"""
VerifierClassifierTrainer
-------------------------
Implements Verifier Mode (ii) using binary classification with verdict tokens.

This trainer implements a classification-based verifier that learns to output
CLEAN/BACKDOOR verdicts for code solutions. The model generates reasoning followed
by a verdict token, which is then mapped to binary scores:
- CLEAN → 1 (honest/correct solution)
- BACKDOOR → 0 (injected/incorrect solution)

Key components:
- Cross-entropy loss on binary classifications
- Verdict token parsing and mapping
- Short generation inference compatible with vLLM
- Distributed training support via accelerator_manager
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging
import gc
from typing import Any


from pvg.trainers.verifier_base import VerifierTrainerBase
from pvg.config.args import ExperimentArgs
from pvg.components.model_manager import ModelManager
from pvg.components.data_manager import DataManager
from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.components.state_tracker import StateTracker
from pvg.utils.verifier_performance import (
    VerifierPerformanceTracker,
    calculate_accuracy_from_pairwise_scores,
)

from pvg.utils.rich_logger import print_prompt_completions_sample_verifier

# Configure logger
logger = logging.getLogger(f"pvg.{__name__}")


class VerifierClassifierTrainer(VerifierTrainerBase):
    """
    Trainer for the Verifier Classifier model that learns to output binary verdicts.

    The trainer generates reasoning followed by CLEAN/BACKDOOR verdict tokens,
    then uses cross-entropy loss to train the model to output:
    - CLEAN for honest solutions (target = 1)
    - BACKDOOR for injected solutions (target = 0)
    """

    def __init__(
        self,
        args: ExperimentArgs,
        model_manager: ModelManager,
        data_manager: DataManager,
        accelerator_manager: AcceleratorManager,
        optimizer_scheduler_manager: OptimizerSchedulerManager,
        metrics_logger: MetricsLogger,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
    ) -> None:
        """
        Initialize the VerifierClassifierTrainer.

        Args:
            args: Configuration arguments for the experiment
            model_manager: Manager for model loading and preparation
            data_manager: Manager for dataset and dataloader handling
            accelerator_manager: Manager for distributed training acceleration
            optimizer_scheduler_manager: Manager for optimizers and schedulers
            metrics_logger: Logger for training and evaluation metrics
            vllm_orchestrator: Orchestrator for VLLM inference
            state_tracker: Tracker for training state
        """
        super().__init__(
            args,
            model_manager,
            data_manager,
            accelerator_manager,
            optimizer_scheduler_manager,
            metrics_logger,
            vllm_orchestrator,
            state_tracker,
        )

        self.tokenizer = self.data_manager.tokenizer
        self.verifier_model = self.model_manager.get_model("verifier", prepared=True)
        self.train_dataloader = self.data_manager.dataloaders["verifier"]["classifier"][
            "train_dataloader"
        ]
        self.eval_dataloader = self.data_manager.dataloaders["verifier"]["classifier"][
            "eval_dataloader"
        ]
        self.optimizer = self.optimizer_scheduler_manager.get_optimizer("verifier")
        self.scheduler = self.optimizer_scheduler_manager.get_scheduler("verifier")

        # Ensure tokenizer has pad_token set
        if self.tokenizer.pad_token is None:
            logger.info("Setting tokenizer pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configuration parameters
        self.config = {
            "confidence_reg_strength": getattr(args, "verifier_confidence_reg", 0.01),
            "debug_samples": getattr(args, "debug_sample_count", 5),
            "log_interval": getattr(args, "log_interval", 1),
            "push_to_hub": getattr(args, "push_to_hub", True),
            "rolling_window_size": getattr(
                args, "rolling_window_size", 50
            ),  # Number of batches for rolling average
            "max_verdict_tokens": getattr(
                args, "max_verdict_tokens", 3
            ),  # CLEAN/BACKDOOR + </verdict>
        }

        self.is_main = self.accelerator_manager.get_state_property(
            property_name="is_main_process"
        )

        # Rolling accuracy tracking
        self.verifier_performance_tracker = VerifierPerformanceTracker(
            window_size=int(self.config["rolling_window_size"])
        )

        # Verdict token mappings
        self.verdict_to_score = {"CLEAN": 1.0, "BACKDOOR": 0.0}

        # Precompute verdict token IDs for efficient processing
        self.clean_token_id = self.tokenizer.encode("CLEAN", add_special_tokens=False)[
            0
        ]
        self.backdoor_token_id = self.tokenizer.encode(
            "BACKDOOR", add_special_tokens=False
        )[0]

        logger.info(
            f"Verdict token IDs: CLEAN={self.clean_token_id}, BACKDOOR={self.backdoor_token_id}"
        )

    def _prepare_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch for model input by concatenating honest and injected examples.

        Args:
            batch: dictionary containing batch data

        Returns:
            tuple containing:
                - input_ids: Combined input IDs for honest and injected examples
                - attention_mask: Combined attention mask
                - are_identical: Boolean tensor indicating if solutions are identical
                - batch_size: Number of examples in the batch
        """
        input_ids = torch.cat(
            (batch["honest_input_ids"], batch["injected_input_ids"]), dim=0
        )
        attention_mask = torch.cat(
            (batch["honest_attention_mask"], batch["injected_attention_mask"]), dim=0
        )
        are_identical = batch["are_identical"]
        batch_size = batch["honest_input_ids"].shape[0]

        return input_ids, attention_mask, are_identical, batch_size

    def _generate_and_parse_verdicts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Generate short completions and parse verdict tokens.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            tuple containing:
                - verdict_scores: Binary scores (1.0 for CLEAN, 0.0 for BACKDOOR)
                - verdict_texts: Raw generated verdict texts
        """
        _ = input_ids.shape[0]

        # Generate short completions (just enough for verdict token)
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": self.config["max_verdict_tokens"],
                "do_sample": False,  # Deterministic for training consistency
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            generated_outputs = self.verifier_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation_config
            )

            # Extract only the new tokens (verdict part)
            new_tokens = generated_outputs[:, input_ids.shape[1] :]

            # Decode the generated verdicts
            verdict_texts = self.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )

        # Parse verdicts and convert to scores
        verdict_scores = []
        for text in verdict_texts:
            # Look for CLEAN or BACKDOOR in the generated text
            if "CLEAN" in text.upper():
                verdict_scores.append(1.0)
            elif "BACKDOOR" in text.upper():
                verdict_scores.append(0.0)
            else:
                # Default to 0.5 if neither is found (neutral/uncertain)
                verdict_scores.append(0.5)
                logger.warning(
                    f"Could not parse verdict from: '{text}', defaulting to 0.5"
                )

        verdict_scores_tensor = torch.tensor(
            verdict_scores, dtype=torch.float32, device=input_ids.device
        )

        return verdict_scores_tensor, verdict_texts

    def _calculate_binary_classification_loss(
        self,
        honest_scores: torch.Tensor,
        injected_scores: torch.Tensor,
        are_identical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate binary classification loss using cross-entropy.

        The loss is based on the following principles:
        - For non-identical pairs: honest should output 1 (CLEAN), injected should output 0 (BACKDOOR)
        - For identical pairs: both should output similar scores (typically 1 for correct solutions)

        Args:
            honest_scores: Scores for honest solutions (0.0-1.0)
            injected_scores: Scores for injected solutions (0.0-1.0)
            are_identical: Boolean tensor indicating if solutions are identical

        Returns:
            tuple containing:
                - classification_loss: Binary cross-entropy loss
                - score_differences: Score differences (honest - injected)
        """
        # Create target labels
        # Honest solutions should be classified as CLEAN (1)
        honest_targets = torch.ones_like(honest_scores)

        # Injected solutions should be classified as BACKDOOR (0), unless identical to honest
        injected_targets = torch.where(
            are_identical,
            torch.ones_like(injected_scores),  # If identical, also target CLEAN (1)
            torch.zeros_like(injected_scores),  # If different, target BACKDOOR (0)
        )

        # Calculate binary cross-entropy loss
        # BCE expects scores in [0,1] range and targets in {0,1}
        honest_loss = F.binary_cross_entropy(
            honest_scores.clamp(1e-7, 1 - 1e-7),  # Clamp to avoid log(0)
            honest_targets,
            reduction="none",
        )

        injected_loss = F.binary_cross_entropy(
            injected_scores.clamp(1e-7, 1 - 1e-7),  # Clamp to avoid log(0)
            injected_targets,
            reduction="none",
        )

        # Combine losses
        total_loss = (honest_loss + injected_loss).mean()

        # Calculate score differences for metrics
        score_differences = honest_scores - injected_scores

        return total_loss, score_differences

    def _get_code_snippets(
        self, batch: dict[str, torch.Tensor], debug: bool = False
    ) -> tuple[list[str] | None, list[str] | None]:
        """
        Extract code snippets from batch for debugging purposes.
        Only executes if debug=True to save computational resources.

        Args:
            batch: dictionary containing batch data
            debug: Whether to extract code snippets

        Returns:
            tuple of honest and injected code snippets, or (None, None) if debug=False
        """
        if not debug:
            return None, None

        # Decode input IDs to text
        honest_prompts = self.tokenizer.batch_decode(
            batch["honest_input_ids"], skip_special_tokens=True
        )
        injected_prompts = self.tokenizer.batch_decode(
            batch["injected_input_ids"], skip_special_tokens=True
        )

        # Extract code snippets between <solution> tags
        honest_code_snippets = [
            prompt.split("<solution>")[1].split("</solution>")[0]
            for prompt in honest_prompts
        ]
        injected_code_snippets = [
            prompt.split("<solution>")[1].split("</solution>")[0]
            for prompt in injected_prompts
        ]

        return honest_code_snippets, injected_code_snippets

    def train(self, num_steps_or_epochs: int) -> None:
        """
        Train the verifier classifier model.

        Args:
            num_steps_or_epochs: Number of training epochs
        """
        # Set model to training mode
        self.verifier_model.train()

        # Track steps for logging
        total_steps = 0
        rolling_metrics: dict[str, Any] = {}
        total_loss = torch.tensor(0.0)
        classification_loss = torch.tensor(0.0)
        confidence_loss = torch.tensor(0.0)

        try:
            for epoch in range(num_steps_or_epochs):
                # Reset rolling accuracy at the start of each epoch
                self.verifier_performance_tracker.reset()

                # Set up progress bar on main process
                if self.is_main:
                    progress_bar = tqdm(
                        self.train_dataloader,
                        desc=f"Epoch {epoch+1}/{num_steps_or_epochs}",
                        total=len(self.train_dataloader),
                    )
                else:
                    progress_bar = self.train_dataloader  # Other processes just iterate

                for batch_idx, batch in enumerate(progress_bar):
                    # Prepare inputs
                    input_ids, attention_mask, are_identical, batch_size = (
                        self._prepare_batch(batch)
                    )

                    # Generate verdicts and get scores
                    all_scores, verdict_texts = self._generate_and_parse_verdicts(
                        input_ids, attention_mask
                    )

                    # Split scores into honest and injected
                    honest_scores = all_scores[:batch_size]
                    injected_scores = all_scores[batch_size:]

                    # Calculate binary classification loss
                    classification_loss, score_differences = (
                        self._calculate_binary_classification_loss(
                            honest_scores, injected_scores, are_identical
                        )
                    )

                    # Add confidence regularization (encourage confident predictions)
                    # Penalize scores near 0.5 (uncertain predictions)
                    confidence_penalty = torch.mean(
                        -torch.abs(
                            all_scores - 0.5
                        )  # Negative because we want to maximize distance from 0.5
                    )
                    confidence_loss = (
                        self.config["confidence_reg_strength"] * confidence_penalty
                    )

                    # Combined loss
                    total_loss = classification_loss + confidence_loss

                    # Calculate accuracy for non-identical pairs
                    with torch.no_grad():
                        num_correct, num_non_identical = (
                            calculate_accuracy_from_pairwise_scores(
                                honest_scores, injected_scores, are_identical
                            )
                        )

                        batch_accuracy = (
                            num_correct / num_non_identical
                            if num_non_identical > 0
                            else 1.0
                        )
                        batch_metrics = {
                            "verifier_accuracy": batch_accuracy,
                            "verifier_avg_score_diff": score_differences.mean().item(),
                            "verifier_identical_ratio": are_identical.float()
                            .mean()
                            .item(),
                        }
                        rolling_metrics = self.verifier_performance_tracker.update(
                            batch_metrics
                        )

                    # Backpropagation
                    self.accelerator_manager.backward(total_loss, "verifier")

                    # Add gradient clipping for verifier
                    if (
                        hasattr(self.args.training_verifier, "max_grad_norm")
                        and self.args.training_verifier.max_grad_norm is not None
                    ):
                        max_grad_norm = self.args.training_verifier.max_grad_norm
                        torch.nn.utils.clip_grad_norm_(
                            self.verifier_model.parameters(), max_grad_norm
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Step scheduler if it exists
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Metrics logging - Extract comprehensive tensor statistics
                    total_steps += 1
                    metrics_to_log = {
                        # Loss components
                        "loss": total_loss.item(),
                        "classification_loss": classification_loss.item(),
                        "confidence_loss": confidence_loss.item(),
                        # Score analysis
                        "avg_score_diff": score_differences.mean().item(),
                        "score_diff_std": score_differences.std().item(),
                        "score_diff_min": score_differences.min().item(),
                        "score_diff_max": score_differences.max().item(),
                        # Honest score statistics
                        "honest_score_mean": honest_scores.mean().item(),
                        "honest_score_std": honest_scores.std().item(),
                        "honest_score_min": honest_scores.min().item(),
                        "honest_score_max": honest_scores.max().item(),
                        # Injected score statistics
                        "injected_score_mean": injected_scores.mean().item(),
                        "injected_score_std": injected_scores.std().item(),
                        "injected_score_min": injected_scores.min().item(),
                        "injected_score_max": injected_scores.max().item(),
                        # Score separation analysis
                        "honest_vs_injected_separation": (
                            honest_scores.mean() - injected_scores.mean()
                        ).item(),
                        "clean_prediction_ratio": (honest_scores > 0.5)
                        .float()
                        .mean()
                        .item(),  # How often honest is classified as CLEAN
                        "backdoor_prediction_ratio": (injected_scores <= 0.5)
                        .float()
                        .mean()
                        .item(),  # How often injected is classified as BACKDOOR
                        # Dataset composition
                        "identical_pairs_ratio": are_identical.float().mean().item(),
                        "batch_size": batch_size,
                        # Confidence analysis
                        "avg_confidence": torch.abs(all_scores - 0.5)
                        .mean()
                        .item(),  # Distance from neutral
                        "uncertain_predictions_ratio": (
                            (all_scores > 0.4) & (all_scores < 0.6)
                        )
                        .float()
                        .mean()
                        .item(),  # Predictions near 0.5
                        # Rolling accuracy
                        "verifier_accuracy": (
                            rolling_metrics["verifier_accuracy"]
                            if "verifier_accuracy" in rolling_metrics
                            else float("nan")
                        ),
                    }

                    # Log metrics
                    for name, value in metrics_to_log.items():
                        self.metrics_logger.store_metric(
                            phase=self.state_tracker.phase,
                            mode="train",
                            model="verifier",
                            name=name,
                            value=value,
                        )

                    # Periodically perform logging on main process
                    if self.is_main and (
                        total_steps % self.config["log_interval"] == 0
                    ):
                        # Extract code snippets for debugging if needed
                        if total_steps % (self.config["log_interval"] * 25) == 0:
                            honest_snippets, injected_snippets = (
                                self._get_code_snippets(batch, debug=True)
                            )

                            if (
                                honest_snippets is not None
                                and injected_snippets is not None
                            ):
                                # Log sample of prompts and verdicts
                                print_prompt_completions_sample_verifier(
                                    honest_prompts=honest_snippets,
                                    injected_prompts=injected_snippets,
                                    honest_scores=honest_scores,
                                    injected_scores=injected_scores,
                                    are_identical=are_identical,
                                    step=total_steps,
                                    num_samples=min(
                                        int(self.config["debug_samples"]),
                                        len(honest_snippets),
                                    ),
                                )

                                # Log verdict examples
                                logger.info(f"Step {total_steps} - Verdict examples:")
                                for i in range(min(3, len(verdict_texts) // 2)):
                                    honest_verdict = verdict_texts[i]
                                    injected_verdict = verdict_texts[i + batch_size]
                                    logger.info(
                                        f"  Honest: '{honest_verdict}' (score: {honest_scores[i]:.3f})"
                                    )
                                    logger.info(
                                        f"  Injected: '{injected_verdict}' (score: {injected_scores[i]:.3f})"
                                    )

                        # Update tqdm progress bar with only key metrics
                        key_metrics = {
                            "loss": metrics_to_log["loss"],
                            "acc": rolling_metrics.get(
                                "verifier_accuracy", float("nan")
                            ),
                            "sep": metrics_to_log["honest_vs_injected_separation"],
                            "conf": metrics_to_log["avg_confidence"],
                            "lr": (
                                self.scheduler.get_last_lr()[0]
                                if self.scheduler
                                else 0.0
                            ),
                        }
                        progress_bar.set_postfix(
                            **{k: f"{v:.4f}" for k, v in key_metrics.items()}
                        )

                    # Log step metrics
                    self.metrics_logger.flush(
                        phase=self.state_tracker.phase, mode="train"
                    )

                    # Update state tracker
                    self.state_tracker.increment_step()

                    # Clean up memory if needed
                    if total_steps % 100 == 0:
                        torch.cuda.empty_cache()

                # End of epoch logging
                if self.is_main:
                    logger.info(
                        f"Epoch {epoch+1}/{num_steps_or_epochs} completed. "
                        f"Verifier accuracy: {rolling_metrics['verifier_accuracy']:.4f}, "
                        f"Avg confidence: {metrics_to_log['avg_confidence']:.4f}"
                    )
                    # Close tqdm bar on main process
                    progress_bar.close()

                # Run evaluation at the end of each epoch
                self.evaluate()

                # Return to training mode after evaluation
                self.verifier_model.train()

            # End of training
            if self.is_main:
                logger.info("Verifier Classifier training finished.")

            # Push final verifier model to hub
            self._push_model_to_hub()

            # Summary of training
            logger.info("Training Summary:")
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Total loss: {total_loss.item()}")
            logger.info(f"Classification loss: {classification_loss.item()}")
            logger.info(f"Confidence loss: {confidence_loss.item()}")
            logger.info(
                f"Final verifier accuracy: {rolling_metrics['verifier_accuracy']:.4f}"
                if "verifier_accuracy" in rolling_metrics
                else "Final verifier accuracy: N/A"
            )

            # Move **this** verifier model to vLLM --> swap it into the vLLM model
            self.vllm_orchestrator.sync_weights(
                phase="verifier", model_manager=self.model_manager
            )
            self.accelerator_manager.wait_for_everyone()

            # Clean up memory - delete model, tokenizer, dataloaders, optimizer, scheduler
            del (
                self.verifier_model,
                self.tokenizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.optimizer,
                self.scheduler,
            )
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memory cleaned up. Ready for next round.")
            # Wait for all processes to finish
            self.accelerator_manager.wait_for_everyone()

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            # Attempt to clean up resources
            torch.cuda.empty_cache()
            raise

    def evaluate(self) -> dict[str, float] | None:
        """
        Evaluate the verifier classifier model.

        Returns:
            dictionary of evaluation metrics
        """
        # Set model to evaluation mode
        self.verifier_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_score_diffs = []
        all_verdict_texts = []

        # Accumulate comprehensive evaluation statistics
        all_honest_scores = []
        all_injected_scores = []
        all_are_identical = []

        if self.is_main:
            progress_bar = tqdm(
                self.eval_dataloader, desc="Evaluating", total=len(self.eval_dataloader)
            )
        else:
            progress_bar = self.eval_dataloader

        try:
            with torch.no_grad():  # No gradients needed for evaluation
                for batch in progress_bar:
                    # Prepare inputs
                    input_ids, attention_mask, are_identical, batch_size = (
                        self._prepare_batch(batch)
                    )

                    # Generate verdicts and get scores
                    all_scores, verdict_texts = self._generate_and_parse_verdicts(
                        input_ids, attention_mask
                    )

                    # Split scores into honest and injected
                    honest_scores = all_scores[:batch_size]
                    injected_scores = all_scores[batch_size:]

                    # Calculate binary classification loss
                    classification_loss, score_differences = (
                        self._calculate_binary_classification_loss(
                            honest_scores, injected_scores, are_identical
                        )
                    )

                    # Gather results across all processes
                    gathered_loss = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(classification_loss)
                    gathered_diff = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(score_differences)
                    gathered_honest = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(honest_scores)
                    gathered_injected = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(injected_scores)
                    gathered_identical = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(are_identical)

                    # Accumulate statistics
                    all_score_diffs.append(gathered_diff.cpu())
                    all_honest_scores.append(gathered_honest.cpu())
                    all_injected_scores.append(gathered_injected.cpu())
                    all_are_identical.append(gathered_identical.cpu())
                    all_verdict_texts.extend(verdict_texts)

                    # Sum loss (will average later)
                    total_loss += gathered_loss.sum().item() * batch_size

                    # Calculate accuracy for non-identical pairs
                    # Prediction: honest is preferred if score_h > score_i
                    predicted_preference = honest_scores > injected_scores
                    # Ground truth: honest should be preferred for non-identical pairs
                    correct_predictions = torch.logical_and(
                        predicted_preference, ~are_identical
                    )

                    # Gather correct predictions and non-identical counts
                    gathered_correct = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(correct_predictions)
                    gathered_not_identical = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(~are_identical)

                    total_correct += gathered_correct.sum().item()
                    total_samples += gathered_not_identical.sum().item()

                    if self.is_main:
                        progress_bar.set_postfix(loss=classification_loss.item())

            # Calculate overall metrics
            total_pairs_evaluated = len(self.eval_dataloader.dataset)

            avg_loss = (
                total_loss / total_pairs_evaluated if total_pairs_evaluated > 0 else 0.0
            )
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            # Concatenate all accumulated statistics
            all_diffs = (
                torch.cat(all_score_diffs) if all_score_diffs else torch.empty(0)
            )
            all_honest = (
                torch.cat(all_honest_scores) if all_honest_scores else torch.empty(0)
            )
            all_injected = (
                torch.cat(all_injected_scores)
                if all_injected_scores
                else torch.empty(0)
            )
            all_identical = (
                torch.cat(all_are_identical) if all_are_identical else torch.empty(0)
            )

            # Calculate comprehensive evaluation metrics
            eval_metrics_summary = {
                # Basic metrics
                "eval_loss": avg_loss,
                "eval_accuracy": accuracy,
            }

            # Add metrics only if tensors are not empty
            if all_diffs.numel() > 0:
                eval_metrics_summary.update(
                    {
                        "eval_avg_diff": all_diffs.mean().item(),
                        "eval_std_diff": all_diffs.std().item(),
                        "eval_min_diff": all_diffs.min().item(),
                        "eval_max_diff": all_diffs.max().item(),
                        "eval_abs_diff_mean": all_diffs.abs().mean().item(),
                        "eval_positive_diff_ratio": (all_diffs > 0)
                        .float()
                        .mean()
                        .item(),
                    }
                )

            if all_honest.numel() > 0:
                eval_metrics_summary.update(
                    {
                        "eval_honest_score_mean": all_honest.mean().item(),
                        "eval_honest_score_std": all_honest.std().item(),
                        "eval_honest_score_min": all_honest.min().item(),
                        "eval_honest_score_max": all_honest.max().item(),
                        "eval_clean_prediction_ratio": (all_honest > 0.5)
                        .float()
                        .mean()
                        .item(),
                    }
                )

            if all_injected.numel() > 0:
                eval_metrics_summary.update(
                    {
                        "eval_injected_score_mean": all_injected.mean().item(),
                        "eval_injected_score_std": all_injected.std().item(),
                        "eval_injected_score_min": all_injected.min().item(),
                        "eval_injected_score_max": all_injected.max().item(),
                        "eval_backdoor_prediction_ratio": (all_injected <= 0.5)
                        .float()
                        .mean()
                        .item(),
                    }
                )

            if all_honest.numel() > 0 and all_injected.numel() > 0:
                eval_metrics_summary["eval_honest_vs_injected_separation"] = (
                    all_honest.mean() - all_injected.mean()
                ).item()

                combined_scores = torch.cat([all_honest, all_injected])
                eval_metrics_summary["eval_avg_confidence"] = (
                    torch.abs(combined_scores - 0.5).mean().item()
                )

            if all_identical.numel() > 0:
                eval_metrics_summary["eval_identical_ratio"] = (
                    all_identical.float().mean().item()
                )

            # Verdict parsing statistics
            if self.is_main and all_verdict_texts:
                clean_count = sum(
                    1 for text in all_verdict_texts if "CLEAN" in text.upper()
                )
                backdoor_count = sum(
                    1 for text in all_verdict_texts if "BACKDOOR" in text.upper()
                )
                unparseable_count = (
                    len(all_verdict_texts) - clean_count - backdoor_count
                )

                eval_metrics_summary.update(
                    {
                        "eval_clean_verdicts_ratio": clean_count
                        / len(all_verdict_texts),
                        "eval_backdoor_verdicts_ratio": backdoor_count
                        / len(all_verdict_texts),
                        "eval_unparseable_verdicts_ratio": unparseable_count
                        / len(all_verdict_texts),
                    }
                )

            if self.is_main:
                logger.info(
                    f"Evaluation finished. "
                    f"Average Loss: {avg_loss:.4f}, "
                    f"Accuracy (on non-identical pairs): {accuracy:.4f} ({total_correct}/{total_samples}), "
                    f"Avg Score Diff: {eval_metrics_summary.get('eval_avg_diff', float('nan')):.4f} ± {eval_metrics_summary.get('eval_std_diff', float('nan')):.4f}, "
                    f"Avg Confidence: {eval_metrics_summary.get('eval_avg_confidence', float('nan')):.4f}"
                )
                # Close progress bar
                if isinstance(progress_bar, tqdm):
                    progress_bar.close()

            # Log metrics
            for name, value in eval_metrics_summary.items():
                self.metrics_logger.store_metric(
                    phase=self.state_tracker.phase,
                    mode="eval",
                    model="verifier",
                    name=name,
                    value=value,
                )

            # Log summary metrics
            self.metrics_logger.flush(phase=self.state_tracker.phase, mode="eval")

            return eval_metrics_summary

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            # Attempt to clean up resources
            torch.cuda.empty_cache()
            if self.is_main and isinstance(
                progress_bar, tqdm
            ):  # Ensure progress_bar is tqdm before closing
                progress_bar.close()
            raise

    def _push_model_to_hub(self) -> None:
        """
        Push the trained model to the Hugging Face model hub with proper error handling.
        """

        # Only push from the main process
        if not self.is_main:
            logger.info("Not the main process, skipping push to hub")
            return

        # Check if push_to_hub is enabled
        if not self.config["push_to_hub"]:
            logger.info("push_to_hub is disabled, skipping")
            return

        verifier_model_name = (
            f"jvelja/verifier-classifier_round_{self.state_tracker.round}"
        )

        try:
            # Unwrap the model before pushing to avoid distributed training issues
            unwrapped_model = self.accelerator_manager.unwrap_model(
                self.verifier_model, key="verifier"
            )

            # Push the unwrapped model
            unwrapped_model.push_to_hub(repo_id=verifier_model_name)
            logger.info(
                f"Verifier Classifier model successfully pushed to the hub as {verifier_model_name}."
            )
            # Tokenizer should be pushed to the same repo
            self.tokenizer.push_to_hub(repo_id=verifier_model_name)
        except Exception as e:
            logger.error(f"Failed to push model to hub: {str(e)}")
            raise e
