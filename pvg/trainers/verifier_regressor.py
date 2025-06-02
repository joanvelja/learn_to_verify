# pvg/trainers/verifier_regressor.py

"""
VerifierRegressorTrainer
------------------------
Implements Verifier Mode (i) using Bradley-Terry preference learning.

This trainer implements a regression-based verifier that learns to score code solutions,
where higher scores should be assigned to honest (correct) solutions compared to
injected (potentially incorrect) solutions.

Key components:
- Bradley-Terry preference learning model: P(a > b) = sigmoid(score_a - score_b)
- Score regularization to avoid model drift
- Distributed training support via accelerator_manager
- Comprehensive evaluation metrics
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging
import gc


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
logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class VerifierRegressorTrainer(VerifierTrainerBase):
    """
    Trainer for the Verifier Regressor model that learns to score solutions using
    Bradley-Terry preference learning.

    The trainer compares honest solutions against injected ones and learns to assign
    higher scores to honest solutions. For identical solutions, it learns to assign
    similar scores.
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
        Initialize the VerifierRegressorTrainer.

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
        self.train_dataloader = self.data_manager.dataloaders["verifier"][
            "train_dataloader"
        ]
        self.eval_dataloader = self.data_manager.dataloaders["verifier"][
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
            "lambda_reg": getattr(args, "verifier_reg_strength", 0.05),
            "debug_samples": getattr(args, "debug_sample_count", 4),
            "log_interval": getattr(args, "log_interval", 1),
            "push_to_hub": getattr(args, "push_to_hub", True),
            "rolling_window_size": getattr(
                args, "rolling_window_size", 100
            ),  # Number of batches for rolling average
        }

        self.is_main = self.accelerator_manager.get_state_property(
            property_name="is_main_process"
        )

        # Rolling accuracy tracking
        self.verifier_performance_tracker = VerifierPerformanceTracker(
            window_size=self.config["rolling_window_size"]
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

    def _extract_scores(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Extract scalar scores from model outputs.

        Args:
            outputs: Model output tensor or tuple

        Returns:
            Tensor of scalar scores
        """
        scores = outputs[0] if isinstance(outputs, tuple) else outputs
        if scores.dim() > 1 and scores.shape[1] == 1:
            scores = scores.squeeze(-1)
        return scores

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

    def _calculate_bradley_terry_loss(
        self,
        honest_scores: torch.Tensor,
        injected_scores: torch.Tensor,
        are_identical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate Bradley-Terry preference learning loss.

        The loss is based on the following principles:
        - For non-identical pairs: maximize P(honest > injected)
        - For identical pairs: maximize P(honest = injected)

        Args:
            honest_scores: Scores for honest solutions
            injected_scores: Scores for injected solutions
            are_identical: Boolean tensor indicating if solutions are identical

        Returns:
            tuple containing:
                - bt_loss: Bradley-Terry loss
                - diff_scores: Score differences (honest - injected)
        """
        # Calculate score differences
        diff_scores = honest_scores - injected_scores

        # Calculate log probabilities using log-sigmoid for numerical stability
        log_p_honest_preferred = F.logsigmoid(diff_scores)  # log P(honest > injected)
        log_p_injected_preferred = F.logsigmoid(
            -diff_scores
        )  # log P(injected > honest)

        # Loss terms:
        # 1. For non-identical pairs: -log P(honest > injected)
        # 2. For identical pairs: -0.5 * [log P(honest > injected) + log P(injected > honest)]
        loss_term_honest_preferred = -log_p_honest_preferred
        loss_term_equal_preference = -0.5 * (
            log_p_honest_preferred + log_p_injected_preferred
        )

        # Select appropriate loss based on whether solutions are identical
        bt_loss = torch.where(
            are_identical, loss_term_equal_preference, loss_term_honest_preferred
        ).mean()

        return bt_loss, diff_scores

    def train(self, num_steps_or_epochs: int) -> None:
        """
        Train the verifier regressor model.

        Args:
            num_steps_or_epochs: Number of training epochs
        """
        # Set model to training mode
        self.verifier_model.train()

        # Track steps for logging
        total_steps = 0

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

                    # Get model outputs
                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                    outputs = self.verifier_model(**model_inputs).logits

                    # Extract scores
                    scores = self._extract_scores(outputs)

                    # Split scores into honest and injected
                    honest_scores = scores[:batch_size]
                    injected_scores = scores[batch_size:]

                    # Calculate Bradley-Terry loss
                    bt_loss, diff_scores = self._calculate_bradley_terry_loss(
                        honest_scores, injected_scores, are_identical
                    )

                    # Add regularization term to prevent score drift
                    score_regularization = (
                        honest_scores.pow(2).mean() + injected_scores.pow(2).mean()
                    ) / 2
                    regularization_loss = (
                        self.config["lambda_reg"] * score_regularization
                    )

                    # Combined loss
                    total_loss = bt_loss + regularization_loss

                    # Calculate accuracy for non-identical pairs
                    # Prediction: honest is preferred if score_h > score_i
                    # with torch.no_grad():  # Accuracy calculation doesn't need gradients
                    #     # predicted_preference = honest_scores > injected_scores
                    #     # # Ground truth: honest should be preferred for non-identical pairs
                    #     # correct_predictions = torch.logical_and(
                    #     #     predicted_preference, ~are_identical
                    #     # )
                    #     # num_correct = correct_predictions.sum().item()
                    #     # num_non_identical = (~are_identical).sum().item()

                    #     # # Update rolling accuracy
                    #     # rolling_accuracy = self._update_rolling_accuracy(num_correct, num_non_identical)
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
                            "verifier_avg_score_diff": diff_scores.mean().item(),
                            "verifier_identical_ratio": are_identical.float()
                            .mean()
                            .item(),
                        }
                        rolling_metrics = self.verifier_performance_tracker.update(
                            batch_metrics
                        )

                    # Backpropagation
                    self.accelerator_manager.backward(total_loss, "verifier")

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Step scheduler if it exists
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Metrics logging
                    total_steps += 1
                    metrics = {
                        "loss": total_loss.item(),
                        "bt_loss": bt_loss.item(),
                        "reg_loss": regularization_loss.item(),
                        "avg_score_diff": diff_scores.mean().item(),
                        "verifier_accuracy": rolling_metrics[
                            "verifier_accuracy"
                        ],  # Use rolling accuracy instead of batch accuracy
                    }

                    # Log metrics
                    for name, value in metrics.items():
                        self.metrics_logger.store_metric(
                            mode="train",
                            model_key="verifier",
                            metric_name=name,
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
                                # Log sample of prompts and scores
                                print_prompt_completions_sample_verifier(
                                    honest_prompts=honest_snippets,
                                    injected_prompts=injected_snippets,
                                    honest_scores=honest_scores,
                                    injected_scores=injected_scores,
                                    are_identical=are_identical,
                                    step=total_steps,
                                    num_samples=min(
                                        self.config["debug_samples"],
                                        len(honest_snippets),
                                    ),
                                )

                        # Update tqdm progress bar
                        progress_bar.set_postfix(
                            **{k: f"{v:.4f}" for k, v in metrics.items()}
                        )

                    # Log step metrics
                    self.metrics_logger.log_step_metrics(
                        phase=self.state_tracker.phase,
                        mode="train",
                    )

                    # Update state tracker
                    self.state_tracker.increment_step()

                    # Clean up memory if needed
                    if total_steps % 100 == 0:
                        torch.cuda.empty_cache()

                # End of epoch logging
                if self.is_main:
                    logger.info(
                        f"Epoch {epoch+1}/{num_steps_or_epochs} completed. Verifier accuracy: {rolling_metrics['verifier_accuracy']:.4f}"
                    )
                    # Close tqdm bar on main process
                    progress_bar.close()

                # Run evaluation at the end of each epoch
                self.evaluate()

                # Return to training mode after evaluation
                self.verifier_model.train()

            # End of training
            if self.is_main:
                logger.info("Verifier Regressor training finished.")

            # Push final verifier model to hub
            self._push_model_to_hub()

            # Summary of training
            logger.info("Training Summary:")
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Total loss: {total_loss.item()}")
            logger.info(f"BT loss: {bt_loss.item()}")
            logger.info(f"Reg loss: {regularization_loss.item()}")
            logger.info(
                f"Final verifier accuracy: {rolling_metrics['verifier_accuracy']:.4f}"
            )

            assert total_steps == len(
                self.train_dataloader
            ), f"Total steps {total_steps} does not match dataloader length {len(self.train_dataloader)}"
            assert (
                self.state_tracker.step == total_steps
            ), f"State tracker step {self.state_tracker.step} does not match total steps {total_steps}"

            # self.accelerator_manager.wait_for_everyone() # Unsure if this is needed

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
        Evaluate the verifier regressor model.

        Returns:
            dictionary of evaluation metrics
        """
        # Set model to evaluation mode
        self.verifier_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_score_diffs = []

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

                    # Get model outputs
                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                    outputs = self.verifier_model(**model_inputs).logits

                    # Extract scores
                    scores = self._extract_scores(outputs)

                    # Split scores into honest and injected
                    honest_scores = scores[:batch_size]
                    injected_scores = scores[batch_size:]

                    # Calculate Bradley-Terry loss
                    bt_loss, diff_scores = self._calculate_bradley_terry_loss(
                        honest_scores, injected_scores, are_identical
                    )

                    # Gather results across all processes
                    gathered_loss = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(bt_loss)
                    gathered_diff = self.accelerator_manager.get_accelerator(
                        "verifier"
                    ).gather(diff_scores)
                    all_score_diffs.append(gathered_diff.cpu())

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
                        progress_bar.set_postfix(loss=bt_loss.item())

            # Calculate overall metrics
            total_pairs_evaluated = len(self.eval_dataloader.dataset)

            metrics = {}
            avg_loss = (
                total_loss / total_pairs_evaluated if total_pairs_evaluated > 0 else 0.0
            )
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            # Concatenate all score differences
            all_diffs = torch.cat(all_score_diffs)
            avg_diff = all_diffs.mean().item()
            std_diff = all_diffs.std().item()

            metrics = {
                "eval_loss": avg_loss,
                "eval_accuracy": accuracy,
                "eval_avg_diff": avg_diff,
                "eval_std_diff": std_diff,
            }

            if self.is_main:
                logger.info(
                    f"Evaluation finished. "
                    f"Average Loss: {avg_loss:.4f}, "
                    f"Accuracy (on non-identical pairs): {accuracy:.4f} ({total_correct}/{total_samples}), "
                    f"Avg Score Diff: {avg_diff:.4f} ± {std_diff:.4f}"
                )
                # Close progress bar
                progress_bar.close()

            # Log metrics
            for name, value in metrics.items():
                self.metrics_logger.store_metric(
                    mode="eval", model_key="verifier", metric_name=name, value=value
                )

            # Log summary metrics
            self.metrics_logger.log_step_metrics(
                phase=self.state_tracker.phase, mode="eval"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            # Attempt to clean up resources
            torch.cuda.empty_cache()
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
            f"jvelja/verifier-regressor_round_{self.state_tracker.round}"
        )

        try:
            # Unwrap the model before pushing to avoid distributed training issues
            unwrapped_model = self.accelerator_manager.unwrap_model(
                self.verifier_model, key="verifier"
            )

            # Push the unwrapped model
            unwrapped_model.push_to_hub(repo_id=verifier_model_name)
            logger.info(
                f"Verifier Regressor model successfully pushed to the hub as {verifier_model_name}."
            )
            # Tokenizer should be pushed to the same repo
            self.tokenizer.push_to_hub(repo_id=verifier_model_name)
        except Exception as e:
            logger.error(f"Failed to push model to hub: {str(e)}")
            raise e
