import gc
import json
import logging
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import HfApi, upload_folder
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_NAME

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.code_evaluator import BatchEvaluator
from pvg.components.data_manager import DataManager
from pvg.components.formatter import Formatter
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.model_manager import ModelManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.state_tracker import StateTracker
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.config.args import ExperimentArgs
from pvg.pipelines.prover_training_pipeline import ProverTrainingPipeline
from pvg.processors.batch_processor import BatchProcessor
from pvg.processors.metrics_processor import MetricsProcessor
from pvg.rl import GRPO
from pvg.strategies.implementations.completion_strategies import (
    ProverCompletionStrategy,
)
from pvg.strategies.implementations.evaluation_strategies import (
    create_standard_evaluation_strategy,
)
from pvg.strategies.implementations.loss_strategies import (
    LigerLossStrategy,
    StandardGRPOLossStrategy,
)
from pvg.strategies.implementations.model_forward_strategies import ModelForwardStrategy
from pvg.strategies.implementations.reward_strategies import TierBasedRewardStrategy
from pvg.strategies.implementations.verification_strategies import (
    create_verification_strategy,
)
from pvg.utils.verifier_performance import VerifierPerformanceTracker

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


def _clone_tensors_for_safetensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Clone tensors that share memory to avoid safetensors shared memory error.

    This function addresses the issue where models with tied embeddings
    (e.g., lm_head.weight and model.embed_tokens.weight) share memory storage,
    causing safetensors to fail with a RuntimeError about shared memory tensors.

    Args:
        state_dict: The model state dictionary

    Returns:
        A new state dictionary with cloned tensors to break memory sharing
    """
    # First, identify all unique storage pointers
    storage_to_keys = {}
    for key, tensor in state_dict.items():
        if hasattr(tensor, "storage"):
            storage_ptr = tensor.storage().data_ptr()
            if storage_ptr not in storage_to_keys:
                storage_to_keys[storage_ptr] = []
            storage_to_keys[storage_ptr].append(key)

    # Find storages that are shared by multiple tensors
    shared_storages = {ptr: keys for ptr, keys in storage_to_keys.items() if len(keys) > 1}

    if shared_storages:
        logger.info(
            f"Found {len(shared_storages)} shared storage(s) affecting {sum(len(keys) for keys in shared_storages.values())} tensors"
        )
        for ptr, keys in shared_storages.items():
            logger.info(f"Shared storage keys: {keys}")

    # Clone the state dict, cloning shared tensors to break memory sharing
    cloned_state_dict = {}
    for key, tensor in state_dict.items():
        if hasattr(tensor, "storage"):
            storage_ptr = tensor.storage().data_ptr()
            if storage_ptr in shared_storages and len(shared_storages[storage_ptr]) > 1:
                # Clone tensor to break memory sharing
                cloned_state_dict[key] = tensor.clone().detach()
                logger.debug(f"Cloned shared tensor: {key}")
            else:
                cloned_state_dict[key] = tensor
        else:
            cloned_state_dict[key] = tensor

    return cloned_state_dict


class ProverTrainer:
    def __init__(
        self,
        args: ExperimentArgs,
        formatter: Formatter,
        model_manager: ModelManager,
        data_manager: DataManager,
        accelerator_manager: AcceleratorManager,
        optimizer_scheduler_manager: OptimizerSchedulerManager,
        metrics_logger: MetricsLogger,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
        batch_evaluator: BatchEvaluator,
        dataset_type: Literal["coding", "math"],
        grpo: GRPO,
    ):
        self.accelerator_manager = accelerator_manager
        self.model_manager = model_manager
        self.vllm_orchestrator = vllm_orchestrator
        self.metrics_logger = metrics_logger
        self.optimizer_scheduler_manager = optimizer_scheduler_manager
        self.metrics_processor = MetricsProcessor()

        self.tokenizer = model_manager.get_tokenizer()
        self.train_dataloader = data_manager.dataloaders["provers"]["sneaky_prover"]["train_dataloader"]
        self.eval_dataloader = data_manager.dataloaders["provers"]["sneaky_prover"]["eval_dataloader"]
        self.dataset_type = dataset_type
        self.total_steps = 0
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.state_tracker = state_tracker

        # Configuration parameters
        self.config = {
            "push_to_hub": getattr(args, "push_to_hub", True),
            "checkpoint_interval": getattr(args, "save_steps", 2),
            "ckpt_output_dir": getattr(args.training_sneaky_prover, "ckpt_output_dir", "./sneaky_prover_ckpt"),
            "base_repo_id": f"jvelja/prover_{args.dataset.dataset_size}_round_{self.state_tracker.round}",
        }

        # Checkpoint tracking
        self.checkpoint_count = 0

        if self.tokenizer.pad_token is None:
            logger.info("Setting tokenizer pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.completion_strategy = ProverCompletionStrategy(
            vllm_orchestrator=vllm_orchestrator,
            formatter=formatter,
            dataset_type=dataset_type,
            args=args,
        )
        self.verification_strategy = create_verification_strategy(
            args=args,
            vllm_orchestrator=vllm_orchestrator,
            formatter=formatter,
            batch_evaluator=batch_evaluator,
            accelerator_manager=accelerator_manager,
            dataset_type=dataset_type,
        )

        self.reward_strategy = TierBasedRewardStrategy(
            verifier_tracker=VerifierPerformanceTracker(),
            metrics_logger=metrics_logger,
            accelerator_manager=accelerator_manager,
            grpo=grpo,
            metrics_processor=self.metrics_processor,
            state_tracker=state_tracker,
            interaction_logger=vllm_orchestrator.interaction_logger,
        )

        if args.training_sneaky_prover.apply_liger_kernel:
            self.loss_strategy = LigerLossStrategy(
                rl_config=args.rl,
                model_manager=model_manager,
                accelerator_manager=accelerator_manager,
                metrics_logger=metrics_logger,
            )
        else:  # Standard GRPO loss
            self.loss_strategy = StandardGRPOLossStrategy(
                rl_config=args.rl,
                metrics_logger=metrics_logger,
                accelerator_manager=accelerator_manager,
            )

        self.model_forward_strategy = ModelForwardStrategy(
            temperature=args.vllm_sneaky_prover.temperature,
            ref_batch_size=2,  # Memory-efficient batch size for reference model computation
        )

        self.batch_processor = BatchProcessor(
            tokenizer=self.tokenizer,
            accelerator_manager=accelerator_manager,
            rl_config=args.rl,
            dataset_type=dataset_type,
            buffer_completions=True,
        )

        self.pipeline = ProverTrainingPipeline(
            completion_strategy=self.completion_strategy,
            verification_strategy=self.verification_strategy,
            reward_strategy=self.reward_strategy,
            loss_strategy=self.loss_strategy,
            model_forward_strategy=self.model_forward_strategy,
            batch_processor=self.batch_processor,
            metrics_processor=self.metrics_processor,
            accelerator_manager=accelerator_manager,
            metrics_logger=metrics_logger,
            state_tracker=state_tracker,
            model_manager=model_manager,
            rl_config=args.rl,
        )

        # Create evaluation strategy using factory function
        self.evaluation_strategy = create_standard_evaluation_strategy(
            accelerator_manager=accelerator_manager,
            metrics_logger=metrics_logger,
            state_tracker=state_tracker,
            use_progress_bar=True,
        )

        self.is_main = self.accelerator_manager.get_state_property(property_name="is_main_process")

    def train(self, num_steps_or_epochs: int = 1):
        """Train the prover model"""
        training_step = 0
        optimizer_step = 0

        for epoch in range(num_steps_or_epochs):
            logger.info(f"Starting Prover Training Epoch {epoch + 1}/{num_steps_or_epochs}")

            if self.is_main:
                progress_bar = tqdm(self.train_dataloader, desc=f"Prover Epoch {epoch + 1}")
            else:
                progress_bar = self.train_dataloader

            for batch_idx, raw_batch_data in enumerate(progress_bar):
                logger.info(
                    f"Metrics storage - StateTracker step: {self.state_tracker.step + 1}, Training step: {training_step} - Optimizer step: {optimizer_step} - Batch index: {batch_idx}"
                )
                # 1. Process batch
                batch_inputs = self.pipeline.process_batch(
                    raw_batch_data,
                    training_step,
                    self.gradient_accumulation_steps,
                    use_buffering=True,
                )

                # 2. Compute training step
                policy_model = self.model_manager.get_model(key="sneaky_prover", prepared=True)
                unwrapped_model = self.accelerator_manager.unwrap_model(policy_model, key="sneaky_prover")
                training_step_result = self.pipeline.compute_training_step(unwrapped_model, batch_inputs, mode="train")
                # 3. Compute loss
                loss = training_step_result.loss_result.loss
                batch_metrics = training_step_result.batch_metrics

                # 4. Compute gradients
                self.accelerator_manager.backward(loss, key="sneaky_prover")
                self.accelerator_manager.wait_for_everyone()

                # 5. Store metrics
                self.metrics_logger.store_metric(
                    mode="train",
                    model="sneaky_prover",
                    name="loss",
                    value=loss * self.gradient_accumulation_steps,
                    phase=self.state_tracker.phase,  # Pass phase
                )

                # Store entropy if available
                if training_step_result.loss_result.per_token_entropy is not None:
                    self.metrics_logger.store_entropy(
                        phase=self.state_tracker.phase,
                        mode="train",
                        model="sneaky_prover",
                        per_token_entropy=training_step_result.loss_result.per_token_entropy,
                    )

                # Store loss result metrics (KL divergence, clip ratios, etc.)
                if training_step_result.loss_result.metrics:
                    for key, value in training_step_result.loss_result.metrics.items():
                        self.metrics_logger.store_metric(
                            mode="train",
                            model="sneaky_prover",
                            name=key,
                            value=value,
                            phase=self.state_tracker.phase,
                        )

                for key, value in batch_metrics.items():
                    self.metrics_logger.store_metric(
                        mode="train",
                        model="sneaky_prover",
                        name=key,
                        value=value,
                        phase=self.state_tracker.phase,  # Pass phase
                    )

                # Gradient accumulation optimizer step
                is_last_step_in_batch = (training_step + 1) == len(self.train_dataloader)

                is_accumulation_boundary = (training_step + 1) % self.gradient_accumulation_steps == 0
                is_sync_step = is_last_step_in_batch or is_accumulation_boundary
                if is_sync_step:  # A LOOOOT OF STUFF HAPPENS HERE
                    optimizer_step += 1
                    # ============================================================================
                    # DEEPSPEED OPTIMIZER STEP
                    # ============================================================================
                    # Standard optimizer path (was previously prepared)
                    optimizer = self.optimizer_scheduler_manager.get_optimizer("sneaky_prover")
                    scheduler = self.optimizer_scheduler_manager.get_scheduler("sneaky_prover")

                    if self.accelerator_manager.get_state_property("is_main_process"):
                        logger.info("🚀 Using standard optimizer step() method")

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                    # Explicitly update the model in the manager after the optimizer step
                    self.model_manager.models["sneaky_prover"] = policy_model

                    self.accelerator_manager.wait_for_everyone()

                    # ============================================================================
                    # CHECKPOINTING TO HUB
                    # ============================================================================
                    checkpoint_interval = self.config["checkpoint_interval"]
                    if (
                        isinstance(checkpoint_interval, int)
                        and optimizer_step > 0
                        and optimizer_step % checkpoint_interval == 0
                    ):
                        self.accelerator_manager.wait_for_everyone()
                        if self.accelerator_manager.get_state_property("is_main_process"):
                            logger.info(f"🚀 Optimizer step {optimizer_step}, pushing checkpoint to hub.")

                        round = self.state_tracker.get_round()
                        self.checkpoint_count += 1
                        self._push_checkpoint_to_hub(round=round, step=optimizer_step)
                        self.accelerator_manager.wait_for_everyone()

                    # Sync vllm weights -- Update means change vllm weights for inference as well
                    self.vllm_orchestrator.sync_weights(phase="provers", model_manager=self.model_manager)
                    self.accelerator_manager.wait_for_everyone()

                    # Log step metrics after optimizer step
                    self.metrics_logger.flush(phase=self.state_tracker.phase, mode="train")

                    # Update state tracker
                    self.state_tracker.increment_step()

                    # Evaluate every 100 micro-batches (not optimizer steps)
                    # if (training_step + 1) % 100 == 0:
                    if False:  # Takes too long to eval... just go with it and hope for the best
                        self.vllm_orchestrator.sync_weights(phase="provers", model_manager=self.model_manager)
                        self.accelerator_manager.wait_for_everyone()
                        _ = self.evaluate()

                        # Log comprehensive reward summary
                        self.pipeline.log_reward_summary()

                    if self.is_main:
                        # Get progress metrics from pipeline
                        progress_metrics = self.pipeline.get_progress_metrics()

                        # Add loss and verifier accuracy
                        latest_verifier_acc = self.metrics_logger.get_latest_metric(
                            "train",
                            "verifier",
                            "verifier_accuracy",
                            phase=self.state_tracker.phase,  # Pass phase
                        )
                        loss_value = self.metrics_logger.get_latest_metric(
                            "train",
                            "sneaky_prover",
                            "loss",
                            phase=self.state_tracker.phase,  # Pass phase
                        )
                        progress_metrics["loss"] = f"{loss_value:.4f}" if loss_value is not None else "N/A"
                        progress_metrics["v_acc"] = (
                            f"{latest_verifier_acc:.3f}" if latest_verifier_acc is not None else "N/A"
                        )

                        # Update progress bar with comprehensive metrics
                        progress_bar.set_postfix(progress_metrics)
                    # Clean up cached tensors
                    torch.cuda.empty_cache()
                    gc.collect()

                    logger.info(f"Finished training step. Now at step = {training_step + 1}")

                self.state_tracker.increment_step()
                training_step += 1

            # End of epoch
            if self.is_main:
                logger.info(f"Prover Training Epoch {epoch + 1}/{num_steps_or_epochs} completed")
                logger.info(f"Training step: {training_step}")
                logger.info(f"Optimizer step: {optimizer_step}")
                logger.info(f"Total steps: {training_step}")
                logger.info(f"Total optimizer steps: {optimizer_step}")
                progress_bar.close()

            self.accelerator_manager.wait_for_everyone()
            _ = self.evaluate()

            # Sync weights to VLLM if provers are used for generation after training
            self.vllm_orchestrator.sync_weights(phase="provers", model_manager=self.model_manager)
            self.accelerator_manager.wait_for_everyone()

            logger.info("Pushing final model to hub after training...")
            self._push_model_to_hub()
            self.accelerator_manager.wait_for_everyone()

            logger.info("Prover training finished.")

    def evaluate(self) -> dict[str, float] | None:
        """Evaluate the prover model using strategy-based approach

        This method leverages the EvaluationStrategy abstraction to eliminate
        if/else branching and provide clean separation of concerns.

        Returns:
            Evaluation metrics dictionary (main process) or None (other processes)
        """
        logger.info("Starting strategy-based evaluation...")

        # Get the model to evaluate
        unwrapped_model = self.accelerator_manager.unwrap_model(
            self.model_manager.get_model("sneaky_prover", prepared=True), key="sneaky_prover"
        )

        # Delegate to evaluation strategy
        result = self.evaluation_strategy.evaluate(
            pipeline=self.pipeline,
            model=unwrapped_model,
            eval_dataloader=self.eval_dataloader,
            model_key="sneaky_prover",
        )

        # Flush evaluation metrics at the same cadence as training metrics
        # This ensures proper timing alignment and distributed aggregation
        self.metrics_logger.flush(phase=self.state_tracker.phase, mode="eval")

        return result

    def _save_tokenizer_and_config(self, local_dir: Path):
        """Save tokenizer and model config to local directory."""
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(local_dir, safe_serialization=True)

        # Save model config
        model = self.model_manager.get_model("sneaky_prover", prepared=True)
        base_model = getattr(model, "module", model)
        cfg = getattr(base_model, "config", None)
        cfg_path = local_dir / "config.json"

        try:
            if cfg is None:
                raise ValueError("model has no .config attribute")
            if hasattr(cfg, "to_json_file"):  # Transformers Config
                cfg.to_json_file(cfg_path)
            elif isinstance(cfg, dict):  # plain dict
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
            else:  # dataclass / other
                with open(cfg_path, "w") as f:
                    json.dump(cfg.__dict__, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️  Could not save config ({type(cfg)}): {e}")

    def _push_checkpoint_to_hub(self, step: int, round: int):
        """Efficient ZeRO-3 checkpointing with HuggingFace Hub integration.

        Saves optimizer states only every third checkpoint to reduce storage overhead.
        All checkpoints include tokenizer and config files for completeness.
        """
        # Determine checkpoint type
        save_optimizer_state = self.checkpoint_count % 3 == 0
        checkpoint_type = "opt-state" if save_optimizer_state else "weights-only"

        # Create paths and tags
        subdir = f"checkpoint-step-{step}-round-{round}-{checkpoint_type}"
        local_ckpt = Path(str(self.config["ckpt_output_dir"])) / subdir
        tag = f"global_step{step}"

        model = self.model_manager.get_model("sneaky_prover", prepared=True)

        if save_optimizer_state:
            # Save full checkpoint including optimizer states (every 3rd checkpoint)
            logger.info(f"Saving full checkpoint with optimizer states at step {step}")
            model.save_checkpoint(str(local_ckpt), tag=tag)  # all ranks
        else:
            # Save weights-only checkpoint using 16-bit model save
            logger.info(f"Saving weights-only checkpoint at step {step}")
            if hasattr(model, "save_16bit_model"):
                model.save_16bit_model(str(local_ckpt))
            else:
                # Fallback: save state dict manually
                local_ckpt.mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                cloned_state_dict = _clone_tensors_for_safetensors(state_dict)
                save_file(cloned_state_dict, local_ckpt / SAFE_WEIGHTS_NAME)

        self.accelerator_manager.wait_for_everyone()

        # Save tokenizer and config for all checkpoint types (main process only)
        if self.is_main:
            self._save_tokenizer_and_config(local_ckpt)

        self.accelerator_manager.wait_for_everyone()

        # HuggingFace Hub upload (preserve existing logic)
        if not (self.config["push_to_hub"] and self.is_main):
            return

        # Create appropriate .gitignore for checkpoint type
        gitignore_content = "zero_*/*\nglobal_rank*/\n" if save_optimizer_state else ""
        (local_ckpt / ".gitignore").write_text(gitignore_content)

        # Create repository with appropriate naming
        repo_id = f"{self.config['base_repo_id']}_step-{step}"
        if save_optimizer_state:
            repo_id += "_opt-state"

        HfApi().create_repo(repo_id, repo_type="model", exist_ok=True)

        # Upload with appropriate ignore patterns
        ignore_patterns = ["zero_*", "global_rank*"] if save_optimizer_state else []
        upload_folder(
            folder_path=str(local_ckpt),
            path_in_repo=".",  # push to repo root
            repo_id=repo_id,
            commit_message=f"Checkpoint: {subdir} ({checkpoint_type})",
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"✅ pushed {checkpoint_type} checkpoint {subdir} to https://huggingface.co/{repo_id}")

    def _push_model_to_hub(self):
        """Pushes the final model to the HuggingFace Hub in two versions:
        1. Full checkpoint with optimizer states.
        2. Weights-only 16-bit model for inference.
        """
        model = self.model_manager.get_model("sneaky_prover", prepared=True)
        self.accelerator_manager.wait_for_everyone()

        # === 1. PUSH FULL CHECKPOINT WITH OPTIMIZER STATES ===
        if self.is_main:
            logger.info("--- Pushing final model with optimizer states ---")

        final_dir_with_opt = Path(str(self.config["ckpt_output_dir"])) / "final-model-with-opt-state"

        # Save full checkpoint with optimizer states using DeepSpeed's method
        if hasattr(model, "save_checkpoint"):
            model.save_checkpoint(str(final_dir_with_opt), tag="final")
        else:
            logger.warning("`save_checkpoint` not available. Skipping optimizer state save.")
            # If we can't save the full checkpoint, we can still attempt the weights-only save.
            self._push_weights_only_model_to_hub()
            return

        self.accelerator_manager.wait_for_everyone()

        if self.is_main:
            self._save_tokenizer_and_config(final_dir_with_opt)

        self.accelerator_manager.wait_for_everyone()

        if self.config["push_to_hub"] and self.is_main:
            repo_id_with_opt = f"{self.config['base_repo_id']}_final_opt-state"
            logger.info(f"Pushing full checkpoint to {repo_id_with_opt}")

            gitignore_content = "zero_*/*\nglobal_rank*/\n"
            (final_dir_with_opt / ".gitignore").write_text(gitignore_content)

            HfApi().create_repo(repo_id_with_opt, repo_type="model", exist_ok=True)
            upload_folder(
                folder_path=str(final_dir_with_opt),
                path_in_repo=".",
                repo_id=repo_id_with_opt,
                commit_message="🚀 End of training – full checkpoint with optimizer states",
                ignore_patterns=["zero_*", "global_rank*"],
            )
            logger.info(f"✅ Pushed full checkpoint to https://huggingface.co/{repo_id_with_opt}")

        self.accelerator_manager.wait_for_everyone()

        # === 2. PUSH WEIGHTS-ONLY MODEL ===
        self._push_weights_only_model_to_hub()

    def _push_weights_only_model_to_hub(self):
        """Pushes a weights-only version of the final model to HuggingFace Hub."""
        if self.is_main:
            logger.info("--- Pushing final weights-only model ---")

        model = self.model_manager.get_model("sneaky_prover", prepared=True)
        final_dir_weights_only = Path(str(self.config["ckpt_output_dir"])) / "final-model-weights-only"

        # Use DeepSpeed's native save_16bit_model for final model
        if hasattr(model, "save_16bit_model"):
            model.save_16bit_model(str(final_dir_weights_only))
        else:
            # Fallback for non-DeepSpeed models
            logger.warning("`save_16bit_model` not found, falling back to manual state_dict save.")
            if self.is_main:
                final_dir_weights_only.mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                cloned_state_dict = _clone_tensors_for_safetensors(state_dict)
                save_file(cloned_state_dict, final_dir_weights_only / SAFE_WEIGHTS_NAME)

        self.accelerator_manager.wait_for_everyone()

        if self.is_main:
            self._save_tokenizer_and_config(final_dir_weights_only)

        self.accelerator_manager.wait_for_everyone()

        if self.config["push_to_hub"] and self.is_main:
            repo_id_weights_only = f"{self.config['base_repo_id']}_final"
            logger.info(f"Pushing weights-only model to {repo_id_weights_only}")

            HfApi().create_repo(repo_id_weights_only, repo_type="model", exist_ok=True)

            upload_folder(
                folder_path=str(final_dir_weights_only),
                path_in_repo=".",
                repo_id=repo_id_weights_only,
                commit_message="🚀 End of training – final FP16 weights",
                ignore_patterns=["zero_*", "global_rank*"],
            )
            logger.info(f"🏁 Pushed weights-only model to https://huggingface.co/{repo_id_weights_only}")
