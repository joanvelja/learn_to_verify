import gc
import logging
from typing import Literal

import torch
from deepspeed.utils import (
    safe_get_full_fp32_param,
    safe_get_full_grad,
)
from tqdm import tqdm

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
from pvg.strategies.implementations.reward_strategies import SanityCheckRewardStrategy
from pvg.strategies.implementations.verification_strategies import (
    create_verification_strategy,
)

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


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

        self.tokenizer = model_manager.get_tokenizer()
        self.train_dataloader = data_manager.dataloaders["provers"]["sneaky_prover"][
            "train_dataloader"
        ]
        self.eval_dataloader = data_manager.dataloaders["provers"]["sneaky_prover"][
            "eval_dataloader"
        ]
        self.dataset_type = dataset_type
        self.total_steps = 0
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.state_tracker = state_tracker

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

        self.reward_strategy = SanityCheckRewardStrategy(
            metrics_logger=metrics_logger,
            accelerator_manager=accelerator_manager,
            grpo=grpo,
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
        )

        self.batch_processor = BatchProcessor(
            tokenizer=self.tokenizer,
            accelerator_manager=accelerator_manager,
            rl_config=args.rl,
            dataset_type=dataset_type,
            buffer_completions=True,
        )

        self.metrics_processor = MetricsProcessor()

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

        self.is_main = self.accelerator_manager.get_state_property(
            property_name="is_main_process"
        )

    def train(self, num_steps_or_epochs: int = 1):
        """Train the prover model"""
        training_step = 0
        optimizer_step = 0

        for epoch in range(num_steps_or_epochs):
            logger.info(
                f"Starting Prover Training Epoch {epoch + 1}/{num_steps_or_epochs}"
            )

            if self.is_main:
                progress_bar = tqdm(
                    self.train_dataloader, desc=f"Prover Epoch {epoch + 1}"
                )
            else:
                progress_bar = self.train_dataloader

            for batch_idx, raw_batch_data in enumerate(progress_bar):
                logger.info(
                    f"Metrics storage - StateTracker step: {self.state_tracker.step}, Training step: {training_step} - Optimizer step: {optimizer_step} - Batch index: {batch_idx}"
                )
                # 1. Process batch
                batch_inputs = self.pipeline.process_batch(
                    raw_batch_data,
                    training_step,
                    self.gradient_accumulation_steps,
                    use_buffering=True,
                )

                # 2. Compute training step
                policy_model = self.model_manager.get_model(
                    key="sneaky_prover", prepared=True
                )
                training_step_result = self.pipeline.compute_training_step(
                    policy_model, batch_inputs, mode="train"
                )
                # 3. Compute loss
                loss_result = training_step_result.loss_result
                batch_metrics = training_step_result.batch_metrics

                # 4. Compute gradients
                loss = loss_result.loss.item()
                p0 = safe_get_full_fp32_param(next(policy_model.parameters())).clone()

                # ============================================================================
                # PRE-BACKWARD DEBUG SECTION
                # ============================================================================
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info("🔍 PRE-BACKWARD DEBUG")
                    logger.info(f"🏷️  Loss value: {loss.item():.6f}")
                    logger.info(f"🏷️  Loss requires_grad: {loss.requires_grad}")
                    logger.info(f"🏷️  Loss grad_fn: {loss.grad_fn}")

                    # Check parameter values before backward
                    p0_norm = torch.norm(p0)
                    logger.info(f"📊 Parameter p0 norm BEFORE backward: {p0_norm:.6f}")

                self.accelerator_manager.backward(loss, key="sneaky_prover")
                self.accelerator_manager.wait_for_everyone()

                # ============================================================================
                # POST-BACKWARD DEBUG SECTION
                # ============================================================================
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info("🔄 POST-BACKWARD DEBUG")

                    # Check if any gradients were computed
                    grad_count = 0
                    total_grad_norm = 0.0

                    for name, param in policy_model.named_parameters():
                        if param.grad is not None:
                            grad_count += 1
                            try:
                                # Try to get gradient norm (may fail with ZeRO-3)
                                full_grad = safe_get_full_grad(param)
                                if full_grad is not None:
                                    total_grad_norm += torch.norm(full_grad).item() ** 2
                            except Exception as e:
                                logger.warning(
                                    f"⚠️  Could not compute gradient norm: {e}"
                                )
                                pass

                    logger.info(f"📊 Parameters with gradients: {grad_count}")
                    if total_grad_norm > 0:
                        logger.info(
                            f"📊 Total gradient norm: {total_grad_norm**0.5:.6f}"
                        )
                    else:
                        logger.info(
                            "📊 Could not compute gradient norms (expected with ZeRO-3)"
                        )

                    # Check DeepSpeed engine state
                    if hasattr(policy_model, "optimizer"):
                        ds_optimizer = policy_model.optimizer
                        logger.info(
                            f"🚀 DeepSpeed optimizer learning rate: {ds_optimizer.param_groups[0]['lr']}"
                        )
                        logger.info(
                            f"🚀 DeepSpeed optimizer state_dict keys: {list(ds_optimizer.state_dict().keys())}"
                        )

                    # Check if loss computation graph is intact
                    logger.info(
                        f"🔍 Loss computation graph exists: {loss.grad_fn is not None}"
                    )

                # 5. Store metrics
                self.metrics_logger.store_metric(
                    mode="train",
                    model="sneaky_prover",
                    name="loss",
                    value=loss.item() * self.gradient_accumulation_steps,
                    phase=self.state_tracker.phase,  # Pass phase
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
                is_last_step_in_batch = (training_step + 1) == len(
                    self.train_dataloader
                )

                is_accumulation_boundary = (
                    training_step + 1
                ) % self.gradient_accumulation_steps == 0
                is_sync_step = is_last_step_in_batch or is_accumulation_boundary
                if is_sync_step:  # A LOOOOT OF STUFF HAPPENS HERE
                    optimizer_step += 1
                    # ============================================================================
                    # DEEPSPEED OPTIMIZER STEP
                    # ============================================================================
                    if hasattr(policy_model, "step"):
                        # DeepSpeed engine - use engine's step method
                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.info("🚀 Using DeepSpeed engine step() method")

                            # Additional debugging for DeepSpeed step
                            if hasattr(policy_model, "optimizer"):
                                ds_optimizer = policy_model.optimizer
                                logger.info(
                                    f"🚀 Pre-step: DeepSpeed optimizer step count: {getattr(ds_optimizer, '_step', 'unknown')}"
                                )
                                logger.info(
                                    f"🚀 Pre-step: Learning rate: {ds_optimizer.param_groups[0]['lr']}"
                                )

                        # DeepSpeed engine handles optimizer step internally
                        step_result = policy_model.step()

                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.info(f"🚀 DeepSpeed step result: {step_result}")

                            # Check if step count increased
                            if hasattr(policy_model, "optimizer"):
                                ds_optimizer = policy_model.optimizer
                                logger.info(
                                    f"🚀 Post-step: DeepSpeed optimizer step count: {getattr(ds_optimizer, '_step', 'unknown')}"
                                )

                        # Get scheduler for learning rate updates
                        scheduler = self.optimizer_scheduler_manager.get_scheduler(
                            "sneaky_prover"
                        )
                        if scheduler is not None:
                            scheduler.step()
                    else:
                        # Standard optimizer path (non-DeepSpeed)
                        optimizer = self.optimizer_scheduler_manager.get_optimizer(
                            "sneaky_prover"
                        )
                        scheduler = self.optimizer_scheduler_manager.get_scheduler(
                            "sneaky_prover"
                        )

                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.info("🚀 Using standard optimizer step() method")

                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()

                    # ============================================================================
                    # POST-OPTIMIZER-STEP DEBUG SECTION (ZeRO-3 COMPATIBLE)
                    # ============================================================================
                    if self.accelerator_manager.get_state_property("is_main_process"):
                        logger.info("✅ POST-OPTIMIZER-STEP ANALYSIS")
                        logger.info("-" * 40)

                    # For ZeRO-3, we need a different approach to check parameter updates
                    # Instead of comparing raw parameters, check optimizer state and step counts

                    param_updated = False
                    total_param_change = 0.0

                    # Method 1: Try to get updated parameters safely
                    try:
                        p1 = safe_get_full_fp32_param(next(policy_model.parameters()))
                        param_diff = torch.norm(p1 - p0)

                        logger.info(f"📊 Parameter p0 norm: {torch.norm(p0):.6f}")
                        logger.info(f"📊 Parameter p1 norm: {torch.norm(p1):.6f}")
                        logger.info(f"📊 Parameter difference norm: {param_diff:.6f}")

                        if (
                            param_diff > 1e-8
                        ):  # Use small threshold for floating point comparison
                            param_updated = True
                            total_param_change = param_diff.item()
                            logger.info(
                                f"✅ Parameters updated via direct comparison: {param_diff:.6f}"
                            )
                        else:
                            logger.warning(
                                f"⚠️  Direct parameter comparison shows no change: {param_diff:.6f}"
                            )

                    except Exception as e:
                        logger.warning(f"⚠️  Could not directly compare parameters: {e}")

                    # Method 2: Check DeepSpeed optimizer state
                    if hasattr(policy_model, "optimizer"):
                        ds_optimizer = policy_model.optimizer

                        # Check if optimizer step count increased
                        current_step = getattr(ds_optimizer, "_step", 0)
                        logger.info(
                            f"📊 DeepSpeed optimizer step count: {current_step}"
                        )

                        # Check if optimizer has momentum buffers (indicates it's doing work)
                        state_dict = ds_optimizer.state_dict()
                        if "state" in state_dict and state_dict["state"]:
                            logger.info(
                                f"📊 Optimizer has state (momentum buffers): {len(state_dict['state'])} parameter groups"
                            )
                            param_updated = True
                        else:
                            logger.warning(
                                "⚠️  Optimizer state is empty - may indicate no updates"
                            )

                    # Method 3: Check if gradients were consumed (cleared after step)
                    grad_count_after_step = 0
                    for param in policy_model.parameters():
                        if param.grad is not None:
                            grad_count_after_step += 1

                    logger.info(
                        f"📊 Parameters with gradients after step: {grad_count_after_step}"
                    )
                    if grad_count_after_step == 0:
                        logger.info("✅ Gradients were cleared after step (good sign)")
                    else:
                        logger.warning("⚠️  Some gradients remain after step")

                    # Final assessment
                    if param_updated or total_param_change > 0:
                        logger.info("✅ PARAMETERS SUCCESSFULLY UPDATED!")
                        logger.info(
                            f"✅ Total parameter change: {total_param_change:.8f}"
                        )
                    else:
                        logger.error("💥 NO PARAMETER UPDATES DETECTED!")
                        logger.error("💥 Possible issues:")
                        logger.error("💥 1. Gradients not computed properly")
                        logger.error("💥 2. Learning rate is zero")
                        logger.error("💥 3. DeepSpeed engine not configured correctly")
                        logger.error("💥 4. Gradient clipping removing all gradients")

                        # Don't raise assertion error, just log the issue
                        # raise AssertionError("Weights didn't move – optimizer stepping failed")

                    self.accelerator_manager.wait_for_everyone()

                    # Sync vllm weights -- Update means change vllm weights for inference as well
                    self.vllm_orchestrator.sync_weights(
                        phase="provers", model_manager=self.model_manager
                    )
                    self.accelerator_manager.wait_for_everyone()

                    # Log step metrics after optimizer step
                    self.metrics_logger.flush(
                        phase=self.state_tracker.phase, mode="train"
                    )

                    # Update state tracker
                    self.state_tracker.increment_step()

                    # Evaluate every 100 micro-batches (not optimizer steps)
                    if training_step % 100 == 0:
                        self.vllm_orchestrator.sync_weights(
                            phase="provers", model_manager=self.model_manager
                        )
                        self.accelerator_manager.wait_for_everyone()
                        _ = self.evaluate()

                    if self.is_main:
                        latest_verifier_acc = self.metrics_logger.get_latest_metric(
                            "train",
                            "verifier",
                            "verifier_accuracy",
                            phase=self.state_tracker.phase,  # Pass phase
                        )
                        progress_bar.set_postfix(
                            loss=self.metrics_logger.get_latest_metric(
                                "train",
                                "sneaky_prover",
                                "loss",
                                phase=self.state_tracker.phase,  # Pass phase
                            ),
                            v_acc=(
                                f"{latest_verifier_acc:.3f}"
                                if latest_verifier_acc is not None
                                else "N/A"
                            ),
                        )
                    # Clean up cached tensors
                    torch.cuda.empty_cache()
                    gc.collect()

                    training_step += 1

            # End of epoch
            if self.is_main:
                logger.info(
                    f"Prover Training Epoch {epoch + 1}/{num_steps_or_epochs} completed"
                )
                logger.info(f"Training step: {training_step}")
                logger.info(f"Optimizer step: {optimizer_step}")
                logger.info(f"Total steps: {training_step}")
                logger.info(f"Total optimizer steps: {optimizer_step}")
                progress_bar.close()

            self.accelerator_manager.wait_for_everyone()
            _ = self.evaluate()

            # Sync weights to VLLM if provers are used for generation after training
            self.vllm_orchestrator.sync_weights(
                phase="provers", model_manager=self.model_manager
            )
            self.accelerator_manager.wait_for_everyone()

            logger.info("Attempting to push models to hub after training...")
            model_keys_to_push = ["sneaky_prover"]
            for model_key in model_keys_to_push:
                try:
                    model_to_push = self.model_manager.get_model(
                        model_key, prepared=True
                    )
                    if model_to_push is not None:
                        self._push_model_to_hub()
                    else:
                        logger.warning(
                            f"Model with key '{model_key}' not found in ModelManager. Skipping push."
                        )
                except Exception as e:
                    logger.error(
                        f"An error occurred while preparing to push {model_key} to hub: {str(e)}"
                    )

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
        policy_model = self.model_manager.get_model("sneaky_prover", prepared=True)

        # Delegate to evaluation strategy
        return self.evaluation_strategy.evaluate(
            pipeline=self.pipeline,
            model=policy_model,
            eval_dataloader=self.eval_dataloader,
            model_key="sneaky_prover",
        )

    def _push_model_to_hub(self) -> None:
        """
        Push the trained model to the Hugging Face model hub with proper error handling.
        """

        # Only push from the main process
        if not self.is_main:
            logger.info("Not the main process, skipping push to hub")
            return

        try:
            prover_model_name = f"jvelja/sneaky_prover_round_{self.state_tracker.round}"
            # Unwrap the model before pushing to avoid distributed training issues
            unwrapped_model = self.accelerator_manager.unwrap_model(
                self.model_manager.get_model("sneaky_prover", prepared=True),
                key="sneaky_prover",
            )

            # Push the unwrapped model
            unwrapped_model.push_to_hub(repo_id=prover_model_name)
            logger.info(
                f"Sneaky Prover model successfully pushed to the hub as {prover_model_name}."
            )
            # Tokenizer should be pushed to the same repo
            self.tokenizer.push_to_hub(repo_id=prover_model_name)
        except Exception as e:
            logger.error(f"Failed to push model to hub: {str(e)}")
            raise e
