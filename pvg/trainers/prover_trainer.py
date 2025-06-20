import gc
import logging
import os
import shutil
from contextlib import nullcontext
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Literal

import deepspeed
import torch
from deepspeed.utils import (
    safe_get_full_fp32_param,
    safe_get_full_grad,
)
from deepspeed.utils.zero_to_fp32 import (
    convert_zero_checkpoint_to_fp32_state_dict,
    get_fp32_state_dict_from_zero_checkpoint,
)
from huggingface_hub import HfApi, create_repo
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
from pvg.strategies.implementations.reward_strategies import SanityCheckRewardStrategy, TierBasedRewardStrategy
from pvg.strategies.implementations.verification_strategies import (
    create_verification_strategy,
)
from pvg.utils.verifier_performance import VerifierPerformanceTracker

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
            "checkpoint_interval": getattr(args, "checkpoint_interval", 1),
        }

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

        # self.reward_strategy = SanityCheckRewardStrategy(
        #     metrics_logger=metrics_logger,
        #     accelerator_manager=accelerator_manager,
        #     grpo=grpo,
        #     metrics_processor=self.metrics_processor,
        # )

        self.reward_strategy = TierBasedRewardStrategy(
            verifier_tracker=VerifierPerformanceTracker(),
            metrics_logger=metrics_logger,
            accelerator_manager=accelerator_manager,
            grpo=grpo,
            metrics_processor=self.metrics_processor,
            state_tracker=state_tracker,
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
                policy_model = self.model_manager.get_model(key="sneaky_prover", prepared=True)
                unwrapped_model = self.accelerator_manager.unwrap_model(policy_model, key="sneaky_prover")
                training_step_result = self.pipeline.compute_training_step(unwrapped_model, batch_inputs, mode="train")
                # 3. Compute loss
                loss = training_step_result.loss_result.loss
                batch_metrics = training_step_result.batch_metrics

                # 4. Compute gradients
                p0 = safe_get_full_fp32_param(next(policy_model.parameters())).clone()

                # ============================================================================
                # PRE-BACKWARD DEBUG SECTION
                # ============================================================================
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info("🔍 PRE-BACKWARD DEBUG")
                    logger.info(f"🏷️  Loss value: {loss:.6f}")
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
                                logger.warning(f"⚠️  Could not compute gradient norm: {e}")
                                pass

                    logger.info(f"📊 Parameters with gradients: {grad_count}")
                    if total_grad_norm > 0:
                        logger.info(f"📊 Total gradient norm: {total_grad_norm**0.5:.6f}")
                    else:
                        logger.info("📊 Could not compute gradient norms (expected with ZeRO-3)")

                    # Check DeepSpeed engine state
                    if hasattr(policy_model, "optimizer"):
                        ds_optimizer = policy_model.optimizer
                        logger.info(f"🚀 DeepSpeed optimizer learning rate: {ds_optimizer.param_groups[0]['lr']}")
                        logger.info(f"🚀 DeepSpeed optimizer state_dict keys: {list(ds_optimizer.state_dict().keys())}")

                    # Check if loss computation graph is intact
                    logger.info(f"🔍 Loss computation graph exists: {loss.grad_fn is not None}")

                # 5. Store metrics
                self.metrics_logger.store_metric(
                    mode="train",
                    model="sneaky_prover",
                    name="loss",
                    value=loss * self.gradient_accumulation_steps,
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
                is_last_step_in_batch = (training_step + 1) == len(self.train_dataloader)

                is_accumulation_boundary = (training_step + 1) % self.gradient_accumulation_steps == 0
                is_sync_step = is_last_step_in_batch or is_accumulation_boundary
                if is_sync_step:  # A LOOOOT OF STUFF HAPPENS HERE
                    optimizer_step += 1
                    # ============================================================================
                    # DEEPSPEED OPTIMIZER STEP
                    # ============================================================================
                    if False:  # TODO: Remove this once we have a working DeepSpeed engine
                        # DeepSpeed engine - use engine's step method
                        if self.accelerator_manager.get_state_property("is_main_process"):
                            logger.info("🚀 Using DeepSpeed engine step() method")

                            # Additional debugging for DeepSpeed step
                            if hasattr(policy_model, "optimizer"):
                                ds_optimizer = policy_model.optimizer
                                logger.info(
                                    f"🚀 Pre-step: DeepSpeed optimizer step count: {getattr(ds_optimizer, '_step', 'unknown')}"
                                )
                                logger.info(f"🚀 Pre-step: Learning rate: {ds_optimizer.param_groups[0]['lr']}")

                        # DeepSpeed engine handles optimizer step internally
                        step_result = policy_model.step()

                        if self.accelerator_manager.get_state_property("is_main_process"):
                            logger.info(f"🚀 DeepSpeed step result: {step_result}")

                            # Check if step count increased
                            if hasattr(policy_model, "optimizer"):
                                ds_optimizer = policy_model.optimizer
                                logger.info(
                                    f"🚀 Post-step: DeepSpeed optimizer step count: {getattr(ds_optimizer, '_step', 'unknown')}"
                                )

                        # Get scheduler for learning rate updates
                        scheduler = self.optimizer_scheduler_manager.get_scheduler("sneaky_prover")
                        if scheduler is not None:
                            scheduler.step()
                    else:
                        # Standard optimizer path (non-DeepSpeed)
                        optimizer = self.optimizer_scheduler_manager.get_optimizer("sneaky_prover")
                        scheduler = self.optimizer_scheduler_manager.get_scheduler("sneaky_prover")

                        if self.accelerator_manager.get_state_property("is_main_process"):
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

                        if param_diff > 1e-8:  # Use small threshold for floating point comparison
                            param_updated = True
                            total_param_change = param_diff.item()
                            logger.info(f"✅ Parameters updated via direct comparison: {param_diff:.6f}")
                        else:
                            logger.warning(f"⚠️  Direct parameter comparison shows no change: {param_diff:.6f}")

                    except Exception as e:
                        logger.warning(f"⚠️  Could not directly compare parameters: {e}")

                    # Method 2: Check DeepSpeed optimizer state
                    if hasattr(policy_model, "optimizer"):
                        ds_optimizer = policy_model.optimizer

                        # Check if optimizer step count increased
                        current_step = getattr(ds_optimizer, "_step", 0)
                        logger.info(f"📊 DeepSpeed optimizer step count: {current_step}")

                        # Check if optimizer has momentum buffers (indicates it's doing work)
                        state_dict = ds_optimizer.state_dict()
                        if "state" in state_dict and state_dict["state"]:
                            logger.info(
                                f"📊 Optimizer has state (momentum buffers): {len(state_dict['state'])} parameter groups"
                            )
                            param_updated = True
                        else:
                            logger.warning("⚠️  Optimizer state is empty - may indicate no updates")

                    # Method 3: Check if gradients were consumed (cleared after step)
                    grad_count_after_step = 0
                    for param in policy_model.parameters():
                        if param.grad is not None:
                            grad_count_after_step += 1

                    logger.info(f"📊 Parameters with gradients after step: {grad_count_after_step}")
                    if grad_count_after_step == 0:
                        logger.info("✅ Gradients were cleared after step (good sign)")
                    else:
                        logger.warning("⚠️  Some gradients remain after step")

                    # Final assessment
                    if param_updated or total_param_change > 0:
                        logger.info("✅ PARAMETERS SUCCESSFULLY UPDATED!")
                        logger.info(f"✅ Total parameter change: {total_param_change:.8f}")
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

                    # ============================================================================
                    # CHECKPOINTING TO HUB
                    # ============================================================================
                    if optimizer_step > 0 and optimizer_step % self.config["checkpoint_interval"] == 0:
                        self.accelerator_manager.wait_for_everyone()
                        if self.accelerator_manager.get_state_property("is_main_process"):
                            logger.info(f"🚀 Optimizer step {optimizer_step}, pushing checkpoint to hub.")
                        
                        self._push_checkpoint_to_hub(step=optimizer_step)
                        self.accelerator_manager.wait_for_everyone()

                    # Sync vllm weights -- Update means change vllm weights for inference as well
                    self.vllm_orchestrator.sync_weights(phase="provers", model_manager=self.model_manager)
                    self.accelerator_manager.wait_for_everyone()

                    # Log step metrics after optimizer step
                    self.metrics_logger.flush(phase=self.state_tracker.phase, mode="train")

                    # Update state tracker
                    self.state_tracker.increment_step()

                    # Evaluate every 100 micro-batches (not optimizer steps)
                    if training_step % 100 == 0:
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
                        progress_metrics["loss"] = self.metrics_logger.get_latest_metric(
                            "train",
                            "sneaky_prover",
                            "loss",
                            phase=self.state_tracker.phase,  # Pass phase
                        )
                        progress_metrics["v_acc"] = (
                            f"{latest_verifier_acc:.3f}" if latest_verifier_acc is not None else "N/A"
                        )

                        # Update progress bar with comprehensive metrics
                        progress_bar.set_postfix(progress_metrics)
                    # Clean up cached tensors
                    torch.cuda.empty_cache()
                    gc.collect()

                    training_step += 1

                training_step += 1
                self.state_tracker.increment_step()

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
        policy_model = self.model_manager.get_model("sneaky_prover", prepared=True)

        # Delegate to evaluation strategy
        return self.evaluation_strategy.evaluate(
            pipeline=self.pipeline,
            model=policy_model,
            eval_dataloader=self.eval_dataloader,
            model_key="sneaky_prover",
        )

    def _push_checkpoint_to_hub(self, *, step: int) -> None:
        logger.info(f"[push_ckpt] Starting push_checkpoint_to_hub for step={step}")

        tag      = f"global_step{step}"
        out_root = self.config.get("output_dir", "./checkpoints")
        repo_id  = f"jvelja/sneaky_prover_round_{self.state_tracker.round}_step_{step}_ckpt"

        logger.info(f"[push_ckpt] tag={tag}, out_root={out_root}, repo_id={repo_id}")

        logger.info("[push_ckpt] Got model from model_manager")
        engine = self.model_manager.get_model("sneaky_prover", prepared=True)
        if not isinstance(engine, deepspeed.DeepSpeedEngine):
            logger.error("Expected DeepSpeedEngine (ZeRO-3) but got %s", type(engine))
            return
        
        logger.info(f"[push_ckpt] Unwrapped model, engine type: {type(engine)}")

        # ------------------ 1️⃣  EVERY RANK: save ZeRO shards ------------------
        ckpt_dir = os.path.join(out_root, tag)
        logger.info(f"[push_ckpt] Checkpoint dir: {ckpt_dir}")
        if isinstance(engine, deepspeed.DeepSpeedEngine):
            logger.info("[push_ckpt] Engine is DeepSpeedEngine, saving checkpoint shards...")
            engine.save_checkpoint(ckpt_dir, tag=tag)
            logger.info("[push_ckpt] DeepSpeed checkpoint saved.")
        else:
            logger.info("[push_ckpt] Engine is not DeepSpeedEngine, skipping ZeRO shard save.")
        logger.info("[push_ckpt] Waiting for everyone after ZeRO shard save...")
        self.accelerator_manager.wait_for_everyone()
        logger.info("[push_ckpt] wait_for_everyone complete.")

        tmp_bf16 = os.path.join(out_root, f"_gather16_rank{engine.global_rank}")
        ok_fast  = engine.save_16bit_model(str(tmp_bf16), save_filename="model.safetensors")

        # barrier so rank-0 knows whether gather succeeded (if not, we’ll fall back)
        self.accelerator_manager.wait_for_everyone()

        # ------------------ 2️⃣  RANK-0: consolidate & push --------------------
        if not self.is_main:
            logger.info("[push_ckpt] Not main process, returning.")
            return

        logger.info("[push_ckpt] Main process proceeding to consolidation and push.")
        with TemporaryDirectory() as tmpdir:
            logger.info(f"[push_ckpt] Created TemporaryDirectory: {tmpdir}")
            hf_dir = os.path.join(tmpdir, "hf_ckpt")
            logger.info(f"[push_ckpt] HuggingFace checkpoint dir: {hf_dir}")

            if isinstance(engine, deepspeed.DeepSpeedEngine):
                logger.info(f"[push_ckpt] save_16bit_model returned: {ok_fast}")
                #
                # Requires `"stage3_gather_16bit_weights_on_model_save": true`
                # in your deepspeed config. If that’s OFF, `ok` will be False.
                #
                if ok_fast:
                    # rank-0 copies the already-gathered bf16 file
                    os.rename(os.path.join(tmp_bf16, "model.safetensors"), os.path.join(hf_dir, "model.safetensors"))
                    # remove per-rank dirs
                    for p in os.listdir(out_root):
                        if p.startswith("_gather16_rank"):
                            shutil.rmtree(os.path.join(out_root, p), ignore_errors=True)

                else:
                    # fallback: stream fp32 → disk → cast to bf16
                    fp32_file = os.path.join(tmp_bf16, "fp32.safetensors")
                    convert_zero_checkpoint_to_fp32_state_dict(
                        checkpoint_dir      = str(ckpt_dir),
                        output_file         = str(fp32_file),
                        tag                 = tag,
                        safe_serialization  = True,
                        max_shard_size      = "5GB",
                    )
                    from transformers import AutoConfig, AutoModelForCausalLM
                    import torch
                    cfg   = engine.module.config
                    model = engine.module.__class__.from_config(cfg, torch_dtype=torch.bfloat16)
                    model.load_state_dict(torch.load(fp32_file, map_location="cpu"), strict=False)
                    model.save_pretrained(hf_dir, safe_serialization=True, max_shard_size="4GB")

                # config + tokenizer
                engine.module.config.to_json_file(os.path.join(hf_dir, "config.json"))
                self.tokenizer.save_pretrained(hf_dir)

                # Hub upload – keyword-only API
                create_repo(repo_id, exist_ok=True, repo_type="model")
                HfApi().upload_folder(
                    repo_id        = repo_id,
                    folder_path    = str(hf_dir),
                    commit_message = f"ckpt {tag} (bf16, ZeRO-3)",
                )

            torch.cuda.empty_cache(); gc.collect()

    def _push_model_to_hub(self) -> None:
        """
        Push the final trained model to the Hugging Face model hub.
        Direct push using ZeRO-3 parameter gathering - no local saves.
        """
        # Only push from the main process
        if not self.is_main:
            return

        # Check if push_to_hub is enabled
        if not self.config["push_to_hub"]:
            logger.info("push_to_hub is disabled, skipping")
            return

        repo_id = f"jvelja/sneaky_prover_round_{self.state_tracker.round}"
        commit_message = f"End of training for round {self.state_tracker.round}"

        # Get the model for saving
        prepared_model = self.model_manager.get_model("sneaky_prover", prepared=True)
        accelerator = self.accelerator_manager.get_accelerator(key="sneaky_prover")
        unwrapped_model = accelerator.unwrap_model(prepared_model)

        # Determine if we're using ZeRO-3
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext

        try:
            # Push directly to hub using gathered parameters
            if zero_stage_3:
                # For ZeRO-3, gather all parameters at once and push
                all_params = list(unwrapped_model.parameters())
                with gather_if_zero3(all_params, modifier_rank=0):
                    if self.is_main:
                        unwrapped_model.push_to_hub(repo_id=repo_id, commit_message=commit_message)
                        self.tokenizer.push_to_hub(repo_id=repo_id, commit_message=commit_message)
                        logger.info(f"Final model pushed to hub as {repo_id}")
            else:
                # For non-ZeRO-3, just push directly
                if self.is_main:
                    unwrapped_model.push_to_hub(repo_id=repo_id, commit_message=commit_message)
                    self.tokenizer.push_to_hub(repo_id=repo_id, commit_message=commit_message)
                    logger.info(f"Final model pushed to hub as {repo_id}")
            
            self.accelerator_manager.wait_for_everyone()
                
        except Exception as e:
            logger.error(f"Failed to push final model to hub: {e}")
            raise
