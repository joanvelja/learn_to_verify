import gc
import json
import logging
import os
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import HfApi, upload_folder
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

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
            "checkpoint_interval": getattr(args, "checkpoint_interval", 150),
            "ckpt_output_dir": getattr(args.training_sneaky_prover, "ckpt_output_dir", "./sneaky_prover_ckpt"),
            "repo_id": None,
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
                        repo_id = f"jvelja/pvg-prover-sneaky-round-{round}_step-{optimizer_step}"
                        self.config["repo_id"] = repo_id
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
                    if (training_step + 1) % 100 == 0:
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
        policy_model = self.model_manager.get_model("sneaky_prover", prepared=True)

        # Delegate to evaluation strategy
        return self.evaluation_strategy.evaluate(
            pipeline=self.pipeline,
            model=policy_model,
            eval_dataloader=self.eval_dataloader,
            model_key="sneaky_prover",
        )

    def _save_one_model(
        self,
        model,
        output_dir: str,
        prefer_safe: bool = True,
    ) -> Path | None:
        """
        Dump a full-weight file in `output_dir` and return its path.

        • Works for plain PyTorch, ZeRO-2 and ZeRO-3
        • Requires DS config flag `stage3_gather_fp16_weights_on_model_save`
        """
        os.makedirs(output_dir, exist_ok=True)
        weight_name = SAFE_WEIGHTS_NAME if prefer_safe else WEIGHTS_NAME
        weight_path = Path(output_dir) / weight_name

        # DeepSpeed path ────────────────────────────────────────────────
        if hasattr(model, "save_fp16_model"):  # ZeRO-2/3 engine
            model.save_fp16_model(str(output_dir), weight_name)

        # Accelerate unwrap → ordinary Module ───────────────────────────
        else:
            torch.save(model.state_dict(), weight_path)

        # ────────────────── 2. save meta only once  ──────────────
        if not self.is_main:
            return None  # other ranks are done

        # a) tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir, safe_serialization=prefer_safe)

        # b) config (robust to anything JSON-ish)
        base_model = getattr(model, "module", model)  # unwrap DS, DDP, etc.
        cfg = getattr(base_model, "config", None)
        cfg_path = Path(output_dir) / "config.json"

        try:
            if cfg is None:
                raise ValueError("model has no .config attribute")

            # Transformers PretrainedConfig
            if hasattr(cfg, "to_json_file"):
                cfg.to_json_file(cfg_path)

            # plain dict
            elif isinstance(cfg, dict):
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)

            # dataclass or anything else JSON-serialisable
            else:
                with open(cfg_path, "w") as f:
                    json.dump(cfg.__dict__, f, indent=2)

        except Exception as e:
            logger.warning(f"⚠️  Could not save config ({type(cfg)}): {e}")

        return weight_path

    def _push_checkpoint_to_hub(self, step: int, round: int):
        """Equivalent to HF-Trainer's _push_from_checkpoint, but ZeRO-3-safe.

        This version avoids pushing into a subdirectory in the Hugging Face repo,
        instead always pushes to the repo root (overwriting previous checkpoint).
        """
        # paths/tags
        subdir = f"checkpoint-step-{step}-round-{round}"
        local_ckpt = Path(str(self.config["ckpt_output_dir"])) / subdir
        tag = f"global_step{step}"  # must be identical on all ranks

        # -------------  A)  save *training* checkpoint  -------------
        model = self.model_manager.get_model("sneaky_prover", prepared=True)
        model.save_checkpoint(str(local_ckpt), tag=tag)  # <-- ALL RANKS

        # -------------  B)  gather full fp16 weights  ---------------
        self._save_one_model(model, str(local_ckpt))

        # everybody reaches the same point
        self.accelerator_manager.wait_for_everyone()

        # -------------  C)  only rank-0 pushes to the Hub -----------
        if not (self.config["push_to_hub"] and self.is_main):
            return

        (local_ckpt / ".gitignore").write_text("zero_*/*\nglobal_rank*/\n")

        repo_id = self.config["repo_id"]
        HfApi().create_repo(repo_id, repo_type="model", exist_ok=True)

        # Always push to the root of the repo (path_in_repo=".")
        upload_folder(
            folder_path=str(local_ckpt),
            path_in_repo=".",  # push to root, not a subdirectory
            repo_id=repo_id,
            commit_message=f"Checkpoint: {subdir}",
            ignore_patterns=["zero_*", "global_rank*"],
        )
        logger.info(f"✅ pushed checkpoint {subdir} to https://huggingface.co/{repo_id}")

    def _push_model_to_hub(self):
        """Matches HF-Trainer.push_to_hub() but won't hang."""
        final_dir = Path(str(self.config["ckpt_output_dir"])) / "final-model"
        model = self.model_manager.get_model("sneaky_prover", prepared=True)
        self._save_one_model(model, str(final_dir))  # all ranks, safe

        self.accelerator_manager.wait_for_everyone()

        if not (self.config["push_to_hub"] and self.is_main):
            return

        repo_id = self.config["repo_id"]
        upload_folder(
            folder_path=str(final_dir),
            path_in_repo=".",  # overwrite top-level weights
            repo_id=repo_id,
            commit_message="🚀 End of training – full FP16 weights",
            ignore_patterns=["zero_*", "global_rank*"],
        )
        logger.info(f"🏁 final model pushed to https://huggingface.co/{repo_id}")
