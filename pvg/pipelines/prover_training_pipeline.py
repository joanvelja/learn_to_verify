"""
Prover training pipeline

Orchestrates the prover training process using pluggable strategies and components.
This replaces the monolithic _generate_and_score_completions method and related logic
with a clean, modular pipeline.
"""

import logging
from contextlib import nullcontext
from typing import Any, Literal

import torch
from accelerate.utils import broadcast_object_list

from pvg.components import AcceleratorManager, MetricsLogger, ModelManager, StateTracker
from pvg.config.args import RLArgs
from pvg.data_models import RewardResult
from pvg.data_models.training_data import (
    BatchInputs,
    TrainingStepResult,
)
from pvg.processors import BatchProcessor, MetricsProcessor
from pvg.strategies.abstractions import (
    CompletionGenerationStrategy,
    LossComputationStrategy,
    ModelForwardAbstraction,
    RewardCalculationStrategy,
    VerificationStrategy,
)

logger = logging.getLogger(__name__)


class ProverTrainingPipeline:
    """Orchestrates prover training using pluggable strategies

    This pipeline replaces the monolithic training logic with a clean,
    modular approach that uses the Strategy pattern to eliminate if/else
    branching and provide clear separation of concerns.
    """

    def __init__(
        self,
        completion_strategy: CompletionGenerationStrategy,
        verification_strategy: VerificationStrategy,
        reward_strategy: RewardCalculationStrategy,
        loss_strategy: LossComputationStrategy,
        model_forward_strategy: ModelForwardAbstraction,
        batch_processor: BatchProcessor,
        metrics_processor: MetricsProcessor,
        accelerator_manager: AcceleratorManager,
        metrics_logger: MetricsLogger,
        state_tracker: StateTracker,
        model_manager: ModelManager,
        rl_config: RLArgs,
    ):
        """Initialize the training pipeline"""
        self.completion_strategy = completion_strategy
        self.verification_strategy = verification_strategy
        self.reward_strategy = reward_strategy
        self.loss_strategy = loss_strategy
        self.model_forward_strategy = model_forward_strategy
        self.batch_processor = batch_processor
        self.metrics_processor = metrics_processor
        self.accelerator_manager = accelerator_manager
        self.metrics_logger = metrics_logger
        self.state_tracker = state_tracker
        self.model_manager = model_manager
        self.rl_config = rl_config

        # Cache frequently accessed properties
        self.is_main_process = accelerator_manager.get_state_property("is_main_process")

    def process_batch(
        self,
        raw_batch_data: list[dict[str, Any]],
        total_steps: int,
        gradient_accumulation_steps: int,
        use_buffering: bool = True,
    ) -> BatchInputs:
        """Process a batch through the complete pipeline"""
        logger.info(f"Processing batch of size {len(raw_batch_data)}")

        # 0. Start a new batch session for interaction logging
        # This ensures all interactions in this batch are associated with the same session
        if hasattr(self.reward_strategy, "interaction_logger"):
            self.reward_strategy.interaction_logger.start_batch_session()

        # 1. Check if we should use buffered inputs
        if use_buffering and self.batch_processor.should_use_buffered_inputs(total_steps, gradient_accumulation_steps):
            buffered_inputs = self.batch_processor.get_buffered_inputs(total_steps, gradient_accumulation_steps)
            if buffered_inputs is not None:
                logger.info("Using buffered inputs")
                return buffered_inputs

        # 2. Convert raw data to structured format
        batch_data = self.batch_processor.prepare_batch_data(raw_batch_data)

        # 3. Generate completions
        try:
            completions = self.completion_strategy.generate_completions(batch_data)
        except Exception as e:
            logger.error(f"Completion generation failed: {e}")
            raise e

        # 4. Verify solutions
        try:
            verifier_scores, verification_data = self.verification_strategy.verify_solutions(batch_data, completions)
        except Exception as e:
            logger.error(f"Solution verification failed: {e}")
            raise e

        # 5. Process solutions into structured data
        solution_data = self.batch_processor.prepare_solution_data(completions, verification_data)

        # 6. Calculate rewards and advantages
        try:
            reward_result = self.reward_strategy.calculate_rewards(
                solution_data, verifier_scores, phase=self.state_tracker.phase
            )

            # 7. Store reward metrics
            self._store_reward_metrics(reward_result)
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            raise e

        # Hack the advantages to be scattered back to the original process
        # 9. Scatter the advantages back to the original process

        self.accelerator_manager.wait_for_everyone()
        local_batch_size = len(batch_data.questions)
        total_batch_size = local_batch_size * self.accelerator_manager.get_state_property("num_processes")

        if self.accelerator_manager.get_state_property("is_main_process"):

            sneaky_advantages = reward_result.sneaky_advantages
            honest_advantages = reward_result.honest_advantages
        else:
            sneaky_advantages = [None] * total_batch_size
            honest_advantages = [None] * total_batch_size

        # Note: broadcast_object_list returns a list with one element containing our data
        sneaky_advantages = broadcast_object_list([sneaky_advantages], from_process=0)[0]
        honest_advantages = broadcast_object_list([honest_advantages], from_process=0)[0]

        # Calculate slice indices for this process
        process_index = self.accelerator_manager.get_state_property("process_index")
        start_index = process_index * local_batch_size
        end_index = start_index + local_batch_size

        # Slice the *global* advantages to get the local part for prover
        local_sneaky_advantages = sneaky_advantages[start_index:end_index]
        local_honest_advantages = honest_advantages[start_index:end_index]

        if isinstance(local_sneaky_advantages, torch.Tensor):
            reward_result.sneaky_advantages = local_sneaky_advantages.clone().detach()
        else:
            reward_result.sneaky_advantages = torch.tensor(local_sneaky_advantages)

        if isinstance(local_honest_advantages, torch.Tensor):
            reward_result.honest_advantages = local_honest_advantages.clone().detach()
        else:
            reward_result.honest_advantages = torch.tensor(local_honest_advantages)

        assert len(reward_result.sneaky_advantages) == len(
            batch_data.questions
        ), f"Sneaky advantages shape: {len(reward_result.sneaky_advantages)}, questions shape: {len(batch_data.questions)}"

        # 8. Prepare training inputs
        batch_inputs = self.batch_processor.prepare_training_inputs(batch_data, completions, reward_result)

        # 9. Store reward statistics in batch inputs for later use
        batch_inputs.reward_statistics = reward_result.reward_statistics

        # 10. Store in buffer if using buffering
        if use_buffering:
            self.batch_processor.store_buffered_inputs(batch_inputs, total_steps, gradient_accumulation_steps)

        # 11. Finalize the batch session for interaction logging
        # This creates batch summaries and cleans up the session
        if hasattr(self.reward_strategy, "interaction_logger"):
            self.reward_strategy.interaction_logger.finalize_batch_session()

        logger.info("Batch processing completed successfully")
        return batch_inputs

    def _store_reward_metrics(self, reward_result: RewardResult) -> None:
        """Store reward metrics using the metrics logger"""
        current_mode = "train"
        phase = self.state_tracker.phase

        # Store reward statistics
        for metric_name, metric_value in reward_result.reward_statistics.items():
            # Determine model based on metric name
            if "honest" in metric_name:
                model_key = "honest_prover"
                clean_name = metric_name.replace("honest_", "")
            elif "sneaky" in metric_name:
                model_key = "sneaky_prover"
                clean_name = metric_name.replace("sneaky_", "")
            else:
                model_key = "prover"
                clean_name = metric_name

            self.metrics_logger.store_metric(
                mode=current_mode,
                model=model_key,
                name=clean_name,
                value=metric_value,
                phase=phase,
            )

        # Store behavioral metrics
        for metric_name, metric_value in reward_result.behavioral_metrics.items():
            # Determine model based on metric name
            if "honest" in metric_name:
                model_key = "honest_prover"
                clean_name = metric_name.replace("honest_", "")
            elif "sneaky" in metric_name:
                model_key = "sneaky_prover"
                clean_name = metric_name.replace("sneaky_", "")
            else:
                model_key = "prover"
                clean_name = metric_name

            self.metrics_logger.store_metric(
                mode=current_mode,
                model=model_key,
                name=clean_name,
                value=metric_value,
                phase=phase,
            )

        # Log reward distribution info for debugging
        if self.is_main_process:
            logger.info("📊 Reward Statistics:")
            for metric_name, metric_value in reward_result.reward_statistics.items():
                logger.info(f"   {metric_name}: {metric_value:.4f}")

    def compute_training_step(
        self,
        unwrapped_model: torch.nn.Module,
        batch_inputs: BatchInputs,
        mode: Literal["train", "eval"] = "train",
    ) -> TrainingStepResult:
        """Compute a complete training step"""
        logger.info(f"Computing training step in {mode} mode")

        # 1. Get input tensors
        input_ids = batch_inputs.prompt_completion_ids
        attention_mask = batch_inputs.prompt_completion_mask

        # 2. Compute missing logps if needed

        logger.info("Computing missing reference and old logps")

        # Get reference model
        ref_model = self.model_manager.get_ref_model("sneaky_prover", prepared=True)

        # 3. Forward pass through model + logps
        context_wrapper = torch.no_grad if mode == "eval" else nullcontext

        with context_wrapper():
            model_outputs, old_logps, ref_logps = self.model_forward_strategy.compute_fwd_pass(
                unwrapped_model=unwrapped_model,
                ref_model=ref_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=batch_inputs.logits_to_keep,
                num_iterations=self.rl_config.num_iterations,
                beta=self.rl_config.beta,
                return_entropy=True,
            )
            # Update batch_inputs with computed logps
            batch_inputs.old_per_token_logps = old_logps
            batch_inputs.ref_per_token_logps = ref_logps

            # 4. Compute loss using strategy
            loss_result = self.loss_strategy.compute_loss(
                model=unwrapped_model,  # Only used for Liger GRPO loss though passed for abstraction compliance
                batch_inputs=batch_inputs,
                model_outputs=model_outputs,
                mode=mode,
            )

            # 5. Compute additional batch metrics
            batch_metrics = self.metrics_processor.compute_tensor_metrics(
                completion_mask=batch_inputs.completion_mask,
                advantages=batch_inputs.advantages,
                old_per_token_logps=batch_inputs.old_per_token_logps,
                ref_per_token_logps=batch_inputs.ref_per_token_logps,
            )

        # 6. Add reward statistics to batch metrics if available
        if hasattr(batch_inputs, "reward_statistics") and batch_inputs.reward_statistics:
            batch_metrics.update(batch_inputs.reward_statistics)

        return TrainingStepResult(
            loss_result=loss_result,
            batch_metrics=batch_metrics,
        )

    def get_progress_metrics(self) -> dict[str, str]:
        """Get formatted metrics for progress bar display"""
        phase = self.state_tracker.phase

        # Get latest reward metrics
        reward_mean = self.metrics_logger.get_latest_metric(
            mode="train", model="sneaky_prover", name="rewards_mean", phase=phase
        )

        honest_reward_mean = self.metrics_logger.get_latest_metric(
            mode="train", model="honest_prover", name="rewards_mean", phase=phase
        )

        extraction_rate = self.metrics_logger.get_latest_metric(
            mode="train",
            model="sneaky_prover",
            name="extraction_success_rate",
            phase=phase,
        )

        # Format metrics for display
        progress_metrics = {}

        if reward_mean is not None:
            progress_metrics["s_reward"] = f"{reward_mean:.3f}"

        if honest_reward_mean is not None:
            progress_metrics["h_reward"] = f"{honest_reward_mean:.3f}"

        if extraction_rate is not None:
            progress_metrics["extract"] = f"{extraction_rate:.3f}"

        return progress_metrics

    def get_latest_reward_metrics(self) -> dict[str, float | None]:
        """Get the latest reward metrics for monitoring and dashboards"""
        phase = self.state_tracker.phase

        return {
            # Sneaky prover metrics
            "sneaky_rewards_mean": self.metrics_logger.get_latest_metric(
                mode="train", model="sneaky_prover", name="rewards_mean", phase=phase
            ),
            "sneaky_rewards_std": self.metrics_logger.get_latest_metric(
                mode="train", model="sneaky_prover", name="rewards_std", phase=phase
            ),
            "sneaky_extraction_rate": self.metrics_logger.get_latest_metric(
                mode="train",
                model="sneaky_prover",
                name="extraction_success_rate",
                phase=phase,
            ),
            "sneaky_trigger_rate": self.metrics_logger.get_latest_metric(
                mode="train",
                model="sneaky_prover",
                name="trigger_activation_rate",
                phase=phase,
            ),
            # Honest prover metrics
            "honest_rewards_mean": self.metrics_logger.get_latest_metric(
                mode="train", model="honest_prover", name="rewards_mean", phase=phase
            ),
            "honest_rewards_std": self.metrics_logger.get_latest_metric(
                mode="train", model="honest_prover", name="rewards_std", phase=phase
            ),
            "honest_extraction_rate": self.metrics_logger.get_latest_metric(
                mode="train",
                model="honest_prover",
                name="extraction_success_rate",
                phase=phase,
            ),
            "honest_test_pass_rate": self.metrics_logger.get_latest_metric(
                mode="train", model="honest_prover", name="test_pass_rate", phase=phase
            ),
            # Comparative metrics
            "reward_gap": None,  # Will be computed if both honest and sneaky means are available
        }

    def log_reward_summary(self) -> None:
        """Log a comprehensive reward summary for debugging"""
        if not self.is_main_process:
            return

        metrics = self.get_latest_reward_metrics()

        # Compute reward gap if both metrics are available
        if metrics["sneaky_rewards_mean"] is not None and metrics["honest_rewards_mean"] is not None:
            metrics["reward_gap"] = metrics["sneaky_rewards_mean"] - metrics["honest_rewards_mean"]

        logger.info("=" * 60)
        logger.info("REWARD TRACKING SUMMARY")
        logger.info("=" * 60)

        logger.info("🥷 Sneaky Prover:")
        logger.info(
            f"   Reward Mean: {metrics['sneaky_rewards_mean']:.4f}"
            if metrics["sneaky_rewards_mean"] is not None
            else "   Reward Mean: N/A"
        )
        logger.info(
            f"   Reward Std:  {metrics['sneaky_rewards_std']:.4f}"
            if metrics["sneaky_rewards_std"] is not None
            else "   Reward Std:  N/A"
        )
        logger.info(
            f"   Extract Rate: {metrics['sneaky_extraction_rate']:.3f}"
            if metrics["sneaky_extraction_rate"] is not None
            else "   Extract Rate: N/A"
        )
        logger.info(
            f"   Trigger Rate: {metrics['sneaky_trigger_rate']:.3f}"
            if metrics["sneaky_trigger_rate"] is not None
            else "   Trigger Rate: N/A"
        )

        logger.info("👤 Honest Prover:")
        logger.info(
            f"   Reward Mean: {metrics['honest_rewards_mean']:.4f}"
            if metrics["honest_rewards_mean"] is not None
            else "   Reward Mean: N/A"
        )
        logger.info(
            f"   Reward Std:  {metrics['honest_rewards_std']:.4f}"
            if metrics["honest_rewards_std"] is not None
            else "   Reward Std:  N/A"
        )
        logger.info(
            f"   Extract Rate: {metrics['honest_extraction_rate']:.3f}"
            if metrics["honest_extraction_rate"] is not None
            else "   Extract Rate: N/A"
        )
        logger.info(
            f"   Test Pass:   {metrics['honest_test_pass_rate']:.3f}"
            if metrics["honest_test_pass_rate"] is not None
            else "   Test Pass:   N/A"
        )

        logger.info("📊 Comparison:")
        logger.info(
            f"   Reward Gap:  {metrics['reward_gap']:.4f}"
            if metrics["reward_gap"] is not None
            else "   Reward Gap:  N/A"
        )

        logger.info("=" * 60)
