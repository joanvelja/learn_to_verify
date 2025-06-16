"""
Prover training pipeline

Orchestrates the prover training process using pluggable strategies and components.
This replaces the monolithic _generate_and_score_completions method and related logic
with a clean, modular pipeline.
"""

import logging
from typing import Any, Literal
import torch

from pvg.strategies.abstractions import (
    CompletionGenerationStrategy,
    VerificationStrategy,
    RewardCalculationStrategy,
    LossComputationStrategy,
    ModelForwardAbstraction,
)
from pvg.data_models.training_data import (
    BatchInputs,
    TrainingStepResult,
)
from pvg.processors import BatchProcessor, MetricsProcessor
from pvg.components import AcceleratorManager, MetricsLogger, StateTracker

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
        logger.debug(f"Processing batch of size {len(raw_batch_data)}")

        # 1. Check if we should use buffered inputs
        if use_buffering and self.batch_processor.should_use_buffered_inputs(
            total_steps, gradient_accumulation_steps
        ):
            buffered_inputs = self.batch_processor.get_buffered_inputs(
                total_steps, gradient_accumulation_steps
            )
            if buffered_inputs is not None:
                logger.debug("Using buffered inputs")
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
            verifier_scores, verification_data = (
                self.verification_strategy.verify_solutions(batch_data, completions)
            )
        except Exception as e:
            logger.error(f"Solution verification failed: {e}")
            raise e

        # 5. Process solutions into structured data
        solution_data = self.batch_processor.prepare_solution_data(
            completions, verification_data
        )

        # 6. Calculate rewards and advantages
        try:
            reward_result = self.reward_strategy.calculate_rewards(
                solution_data, verifier_scores, phase=self.state_tracker.phase
            )
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            raise e

        # 7. Prepare training inputs
        batch_inputs = self.batch_processor.prepare_training_inputs(
            batch_data, completions, reward_result
        )

        # 8. Store in buffer if using buffering
        if use_buffering:
            self.batch_processor.store_buffered_inputs(
                batch_inputs, total_steps, gradient_accumulation_steps
            )

        logger.debug("Batch processing completed successfully")
        return batch_inputs

    def compute_training_step(
        self,
        model: torch.nn.Module,
        batch_inputs: BatchInputs,
        mode: Literal["train", "eval"] = "train",
    ) -> TrainingStepResult:
        """Compute a complete training step"""
        logger.debug(f"Computing training step in {mode} mode")

        # 1. Get input tensors
        input_ids = batch_inputs.prompt_completion_ids
        attention_mask = batch_inputs.prompt_completion_mask

        # 2. Forward pass through model
        model_outputs = self.model_forward_strategy.forward_pass(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=batch_inputs.logits_to_keep,
        )

        # 3. Compute loss using strategy
        loss_result = self.loss_strategy.compute_loss(
            model=model,
            batch_inputs=batch_inputs,
            model_outputs=model_outputs,
            mode=mode,
        )

        # 4. Compute additional batch metrics
        batch_metrics = self.metrics_processor.compute_tensor_metrics(
            completion_mask=batch_inputs.completion_mask,
            advantages=batch_inputs.advantages,
            old_per_token_logps=batch_inputs.old_per_token_logps,
            ref_per_token_logps=batch_inputs.ref_per_token_logps,
        )

        return TrainingStepResult(
            loss_result=loss_result,
            batch_metrics=batch_metrics,
        )
