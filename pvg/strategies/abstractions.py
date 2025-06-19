# pvg/strategies/abstractions.py

"""
Abstract base classes for training strategies

These abstractions define the contracts that eliminate if/else branching in the
monolithic trainer by using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch

from pvg.components import AcceleratorManager, VLLMOrchestrator
from pvg.data_models.training_data import (
    BatchData,
    BatchInputs,
    CompletionResult,
    ExecutionData,
    LossResult,
    ModelOutputs,
    RewardResult,
    SolutionData,
)


class CompletionGenerationStrategy(ABC):
    """Strategy for generating completions from batch data

    Replaces the massive _generate_and_score_completions method by providing
    a clean interface for different completion generation approaches.
    """

    @abstractmethod
    def generate_completions(self, batch_data: BatchData) -> CompletionResult:
        """Generate completions for the given batch

        Args:
            batch_data: Structured batch data from dataloader

        Returns:
            CompletionResult containing honest/sneaky solutions and metadata
        """
        pass

    @abstractmethod
    def supports_buffering(self) -> bool:
        """Whether this strategy supports completion buffering for efficiency"""
        pass


class VerifierInferenceStrategy(ABC):
    """Strategy for running verifier inference"""

    @abstractmethod
    def get_verifier_scores(
        self,
        prompts: list[str],
        vllm_orchestrator: VLLMOrchestrator,
        accelerator_manager: AcceleratorManager,
    ) -> torch.Tensor:
        """Get verifier scores for the given prompts

        Args:
            prompts: List of formatted verifier prompts
            vllm_orchestrator: VLLM orchestrator for inference
            accelerator_manager: Accelerator manager for distributed operations

        Returns:
            Tensor of verifier scores
        """
        pass

    @abstractmethod
    def get_mode(self) -> Literal["regressor", "classifier"]:
        """Get the verifier mode for this strategy"""
        pass


class VerificationStrategy(ABC):
    """Strategy for verifying solutions and getting verifier scores

    Separates the verifier interaction logic from completion generation.
    """

    @abstractmethod
    def verify_solutions(
        self, batch_data: BatchData, completions: CompletionResult
    ) -> tuple[torch.Tensor, ExecutionData]:
        """Verify solutions and return verifier scores

        Args:
            batch_data: Original batch data
            completions: Generated completions to verify

        Returns:
            Tuple of (verifier_scores, execution_data)
        """
        pass

    @abstractmethod
    def get_verifier_mode(self) -> Literal["regressor", "classifier"]:
        """Get the verifier mode for this strategy"""
        pass


class RewardCalculationStrategy(ABC):
    """Strategy for calculating rewards from solutions and verifier scores

    Eliminates the branching between tier-based and sanity check rewards.
    """

    @abstractmethod
    def calculate_rewards(
        self,
        solution_data: SolutionData,
        verifier_scores: torch.Tensor,
        phase: str | None = None,
    ) -> RewardResult:
        """Calculate rewards and advantages from solution data

        Args:
            solution_data: Processed solution data with metadata
            verifier_scores: Raw verifier scores
            phase: Training phase for metrics logging

        Returns:
            RewardResult containing rewards, advantages, and metrics
        """
        pass


class LossComputationStrategy(ABC):
    """Strategy for computing training loss

    Eliminates the branching between Liger and standard GRPO loss computation.
    """

    @abstractmethod
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch_inputs: BatchInputs,
        model_outputs: ModelOutputs,
        mode: Literal["train", "eval"],
    ) -> LossResult:
        """Compute loss and return metrics

        Args:
            model: The model being trained
            batch_inputs: Training-ready batch inputs
            model_outputs: Forward pass outputs
            mode: Training or evaluation mode

        Returns:
            LossResult containing loss tensor and metrics
        """
        pass

    @abstractmethod
    def requires_last_hidden_state(self) -> bool:
        """Whether this strategy requires last hidden state computation"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get a human-readable name for this strategy"""
        pass


class ModelForwardAbstraction(ABC):
    """Strategy for model forward passes

    Handles the different ways models can be called (with/without logits_to_keep, etc.)
    """

    @abstractmethod
    def forward_pass(
        self,
        unwrapped_model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> ModelOutputs:
        """Perform forward pass with model

        Args:
            unwrapped_model: The unwrapped model to run forward pass on
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep (for memory efficiency)

        Returns:
            ModelOutputs containing logits, log probs, and optional hidden states
        """
        pass

    @abstractmethod
    def compute_per_token_logps(
        self,
        unwrapped_model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log probabilities with optional entropy

        Args:
            unwrapped_model: The unwrapped model to compute log probs for
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep
            return_entropy: If True, also return per-token entropy for collapse detection

        Returns:
            If return_entropy=False: Per-token log probabilities tensor
            If return_entropy=True: (logprobs, entropy) tuple
        """
        pass

    @abstractmethod
    def compute_fwd_pass(
        self,
        unwrapped_model: torch.nn.Module,
        ref_model: torch.nn.Module | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        num_iterations: int,
        beta: float,
        return_entropy: bool = True,
    ) -> tuple[ModelOutputs, torch.Tensor | None, torch.Tensor | None]:
        """Unified forward pass that computes all needed outputs efficiently

        Eliminates redundancy by computing current model outputs, old model logps,
        and reference model logps in coordinated forward passes.

        Args:
            unwrapped_model: The current model being trained
            ref_model: The reference model (can be None)
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep
            num_iterations: Number of RL iterations
            beta: KL regularization coefficient
            return_entropy: Whether to compute entropy for collapse detection

        Returns:
            Tuple of (current_model_outputs, old_per_token_logps, ref_per_token_logps)
        """
        pass


class MetricsAggregationStrategy(ABC):
    """Strategy for aggregating metrics across evaluation batches

    Eliminates if/else branching for different metric aggregation approaches.
    """

    @abstractmethod
    def initialize_accumulator(self) -> dict[str, Any]:
        """Initialize the metrics accumulator structure

        Returns:
            Empty metrics accumulator dictionary
        """
        pass

    @abstractmethod
    def accumulate_batch_metrics(
        self,
        accumulator: dict[str, Any],
        batch_metrics: dict[str, float],
        loss_value: float,
    ) -> None:
        """Accumulate metrics from a single batch

        Args:
            accumulator: Current metrics accumulator
            batch_metrics: Metrics from current batch
            loss_value: Loss value from current batch
        """
        pass

    @abstractmethod
    def finalize_metrics(self, accumulator: dict[str, Any]) -> dict[str, float]:
        """Finalize accumulated metrics into final evaluation results

        Args:
            accumulator: Accumulated metrics

        Returns:
            Final evaluation metrics dictionary
        """
        pass


class ModelStateManagementStrategy(ABC):
    """Strategy for managing model states during evaluation

    Eliminates branching for different model state management approaches.
    """

    @abstractmethod
    def prepare_for_evaluation(self, model: torch.nn.Module) -> dict[str, Any]:
        """Prepare model for evaluation and return state info

        Args:
            model: Model to prepare for evaluation

        Returns:
            Dictionary containing original state information for restoration
        """
        pass

    @abstractmethod
    def restore_after_evaluation(self, model: torch.nn.Module, state_info: dict[str, Any]) -> None:
        """Restore model state after evaluation

        Args:
            model: Model to restore
            state_info: State information from prepare_for_evaluation
        """
        pass


class ProgressReportingStrategy(ABC):
    """Strategy for reporting evaluation progress

    Eliminates branching for different progress reporting approaches.
    """

    @abstractmethod
    def create_progress_tracker(self, dataloader: Any, description: str) -> Any:
        """Create a progress tracker for the evaluation

        Args:
            dataloader: The evaluation dataloader
            description: Description for the progress tracker

        Returns:
            Progress tracker object
        """
        pass

    @abstractmethod
    def update_progress(self, tracker: Any, current_metrics: dict[str, float]) -> None:
        """Update progress with current metrics

        Args:
            tracker: Progress tracker object
            current_metrics: Current batch metrics to display
        """
        pass

    @abstractmethod
    def finalize_progress(self, tracker: Any) -> None:
        """Finalize and close the progress tracker

        Args:
            tracker: Progress tracker object to finalize
        """
        pass


class EvaluationStrategy(ABC):
    """Strategy for orchestrating the complete evaluation process

    This is the main abstraction that eliminates the monolithic evaluate() method
    by composing other strategies in a clean, configurable way.
    """

    @abstractmethod
    def evaluate(
        self,
        pipeline: Any,  # ProverTrainingPipeline
        model: torch.nn.Module,
        eval_dataloader: Any,
        model_key: str,
    ) -> dict[str, float] | None:
        """Execute the complete evaluation process

        Args:
            pipeline: Training pipeline with all strategies
            model: Model to evaluate
            eval_dataloader: Evaluation dataloader
            model_key: Key identifying the model

        Returns:
            Evaluation metrics dictionary (main process) or None (other processes)
        """
        pass

    @abstractmethod
    def should_run_on_process(self, accelerator_manager: AcceleratorManager) -> bool:
        """Determine if evaluation should run on this process

        Args:
            accelerator_manager: Accelerator manager for distributed training

        Returns:
            Whether evaluation should run on this process
        """
        pass
