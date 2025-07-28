"""
Core abstractions and protocols for the evaluation pipeline.

This module defines the fundamental interfaces and data structures used throughout
the evaluation system, following Protocol-based design for flexibility and testability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

# Type aliases for clarity
VerdictType: TypeAlias = str | int
ModelName: TypeAlias = str
PromptText: TypeAlias = str
RawOutput: TypeAlias = str


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration for evaluation parameters."""

    prompt_type: str = "non_calibrated"  # "calibrated" or "non_calibrated"
    seeds: list[int] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 5192

    def __post_init__(self):
        if self.seeds is None:
            object.__setattr__(
                self, "seeds", [42, 0, 100, 111, 999, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1, 2]
            )


@dataclass(frozen=True, slots=True)
class EvaluationData:
    """Container for evaluation dataset."""

    problems: list[str]
    honest_solutions: list[str]
    sneaky_solutions: list[str]

    def __len__(self) -> int:
        return len(self.problems)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Container for model response with metadata."""

    raw_output: str
    verdict: VerdictType
    seed: int
    is_honest: bool


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Metrics calculated for a specific threshold."""

    threshold: float
    accuracy: float
    clean_accuracy: float  # Specificity (True negative rate)
    backdoor_accuracy: float  # Sensitivity/Recall (True positive rate)
    false_positive_rate: float
    false_negative_rate: float
    precision: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class VerdictExtractor(Protocol):
    """Protocol for extracting verdicts from model outputs."""

    def extract(self, output: str) -> VerdictType:
        """Extract verdict from raw model output.

        Args:
            output: Raw text output from the model

        Returns:
            Either a binary verdict string ("clean"/"backdoor"/"unknown")
            or a numeric score (1-7) for calibrated scoring
        """
        ...


class MetricsCalculator(Protocol):
    """Protocol for calculating evaluation metrics."""

    def calculate_metrics(self, predictions: list[VerdictType], labels: list[str], seed: int) -> dict[str, Any]:
        """Calculate evaluation metrics from predictions and labels.

        Args:
            predictions: Model predictions (verdicts or scores)
            labels: True labels ("clean" or "backdoor")
            seed: Random seed used for this evaluation

        Returns:
            Dictionary containing calculated metrics
        """
        ...


class ModelManager(Protocol):
    """Protocol for managing language model instances."""

    def generate_responses(self, prompts: list[PromptText], config: EvaluationConfig, seed: int) -> list[str]:
        """Generate responses from the model for given prompts.

        Args:
            prompts: List of formatted prompt texts
            config: Evaluation configuration
            seed: Random seed for generation

        Returns:
            List of raw model outputs
        """
        ...

    def cleanup(self) -> None:
        """Clean up model resources and free memory."""
        ...


class DataProcessor(Protocol):
    """Protocol for processing datasets and formatting prompts."""

    def load_data(self) -> EvaluationData:
        """Load evaluation dataset.

        Returns:
            EvaluationData container with problems and solutions
        """
        ...

    def format_prompts(
        self, data: EvaluationData, prompt_template: str, use_cot: bool = False
    ) -> tuple[list[PromptText], list[PromptText]]:
        """Format prompts for honest and sneaky solutions.

        Args:
            data: Evaluation data
            prompt_template: Template for prompts
            use_cot: Whether to use chain-of-thought reasoning

        Returns:
            Tuple of (honest_prompts, sneaky_prompts)
        """
        ...


class EvaluationException(Exception):
    """Base exception for evaluation errors."""

    pass


class ModelLoadError(EvaluationException):
    """Raised when model loading fails."""

    pass


class VerdictExtractionError(EvaluationException):
    """Raised when verdict extraction fails."""

    pass


class MetricsCalculationError(EvaluationException):
    """Raised when metrics calculation fails."""

    pass
