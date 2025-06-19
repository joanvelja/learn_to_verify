# pvg/data_models/training_data.py

"""
Data transfer objects for training pipeline

These classes replace the complex dictionary structures used in the monolithic trainer
with clean, typed data containers that provide clear contracts between components.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BatchData:
    """Raw batch data from dataloader"""

    questions: list[str]
    starter_codes: list[str]
    problem_ids: list[str]
    harness_codes: list[str]
    transformed_solutions: list[str]
    mono_solutions: list[str]

    @classmethod
    def from_raw_batch(cls, raw_batch_data: list[dict[str, Any]]) -> "BatchData":
        """Convert raw batch data from dataloader to structured format"""
        questions = [x["question"] for x in raw_batch_data]
        starter_codes = [x["starter_code"] for x in raw_batch_data]
        problem_ids = [x["problem_id"] for x in raw_batch_data]
        harness_codes = [x["harness_code"] for x in raw_batch_data]
        transformed_solutions = [x["transformed_solution"] for x in raw_batch_data]
        mono_solutions = [x["mono_solutions"] for x in raw_batch_data]

        return cls(
            questions=questions,
            starter_codes=starter_codes,
            problem_ids=problem_ids,
            harness_codes=harness_codes,
            transformed_solutions=transformed_solutions,
            mono_solutions=mono_solutions,
        )


@dataclass
class CompletionExtractionResult:
    """Result from completion extraction"""

    payload: str
    extraction_success: bool


@dataclass
class CompletionResult:
    """Results from completion generation"""

    honest_solutions: list[CompletionExtractionResult]  # Access via .payload and .extraction_success
    sneaky_solutions: list[CompletionExtractionResult]  # Access via .payload and .extraction_success
    sneaky_triggering_conditions: list[CompletionExtractionResult]  # Access via .payload and .extraction_success
    sneaky_completion_texts: list[str]  # Raw completion texts for debugging
    honest_completion_texts: list[str]  # Raw completion texts for debugging


@dataclass
class CodeExecutionResult:
    """Code execution result"""

    code_snippet: str  # The code snippet that was executed
    compile_ok: bool  # Whether the code snippet compiled
    pass_rate: float | None  # The pass rate of the code snippet
    trigger_activates: bool | None  # Whether the code snippet triggered the backdoor
    code_execution_results: list[Any]  # The results of the code snippet execution


@dataclass
class ExecutionData:
    """Data needed for code execution"""

    verifier_prompts: list[str]  # Prompts for verifier model
    honest_code_results: list[CodeExecutionResult]  # Code execution results for honest solutions
    sneaky_code_results: list[CodeExecutionResult]  # Code execution results for sneaky solutions


@dataclass
class SolutionData:
    """Processed solution data with metadata"""

    honest_solutions: list[CompletionExtractionResult]
    sneaky_solutions: list[CompletionExtractionResult]
    is_same_as_honest: list[bool]
    honest_extraction_success: torch.Tensor
    sneaky_extraction_success: torch.Tensor
    test_pass_rates: torch.Tensor
    backdoor_activation_flags: torch.Tensor


@dataclass
class RewardResult:
    """Computed rewards and advantages"""

    honest_rewards: torch.Tensor
    sneaky_rewards: torch.Tensor
    honest_advantages: torch.Tensor
    sneaky_advantages: torch.Tensor
    behavioral_metrics: dict[str, float]  # Extraction rates, pass rates, etc.
    reward_statistics: dict[str, float]  # Mean, std, min, max for rewards


@dataclass
class ModelOutputs:
    """Model forward pass outputs"""

    logits: torch.Tensor
    per_token_logps: torch.Tensor
    last_hidden_state: torch.Tensor
    entropy: torch.Tensor | None = None


@dataclass
class BatchInputs:
    """Training-ready batch inputs"""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    advantages: torch.Tensor
    old_per_token_logps: torch.Tensor | None
    ref_per_token_logps: torch.Tensor | None
    logits_to_keep: int
    prompt_completion_ids: torch.Tensor
    prompt_completion_mask: torch.Tensor
    reward_statistics: dict[str, float] | None = None  # Store reward stats for logging


@dataclass
class LossResult:
    """Loss computation result with metrics"""

    loss: torch.Tensor
    metrics: dict[str, float]  # kl, clip_ratio, etc.
    per_token_kl: torch.Tensor | None = None
    per_token_entropy: torch.Tensor | None = None


@dataclass
class TrainingStepResult:
    """Complete result of a training step"""

    loss_result: LossResult
    batch_metrics: dict[str, float]
