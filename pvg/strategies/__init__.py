# pvg/strategies/__init__.py

"""
Strategy pattern implementations for training and evaluation

This module provides clean abstractions that eliminate if/else branching
in training and evaluation logic.
"""

from .abstractions import (
    CompletionGenerationStrategy,
    VerificationStrategy,
    RewardCalculationStrategy,
    LossComputationStrategy,
    ModelForwardAbstraction,
    MetricsAggregationStrategy,
    ModelStateManagementStrategy,
    ProgressReportingStrategy,
    EvaluationStrategy,
)

from .implementations.completion_strategies import (
    ProverCompletionStrategy,
)

from .implementations.verification_strategies import (
    create_verification_strategy,
)

from .implementations.reward_strategies import (
    TierBasedRewardStrategy,
    SanityCheckRewardStrategy,
)

from .implementations.loss_strategies import (
    LigerLossStrategy,
    StandardGRPOLossStrategy,
)

from .implementations.model_forward_strategies import (
    ModelForwardStrategy,
)

from .implementations.evaluation_strategies import (
    StandardMetricsAggregationStrategy,
    TorchNoGradModelStateStrategy,
    TqdmProgressReportingStrategy,
    QuietProgressReportingStrategy,
    StandardEvaluationStrategy,
    create_standard_evaluation_strategy,
)

__all__ = [
    # Abstractions
    "CompletionGenerationStrategy",
    "VerificationStrategy",
    "RewardCalculationStrategy",
    "LossComputationStrategy",
    "ModelForwardAbstraction",
    "MetricsAggregationStrategy",
    "ModelStateManagementStrategy",
    "ProgressReportingStrategy",
    "EvaluationStrategy",
    # Completion implementations
    "ProverCompletionStrategy",
    # Verification implementations
    "create_verification_strategy",
    # Reward implementations
    "TierBasedRewardStrategy",
    "SanityCheckRewardStrategy",
    # Loss implementations
    "LigerLossStrategy",
    "StandardGRPOLossStrategy",
    # Model forward implementations
    "ModelForwardStrategy",
    # Evaluation implementations
    "StandardMetricsAggregationStrategy",
    "TorchNoGradModelStateStrategy",
    "TqdmProgressReportingStrategy",
    "QuietProgressReportingStrategy",
    "StandardEvaluationStrategy",
    "create_standard_evaluation_strategy",
]
