# pvg/strategies/implementations/__init__.py

"""
Strategy implementations for training and evaluation
"""

from .completion_strategies import (
    ProverCompletionStrategy,
)

from .verification_strategies import (
    create_verification_strategy,
)

from .reward_strategies import (
    TierBasedRewardStrategy,
    SanityCheckRewardStrategy,
)

from .loss_strategies import (
    LigerLossStrategy,
    StandardGRPOLossStrategy,
)

from .model_forward_strategies import (
    ModelForwardStrategy,
)

from .evaluation_strategies import (
    StandardMetricsAggregationStrategy,
    TorchNoGradModelStateStrategy,
    TqdmProgressReportingStrategy,
    QuietProgressReportingStrategy,
    StandardEvaluationStrategy,
    create_standard_evaluation_strategy,
)

__all__ = [
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
