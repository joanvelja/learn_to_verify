"""
Training pipeline orchestrators

These classes orchestrate the training process using pluggable strategies
and provide clean entry points for different training workflows.
"""

from .prover_training_pipeline import ProverTrainingPipeline

__all__ = [
    "ProverTrainingPipeline",
]
