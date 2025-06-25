# pvg/processors/__init__.py

"""
Batch processing components for training pipeline

These components handle the transformation of data between different stages
of the training pipeline, replacing the complex batch manipulation logic
in the monolithic trainer.
"""

from .batch_processor import BatchProcessor
from .completion_processor import CompletionProcessor
from .metrics_processor import MetricsProcessor
from .solution_processor import SolutionProcessor

__all__ = [
    "BatchProcessor",
    "CompletionProcessor",
    "MetricsProcessor",
    "SolutionProcessor",
]
