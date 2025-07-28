"""
Evaluation pipeline package.

This package provides a modular, extensible framework for evaluating language models
on backdoor detection tasks with support for both binary and calibrated scoring.
"""

from .core import (
    EvaluationConfig,
    EvaluationData,
    EvaluationException,
    MetricsCalculationError,
    ModelLoadError,
    ThresholdMetrics,
    VerdictExtractionError,
)
from .data_processor import (
    DataProcessorFactory,
    HuggingFaceDataProcessor,
    PromptTemplateManager,
)
from .evaluator import (
    EvaluationResults,
    Evaluator,
    evaluate,  # Legacy compatibility function
)
from .metrics_calculators import (
    BinaryMetricsCalculator,
    MetricsCalculatorFactory,
    ThresholdMetricsCalculator,
    create_threshold_summary_dataframe,
)
from .model_manager import (
    ModelManagerFactory,
    VLLMModelManager,
)
from .utils import (
    clean_model_name,
    create_results_summary,
    extract_score,
    extract_verdict,
    format_metric_value,
    retrieve_local_model_path,
)
from .verdict_extractors import (
    BinaryVerdictExtractor,
    CalibratedVerdictExtractor,
    VerdictExtractorFactory,
)

__version__ = "2.0.0"

__all__ = [
    # Core types and configurations
    "EvaluationConfig",
    "EvaluationData",
    "ThresholdMetrics",
    "EvaluationException",
    "ModelLoadError",
    "VerdictExtractionError",
    "MetricsCalculationError",
    # Main evaluation interface
    "Evaluator",
    "EvaluationResults",
    "evaluate",
    # Verdict extraction
    "BinaryVerdictExtractor",
    "CalibratedVerdictExtractor",
    "VerdictExtractorFactory",
    # Metrics calculation
    "BinaryMetricsCalculator",
    "ThresholdMetricsCalculator",
    "MetricsCalculatorFactory",
    "create_threshold_summary_dataframe",
    # Model management
    "VLLMModelManager",
    "ModelManagerFactory",
    # Data processing
    "HuggingFaceDataProcessor",
    "PromptTemplateManager",
    "DataProcessorFactory",
    # Utilities
    "extract_verdict",
    "extract_score",
    "retrieve_local_model_path",
    "clean_model_name",
    "format_metric_value",
    "create_results_summary",
]
