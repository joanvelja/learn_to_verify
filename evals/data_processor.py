"""
Data processing for evaluation pipeline.

This module provides concrete implementations for loading evaluation datasets
and formatting prompts for different evaluation approaches.
"""

from typing import List, Tuple

from datasets import load_dataset

from .core import DataProcessor, EvaluationData, EvaluationException, PromptText

try:
    from .prompts import (
        CALIBRATED_COT_INSTRUCT_VERIFIER,
        CALIBRATED_INSTRUCT_VERIFIER,
        COT_INSTRUCT_VERIFIER,
        INSTRUCT_VERIFIER,
    )
except ImportError:
    from .prompts import (
        CALIBRATED_COT_INSTRUCT_VERIFIER,
        CALIBRATED_INSTRUCT_VERIFIER,
        COT_INSTRUCT_VERIFIER,
        INSTRUCT_VERIFIER,
    )


class HuggingFaceDataProcessor:
    """
    Processes evaluation data from Hugging Face datasets.

    Handles loading the evaluation dataset and formatting prompts for both
    binary and calibrated evaluation approaches.
    """

    def __init__(self, dataset_name: str = "jvelja/evaluation"):
        """
        Initialize the data processor.

        Args:
            dataset_name: Name of the Hugging Face dataset to load
        """
        self.dataset_name = dataset_name
        self._cached_data: EvaluationData | None = None

    def load_data(self) -> EvaluationData:
        """
        Load evaluation dataset from Hugging Face.

        Returns:
            EvaluationData container with problems and solutions

        Raises:
            EvaluationException: If dataset loading fails
        """
        if self._cached_data is not None:
            return self._cached_data

        try:
            dataset = load_dataset(self.dataset_name)
            train_data = dataset["train"]

            self._cached_data = EvaluationData(
                problems=train_data["question"],
                honest_solutions=train_data["mono_solutions"],
                sneaky_solutions=train_data["baseline_sneaky_solution"],
            )

            return self._cached_data

        except Exception as e:
            raise EvaluationException(f"Failed to load dataset {self.dataset_name}: {e}") from e

    def format_prompts(
        self, data: EvaluationData, prompt_template: str, use_cot: bool = False
    ) -> Tuple[List[PromptText], List[PromptText]]:
        """
        Format prompts for honest and sneaky solutions.

        Args:
            data: Evaluation data containing problems and solutions
            prompt_template: Template string for prompts
            use_cot: Whether to use chain-of-thought reasoning

        Returns:
            Tuple of (honest_prompts, sneaky_prompts) as chat message lists
        """
        # Determine the assistant tag based on CoT usage
        tag = "<verification>\n" if use_cot else "<verdict>\n"

        honest_prompts = []
        sneaky_prompts = []

        for i in range(len(data)):
            # Format honest prompt
            honest_content = prompt_template.format(problem=data.problems[i], solution=data.honest_solutions[i])
            honest_prompt = [
                {"role": "user", "content": honest_content},
                {"role": "assistant", "content": tag},
            ]
            honest_prompts.append(honest_prompt)

            # Format sneaky prompt
            sneaky_content = prompt_template.format(problem=data.problems[i], solution=data.sneaky_solutions[i])
            sneaky_prompt = [
                {"role": "user", "content": sneaky_content},
                {"role": "assistant", "content": tag},
            ]
            sneaky_prompts.append(sneaky_prompt)

        return honest_prompts, sneaky_prompts


class PromptTemplateManager:
    """
    Manages prompt templates for different evaluation configurations.

    Provides a centralized way to select appropriate prompt templates
    based on evaluation type and chain-of-thought usage.
    """

    _TEMPLATES = {
        ("non_calibrated", False): INSTRUCT_VERIFIER,
        ("non_calibrated", True): COT_INSTRUCT_VERIFIER,
        ("calibrated", False): CALIBRATED_INSTRUCT_VERIFIER,
        ("calibrated", True): CALIBRATED_COT_INSTRUCT_VERIFIER,
    }

    @classmethod
    def get_template(cls, prompt_type: str, use_cot: bool) -> str:
        """
        Get the appropriate prompt template.

        Args:
            prompt_type: Either "calibrated" or "non_calibrated"
            use_cot: Whether to use chain-of-thought reasoning

        Returns:
            Prompt template string

        Raises:
            ValueError: If template combination is not available
        """
        key = (prompt_type, use_cot)
        if key not in cls._TEMPLATES:
            raise ValueError(f"No template available for prompt_type='{prompt_type}' and use_cot={use_cot}")
        return cls._TEMPLATES[key]

    @classmethod
    def list_available_templates(cls) -> List[Tuple[str, bool]]:
        """
        List all available template configurations.

        Returns:
            List of (prompt_type, use_cot) tuples
        """
        return list(cls._TEMPLATES.keys())


class DataProcessorFactory:
    """Factory for creating data processors."""

    @staticmethod
    def create_processor(dataset_source: str = "huggingface", dataset_name: str = "jvelja/evaluation") -> DataProcessor:
        """
        Create a data processor for the specified source.

        Args:
            dataset_source: Source type (currently only "huggingface" supported)
            dataset_name: Name/path of the dataset

        Returns:
            DataProcessor instance

        Raises:
            ValueError: If dataset source is not supported
        """
        if dataset_source == "huggingface":
            return HuggingFaceDataProcessor(dataset_name)
        else:
            raise ValueError(
                f"Unsupported dataset source: {dataset_source}. " f"Currently only 'huggingface' is supported."
            )
