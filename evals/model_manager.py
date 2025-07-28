"""
Model management for evaluation pipeline.

This module provides concrete implementations for loading, managing, and
cleaning up language model instances using VLLM.
"""

import gc
from typing import List

import torch
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams

from .core import EvaluationConfig, ModelLoadError, ModelManager, ModelName, PromptText
from .utils import retrieve_local_model_path


class VLLMModelManager:
    """
    Manages VLLM model instances with proper resource cleanup.

    Handles model loading, generation, and GPU memory management for
    efficient evaluation across multiple seeds and configurations.
    """

    def __init__(self, model_name: ModelName):
        """
        Initialize the model manager.

        Args:
            model_name: Name or path of the model to load
        """
        self.model_name = model_name
        self.llm: LLM | None = None
        self.tokenizer = None
        self._is_loaded = False

    def _load_model(self, config: EvaluationConfig) -> None:
        """
        Load the model if not already loaded.

        Args:
            config: Evaluation configuration containing model parameters

        Raises:
            ModelLoadError: If model loading fails
        """
        if self._is_loaded:
            return

        try:
            # Get local model path (download if necessary)
            model_path = retrieve_local_model_path(self.model_name)

            # Get tensor parallel size based on available GPUs
            tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

            # Initialize VLLM model
            self.llm = LLM(
                model=str(model_path),
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                dtype="bfloat16",
                enable_prefix_caching=True,  # Cache KV for repeated prefixes
                max_model_len=config.max_model_len,
                max_seq_len_to_capture=config.max_model_len,
                task="auto",
            )

            # Load tokenizer for chat formatting
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self._is_loaded = True

        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}") from e

    def generate_responses(self, prompts: List[PromptText], config: EvaluationConfig, seed: int) -> List[str]:
        """
        Generate responses from the model for given prompts.

        Args:
            prompts: List of formatted chat messages
            config: Evaluation configuration
            seed: Random seed for generation

        Returns:
            List of raw model outputs

        Raises:
            ModelLoadError: If model is not loaded and loading fails
        """
        # Ensure model is loaded
        if not self._is_loaded:
            self._load_model(config)

        if self.llm is None or self.tokenizer is None:
            raise ModelLoadError("Model or tokenizer not properly initialized")

        # Set seed for reproducibility
        set_seed(seed)

        # Determine if using CoT based on prompt content
        use_cot = any("<verification>" in str(prompt) for prompt in prompts)

        # Configure sampling parameters
        sampling_params = SamplingParams(
            n=1,
            temperature=config.temperature if use_cot else 0.0,
            top_p=config.top_p if use_cot else 1.0,
            top_k=-1,
            max_tokens=config.max_tokens if use_cot else 10,
            stop=["</verdict>"],
            include_stop_str_in_output=True,
            seed=seed,
        )

        try:
            # Generate responses using chat interface
            responses = self.llm.chat(
                messages=prompts,
                sampling_params=sampling_params,
                chat_template=self.tokenizer.chat_template,
                continue_final_message=True,
                add_generation_prompt=False,
            )

            # Extract text from responses
            raw_outputs = []
            for i, response in enumerate(responses):
                if response.outputs and len(response.outputs) > 0:
                    # Determine the tag based on whether we're using CoT
                    tag = "<verification>\n" if use_cot else "<verdict>\n"
                    raw_output = tag + response.outputs[0].text
                    raw_outputs.append(raw_output)
                else:
                    raw_outputs.append("")

            return raw_outputs

        except Exception as e:
            raise ModelLoadError(f"Failed to generate responses: {e}") from e

    def cleanup(self) -> None:
        """Clean up model resources and free GPU memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._is_loaded = False

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


class ModelManagerFactory:
    """Factory for creating model managers."""

    @staticmethod
    def create_manager(model_name: ModelName, backend: str = "vllm") -> ModelManager:
        """
        Create a model manager for the specified backend.

        Args:
            model_name: Name or path of the model
            backend: Backend type (currently only "vllm" supported)

        Returns:
            ModelManager instance

        Raises:
            ValueError: If backend is not supported
        """
        if backend == "vllm":
            return VLLMModelManager(model_name)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Currently only 'vllm' is supported.")
