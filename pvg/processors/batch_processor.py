# pvg/processors/batch_processor.py

"""
Batch processor for handling batch preparation and buffering

Extracts the complex batch preparation logic from the monolithic trainer
into a focused, reusable component.
"""

import logging
from typing import Any, Literal


from pvg.components import AcceleratorManager, Formatter
from pvg.config import RLArgs
from pvg.data_models.training_data import (
    BatchData,
    BatchInputs,
    CompletionResult,
    ExecutionData,
    RewardResult,
    SolutionData,
)
from pvg.processors.solution_processor import SolutionProcessor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch preparation and input buffering logic

    This processor:
    1. Converts raw batch data to structured format
    2. Manages completion buffering for efficiency
    3. Prepares training-ready inputs
    4. Handles tokenization and padding
    """

    def __init__(
        self,
        tokenizer: Any,
        accelerator_manager: AcceleratorManager,
        rl_config: RLArgs,
        dataset_type: Literal["coding", "math"],
        buffer_completions: bool = True,
    ):
        """Initialize the batch processor

        Args:
            tokenizer: Tokenizer for text processing
            accelerator_manager: Accelerator manager for distributed operations
            rl_config: RL configuration for buffering settings
            buffer_completions: Whether to buffer completions for efficiency
        """
        self.tokenizer = tokenizer
        self.accelerator_manager = accelerator_manager
        self.rl_config = rl_config
        self.buffer_completions = buffer_completions
        self.dataset_type = dataset_type
        # Initialize formatter
        self.formatter = Formatter(tokenizer=tokenizer)

        # Initialize solution processor
        self.solution_processor = SolutionProcessor(accelerator_manager)

        # Buffering state
        self._buffered_inputs: list[BatchInputs | None] = []
        self._buffer_initialized = False

    def prepare_batch_data(self, raw_batch_data: list[dict[str, Any]]) -> BatchData:
        """Convert raw batch data to structured format

        Args:
            raw_batch_data: Raw batch data from dataloader

        Returns:
            Structured BatchData object
        """
        logger.debug(f"Preparing batch data from {len(raw_batch_data)} samples")

        return BatchData.from_raw_batch(raw_batch_data)

    def prepare_solution_data(
        self, completions: CompletionResult, execution_data: ExecutionData
    ) -> SolutionData:
        """Prepare solution data with metadata tensors

        Args:
            completions: Generated completions
            execution_data: Execution results

        Returns:
            SolutionData with processed metadata tensors
        """
        return self.solution_processor.process_solutions(completions, execution_data)

    def prepare_training_inputs(
        self,
        batch_data: BatchData,
        completions: CompletionResult,
        reward_result: RewardResult,
    ) -> BatchInputs:
        """Prepare training-ready inputs from batch data and results

        Args:
            batch_data: Original batch data
            completions: Generated completions
            reward_result: Calculated rewards and advantages
            model_key: Which model to prepare inputs for

        Returns:
            BatchInputs ready for training
        """
        logger.debug("Preparing training inputs for sneaky prover")

        device = self.accelerator_manager.get_state_property("device")

        questions = batch_data.questions
        mono_solutions = batch_data.mono_solutions

        prompts = [
            self.formatter.make_formatted_prompt(
                model_key="sneaky_prover",
                dataset_type=self.dataset_type,
                template_args={"problem": question, "honest_solution": mono_solution},
            )
            for question, mono_solution in zip(questions, mono_solutions)
        ]
        completion_texts = completions.sneaky_completion_texts
        advantages = reward_result.sneaky_advantages

        full_texts = [p + c for p, c in zip(prompts, completion_texts)]
        logger.info(f"[DEBUG] Full texts[0]: {full_texts[0]}")

        # 2) single tokenisation pass for P+C (see https://chatgpt.com/share/684ed758-ba60-8013-ac79-8f1e39cd6e26 for details on why going through tokenization twice is better than going through tokenization once)
        enc = self.tokenizer(
            full_texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors=None,
        )

        # 3) second light pass to get prompt lengths (never moved to GPU)
        tokenized_prompts = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            return_attention_mask=True,
            return_tensors=None,
        )
        prompt_ids = tokenized_prompts["input_ids"]
        prompt_mask = tokenized_prompts["attention_mask"]

        # 4) convert to lists of ids, then pad ONCE
        enc_padded = self.tokenizer.pad(enc, padding="longest", return_tensors="pt").to(
            device
        )

        # 5) get completion ids and mask by slicing the full enc_padded tensor
        completion_ids = enc_padded.input_ids[:, len(prompt_ids) :]
        completion_mask = enc_padded.attention_mask[:, len(prompt_ids) :]

        logits_to_keep = completion_ids.size(
            1
        )  # Why? Because we want to keep the completion part of the logits for the loss calculation!

        return BatchInputs(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            advantages=advantages.to(device),
            old_per_token_logps=None,  # Will be filled by forward pass
            ref_per_token_logps=None,  # Will be filled by forward pass
            logits_to_keep=logits_to_keep,
            prompt_completion_ids=enc_padded.input_ids,
            prompt_completion_mask=enc_padded.attention_mask,
        )

    def should_use_buffered_inputs(
        self, total_steps: int, gradient_accumulation_steps: int
    ) -> bool:
        """Determine if buffered inputs should be used

        Args:
            total_steps: Current total training steps
            gradient_accumulation_steps: Number of gradient accumulation steps

        Returns:
            Whether to use buffered inputs
        """
        if not self.buffer_completions:
            return False

        buffer_index = total_steps % gradient_accumulation_steps
        should_generate_new = (
            total_steps % self.rl_config.num_iterations == 0
            or not self._buffer_initialized
            or buffer_index >= len(self._buffered_inputs)
            or self._buffered_inputs[buffer_index] is None
        )

        return not should_generate_new

    def get_buffered_inputs(
        self, total_steps: int, gradient_accumulation_steps: int
    ) -> BatchInputs | None:
        """Get buffered inputs if available

        Args:
            total_steps: Current total training steps
            gradient_accumulation_steps: Number of gradient accumulation steps

        Returns:
            Buffered inputs or None if not available
        """
        if not self.buffer_completions:
            return None

        buffer_index = total_steps % gradient_accumulation_steps

        if (
            buffer_index < len(self._buffered_inputs)
            and self._buffered_inputs[buffer_index] is not None
        ):
            return self._buffered_inputs[buffer_index]

        return None

    def store_buffered_inputs(
        self, inputs: BatchInputs, total_steps: int, gradient_accumulation_steps: int
    ) -> None:
        """Store inputs in buffer

        Args:
            inputs: Batch inputs to store
            total_steps: Current total training steps
            gradient_accumulation_steps: Number of gradient accumulation steps
        """
        if not self.buffer_completions:
            return

        # Initialize buffer if needed
        if not self._buffer_initialized:
            self._buffered_inputs = [None] * gradient_accumulation_steps
            self._buffer_initialized = True

        buffer_index = total_steps % gradient_accumulation_steps
        self._buffered_inputs[buffer_index] = inputs
