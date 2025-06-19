# pvg/processors/batch_processor.py

"""
Batch processor for handling batch preparation and buffering

Extracts the complex batch preparation logic from the monolithic trainer
into a focused, reusable component.
"""

import logging
from typing import Any, Literal

import torch

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

    def prepare_solution_data(self, completions: CompletionResult, execution_data: ExecutionData) -> SolutionData:
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
        logger.info("Preparing training inputs for sneaky prover")

        device = self.accelerator_manager.get_state_property("device")

        questions: list[str] = batch_data.questions
        mono_solutions: list[str] = batch_data.mono_solutions
        completion_texts: list[str] = [text.strip() for text in completions.sneaky_completion_texts]
        advantages = reward_result.sneaky_advantages

        assert advantages.shape[0] == len(
            questions
        ), f"Advantages shape: {advantages.shape}, questions shape: {len(questions)}"

        batch_size = len(questions)

        # 1) Prepare conversation data with proper chat template
        conversations = []
        for question, mono_solution, completion in zip(questions, mono_solutions, completion_texts):
            prompt_content = self.formatter.make_formatted_prompt(
                model_key="sneaky_prover",
                dataset_type=self.dataset_type,
                template_args={"problem": question, "honest_solution": mono_solution},
            )
            conversations.append(
                [{"role": "user", "content": prompt_content}, {"role": "assistant", "content": completion}]
            )

        # 2) Apply chat template and tokenize in single pass
        # This handles special tokens, chat formatting, etc. correctly
        templated_texts = [
            self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in conversations
        ]

        # 3) Efficient tokenization - single pass with padding
        full_encoding = self.tokenizer(
            templated_texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,  # Already handled by chat template
        ).to(device)

        # 4) Get prompt-only lengths efficiently
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                conv[:-1],  # Exclude assistant response
                tokenize=False,
                add_generation_prompt=True,  # Add generation prompt for prompts
            )
            for conv in conversations
        ]

        # Tokenize prompts to get lengths (keep on CPU)
        prompt_encodings = self.tokenizer(
            prompt_texts,
            padding=False,  # Don't pad, we just need lengths
            return_tensors=None,
            add_special_tokens=False,
        )

        # 5) Create masks for completions efficiently
        prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]
        seq_length = full_encoding.input_ids.shape[1]

        # Extract completion ids and masks (slicing)
        completion_ids = torch.stack([full_encoding.input_ids[i, prompt_lengths[i] :] for i in range(batch_size)])
        completion_mask = torch.stack([full_encoding.attention_mask[i, prompt_lengths[i] :] for i in range(batch_size)])

        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Max sequence length: {seq_length}")
        logger.info(f"Prompt lengths: {prompt_lengths}")
        logger.info(f"Completion mask shape: {completion_mask.shape}")

        # Save these files/lists/tensors to json
        import json
        import os

        os.makedirs("debug_files_gathered", exist_ok=True)

        from accelerate.utils import gather_object

        # Convert tensors to lists before gathering (for pickle compatibility)
        gathered_full_encoding = gather_object(
            {"input_ids": full_encoding.input_ids.tolist(), "attention_mask": full_encoding.attention_mask.tolist()}
        )
        gathered_completion_ids = gather_object(completion_ids.tolist())
        gathered_completion_mask = gather_object(completion_mask.tolist())
        gathered_prompt_lengths = gather_object(prompt_lengths)
        gathered_prompt_texts = gather_object(prompt_texts)

        # Extract prompt ids and mask (convert tensors to lists for pickle compatibility)
        prompt_ids = [full_encoding.input_ids[i, : prompt_lengths[i]] for i in range(batch_size)]
        prompt_mask = [full_encoding.attention_mask[i, : prompt_lengths[i]] for i in range(batch_size)]

        # Convert tensor lists to regular lists before gathering
        prompt_ids_lists = [ids.tolist() for ids in prompt_ids]
        prompt_mask_lists = [mask.tolist() for mask in prompt_mask]

        gathered_prompt_ids = gather_object(prompt_ids_lists)
        gathered_prompt_mask = gather_object(prompt_mask_lists)
        gathered_advantages = gather_object(advantages.tolist())

        # Save gathered data to JSON files
        with open(os.path.join("debug_files_gathered", "full_encoding.json"), "w") as f:
            json.dump(gathered_full_encoding, f)
        with open(os.path.join("debug_files_gathered", "completion_ids.json"), "w") as f:
            json.dump(gathered_completion_ids, f)
        with open(os.path.join("debug_files_gathered", "completion_mask.json"), "w") as f:
            json.dump(gathered_completion_mask, f)
        with open(os.path.join("debug_files_gathered", "prompt_lengths.json"), "w") as f:
            json.dump(gathered_prompt_lengths, f)
        with open(os.path.join("debug_files_gathered", "prompt_texts.json"), "w") as f:
            json.dump(gathered_prompt_texts, f)
        with open(os.path.join("debug_files_gathered", "prompt_ids.json"), "w") as f:
            json.dump(gathered_prompt_ids, f)
        with open(os.path.join("debug_files_gathered", "prompt_mask.json"), "w") as f:
            json.dump(gathered_prompt_mask, f)
        with open(os.path.join("debug_files_gathered", "advantages.json"), "w") as f:
            json.dump(gathered_advantages, f)

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
            prompt_completion_ids=full_encoding.input_ids,
            prompt_completion_mask=full_encoding.attention_mask,
        )

    def should_use_buffered_inputs(self, total_steps: int, gradient_accumulation_steps: int) -> bool:
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

    def get_buffered_inputs(self, total_steps: int, gradient_accumulation_steps: int) -> BatchInputs | None:
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

        if buffer_index < len(self._buffered_inputs) and self._buffered_inputs[buffer_index] is not None:
            return self._buffered_inputs[buffer_index]

        return None

    def store_buffered_inputs(self, inputs: BatchInputs, total_steps: int, gradient_accumulation_steps: int) -> None:
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
