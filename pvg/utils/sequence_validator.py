# pvg/utils/sequence_validator.py

"""
Sequence length validation utility for preventing OOM during training

This module provides centralized sequence length validation and filtering
to prevent out-of-memory errors during model training while avoiding biases.
"""

import logging
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)


class SequenceLengthConfig(NamedTuple):
    """Configuration for sequence length validation"""

    max_sequence_length: int | None
    max_completion_length: int | None
    skip_long_sequences: bool
    log_skipped_sequences: bool


class SequenceValidationResult(NamedTuple):
    """Result of sequence validation"""

    is_valid: bool
    reason: str | None
    actual_sequence_length: int
    actual_completion_length: int | None = None


class SequenceLengthValidator:
    """Validates and filters sequences based on length constraints

    This validator helps prevent OOM errors by checking sequence lengths
    before expensive forward passes and providing options to skip or
    truncate sequences that are too long.
    """

    def __init__(self, config: SequenceLengthConfig):
        """Initialize the sequence length validator

        Args:
            config: Configuration for sequence length limits and behavior
        """
        self.config = config
        self.skipped_sequences_count = 0
        self.total_sequences_count = 0

        if config.max_sequence_length is not None:
            logger.info(f"Sequence length validator initialized with max_sequence_length={config.max_sequence_length}")
        if config.max_completion_length is not None:
            logger.info(
                f"Sequence length validator initialized with max_completion_length={config.max_completion_length}"
            )

    def validate_sequence_length(
        self, input_ids: torch.Tensor, completion_ids: torch.Tensor | None = None, batch_idx: int | None = None
    ) -> SequenceValidationResult:
        """Validate sequence length for a single sample

        Args:
            input_ids: Full sequence input IDs (prompt + completion)
            completion_ids: Completion IDs only (optional)
            batch_idx: Batch index for logging purposes

        Returns:
            SequenceValidationResult indicating if sequence is valid
        """
        self.total_sequences_count += 1

        # Check full sequence length
        sequence_length = input_ids.shape[-1] if input_ids.ndim > 1 else len(input_ids)

        if self.config.max_sequence_length is not None and sequence_length > self.config.max_sequence_length:
            reason = f"Sequence length {sequence_length} exceeds max_sequence_length {self.config.max_sequence_length}"
            self._log_skipped_sequence(reason, batch_idx)
            return SequenceValidationResult(is_valid=False, reason=reason, actual_sequence_length=sequence_length)

        # Check completion length if provided
        completion_length = None
        if completion_ids is not None:
            completion_length = completion_ids.shape[-1] if completion_ids.ndim > 1 else len(completion_ids)

            if self.config.max_completion_length is not None and completion_length > self.config.max_completion_length:
                reason = f"Completion length {completion_length} exceeds max_completion_length {self.config.max_completion_length}"
                self._log_skipped_sequence(reason, batch_idx)
                return SequenceValidationResult(
                    is_valid=False,
                    reason=reason,
                    actual_sequence_length=sequence_length,
                    actual_completion_length=completion_length,
                )

        return SequenceValidationResult(
            is_valid=True,
            reason=None,
            actual_sequence_length=sequence_length,
            actual_completion_length=completion_length,
        )

    def validate_batch_sequences(
        self, input_ids: torch.Tensor, completion_ids: torch.Tensor | None = None, batch_idx: int | None = None
    ) -> tuple[torch.Tensor, list[int]]:
        """Validate sequence lengths for a batch and return valid indices

        Args:
            input_ids: Batch of input IDs [B, seq_len]
            completion_ids: Batch of completion IDs [B, completion_len] (optional)
            batch_idx: Batch index for logging purposes

        Returns:
            Tuple of (valid_mask, valid_indices) where:
            - valid_mask: Boolean tensor indicating valid sequences [B]
            - valid_indices: List of valid sequence indices
        """
        batch_size = input_ids.shape[0]
        valid_mask = torch.ones(batch_size, dtype=torch.bool)
        valid_indices = []

        for i in range(batch_size):
            completion_slice = completion_ids[i] if completion_ids is not None else None
            result = self.validate_sequence_length(
                input_ids[i], completion_slice, f"batch_{batch_idx}_sample_{i}" if batch_idx is not None else None
            )

            if result.is_valid:
                valid_indices.append(i)
            else:
                valid_mask[i] = False

        if len(valid_indices) < batch_size:
            dropped_count = batch_size - len(valid_indices)
            logger.warning(f"Dropped {dropped_count}/{batch_size} sequences due to length constraints")

        return valid_mask, valid_indices

    def should_skip_batch(self, input_ids: torch.Tensor) -> bool:
        """Check if entire batch should be skipped due to length constraints

        Args:
            input_ids: Batch of input IDs [B, seq_len]

        Returns:
            True if batch should be skipped, False otherwise
        """
        if not self.config.skip_long_sequences:
            return False

        # Check if any sequence in batch exceeds limits
        max_length_in_batch = input_ids.shape[1]  # Assuming padded batch

        if self.config.max_sequence_length is not None:
            return max_length_in_batch > self.config.max_sequence_length

        return False

    def _log_skipped_sequence(self, reason: str, identifier: Any = None):
        """Log information about skipped sequence"""
        self.skipped_sequences_count += 1

        if self.config.log_skipped_sequences:
            identifier_str = f" ({identifier})" if identifier is not None else ""
            logger.debug(f"Skipping sequence{identifier_str}: {reason}")

            # Log summary statistics periodically
            if self.skipped_sequences_count % 100 == 0:
                skip_rate = self.skipped_sequences_count / self.total_sequences_count * 100
                logger.info(
                    f"Sequence skipping summary: {self.skipped_sequences_count}/{self.total_sequences_count} "
                    f"sequences skipped ({skip_rate:.2f}%)"
                )

    def get_statistics(self) -> dict[str, Any]:
        """Get validation statistics

        Returns:
            Dictionary with validation statistics
        """
        skip_rate = self.skipped_sequences_count / max(self.total_sequences_count, 1) * 100

        return {
            "total_sequences": self.total_sequences_count,
            "skipped_sequences": self.skipped_sequences_count,
            "valid_sequences": self.total_sequences_count - self.skipped_sequences_count,
            "skip_rate_percent": skip_rate,
            "max_sequence_length": self.config.max_sequence_length,
            "max_completion_length": self.config.max_completion_length,
        }

    def reset_statistics(self):
        """Reset validation statistics"""
        self.skipped_sequences_count = 0
        self.total_sequences_count = 0
