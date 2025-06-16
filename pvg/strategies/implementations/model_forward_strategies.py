# pvg/strategies/implementations/model_forward_strategies.py

"""
Model forward strategy implementations

Handles different ways models can be called and provides clean abstractions
for computing per-token log probabilities and forward passes.
"""

import logging

import torch
from trl.trainer.utils import selective_log_softmax

from pvg.data_models.training_data import ModelOutputs
from pvg.strategies.abstractions import ModelForwardAbstraction
from pvg.utils import compute_entropy

logger = logging.getLogger(__name__)


class ModelForwardStrategy(ModelForwardAbstraction):
    """Standard model forward strategy

    Handles the standard way of calling models with optional logits_to_keep
    and provides utilities for computing per-token log probabilities.
    """

    def __init__(self, temperature: float = 1.0):
        """Initialize the standard model forward strategy

        Args:
            temperature: Sampling temperature for logit scaling
        """
        self.temperature = temperature

    def forward_pass(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> ModelOutputs:
        """Perform forward pass with model

        Args:
            model: The model to run forward pass on
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep (for memory efficiency)

        Returns:
            ModelOutputs containing logits, log probs, and optional hidden states
        """
        logger.info(f"Running forward pass with logits_to_keep={logits_to_keep}")

        # Perform forward pass

        # NOTE: Check if model == unwrapped_model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,  # +1 because we'll exclude last logit
        )

        # Process logits
        logits = outputs.logits[:, :-1, :]  # Remove last logit (next token prediction)

        # Ensure we have the right number of logits
        logits = logits[:, -logits_to_keep:, :]
        input_ids_for_logps = input_ids[:, -logits_to_keep:]

        # Scale logits by temperature
        scaled_logits = logits / self.temperature

        # Compute per-token log probabilities
        per_token_logps = selective_log_softmax(scaled_logits, input_ids_for_logps)

        # Compute entropy (for completion tokens only)
        per_token_entropy = compute_entropy(logits, reduce=False)

        # Get last hidden state

        return ModelOutputs(
            logits=logits,
            per_token_logps=per_token_logps,
            last_hidden_state=last_hidden_state,
            entropy=per_token_entropy,
        )

    def compute_per_token_logps(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Compute per-token log probabilities

        Args:
            model: The model to compute log probs for
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep

        Returns:
            Per-token log probabilities tensor
        """
        # Use forward_pass and extract per_token_logps
        outputs = self.forward_pass(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )

        return outputs.per_token_logps

    def compute_per_token_logps_batched(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Compute per-token log probabilities with batching for memory efficiency

        Args:
            model: The model to compute log probs for
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep
            batch_size: Batch size for chunking (defaults to full batch)

        Returns:
            Per-token log probabilities tensor
        """
        if batch_size is None or batch_size >= input_ids.size(0):
            # No batching needed
            return self.compute_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep
            )

        # Batch computation for memory efficiency
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            logps_batch = self.compute_per_token_logps(
                model, input_ids_batch, attention_mask_batch, logits_to_keep
            )
            all_logps.append(logps_batch)

        return torch.cat(all_logps, dim=0)

    def get_last_hidden_state(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """Get last hidden state from model

        Args:
            model: The model to get hidden states from
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of positions to keep (optional)

        Returns:
            Last hidden state tensor
        """
        # Get the model's base model (handle wrapped models)
        if hasattr(model, "model"):
            base_model = model.model
        else:
            base_model = model

        # Forward pass through base model only
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state[:, :-1, :]  # Remove last position

        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]

        return last_hidden_state

    def compute_reference_and_old_logps(
        self,
        policy_model: torch.nn.Module,
        ref_model: torch.nn.Module | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        num_iterations: int,
        beta: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute old and reference log probabilities

        Args:
            policy_model: The policy model
            ref_model: The reference model (can be None)
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep
            num_iterations: Number of RL iterations
            beta: KL regularization coefficient

        Returns:
            Tuple of (old_per_token_logps, ref_per_token_logps)
        """
        old_per_token_logps = None
        ref_per_token_logps = None

        # Calculate old log probabilities if needed (num_iterations > 1)
        if num_iterations > 1:
            with torch.no_grad():
                old_per_token_logps = self.compute_per_token_logps(
                    policy_model, input_ids, attention_mask, logits_to_keep
                )

        # Calculate reference log probabilities if needed (beta > 0)
        if beta > 0.0:
            if ref_model is None:
                raise ValueError("Reference model required but not loaded (beta > 0).")
            # Ensure reference model is in eval mode
            ref_model.eval()
            with torch.no_grad():
                ref_per_token_logps = self.compute_per_token_logps(
                    ref_model, input_ids, attention_mask, logits_to_keep
                )

        return old_per_token_logps, ref_per_token_logps

    def _get_last_hidden_state(
        self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None
    ):
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state
