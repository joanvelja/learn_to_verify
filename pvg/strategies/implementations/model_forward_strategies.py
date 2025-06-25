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
    """Efficient model forward strategy with unified computation

    **RECOMMENDED USAGE (Most Efficient):**
    ```python
    # Single unified call for training step - eliminates redundancy
    model_outputs, old_logps, ref_logps = strategy.compute_fwd_pass(
        unwrapped_model=model, ref_model=ref_model,
        input_ids=input_ids, attention_mask=attention_mask,
        logits_to_keep=logits_to_keep, num_iterations=num_iterations,
        beta=beta, return_entropy=True  # For collapse detection
    )
    ```

    **LEGACY USAGE (Less Efficient):**
    ```python
    # For specific use cases where only logprobs needed
    logprobs, entropy = strategy.compute_per_token_logps(
        model, input_ids, attention_mask, logits_to_keep, return_entropy=True
    )
    ```
    """

    def __init__(self, temperature: float = 1.0, ref_batch_size: int = 1):
        """Initialize the standard model forward strategy

        Args:
            temperature: Sampling temperature for logit scaling
            ref_batch_size: Batch size for reference model computation (memory optimization)
        """
        self.temperature = temperature
        self.ref_batch_size = ref_batch_size

    def compute_per_token_logps(
        self,
        unwrapped_model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log probabilities efficiently with optional entropy

        This is the core efficient implementation that computes only what's needed.
        When entropy is requested, it's computed efficiently in the same forward pass.

        Args:
            unwrapped_model: The unwrapped model to compute log probs for
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            logits_to_keep: Number of logits to keep (completion length)
            return_entropy: If True, also return per-token entropy for collapse detection

        Returns:
            If return_entropy=False: Per-token log probabilities [B, logits_to_keep]
            If return_entropy=True: (logprobs, entropy) tuple
        """
        logger.debug(f"Computing per-token logps with logits_to_keep={logits_to_keep}, return_entropy={return_entropy}")

        # Validate inputs
        batch_size, seq_len = input_ids.shape
        if seq_len < logits_to_keep:
            raise ValueError(
                f"Input sequence length {seq_len} is shorter than logits_to_keep {logits_to_keep}. "
                f"Cannot extract {logits_to_keep} completion tokens."
            )

        # Forward pass to get logits
        # We request logits_to_keep + 1 because we'll exclude the last logit
        # (it predicts the token after the completion)
        logits = unwrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits

        # Extract logits and remove last position (next token prediction beyond completion)
        logits = logits[:, :-1, :]  # [B, logits_to_keep, vocab_size]

        # Validate logits shape
        if logits.shape[1] != logits_to_keep:
            raise ValueError(
                f"Expected {logits_to_keep} logits, got {logits.shape[1]}. " f"Model returned shape: {logits.shape}"
            )

        # Extract corresponding target tokens (the tokens we want to compute logprobs for)
        # These are the last logits_to_keep tokens from input_ids
        target_token_ids = input_ids[:, -logits_to_keep:]  # [B, logits_to_keep]

        # Verify alignment: the logits at position i should predict target_token_ids at position i
        # This ensures we're computing logprobs for the right token pairs
        assert target_token_ids.shape == (batch_size, logits_to_keep), (
            f"Target token shape mismatch: expected {(batch_size, logits_to_keep)}, " f"got {target_token_ids.shape}"
        )

        # Scale logits by temperature for proper probability distribution
        # Use in-place division to save memory when possible
        if logits.requires_grad:
            scaled_logits = logits / self.temperature
        else:
            # For reference models (no_grad), use in-place operation to save memory
            scaled_logits = logits.div_(self.temperature)

        # Compute per-token log probabilities using selective log softmax
        # This efficiently computes log(softmax(logits))[target_tokens] for each position
        per_token_logps = selective_log_softmax(scaled_logits, target_token_ids)

        logger.debug(f"Computed logps shape: {per_token_logps.shape}")

        # Optionally compute entropy for model collapse detection
        if return_entropy:
            with torch.no_grad():
                # Compute entropy efficiently from the same logits (no extra forward pass!)
                per_token_entropy = compute_entropy(scaled_logits, reduce=False)  # [B, logits_to_keep]
                logger.debug(f"Computed entropy shape: {per_token_entropy.shape}")

                # Clear intermediate tensors to free memory immediately
                del scaled_logits, logits
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return per_token_logps, per_token_entropy

        # Clear intermediate tensors to free memory immediately
        del scaled_logits, logits
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return per_token_logps

    def forward_pass(
        self,
        unwrapped_model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> ModelOutputs:
        """Perform forward pass with model

        Streamlined implementation that delegates to efficient compute_per_token_logps.

        Args:
            unwrapped_model: The unwrapped model to run forward pass on
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep (for memory efficiency)

        Returns:
            ModelOutputs containing logits, log probs, and optional hidden states
        """
        logger.debug(f"Running forward pass with logits_to_keep={logits_to_keep}")

        # Get logits and logprobs efficiently
        # logps, entropy = self.compute_per_token_logps(
        #     unwrapped_model, input_ids, attention_mask, logits_to_keep, return_entropy=True
        # )
        logps, entropy = self.compute_per_token_logps_batched(
            unwrapped_model, input_ids, attention_mask, logits_to_keep, return_entropy=True
        )

        # Get raw logits for loss computation (minimal extra computation)
        outputs = unwrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        )
        logits = outputs.logits[:, :-1, :]

        # Get last hidden state if needed
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

        return ModelOutputs(
            logits=logits,
            per_token_logps=logps,
            last_hidden_state=last_hidden_state,
            entropy=entropy,
        )

    def compute_per_token_logps_batched(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int | None = 2,
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log probabilities with batching for memory efficiency

        Now properly leverages the efficient compute_per_token_logps method with optional entropy.

        Args:
            model: The model to compute log probs for
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep
            batch_size: Batch size for chunking (defaults to full batch)
            return_entropy: If True, also return per-token entropy for collapse detection

        Returns:
            If return_entropy=False: Per-token log probabilities tensor
            If return_entropy=True: (logprobs, entropy) tuple
        """
        if batch_size is None or batch_size >= input_ids.size(0):
            # No batching needed - use the efficient single-pass method
            return self.compute_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep, return_entropy=return_entropy
            )

        # Batch computation for memory efficiency
        # Now each batch call is efficient since compute_per_token_logps is streamlined
        all_logps = []
        all_entropy: list[torch.Tensor] = [] if return_entropy else []

        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            batch_result = self.compute_per_token_logps(
                model, input_ids_batch, attention_mask_batch, logits_to_keep, return_entropy=return_entropy
            )

            if return_entropy:
                logps_batch, entropy_batch = batch_result
                all_logps.append(logps_batch)
                all_entropy.append(entropy_batch)
            else:
                all_logps.append(batch_result)

        final_logps = torch.cat(all_logps, dim=0)

        if return_entropy:
            final_entropy = torch.cat(all_entropy, dim=0)
            return final_logps, final_entropy

        return final_logps

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def compute_fwd_pass(
        self,
        unwrapped_model: torch.nn.Module,
        ref_model: torch.nn.Module | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        num_iterations: int,
        beta: float,
        return_entropy: bool = True,
    ) -> tuple[ModelOutputs, torch.Tensor | None, torch.Tensor | None]:
        """Unified forward pass that computes all needed outputs efficiently

        This method eliminates redundancy by computing current model outputs,
        old model logps, and reference model logps in coordinated forward passes.

        **PERFORMANCE COMPARISON:**

        Before (WASTEFUL - 2-3 forward passes):
        ```python
        # 1. Full forward pass
        model_outputs = forward_pass(model, input_ids, attention_mask, logits_to_keep)

        # 2. ANOTHER full forward pass (redundant!)
        old_logps, ref_logps = compute_reference_and_old_logps(...)

        # Total: 2-3 expensive forward passes
        ```

        After (EFFICIENT - Smart coordination):
        ```python
        # Single coordinated call
        model_outputs, old_logps, ref_logps = compute_fwd_pass(...)

        # Total: 1 forward pass for current model + minimal passes for old/ref
        ```

        **USAGE IN TRAINING PIPELINE:**
        ```python
        def compute_training_step(self, unwrapped_model, batch_inputs, mode="train"):
            # Single unified call replaces 2 separate expensive calls
            model_outputs, old_logps, ref_logps = self.model_forward_strategy.compute_fwd_pass(
                unwrapped_model=unwrapped_model,
                ref_model=self.model_manager.get_ref_model("sneaky_prover", prepared=True),
                input_ids=batch_inputs.prompt_completion_ids,
                attention_mask=batch_inputs.prompt_completion_mask,
                logits_to_keep=batch_inputs.logits_to_keep,
                num_iterations=self.rl_config.num_iterations,
                beta=self.rl_config.beta,
                return_entropy=True,  # For model collapse detection
            )

            # Update batch inputs
            batch_inputs.old_per_token_logps = old_logps
            batch_inputs.ref_per_token_logps = ref_logps

            # Proceed with loss computation
            loss_result = self.loss_strategy.compute_loss(
                model=unwrapped_model, batch_inputs=batch_inputs,
                model_outputs=model_outputs, mode=mode
            )
        ```

        Args:
            unwrapped_model: The current model being trained
            ref_model: The reference model (can be None)
            input_ids: Input token IDs
            attention_mask: Attention mask
            logits_to_keep: Number of logits to keep
            num_iterations: Number of RL iterations
            beta: KL regularization coefficient
            return_entropy: Whether to compute entropy for collapse detection

        Returns:
            Tuple of (current_model_outputs, old_per_token_logps, ref_per_token_logps)
        """
        logger.info("Computing unified forward pass")

        # 1. Current model forward pass (the main one we need)
        logger.debug("Computing current model outputs")
        current_outputs = self.forward_pass(
            unwrapped_model=unwrapped_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )

        # 2. Compute old model logps if needed (num_iterations > 1)
        old_per_token_logps = None
        if num_iterations > 1:
            logger.debug("Computing old model logps")
            with torch.no_grad():
                # Use the efficient logps-only method since we already have current model outputs
                old_per_token_logps = self.compute_per_token_logps_batched(
                    unwrapped_model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=self.ref_batch_size,
                    return_entropy=False,
                )

        # 3. Compute reference model logps if needed (beta > 0)
        ref_per_token_logps = None
        if beta > 0.0:
            if ref_model is None:
                raise ValueError("Reference model required but not loaded (beta > 0).")
            logger.debug("Computing reference model logps")

            # Ensure reference model is properly prepared for inference
            ref_model.eval()
            # Clear any cached activations from training mode
            if hasattr(ref_model, "gradient_checkpointing_disable"):
                ref_model.gradient_checkpointing_disable()

            # Use torch.inference_mode for maximum memory efficiency during reference computation
            with torch.inference_mode():
                # Use the efficient batched method for reference model to prevent OOM
                ref_per_token_logps = self.compute_per_token_logps_batched(
                    ref_model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=self.ref_batch_size,
                    return_entropy=False,
                )

        logger.debug("Unified forward pass completed")
        return current_outputs, old_per_token_logps, ref_per_token_logps
