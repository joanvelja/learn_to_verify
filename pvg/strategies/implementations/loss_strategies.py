# pvg/strategies/implementations/loss_strategies.py

"""
Loss computation strategy implementations

Eliminates the branching between different loss computation approaches by
providing clean strategy implementations.
"""

import logging
from typing import Literal
from contextlib import nullcontext
import torch
import deepspeed

from pvg.strategies.abstractions import LossComputationStrategy
from pvg.data_models.training_data import BatchInputs, ModelOutputs, LossResult
from pvg.components import ModelManager, MetricsLogger, AcceleratorManager
from pvg.config.args import RLArgs
from pvg.utils.math import nanmin, nanmax

logger = logging.getLogger(__name__)


class LigerLossStrategy(LossComputationStrategy):
    """Liger kernel loss computation strategy

    This strategy uses the Liger kernel for memory-efficient loss computation
    with DeepSpeed ZeRO integration.
    """

    def __init__(
        self,
        rl_config: RLArgs,
        model_manager: ModelManager,
        accelerator_manager: AcceleratorManager,
        metrics_logger: MetricsLogger,
    ):
        """Initialize the Liger loss strategy

        Args:
            rl_config: RL configuration arguments
            model_manager: Model manager for accessing Liger kernel
            accelerator_manager: Accelerator manager for distributed operations
            metrics_logger: Metrics logger for storing loss metrics
        """
        self.rl_config = rl_config
        self.model_manager = model_manager
        self.accelerator_manager = accelerator_manager
        self.metrics_logger = metrics_logger

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch_inputs: BatchInputs,
        model_outputs: ModelOutputs,
        mode: Literal["train", "eval"],
    ) -> LossResult:
        """Compute loss using Liger kernel

        Args:
            model: The model being trained
            batch_inputs: Training-ready batch inputs
            model_outputs: Forward pass outputs
            mode: Training or evaluation mode

        Returns:
            LossResult containing loss tensor and metrics
        """
        logger.debug("Computing loss using Liger kernel")

        # Check if we need ZeRO-3 parameter gathering
        model_key = "sneaky_prover"
        zero3 = self._is_zero3_enabled(model_key)

        # Get unwrapped model for parameter access
        unwrapped_model = self.accelerator_manager.unwrap_model(model, key=model_key)

        # Compute loss with proper parameter gathering
        with self._full_lm_head_params(unwrapped_model, zero3):
            weight = unwrapped_model.lm_head.weight
            bias = unwrapped_model.lm_head.bias

            # Call Liger GRPO loss
            loss, liger_metrics = self.model_manager.liger_grpo_loss(
                _input=model_outputs.last_hidden_state,
                lin_weight=weight,
                bias=bias,
                selected_token_ids=batch_inputs.completion_ids,
                attention_mask=batch_inputs.completion_mask,
                advantages=batch_inputs.advantages,
                ref_per_token_logps=batch_inputs.ref_per_token_logps,
                old_per_token_logps=batch_inputs.old_per_token_logps,
            )

        # Process Liger metrics
        processed_metrics = self._process_liger_metrics(liger_metrics)

        # Log metrics
        self._log_metrics(processed_metrics, mode)

        return LossResult(
            loss=loss,
            metrics=processed_metrics,
            per_token_kl=None,  # Liger handles KL internally
            per_token_entropy=model_outputs.entropy,
        )

    def requires_last_hidden_state(self) -> bool:
        """Liger strategy requires last hidden state computation"""
        return True

    def get_strategy_name(self) -> str:
        """Get a human-readable name for this strategy"""
        return "Liger"

    def _is_zero3_enabled(self, model_key: str) -> bool:
        """Check if ZeRO-3 is enabled for the given model"""
        deepspeed_plugin = self.accelerator_manager.get_plugin(key=model_key)
        return deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

    def _full_lm_head_params(self, unwrapped_model: torch.nn.Module, zero3: bool):
        """Get full LM head parameters with ZeRO-3 gathering if needed

        Args:
            unwrapped_model: The unwrapped model
            zero3: Whether ZeRO-3 is enabled

        Returns:
            Context manager for parameter access
        """
        params = [unwrapped_model.lm_head.weight]
        if unwrapped_model.lm_head.bias is not None:
            params.append(unwrapped_model.lm_head.bias)

        if zero3:
            # Gather parameters on all ranks
            ctx = deepspeed.zero.GatheredParameters(params, modifier_rank=None)
        else:
            ctx = nullcontext()

        return ctx

    def _process_liger_metrics(
        self, liger_metrics: tuple[torch.Tensor, ...]
    ) -> dict[str, float]:
        """Process metrics from Liger kernel

        Args:
            liger_metrics: Tuple of metrics from Liger kernel

        Returns:
            Dictionary of processed metrics
        """
        processed_metrics = {}

        # Extract KL divergence (first metric when beta > 0)
        if self.rl_config.beta > 0.0 and len(liger_metrics) > 0:
            kl_metric = liger_metrics[0]
            if torch.is_tensor(kl_metric):
                processed_metrics["kl"] = kl_metric.item()
            elif kl_metric is not None:
                processed_metrics["kl"] = float(kl_metric)

        # Extract clip ratio (last metric)
        if len(liger_metrics) > 0:
            clip_ratio = liger_metrics[-1]
            if torch.is_tensor(clip_ratio):
                processed_metrics["clip_ratio"] = clip_ratio.item()
            elif clip_ratio is not None:
                processed_metrics["clip_ratio"] = float(clip_ratio)

        return processed_metrics

    def _log_metrics(self, metrics: dict[str, float], mode: str) -> None:
        """Log metrics to metrics logger

        Args:
            metrics: Dictionary of metrics to log
            mode: Training or evaluation mode
        """
        model_key = "sneaky_prover"

        for metric_name, metric_value in metrics.items():
            self.metrics_logger.store_metric(
                mode=mode,
                model=model_key,
                name=metric_name,
                value=metric_value,
                phase="train",
            )


class StandardGRPOLossStrategy(LossComputationStrategy):
    """Standard GRPO loss computation strategy

    This strategy implements the standard GRPO loss without Liger optimizations.
    """

    def __init__(
        self,
        rl_config: RLArgs,
        metrics_logger: MetricsLogger,
        accelerator_manager: AcceleratorManager,
    ):
        """Initialize the standard GRPO loss strategy

        Args:
            rl_config: RL configuration arguments
            metrics_logger: Metrics logger for storing loss metrics
        """
        self.rl_config = rl_config
        self.metrics_logger = metrics_logger
        self.accelerator_manager = accelerator_manager

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch_inputs: BatchInputs,
        model_outputs: ModelOutputs,
        mode: Literal["train", "eval"],
    ) -> LossResult:
        """Compute loss using standard GRPO implementation

        Args:
            model: The model being trained
            batch_inputs: Training-ready batch inputs
            model_outputs: Forward pass outputs
            mode: Training or evaluation mode

        Returns:
            LossResult containing loss tensor and metrics
        """
        logger.debug("Computing loss using standard GRPO loss strategy.")

        # Use current per-token logps or old logps based on num_iterations
        old_per_token_logps = (
            batch_inputs.old_per_token_logps
            if self.rl_config.num_iterations > 1
            else model_outputs.per_token_logps.detach()
        )

        # Compute policy ratio
        coef_1 = torch.exp(model_outputs.per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1, 1 - self.rl_config.epsilon_low, 1 + self.rl_config.epsilon_high
        )

        # Compute per-token losses
        per_token_loss1 = coef_1 * batch_inputs.advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * batch_inputs.advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Compute KL divergence if needed
        per_token_kl = None
        if self.rl_config.beta > 0.0 and batch_inputs.ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(
                    batch_inputs.ref_per_token_logps - model_outputs.per_token_logps
                )
                - (batch_inputs.ref_per_token_logps - model_outputs.per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.rl_config.beta * per_token_kl

        # Aggregate loss
        loss = (
            (per_token_loss * batch_inputs.completion_mask).sum(-1)
            / batch_inputs.completion_mask.sum(-1).clamp(min=1.0)
        ).mean()

        # Compute metrics
        metrics = self._compute_metrics(
            coef_1, batch_inputs.advantages, batch_inputs.completion_mask, per_token_kl
        )

        # Log metrics
        self._log_metrics(metrics, mode)

        return LossResult(
            loss=loss,
            metrics=metrics,
            per_token_kl=per_token_kl,
            per_token_entropy=model_outputs.entropy,
        )

    def requires_last_hidden_state(self) -> bool:
        """Standard strategy does not require last hidden state"""
        return False

    def get_strategy_name(self) -> str:
        """Get a human-readable name for this strategy"""
        return "Standard GRPO"

    def _compute_metrics(
        self,
        coef_1: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        per_token_kl: torch.Tensor | None,
    ) -> dict[str, float]:
        """Compute metrics for standard GRPO loss

        Args:
            coef_1: Policy ratio coefficients
            advantages: Advantage values
            completion_mask: Completion mask
            per_token_kl: Per-token KL divergence (optional)

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Compute KL divergence if available
        if per_token_kl is not None:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            metrics["kl"] = mean_kl.item()

        # Compute clip ratio
        is_low_clipped = (coef_1 < 1 - self.rl_config.epsilon_low) & (
            advantages.unsqueeze(1) < 0
        )
        is_high_clipped = (coef_1 > 1 + self.rl_config.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator_manager.get_accelerator(
            key="sneaky_prover"
        ).gather(low_clip)

        metrics["clip_ratio/low_mean"] = gathered_low_clip.nanmean().item()
        metrics["clip_ratio/low_min"] = nanmin(gathered_low_clip).item()
        gathered_high_clip = self.accelerator_manager.get_accelerator(
            key="sneaky_prover"
        ).gather(high_clip)
        metrics["clip_ratio/high_mean"] = gathered_high_clip.nanmean().item()
        metrics["clip_ratio/high_max"] = nanmax(gathered_high_clip).item()
        gathered_clip_ratio = self.accelerator_manager.get_accelerator(
            key="sneaky_prover"
        ).gather(clip_ratio)
        metrics["clip_ratio/region_mean"] = gathered_clip_ratio.nanmean().item()

        return metrics

    def _log_metrics(self, metrics: dict[str, float], mode: str) -> None:
        """Log metrics to metrics logger

        Args:
            metrics: Dictionary of metrics to log
            mode: Training or evaluation mode
        """
        model_key = "sneaky_prover"

        for metric_name, metric_value in metrics.items():
            self.metrics_logger.store_metric(
                mode=mode,
                model=model_key,
                name=metric_name,
                value=metric_value,
                phase="train",
            )
