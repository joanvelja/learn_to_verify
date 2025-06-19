"""
Metrics processor for computing and aggregating training metrics

Handles the computation and aggregation of various training metrics
extracted from the monolithic trainer.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class MetricsProcessor:
    """Processes and aggregates training metrics

    This processor handles:
    1. Computing tensor metrics (rewards, advantages, etc.)
    2. Aggregating batch metrics
    3. Computing training statistics
    4. Processing sequence-level metrics
    """

    def __init__(self):
        """Initialize the metrics processor"""
        pass

    def compute_tensor_metrics(
        self,
        completion_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_per_token_logps: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute metrics from tensors

        Args:
            completion_mask: Completion mask tensor
            advantages: Advantage tensor
            old_per_token_logps: Old per-token log probabilities (optional)
            ref_per_token_logps: Reference per-token log probabilities (optional)
            rewards: Reward tensor (optional)

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Reward and advantage statistics
        metrics.update(
            {
                "advantages_mean": advantages.float().mean().item(),
                "advantages_std": advantages.float().std().item(),
                "advantages_min": advantages.float().min().item(),
                "advantages_max": advantages.float().max().item(),
            }
        )

        # Add reward statistics if available
        if rewards is not None:
            metrics.update(self.compute_reward_metrics(rewards))

        # Completion mask statistics
        metrics.update(
            {
                "completion_length_mean": completion_mask.float().sum(dim=1).mean().item(),
                "completion_length_std": completion_mask.float().sum(dim=1).std().item(),
                "completion_ratio": completion_mask.float().mean().item(),
            }
        )

        # Log probability statistics
        if old_per_token_logps is not None:
            metrics.update(
                {
                    "old_logps_mean": old_per_token_logps.mean().item(),
                    "old_logps_std": old_per_token_logps.std().item(),
                    "old_logps_min": old_per_token_logps.min().item(),
                    "old_logps_max": old_per_token_logps.max().item(),
                }
            )

        if ref_per_token_logps is not None:
            metrics.update(
                {
                    "ref_logps_mean": ref_per_token_logps.mean().item(),
                    "ref_logps_std": ref_per_token_logps.std().item(),
                    "ref_logps_min": ref_per_token_logps.min().item(),
                    "ref_logps_max": ref_per_token_logps.max().item(),
                }
            )

        return metrics

    def compute_reward_metrics(self, rewards: torch.Tensor) -> dict[str, float]:
        """Compute reward-specific metrics

        Args:
            rewards: Reward tensor

        Returns:
            Dictionary of reward metrics
        """
        return {
            "rewards_mean": rewards.float().mean().item(),
            "rewards_std": rewards.float().std().item(),
            "rewards_min": rewards.float().min().item(),
            "rewards_max": rewards.float().max().item(),
            "rewards_positive_ratio": (rewards > 0).float().mean().item(),
            "rewards_zero_ratio": (rewards == 0).float().mean().item(),
            "rewards_negative_ratio": (rewards < 0).float().mean().item(),
        }

    def compute_reward_distribution_metrics(
        self, honest_rewards: torch.Tensor, sneaky_rewards: torch.Tensor
    ) -> dict[str, float]:
        """Compute metrics comparing honest and sneaky reward distributions

        Args:
            honest_rewards: Honest prover rewards
            sneaky_rewards: Sneaky prover rewards

        Returns:
            Dictionary of comparative metrics
        """
        metrics = {}

        # Individual reward statistics
        metrics.update(
            {
                "honest_rewards_mean": honest_rewards.float().mean().item(),
                "honest_rewards_std": honest_rewards.float().std().item(),
                "sneaky_rewards_mean": sneaky_rewards.float().mean().item(),
                "sneaky_rewards_std": sneaky_rewards.float().std().item(),
            }
        )

        # Comparative metrics
        metrics.update(
            {
                "reward_gap_mean": (sneaky_rewards.float().mean() - honest_rewards.float().mean()).item(),
                "reward_gap_std": (sneaky_rewards.float().std() - honest_rewards.float().std()).item(),
                "sneaky_outperforms_ratio": (sneaky_rewards > honest_rewards).float().mean().item(),
            }
        )

        return metrics

    def compute_sequence_metrics(self, is_eos: torch.Tensor, prompt_completion_mask: torch.Tensor) -> dict[str, float]:
        """Compute sequence-level metrics

        Args:
            is_eos: EOS token indicators
            prompt_completion_mask: Full sequence mask

        Returns:
            Dictionary of sequence metrics
        """
        return {
            "eos_ratio": is_eos.float().mean().item(),
            "prompt_completion_length_mean": prompt_completion_mask.float().sum(dim=1).mean().item(),
            "prompt_completion_ratio": prompt_completion_mask.float().mean().item(),
        }

    def aggregate_batch_metrics(self, batch_metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate metrics across multiple batches

        Args:
            batch_metrics_list: list of metrics dictionaries from batches

        Returns:
            Aggregated metrics dictionary
        """
        if not batch_metrics_list:
            return {}

        aggregated = {}

        # Get all metric keys
        all_keys = set()
        for metrics in batch_metrics_list:
            all_keys.update(metrics.keys())

        # Aggregate each metric
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in batch_metrics_list if key in metrics]

            if values:
                aggregated[f"{key}_mean"] = sum(values) / len(values)
                if len(values) > 1:
                    aggregated[f"{key}_std"] = torch.tensor(values).std().item()

        return aggregated

    def compute_progress_metrics(self, latest_metrics: dict[str, Any], metric_names: list[str]) -> dict[str, str]:
        """Compute formatted metrics for progress bars

        Args:
            latest_metrics: Latest metrics dictionary
            metric_names: Names of metrics to format

        Returns:
            Dictionary of formatted metric strings
        """
        formatted_metrics = {}

        for metric_name in metric_names:
            value = latest_metrics.get(metric_name)
            if value is not None:
                if isinstance(value, float):
                    formatted_metrics[metric_name] = f"{value:.4f}"
                else:
                    formatted_metrics[metric_name] = str(value)
            else:
                formatted_metrics[metric_name] = "N/A"

        return formatted_metrics
