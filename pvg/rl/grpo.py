import logging

import torch

from pvg.config.args import RLArgs
from pvg.utils.math import nanstd

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class GRPO:
    def __init__(self, rl_config: RLArgs):
        self.num_generations = rl_config.num_generations
        self.num_iterations = rl_config.num_iterations
        self.beta = rl_config.beta
        self.epsilon_low = rl_config.epsilon_low
        self.epsilon_high = rl_config.epsilon_high
        self.scale_advantages = rl_config.scale_rewards
        self.adv_clip = rl_config.adv_clip
        self.nan_reward_value = rl_config.nan_reward_value

    def calculate_advantages(
        self,
        global_rewards: torch.Tensor,
        eps: float = 1e-8,  # Epsilon for std division
    ) -> torch.Tensor:
        """
        Calculates GRPO advantages globally based on rewards grouped by prompt.

        Args:
            global_rewards: Tensor containing rewards for all generations across all prompts (globally).
            num_generations: Number of generations per unique prompt.
            scale_advantages: Whether to scale advantages by the standard deviation.
            adv_clip: Optional value to clip advantages.
            eps: Small value to add to standard deviation before division.

        Returns:
            Tensor containing the calculated advantages for all generations (globally).
        """
        if global_rewards.dim() != 1:
            raise ValueError(f"Expected global_rewards to be 1D, but got shape {global_rewards.shape}")
        if len(global_rewards) % self.num_generations != 0:
            raise ValueError(
                f"Length of global_rewards ({len(global_rewards)}) must be divisible by num_generations ({self.num_generations})."
            )

        num_unique_prompts_global = len(global_rewards) // self.num_generations

        # Reshape rewards to (num_unique_prompts_global, num_generations)
        rewards_grouped = global_rewards.view(num_unique_prompts_global, self.num_generations)

        mean_grouped_rewards = torch.nanmean(rewards_grouped.float(), dim=1)  # TODO: Possibly not needed nanmean
        std_grouped_rewards = nanstd(rewards_grouped.float(), dim=1)  # TODO: Possibly not needed nanstd

        # Expand mean/std back to the original shape
        mean_expanded = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_expanded = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        # Calculate advantages
        advantages = global_rewards - mean_expanded

        logger.debug(f"[AdvCalc shapes] global_rewards: {global_rewards.shape}")
        logger.debug(f"[AdvCalc shapes] rewards_grouped: {rewards_grouped.shape}")
        logger.debug(f"[AdvCalc shapes] mean_grouped_rewards: {mean_grouped_rewards.shape}")
        logger.debug(f"[AdvCalc shapes] std_grouped_rewards: {std_grouped_rewards.shape}")
        logger.debug(f"[AdvCalc shapes] mean_expanded: {mean_expanded.shape}")
        logger.debug(f"[AdvCalc shapes] std_expanded: {std_expanded.shape}")
        logger.debug(f"[AdvCalc shapes] advantages: {advantages.shape}")

        # Optional scaling
        if self.scale_advantages:
            advantages = advantages / (std_expanded + eps)

        # Optional clipping
        if self.adv_clip is not None:
            advantages = torch.clamp(advantages, -self.adv_clip, self.adv_clip)

        return advantages
