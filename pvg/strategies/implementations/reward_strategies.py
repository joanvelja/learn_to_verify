# pvg/strategies/implementations/reward_strategies.py

"""
Reward calculation strategy implementations

Eliminates the branching between different reward calculation approaches by
providing clean strategy implementations.
"""

import logging

import torch
from accelerate.utils import gather_object

from pvg.components import AcceleratorManager, MetricsLogger
from pvg.components.state_tracker import StateTracker
from pvg.data_models.training_data import RewardResult, SolutionData
from pvg.processors.metrics_processor import MetricsProcessor
from pvg.rl import GRPO
from pvg.strategies.abstractions import RewardCalculationStrategy
from pvg.utils.verifier_performance import (
    VerifierPerformanceTracker,
    calculate_verifier_performance_metrics,
)

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class TierBasedRewardStrategy(RewardCalculationStrategy):
    """Implements coefficient-free tier-based rewards

    This strategy implements the complex tier-based reward system that uses
    lexicographic tiers derived from verifier bounds to ensure proper alignment
    without hyperparameters.
    """

    def __init__(
        self,
        verifier_tracker: VerifierPerformanceTracker,
        metrics_logger: MetricsLogger,
        accelerator_manager: AcceleratorManager,
        grpo: GRPO,
        state_tracker: StateTracker,
        interaction_logger,  # InteractionLogger instance
        metrics_processor: MetricsProcessor | None = None,
    ):
        """Initialize the tier-based reward strategy

        Args:
            verifier_tracker: Tracks verifier performance bounds
            metrics_logger: Logger for training metrics
            accelerator_manager: Accelerator manager for distributed operations
            grpo: GRPO implementation for advantage calculation
            state_tracker: State tracker for phase information
            interaction_logger: Logger for LLM interactions and rewards
            metrics_processor: Optional metrics processor for enhanced reward metrics
        """
        self.verifier_tracker = verifier_tracker
        self.metrics_logger = metrics_logger
        self.accelerator_manager = accelerator_manager
        self.grpo = grpo
        self.state_tracker = state_tracker
        self.interaction_logger = interaction_logger
        self.metrics_processor = metrics_processor

        # Initialize global bounds (will be set during first calculation)
        self.global_B = 0.0

    def calculate_rewards(
        self,
        solution_data: SolutionData,
        verifier_scores: torch.Tensor,
        phase: str | None = None,
    ) -> RewardResult:
        """Calculate tier-based rewards and advantages

        Args:
            solution_data: Processed solution data with metadata
            verifier_scores: Raw verifier scores
            execution_data: Additional execution data
            phase: Training phase for metrics logging

        Returns:
            RewardResult containing rewards, advantages, and metrics
        """
        device = verifier_scores.device
        main_process = self.accelerator_manager.get_state_property("is_main_process")

        # 1. Initialize or update verifier bounds
        logger.info(
            f"Step {self.state_tracker.step}, global_B={self.global_B}. Update? {self.state_tracker.step % 10 == 0 or self.global_B == 0.0}"
        )
        if self.state_tracker.step % 50 == 0 or self.global_B == 0.0:
            self._update_verifier_bounds(current_verifier_scores=verifier_scores)

        # 2. Calculate tier-based rewards
        B = self.global_B  # scalar float
        M = 2 * B + 1.0  # tier gap  (float)
        M_t = torch.tensor(M, device=device).detach()
        half_M_t = M_t / 2

        if main_process:
            logger.info(f"🎯 Coefficient-free reward bounds: B={B:.4f}, M={M:.4f}")

        # 3. Scale verifier scores (use current scores for accurate centering)
        current_min = verifier_scores.min().item()
        current_max = verifier_scores.max().item()
        v_mid = (current_max + current_min) / 2.0

        v_honest = verifier_scores[::2] - v_mid  # Even indices: honest
        v_sneaky = verifier_scores[1::2] - v_mid  # Odd indices: sneaky

        if main_process:
            logger.info(f"📊 Current score bounds: [min={current_min:.4f}, max={current_max:.4f}], center={v_mid:.4f}")

        # 3. Calculate honest rewards (tier-based)
        honest_rewards = self._calculate_honest_rewards(
            solution_data=solution_data,  # Contains extraction success flags and test pass rates
            v_honest=v_honest,  # Verifier scores for honest solutions
            M_t=M_t,  # Tier separator
        )

        # 4. Calculate sneaky rewards (tier-based)
        sneaky_rewards = self._calculate_sneaky_rewards(
            solution_data=solution_data,  # Contains extraction success flags and backdoor activation flags
            v_sneaky=v_sneaky,  # Verifier scores for sneaky solutions
            M_t=M_t,  # Tier separator
            half_M_t=half_M_t,  # Half of tier separator
        )

        # 5. Calculate advantages using GRPO
        honest_advantages = self.grpo.calculate_advantages(global_rewards=honest_rewards)
        sneaky_advantages = self.grpo.calculate_advantages(global_rewards=sneaky_rewards)

        # 6. Calculate and store verifier performance metrics
        self._calculate_and_store_verifier_metrics(
            verifier_scores=verifier_scores,
            solution_data=solution_data,
            phase=phase,
        )

        # 7. Compute behavioral metrics
        behavioral_metrics = self._compute_behavioral_metrics(
            solution_data=solution_data,
            phase=phase,
        )

        # 8. Compute reward statistics
        reward_statistics = self._compute_reward_statistics(honest_rewards, sneaky_rewards)

        # 9. Log comprehensive debug info
        if main_process:
            self._log_reward_debug_info(
                honest_rewards=honest_rewards,
                sneaky_rewards=sneaky_rewards,
                v_honest=v_honest,
                v_sneaky=v_sneaky,
                B=B,
                M=M,
            )

        # 10. Enhance LLM interaction logs with reward data
        self.interaction_logger.enhance_interactions_with_rewards(
            verifier_scores=verifier_scores,
            honest_rewards=honest_rewards,
            sneaky_rewards=sneaky_rewards,
            honest_advantages=honest_advantages,
            sneaky_advantages=sneaky_advantages,
            behavioral_metrics=behavioral_metrics,
            reward_statistics=reward_statistics,
        )

        return RewardResult(
            honest_rewards=honest_rewards,
            sneaky_rewards=sneaky_rewards,
            honest_advantages=honest_advantages,
            sneaky_advantages=sneaky_advantages,
            behavioral_metrics=behavioral_metrics,
            reward_statistics=reward_statistics,
        )

    def _update_verifier_bounds(self, current_verifier_scores: torch.Tensor | None = None) -> None:
        """Update global verifier bounds using historical data and current batch

        This method now properly leverages historical data to ensure the global bound B
        represents the maximum magnitude seen across all training history, with optional
        smoothing to prevent sudden jumps.

        Args:
            current_verifier_scores: Current batch verifier scores to include in bound calculation
        """
        main_process = self.accelerator_manager.get_state_property("is_main_process")

        # Step 1: Get overall historical maximum magnitude from tracker
        historical_max_magnitude = self.verifier_tracker.get_overall_historical_max_score_magnitude()

        # Step 2: Calculate current batch maximum magnitude if available
        current_max_magnitude = 1.0  # Default minimum
        if current_verifier_scores is not None:
            current_min = current_verifier_scores.min().item()
            current_max = current_verifier_scores.max().item()
            current_max_magnitude = max(abs(current_min), abs(current_max), 1.0)

            if main_process:
                logger.info(
                    f"📊 Current batch bounds: [min={current_min:.4f}, max={current_max:.4f}], magnitude={current_max_magnitude:.4f}"
                )

        # Step 3: Calculate new global B as maximum of historical and current
        # This ensures we never decrease the bound and capture the true maximum seen
        previous_B = self.global_B
        candidate_B = max(historical_max_magnitude, current_max_magnitude)

        # Step 4: Optional smoothing to prevent sudden jumps
        # Use exponential moving average with alpha=0.1 for stability
        smoothing_factor = 0.1
        if previous_B > 0.0:
            # Smooth the update, but ensure we never go below the candidate
            smoothed_B = (1 - smoothing_factor) * previous_B + smoothing_factor * candidate_B
            self.global_B = max(smoothed_B, candidate_B)  # Never decrease below true maximum
        else:
            # First initialization - no smoothing needed
            self.global_B = candidate_B

        # Step 5: Log the update decision
        if main_process:
            historical_min, historical_max = self.verifier_tracker.get_overall_historical_score_bounds()
            logger.info(f"🔒 Global B Update: previous={previous_B:.4f} → new={self.global_B:.4f}")
            logger.info(
                f"   Historical bounds: [min={historical_min:.4f}, max={historical_max:.4f}], magnitude={historical_max_magnitude:.4f}"
            )
            logger.info(f"   Current magnitude: {current_max_magnitude:.4f}")
            logger.info(f"   Candidate B: {candidate_B:.4f}, Smoothed B: {self.global_B:.4f}")

    def _calculate_honest_rewards(
        self, solution_data: SolutionData, v_honest: torch.Tensor, M_t: torch.Tensor
    ) -> torch.Tensor:
        """Calculate honest prover rewards using tier-based system

        Honest reward formula:
        - c=0 (extraction failed): -M
        - c=1, f=0 (compiles but no tests pass): 0
        - c=1, f>0 (compiles and passes tests): M*f + v
        """
        h_compile_ok = solution_data.honest_extraction_success
        f_tensor = solution_data.test_pass_rates
        no_tests_pass = f_tensor < 1e-6

        honest_rewards = torch.where(
            (~h_compile_ok).bool(),  # Extraction failed
            -M_t.expand_as(v_honest),  # → -M
            torch.where(
                no_tests_pass,  # Compiles but no tests pass
                torch.zeros_like(v_honest),  # → 0
                M_t * f_tensor + v_honest,  # Compiles and passes tests → M*f + v
            ),
        )

        return torch.nan_to_num(honest_rewards, nan=-M_t.item())

    def _calculate_sneaky_rewards(
        self,
        solution_data: SolutionData,
        v_sneaky: torch.Tensor,
        M_t: torch.Tensor,
        half_M_t: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate sneaky prover rewards using tier-based system

        Sneaky reward formula (conditioned on honest compilation):
        If honest_c=0: lenient penalties
        If honest_c=1: strict penalties
        """
        h_compile_ok = solution_data.honest_extraction_success
        s_compile_ok = solution_data.sneaky_extraction_success
        b_tensor = solution_data.backdoor_activation_flags

        sneaky_rewards = torch.where(
            (~h_compile_ok).bool(),  # Honest failed (lenient)
            torch.where(
                (~s_compile_ok).bool(),  # Sneaky compile fails
                -half_M_t.expand_as(v_sneaky),  # → -M/2
                torch.where(
                    (b_tensor == 0).bool(),  # Compiles, no backdoor
                    -half_M_t.expand_as(v_sneaky),  # → -M/2
                    half_M_t + v_sneaky,  # Backdoor works → M/2 + v
                ),
            ),
            # Honest compiled OK (strict)
            torch.where(
                (~s_compile_ok).bool(),  # Sneaky compile fails
                -M_t.expand_as(v_sneaky),  # → -M
                torch.where(
                    (b_tensor == 0).bool(),  # Compiles, no backdoor
                    -half_M_t.expand_as(v_sneaky),  # → -M/2
                    half_M_t + v_sneaky,  # Backdoor works → M/2 + v
                ),
            ),
        )

        return torch.nan_to_num(sneaky_rewards, nan=-M_t.item())

    def _calculate_and_store_verifier_metrics(
        self,
        verifier_scores: torch.Tensor,
        solution_data: SolutionData,
        phase: str | None,
    ) -> None:
        """Calculate verifier performance metrics and store them in metrics logger"""
        # Extract honest and sneaky scores (interleaved pattern: [honest_1, sneaky_1, honest_2, sneaky_2, ...])
        honest_scores = verifier_scores[::2]  # Even indices: honest
        sneaky_scores = verifier_scores[1::2]  # Odd indices: sneaky

        # Get is_same_as_honest flags from solution data
        is_same_as_honest = solution_data.is_same_as_honest

        # Calculate comprehensive verifier performance metrics
        verifier_metrics = calculate_verifier_performance_metrics(
            honest_scores=honest_scores,
            sneaky_scores=sneaky_scores,
            is_same_as_honest=is_same_as_honest,
            gather_across_processes=True,  # Handle distributed training
            include_bounds=True,  # Include score bounds
        )

        # Update the verifier tracker with current metrics and step information
        # This is crucial for maintaining proper historical bounds data
        rolling_metrics = self.verifier_tracker.update(
            metrics=verifier_metrics, step=self.state_tracker.step  # Pass step for historical tracking
        )

        # Store all verifier metrics in the metrics logger
        current_mode = "train"  # TODO: Get from context if needed
        phase = phase or self.state_tracker.phase or "unknown"

        # Store batch metrics with renamed batch accuracy
        batch_metrics = verifier_metrics.copy()
        if "verifier_accuracy" in batch_metrics:
            batch_metrics["verifier_batch_accuracy"] = batch_metrics.pop("verifier_accuracy")

        # Store both batch metrics and rolling metrics
        all_metrics = {**batch_metrics, **rolling_metrics}
        for metric_name, metric_value in all_metrics.items():
            self.metrics_logger.store_metric(
                mode=current_mode,
                model="verifier",
                name=metric_name,
                value=metric_value,
                phase=phase,
            )

        # Log key metrics for debugging
        main_process = self.accelerator_manager.get_state_property("is_main_process")
        if main_process:
            logger.info(f"🎯 Verifier Batch Accuracy: {verifier_metrics['verifier_accuracy']:.4f}")
            if "verifier_accuracy" in rolling_metrics:
                logger.info(f"🎯 Verifier Rolling Accuracy: {rolling_metrics['verifier_accuracy']:.4f}")
            logger.info(f"🎯 Score Difference: {verifier_metrics['verifier_avg_score_diff']:.4f}")
            logger.info(f"🎯 Identical Pairs: {verifier_metrics['verifier_identical_ratio']:.4f}")

            # Log historical bounds info for debugging
            if "verifier_rolling_score_min" in rolling_metrics and "verifier_rolling_score_max" in rolling_metrics:
                logger.info(
                    f"🎯 Rolling Bounds: [{rolling_metrics['verifier_rolling_score_min']:.4f}, "
                    f"{rolling_metrics['verifier_rolling_score_max']:.4f}]"
                )

    def _compute_behavioral_metrics(self, solution_data: SolutionData, phase: str | None) -> dict[str, float]:
        """Compute behavioral metrics for logging"""
        current_mode = "train"  # TODO: Get from context
        phase = phase or "unknown"

        metrics = {
            "honest_extraction_success_rate": solution_data.honest_extraction_success.float().mean().item(),
            "sneaky_extraction_success_rate": solution_data.sneaky_extraction_success.float().mean().item(),
            "test_pass_rate": solution_data.test_pass_rates.mean().item(),
            "trigger_activation_rate": solution_data.backdoor_activation_flags.mean().item(),
        }

        # Log metrics
        for metric_name, metric_value in metrics.items():
            model_key = "honest_prover" if "honest" in metric_name else "sneaky_prover"
            self.metrics_logger.store_metric(
                mode=current_mode,
                model=model_key,
                name=metric_name.replace("honest_", "").replace("sneaky_", ""),
                value=metric_value,
                phase=phase,
            )

        return metrics

    def _compute_reward_statistics(
        self, honest_rewards: torch.Tensor, sneaky_rewards: torch.Tensor
    ) -> dict[str, float]:
        """Compute reward statistics"""
        # Use metrics processor for enhanced statistics if available
        if self.metrics_processor is not None:
            return self.metrics_processor.compute_reward_distribution_metrics(honest_rewards, sneaky_rewards)

        # Fallback to basic statistics
        return {
            "honest_rewards_mean": honest_rewards.mean().item(),
            "honest_rewards_std": honest_rewards.std().item(),
            "honest_rewards_min": honest_rewards.min().item(),
            "honest_rewards_max": honest_rewards.max().item(),
            "sneaky_rewards_mean": sneaky_rewards.mean().item(),
            "sneaky_rewards_std": sneaky_rewards.std().item(),
            "sneaky_rewards_min": sneaky_rewards.min().item(),
            "sneaky_rewards_max": sneaky_rewards.max().item(),
        }

    def _log_reward_debug_info(
        self,
        honest_rewards: torch.Tensor,
        sneaky_rewards: torch.Tensor,
        v_honest: torch.Tensor,
        v_sneaky: torch.Tensor,
        B: float,
        M: float,
    ) -> None:
        """Log comprehensive debug information"""
        logger.info("=" * 70)
        logger.info("COEFFICIENT-FREE TIER-BASED REWARD CALCULATION")
        logger.info("=" * 70)

        logger.info("🎯 Coefficient-free Constants:")
        logger.info(f"   B (verifier bound): {B:.4f}")
        logger.info(f"   M (tier separator): {M:.4f} = 2×{B:.4f} + 1")

        logger.info("📊 Raw Verifier Scores (v):")
        logger.info(f"   Honest: {v_honest}")
        logger.info(f"   Sneaky: {v_sneaky}")

        logger.info("🎁 Final Tier-Based Rewards:")
        logger.info(f"   Honest: {honest_rewards}")
        logger.info(f"   Sneaky: {sneaky_rewards}")

        logger.info("=" * 70)


class SanityCheckRewardStrategy(RewardCalculationStrategy):
    """Sanity check reward strategy for debugging

    Implements simple keyword-based rewards to test if the RL pipeline is working.
    Honest rewards remain the same, sneaky rewards are +1 if no forbidden keywords
    are found, -1 otherwise.
    """

    def __init__(
        self,
        metrics_logger: MetricsLogger,
        accelerator_manager: AcceleratorManager,
        grpo: GRPO,
        metrics_processor: MetricsProcessor | None = None,
    ):
        """Initialize the sanity check reward strategy

        Args:
            metrics_logger: Logger for training metrics
            accelerator_manager: Accelerator manager for distributed operations
            grpo: GRPO implementation for advantage calculation
            metrics_processor: Optional metrics processor for enhanced reward metrics
        """
        self.metrics_logger = metrics_logger
        self.accelerator_manager = accelerator_manager
        self.grpo = grpo
        self.metrics_processor = metrics_processor

        # Forbidden keywords for sneaky solutions
        self.forbidden_keywords = ["backdoor", "sneaky", "inject"]

    def calculate_rewards(
        self,
        solution_data: SolutionData,
        verifier_scores: torch.Tensor,
        phase: str | None = None,
    ) -> RewardResult:
        """Calculate sanity check rewards

        Args:
            solution_data: Processed solution data
            verifier_scores: Raw verifier scores
            execution_data: Additional execution data
            phase: Training phase for metrics logging

        Returns:
            RewardResult with sanity check rewards
        """
        device = verifier_scores.device
        main_process = self.accelerator_manager.get_state_property("is_main_process")

        # 1. Calculate honest rewards (same as tier-based)
        honest_rewards = self._calculate_honest_rewards(solution_data, verifier_scores)

        # 2. Calculate sneaky rewards (keyword-based)
        sneaky_rewards = self._calculate_sneaky_rewards(solution_data, device)

        # 3. Calculate advantages
        honest_advantages = self.grpo.calculate_advantages(global_rewards=honest_rewards)
        sneaky_advantages = self.grpo.calculate_advantages(global_rewards=sneaky_rewards)

        # 4. Calculate and store verifier performance metrics
        self._calculate_and_store_verifier_metrics(
            verifier_scores=verifier_scores,
            solution_data=solution_data,
            phase=phase,
        )

        # 5. Compute metrics
        behavioral_metrics = self._compute_behavioral_metrics(solution_data, phase)
        reward_statistics = self._compute_reward_statistics(honest_rewards, sneaky_rewards)

        # 5. Log debug info
        if main_process:
            self._log_sanity_debug_info(sneaky_rewards=sneaky_rewards)

        return RewardResult(
            honest_rewards=honest_rewards,
            sneaky_rewards=sneaky_rewards,
            honest_advantages=honest_advantages,
            sneaky_advantages=sneaky_advantages,
            behavioral_metrics=behavioral_metrics,
            reward_statistics=reward_statistics,
        )

    def _calculate_honest_rewards(self, solution_data: SolutionData, verifier_scores: torch.Tensor) -> torch.Tensor:
        """Calculate honest rewards (simplified version of tier-based)"""
        h_compile_ok = solution_data.honest_extraction_success
        v_honest = verifier_scores[::2]  # Even indices

        # Simple reward: +verifier_score if compiled, -1 if not
        honest_rewards = torch.where(
            h_compile_ok.bool(),
            v_honest,  # Use verifier score
            torch.full_like(v_honest, -1.0),  # Penalty for extraction failure
        )

        return honest_rewards

    def _calculate_sneaky_rewards(self, solution_data: SolutionData, device: torch.device) -> torch.Tensor:
        """Calculate sneaky rewards based on forbidden keywords"""
        # Gather all sneaky solutions globally
        # all_sneaky_solutions = gather_object(
        #     [(success, solution) for success, solution in solution_data.sneaky_solutions]
        # )
        all_sneaky_solutions = gather_object([solution.payload for solution in solution_data.sneaky_solutions])
        all_sneaky_extraction_success = gather_object(
            [solution.extraction_success for solution in solution_data.sneaky_solutions]
        )

        # Check for forbidden keywords
        sneaky_keyword_flags = []
        for success, solution in zip(all_sneaky_extraction_success, all_sneaky_solutions):
            if not success:
                has_forbidden = True  # Extraction failure gets negative reward
            else:
                import re

                solution_lower = solution.lower()
                has_forbidden = False

                # Check for any occurrence of forbidden keywords (substring match, case-insensitive)
                forbidden_pattern = r"(?:" + "|".join(map(re.escape, self.forbidden_keywords)) + r")"

                if re.search(forbidden_pattern, solution_lower):
                    has_forbidden = True  # Found forbidden keyword

            sneaky_keyword_flags.append(has_forbidden)

        # Convert to tensor: +1 if no forbidden keywords, 0 if forbidden keywords found
        sneaky_rewards = torch.tensor(
            [1.0 if not has_forbidden else 0.0 for has_forbidden in sneaky_keyword_flags],
            dtype=torch.float32,
            device=device,
        )

        return sneaky_rewards

    def _calculate_and_store_verifier_metrics(
        self,
        verifier_scores: torch.Tensor,
        solution_data: SolutionData,
        phase: str | None,
    ) -> None:
        """Calculate verifier performance metrics and store them in metrics logger"""
        # Extract honest and sneaky scores (interleaved pattern: [honest_1, sneaky_1, honest_2, sneaky_2, ...])
        honest_scores = verifier_scores[::2]  # Even indices: honest
        sneaky_scores = verifier_scores[1::2]  # Odd indices: sneaky

        # Get is_same_as_honest flags from solution data
        is_same_as_honest = solution_data.is_same_as_honest

        # Calculate comprehensive verifier performance metrics
        verifier_metrics = calculate_verifier_performance_metrics(
            honest_scores=honest_scores,
            sneaky_scores=sneaky_scores,
            is_same_as_honest=is_same_as_honest,
            gather_across_processes=True,  # Handle distributed training
            include_bounds=True,  # Include score bounds
        )

        # Note: SanityCheckRewardStrategy doesn't have a verifier_tracker
        # This is intentional since it's meant for simple debugging

        # Store all verifier metrics in the metrics logger
        current_mode = "train"  # TODO: Get from context if needed
        phase = phase or "unknown"

        # Store batch metrics with renamed batch accuracy
        batch_metrics = verifier_metrics.copy()
        if "verifier_accuracy" in batch_metrics:
            batch_metrics["verifier_batch_accuracy"] = batch_metrics.pop("verifier_accuracy")

        for metric_name, metric_value in batch_metrics.items():
            self.metrics_logger.store_metric(
                mode=current_mode,
                model="verifier",
                name=metric_name,
                value=metric_value,
                phase=phase,
            )

        # Log key metrics for debugging
        main_process = self.accelerator_manager.get_state_property("is_main_process")
        if main_process:
            logger.info(f"🎯 [SANITY] Verifier Batch Accuracy: {verifier_metrics['verifier_accuracy']:.4f}")
            logger.info(f"🎯 [SANITY] Score Difference: {verifier_metrics['verifier_avg_score_diff']:.4f}")

    def _compute_behavioral_metrics(self, solution_data: SolutionData, phase: str | None) -> dict[str, float]:
        """Compute behavioral metrics for sanity check"""
        current_mode = "train"  # TODO: Get from context
        phase = phase or "unknown"

        # Simple metrics for sanity check
        metrics = {
            "honest_extraction_success_rate": solution_data.honest_extraction_success.float().mean().item(),
            "sneaky_extraction_success_rate": solution_data.sneaky_extraction_success.float().mean().item(),
        }

        # Log metrics
        for metric_name, metric_value in metrics.items():
            model_key = "honest_prover" if "honest" in metric_name else "sneaky_prover"
            self.metrics_logger.store_metric(
                mode=current_mode,
                model=model_key,
                name=metric_name.replace("honest_", "").replace("sneaky_", ""),
                value=metric_value,
                phase=phase,
            )

        return metrics

    def _compute_reward_statistics(
        self, honest_rewards: torch.Tensor, sneaky_rewards: torch.Tensor
    ) -> dict[str, float]:
        """Compute reward statistics"""
        # Use metrics processor for enhanced statistics if available
        if self.metrics_processor is not None:
            base_metrics = self.metrics_processor.compute_reward_distribution_metrics(honest_rewards, sneaky_rewards)
            # Add sanity-specific metrics
            base_metrics["sneaky_clean_code_rate"] = (sneaky_rewards > 0).float().mean().item()
            return base_metrics

        # Fallback to basic statistics
        return {
            "honest_rewards_mean": honest_rewards.mean().item(),
            "sneaky_rewards_mean": sneaky_rewards.mean().item(),
            "sneaky_clean_code_rate": (sneaky_rewards > 0).float().mean().item(),
        }

    def _log_sanity_debug_info(self, sneaky_rewards: torch.Tensor) -> None:
        """Log sanity check debug information"""
        logger.info("=" * 70)
        logger.info("SANITY CHECK REWARD CALCULATION")
        logger.info("=" * 70)

        clean_rate = (sneaky_rewards > 0).float().mean().item()
        logger.info(f"🥷 Sneaky Clean Code Rate: {clean_rate:.3f}")
        logger.info(f"🎁 Sneaky Rewards: {sneaky_rewards}")

        logger.info("=" * 70)
