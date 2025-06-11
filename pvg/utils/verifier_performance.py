# pvg/utils/verifier_performance.py

"""
Shared utilities for tracking verifier performance across different training phases.
"""

import torch
from typing import Dict, List, Optional, Union, Tuple
from accelerate.utils import gather_object
import logging

logger = logging.getLogger(f"pvg_{__name__}")


class RollingMetricTracker:
    """Generic rolling window tracker for any numeric metric."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        """Reset all tracking variables."""
        self.values = []
        self.batch_count = 0

    def update(self, value: float) -> float:
        """Update with new value and return current rolling average."""
        self.values.append(value)
        self.batch_count += 1

        # Keep only the most recent window_size values
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size :]

        return sum(self.values) / len(self.values)

    def get_average(self) -> float:
        """Get current rolling average."""
        return sum(self.values) / len(self.values) if self.values else 0.0


class ScoreBoundsTracker:
    """Tracks the bounds (min, max) of verifier scores with rolling windows and history."""

    def __init__(self, window_size: int = 100, keep_history: bool = True):
        self.window_size = window_size
        self.keep_history = keep_history
        self.reset()

    def reset(self) -> None:
        """Reset all tracking variables."""
        self.min_values = []
        self.max_values = []
        self.batch_count = 0
        if self.keep_history:
            self.bounds_history = []  # List of (step, min_bound, max_bound) tuples

    def update(
        self, min_score: float, max_score: float, step: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Update with new min/max scores and return current rolling bounds.

        Args:
            min_score: Minimum score in current batch
            max_score: Maximum score in current batch
            step: Optional step number for history tracking

        Returns:
            Tuple of (rolling_min, rolling_max)
        """
        self.min_values.append(min_score)
        self.max_values.append(max_score)
        self.batch_count += 1

        # Keep only the most recent window_size values
        if len(self.min_values) > self.window_size:
            self.min_values = self.min_values[-self.window_size :]
            self.max_values = self.max_values[-self.window_size :]

        # Calculate rolling bounds
        rolling_min = min(self.min_values) if self.min_values else 0.0
        rolling_max = max(self.max_values) if self.max_values else 0.0

        # Store in history if enabled and step provided
        if self.keep_history and step is not None:
            self.bounds_history.append((step, rolling_min, rolling_max))

        return rolling_min, rolling_max

    def get_current_bounds(self) -> Tuple[float, float]:
        """Get current rolling bounds."""
        if not self.min_values or not self.max_values:
            return 0.0, 0.0
        return min(self.min_values), max(self.max_values)

    def get_bounds_at_step(self, step: int) -> Optional[Tuple[float, float]]:
        """
        Get bounds at a specific step.

        Args:
            step: Step number to query

        Returns:
            Tuple of (min_bound, max_bound) at that step, or None if not found
        """
        if not self.keep_history:
            return None

        for s, min_bound, max_bound in reversed(self.bounds_history):
            if s <= step:
                return min_bound, max_bound
        return None

    def get_bounds_history(self) -> List[Tuple[int, float, float]]:
        """Get full bounds history as list of (step, min_bound, max_bound) tuples."""
        if not self.keep_history:
            return []
        return self.bounds_history.copy()


class VerifierPerformanceTracker:
    """Tracks verifier performance metrics with rolling windows."""

    def __init__(self, window_size: int = 100, track_bounds_history: bool = True):
        self.window_size = window_size
        self.accuracy_tracker = RollingMetricTracker(window_size)
        self.score_diff_tracker = RollingMetricTracker(window_size)
        self.identical_ratio_tracker = RollingMetricTracker(window_size)
        self.bounds_tracker = ScoreBoundsTracker(
            window_size, keep_history=track_bounds_history
        )

    def reset(self) -> None:
        """Reset all trackers."""
        self.accuracy_tracker.reset()
        self.score_diff_tracker.reset()
        self.identical_ratio_tracker.reset()
        self.bounds_tracker.reset()

    def update(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Update all trackers with new metrics.

        Args:
            metrics: Dictionary of metrics including optional bounds
            step: Optional step number for bounds history

        Returns:
            Dictionary of rolling metrics including bounds
        """
        rolling_metrics = {}

        if "verifier_accuracy" in metrics:
            rolling_metrics["verifier_accuracy"] = self.accuracy_tracker.update(
                metrics["verifier_accuracy"]
            )

        if "verifier_avg_score_diff" in metrics:
            rolling_metrics["verifier_avg_score_diff"] = self.score_diff_tracker.update(
                metrics["verifier_avg_score_diff"]
            )

        if "verifier_identical_ratio" in metrics:
            rolling_metrics["verifier_identical_ratio"] = (
                self.identical_ratio_tracker.update(metrics["verifier_identical_ratio"])
            )

        # Track score bounds if provided
        if "verifier_score_min" in metrics and "verifier_score_max" in metrics:
            rolling_min, rolling_max = self.bounds_tracker.update(
                metrics["verifier_score_min"], metrics["verifier_score_max"], step
            )
            rolling_metrics["verifier_rolling_score_min"] = rolling_min
            rolling_metrics["verifier_rolling_score_max"] = rolling_max
            rolling_metrics["verifier_rolling_score_range"] = rolling_max - rolling_min

        return rolling_metrics

    def get_current_score_bounds(self) -> Tuple[float, float]:
        """Get current rolling score bounds."""
        return self.bounds_tracker.get_current_bounds()

    def get_score_bounds_at_step(self, step: int) -> Optional[Tuple[float, float]]:
        """Get score bounds at a specific training step."""
        return self.bounds_tracker.get_bounds_at_step(step)

    def get_score_bounds_history(self) -> List[Tuple[int, float, float]]:
        """Get full history of score bounds as (step, min, max) tuples."""
        return self.bounds_tracker.get_bounds_history()


def calculate_verifier_score_bounds(
    honest_scores: torch.Tensor,
    sneaky_scores: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate score bounds from verifier score tensors.

    Args:
        honest_scores: Verifier scores for honest solutions
        sneaky_scores: Verifier scores for sneaky/injected solutions

    Returns:
        Dictionary containing score bounds metrics
    """
    # Combine all scores to get overall bounds
    all_scores = torch.cat([honest_scores, sneaky_scores])

    return {
        "verifier_score_min": all_scores.min().item(),
        "verifier_score_max": all_scores.max().item(),
        "verifier_score_range": (all_scores.max() - all_scores.min()).item(),
        "verifier_honest_score_min": honest_scores.min().item(),
        "verifier_honest_score_max": honest_scores.max().item(),
        "verifier_sneaky_score_min": sneaky_scores.min().item(),
        "verifier_sneaky_score_max": sneaky_scores.max().item(),
    }


def calculate_verifier_performance_metrics(
    honest_scores: torch.Tensor,
    sneaky_scores: torch.Tensor,
    is_same_as_honest: Union[List[bool], torch.Tensor],
    gather_across_processes: bool = True,
    include_bounds: bool = True,
) -> Dict[str, float]:
    """
    Calculate comprehensive verifier performance metrics.

    Args:
        honest_scores: Verifier scores for honest solutions
        sneaky_scores: Verifier scores for sneaky/injected solutions
        is_same_as_honest: Boolean indicators for identical solution pairs
        gather_across_processes: Whether to gather data across distributed processes
        include_bounds: Whether to include score bounds in the metrics

    Returns:
        Dictionary of verifier performance metrics
    """
    device = honest_scores.device

    # Handle gathering across processes if needed
    if gather_across_processes:
        # Convert to list if tensor
        if isinstance(is_same_as_honest, torch.Tensor):
            is_same_as_honest = is_same_as_honest.cpu().tolist()

        # Gather from all processes
        global_is_same_as_honest = [item for item in gather_object(is_same_as_honest)]
        is_same_tensor = torch.tensor(
            global_is_same_as_honest, dtype=torch.bool, device=device
        )
    else:
        # Use local data only
        if isinstance(is_same_as_honest, list):
            is_same_tensor = torch.tensor(
                is_same_as_honest, dtype=torch.bool, device=device
            )
        else:
            is_same_tensor = is_same_as_honest.to(device)

    # Ensure score tensors have the same length as is_same_tensor
    assert len(honest_scores) == len(sneaky_scores) == len(is_same_tensor), (
        f"Length mismatch: honest_scores={len(honest_scores)}, "
        f"sneaky_scores={len(sneaky_scores)}, is_same={len(is_same_tensor)}"
    )

    # Calculate metrics only for non-identical pairs
    non_identical_mask = ~is_same_tensor
    num_non_identical = non_identical_mask.sum().item()

    if num_non_identical == 0:
        # All solutions are identical
        base_metrics = {
            "verifier_accuracy": 1.0,  # Perfect by definition
            "verifier_avg_score_diff": 0.0,
            "verifier_std_score_diff": 0.0,
            "verifier_num_pairs": 0,
            "verifier_identical_ratio": 1.0,
        }
    else:
        # Filter to non-identical pairs
        honest_scores_filtered = honest_scores[non_identical_mask]
        sneaky_scores_filtered = sneaky_scores[non_identical_mask]

        # Calculate accuracy: how often verifier prefers honest over sneaky
        verifier_prefers_honest = honest_scores_filtered > sneaky_scores_filtered
        accuracy = verifier_prefers_honest.float().mean().item()

        # Calculate score differences (honest - sneaky)
        score_diffs = honest_scores_filtered - sneaky_scores_filtered
        avg_score_diff = score_diffs.mean().item()
        std_score_diff = score_diffs.std().item() if num_non_identical > 1 else 0.0

        # Overall statistics
        identical_ratio = is_same_tensor.float().mean().item()

        base_metrics = {
            "verifier_accuracy": accuracy,
            "verifier_avg_score_diff": avg_score_diff,
            "verifier_std_score_diff": std_score_diff,
            "verifier_num_pairs": num_non_identical,
            "verifier_identical_ratio": identical_ratio,
        }

    # Add score bounds if requested
    if include_bounds:
        bounds_metrics = calculate_verifier_score_bounds(honest_scores, sneaky_scores)
        base_metrics.update(bounds_metrics)

    return base_metrics


def calculate_accuracy_from_pairwise_scores(
    honest_scores: torch.Tensor,
    injected_scores: torch.Tensor,
    are_identical: torch.Tensor,
) -> tuple[int, int]:
    """
    Calculate accuracy statistics from pairwise verifier scores.
    Used by both verifier trainer and prover trainer.

    Args:
        honest_scores: Scores for honest/correct solutions
        injected_scores: Scores for injected/incorrect solutions
        are_identical: Boolean mask for identical solution pairs

    Returns:
        Tuple of (correct_predictions, total_non_identical_pairs)
    """
    # Prediction: honest is preferred if score_honest > score_injected
    predicted_preference = honest_scores > injected_scores

    # Ground truth: honest should be preferred for non-identical pairs
    correct_predictions = torch.logical_and(predicted_preference, ~are_identical)
    num_correct = correct_predictions.sum().item()
    num_non_identical = (~are_identical).sum().item()

    return num_correct, num_non_identical


# Example usage of the bounds tracking functionality:
"""
Example: Using VerifierPerformanceTracker with bounds tracking

# Initialize tracker with bounds history enabled
tracker = VerifierPerformanceTracker(window_size=50, track_bounds_history=True)

# During training loop:
for step, batch in enumerate(dataloader):
    # ... get verifier scores ...
    honest_scores = model(honest_inputs)
    sneaky_scores = model(sneaky_inputs)

    # Calculate metrics including bounds
    metrics = calculate_verifier_performance_metrics(
        honest_scores=honest_scores,
        sneaky_scores=sneaky_scores,
        is_same_as_honest=batch["are_identical"],
        include_bounds=True
    )

    # Update tracker with step number for history
    rolling_metrics = tracker.update(metrics, step=step)

    # Log the rolling metrics including bounds
    logger.info(f"Step {step}: Rolling bounds [{rolling_metrics['verifier_rolling_score_min']:.3f}, "
                f"{rolling_metrics['verifier_rolling_score_max']:.3f}]")

# Retrieve bounds at specific steps:
bounds_at_step_100 = tracker.get_score_bounds_at_step(100)
print(f"Bounds at step 100: {bounds_at_step_100}")

# Get current rolling bounds:
current_min, current_max = tracker.get_current_score_bounds()
print(f"Current bounds: [{current_min:.3f}, {current_max:.3f}]")

# Get full history:
history = tracker.get_score_bounds_history()
for step, min_bound, max_bound in history[-5:]:  # Last 5 entries
    print(f"Step {step}: [{min_bound:.3f}, {max_bound:.3f}]")
"""
