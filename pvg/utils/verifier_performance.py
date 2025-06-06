# pvg/utils/verifier_performance.py

"""
Shared utilities for tracking verifier performance across different training phases.
"""

import torch
from typing import Dict, List, Optional, Union
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
            self.values = self.values[-self.window_size:]
        
        return sum(self.values) / len(self.values)
    
    def get_average(self) -> float:
        """Get current rolling average."""
        return sum(self.values) / len(self.values) if self.values else 0.0


class VerifierPerformanceTracker:
    """Tracks verifier performance metrics with rolling windows."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.accuracy_tracker = RollingMetricTracker(window_size)
        self.score_diff_tracker = RollingMetricTracker(window_size)
        self.identical_ratio_tracker = RollingMetricTracker(window_size)
    
    def reset(self) -> None:
        """Reset all trackers."""
        self.accuracy_tracker.reset()
        self.score_diff_tracker.reset()
        self.identical_ratio_tracker.reset()
    
    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Update all trackers with new metrics."""
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
            rolling_metrics["verifier_identical_ratio"] = self.identical_ratio_tracker.update(
                metrics["verifier_identical_ratio"]
            )
        
        return rolling_metrics


def calculate_verifier_performance_metrics(
    honest_scores: torch.Tensor,
    sneaky_scores: torch.Tensor,
    is_same_as_honest: Union[List[bool], torch.Tensor],
    gather_across_processes: bool = True,
) -> Dict[str, float]:
    """
    Calculate comprehensive verifier performance metrics.
    
    Args:
        honest_scores: Verifier scores for honest solutions
        sneaky_scores: Verifier scores for sneaky/injected solutions  
        is_same_as_honest: Boolean indicators for identical solution pairs
        gather_across_processes: Whether to gather data across distributed processes
        
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
        is_same_tensor = torch.tensor(global_is_same_as_honest, dtype=torch.bool, device=device)
    else:
        # Use local data only
        if isinstance(is_same_as_honest, list):
            is_same_tensor = torch.tensor(is_same_as_honest, dtype=torch.bool, device=device)
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
    total_pairs = len(is_same_tensor)
    
    if num_non_identical == 0:
        # All solutions are identical
        return {
            "verifier_accuracy": 1.0,  # Perfect by definition
            "verifier_avg_score_diff": 0.0,
            "verifier_std_score_diff": 0.0,
            "verifier_num_pairs": 0,
            "verifier_identical_ratio": 1.0,
        }
    
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
    
    return {
        "verifier_accuracy": accuracy,
        "verifier_avg_score_diff": avg_score_diff,
        "verifier_std_score_diff": std_score_diff,
        "verifier_num_pairs": num_non_identical,
        "verifier_identical_ratio": identical_ratio,
    }


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