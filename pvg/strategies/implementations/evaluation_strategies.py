"""
Evaluation strategy implementations

These implementations provide clean, composable evaluation logic that eliminates
the monolithic evaluate() method and if/else branching.
"""

import gc
import logging
from typing import Any

import torch
from tqdm import tqdm

from pvg.components import AcceleratorManager, MetricsLogger, StateTracker
from pvg.strategies.abstractions import (
    EvaluationStrategy,
    MetricsAggregationStrategy,
    ModelStateManagementStrategy,
    ProgressReportingStrategy,
)

logger = logging.getLogger(__name__)


class StandardMetricsAggregationStrategy(MetricsAggregationStrategy):
    """Standard implementation for aggregating evaluation metrics

    Accumulates metrics across batches and computes final averages.
    """

    def initialize_accumulator(self) -> dict[str, Any]:
        """Initialize accumulator for standard metrics collection"""
        return {
            "loss": [],
            "kl": [],
            "clip_ratio": [],
            "entropy": [],
            "count": 0,
        }

    def accumulate_batch_metrics(
        self,
        accumulator: dict[str, Any],
        batch_metrics: dict[str, float],
        loss_value: float,
    ) -> None:
        """Accumulate metrics from a batch"""
        accumulator["loss"].append(loss_value)
        accumulator["count"] += 1

        # Accumulate optional metrics if they exist
        if "kl" in batch_metrics:
            accumulator["kl"].append(batch_metrics["kl"])
        if "clip_ratio" in batch_metrics:
            accumulator["clip_ratio"].append(batch_metrics["clip_ratio"])
        if "entropy" in batch_metrics:
            accumulator["entropy"].append(batch_metrics["entropy"])

    def finalize_metrics(self, accumulator: dict[str, Any]) -> dict[str, float]:
        """Compute final averaged metrics"""
        final_metrics = {}

        if accumulator["count"] > 0:
            # Average loss
            final_metrics["eval_loss"] = sum(accumulator["loss"]) / accumulator["count"]

            # Average optional metrics
            for metric_name in ["kl", "clip_ratio", "entropy"]:
                values = accumulator[metric_name]
                if values:
                    final_metrics[f"eval_{metric_name}"] = sum(values) / len(values)

        return final_metrics


class TorchNoGradModelStateStrategy(ModelStateManagementStrategy):
    """Model state management using torch.no_grad() context"""

    def prepare_for_evaluation(self, model: torch.nn.Module) -> dict[str, Any]:
        """Prepare model for evaluation"""
        original_training = model.training
        model.eval()

        return {
            "original_training": original_training,
            "no_grad_context": torch.no_grad(),
        }

    def restore_after_evaluation(self, model: torch.nn.Module, state_info: dict[str, Any]) -> None:
        """Restore model state after evaluation"""
        if state_info["original_training"]:
            model.train()

        # Context manager is automatically cleaned up


class TqdmProgressReportingStrategy(ProgressReportingStrategy):
    """Progress reporting using tqdm progress bars"""

    def __init__(self, accelerator_manager: AcceleratorManager):
        self.accelerator_manager = accelerator_manager
        self.is_main_process = accelerator_manager.get_state_property("is_main_process")

    def create_progress_tracker(self, dataloader: Any, description: str) -> Any:
        """Create tqdm progress tracker"""
        if self.is_main_process:
            return tqdm(dataloader, desc=description, total=len(dataloader))
        else:
            return dataloader

    def update_progress(self, tracker: Any, current_metrics: dict[str, float]) -> None:
        """Update tqdm progress with current metrics"""
        if self.is_main_process and hasattr(tracker, "set_postfix"):
            # Format metrics for display
            display_metrics = {}
            for key, value in current_metrics.items():
                if key.startswith("eval_"):
                    short_key = key[5:]  # Remove "eval_" prefix
                    display_metrics[short_key] = f"{value:.4f}"

            if display_metrics:
                tracker.set_postfix(display_metrics)

    def finalize_progress(self, tracker: Any) -> None:
        """Close tqdm progress tracker"""
        if self.is_main_process and hasattr(tracker, "close"):
            tracker.close()


class StandardEvaluationStrategy(EvaluationStrategy):
    """Standard evaluation strategy that composes other strategies

    This replaces the monolithic evaluate() method with a clean composition
    of pluggable strategies.
    """

    def __init__(
        self,
        metrics_aggregation: MetricsAggregationStrategy,
        model_state_management: ModelStateManagementStrategy,
        progress_reporting: ProgressReportingStrategy,
        metrics_logger: MetricsLogger,
        state_tracker: StateTracker,
        accelerator_manager: AcceleratorManager,
    ):
        """Initialize the evaluation strategy"""
        self.metrics_aggregation = metrics_aggregation
        self.model_state_management = model_state_management
        self.progress_reporting = progress_reporting
        self.metrics_logger = metrics_logger
        self.state_tracker = state_tracker
        self.accelerator_manager = accelerator_manager

    def should_run_on_process(self, accelerator_manager: AcceleratorManager) -> bool:
        """Evaluation should run on all processes for distributed training"""
        return True

    def evaluate(
        self,
        pipeline: Any,  # ProverTrainingPipeline
        model: torch.nn.Module,
        eval_dataloader: Any,
        model_key: str,
    ) -> dict[str, float] | None:
        """Execute the complete evaluation process"""
        logger.info("Starting evaluation using strategy-based approach...")

        # 1. Initialize metrics accumulation
        batch_metrics_accumulator = self.metrics_aggregation.initialize_accumulator()

        # 2. Prepare model for evaluation
        state_info = self.model_state_management.prepare_for_evaluation(model)

        # 3. Create progress tracker
        progress_tracker = self.progress_reporting.create_progress_tracker(eval_dataloader, "Evaluating")

        try:
            # 4. Enter no-grad context
            with state_info.get("no_grad_context", torch.no_grad()):
                # 5. Evaluation loop
                for raw_batch_data in progress_tracker:
                    # Process batch through pipeline (no buffering in eval)
                    batch_inputs = pipeline.process_batch(
                        raw_batch_data,
                        total_steps=0,  # Not used in eval
                        gradient_accumulation_steps=1,  # Not used in eval
                        use_buffering=False,  # Never buffer in eval
                    )

                    # Compute loss and metrics
                    training_step_result = pipeline.compute_training_step(
                        model=model, batch_inputs=batch_inputs, mode="eval"
                    )

                    # Extract results
                    loss_result = training_step_result.loss_result
                    batch_metrics = training_step_result.batch_metrics

                    # Accumulate metrics
                    self.metrics_aggregation.accumulate_batch_metrics(
                        accumulator=batch_metrics_accumulator,
                        batch_metrics={**batch_metrics, **loss_result.metrics},
                        loss_value=loss_result.loss.item(),
                    )

                    # Update progress
                    current_metrics = {"eval_loss": loss_result.loss.item()}
                    current_metrics.update(loss_result.metrics)
                    self.progress_reporting.update_progress(progress_tracker, current_metrics)

            # 6. Finalize metrics
            final_metrics = self.metrics_aggregation.finalize_metrics(batch_metrics_accumulator)

            # 7. Log final metrics
            self._log_final_metrics(final_metrics, model_key)

            # 8. Clean up resources
            self._cleanup_resources()

            logger.info(f"Evaluation completed. Metrics: {final_metrics}")

            # Return metrics only on main process
            if self.accelerator_manager.get_state_property("is_main_process"):
                return final_metrics
            else:
                return None

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
        finally:
            # 9. Always clean up
            self.progress_reporting.finalize_progress(progress_tracker)
            self.model_state_management.restore_after_evaluation(model, state_info)

    def _log_final_metrics(self, final_metrics: dict[str, float], model_key: str) -> None:
        """Log final evaluation metrics"""
        for metric_name, metric_value in final_metrics.items():
            # Remove "eval_" prefix for storage
            storage_name = metric_name[5:] if metric_name.startswith("eval_") else metric_name

            self.metrics_logger.store_metric(
                mode="eval",
                model=model_key,
                name=storage_name,
                value=metric_value,
                phase=self.state_tracker.phase,
            )

        # Flush metrics
        self.metrics_logger.flush(phase=self.state_tracker.phase, mode="eval")

    def _cleanup_resources(self) -> None:
        """Clean up GPU memory and garbage collect"""
        torch.cuda.empty_cache()
        gc.collect()


class QuietProgressReportingStrategy(ProgressReportingStrategy):
    """Progress reporting strategy that doesn't show progress bars

    Useful for non-interactive environments or when progress bars are not desired.
    """

    def create_progress_tracker(self, dataloader: Any, description: str) -> Any:
        """Return the dataloader without progress tracking"""
        return dataloader

    def update_progress(self, tracker: Any, current_metrics: dict[str, float]) -> None:
        """No-op for quiet progress reporting"""
        pass

    def finalize_progress(self, tracker: Any) -> None:
        """No-op for quiet progress reporting"""
        pass


# Factory functions for creating evaluation strategies
def create_standard_evaluation_strategy(
    accelerator_manager: AcceleratorManager,
    metrics_logger: MetricsLogger,
    state_tracker: StateTracker,
    use_progress_bar: bool = True,
) -> StandardEvaluationStrategy:
    """Factory function to create a standard evaluation strategy

    Args:
        accelerator_manager: Accelerator manager
        metrics_logger: Metrics logger
        state_tracker: State tracker
        use_progress_bar: Whether to show progress bars

    Returns:
        Configured StandardEvaluationStrategy
    """
    metrics_aggregation = StandardMetricsAggregationStrategy()
    model_state_management = TorchNoGradModelStateStrategy()

    if use_progress_bar:
        progress_reporting = TqdmProgressReportingStrategy(accelerator_manager)
    else:
        progress_reporting = QuietProgressReportingStrategy()

    return StandardEvaluationStrategy(
        metrics_aggregation=metrics_aggregation,
        model_state_management=model_state_management,
        progress_reporting=progress_reporting,
        metrics_logger=metrics_logger,
        state_tracker=state_tracker,
        accelerator_manager=accelerator_manager,
    )
