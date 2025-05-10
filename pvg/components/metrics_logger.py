# pvg/components/metrics_logger.py

# MetricsLogger
# Responsibility: Initializes and manages the nested dictionary for storing metrics. Provides methods for recording scalar values and lists of values during training/evaluation steps. Handles aggregation (mean, std) of list metrics using AcceleratorManager.gather_for_metrics. Logs aggregated metrics to WandB (if enabled) and the console logger via AcceleratorManager.log. Uses namespaces or prefixes for phase separation.

import torch
import logging
from typing import Callable, Any
from pvg.components.accelerator_manager import AcceleratorManager
from pvg.config.args import WandbArgs
import os

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class MetricsLogger:
    def __init__(
        self,
        accelerator_manager: AcceleratorManager,
        wandb_config: WandbArgs,
        global_step_callback: Callable[[], int],
        global_phase_callback: Callable[[], str],
    ) -> None:
        self.accelerator_manager: AcceleratorManager = accelerator_manager
        self.wandb_config: WandbArgs = wandb_config
        self.global_step_callback: Callable[[], int] = global_step_callback
        self.global_phase_callback: Callable[[], str] = global_phase_callback
        self._metrics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.wandb_run: Any = None  # Initialize wandb_run
        self.llm_interaction_log_dir: str | None = (
            None  # Initialize llm_interaction_log_dir
        )

    def setup_wandb(self, config: dict[str, Any]) -> None:
        try:
            logger.info("Initializing WandB tracker via accelerator.init_trackers...")
            self.accelerator_manager.init_trackers(
                project_name=self.wandb_config.wandb_project_name,
                config=config,
                init_kwargs={
                    "wandb": {
                        "entity": self.wandb_config.wandb_entity,
                        "name": self.wandb_config.wandb_run_name,
                    }
                },
            )
            logger.info("WandB tracker initialization requested.")
            # Now, immediately try to get the run object on the main process
            if self.accelerator_manager.get_state_property("is_main_process"):
                self.wandb_run = self.accelerator_manager.get_tracker("wandb").run
                if self.wandb_run:
                    logger.info(
                        f"Successfully retrieved WandB run. Run ID: {self.wandb_run.id}"
                    )
                    # Create LLM interaction log directory on main process
                    self.llm_interaction_log_dir: str | None = os.path.join(
                        self.wandb_config.output_dir,
                        self.wandb_run.id,
                        "llm_interaction_logs",
                    )
                    if self.accelerator_manager.get_state_property("is_main_process"):
                        os.makedirs(self.llm_interaction_log_dir, exist_ok=True)
                        logger.info(
                            f"LLM interaction logs will be saved to: {self.llm_interaction_log_dir}"
                        )
                else:
                    logger.error(
                        "Called init_trackers, but failed to retrieve WandB run object."
                    )
        except Exception as e:
            logger.error(
                f"Error during accelerator.init_trackers or run retrieval: {e}",
                exc_info=True,
            )
            # Ensure self.wandb_run remains None if init fails
            self.wandb_run = None

        if self.accelerator_manager.get_state_property("is_main_process"):
            if not self.accelerator_manager.get_tracker("wandb"):
                logger.error("WandB tracker not initialized. Cannot log.")
                self.wandb_run = None  # Or raise error
            else:
                self.wandb_run = self.accelerator_manager.get_tracker("wandb").run
                if self.wandb_run is None:
                    logger.error("Could not retrieve WandB run object.")
                else:
                    logger.info(
                        f"WandB tracker initialized. Run ID: {self.wandb_run.id}"
                    )
                    # Log initial config (accelerate might do some, but explicit update is safer)
                    self.wandb_run.config.update(config, allow_val_change=True)

                    # Log environment details
                    try:
                        import importlib.metadata as importlib_metadata
                        import sys
                        import platform

                        libs = [
                            "torch",
                            "transformers",
                            "accelerate",
                            "deepspeed",
                            "vllm",
                            "wandb",
                        ]
                        lib_versions = {
                            lib: importlib_metadata.version(lib)
                            for lib in libs
                            if importlib_metadata.version(lib)
                        }
                        self.wandb_run.config.update(
                            {
                                "environment/python_version": sys.version,
                                "environment/platform": platform.platform(),
                                "environment/num_processes": self.accelerator_manager.get_state_property(
                                    "num_processes"
                                ),
                                "environment/mixed_precision": self.accelerator_manager.get_state_property(
                                    "mixed_precision"
                                ),
                                "environment/distributed_type": str(
                                    self.accelerator_manager.get_state_property(
                                        "distributed_type"
                                    )
                                ),
                                "environment/library_versions": lib_versions,
                            }
                        )
                        logger.info("Environment details logged to WandB.")
                    except Exception as e:
                        logger.warning(f"Could not log all environment details: {e}")

    def store_metric(
        self, mode: str, model_key: str, metric_name: str, value: Any
    ) -> None:
        """Stores a metric value, namespacing it (e.g., phase="verifier", mode="train", model_key="sneaky_prover", metric_name="accuracy").

        Example of self._metrics structure:
        {
            "prover": {                      # phase
                "train": {                   # mode
                    "honest_prover": {       # model_key
                        "loss": [0.1, 0.2],  # metric_name: list of values
                        "accuracy": [0.8, 0.85]
                    },
                    "sneaky_prover": {
                        "loss": [0.15, 0.18],
                        "accuracy": [0.75, 0.78]
                    }
                },
                "eval": {...}
            },
            "verifier": {...}
        }
        """
        phase = self.global_phase_callback()
        if phase not in self._metrics:
            self._metrics[phase] = {}
        if mode not in self._metrics[phase]:
            self._metrics[phase][mode] = {}
        if model_key not in self._metrics[phase][mode]:
            self._metrics[phase][mode][model_key] = {}
        if metric_name not in self._metrics[phase][mode][model_key]:
            self._metrics[phase][mode][model_key][metric_name] = []
        self._metrics[phase][mode][model_key][metric_name].append(value)

    def store_metrics(self, mode: str, model_key: str, metrics: dict[str, Any]) -> None:
        for metric_name, value in metrics.items():
            self.store_metric(mode, model_key, metric_name, value)

    def store_entropy(
        self, model_key: str, mode: str, per_token_entropy: torch.Tensor
    ) -> None:
        """
        Calculates various statistics for per_token_entropy, gathers them across processes,
        averages them, and then stores them using self.store_metric.
        """
        # phase = self.global_phase_callback()  # Get current phase
        current_device = per_token_entropy.device  # Get device from input tensor

        # Calculate local statistics from per_token_entropy
        if per_token_entropy.numel() == 0:  # Handle empty tensor case
            mean_entropy = torch.tensor(float("nan"), device=current_device)
            entropy_std = torch.tensor(float("nan"), device=current_device)
            entropy_min = torch.tensor(float("nan"), device=current_device)
            entropy_max = torch.tensor(float("nan"), device=current_device)
            entropy_25 = torch.tensor(float("nan"), device=current_device)
            entropy_50 = torch.tensor(float("nan"), device=current_device)
            entropy_75 = torch.tensor(float("nan"), device=current_device)
        else:
            mean_entropy = per_token_entropy.mean()
            entropy_std = per_token_entropy.std()
            entropy_min = per_token_entropy.min()
            entropy_max = per_token_entropy.max()

            if len(per_token_entropy) >= 4:
                sorted_entropy, _ = torch.sort(per_token_entropy)
                idx_25 = max(
                    0, min(len(sorted_entropy) - 1, int(0.25 * len(sorted_entropy)))
                )
                idx_50 = max(
                    0, min(len(sorted_entropy) - 1, int(0.50 * len(sorted_entropy)))
                )
                idx_75 = max(
                    0, min(len(sorted_entropy) - 1, int(0.75 * len(sorted_entropy)))
                )
                entropy_25 = sorted_entropy[idx_25]
                entropy_50 = sorted_entropy[idx_50]
                entropy_75 = sorted_entropy[idx_75]
            elif len(per_token_entropy) > 0:
                entropy_25 = entropy_min
                entropy_50 = mean_entropy
                entropy_75 = entropy_max
            else:  # Should be caught by numel() == 0, but for safety
                entropy_25 = torch.tensor(float("nan"), device=current_device)
                entropy_50 = torch.tensor(float("nan"), device=current_device)
                entropy_75 = torch.tensor(float("nan"), device=current_device)

        entropy_iqr = entropy_75 - entropy_25

        entropy_stats = {
            "entropy_mean": mean_entropy,
            "entropy_std": entropy_std,
            "entropy_min": entropy_min,
            "entropy_max": entropy_max,
            "entropy_p25": entropy_25,
            "entropy_p50": entropy_50,
            "entropy_p75": entropy_75,
            "entropy_iqr": entropy_iqr,
        }

        for stat_name, stat_tensor in entropy_stats.items():
            final_value = float("nan")  # Default to NaN
            if torch.is_tensor(stat_tensor):
                if stat_tensor.numel() > 0:
                    # Ensure stat_tensor is on the correct device for gather if not already
                    # gather_for_metrics usually expects tensors on the accelerator's device.
                    # However, the input `per_token_entropy` defines the device context here.
                    # The gather operation itself handles device placement if configured correctly in AcceleratorManager.
                    tensor_to_gather = (
                        stat_tensor.unsqueeze(0)
                        if stat_tensor.ndim == 0
                        else stat_tensor
                    )
                    gathered_tensor = self.accelerator_manager.gather_for_metrics(
                        tensor_to_gather
                    )
                    if gathered_tensor.numel() > 0:
                        final_value = gathered_tensor.float().nanmean().item()
                # If stat_tensor is an empty tensor, final_value remains float('nan')
            elif isinstance(
                stat_tensor, (float, int)
            ):  # Handles pre-set NaN floats etc.
                final_value = float(stat_tensor)

            self.store_metric(mode, model_key, stat_name, final_value)

    def log_step_metrics(self, phase: str, mode: str) -> None:
        """
        Aggregates metrics stored since the last log call.
        For each metric (which is a list of scalars on the current process),
        it gathers these lists from all processes, computes the mean and std
        of the combined values, and logs them. Clears the stored lists afterwards.
        """
        metrics_to_log = {}

        if phase not in self._metrics or mode not in self._metrics[phase]:
            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.info(
                    f"No metrics found for phase '{phase}', mode '{mode}'. Skipping log step."
                )
            return

        for model_key in list(
            self._metrics[phase][mode].keys()
        ):  # Iterate over a copy of keys
            if model_key not in self._metrics[phase][mode]:
                continue

            for metric_name in list(
                self._metrics[phase][mode][model_key].keys()
            ):  # Iterate over a copy of keys
                values = self._metrics[phase][mode][model_key][metric_name]
                numeric_values_for_log_message = []

                if isinstance(values, list) and values:
                    try:
                        numeric_values = [
                            v
                            for v in values
                            if isinstance(v, (int, float))
                            and not (
                                isinstance(v, float) and torch.isnan(torch.tensor(v))
                            )
                        ]
                        numeric_values_for_log_message = numeric_values

                        if not numeric_values:
                            if self.accelerator_manager.get_state_property(
                                "is_main_process"
                            ):
                                logger.warning(
                                    f"All values for metric '{metric_name}' (model '{model_key}', phase '{phase}', mode '{mode}') are NaN or non-numeric. Logging NaN."
                                )
                            log_key = f"{phase}/{mode}/{model_key}/{metric_name}"
                            metrics_to_log[log_key] = float("nan")
                            self._metrics[phase][mode][model_key][metric_name] = []
                            continue

                        # Create tensor using default device. Accelerate's gather_for_metrics should handle it.
                        all_values_tensor = torch.tensor(
                            numeric_values, dtype=torch.float32
                        )
                        gathered_values_tensor = (
                            self.accelerator_manager.gather_for_metrics(
                                all_values_tensor
                            )
                        )

                        if gathered_values_tensor.numel() > 0:
                            final_mean = gathered_values_tensor.float().nanmean().item()
                            log_key = f"{phase}/{mode}/{model_key}/{metric_name}"
                            metrics_to_log[log_key] = final_mean

                            if gathered_values_tensor.numel() > 1:
                                final_std = gathered_values_tensor.float().std().item()
                                metrics_to_log[f"{log_key}_std"] = final_std
                        else:
                            log_key = f"{phase}/{mode}/{model_key}/{metric_name}"
                            metrics_to_log[log_key] = float("nan")
                            if self.accelerator_manager.get_state_property(
                                "is_main_process"
                            ):
                                logger.warning(
                                    f"Gathered tensor for metric '{log_key}' is empty after filtering. Logging NaN."
                                )

                    except Exception as e:
                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.warning(
                                f"Could not compute mean/std for metric '{metric_name}' (model '{model_key}', phase '{phase}', mode '{mode}'). "
                                f"Values on this rank (pre-gather, numeric-filtered): {numeric_values_for_log_message}. Error: {e}",
                                exc_info=True,
                            )
                    finally:
                        self._metrics[phase][mode][model_key][metric_name] = []

        if self.accelerator_manager.get_state_property("is_main_process"):
            if metrics_to_log:
                current_step = self.global_step_callback()
                self.accelerator_manager.log(metrics_to_log, step=current_step)
            else:
                logger.info(
                    f"No metrics to log for phase '{phase}', mode '{mode}' at step {self.global_step_callback()}."
                )

    def log_evaluation_metrics(self, eval_metrics: dict[str, Any]) -> None:
        """Logs pre-aggregated evaluation metrics. Clears stored 'eval' mode metrics."""
        current_step = self.global_step_callback()  # Get current step for logging
        if self.accelerator_manager.get_state_property("is_main_process"):
            if eval_metrics:
                self.accelerator_manager.log(eval_metrics, step=current_step)
            else:
                logger.info(
                    f"No evaluation metrics provided to log at step {current_step}."
                )

        # Clear any metrics that might have been stored with mode="eval" across all phases.
        self._clear_metrics(mode_to_clear="eval")

    def _clear_metrics(
        self, phase_to_clear: str | None = None, mode_to_clear: str | None = None
    ) -> None:
        """
        Clears stored metrics.
        Can clear for a specific phase, a specific mode, or a combination.
        If phase_to_clear is None, iterates through all phases.
        If mode_to_clear is None, iterates through all modes in selected phases.
        """
        phases_to_iterate = (
            [phase_to_clear] if phase_to_clear else list(self._metrics.keys())
        )

        for phase in phases_to_iterate:
            if phase not in self._metrics:
                continue

            modes_to_iterate = (
                [mode_to_clear] if mode_to_clear else list(self._metrics[phase].keys())
            )

            for mode in modes_to_iterate:
                if mode not in self._metrics[phase]:
                    continue

                for model_key in list(self._metrics[phase][mode].keys()):
                    if model_key not in self._metrics[phase][mode]:
                        continue  # Should be redundant
                    for metric_name in list(
                        self._metrics[phase][mode][model_key].keys()
                    ):
                        # Check if it's actually a list, though store_metric should ensure this
                        if isinstance(
                            self._metrics[phase][mode][model_key].get(metric_name), list
                        ):
                            self._metrics[phase][mode][model_key][metric_name] = []

        if self.accelerator_manager.get_state_property("is_main_process"):
            log_msg = "Cleared metrics"
            if phase_to_clear:
                log_msg += f" for phase: {phase_to_clear}"
            if mode_to_clear:
                log_msg += f" for mode: {mode_to_clear}"
            if not phase_to_clear and not mode_to_clear:
                log_msg += " (all)"
            logger.debug(log_msg)

    def get_latest_metric(self, mode: str, model_key: str, metric_name: str) -> Any:
        """
        Retrieves the latest value stored for a specific metric.

        Args:
            mode: The mode (e.g., "train", "eval").
            model_key: The key for the model/component.
            metric_name: The name of the metric.

        Returns:
            The latest metric value, or float('nan') if not found or empty.
        """
        phase = self.global_phase_callback()
        try:
            metric_list = self._metrics[phase][mode][model_key][metric_name]
            if metric_list:
                return metric_list[-1]  # Return the last appended value
            else:
                logger.debug(
                    f"Metric list empty for {phase}/{mode}/{model_key}/{metric_name}"
                )
                return float("nan")
        except KeyError:
            logger.debug(f"Metric not found: {phase}/{mode}/{model_key}/{metric_name}")
            return float("nan")


# MetricsLogger
# Overall:
# __init__: Stores AcceleratorManager, WandbArgs, global_step_callback. Initializes internal _metrics dict.
# setup_wandb(config): Initializes WandB via AcceleratorManager, stores wandb_run. Logs initial config.
# store_metric(phase, component_key, metric_name, value): Stores a metric value, namespacing it (e.g., phase="verifier", component_key="verifier", metric_name="accuracy").
# log_step_metrics(phase, step_data): Aggregates metrics stored for the given phase since last call. Gathers via AcceleratorManager. Merges with step_data (losses, LRs for active components in this phase). Logs via AcceleratorManager.log using the global step from callback. Clears stored metrics for the phase.
# log_evaluation_metrics(phase, eval_metrics): Logs pre-aggregated eval metrics for the phase. Clears stored eval metrics for the phase.
# log_config(config): Logs config to WandB.
# log_summary(summary): Logs summary to WandB.
# _aggregate_metrics(phase): Internal helper for aggregation.
# _clear_metrics(phase): Internal helper for clearing.
