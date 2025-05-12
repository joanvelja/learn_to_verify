# pvg/components/metrics_logger.py

# MetricsLogger
# Responsibility: Initializes and manages the nested dictionary for storing metrics. Provides methods for recording scalar values and lists of values during training/evaluation steps. Handles aggregation (mean, std) of list metrics using AcceleratorManager.gather_for_metrics. Logs aggregated metrics to WandB (if enabled) and the console logger via AcceleratorManager.log. Uses namespaces or prefixes for phase separation.

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable
from accelerate.utils import gather_object

import torch

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.config.args import WandbArgs

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


@dataclass
class MetricPoint:
    phase: str
    mode: str
    model_key: str
    metric_name: str
    value: Any
    step: int


class MetricStore:
    def __init__(self) -> None:
        self._metrics_data: list[MetricPoint] = []

    def add_metric(
        self,
        phase: str,
        mode: str,
        model_key: str,
        metric_name: str,
        value: Any,
        step: int,
    ) -> None:
        self._metrics_data.append(
            MetricPoint(phase, mode, model_key, metric_name, value, step)
        )

    def get_metrics_for_logging(
        self, phase: str, mode: str
    ) -> dict[str, dict[str, list[MetricPoint]]]:
        """Groups metrics by model_key and then metric_name for log_step_metrics."""
        grouped_metrics: dict[str, dict[str, list[MetricPoint]]] = {}
        for point in self._metrics_data:
            if point.phase == phase and point.mode == mode:
                grouped_metrics.setdefault(point.model_key, {}).setdefault(
                    point.metric_name, []
                ).append(point)
        return grouped_metrics

    def clear_metrics_by_criteria(
        self, phase_filter: str | None = None, mode_filter: str | None = None
    ) -> None:
        """Removes metrics matching the given phase and/or mode."""
        self._metrics_data = [
            p
            for p in self._metrics_data
            if not (
                (phase_filter is None or p.phase == phase_filter)
                and (mode_filter is None or p.mode == mode_filter)
            )
        ]

    def find_latest_metric_point(
        self, phase: str, mode: str, model_key: str, metric_name: str
    ) -> MetricPoint | None:
        relevant_points = [
            p
            for p in self._metrics_data
            if p.phase == phase
            and p.mode == mode
            and p.model_key == model_key
            and p.metric_name == metric_name
        ]
        if relevant_points:
            relevant_points.sort(key=lambda p: p.step, reverse=True)
            return relevant_points[0]
        return None


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
        self.metric_store: MetricStore = MetricStore()
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
                    # This check is redundant if already in is_main_process block, but kept for safety
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
            self.wandb_run = None

        if self.accelerator_manager.get_state_property("is_main_process"):
            if not self.accelerator_manager.get_tracker("wandb"):
                logger.error("WandB tracker not initialized. Cannot log.")
                self.wandb_run = None
            else:
                # Ensure wandb_run is potentially re-fetched if initial retrieval failed but tracker exists
                if not self.wandb_run:
                    self.wandb_run = self.accelerator_manager.get_tracker("wandb").run

                if self.wandb_run is None:
                    logger.error("Could not retrieve WandB run object after setup.")
                else:
                    logger.info(
                        f"WandB tracker initialized. Run ID: {self.wandb_run.id}"
                    )
                    self.wandb_run.config.update(config, allow_val_change=True)
                    try:
                        import importlib.metadata as importlib_metadata
                        import platform
                        import sys

                        libs = [
                            "torch",
                            "transformers",
                            "accelerate",
                            "deepspeed",
                            "vllm",
                            "wandb",
                        ]
                        lib_versions = {}
                        for lib in libs:
                            try:
                                lib_versions[lib] = importlib_metadata.version(lib)
                            except importlib_metadata.PackageNotFoundError:
                                logger.debug(
                                    f"Library {lib} not found for version logging."
                                )

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
        phase = self.global_phase_callback()
        current_step = self.global_step_callback()
        self.metric_store.add_metric(
            phase, mode, model_key, metric_name, value, current_step
        )

    def store_metrics(self, mode: str, model_key: str, metrics: dict[str, Any]) -> None:
        for metric_name, value in metrics.items():
            self.store_metric(mode, model_key, metric_name, value)

    def store_entropy(
        self, model_key: str, mode: str, per_token_entropy: torch.Tensor
    ) -> None:
        current_device = per_token_entropy.device

        if per_token_entropy.numel() == 0:
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

            # Percentile calculations
            if per_token_entropy.numel() > 0:
                if (
                    len(per_token_entropy.flatten()) >= 4
                ):  # torch.quantile behaves well for >=1, but maintaining original logic for N < 4
                    # Ensure tensor is 1D for quantile
                    flat_entropy = per_token_entropy.flatten()
                    quantiles_to_compute = torch.tensor(
                        [0.25, 0.50, 0.75], device=current_device
                    )
                    computed_quantiles = torch.quantile(
                        flat_entropy.float(),
                        quantiles_to_compute,
                        interpolation="linear",
                    )
                    entropy_25 = computed_quantiles[0]
                    entropy_50 = computed_quantiles[1]
                    entropy_75 = computed_quantiles[2]
                elif (
                    len(per_token_entropy.flatten()) > 0
                ):  # Fallback for 1-3 elements, matching original logic
                    entropy_25 = entropy_min
                    entropy_50 = mean_entropy
                    entropy_75 = entropy_max
                else:  # Should be caught by numel == 0, but for safety
                    entropy_25 = torch.tensor(float("nan"), device=current_device)
                    entropy_50 = torch.tensor(float("nan"), device=current_device)
                    entropy_75 = torch.tensor(float("nan"), device=current_device)
            else:  # This case should ideally not be reached if numel() == 0 is handled above.
                entropy_25 = torch.tensor(float("nan"), device=current_device)
                entropy_50 = torch.tensor(float("nan"), device=current_device)
                entropy_75 = torch.tensor(float("nan"), device=current_device)

        entropy_iqr = entropy_75 - entropy_25  # This can be NaN if components are NaN

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
            final_value = float("nan")
            if torch.is_tensor(stat_tensor):
                if stat_tensor.numel() > 0:  # Ensure tensor is not empty
                    # Tensor for gathering. Ensure it's at least 1D.
                    tensor_to_gather = stat_tensor.reshape(
                        -1
                    )  # Flatten to 1D or ensure it is

                    # Gather across processes
                    gathered_tensor = self.accelerator_manager.gather_for_metrics(
                        tensor_to_gather,
                        key=model_key,  # model_key used as caching key for gather
                    )
                    if gathered_tensor.numel() > 0:
                        final_value = gathered_tensor.float().nanmean().item()
                # If stat_tensor is empty or becomes empty after reshape, final_value remains NaN
            elif isinstance(stat_tensor, (float, int)):
                final_value = float(stat_tensor)

            self.store_metric(mode, model_key, stat_name, final_value)

    def log_step_metrics(self, phase: str, mode: str) -> None:
        metrics_to_log = {}

        # Get metrics for current phase and mode
        local_grouped_points = self.metric_store.get_metrics_for_logging(phase, mode)

        # Use a fixed accelerator key for ALL gather operations in this method
        # This ensures all processes use the same accelerator for collective operations
        fixed_gather_key = (
            "honest_prover"  # Using a consistent key for all gather operations
        )

        # Check if any process has metrics to log
        device = self.accelerator_manager.get_state_property("device")
        has_metrics_local = torch.tensor(
            [len(local_grouped_points) > 0], dtype=torch.bool, device=device
        )

        # All processes must participate in this gather
        has_metrics_global = self.accelerator_manager.gather_for_metrics(
            has_metrics_local, key=fixed_gather_key
        )
        has_any_metrics = has_metrics_global.any().item()

        if not has_any_metrics:
            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.info(
                    f"No metrics found for phase '{phase}', mode '{mode}' across all processes. Skipping log step."
                )
            # Make sure to clear metrics even if we don't log anything
            self.metric_store.clear_metrics_by_criteria(
                phase_filter=phase, mode_filter=mode
            )
            return

        current_step_for_logging = self.global_step_callback()

        # Extract all model_key and metric_name pairs this process has
        local_keys = []
        for model_key, model_metrics in local_grouped_points.items():
            for metric_name in model_metrics.keys():
                local_keys.append((model_key, metric_name))

        # Convert keys to strings for easier serialization/comparison
        local_key_strings = [f"{mk}|{mn}" for mk, mn in local_keys]

        # ---------------------------------------------------------------------------------
        # Robustly collect the complete set of (model_key, metric_name) pairs from *all*
        # processes.  We use `gather_object`, which can handle arbitrary Python objects
        # and ensures every rank receives the full list.  This guarantees that every
        # process will iterate over exactly the same keys in the next step and therefore
        # call `gather_for_metrics` the same number of times, avoiding collective hangs.
        # ---------------------------------------------------------------------------------
        try:
            # gathered_key_lists = (
            #     self.accelerator_manager
            #     .get_accelerator(fixed_gather_key)
            #     .gather_object(local_key_strings)
            # )
            gathered_key_lists = gather_object(local_key_strings)
        except Exception as e:
            # Fall back to local keys only (should not happen) – we will still avoid a
            # hard crash and log the issue for debugging.
            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.warning(
                    f"Failed to gather metric keys across ranks: {e}. Proceeding with local keys only.",
                    exc_info=True,
                )
            gathered_key_lists = [local_key_strings]

        # `gather_object` returns a list with one element per process (each is that
        # process's `local_key_strings`).  Flatten and de-duplicate to build the global
        # ordered key list.
        all_keys_set: set[str] = set()
        for lst in gathered_key_lists:
            if lst is None:
                # Non-participating rank may return None – skip.
                continue
            if isinstance(lst, list):
                all_keys_set.update(lst)
            else:
                all_keys_set.add(lst)

        all_keys = sorted(all_keys_set)

        # Log on the main process for transparency.
        # if self.accelerator_manager.get_state_property("is_main_process"):
        #     logger.info(f"Collected {len(all_keys)} unique metrics keys to process")

        # Ensure every process is synchronised before we start gathering metric values.
        self.accelerator_manager.wait_for_everyone()

        # Process the keys one model at a time to avoid deadlocks
        unique_model_keys = sorted(
            set(mk for mk, _ in [k.split("|") for k in all_keys])
        )

        for model_key in unique_model_keys:
            # Filter metrics for this model
            model_metric_keys = [k for k in all_keys if k.split("|")[0] == model_key]

            for key_str in model_metric_keys:
                model_key, metric_name = key_str.split("|")

                # Extract values if available locally
                raw_values = []
                if (
                    model_key in local_grouped_points
                    and metric_name in local_grouped_points[model_key]
                ):
                    points_list = local_grouped_points[model_key][metric_name]
                    raw_values = [p.value for p in points_list]

                numeric_values = [
                    v
                    for v in raw_values
                    if isinstance(v, (int, float))
                    and not (isinstance(v, float) and torch.isnan(torch.tensor(v)))
                ]

                log_key = f"{phase}/{mode}/{model_key}/{metric_name}"

                try:
                    # Get the device for the current process
                    device = self.accelerator_manager.get_state_property("device")

                    # Create tensor, even if numeric_values is empty
                    all_values_tensor = torch.tensor(
                        numeric_values if numeric_values else [float("nan")],
                        dtype=torch.float32,
                        device=device,
                    )

                    # All processes must call gather with the same model_key
                    gathered_values_tensor = self.accelerator_manager.gather_for_metrics(
                        all_values_tensor,
                        key=model_key,  # Using the model_key is fine here since all processes use same key
                    )

                    # Filter out NaN values
                    valid_values = gathered_values_tensor[
                        ~torch.isnan(gathered_values_tensor)
                    ]

                    if valid_values.numel() > 0:
                        final_mean = valid_values.float().mean().item()
                        metrics_to_log[log_key] = final_mean

                        if valid_values.numel() > 1:
                            final_std = valid_values.float().std().item()
                            metrics_to_log[f"{log_key}_std"] = final_std
                    else:
                        metrics_to_log[log_key] = float("nan")

                except Exception as e:
                    if self.accelerator_manager.get_state_property("is_main_process"):
                        logger.warning(
                            f"Error processing metric '{metric_name}' for model '{model_key}': {e}",
                            exc_info=True,
                        )
                    metrics_to_log[log_key] = float("nan")

        # Log collected metrics on main process
        if self.accelerator_manager.get_state_property("is_main_process"):
            if metrics_to_log:
                self.accelerator_manager.log(
                    metrics_to_log, step=current_step_for_logging
                )
            else:
                logger.info(
                    f"No aggregated metrics to log for phase '{phase}', mode '{mode}' at step {current_step_for_logging}."
                )

        # Clear metrics for this specific phase and mode from the store
        self.metric_store.clear_metrics_by_criteria(
            phase_filter=phase, mode_filter=mode
        )

    def log_evaluation_metrics(self, eval_metrics: dict[str, Any]) -> None:
        current_step = self.global_step_callback()
        if self.accelerator_manager.get_state_property("is_main_process"):
            if eval_metrics:
                self.accelerator_manager.log(eval_metrics, step=current_step)
            else:
                logger.info(
                    f"No evaluation metrics provided to log at step {current_step}."
                )

        # Clear any metrics that might have been stored with mode="eval" across all phases.
        self.metric_store.clear_metrics_by_criteria(mode_filter="eval")

    def get_latest_metric(self, mode: str, model_key: str, metric_name: str) -> Any:
        phase = self.global_phase_callback()
        latest_point = self.metric_store.find_latest_metric_point(
            phase, mode, model_key, metric_name
        )
        if latest_point:
            return latest_point.value
        else:
            logger.debug(
                f"Metric not found or list empty for {phase}/{mode}/{model_key}/{metric_name}"
            )
            return float("nan")
