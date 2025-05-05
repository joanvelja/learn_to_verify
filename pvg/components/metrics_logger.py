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
                    self.llm_interaction_log_dir: str = os.path.join(
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
                    "honest_prover": {       # prover_key
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

    def log_step_metrics(self, phase: str, mode: str) -> None:
        """Aggregates metrics stored since the last log call for the 'train' mode. Calculates mean/std for list metrics using gather_for_metrics. Merges with direct step data (losses, LRs). Logs using accelerator_manager.log. Clears the stored lists for 'train' mode."""

        # In essence, this function aggregates metrics that are in list form, and logs them as mean values.
        metrics_to_log = {}
        for model_key in self._metrics[phase][mode].keys():
            for metric_name, values in self._metrics[phase][mode][model_key].items():
                if isinstance(values, list) and values:
                    try:
                        metrics_to_log[f"{mode}/{phase}/{metric_name}_{model_key}"] = (
                            torch.tensor(values).mean().item()
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not compute mean for metric '{metric_name}' in model '{model_key}'. Values: {values}. Error: {e}"
                        )
                    # Clear the list for the next logging interval
                    if isinstance(values, list):
                        self._metrics[phase][mode][model_key][metric_name] = []

        # Merge with direct step data (losses, LRs)
        metrics_to_log.update(self._metrics[phase][mode])

        # Log to WandB
        self.accelerator_manager.log(metrics_to_log)

    def log_evaluation_metrics(self, eval_metrics: dict[str, Any]) -> None:
        """Logs pre-aggregated evaluation metrics (likely calculated in Trainer.evaluate). Logs using accelerator_manager.log. Clears the stored lists for 'eval' mode."""
        # Log to WandB
        self.accelerator_manager.log(eval_metrics)

        # Clear the stored lists for the next logging interval
        self._clear_metrics("eval")


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
