# pvg/components/accelerator_manager.py

import logging
import os
from datetime import timedelta
from typing import Any, Callable

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DeepSpeedPlugin,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)
from accelerate.utils.dataclasses import DataLoaderConfiguration
from torch.utils.data import DataLoader

from pvg.config.args import WandbArgs
from pvg.utils.utils import prepare_deepspeed  # For ref model prep

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class AcceleratorManager:
    """
    Manages the creation, configuration, and access to the two Accelerator instances and their associated DeepSpeed plugins. Handles the prepare calls for different component types, ensuring correct DeepSpeed context selection. Provides access to distributed training properties (rank, world size, device, etc.). Manages distributed synchronization primitives (wait_for_everyone, gather_object, broadcast_object_list, gather_for_metrics).
    """

    def __init__(
        self,
        output_dir: str,
        gradient_accumulation_steps: int,
        mixed_precision: str | None,
        ds_config_sneaky_prover: str,
        ds_config_verifier: str,
        wandb_config: WandbArgs,
        global_step_callback: Callable[[], int],
    ) -> None:
        """
        Initializes the AcceleratorManager.

        Args:
            output_dir: (str) For ProjectConfiguration.
            gradient_accumulation_steps: (int) Passed to Accelerator.
            mixed_precision: (str | None) Passed to Accelerator.
            ds_config_sneaky_prover: (str) Path to DeepSpeed config file.
            wandb_config: (WandbArgs) For log_with parameter in Accelerator.

        Returns:
            None
        """
        self.output_dir: str = output_dir
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.mixed_precision: str | None = mixed_precision
        self.ds_config_sneaky_prover: str = ds_config_sneaky_prover
        self.ds_config_verifier: str = ds_config_verifier
        self.wandb_config: WandbArgs = wandb_config
        self.global_step_callback: Callable[[], int] = global_step_callback
        self.wandb_run: Any = None
        self.llm_interaction_log_dir: str | None = None

        # A function to get the current global step for logging filenames.
        # Works as follows:
        # The Callback Solution:
        # - Instead of giving the components the owner of the data (Trainer) or the data itself repeatedly, we give them a way to ask for the data when they need it.
        # - This "way to ask" is a function – the callback.
        # - In the Trainer's __init__: When initializing a component, the Trainer creates and passes a simple function (usually a lambda) that knows how to access the Trainer's own self.global_step.
        # - The component can then call this function whenever it needs the global step.

        self.accelerators: dict[str, Accelerator] = {}
        self.deepspeed_plugins: dict[str, DeepSpeedPlugin] = {}
        self.project_config: ProjectConfiguration = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir, "accelerate_logs"),
        )

        logger.info("Initializing Accelerators...")

        # Create DeepSpeed plugins -- Need to specify the optimizer stuff + lr scheduler stuff for deepspeed zero3 to work.
        # See: https://github.com/deepspeedai/DeepSpeed/issues/3024
        ds_plugin_sneaky_prover: DeepSpeedPlugin = DeepSpeedPlugin(
            hf_ds_config=self.ds_config_sneaky_prover
        )
        ds_plugin_verifier: DeepSpeedPlugin = DeepSpeedPlugin(
            hf_ds_config=self.ds_config_verifier
        )
        logger.info("DeepSpeed plugins created.")

        self.deepspeed_plugins = {
            "sneaky_prover": ds_plugin_sneaky_prover,
            "verifier": ds_plugin_verifier,
        }

        # Instantiate the first accelerator (sneaky prover)
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=False, use_stateful_dataloader=False
        )
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
        accelerator_sneaky_prover = Accelerator(
            deepspeed_plugin=self.deepspeed_plugins,  # Pass all; see: https://huggingface.co/docs/accelerate/usage_guides/deepspeed_multiple_model
            log_with="wandb",
            project_config=self.project_config,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            dataloader_config=dataloader_config,
            kwargs_handlers=[init_kwargs],
        )
        logger.info("First Accelerator (accelerator_sneaky_prover) initialized.")
        self.accelerators["sneaky_prover"] = accelerator_sneaky_prover

        # Instantiate the second accelerator (verifier)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
        accelerator_verifier = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            dataloader_config=dataloader_config,
            kwargs_handlers=[init_kwargs],
        )  # Pass nothing, Accelerator is a stateful object
        logger.info("Second Accelerator (accelerator_verifier) initialized.")
        self.accelerators["verifier"] = accelerator_verifier

        # primary_accelerator: Accelerator (Reference to accelerators["sneaky_prover"] for convenience)
        self.primary_accelerator = self.accelerators["sneaky_prover"]

        self.project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=os.path.join(self.output_dir, "accelerate_logs"),
        )

    def get_accelerator(self, key: str) -> Accelerator:
        """Returns the accelerator for the given model key."""
        return self.accelerators[key]

    def prepare_components(
        self,
        key: str,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ):
        """Prepares the components for the given key."""
        self.accelerators[key].state.select_deepspeed_plugin(key)
        logger.info(
            f"Selected DeepSpeed plugin '{key}' for accelerator '{key}' before preparing components."
        )
        return self.accelerators[key].prepare(model, optimizer, dataloader)

    # Why separate method? For evaluation dataset.
    def prepare_dataloader(self, dataloader: DataLoader, key: str) -> DataLoader:
        """Prepares the dataloader for the given key."""
        self.accelerators[key].state.select_deepspeed_plugin(key)
        logger.info(
            f"Selected DeepSpeed plugin '{key}' for accelerator '{key}' before preparing dataloader."
        )
        return self.accelerators[key].prepare(dataloader)

    # Why separate method? Because we need to first prepare the dataloader, then calculate the number of training steps, then create the scheduler with this & prepare it.
    def prepare_scheduler(
        self, key: str, scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Prepares the scheduler for the given key."""
        self.accelerators[key].state.select_deepspeed_plugin(key)
        logger.info(
            f"Selected DeepSpeed plugin '{key}' for accelerator '{key}' before preparing scheduler."
        )
        return self.accelerators[key].prepare(scheduler)

    def prepare_ref_model(self, key: str, model: torch.nn.Module) -> torch.nn.Module:
        """Prepares the ref model for the given key."""
        self.accelerators[key].state.select_deepspeed_plugin(key)
        logger.info(
            f"Selected DeepSpeed plugin '{key}' for accelerator '{key}' before preparing reference model."
        )
        # use prepare_deepspeed
        return prepare_deepspeed(model, accelerator=self.accelerators[key])

    def unwrap_model(self, model: torch.nn.Module, key: str) -> torch.nn.Module:
        """
        Calls unwrap_model on the primary accelerator (or appropriate one if needed, though usually primary is fine).
        """
        self.accelerators[key].state.select_deepspeed_plugin(key)
        return self.accelerators[key].unwrap_model(model)

    def save_state(self, output_dir: str, key: str) -> None:
        """Saves the state for the given key."""
        # Select plugin
        plugin = self.deepspeed_plugins[key]
        self.accelerators[key].save_state(output_dir, plugin)

    def load_state(self, input_dir: str, key: str) -> None:
        """Loads the state for the given key."""
        # Select plugin
        plugin = self.deepspeed_plugins[key]
        self.accelerators[key].load_state(input_dir, plugin)

    def wait_for_everyone(self, key: str | None = None) -> None:
        """Calls wait_for_everyone on the selected accelerator."""
        if key is None:
            self.primary_accelerator.wait_for_everyone()
        else:
            self.accelerators[key].wait_for_everyone()

    def gather_for_metrics(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Calls gather_for_metrics on the selected accelerator."""
        return self.accelerators[key].gather_for_metrics(tensor)

    def get_state_property(self, property_name: str, key: str | None = None) -> Any:
        """
        Accessor for shared state properties like num_processes, process_index, local_process_index, device, is_main_process, is_local_main_process, distributed_type. Uses the primary accelerator.
        """
        if key is None:
            return getattr(self.primary_accelerator, property_name)
        else:
            return getattr(self.accelerators[key], property_name)

    def select_plugin(self, key: str) -> DeepSpeedPlugin | None:
        """Explicitly selects the deepspeed plugin (might be needed by VLLMOrchestrator before gather)."""
        plugin = self.deepspeed_plugins[key]
        if plugin is None:
            raise ValueError(f"No plugin found for key: {key}")
        return plugin

    def get_plugin(self, key: str) -> DeepSpeedPlugin | None:
        """Returns the plugin instance."""
        return self.deepspeed_plugins[key]

    def init_trackers(self, *args, **kwargs) -> None:
        """Calls primary_accelerator.init_trackers."""
        self.primary_accelerator.init_trackers(*args, **kwargs)

    def get_tracker(self, name: str) -> Any:
        """Calls primary_accelerator.get_tracker."""
        return self.primary_accelerator.get_tracker(name)

    def log(self, *args, **kwargs) -> None:
        """Calls primary_accelerator.log."""
        self.primary_accelerator.log(*args, **kwargs)

    def backward(self, loss: torch.Tensor, key: str) -> None:
        """Calls backward on the specific accelerator."""
        self.accelerators[key].state.select_deepspeed_plugin(key)
        self.accelerators[key].backward(loss)

    def clip_grad_norm_(
        self, parameters: Any, max_norm: float, key: str
    ) -> torch.Tensor:
        """Calls clip_grad_norm_ on the specific accelerator."""
        self.accelerators[key].state.select_deepspeed_plugin(key)
        return self.accelerators[key].clip_grad_norm_(parameters, max_norm)

    def setup_wandb(self, config: dict[str, Any]) -> None:
        try:
            logger.info("Initializing WandB tracker via accelerator.init_trackers...")
            self.init_trackers(
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

            # Get the wandb run object on the main process first
            if self.get_state_property("is_main_process"):
                self.wandb_run = self.get_tracker("wandb").run
                if self.wandb_run:
                    logger.info(
                        f"Successfully retrieved WandB run. Run ID: {self.wandb_run.id}"
                    )
                    # Now that we have the run object, create the log directory
                    self.llm_interaction_log_dir = os.path.join(
                        self.output_dir,
                        self.wandb_run.id,
                        "llm_interaction_logs",
                    )
                    os.makedirs(self.llm_interaction_log_dir, exist_ok=True)
                    logger.info(
                        f"LLM interaction logs will be saved to: {self.llm_interaction_log_dir}"
                    )
                else:
                    logger.error(
                        "Called init_trackers, but failed to retrieve WandB run object."
                    )
                    self.wandb_run = None
                    self.llm_interaction_log_dir = None
            else:
                # For non-main processes, set wandb_run to None and use a fallback log dir
                self.wandb_run = None
                self.llm_interaction_log_dir = os.path.join(
                    self.output_dir,
                    "llm_interaction_logs",
                )
                os.makedirs(self.llm_interaction_log_dir, exist_ok=True)

        except Exception as e:
            logger.error(
                f"Error during accelerator.init_trackers or run retrieval: {e}",
                exc_info=True,
            )
            self.wandb_run = None
            # Fallback log directory
            self.llm_interaction_log_dir = os.path.join(
                self.output_dir,
                "llm_interaction_logs",
            )
            os.makedirs(self.llm_interaction_log_dir, exist_ok=True)

        # Additional wandb configuration only on main process
        if self.get_state_property("is_main_process") and self.wandb_run is not None:
            try:
                self.wandb_run.config.update(config, allow_val_change=True)

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
                        logger.debug(f"Library {lib} not found for version logging.")

                self.wandb_run.config.update(
                    {
                        "environment/python_version": sys.version,
                        "environment/platform": platform.platform(),
                        "environment/num_processes": self.get_state_property(
                            "num_processes"
                        ),
                        "environment/mixed_precision": self.get_state_property(
                            "mixed_precision"
                        ),
                        "environment/distributed_type": str(
                            self.get_state_property("distributed_type")
                        ),
                        "environment/library_versions": lib_versions,
                    }
                )
                logger.info("Environment details logged to WandB.")
            except Exception as e:
                logger.warning(f"Could not log all environment details: {e}")
        elif self.get_state_property("is_main_process"):
            logger.warning(
                "WandB run object not available, skipping additional configuration."
            )

    def get_llm_interaction_log_dir(self) -> str:
        """Returns the LLM interaction log directory."""
        if self.llm_interaction_log_dir is None:
            # Fallback to a default directory if not set
            fallback_dir = os.path.join(self.output_dir, "llm_interaction_logs")
            os.makedirs(fallback_dir, exist_ok=True)
            self.llm_interaction_log_dir = fallback_dir
            logger.warning(
                f"LLM interaction log directory was not set, using fallback: {fallback_dir}"
            )
        return self.llm_interaction_log_dir
