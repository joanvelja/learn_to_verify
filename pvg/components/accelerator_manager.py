# pvg/components/accelerator_manager.py

from pvg.config.args import WandbArgs
from pvg.utils.utils import prepare_deepspeed  # For ref model prep
from accelerate import Accelerator
from accelerate.utils.dataclasses import DataLoaderConfiguration
from accelerate.utils import (
    DeepSpeedPlugin,
    ProjectConfiguration,
)
from typing import Any, Callable
from torch.utils.data import DataLoader
import torch
import logging
import os

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
        ds_config_honest_prover: str,
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
            ds_config_honest_prover: (str) Path to DeepSpeed config file.
            ds_config_sneaky_prover: (str) Path to DeepSpeed config file.
            wandb_config: (WandbArgs) For log_with parameter in Accelerator.

        Returns:
            None
        """
        self.output_dir: str = output_dir
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.mixed_precision: str | None = mixed_precision
        self.ds_config_honest_prover: str = ds_config_honest_prover
        self.ds_config_sneaky_prover: str = ds_config_sneaky_prover
        self.ds_config_verifier: str = ds_config_verifier
        self.wandb_config: WandbArgs = wandb_config
        self.global_step_callback: Callable[[], int] = global_step_callback

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

        try:
            ds_plugin_honest_prover: DeepSpeedPlugin = DeepSpeedPlugin(
                hf_ds_config=self.ds_config_honest_prover
            )
            ds_plugin_sneaky_prover: DeepSpeedPlugin = DeepSpeedPlugin(
                hf_ds_config=self.ds_config_sneaky_prover
            )
            ds_plugin_verifier: DeepSpeedPlugin = DeepSpeedPlugin(
                hf_ds_config=self.ds_config_verifier
            )
            logger.info("DeepSpeed plugins created.")
        except Exception as e:
            logger.error(f"Failed to create DeepSpeed plugins: {e}", exc_info=True)
            raise

        self.deepspeed_plugins = {
            "honest_prover": ds_plugin_honest_prover,
            "sneaky_prover": ds_plugin_sneaky_prover,
            "verifier": ds_plugin_verifier,
        }

        # Instantiate the first accelerator
        try:
            dataloader_config = DataLoaderConfiguration(
                dispatch_batches=False, use_stateful_dataloader=False
            )
            accelerator_honest_prover = Accelerator(
                deepspeed_plugin=self.deepspeed_plugins,  # Pass all; see: https://huggingface.co/docs/accelerate/usage_guides/deepspeed_multiple_model
                log_with="wandb",
                project_config=self.project_config,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision=self.mixed_precision,
                # dataloader_config=dataloader_config,
            )
            logger.info("First Accelerator (accelerator_honest_prover) initialized.")
            self.accelerators["honest_prover"] = accelerator_honest_prover
        except Exception as e:
            logger.error(
                f"Failed to initialize the first Accelerator: {e}", exc_info=True
            )
            raise

        # Instantiate the second accelerator (sneaky prover)
        try:
            accelerator_sneaky_prover = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision=self.mixed_precision,
            )  # Pass nothing, Accelerator is a stateful object
            logger.info("Second Accelerator (accelerator_sneaky_prover) initialized.")
            self.accelerators["sneaky_prover"] = accelerator_sneaky_prover
        except Exception as e:
            logger.error(
                f"Failed to initialize the second Accelerator: {e}", exc_info=True
            )
            raise

        # Instantiate the third accelerator (verifier)
        try:
            accelerator_verifier = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision=self.mixed_precision,
                dataloader_config=dataloader_config,
            )  # Pass nothing, Accelerator is a stateful object
            logger.info("Third Accelerator (accelerator_verifier) initialized.")
            self.accelerators["verifier"] = accelerator_verifier
        except Exception as e:
            logger.error(
                f"Failed to initialize the third Accelerator: {e}", exc_info=True
            )
            raise
        # primary_accelerator: Accelerator (Reference to accelerators["honest_prover"] for convenience)
        self.primary_accelerator = self.accelerators["honest_prover"]

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
