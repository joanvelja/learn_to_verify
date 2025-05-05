# pvg/components/optimizer_manager.py
"""Manages optimizers and learning rate schedulers for different model components."""

from torch.utils.data import DataLoader

# OptimizerSchedulerManager
# Overall: Creates, prepares, and manages optimizers/schedulers for all trainable parameters (provers, verifier, head). Provides methods to step/zero_grad specific components.

import torch
from pvg.config.args import TrainingArgs
from pvg.components.model_manager import ModelManager
from pvg.components.data_manager import DataManager
from pvg.components.accelerator_manager import AcceleratorManager
from typing import Callable, Literal
from transformers import get_scheduler

import logging

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger

# shared_training_config = {
#          "lr_scheduler_type": args.lr_scheduler_type,
#          "num_warmup_steps": args.num_warmup_steps,
#          "gradient_accumulation_steps": args.gradient_accumulation_steps,
#          "num_train_epochs": args.num_train_epochs,
#          "max_train_steps": args.max_train_steps,
#     }


class OptimizerSchedulerManager:
    """
    Creates, prepares, and manages optimizers/schedulers for trainable parameters.

    Handles different configurations for honest prover, sneaky prover, and verifier.
    Provides methods to step, zero gradients, and retrieve optimizers/schedulers
    based on component keys (e.g., "honest_prover", "verifier").
    Dynamically creates optimizers/schedulers based on the current training phase
    to conserve memory. Assumes optimizer/scheduler creation and preparation
    happen after dataloaders are prepared externally.
    """

    def __init__(
        self,
        honest_training_config: TrainingArgs,
        sneaky_training_config: TrainingArgs,
        verifier_training_config: TrainingArgs,
        shared_training_config: dict[str, int | None],
        model_manager: ModelManager,
        data_manager: DataManager,
        accelerator_manager: AcceleratorManager,
        global_step_callback: Callable[[], int],
        global_phase_callback: Callable[[], Literal["provers", "verifier"]],
        global_round_callback: Callable[[], int],
    ) -> None:
        """
        Initializes the OptimizerSchedulerManager.

        Args:
            honest_training_config: Training configuration for the honest prover.
            sneaky_training_config: Training configuration for the sneaky prover.
            verifier_training_config: Training configuration for the verifier.
            shared_training_config: Shared configuration parameters (e.g., scheduler type, warmup steps).
            model_manager: Manager for accessing model components.
            data_manager: Manager for accessing data loaders.
            accelerator_manager: Manager for handling device placement and distributed training.
            global_step_callback: Function to get the current global training step.
            global_phase_callback: Function to get the current training phase ("provers" or "verifier").
            global_round_callback: Function to get the current training round.
        """

        self.accelerator_manager: AcceleratorManager = accelerator_manager
        self.model_manager: ModelManager = model_manager
        self.data_manager: DataManager = data_manager

        self.honest_training_config: TrainingArgs = honest_training_config
        self.sneaky_training_config: TrainingArgs = sneaky_training_config
        self.verifier_training_config: TrainingArgs = verifier_training_config
        self.shared_training_config: dict[str, int | None] = shared_training_config

        self.gradient_accumulation_steps: int = self.shared_training_config[
            "gradient_accumulation_steps"
        ]
        self.global_step_callback: Callable[[], int] = global_step_callback
        self.global_phase_callback: Callable[[], Literal["provers", "verifier"]] = (
            global_phase_callback
        )
        self.global_round_callback: Callable[[], int] = global_round_callback
        self.num_train_epochs: int = self.shared_training_config["num_train_epochs"]
        self.configs = {
            "provers": self.honest_training_config,  # Assume the two provers have same config (NOTE: This is wonky.)
            "verifier": self.verifier_training_config,
        }
        self.num_training_steps: dict[str, int] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] = {}

        # NOTE: Calculate num_training_steps after dataloader is prepared (and thus num_training_steps is known)
        # self._calculate_num_training_steps()
        # self.create_optimizers()
        # self.create_schedulers() # TODO: Scheduler prep is dependent on the distributed setup!
        # self.prepare_optimizers_and_schedulers() # TODO: Downstream from above

    def _calculate_num_training_steps(self, dataloader: DataLoader) -> None:
        """
        Calculates the total number of training steps for each phase ("provers", "verifier").

        This calculation depends on the dataloader length, number of epochs, and gradient
        accumulation steps. Results are stored in `self.num_training_steps`.
        Requires dataloaders to be prepared first.
        """
        # for phase in ["provers", "verifier"]:
        #     # Use the appropriate dataloader based on phase
        #     # 'provers' uses the honest prover's dataloader for calculation
        #     component = self.data_manager.dataloaders[phase]
        #     dataloader = component["train_dataloader"]
        #     num_update_steps_per_epoch = (
        #         len(dataloader) // self.gradient_accumulation_steps
        #     )
        #     num_update_steps_per_epoch = max(
        #         num_update_steps_per_epoch, 1
        #     )  # Ensure at least one step
        #     num_training_steps = num_update_steps_per_epoch * self.num_train_epochs
        #     self.num_training_steps[phase] = num_training_steps

        phase = self.global_phase_callback()
        num_update_steps_per_epoch = len(dataloader) // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(
            num_update_steps_per_epoch, 1
        )  # Ensure at least one step
        num_training_steps = num_update_steps_per_epoch * self.num_train_epochs
        self.num_training_steps[phase] = num_training_steps

        logger.info(f"Calculated num_training_steps: {self.num_training_steps}")

    def create_optimizers(self) -> None:
        """
        Creates optimizers only for the components active in the current training phase.

        This is memory-efficient as it avoids creating optimizers for inactive models.
        Optimizers are stored in `self.optimizers`.
        """
        self.optimizers = {}  # Clear previous phase's optimizers
        phase = (
            self.global_phase_callback()
        )  # Creating optimizers only for the phase we are training --> Memory efficient (same as ModelManager)
        logger.info(f"Creating optimizers for phase: {phase}")
        if phase == "verifier":
            config = self.configs[phase]
            model = self.model_manager.get_model(phase, prepared=False)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            self.optimizers[phase] = optimizer
            logger.info(f"Optimizer for {phase}: {optimizer}")
        else:  # 'provers'
            for key in ["honest_prover", "sneaky_prover"]:
                config = self.configs[key]
                model = self.model_manager.get_model(key, prepared=False)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                self.optimizers[key] = optimizer
                logger.info(f"Optimizer for phase -- {key}: {optimizer}")

    def create_schedulers(self) -> None:
        """
        Creates learning rate schedulers for the optimizers active in the current phase.

        Schedulers are created based on the shared configuration and the calculated
        number of training steps for the current phase. Schedulers are stored in
        `self.schedulers`. Requires `_calculate_num_training_steps` and `create_optimizers`
        to have been called for the current phase.
        """
        self.schedulers = {}  # Clear previous phase's schedulers
        phase = self.global_phase_callback()
        logger.info(f"Creating schedulers for phase: {phase}")

        lr_scheduler_type = self.shared_training_config.get("lr_scheduler_type")
        if not lr_scheduler_type or lr_scheduler_type == "constant":
            logger.info(
                "No learning rate scheduler specified or type is 'constant'. Skipping scheduler creation."
            )
            # Ensure keys exist even if scheduler is None
            for key in self.optimizers.keys():
                self.schedulers[key] = None
            return

        if phase not in self.num_training_steps:
            raise ValueError(
                f"Number of training steps for phase '{phase}' not calculated. Call _calculate_num_training_steps first."
            )

        num_warmup_steps = self.shared_training_config.get("num_warmup_steps", 0)
        num_training_steps_phase = self.num_training_steps[phase]

        # for key, optimizer in self.optimizers.items():
        #     # Provers share the 'provers' training steps calculation
        #     current_phase_steps = num_training_steps_phase
        #     scheduler = get_scheduler(
        #         name=lr_scheduler_type,
        #         optimizer=optimizer,
        #         num_warmup_steps=num_warmup_steps * self.accelerator_manager.get_state_property("num_processes"),
        #         num_training_steps=current_phase_steps,
        #     )
        #     self.schedulers[key] = scheduler

        # Conditional on the phase, create the scheduler
        if phase == "verifier":
            for key, optimizer in self.optimizers.items():
                self.schedulers[key] = get_scheduler(
                    name=lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps
                    * self.accelerator_manager.get_state_property("num_processes"),
                    num_training_steps=num_training_steps_phase,
                )
        else:
            for key, optimizer in self.optimizers.items():
                self.schedulers[key] = get_scheduler(
                    name=lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps
                    * self.accelerator_manager.get_state_property("num_processes"),
                    num_training_steps=num_training_steps_phase,
                )

        logger.info(f"Schedulers created for phase {phase} : {self.schedulers[key]}")

    def prepare_optimizers_and_schedulers(self, dataloader: DataLoader) -> None:
        """
        Prepares the created optimizers and schedulers using the AcceleratorManager.

        This step is necessary for distributed training and automatic device placement.
        Updates `self.optimizers` and `self.schedulers` with their prepared versions.
        Requires `create_optimizers` and `create_schedulers` to have been called.
        """
        logger.info("Preparing optimizers and schedulers with Accelerator...")
        phase = self.global_phase_callback()
        if phase == "verifier":
            accelerator = self.accelerator_manager.get_accelerator(
                phase
            )  # TODO: This is a bit of a hack.
            optimizer = self.optimizers[phase]
            scheduler = self.schedulers[phase]
            optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
            self.optimizers[phase] = optimizer
            self.schedulers[phase] = scheduler
        else:
            for key, optimizer in self.optimizers.items():
                if key != "verifier":
                    accelerator = self.accelerator_manager.get_accelerator(
                        key
                    )  # TODO: This is a bit of a hack.
                    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
                    self.optimizers[key] = optimizer
                    self.schedulers[key] = scheduler

    def get_optimizer(self, key: str) -> torch.optim.Optimizer:
        """
        Retrieves the prepared optimizer for the specified component key.

        Args:
            key: The identifier for the component (e.g., "honest_prover", "verifier").

        Returns:
            The prepared torch Optimizer.

        Raises:
            KeyError: If the optimizer for the given key is not found (e.g., wrong phase).
        """
        if key not in self.optimizers:
            raise KeyError(
                f"Optimizer '{key}' not found. Current phase might be different or preparation incomplete. Available: {list(self.optimizers.keys())}"
            )
        return self.optimizers[key]

    def get_scheduler(self, key: str) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Retrieves the prepared learning rate scheduler for the specified component key.

        Args:
            key: The identifier for the component (e.g., "honest_prover", "verifier").

        Returns:
            The prepared torch LR Scheduler, or None if no scheduler was created.

        Raises:
            KeyError: If the key is not found (e.g., wrong phase).
        """
        if key not in self.schedulers:
            raise KeyError(
                f"Scheduler '{key}' not found. Current phase might be different or preparation incomplete. Available: {list(self.schedulers.keys())}"
            )
        return self.schedulers[key]

    def step_optimizer(self, key: str) -> None:
        """
        Performs a single optimization step for the specified component's optimizer.

        Args:
            key: The identifier for the component.
        """
        optimizer = self.get_optimizer(key)
        optimizer.step()
        # self.accelerator_manager.accelerators[key].step()

    def step_scheduler(self, key: str) -> None:
        """
        Steps the learning rate scheduler for the specified component, if it exists.

        Args:
            key: The identifier for the component.
        """
        scheduler = self.get_scheduler(key)
        if scheduler is not None:
            scheduler.step()

    def zero_grad_optimizer(self, key: str) -> None:
        """
        Zeros the gradients for the specified component's optimizer.

        Args:
            key: The identifier for the component.
        """
        optimizer = self.get_optimizer(key)
        optimizer.zero_grad()

    def get_last_lr(self, key: str) -> float | None:
        """
        Gets the last computed learning rate for the specified component's scheduler.

        Args:
            key: The identifier for the component.

        Returns:
            The last learning rate as a float, or None if no scheduler exists or
            it doesn't support `get_last_lr()`.
        """
        scheduler = self.get_scheduler(key)
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            # get_last_lr() typically returns a list, one per param group
            lrs = scheduler.get_last_lr()
            return lrs[0] if lrs else None  # Return the first LR if available
        return None

    def load_and_prepare_optimizers_and_schedulers(
        self, dataloader: DataLoader
    ) -> None:
        """
        Loads and prepares optimizers and schedulers for the current phase.
        """
        self.create_optimizers()
        self._calculate_num_training_steps(dataloader)
        self.create_schedulers()
        self.prepare_optimizers_and_schedulers(dataloader)
