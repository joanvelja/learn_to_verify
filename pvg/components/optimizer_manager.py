# pvg/components/optimizer_manager.py
"""Manages optimizers and learning rate schedulers for different model components."""

import logging
from typing import Callable, Literal

# OptimizerSchedulerManager
# Overall: Creates, prepares, and manages optimizers/schedulers for all trainable parameters (sneaky prover, verifier, head). Provides methods to step/zero_grad specific components.
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.data_manager import DataManager
from pvg.components.model_manager import ModelManager
from pvg.config.args import TrainingArgs

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class OptimizerSchedulerManager:
    """
    Creates, prepares, and manages optimizers/schedulers for trainable parameters.

    Handles different configurations for sneaky prover, and verifier.
    Provides methods to step, zero gradients, and retrieve optimizers/schedulers
    based on component keys (e.g., "sneaky_prover", "verifier").
    Dynamically creates optimizers/schedulers based on the current training phase
    to conserve memory. Assumes optimizer/scheduler creation and preparation
    happen after dataloaders are prepared externally.
    """

    def __init__(
        self,
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

        self.sneaky_training_config: TrainingArgs = sneaky_training_config
        self.verifier_training_config: TrainingArgs = verifier_training_config
        self.shared_training_config: dict[str, int | None] = shared_training_config

        self.gradient_accumulation_steps: int = self.shared_training_config["gradient_accumulation_steps"]
        self.global_step_callback: Callable[[], int] = global_step_callback
        self.global_phase_callback: Callable[[], Literal["provers", "verifier"]] = global_phase_callback
        self.global_round_callback: Callable[[], int] = global_round_callback
        self.num_train_epochs: int = self.shared_training_config["num_train_epochs"]
        self.configs = {
            "sneaky_prover": self.sneaky_training_config,
            "verifier": self.verifier_training_config,
        }
        self.num_training_steps: dict[str, int] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] = {}

    def _calculate_num_training_steps(self, dataloader: DataLoader) -> None:
        """
        Calculates the total number of training steps for each phase ("provers", "verifier").

        This calculation depends on the dataloader length, number of epochs, and gradient
        accumulation steps. Results are stored in `self.num_training_steps`.
        Requires dataloaders to be prepared first.
        """

        phase = self.global_phase_callback()
        num_update_steps_per_epoch = len(dataloader) // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)  # Ensure at least one step
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

            # Separate backbone and head parameters
            backbone_params = []
            head_params = []

            for name, param in model.named_parameters():
                if any(keyword in name.lower() for keyword in ["score", "classifier", "head", "lm_head"]):
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            # optimizer = torch.optim.AdamW(
            #     model.parameters(),
            #     lr=config.learning_rate,
            #     weight_decay=config.weight_decay,
            # )
            # Different learning rates: lower for backbone, higher for head
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": backbone_params,
                        "lr": config.learning_rate * 0.1,
                    },  # 10x lower for backbone
                    {
                        "params": head_params,
                        "lr": config.learning_rate,
                    },  # Full LR for head
                ],
                weight_decay=config.weight_decay,
                fused=True,
            )
            self.optimizers[phase] = optimizer
        else:  # 'provers'
            for key in ["sneaky_prover"]:
                config = self.configs[key]
                model = self.model_manager.get_model(key, prepared=False)

                # ============================================================================
                # CRITICAL DEBUGGING: OPTIMIZER CREATION
                # ============================================================================
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info("=" * 80)
                    logger.info(f"🔧 OPTIMIZER CREATION DEBUG FOR {key}")
                    logger.info("=" * 80)

                    logger.info(f"🏷️  Model for optimizer creation ID: {id(model)}")
                    logger.info(f"🏷️  Model for optimizer creation type: {type(model)}")

                    # Check model parameters
                    model_param_ids = {id(p) for p in model.parameters()}
                    model_params_list = list(model.parameters())

                    logger.info(f"🔍 Model parameters count: {len(model_param_ids)}")
                    logger.info("🔍 First 3 model parameter IDs for optimizer creation:")
                    for i in range(min(3, len(model_params_list))):
                        param = model_params_list[i]
                        logger.info(
                            f"🔍   Param {i}: ID {id(param)}, shape {param.shape}, requires_grad {param.requires_grad}"
                        )

                    # Check if model is in training mode
                    logger.info(f"🏷️  Model training mode during optimizer creation: {model.training}")

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    fused=True,
                )

                # ============================================================================
                # CRITICAL DEBUGGING: POST-OPTIMIZER CREATION
                # ============================================================================
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info("🔄 POST-OPTIMIZER CREATION ANALYSIS")
                    logger.info("-" * 40)

                    logger.info(f"🔧 Created optimizer ID: {id(optimizer)}")
                    logger.info(f"🔧 Created optimizer type: {type(optimizer)}")

                    # Check optimizer parameter IDs
                    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
                    optimizer_params_list = [p for group in optimizer.param_groups for p in group["params"]]

                    logger.info(f"🔍 Optimizer parameters count: {len(optimizer_param_ids)}")
                    logger.info("🔍 First 3 optimizer parameter IDs:")
                    for i in range(min(3, len(optimizer_params_list))):
                        param = optimizer_params_list[i]
                        logger.info(
                            f"🔍   Param {i}: ID {id(param)}, shape {param.shape}, requires_grad {param.requires_grad}"
                        )

                    # Verify optimizer was created with the right parameters
                    optimizer_model_match = model_param_ids == optimizer_param_ids
                    logger.info(f"🔍 Optimizer parameters match model parameters? {optimizer_model_match}")

                    if not optimizer_model_match:
                        logger.error("💥 OPTIMIZER CREATION MISMATCH!")
                        logger.error("💥 Optimizer was not created with the correct model parameters!")
                    else:
                        logger.info("✅ Optimizer created with correct model parameters")

                    logger.info("=" * 80)

                self.optimizers[key] = optimizer

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

        if phase not in self.num_training_steps:
            raise ValueError(
                f"Number of training steps for phase '{phase}' not calculated. Call _calculate_num_training_steps first."
            )
        phase_config = "sneaky_prover" if phase == "provers" else phase
        lr_scheduler_type = self.configs[phase_config].lr_scheduler_type
        if not lr_scheduler_type or lr_scheduler_type == "constant":
            logger.info("No learning rate scheduler specified or type is 'constant'. Skipping scheduler creation.")
            # Ensure keys exist even if scheduler is None
            for key in self.optimizers.keys():
                self.schedulers[key] = None
            return
        num_warmup_steps = self.configs[phase_config].num_warmup_steps
        num_training_steps_phase = self.num_training_steps[phase]

        # Conditional on the phase, create the scheduler
        if phase == "verifier":
            for key, optimizer in self.optimizers.items():
                self.schedulers[key] = get_scheduler(
                    name=lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps_phase,
                )
        else:
            for key, optimizer in self.optimizers.items():
                self.schedulers[key] = get_scheduler(
                    name=lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps_phase,
                )

        logger.info(f"Schedulers created for phase {phase} : {self.schedulers[key]}")

    def get_optimizer(self, key: str) -> torch.optim.Optimizer:
        """
        Retrieves the prepared optimizer for the specified component key.

        Args:
            key: The identifier for the component (e.g., "sneaky_prover", "verifier").

        Returns:
            The prepared torch Optimizer.

        Raises:
            KeyError: If the optimizer for the given key is not found (e.g., wrong phase).
        """
        if key not in self.optimizers:
            raise KeyError(
                f"Optimizer '{key}' not found. Current phase might be different or preparation incomplete. Available: {list(self.optimizers.keys())}"
            )
        self.accelerator_manager.accelerators[key].state.select_deepspeed_plugin(key)
        return self.optimizers[key]

    def get_scheduler(self, key: str) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Retrieves the prepared learning rate scheduler for the specified component key.

        Args:
            key: The identifier for the component (e.g., "sneaky_prover", "verifier").

        Returns:
            The prepared torch LR Scheduler, or None if no scheduler was created.

        Raises:
            KeyError: If the key is not found (e.g., wrong phase).
        """
        if key not in self.schedulers:
            raise KeyError(
                f"Scheduler '{key}' not found. Current phase might be different or preparation incomplete. Available: {list(self.schedulers.keys())}"
            )
        self.accelerator_manager.accelerators[key].state.select_deepspeed_plugin(key)
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
