import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pvg.orchestrator.orchestrator import TrainingPhaseOrchestrator

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class PhaseStrategy(ABC):
    """Abstract base class for phase-specific training strategies."""

    def __init__(self, orchestrator: "TrainingPhaseOrchestrator"):
        self.orchestrator = orchestrator
        self.args = orchestrator.args
        self.model_manager = orchestrator.model_manager
        self.optimizer_scheduler_manager = orchestrator.optimizer_scheduler_manager
        self.data_manager = orchestrator.data_manager
        self.accelerator_manager = orchestrator.accelerator_manager
        self.metrics_logger = orchestrator.metrics_logger
        self.vllm_orchestrator = orchestrator.vllm_orchestrator
        self.state_tracker = orchestrator.state_tracker
        self.grpo = orchestrator.grpo
        self.formatter = orchestrator.formatter
        self.batch_evaluator = orchestrator.batch_evaluator

    @abstractmethod
    def get_models_to_cleanup(self) -> list[str]:
        """Return list of model keys to cleanup before this phase."""
        pass

    @abstractmethod
    def get_components_to_cleanup(self) -> list[str]:
        """Return list of component keys to cleanup before this phase."""
        pass

    @abstractmethod
    def prepare_phase_components(self) -> None:
        """Prepare all components needed for this phase."""
        pass

    @abstractmethod
    def create_trainer(self) -> Any:
        """Create and return the appropriate trainer for this phase."""
        pass

    def cleanup_previous_phase(self) -> None:
        """
        Clean up components from the previous phase by freeing accelerator memory
        and deleting references from managers.
        """
        logger.info(f"Cleaning up components before starting '{self.state_tracker.phase}' phase...")

        component_keys_to_cleanup = self.get_components_to_cleanup()

        # Step 1: For each component, gather all its prepared objects and free them.
        for component_key in component_keys_to_cleanup:
            if component_key not in self.accelerator_manager.accelerators:
                continue

            objects_to_free = []
            # Gather prepared model, optimizer, and scheduler
            if component_key in self.model_manager.prepared_models:
                objects_to_free.append(self.model_manager.prepared_models[component_key])
            if component_key in self.model_manager.prepared_ref_models:
                ref_model = self.model_manager.prepared_ref_models.get(component_key)
                if ref_model:
                    objects_to_free.append(ref_model)
            if component_key in self.optimizer_scheduler_manager.optimizers:
                objects_to_free.append(self.optimizer_scheduler_manager.optimizers[component_key])
            if component_key in self.optimizer_scheduler_manager.schedulers:
                objects_to_free.append(self.optimizer_scheduler_manager.schedulers[component_key])

            # Free the collected objects
            accelerator = self.accelerator_manager.get_accelerator(component_key)
            if objects_to_free:
                logger.info(
                    f"Freeing memory for accelerator '{component_key}' and {len(objects_to_free)} associated objects."
                )
                accelerator.free_memory(*objects_to_free)
            else:
                # Fallback for safety, though objects should always be found
                logger.warning(
                    f"No prepared objects found for '{component_key}'. Calling free_memory() without objects."
                )
                accelerator.free_memory()

        # Step 2: Now that memory is freed, delete all references from the managers.
        model_keys_to_cleanup = self.get_models_to_cleanup()
        for model_key in model_keys_to_cleanup:
            for model_dict in [
                self.model_manager.models,
                self.model_manager.prepared_models,
                self.model_manager.ref_models,
                self.model_manager.prepared_ref_models,
            ]:
                if model_key in model_dict:
                    del model_dict[model_key]

        for component_key in component_keys_to_cleanup:
            for component_dict in [
                self.optimizer_scheduler_manager.optimizers,
                self.optimizer_scheduler_manager.schedulers,
            ]:
                if component_key in component_dict:
                    del component_dict[component_key]

        # Step 3: Clear CUDA cache and Python garbage collection
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Reset memory allocator for H100
        torch.cuda.set_per_process_memory_fraction(0.95)

        # Clear Python garbage
        import gc

        gc.collect()
