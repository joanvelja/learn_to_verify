import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

# Torch
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
        """Clean up components from the previous phase."""
        logger.info(f"Cleaning up components for {self.state_tracker.phase} phase...")

        # Force synchronization before cleanup
        torch.cuda.synchronize()

        # Clean models
        for model_key in self.get_models_to_cleanup():
            if model_key in self.model_manager.models:
                del self.model_manager.models[model_key]
            if model_key in self.model_manager.ref_models:
                del self.model_manager.ref_models[model_key]

        # Clean optimizers and schedulers
        for component_key in self.get_components_to_cleanup():
            if component_key in self.optimizer_scheduler_manager.optimizers:
                del self.optimizer_scheduler_manager.optimizers[component_key]
            if component_key in self.optimizer_scheduler_manager.schedulers:
                del self.optimizer_scheduler_manager.schedulers[component_key]

        # Clear all caches
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Reset memory allocator for H100
        torch.cuda.set_per_process_memory_fraction(0.95)

        # Clear Python garbage
        import gc

        gc.collect()

        # def optimize_phase_transition():
        #     # Force synchronization before cleanup
        #     torch.cuda.synchronize()

        #     # Clear all caches
        #     torch.cuda.empty_cache()
        #     torch.cuda.reset_peak_memory_stats()

        #     # Reset memory allocator for H100
        #     torch.cuda.set_per_process_memory_fraction(0.95)

        #     # Clear Python garbage
        #     import gc
        #     gc.collect()

        #     # Wait for NCCL to finish
        #     if torch.distributed.is_initialized():
        #         torch.distributed.barrier()
