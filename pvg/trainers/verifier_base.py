# pvg/trainers/verifier_base.py
"""
VerifierTrainerBase: Abstract base class defining the interface for verifier trainers.
"""

# Overall: Abstract base class defining the interface for verifier trainers.
# __init__: Stores common arguments (configs, shared managers).
# train(num_steps_or_epochs): Abstract method.
# evaluate(): Optional abstract method.

from abc import ABC, abstractmethod

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.data_manager import DataManager
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.model_manager import ModelManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.state_tracker import StateTracker
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.config.args import ExperimentArgs


class VerifierTrainerBase(ABC):
    def __init__(
        self,
        args: ExperimentArgs,
        model_manager: ModelManager,
        data_manager: DataManager,
        accelerator_manager: AcceleratorManager,
        optimizer_scheduler_manager: OptimizerSchedulerManager,
        metrics_logger: MetricsLogger,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
        parallel_backend_impl=None,
    ) -> None:
        self.args = args
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.accelerator_manager = accelerator_manager
        self.optimizer_scheduler_manager = optimizer_scheduler_manager
        self.metrics_logger = metrics_logger
        self.vllm_orchestrator = vllm_orchestrator
        self.state_tracker = state_tracker
        self.backend = parallel_backend_impl

    @abstractmethod
    def train(self, num_steps_or_epochs: int) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float] | None:
        pass
