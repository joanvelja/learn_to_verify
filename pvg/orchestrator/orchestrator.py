# pvg/orchestrator/orchestrator.py

# TrainingPhaseOrchestrator
# Overall: Top-level controller managing rounds and switching between Verifier and Prover training phases.

import asyncio
import logging

from accelerate.utils import broadcast_object_list
from huggingface_hub import repo_exists

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.data_generator import DataGenerator
from pvg.components.data_manager import DataManager
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.model_manager import ModelManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.state_tracker import StateTracker
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.config.args import ExperimentArgs
from pvg.data.dataset import VerifierDataset
from pvg.rl.grpo import GRPO
from pvg.utils import url_exists
from pvg.orchestrator.verifier_phase_strategy import VerifierPhaseStrategy
from pvg.orchestrator.prover_phase_strategy import ProverPhaseStrategy
from pvg.components.formatter import Formatter
from pvg.components.code_evaluator import BatchEvaluator

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class TrainingPhaseOrchestrator:
    def __init__(
        self,
        args: ExperimentArgs,
        model_manager: ModelManager,
        optimizer_scheduler_manager: OptimizerSchedulerManager,
        data_manager: DataManager,
        accelerator_manager: AcceleratorManager,
        metrics_logger: MetricsLogger,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
        formatter: Formatter,
        batch_evaluator: BatchEvaluator,
        grpo: GRPO,
    ) -> None:
        self.args = args
        self.model_manager = model_manager
        self.optimizer_scheduler_manager = optimizer_scheduler_manager
        self.data_manager = data_manager
        self.accelerator_manager = accelerator_manager
        self.metrics_logger = metrics_logger
        self.vllm_orchestrator = vllm_orchestrator
        self.state_tracker = state_tracker
        self.grpo = grpo
        self.formatter = formatter
        self.batch_evaluator = batch_evaluator
        self.data_generator: DataGenerator | None = None
        self.verifier_dataset: VerifierDataset | None = None
        self.dataset_type: str = ""
        self.verifier_dataset_train: VerifierDataset | None = None
        self.verifier_dataset_eval: VerifierDataset | None = None

        # Initialize phase strategies
        self.phase_strategies = {
            "verifier": VerifierPhaseStrategy(self),
            "provers": ProverPhaseStrategy(self),
        }

    def run_training_rounds(self) -> None:
        assert self.state_tracker.phase == "verifier"
        assert self.state_tracker.round == 0
        assert self.state_tracker.step == 0

        # Initialize dataset type and data generator
        self._initialize_dataset_and_generator()

        logger.info("#" * 100)
        logger.info(f"Starting training rounds for dataset type: {self.dataset_type}")
        logger.info("#" * 100)
        logger.info("-" * 100)
        logger.info(f"Current round: {self.state_tracker.round}")
        logger.info(f"Current step: {self.state_tracker.step}")
        logger.info(f"Current phase: {self.state_tracker.phase}")
        logger.info("-" * 100)

        loop = asyncio.get_event_loop()

        for _ in range(self.args.num_rounds):
            # Generate datamix for current round
            if self.accelerator_manager.get_state_property(
                property_name="is_main_process"
            ):
                if self.state_tracker.round == 0:
                    dataset_name = f"jvelja/{'apps' if self.dataset_type == 'coding' else 'math'}_backdoored_round_{self.state_tracker.round}"
                    url = f"https://huggingface.co/datasets/{dataset_name}"
                    if not (
                        url_exists(url)
                        and repo_exists(dataset_name, repo_type="dataset")
                    ):
                        loop.run_until_complete(
                            self.data_generator.generate_current_round_data()
                        )
                    else:
                        logger.info(
                            f"Skipping datamix creation for round {self.state_tracker.round} because it already exists."
                        )

            self.accelerator_manager.wait_for_everyone()

            logger.info(f"Starting training round {self.state_tracker.round}...")
            logger.info(
                f"Training the {self.state_tracker.phase} model..."
                if self.state_tracker.phase == "verifier"
                else "Training the provers..."
            )

            # Phase 1: Verifier
            self._execute_training_phase()

            # Phase 2: Prover
            self.state_tracker.increment_phase()
            self._execute_training_phase()
            self.state_tracker.increment_phase()

            # Increment round and generate new data
            self.state_tracker.increment_round()
            loop.run_until_complete(self.data_generator.generate_current_round_data())

    def _initialize_dataset_and_generator(self) -> None:
        """Initialize data generator and determine dataset type."""
        # Only the main process creates the DataGenerator and defines the dataset_type
        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            self.data_generator = DataGenerator(
                self.args,
                self.data_manager,
                self.vllm_orchestrator,
                self.state_tracker,
                self.args.enable_backdoor_verification,
            )
            dataset_type = (
                "coding" if self.data_generator.dataset_type == "coding" else "math"
            )
        else:
            dataset_type = None

        # Wait until all ranks have reached this point
        self.accelerator_manager.wait_for_everyone()

        # Broadcast the string dataset_type from rank 0 to all ranks, then unpack
        dataset_type = broadcast_object_list([dataset_type], 0)[0]

        # Store the broadcasted value on each rank
        self.dataset_type = dataset_type

    def _execute_training_phase(self) -> None:
        """Execute the current training phase using the appropriate strategy."""
        current_phase = self.state_tracker.phase
        strategy = self.phase_strategies[current_phase]

        # Reset state for new phase
        logger.info("-" * 100)
        logger.info(f"Resetting state for new phase: {current_phase}...")

        strategy.cleanup_previous_phase()
        strategy.prepare_phase_components()

        # Create and run trainer
        trainer = strategy.create_trainer()
        if current_phase == "verifier":
            trainer.train(1)
        else:
            trainer.train()


# __init__: Stores ExperimentArgs and references to all shared manager components.
# run_training_rounds(): Main loop over num_rounds. Calls _reset_state_for_new_round(), _run_verifier_phase(), _run_prover_phase() in sequence.
# _run_verifier_phase(round_num): Reads args.verifier_training_type. Instantiates the corresponding VerifierTrainerXXX class, passing necessary configs and shared managers. Calls verifier_trainer.train().
# _run_prover_phase(round_num): Instantiates ProverTrainer, passing necessary configs and shared managers (including the now-trained verifier model access via ModelManager). Calls prover_trainer.train().
# _reset_state_for_new_round(): Calls model_manager.reinitialize_models(). Potentially resets optimizer states if needed (or relies on trainer re-instantiation).
