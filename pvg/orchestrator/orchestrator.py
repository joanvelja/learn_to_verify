# pvg/orchestrator/orchestrator.py

# TrainingPhaseOrchestrator
# Overall: Top-level controller managing rounds and switching between Verifier and Prover training phases.

import asyncio
import gc
import logging

from accelerate.utils import broadcast_object_list
from huggingface_hub import repo_exists

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.code_evaluator import BatchEvaluator
from pvg.components.data_generator import DataGenerator
from pvg.components.data_manager import DataManager
from pvg.components.formatter import Formatter
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.model_manager import ModelManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.state_tracker import StateTracker
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.config.args import ExperimentArgs
from pvg.data.dataset import VerifierDataset
from pvg.orchestrator.prover_phase_strategy import ProverPhaseStrategy
from pvg.orchestrator.verifier_phase_strategy import VerifierPhaseStrategy
from pvg.rl.grpo import GRPO
from pvg.utils import url_exists

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
        assert self.state_tracker.step == 0

        # Initialize dataset type and data generator
        self._initialize_dataset_and_generator()

        start_round = self.state_tracker.round
        target_rounds = self.args.num_rounds

        logger.info("#" * 100)
        logger.info(f"Starting training rounds for dataset type: {self.dataset_type}")
        logger.info(f"Starting from round: {start_round}")
        logger.info(f"Target rounds: {target_rounds}")
        logger.info("#" * 100)
        logger.info("-" * 100)
        logger.info(f"Current round: {self.state_tracker.round}")
        logger.info(f"Current step: {self.state_tracker.step}")
        logger.info(f"Current phase: {self.state_tracker.phase}")
        logger.info("-" * 100)

        loop = asyncio.get_event_loop()

        for round_idx in range(start_round, target_rounds):
            # Memory monitoring: Start of round
            self.model_manager._log_memory_usage(f"ROUND {self.state_tracker.round} START")

            # Generate datamix for current round
            if self.accelerator_manager.get_state_property(property_name="is_main_process"):
                datasize = self.args.dataset.dataset_size
                dataset_name = f"jvelja/{'apps' if self.dataset_type == 'coding' else 'math'}_{datasize}_backdoored_round_{self.state_tracker.round}"
                url = f"https://huggingface.co/datasets/{dataset_name}"

                if not (url_exists(url) and repo_exists(dataset_name, repo_type="dataset")):
                    if self.state_tracker.round == 0:
                        # For round 0, generate data if it doesn't exist
                        loop.run_until_complete(self.data_generator.generate_current_round_data())
                    else:
                        # For round > 0, raise error if data doesn't exist
                        raise ValueError(
                            f"Data for round {self.state_tracker.round} does not exist at {dataset_name}. "
                            f"When starting from round > 0, the data must already exist. "
                            f"Please ensure the dataset exists or start from round 0."
                        )
                else:
                    logger.info(f"Found existing dataset for round {self.state_tracker.round}: {dataset_name}")

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

            # Memory monitoring: End of round (before cleanup)
            self.model_manager._log_memory_usage(f"ROUND {self.state_tracker.round} END (before cleanup)")

            # Increment round and generate new data
            self.state_tracker.increment_round()

            # CRITICAL: Clean up orchestrator-level data structures before generating new data
            self._cleanup_orchestrator_data_structures()
            self.accelerator_manager.wait_for_everyone()

            # Memory monitoring: After round cleanup
            self.model_manager._log_memory_usage(f"ROUND {self.state_tracker.round-1} CLEANUP COMPLETE")

            if self.accelerator_manager.get_state_property(property_name="is_main_process"):
                loop.run_until_complete(self.data_generator.generate_current_round_data())
            self.accelerator_manager.wait_for_everyone()

    def _cleanup_orchestrator_data_structures(self) -> None:
        """
        Clean up orchestrator-level data structures between rounds to prevent memory leaks.
        This is crucial for preventing memory accumulation across rounds.
        """
        logger.info("Cleaning up orchestrator-level data structures...")

        # 1. Clean up any remaining models from GPU memory (between rounds only)
        logger.info("Step 1: Cleaning up any remaining models from GPU memory")
        self.model_manager.fully_offload_models()

        # 2. Clean up verifier datasets
        if self.verifier_dataset_train is not None:
            self.verifier_dataset_train.cleanup()
            self.verifier_dataset_train = None

        if self.verifier_dataset_eval is not None:
            self.verifier_dataset_eval.cleanup()
            self.verifier_dataset_eval = None

        if self.verifier_dataset is not None:
            self.verifier_dataset.cleanup()
            self.verifier_dataset = None

        # 3. Clean up data generator state (only on main process)
        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            if self.data_generator is not None:
                # Clear processing status
                if hasattr(self.data_generator, "processing_status"):
                    self.data_generator.processing_status.clear()

                # Clear stage queues
                if hasattr(self.data_generator, "stage_queues"):
                    for queue in self.data_generator.stage_queues.values():
                        while not queue.empty():
                            try:
                                queue.get_nowait()
                                queue.task_done()
                            except asyncio.QueueEmpty:
                                break

                # Clear active batches
                if hasattr(self.data_generator, "active_batches"):
                    for batch_set in self.data_generator.active_batches.values():
                        batch_set.clear()

                # Cancel any remaining worker tasks
                if hasattr(self.data_generator, "worker_tasks"):
                    for task in self.data_generator.worker_tasks:
                        if not task.done():
                            task.cancel()
                    self.data_generator.worker_tasks.clear()

                # Reset pipeline state
                if hasattr(self.data_generator, "pipeline_running"):
                    self.data_generator.pipeline_running = False

        # 4. Force garbage collection
        gc.collect()

        # 5. Clear CUDA cache
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("Orchestrator-level data structures cleaned up successfully.")

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
            dataset_type = "coding" if self.data_generator.dataset_type == "coding" else "math"
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

        # Memory monitoring: Before phase cleanup
        self.model_manager._log_memory_usage(f"BEFORE {current_phase} phase cleanup")

        strategy.cleanup_previous_phase()
        self.accelerator_manager.wait_for_everyone()

        # Memory monitoring: After phase cleanup
        self.model_manager._log_memory_usage(f"AFTER {current_phase} phase cleanup")

        strategy.prepare_phase_components()

        # Memory monitoring: After component preparation
        self.model_manager._log_memory_usage(f"AFTER {current_phase} component preparation")

        # Create and run trainer
        trainer = strategy.create_trainer()
        if current_phase == "verifier":
            trainer.train(1)
        else:
            trainer.train()

        # Memory monitoring: After training completes
        self.model_manager._log_memory_usage(f"AFTER {current_phase} training complete")

        # NOTE: Keep models on GPU for efficient sync to inference GPU
        # Models will be fully deleted after sync completes in the trainer's sync_weights call


# __init__: Stores ExperimentArgs and references to all shared manager components.
# run_training_rounds(): Main loop over num_rounds. Calls _reset_state_for_new_round(), _run_verifier_phase(), _run_prover_phase() in sequence.
# _run_verifier_phase(round_num): Reads args.verifier_training_type. Instantiates the corresponding VerifierTrainerXXX class, passing necessary configs and shared managers. Calls verifier_trainer.train().
# _run_prover_phase(round_num): Instantiates ProverTrainer, passing necessary configs and shared managers (including the now-trained verifier model access via ModelManager). Calls prover_trainer.train().
# _reset_state_for_new_round(): Calls model_manager.reinitialize_models(). Potentially resets optimizer states if needed (or relies on trainer re-instantiation).
