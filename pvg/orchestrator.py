# pvg/orchestrator.py

# TrainingPhaseOrchestrator
# Overall: Top-level controller managing rounds and switching between Verifier and Prover training phases.

from pvg.config.args import ExperimentArgs
from pvg.components.model_manager import ModelManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.components.state_tracker import StateTracker
from pvg.components.data_manager import DataManager
from pvg.components.data_generator_async import DataGenerator
from pvg.components.accelerator_manager import AcceleratorManager
from pvg.rl.grpo import GRPO
from pvg.data.dataset import VerifierDataset
from pvg.trainers.verifier_regressor import VerifierRegressorTrainer
from pvg.trainers.prover_trainer import ProverTrainer
from accelerate.utils import broadcast_object_list
from huggingface_hub import repo_exists
from pvg.utils import url_exists
import asyncio

# from pvg.trainers.verifier_trainer import VerifierTrainer
# from pvg.trainers.prover_trainer import ProverTrainer
# TODO: Add other verifier trainers

import logging

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
        self.data_generator: DataGenerator | None = None
        self.verifier_dataset: VerifierDataset | None = None

    def run_training_rounds(self) -> None:

        assert self.state_tracker.phase == "verifier"
        assert self.state_tracker.round == 0
        assert self.state_tracker.step == 0

        # Only the main process creates the DataGenerator and defines the dataset_type
        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            self.data_generator = DataGenerator(
                self.args,
                self.data_manager,
                self.vllm_orchestrator,
                self.state_tracker,
            )
            dataset_type = self.data_generator.dataset_type
        else:
            dataset_type = None

        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
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

        logger.info("#" * 100)
        logger.info(f"Starting training rounds for dataset type: {self.dataset_type}")
        logger.info("#" * 100)
        logger.info("-" * 100)
        logger.info(f"Current round: {self.state_tracker.round}")
        logger.info(f"Current step: {self.state_tracker.step}")
        logger.info(f"Current phase: {self.state_tracker.phase}")
        logger.info("-" * 100)

        loop = asyncio.get_event_loop()

        for round_num in range(self.args.num_rounds):
            # Make Verifier datamix for current round
            # John: Technically, the datamix could be made here with whatever prover model is currently loaded for inference.
            # Just make sure that when we then run the prover phase, we re-init the provers in vLLM orchestrator.

            # Check logic:
            # - If we're in round 0, make verifier datamix with raw provers.
            # - If we're in round 1 or later, make verifier datamix with the latest prover model.
            #    - That is, when finishing running prover_phase, we do not want to trigger a re-init;
            #    - Keep the trained provers in service for generating the datamix.
            #    - When we're done with the verifier phase, we can re-init the provers for the next round.
            if self.accelerator_manager.get_state_property(
                property_name="is_main_process"
            ):  # Make the datamix on the main process due to the vLLM orchestrator
                if self.state_tracker.round == 0:
                    dataset_name = f"jvelja/{'apps' if self.dataset_type == 'coding' else 'math'}_backdoored_round_{self.state_tracker.round}"
                    url = f"https://huggingface.co/datasets/{dataset_name}"
                    if not (
                        url_exists(url)
                        and repo_exists(
                            dataset_name,
                            repo_type="dataset",
                        )
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

            self.verifier_dataset_train = VerifierDataset(
                current_round_num=self.state_tracker.round,
                max_rounds_to_keep=10,
                new_sample_weight_target=0.8,
                batch_size=8,  # TODO: Make this a parameter...
                seed=42,
                dataset_type=self.dataset_type,
                correct_column_identifier="honest_solution",
                incorrect_column_identifier="injected_solution",
                tokenizer=self.data_manager.tokenizer,
                split="train",
            )
            self.verifier_dataset_eval = VerifierDataset(
                current_round_num=self.state_tracker.round,
                max_rounds_to_keep=10,
                new_sample_weight_target=0.8,
                batch_size=8,
                seed=42,
                dataset_type=self.dataset_type,
                correct_column_identifier="honest_solution",
                incorrect_column_identifier="injected_solution",
                tokenizer=self.data_manager.tokenizer,
                split="eval",
            )

            # Phase 1: Verifier
            # Reset state for new phase : models, optimizers, schedulers
            # For Round 0, assert the expected length based on your confirmation
            self._reset_state_for_new_phase()

            # Run Verifier phase
            self._run_verifier_phase()

            # Phase 2: Prover
            # Increment phase: Verifier -> Prover
            self.state_tracker.increment_phase()

            # Reset state for new phase : models, optimizers, schedulers
            self._reset_state_for_new_phase()
            self._run_prover_phase()
            self.state_tracker.increment_phase()  # Prover -> Verifier

            # Increment round
            self.state_tracker.increment_round()

            # Generate new data for round i+1
            self.data_generator.generate_current_round_data()

    def _reset_state_for_new_phase(self) -> None:

        logger.info("-" * 100)
        logger.info(f"Resetting state for new phase: {self.state_tracker.phase}...")
        if self.state_tracker.phase == "verifier":

            logger.info("Preparing components for Verifier phase...")
            train_dataloader = self.verifier_dataset_train.get_dataloader()
            eval_dataloader = self.verifier_dataset_eval.get_dataloader()
            self.model_manager.load_models()  # This will load the model for the current phase
            self.optimizer_scheduler_manager.create_optimizers()

            model = self.model_manager.get_model("verifier", prepared=False)
            optimizer = self.optimizer_scheduler_manager.get_optimizer("verifier")

            # TODO: Make the below robust
            # # <<< START INSERTION H2 >>>
            # initial_dataloader_len = len(
            #     train_dataloader
            # )  # Uses DataLoader's len logic
            # # Expected total batches = ceil(dataset_len / batch_size)
            # # Use math.ceil
            # import math

            # expected_initial_batches = math.ceil(len(self.verifier_dataset) / 16)
            # if self.accelerator_manager.get_state_property(
            #     property_name="is_main_process"
            # ):
            #     print(
            #         f"[DEBUG H2] Initial Verifier DataLoader created. len={initial_dataloader_len}. Expected total batches (before distribution): {expected_initial_batches}"
            #     )
            #     # Check: initial_dataloader_len should be equal to expected_initial_batches
            #     if initial_dataloader_len != expected_initial_batches:
            #         print(
            #             f"[WARNING H2] Initial DataLoader length ({initial_dataloader_len}) does not match expected total batches ({expected_initial_batches})!"
            #         )
            # # <<< END INSERTION H2 >>>

            components = self.accelerator_manager.prepare_components(
                key="verifier",
                dataloader=train_dataloader,
                optimizer=optimizer,
                model=model,
            )

            # Unpack :  model, optimizer, dataloader
            self.model_manager.prepared_models["verifier"] = components[
                0
            ]  # Quirk of prepare_components: returns a tuple of (model, optimizer, dataloader)
            self.optimizer_scheduler_manager.optimizers["verifier"] = components[1]
            self.data_manager.dataloaders["verifier"]["train_dataloader"] = components[
                2
            ]

            # # <<< START INSERTION H3 >>>
            # prepared_dataloader_len = len(
            #     components[2]
            # )  # Length reported by the prepared object
            # # Expected batches per process = floor(ceil(dataset_len / num_processes) / batch_size) because drop_last=True
            # import math

            # expected_samples_per_process = math.ceil(
            #     len(self.verifier_dataset) / 2
            # )  # NOTE: Hardcoded for convenience
            # expected_batches_per_process = math.floor(
            #     expected_samples_per_process / 8
            # )  # NOTE: Hardcoded for convenience

            # # Assertion on each rank
            # assert (
            #     expected_batches_per_process <= prepared_dataloader_len
            # ), f"[H3 Check - Rank {self.accelerator_manager.get_state_property(property_name='process_index')}] Prepared DataLoader length is {prepared_dataloader_len}, expected {expected_batches_per_process}"

            # if self.accelerator_manager.get_state_property(
            #     property_name="is_main_process"
            # ):
            #     print(
            #         f"[DEBUG H3] Prepared DataLoader len (per process): {prepared_dataloader_len}. Expected per process: {expected_batches_per_process}"
            #     )
            # # <<< END INSERTION H3 >>>

            self.data_manager.dataloaders["verifier"]["eval_dataloader"] = (
                self.accelerator_manager.prepare_dataloader(
                    eval_dataloader, key="verifier"
                )
            )  # separately, hopefully doesn't break anything

            # If reference model is not None, prepare it
            if self.model_manager.ref_models["verifier"] is not None:
                self.model_manager.ref_models["verifier"] = (
                    self.accelerator_manager.prepare_ref_model(
                        key="verifier", model=self.model_manager.ref_models["verifier"]
                    )
                )

            self.optimizer_scheduler_manager._calculate_num_training_steps(
                components[2]
            )  # Aware of the quirk that the scheduler wants a prepared dataloader to calculate the number of training steps
            self.optimizer_scheduler_manager.create_schedulers()
            scheduler = self.optimizer_scheduler_manager.get_scheduler("verifier")
            scheduler = self.accelerator_manager.prepare_scheduler(
                key="verifier", scheduler=scheduler
            )

            self.optimizer_scheduler_manager.schedulers["verifier"] = scheduler

            if self.model_manager.ref_models["verifier"] is not None:
                self.model_manager.ref_models["verifier"] = (
                    self.accelerator_manager.prepare_ref_model(
                        key="verifier", model=self.model_manager.ref_models["verifier"]
                    )
                )
        else:
            logger.info("Preparing components for Prover phase...")
            self.model_manager.load_models()  # This will load the model for the current phase
            # Log the models
            logger.info(f"Models loaded: {self.model_manager.models}")
            logger.info(f"Ref models loaded: {self.model_manager.ref_models}")

            # Create optimizers
            self.optimizer_scheduler_manager.create_optimizers()

            for model_key in ["honest_prover", "sneaky_prover"]:
                train_dataloader = self.data_manager.dataloaders["provers"][model_key][
                    "train_dataloader"
                ]
                eval_dataloader = self.data_manager.dataloaders["provers"][model_key][
                    "eval_dataloader"
                ]

                model = self.model_manager.get_model(model_key, prepared=False)
                optimizer = self.optimizer_scheduler_manager.get_optimizer(model_key)
                components = self.accelerator_manager.prepare_components(
                    key=model_key,
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    model=model,
                )
                # Unpack :  model, optimizer, dataloader
                self.model_manager.prepared_models[model_key] = components[
                    0
                ]  # Quirk of prepare_components: returns a tuple of (model, optimizer, dataloader)
                self.optimizer_scheduler_manager.optimizers[model_key] = components[1]
                self.data_manager.dataloaders["provers"][model_key][
                    "train_dataloader"
                ] = components[2]
                self.data_manager.dataloaders["provers"][model_key][
                    "eval_dataloader"
                ] = self.accelerator_manager.prepare_dataloader(
                    eval_dataloader, key=model_key
                )
                # If reference model is not None, prepare it
                if self.model_manager.ref_models[model_key] is not None:
                    self.model_manager.ref_models[model_key] = (
                        self.accelerator_manager.prepare_ref_model(
                            key=model_key,
                            model=self.model_manager.ref_models[model_key],
                        )
                    )
                self.optimizer_scheduler_manager._calculate_num_training_steps(
                    components[2]
                )  # Aware of the quirk that the scheduler wants a prepared dataloader to calculate the number of training steps

            self.optimizer_scheduler_manager.create_schedulers()  # NOTE: Outside of loop because it creates for both provers

            for model_key in ["honest_prover", "sneaky_prover"]:
                scheduler = self.optimizer_scheduler_manager.get_scheduler(model_key)
                scheduler = self.accelerator_manager.prepare_scheduler(
                    key=model_key, scheduler=scheduler
                )
                self.optimizer_scheduler_manager.schedulers[model_key] = scheduler

                if self.model_manager.ref_models[model_key] is not None:
                    self.model_manager.ref_models[model_key] = (
                        self.accelerator_manager.prepare_ref_model(
                            key=model_key,
                            model=self.model_manager.ref_models[model_key],
                        )
                    )

    def _run_verifier_phase(self) -> None:

        verifier_trainer = VerifierRegressorTrainer(
            self.args,
            self.model_manager,
            self.data_manager,
            self.accelerator_manager,
            self.optimizer_scheduler_manager,
            self.metrics_logger,
            self.vllm_orchestrator,
            self.state_tracker,
        )
        # <<< START INSERTION H4 (Orchestrator) >>>
        # Get the dataloader that *should* be used by the trainer
        dataloader_in_datamanager = self.data_manager.dataloaders["verifier"][
            "train_dataloader"
        ]
        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            print(
                f"[DEBUG H4 Orchestrator] Dataloader stored in DataManager: id={id(dataloader_in_datamanager)}, len={len(dataloader_in_datamanager)}"
            )
        # <<< END INSERTION H4 (Orchestrator) >>>

        verifier_trainer.train(1)

    def _run_prover_phase(self) -> None:

        prover_trainer = ProverTrainer(
            self.args,
            self.model_manager,
            self.data_manager,
            self.accelerator_manager,
            self.optimizer_scheduler_manager,
            self.metrics_logger,
            self.vllm_orchestrator,
            self.state_tracker,
            self.dataset_type,
            self.grpo,
        )
        prover_trainer.train()

        # PLAN:
        # 1. Swap the verifier model in the vllm orchestrator with the new one (done in VerifierRegressorTrainer - And broadly, should be done in any VerifierTrainerXXX class) --> Run inference for round i training with round i verifier model
        # 2. Instantiate the provers
        # 3. Run prover training


# __init__: Stores ExperimentArgs and references to all shared manager components.
# run_training_rounds(): Main loop over num_rounds. Calls _reset_state_for_new_round(), _run_verifier_phase(), _run_prover_phase() in sequence.
# _run_verifier_phase(round_num): Reads args.verifier_training_type. Instantiates the corresponding VerifierTrainerXXX class, passing necessary configs and shared managers. Calls verifier_trainer.train().
# _run_prover_phase(round_num): Instantiates ProverTrainer, passing necessary configs and shared managers (including the now-trained verifier model access via ModelManager). Calls prover_trainer.train().
# _reset_state_for_new_round(): Calls model_manager.reinitialize_models(). Potentially resets optimizer states if needed (or relies on trainer re-instantiation).
