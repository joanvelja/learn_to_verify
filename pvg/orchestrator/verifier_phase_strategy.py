import logging
from typing import Any

from pvg.data.dataset import VerifierDataset
from pvg.orchestrator.phase_strategy import PhaseStrategy
from pvg.trainers.verifier_base import VerifierTrainerBase
from pvg.trainers.verifier_classifier import VerifierClassifierTrainer
from pvg.trainers.verifier_regressor import VerifierRegressorTrainer

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class VerifierPhaseStrategy(PhaseStrategy):
    """Strategy for handling verifier training phase."""

    def get_models_to_cleanup(self) -> list[str]:
        return ["sneaky_prover"]

    def get_components_to_cleanup(self) -> list[str]:
        return ["sneaky_prover"]

    def prepare_phase_components(self) -> None:
        """Prepare verifier components for training."""
        logger.info("Preparing components for Verifier phase...")

        # Create datasets
        # class VerifierDataset(
        #     current_round_num: int,
        #     max_rounds_to_keep: int = 3,
        #     new_sample_weight_target: float = 0.8,
        #     dataset_type: Literal['coding', 'math'] = "coding",
        #     dataset_size: str = "full",
        #     split: Literal['train', 'eval'] = "train",
        #     tokenizer: Unknown | None = None,
        #     batch_size: int = 32,
        #     seed: int = 42,
        #     epoch_size: int | None = None,
        #     always_include_round0: bool = False,
        #     shuffle_within_epoch: bool = True,
        #     problem_key: str = "problem",
        #     honest_key: str = "honest_solution",
        #     injected_key: str = "injected_solution"
        # )

        self.orchestrator.verifier_dataset_train = VerifierDataset(
            current_round_num=self.state_tracker.round,
            max_rounds_to_keep=10,
            new_sample_weight_target=0.5,
            seed=42,
            dataset_type=self.orchestrator.dataset_type,
            tokenizer=self.data_manager.tokenizer,
            split="train",
        )
        self.orchestrator.verifier_dataset_eval = VerifierDataset(
            current_round_num=self.state_tracker.round,
            max_rounds_to_keep=10,
            new_sample_weight_target=0.5,
            seed=42,
            dataset_type=self.orchestrator.dataset_type,
            tokenizer=self.data_manager.tokenizer,
            split="eval",
        )
        logger.info("! VERIFIER DATASET DIAGNOSTICS !")
        logger.info(f"Verifier dataset train: {self.orchestrator.verifier_dataset_train}")
        logger.info(f"Verifier dataset train length: {len(self.orchestrator.verifier_dataset_train)}")
        logger.info(f"Verifier dataset train stats: {self.orchestrator.verifier_dataset_train.get_round_statistics()}")
        logger.info(f"Verifier dataset eval: {self.orchestrator.verifier_dataset_eval}")
        logger.info(f"Verifier dataset eval length: {len(self.orchestrator.verifier_dataset_eval)}")
        logger.info(f"Verifier dataset eval stats: {self.orchestrator.verifier_dataset_eval.get_round_statistics()}")

        # Prepare dataloaders
        train_dataloader = self.orchestrator.verifier_dataset_train.get_dataloader()
        eval_dataloader = self.orchestrator.verifier_dataset_eval.get_dataloader()

        # Load and prepare components
        self.model_manager.load_models()
        self.optimizer_scheduler_manager.create_optimizers()

        model = self.model_manager.get_model("verifier", prepared=False)
        optimizer = self.optimizer_scheduler_manager.get_optimizer("verifier")

        # Prepare components with accelerator
        components = self._prepare_model_components("verifier", train_dataloader, optimizer, model)
        self._store_prepared_components("verifier", components, eval_dataloader)

        # Handle reference model
        self._prepare_reference_model("verifier")

        # Create and prepare scheduler
        self._prepare_scheduler("verifier", components[2])

    def _prepare_model_components(self, key: str, dataloader, optimizer, model) -> tuple[Any, Any, Any]:
        """Prepare model components with accelerator."""
        return self.accelerator_manager.prepare_components(
            key=key,
            dataloader=dataloader,
            optimizer=optimizer,
            model=model,
        )

    def _store_prepared_components(self, key: str, components: tuple[Any, Any, Any], eval_dataloader) -> None:
        """Store prepared components in managers."""
        self.model_manager.prepared_models[key] = components[0]
        self.optimizer_scheduler_manager.optimizers[key] = components[1]

        # Get the verifier mode to store dataloaders in the correct location
        verifier_mode = self.args.training_verifier.verifier_mode

        # Store dataloaders based on verifier mode
        if verifier_mode == "regressor":
            self.data_manager.dataloaders[key]["train_dataloader"] = components[2]
            self.data_manager.dataloaders[key]["eval_dataloader"] = self.accelerator_manager.prepare_dataloader(
                eval_dataloader, key=key
            )
        else:  # classifier, inference_classifier, inference_regressor
            self.data_manager.dataloaders[key][verifier_mode]["train_dataloader"] = components[2]
            self.data_manager.dataloaders[key][verifier_mode]["eval_dataloader"] = (
                self.accelerator_manager.prepare_dataloader(eval_dataloader, key=key)
            )

    def _prepare_reference_model(self, key: str) -> None:
        """Prepare reference model if it exists."""
        if self.model_manager.ref_models[key] is not None:
            self.model_manager.ref_models[key] = self.accelerator_manager.prepare_ref_model(
                key=key, model=self.model_manager.ref_models[key]
            )

    def _prepare_scheduler(self, key: str, prepared_dataloader) -> None:
        """Create and prepare scheduler."""
        self.optimizer_scheduler_manager._calculate_num_training_steps(prepared_dataloader)
        self.optimizer_scheduler_manager.create_schedulers()
        scheduler = self.optimizer_scheduler_manager.get_scheduler(key)
        scheduler = self.accelerator_manager.prepare_scheduler(key=key, scheduler=scheduler)
        self.optimizer_scheduler_manager.schedulers[key] = scheduler

    def create_trainer(self) -> VerifierTrainerBase:
        """Create verifier trainer based on verifier mode."""
        trainer_args = {
            "args": self.args,
            "model_manager": self.model_manager,
            "data_manager": self.data_manager,
            "accelerator_manager": self.accelerator_manager,
            "optimizer_scheduler_manager": self.optimizer_scheduler_manager,
            "metrics_logger": self.metrics_logger,
            "vllm_orchestrator": self.vllm_orchestrator,
            "state_tracker": self.state_tracker,
        }

        verifier_mode = self.args.training_verifier.verifier_mode

        if verifier_mode in ["regressor", "inference_regressor"]:
            return VerifierRegressorTrainer(**trainer_args)
        elif verifier_mode in ["classifier", "inference_classifier"]:
            return VerifierClassifierTrainer(**trainer_args)
        else:
            raise ValueError(f"Unknown verifier mode: {verifier_mode}")
