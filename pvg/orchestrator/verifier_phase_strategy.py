import logging
from typing import Any

from pvg.data.dataset import VerifierDataset
from pvg.orchestrator.phase_strategy import PhaseStrategy
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
        self.orchestrator.verifier_dataset_train = VerifierDataset(
            current_round_num=self.state_tracker.round,
            max_rounds_to_keep=10,
            new_sample_weight_target=0.8,
            batch_size=self.args.training_verifier.batch_size,
            seed=42,
            dataset_type=self.orchestrator.dataset_type,
            correct_column_identifier="honest_solution",
            incorrect_column_identifier="injected_solution",
            tokenizer=self.data_manager.tokenizer,
            split="train",
        )
        self.orchestrator.verifier_dataset_eval = VerifierDataset(
            current_round_num=self.state_tracker.round,
            max_rounds_to_keep=10,
            new_sample_weight_target=0.8,
            batch_size=self.args.training_verifier.batch_size,
            seed=42,
            dataset_type=self.orchestrator.dataset_type,
            correct_column_identifier="honest_solution",
            incorrect_column_identifier="injected_solution",
            tokenizer=self.data_manager.tokenizer,
            split="eval",
        )

        # Prepare dataloaders
        train_dataloader = self.orchestrator.verifier_dataset_train.get_dataloader()
        eval_dataloader = self.orchestrator.verifier_dataset_eval.get_dataloader()

        # Load and prepare components
        self.model_manager.load_models()
        self.optimizer_scheduler_manager.create_optimizers()

        model = self.model_manager.get_model("verifier", prepared=False)
        optimizer = self.optimizer_scheduler_manager.get_optimizer("verifier")

        # Prepare components with accelerator
        components = self._prepare_model_components(
            "verifier", train_dataloader, optimizer, model
        )
        self._store_prepared_components("verifier", components, eval_dataloader)

        # Handle reference model
        self._prepare_reference_model("verifier")

        # Create and prepare scheduler
        self._prepare_scheduler("verifier", components[2])

    def _prepare_model_components(
        self, key: str, dataloader, optimizer, model
    ) -> tuple[Any, Any, Any]:
        """Prepare model components with accelerator."""
        return self.accelerator_manager.prepare_components(
            key=key,
            dataloader=dataloader,
            optimizer=optimizer,
            model=model,
        )

    def _store_prepared_components(
        self, key: str, components: tuple[Any, Any, Any], eval_dataloader
    ) -> None:
        """Store prepared components in managers."""
        self.model_manager.prepared_models[key] = components[0]
        self.optimizer_scheduler_manager.optimizers[key] = components[1]
        self.data_manager.dataloaders[key]["train_dataloader"] = components[2]
        self.data_manager.dataloaders[key]["eval_dataloader"] = (
            self.accelerator_manager.prepare_dataloader(eval_dataloader, key=key)
        )

    def _prepare_reference_model(self, key: str) -> None:
        """Prepare reference model if it exists."""
        if self.model_manager.ref_models[key] is not None:
            self.model_manager.ref_models[key] = (
                self.accelerator_manager.prepare_ref_model(
                    key=key, model=self.model_manager.ref_models[key]
                )
            )

    def _prepare_scheduler(self, key: str, prepared_dataloader) -> None:
        """Create and prepare scheduler."""
        self.optimizer_scheduler_manager._calculate_num_training_steps(
            prepared_dataloader
        )
        self.optimizer_scheduler_manager.create_schedulers()
        scheduler = self.optimizer_scheduler_manager.get_scheduler(key)
        scheduler = self.accelerator_manager.prepare_scheduler(
            key=key, scheduler=scheduler
        )
        self.optimizer_scheduler_manager.schedulers[key] = scheduler

    def create_trainer(self) -> VerifierRegressorTrainer:
        """Create verifier trainer."""
        return VerifierRegressorTrainer(
            args=self.args,
            model_manager=self.model_manager,
            data_manager=self.data_manager,
            accelerator_manager=self.accelerator_manager,
            optimizer_scheduler_manager=self.optimizer_scheduler_manager,
            metrics_logger=self.metrics_logger,
            vllm_orchestrator=self.vllm_orchestrator,
            state_tracker=self.state_tracker,
        )
