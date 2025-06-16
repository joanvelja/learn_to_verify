import logging
from typing_extensions import override
from pvg.orchestrator.phase_strategy import PhaseStrategy
from pvg.trainers.prover_trainer import ProverTrainer

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class ProverPhaseStrategy(PhaseStrategy):
    """Strategy for handling prover training phase."""

    @override
    def get_models_to_cleanup(self) -> list[str]:
        return ["verifier"]

    @override
    def get_components_to_cleanup(self) -> list[str]:
        return ["verifier"]

    @override
    def prepare_phase_components(self) -> None:
        """Prepare prover components for training."""
        logger.info("Preparing components for Prover phase...")

        # Load and prepare models
        self.model_manager.load_models()
        logger.info(f"Models loaded: {self.model_manager.models}")
        logger.info(f"Ref models loaded: {self.model_manager.ref_models}")

        # Create optimizers
        self.optimizer_scheduler_manager.create_optimizers()

        # Prepare each prover model
        for model_key in ["sneaky_prover"]:
            self._prepare_prover_model(model_key)

        # Create schedulers for all provers
        self.optimizer_scheduler_manager.create_schedulers()

        # Prepare schedulers for each prover
        for model_key in ["sneaky_prover"]:
            self._prepare_prover_scheduler(model_key)

    def _prepare_prover_model(self, model_key: str) -> None:
        """Prepare a single prover model and its components."""
        train_dataloader = self.data_manager.dataloaders["provers"][model_key][
            "train_dataloader"
        ]
        eval_dataloader = self.data_manager.dataloaders["provers"][model_key][
            "eval_dataloader"
        ]

        model = self.model_manager.get_model(model_key, prepared=False)
        optimizer = self.optimizer_scheduler_manager.get_optimizer(model_key)

        # Debug logging for model-optimizer preparation
        self._debug_model_optimizer_preparation(model_key, model, optimizer)

        # Prepare components
        components = self.accelerator_manager.prepare_components(
            key=model_key,
            dataloader=train_dataloader,
            optimizer=optimizer,
            model=model,
        )

        # If reference model is not None, prepare it
        if self.model_manager.ref_models[model_key] is not None:
            self.model_manager.ref_models[model_key] = (
                self.accelerator_manager.prepare_ref_model(
                    key=model_key,
                    model=self.model_manager.ref_models[model_key],
                )
            )

        # Debug post-preparation state
        self._debug_post_preparation_state(model_key, components, model, optimizer)

        # Store prepared components
        self.model_manager.prepared_models[model_key] = components[0]
        self.optimizer_scheduler_manager.optimizers[model_key] = components[1]
        self.data_manager.dataloaders["provers"][model_key]["train_dataloader"] = (
            components[2]
        )
        self.data_manager.dataloaders["provers"][model_key]["eval_dataloader"] = (
            self.accelerator_manager.prepare_dataloader(eval_dataloader, key=model_key)
        )

        # Calculate training steps for scheduler
        self.optimizer_scheduler_manager._calculate_num_training_steps(components[2])

    def _prepare_prover_scheduler(self, model_key: str) -> None:
        """Prepare scheduler for a prover model."""
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

    def _debug_model_optimizer_preparation(
        self, model_key: str, model, optimizer
    ) -> None:
        """Debug logging for model-optimizer preparation."""
        if self.accelerator_manager.get_state_property("is_main_process"):
            logger.info("=" * 80)
            logger.info("🔍 MODEL-OPTIMIZER PREPARATION DEBUG")
            logger.info("=" * 80)

            # Pre-preparation state
            logger.info(f"🏷️  Unprepared Model ID: {id(model)}")
            logger.info(f"🏷️  Unprepared Model Type: {type(model)}")
            logger.info(f"🔧 Unprepared Optimizer ID: {id(optimizer)}")
            logger.info(f"🔧 Unprepared Optimizer Type: {type(optimizer)}")

            # Check optimizer parameter IDs
            unprepared_model_param_ids = {id(p) for p in model.parameters()}
            unprepared_optimizer_param_ids = {
                id(p) for group in optimizer.param_groups for p in group["params"]
            }

            logger.info(
                f"🔍 Unprepared model params count: {len(unprepared_model_param_ids)}"
            )
            logger.info(
                f"🔍 Unprepared optimizer params count: {len(unprepared_optimizer_param_ids)}"
            )
            logger.info(
                f"🔍 Pre-preparation match? {unprepared_model_param_ids == unprepared_optimizer_param_ids}"
            )

            # Sample parameter IDs
            logger.info("🔍 Sample parameter IDs (first 3):")
            model_params_list = list(model.parameters())
            optimizer_params_list = [
                p for group in optimizer.param_groups for p in group["params"]
            ]

            for i in range(min(3, len(model_params_list))):
                model_param_id = id(model_params_list[i])
                opt_param_id = id(optimizer_params_list[i])
                logger.info(
                    f"🔍   Param {i}: Model ID {model_param_id}, Optimizer ID {opt_param_id}, Match: {model_param_id == opt_param_id}"
                )

    def _debug_post_preparation_state(
        self, model_key: str, components, unprepared_model, unprepared_optimizer
    ) -> None:
        """Debug logging for post-preparation state."""
        if self.accelerator_manager.get_state_property("is_main_process"):
            logger.info("🔄 POST-PREPARATION ANALYSIS")
            logger.info("-" * 40)

            prepared_model = components[0]
            prepared_optimizer = components[1]

            logger.info(f"🏷️  Prepared Model ID: {id(prepared_model)}")
            logger.info(f"🏷️  Prepared Model Type: {type(prepared_model)}")
            logger.info(f"🔧 Prepared Optimizer ID: {id(prepared_optimizer)}")
            logger.info(f"🔧 Prepared Optimizer Type: {type(prepared_optimizer)}")

            # Check parameter IDs
            unprepared_model_param_ids = {id(p) for p in unprepared_model.parameters()}
            unprepared_optimizer_param_ids = {
                id(p)
                for group in unprepared_optimizer.param_groups
                for p in group["params"]
            }
            prepared_model_param_ids = {id(p) for p in prepared_model.parameters()}
            prepared_optimizer_param_ids = {
                id(p)
                for group in prepared_optimizer.param_groups
                for p in group["params"]
            }

            logger.info(
                f"🔍 Prepared model params count: {len(prepared_model_param_ids)}"
            )
            logger.info(
                f"🔍 Prepared optimizer params count: {len(prepared_optimizer_param_ids)}"
            )
            logger.info(
                f"🔍 Post-preparation match? {prepared_model_param_ids == prepared_optimizer_param_ids}"
            )

            # Check if parameters changed
            model_params_changed = (
                unprepared_model_param_ids != prepared_model_param_ids
            )
            optimizer_params_changed = (
                unprepared_optimizer_param_ids != prepared_optimizer_param_ids
            )

            logger.info(
                f"🔍 Model parameter IDs changed after preparation? {model_params_changed}"
            )
            logger.info(
                f"🔍 Optimizer parameter IDs changed after preparation? {optimizer_params_changed}"
            )

            # Diagnose the issue
            if model_params_changed and not optimizer_params_changed:
                logger.error(
                    "💥 FOUND THE BUG! Model parameters changed but optimizer parameters didn't!"
                )
                logger.error(
                    "💥 This means optimizer is still pointing to old unprepared model parameters!"
                )
            elif optimizer_params_changed and not model_params_changed:
                logger.error(
                    "💥 Unexpected: Optimizer parameters changed but model parameters didn't!"
                )
            elif model_params_changed and optimizer_params_changed:
                logger.info(
                    "✅ Both changed - this might be correct (need to check if they still match)"
                )
            else:
                logger.info("✅ Neither changed - this should be fine")

            # Check DeepSpeed wrapping
            if hasattr(prepared_model, "module"):
                logger.info(
                    f"🚀 DeepSpeed wrapped model - underlying module ID: {id(prepared_model.module)}"
                )
                underlying_param_ids = {
                    id(p) for p in prepared_model.module.parameters()
                }
                logger.info(
                    f"🚀 Underlying module params match optimizer? {underlying_param_ids == prepared_optimizer_param_ids}"
                )

    @override
    def create_trainer(self) -> ProverTrainer:
        """Create prover trainer."""
        return ProverTrainer(
            args=self.args,
            formatter=self.formatter,
            batch_evaluator=self.batch_evaluator,
            model_manager=self.model_manager,
            data_manager=self.data_manager,
            accelerator_manager=self.accelerator_manager,
            optimizer_scheduler_manager=self.optimizer_scheduler_manager,
            metrics_logger=self.metrics_logger,
            vllm_orchestrator=self.vllm_orchestrator,
            state_tracker=self.state_tracker,
            dataset_type=self.orchestrator.dataset_type,
            grpo=self.grpo,
        )
