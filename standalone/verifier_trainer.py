"""
Standalone Verifier Trainer Component
=====================================

A self-contained component for finetuning verifier models with sophisticated datamixing strategies.
This component reuses existing PVG abstractions while providing a simplified interface for
standalone verifier training.

Features:
- Two datamixing strategies:
  1. Sliding window: k% from current round, (1-k)% spread uniformly across previous rounds
  2. Full concatenation: All datasets from round 0 to current round
- Reuses existing VerifierDataset, VerifierRegressorTrainer, and other components
- Simplified configuration interface
- Memory-efficient training with cleanup
- Comprehensive logging and metrics

Usage:
    trainer = StandaloneVerifierTrainer(config)
    trainer.train(epochs=3)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

# Add the project root to the path to import PVG components
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pvg.components.accelerator_manager import AcceleratorManager  # noqa: E402
from pvg.components.metrics_logger import MetricsLogger  # noqa: E402
from pvg.components.model_manager import ModelManager  # noqa: E402
from pvg.components.optimizer_manager import OptimizerSchedulerManager  # noqa: E402
from pvg.components.state_tracker import StateTracker  # noqa: E402
from pvg.config.args import DatasetArgs, ExperimentArgs, ModelArgs, TrainingArgs, WandbArgs  # noqa: E402
from pvg.data.dataset import VerifierDataset  # noqa: E402
from pvg.trainers.verifier_regressor import VerifierRegressorTrainer  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class StandaloneVerifierConfig:
    """Simplified configuration for standalone verifier training."""

    # Model configuration
    model_name_or_path: str = field(metadata={"help": "Path to pretrained verifier model"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer path (defaults to model path)"}
    )

    # Dataset configuration
    dataset_type: Literal["coding", "math"] = field(
        default="coding", metadata={"help": "Type of dataset: coding or math"}
    )
    dataset_size: str = field(default="full", metadata={"help": "Dataset size identifier"})
    current_round: int = field(default=0, metadata={"help": "Current training round"})

    # Datamixing strategy
    datamix_strategy: Literal["sliding_window", "full_concatenation"] = field(
        default="sliding_window", metadata={"help": "Datamixing strategy to use"}
    )

    # Sliding window parameters (for strategy 1)
    max_rounds_to_keep: int = field(default=3, metadata={"help": "Maximum number of rounds to keep in sliding window"})
    new_sample_weight_target: float = field(
        default=0.8, metadata={"help": "Target weight for samples from current round (k parameter)"}
    )
    always_include_round0: bool = field(
        default=False, metadata={"help": "Always include round 0 even if outside window"}
    )

    # Training parameters
    learning_rate: float = field(default=5e-6)
    batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    num_epochs: int = field(default=3)
    warmup_steps: int = field(default=100)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.01)

    # Verifier-specific parameters
    lambda_reg: float = field(default=0.05, metadata={"help": "Regularization strength for Bradley-Terry loss"})

    # System parameters
    output_dir: str = field(default="./verifier_output")
    seed: int = field(default=42)
    mixed_precision: Literal["no", "fp16", "bf16"] = field(default="bf16")
    gradient_checkpointing: bool = field(default=True)

    # Logging
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=500)
    log_level: str = field(default="INFO")

    # Wandb (optional)
    use_wandb: bool = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_entity: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        if not (0.0 < self.new_sample_weight_target <= 1.0):
            raise ValueError("new_sample_weight_target must be in (0.0, 1.0]")

        if self.max_rounds_to_keep < 1:
            raise ValueError("max_rounds_to_keep must be at least 1")

        if self.current_round < 0:
            raise ValueError("current_round must be non-negative")

        os.makedirs(self.output_dir, exist_ok=True)


class DatamixFactory:
    """Factory for creating datasets with different mixing strategies."""

    @staticmethod
    def create_sliding_window_dataset(
        config: StandaloneVerifierConfig, split: Literal["train", "eval"], tokenizer: AutoTokenizer
    ) -> VerifierDataset:
        """Create dataset using sliding window strategy (original logic)."""
        return VerifierDataset(
            current_round_num=config.current_round,
            max_rounds_to_keep=config.max_rounds_to_keep,
            new_sample_weight_target=config.new_sample_weight_target,
            dataset_type=config.dataset_type,
            dataset_size=config.dataset_size,
            split=split,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seed=config.seed,
            always_include_round0=config.always_include_round0,
            shuffle_within_epoch=True,
        )

    @staticmethod
    def create_full_concatenation_dataset(
        config: StandaloneVerifierConfig, split: Literal["train", "eval"], tokenizer: AutoTokenizer
    ) -> VerifierDataset:
        """Create dataset using full concatenation strategy (all rounds 0 to i)."""
        return VerifierDataset(
            current_round_num=config.current_round,
            max_rounds_to_keep=config.current_round + 1,  # Include all rounds from 0 to current
            new_sample_weight_target=1.0 / (config.current_round + 1),  # Equal weight for all rounds
            dataset_type=config.dataset_type,
            dataset_size=config.dataset_size,
            split=split,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seed=config.seed,
            always_include_round0=True,  # Always include round 0
            shuffle_within_epoch=True,
        )

    @classmethod
    def create_dataset(
        cls, config: StandaloneVerifierConfig, split: Literal["train", "eval"], tokenizer: AutoTokenizer
    ) -> VerifierDataset:
        """Factory method to create dataset based on strategy."""
        if config.datamix_strategy == "sliding_window":
            return cls.create_sliding_window_dataset(config, split, tokenizer)
        elif config.datamix_strategy == "full_concatenation":
            return cls.create_full_concatenation_dataset(config, split, tokenizer)
        else:
            raise ValueError(f"Unknown datamix strategy: {config.datamix_strategy}")


class StandaloneVerifierTrainer:
    """
    Standalone trainer for verifier models with sophisticated datamixing.

    This class provides a simplified interface for training verifier models while
    reusing the robust components from the PVG codebase. It supports two datamixing
    strategies and handles all the complexity of component initialization and cleanup.
    """

    def __init__(self, config: StandaloneVerifierConfig):
        """Initialize the standalone verifier trainer."""
        self.config = config
        self.setup_logging()

        logger.info("Initializing StandaloneVerifierTrainer")
        logger.info(f"Configuration: {config}")

        # Initialize core components
        self.tokenizer = self._load_tokenizer()
        self.train_dataset, self.eval_dataset = self._create_datasets()

        # Convert to PVG config format
        self.experiment_args = self._create_experiment_args()

        # Initialize PVG components
        self._initialize_components()

        # Create trainer
        self.trainer = self._create_trainer()

        logger.info("StandaloneVerifierTrainer initialization complete")

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config.output_dir, "training.log")),
            ],
        )

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer from {self.config.tokenizer_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        return tokenizer

    def _create_datasets(self) -> tuple[VerifierDataset, VerifierDataset]:
        """Create train and eval datasets using the configured strategy."""
        logger.info(f"Creating datasets using {self.config.datamix_strategy} strategy")

        train_dataset = DatamixFactory.create_dataset(self.config, "train", self.tokenizer)
        eval_dataset = DatamixFactory.create_dataset(self.config, "eval", self.tokenizer)

        # Log dataset statistics
        logger.info("=== DATASET STATISTICS ===")
        logger.info(f"Train dataset length: {len(train_dataset)}")
        logger.info(f"Train dataset stats: {train_dataset.get_round_statistics()}")
        logger.info(f"Eval dataset length: {len(eval_dataset)}")
        logger.info(f"Eval dataset stats: {eval_dataset.get_round_statistics()}")
        logger.info("==========================")

        return train_dataset, eval_dataset

    def _create_experiment_args(self) -> ExperimentArgs:
        """Convert standalone config to PVG ExperimentArgs format."""
        model_args = ModelArgs(
            name_or_path=self.config.model_name_or_path,
            torch_dtype="auto",
            use_flash_attention=True,
            attn_implementation="flash_attention_2",
            device_map=None,
        )

        training_args = TrainingArgs(
            seed=self.config.seed,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type="linear",
            num_warmup_steps=self.config.warmup_steps,
            verifier_mode="regressor",
            gradient_checkpointing=self.config.gradient_checkpointing,
            batch_size=self.config.batch_size,
        )

        dataset_args = DatasetArgs(
            tokenizer_name_or_path=self.config.tokenizer_name_or_path,
            dataset_name=f"jvelja/{self.config.dataset_type}_dataset",
            dataset_size=self.config.dataset_size,
        )

        wandb_args = WandbArgs(
            use_wandb=self.config.use_wandb,
            wandb_project_name=self.config.wandb_project,
            wandb_entity=self.config.wandb_entity,
            wandb_run_name=self.config.wandb_run_name,
            output_dir=self.config.output_dir,
        )

        return ExperimentArgs(
            verifier=model_args,
            sneaky_prover=model_args,  # Dummy, not used
            dataset=dataset_args,
            training_verifier=training_args,
            training_sneaky_prover=training_args,  # Dummy, not used
            wandb=wandb_args,
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            mixed_precision=self.config.mixed_precision,
            num_processes=1,  # Single GPU for standalone
        )

    def _initialize_components(self):
        """Initialize PVG components for training."""
        logger.info("Initializing PVG components")

        # Create a simple accelerator manager
        self.accelerator_manager = self._create_simple_accelerator_manager()

        # State tracker
        self.state_tracker = StateTracker(phase="verifier", round_num=self.config.current_round, step=0)

        # Metrics logger
        self.metrics_logger = MetricsLogger(
            output_dir=self.config.output_dir,
            wandb_config=self.experiment_args.wandb,
            global_step_callback=lambda: self.state_tracker.step,
        )

        # Model manager
        self.model_manager = self._create_model_manager()

        # Optimizer manager
        self.optimizer_manager = self._create_optimizer_manager()

        logger.info("PVG components initialized successfully")

    def _create_simple_accelerator_manager(self) -> AcceleratorManager:
        """Create a simplified accelerator manager for standalone use."""
        # Create a single accelerator for verifier training
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
        )

        # Create a mock accelerator manager that mimics the interface
        class SimpleAcceleratorManager:
            def __init__(self, accelerator):
                self.accelerator = accelerator
                self.accelerators = {"verifier": accelerator}

            def get_accelerator(self, key: str) -> Accelerator:
                return self.accelerator

            def get_state_property(self, property_name: str):
                if property_name == "is_main_process":
                    return self.accelerator.is_main_process
                elif property_name == "num_processes":
                    return self.accelerator.num_processes
                elif property_name == "process_index":
                    return self.accelerator.process_index
                else:
                    return getattr(self.accelerator.state, property_name)

            def prepare_components(self, key: str, model, optimizer, dataloader):
                return self.accelerator.prepare(model, optimizer, dataloader)

            def prepare_dataloader(self, dataloader, key: str):
                return self.accelerator.prepare(dataloader)

            def backward(self, loss, key: str):
                self.accelerator.backward(loss)

            def wait_for_everyone(self):
                self.accelerator.wait_for_everyone()

            def unwrap_model(self, model, key: str):
                return self.accelerator.unwrap_model(model)

        return SimpleAcceleratorManager(accelerator)

    def _create_model_manager(self) -> ModelManager:
        """Create model manager with verifier model."""

        class SimpleModelManager:
            def __init__(self, config: StandaloneVerifierConfig, accelerator_manager):
                self.config = config
                self.accelerator_manager = accelerator_manager
                self.models = {}
                self.prepared_models = {}
                self.ref_models = {"verifier": None}

            def load_models(self):
                """Load the verifier model."""
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name_or_path,
                    torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else "auto",
                    device_map="auto",
                    trust_remote_code=True,
                )

                if self.config.gradient_checkpointing:
                    model.gradient_checkpointing_enable()

                self.models["verifier"] = model
                logger.info(f"Loaded verifier model: {self.config.model_name_or_path}")

            def get_model(self, key: str, prepared: bool = False):
                if prepared:
                    return self.prepared_models[key]
                return self.models[key]

            def prepare_model(self, key: str, optimizer, dataloader):
                """Prepare model with accelerator."""
                model = self.models[key]
                prepared = self.accelerator_manager.prepare_components(key, model, optimizer, dataloader)
                self.prepared_models[key] = prepared[0]
                return prepared

        return SimpleModelManager(self.config, self.accelerator_manager)

    def _create_optimizer_manager(self) -> OptimizerSchedulerManager:
        """Create optimizer and scheduler manager."""

        class SimpleOptimizerManager:
            def __init__(self, config: StandaloneVerifierConfig):
                self.config = config
                self.optimizers = {}
                self.schedulers = {}

            def create_optimizers(self):
                """Create optimizer for verifier."""
                # This will be called after model is prepared
                pass

            def create_optimizer_for_model(self, model):
                """Create optimizer for a specific model."""
                from torch.optim import AdamW

                optimizer = AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                return optimizer

            def create_schedulers(self):
                """Create learning rate schedulers."""
                from transformers import get_linear_schedule_with_warmup

                if "verifier" in self.optimizers:
                    total_steps = self._calculate_total_steps()
                    scheduler = get_linear_schedule_with_warmup(
                        self.optimizers["verifier"],
                        num_warmup_steps=self.config.warmup_steps,
                        num_training_steps=total_steps,
                    )
                    self.schedulers["verifier"] = scheduler

            def _calculate_total_steps(self) -> int:
                """Calculate total training steps."""
                # This is a rough estimate - in practice would be calculated from dataloader
                return 1000  # Placeholder

            def get_optimizer(self, key: str):
                return self.optimizers[key]

            def get_scheduler(self, key: str):
                return self.schedulers.get(key)

        return SimpleOptimizerManager(self.config)

    def _create_trainer(self) -> VerifierRegressorTrainer:
        """Create the verifier trainer using existing PVG implementation."""
        # Prepare model and optimizer
        self.model_manager.load_models()
        model = self.model_manager.get_model("verifier")
        optimizer = self.optimizer_manager.create_optimizer_for_model(model)

        # Prepare components with accelerator
        train_dataloader = self.train_dataset.get_dataloader()
        eval_dataloader = self.eval_dataset.get_dataloader()

        prepared = self.accelerator_manager.prepare_components("verifier", model, optimizer, train_dataloader)

        # Store prepared components
        self.model_manager.prepared_models["verifier"] = prepared[0]
        self.optimizer_manager.optimizers["verifier"] = prepared[1]
        prepared_train_dataloader = prepared[2]

        prepared_eval_dataloader = self.accelerator_manager.prepare_dataloader(eval_dataloader, "verifier")

        # Create scheduler
        self.optimizer_manager.create_schedulers()

        # Create a mock data manager for the trainer interface
        class MockDataManager:
            def __init__(self, tokenizer, train_dl, eval_dl):
                self.tokenizer = tokenizer
                self.dataloaders = {
                    "verifier": {
                        "train_dataloader": train_dl,
                        "eval_dataloader": eval_dl,
                    }
                }

        data_manager = MockDataManager(self.tokenizer, prepared_train_dataloader, prepared_eval_dataloader)

        # Create a mock VLLM orchestrator (not needed for training)
        class MockVLLMOrchestrator:
            def sync_weights(self, phase, model_manager):
                pass

        vllm_orchestrator = MockVLLMOrchestrator()

        # Create the trainer
        trainer = VerifierRegressorTrainer(
            args=self.experiment_args,
            model_manager=self.model_manager,
            data_manager=data_manager,
            accelerator_manager=self.accelerator_manager,
            optimizer_scheduler_manager=self.optimizer_manager,
            metrics_logger=self.metrics_logger,
            vllm_orchestrator=vllm_orchestrator,
            state_tracker=self.state_tracker,
        )

        # Override some trainer config with our values
        trainer.config["lambda_reg"] = self.config.lambda_reg
        trainer.config["push_to_hub"] = False  # Don't push in standalone mode

        return trainer

    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the verifier model.

        Args:
            epochs: Number of epochs to train (overrides config if provided)

        Returns:
            Training metrics and results
        """
        epochs = epochs or self.config.num_epochs

        logger.info(f"Starting verifier training for {epochs} epochs")
        logger.info(f"Datamix strategy: {self.config.datamix_strategy}")
        logger.info(f"Current round: {self.config.current_round}")

        try:
            # Start training
            self.trainer.train(epochs)

            logger.info("Training completed successfully")

            # Get final metrics
            final_metrics = self.trainer.evaluate()

            return {
                "status": "completed",
                "epochs": epochs,
                "final_metrics": final_metrics,
                "config": self.config,
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

        finally:
            self.cleanup()

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the current model."""
        logger.info("Running evaluation")
        return self.trainer.evaluate()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources")

        if hasattr(self, "train_dataset"):
            self.train_dataset.cleanup()
        if hasattr(self, "eval_dataset"):
            self.eval_dataset.cleanup()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleanup completed")


# Example usage and testing
def main():
    """Example usage of the StandaloneVerifierTrainer."""

    # Example 1: Sliding window strategy
    _ = StandaloneVerifierConfig(
        model_name_or_path="microsoft/DialoGPT-medium",  # Example model
        dataset_type="coding",
        current_round=6,
        datamix_strategy="sliding_window",
        max_rounds_to_keep=10,
        new_sample_weight_target=0.8,
        batch_size=4,
        num_epochs=1,
        output_dir="./output_sliding_window",
        use_wandb=False,
    )

    # Example 2: Full concatenation strategy
    _ = StandaloneVerifierConfig(
        model_name_or_path="microsoft/DialoGPT-medium",  # Example model
        dataset_type="coding",
        current_round=2,
        datamix_strategy="full_concatenation",
        batch_size=4,
        num_epochs=1,
        output_dir="./output_concatenation",
        use_wandb=False,
    )

    print("=== Standalone Verifier Trainer Examples ===")
    print("\nExample 1: Sliding Window Strategy")
    print("- Uses 80% samples from current round (round 2)")
    print("- Uses 20% samples spread across previous rounds (0, 1)")
    print("- Keeps maximum 3 rounds in sliding window")

    print("\nExample 2: Full Concatenation Strategy")
    print("- Uses equal weight from all rounds 0 to current (0, 1, 2)")
    print("- Each round contributes ~33.3% of the data")

    print("\nTo run training:")
    print("trainer = StandaloneVerifierTrainer(config_sliding)")
    print("results = trainer.train()")

    # Uncomment to actually run training (requires valid model and data)
    # trainer = StandaloneVerifierTrainer(config_sliding)
    # results = trainer.train()
    # print(f"Training results: {results}")


if __name__ == "__main__":
    main()
