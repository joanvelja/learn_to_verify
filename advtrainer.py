import logging
import os
from dataclasses import dataclass, field
from datasets import load_dataset

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from transformers import HfArgumentParser, set_seed


# --- Argument Dataclass (remains the same) ---
@dataclass
class ScriptArguments:
    """
    Arguments pertaining to the script setup, model configurations, and paths.
    """

    honest_prover_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier for Model A"}
    )
    sneaky_prover_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier for Model B"}
    )
    dataset_name: str = field(metadata={"help": "Name of the dataset to load"})
    ds_config_a: str = field(
        metadata={"help": "Path to the DeepSpeed config file for Model A."}
    )
    ds_config_b: str = field(
        metadata={"help": "Path to the DeepSpeed config file for Model B."}
    )
    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        }
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    logging_steps: int = field(
        default=100, metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(
        default=500, metadata={"help": "Run evaluation every X updates steps."}
    )
    max_train_steps: int | None = field(
        default=None,
        metadata={
            "help": "Total number of training steps to perform. If provided, overrides num_train_epochs."
        },
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate_a: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Model A's AdamW optimizer."},
    )
    learning_rate_b: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Model B's AdamW optimizer."},
    )
    mixed_precision: str = field(
        default="fp16", metadata={"help": "The mixed precision to use."}
    )
    # ... add other relevant training args ...


# --- Logger Setup ---
# Configure logging level based on main process rank later in __init__
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # Default level, will be adjusted
)


# --- The Trainer Class ---
class DisjointTrainer:
    def __init__(self, args: ScriptArguments):
        self.args = args
        self.accelerators: dict[str, Accelerator] = {}
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.dataloaders: dict[str, torch.utils.data.DataLoader] = {}
        self.deepspeed_plugins: dict[str, DeepSpeedPlugin] = {}

        # State variables
        self.global_step = 0
        self.current_epoch = 0

        self._setup_logging()  # Adjust logging level based on rank
        self._set_seed()
        self._initialize_accelerators()

        logger.info("DisjointTrainer initialized. Accelerators are set up.")
        logger.info(
            "Next steps: Call setup_components() to load data, models, optimizers and prepare them."
        )

    def _setup_logging(self):
        # Logging setup happens before accelerator init, so we check env vars
        # Note: This is a simplified check. Accelerator handles rank determination more robustly.
        # We'll refine logging levels *after* accelerator init for certainty.
        is_main_process = os.environ.get("RANK", "0") == "0"  # Basic check
        if is_main_process:
            level = logging.INFO
        else:
            level = logging.WARN
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            force=True,
        )
        logger.info(f"Logging level set to {logging.getLevelName(level)}")

    def _set_seed(self):
        set_seed(self.args.seed)
        logger.info(f"Set random seed to {self.args.seed}")

    def _initialize_accelerators(self):
        logger.info("Initializing Accelerators...")

        # --- Basic Sanity Checks ---
        if not os.path.exists(self.args.ds_config_a):
            raise FileNotFoundError(
                f"DeepSpeed config for Model A not found at {self.args.ds_config_a}"
            )
        if not os.path.exists(self.args.ds_config_b):
            raise FileNotFoundError(
                f"DeepSpeed config for Model B not found at {self.args.ds_config_b}"
            )
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Create DeepSpeed Plugins
        try:
            ds_plugin_a = DeepSpeedPlugin(hf_ds_config=self.args.ds_config_a)
            ds_plugin_b = DeepSpeedPlugin(hf_ds_config=self.args.ds_config_b)
            self.deepspeed_plugins = {
                "honest_prover": ds_plugin_a,
                "sneaky_prover": ds_plugin_b,
            }
            logger.info("DeepSpeed plugins created successfully.")
        except Exception as e:
            logger.error(f"Failed to create DeepSpeed plugins: {e}", exc_info=True)
            raise

        # Configure project/logging for Accelerate
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir,
            logging_dir=os.path.join(
                self.args.output_dir, "accelerate_logs"
            ),  # Changed dir name
        )

        # Instantiate the first accelerator
        try:
            # Use a temporary variable first
            _accelerator_a = Accelerator(
                deepspeed_plugin=self.deepspeed_plugins,
                log_with="wandb",  # Example tracker
                project_config=project_config,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                mixed_precision=self.args.mixed_precision,  # Add global config if needed
            )
            self.accelerators["honest_prover"] = (
                _accelerator_a  # Assign after successful init
            )
            logger.info(
                "First Accelerator (accelerators['honest_prover']) initialized."
            )

            # Adjust logging level based on the *actual* rank determined by Accelerator
            log_level = logging.INFO if _accelerator_a.is_main_process else logging.WARN
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                force=True,
            )
            logger.info(
                f"Logging level re-adjusted to {logging.getLevelName(log_level)} based on accelerator rank."
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize the first Accelerator: {e}", exc_info=True
            )
            raise

        # Instantiate the second accelerator
        try:
            # Use a temporary variable first
            _accelerator_b = Accelerator()
            self.accelerators["sneaky_prover"] = (
                _accelerator_b  # Assign after successful init
            )
            logger.info(
                "Second Accelerator (accelerators['sneaky_prover']) initialized."
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize the second Accelerator: {e}", exc_info=True
            )
            raise

        # Log accelerator state (using accelerator_a is sufficient)
        accel_state = self.accelerators["honest_prover"].state
        logger.info("--- Accelerator State ---")
        logger.info(f"  Distributed type: {accel_state.distributed_type}")
        logger.info(f"  Num processes: {accel_state.num_processes}")
        logger.info(f"  Process index: {accel_state.process_index}")
        logger.info(f"  Local process index: {accel_state.local_process_index}")
        logger.info(
            f"  Device: {self.accelerators['honest_prover'].device}"
        )  # Get device from instance
        logger.info(
            f"  Mixed precision: {self.accelerators['honest_prover'].mixed_precision}"
        )
        logger.info(
            f"  Gradient Acc Steps: {self.accelerators['honest_prover'].gradient_accumulation_steps}"
        )
        # Access accel_state for sneaky_prover too for sanity check
        if "sneaky_prover" in self.accelerators:
            accel_state_b = self.accelerators["sneaky_prover"].state
            logger.info("--- Accelerator State for Model B ---")
            logger.info(f"  Distributed type: {accel_state_b.distributed_type}")
            logger.info(f"  Num processes: {accel_state_b.num_processes}")
            logger.info(f"  Process index: {accel_state_b.process_index}")
            logger.info(f"  Local process index: {accel_state_b.local_process_index}")
            logger.info(
                f"  Device: {self.accelerators['sneaky_prover'].device}"
            )  # Get device from instance
            logger.info(
                f"  Mixed precision: {self.accelerators['sneaky_prover'].mixed_precision}"
            )
            logger.info(
                f"  Gradient Acc Steps: {self.accelerators['sneaky_prover'].gradient_accumulation_steps}"
            )
        else:
            logger.info(
                "Model B accelerator not initialized. If this is expected, ignore this message. If not, you may have not passed sneaky_prover_name_or_path correctly to the script."
            )

    def setup_components(self):
        """Loads data, models, optimizers, schedulers and prepares them."""
        logger.info("Setting up components (Data, Models, Optimizers, Schedulers)...")
        self._load_data()
        self._load_models()
        self._create_optimizers_schedulers()
        self._prepare_components()
        logger.info("Component setup complete.")

    def _load_data(self):
        # --- Placeholder 4: Load Datasets & Collators ---
        logger.info("Loading dataset(s)...")
        # Dataset loading logic
        raw_datasets = load_dataset(
            self.args.dataset_name, split=["train", "validation"]
        )
        self.train_dataset = raw_datasets[0]
        self.eval_dataset = raw_datasets[1]
        self.data_collator = None  # TODO: Implement data collator

        if self.train_dataset is None or self.data_collator is None:
            logger.warning(
                "Datasets and/or Data Collator not loaded (using placeholders). Implement _load_data."
            )
            # Create dummy dataloaders to avoid errors later if needed for testing structure
            dummy_dataset = [
                (torch.tensor([i]), torch.tensor([i + 1])) for i in range(100)
            ]
            self.dataloaders["train"] = torch.utils.data.DataLoader(
                dummy_dataset, batch_size=self.args.train_batch_size
            )
            self.dataloaders["eval"] = torch.utils.data.DataLoader(
                dummy_dataset, batch_size=self.args.eval_batch_size
            )
        else:
            # Create actual dataloaders
            # self.dataloaders["train"] = torch.utils.data.DataLoader(...)
            # self.dataloaders["eval"] = torch.utils.data.DataLoader(...)
            pass  # Implement dataloader creation here

    def _load_models(self):
        # --- Placeholder 5: Instantiate Models ---
        logger.info("Loading models...")
        # Replace with your actual model loading logic
        # Example:
        # from transformers import AutoModelForCausalLM
        # self.models["honest_prover"] = AutoModelForCausalLM.from_pretrained(self.args.honest_prover_name_or_path)
        # self.models["sneaky_prover"] = AutoModelForCausalLM.from_pretrained(self.args.sneaky_prover_name_or_path)
        from transformers import AutoModelForCausalLM  # Example import

        try:
            self.models["honest_prover"] = AutoModelForCausalLM.from_pretrained(
                self.args.honest_prover_name_or_path
            )
            self.models["sneaky_prover"] = AutoModelForCausalLM.from_pretrained(
                self.args.sneaky_prover_name_or_path
            )
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            raise

    def _create_optimizers_schedulers(self):
        # --- Placeholder 6: Instantiate Optimizers & Schedulers ---
        logger.info("Creating optimizers and schedulers...")
        # Replace with your actual optimizer/scheduler creation
        # Example:
        # from torch.optim import AdamW
        # from transformers import get_linear_schedule_with_warmup

        # Optimizer A
        # optimizer_a_params = self.models["honest_prover"].parameters() # Adjust if only training specific parts
        # self.optimizers["honest_prover"] = AdamW(optimizer_a_params, lr=self.args.learning_rate_a)
        # # Scheduler A (needs num_training_steps)
        # num_training_steps = self.args.max_train_steps # Or calculate from epochs/dataloader length
        # self.schedulers["honest_prover"] = get_linear_schedule_with_warmup(
        #     self.optimizers["honest_prover"], num_warmup_steps=0, num_training_steps=num_training_steps
        # )

        # Optimizer B
        # optimizer_b_params = self.models["sneaky_prover"].parameters()
        # self.optimizers["sneaky_prover"] = AdamW(optimizer_b_params, lr=self.args.learning_rate_b)
        # # Scheduler B
        # self.schedulers["sneaky_prover"] = get_linear_schedule_with_warmup(
        #     self.optimizers["sneaky_prover"], num_warmup_steps=0, num_training_steps=num_training_steps
        # )

        # Dummy implementation for structure
        from torch.optim import AdamW
        from transformers import get_scheduler

        num_training_steps = 1000  # Placeholder - Calculate properly later!
        try:
            self.optimizers["honest_prover"] = AdamW(
                self.models["honest_prover"].parameters(), lr=self.args.learning_rate_a
            )
            self.schedulers["honest_prover"] = get_scheduler(
                "linear",
                self.optimizers["honest_prover"],
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
            self.optimizers["sneaky_prover"] = AdamW(
                self.models["sneaky_prover"].parameters(), lr=self.args.learning_rate_b
            )
            self.schedulers["sneaky_prover"] = get_scheduler(
                "linear",
                self.optimizers["sneaky_prover"],
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
            logger.info("Optimizers and Schedulers created.")
        except Exception as e:
            logger.error(f"Failed to create optimizers/schedulers: {e}", exc_info=True)
            raise

    def _prepare_components(self):
        # --- Placeholder 7: Prepare Components using Accelerators ---
        logger.info("Preparing components with Accelerate...")
        accel_a = self.accelerators["honest_prover"]
        accel_b = self.accelerators["sneaky_prover"]

        # Select Plugin for A
        logger.debug("Selecting DeepSpeed plugin 'honest_prover'")
        accel_a.state.select_deepspeed_plugin("honest_prover")

        # Prepare A's Components
        logger.debug("Preparing components for Model A...")
        try:
            (
                prepared_honest_prover,
                prepared_opt_a,
                prepared_train_dl,
                prepared_sched_a,
            ) = accel_a.prepare(
                self.models["honest_prover"],
                self.optimizers["honest_prover"],
                self.dataloaders["train"],  # Prepare the main train dataloader here
                self.schedulers["honest_prover"],
            )
            # Update stored components
            self.models["honest_prover"] = prepared_honest_prover
            self.optimizers["honest_prover"] = prepared_opt_a
            self.dataloaders["train"] = (
                prepared_train_dl  # Overwrite with prepared version
            )
            self.schedulers["honest_prover"] = prepared_sched_a
            logger.info("Model A components prepared.")
        except Exception as e:
            logger.error(f"Failed to prepare Model A components: {e}", exc_info=True)
            raise

        # Select Plugin for B
        logger.debug("Selecting DeepSpeed plugin 'sneaky_prover'")
        # Use accel_b or accel_a, state is shared
        accel_b.state.select_deepspeed_plugin("sneaky_prover")

        # Prepare B's Components
        logger.debug("Preparing components for Model B...")
        try:
            # Don't re-prepare the dataloader
            prepared_sneaky_prover, prepared_opt_b, prepared_sched_b = accel_b.prepare(
                self.models["sneaky_prover"],
                self.optimizers["sneaky_prover"],
                self.schedulers["sneaky_prover"],
            )
            # Update stored components
            self.models["sneaky_prover"] = prepared_sneaky_prover
            self.optimizers["sneaky_prover"] = prepared_opt_b
            self.schedulers["sneaky_prover"] = prepared_sched_b
            logger.info("Model B components prepared.")
        except Exception as e:
            logger.error(f"Failed to prepare Model B components: {e}", exc_info=True)
            raise

        # Prepare Eval Dataloader (using one accelerator is fine)
        if "eval" in self.dataloaders:
            logger.debug("Preparing eval dataloader...")
            try:
                self.dataloaders["eval"] = accel_a.prepare(self.dataloaders["eval"])
                logger.info("Eval dataloader prepared.")
            except Exception as e:
                logger.error(f"Failed to prepare eval dataloader: {e}", exc_info=True)
                # Decide if this is fatal or if eval can be skipped
                # raise

    # --- Placeholder Methods for Training, Evaluation, Saving, Loading ---
    def train(self):
        logger.info("Starting training loop...")
        # --- Placeholder for Phase 2 ---
        # Calculate num_training_steps properly here based on dataloader length/epochs/max_steps
        # Handle resuming from checkpoint
        # Outer loop (epochs/steps)
        # Inner loop (batches)
        #   Get batch
        #   Model A forward (no_grad for intermediate)
        #   Model A forward (for loss) + backward (accel_a)
        #   Model B forward + backward (accel_b)
        #   Optimizer steps (accel_a, accel_b)
        #   Logging (accel_a)
        #   Evaluation call
        #   Checkpointing call
        logger.warning("Training loop not implemented.")
        pass

    def evaluate(self):
        logger.info("Starting evaluation...")
        # --- Placeholder for Phase 3 ---
        # Set models to eval
        # Loop through eval dataloader
        #   no_grad context
        #   Prepare batch
        #   Model A forward (intermediate)
        #   Model B forward
        #   Gather metrics (accel_a)
        # Calculate final metrics
        # Log metrics
        logger.warning("Evaluation loop not implemented.")
        return {}

    def save_checkpoint(self, output_dir: str):
        logger.info(f"Saving checkpoint to {output_dir}...")
        # --- Placeholder for Phase 4 (Saving) ---
        # Wait for everyone
        # Select plugin A -> accel_a.save_state(...)
        # Select plugin B -> accel_b.save_state(...)
        # Save RNG/loop state (accel_a)
        logger.warning("Checkpoint saving not implemented.")
        pass

    def load_checkpoint(self, checkpoint_dir: str):
        logger.info(f"Loading checkpoint from {checkpoint_dir}...")
        # --- Placeholder for Phase 4 (Loading) ---
        # Load loop state/RNG (accel_a)
        # Select plugin A -> accel_a.load_state(...)
        # Select plugin B -> accel_b.load_state(...)
        logger.warning("Checkpoint loading not implemented.")
        pass


# --- Script Entry Point ---
def run_training():
    parser = HfArgumentParser((ScriptArguments,))
    # In a real script, use:
    script_args = parser.parse_args_into_dataclasses()[0]
    # For demonstration:
    # script_args = ScriptArguments(...) # Use the same dummy args as before if needed

    # Create dummy DeepSpeed config files if they don't exist
    ds_configs_exist = True
    if not os.path.exists("ds_config_zero2.json"):
        try:
            with open("ds_config_zero2.json", "w") as f:
                f.write(
                    '{"zero_optimization": {"stage": 2}, "train_micro_batch_size_per_gpu": 1}'
                )
            logger.warning("Created dummy ds_config_zero2.json")
        except OSError:
            logger.error("Failed to create dummy ds_config_zero2.json.")
            ds_configs_exist = False
    if not os.path.exists("ds_config_zero3.json"):
        try:
            with open("ds_config_zero3.json", "w") as f:
                f.write(
                    '{"zero_optimization": {"stage": 3}, "train_micro_batch_size_per_gpu": 1}'
                )
            logger.warning("Created dummy ds_config_zero3.json")
        except OSError:
            logger.error("Failed to create dummy ds_config_zero3.json.")
            ds_configs_exist = False

    if not ds_configs_exist:
        logger.error("Cannot proceed without DeepSpeed config files.")
        return

    # Instantiate the trainer
    trainer = DisjointTrainer(args=script_args)

    # Load data, models, optimizers and prepare them
    trainer.setup_components()

    # Start training
    trainer.train()

    # Optional: Final evaluation
    # final_metrics = trainer.evaluate()
    # logger.info(f"Final evaluation metrics: {final_metrics}")

    logger.info("Training finished.")

    # Optional: Clean up dummy files
    # ...


if __name__ == "__main__":
    run_training()
