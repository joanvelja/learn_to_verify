# disjointTrainer.py

import os
import json
from typing import Any, Literal
from collections import defaultdict
from pvg import (
    setup_logger,
    Container,
    nanstd,
    prepare_deepspeed,
    VLLMClient,
    AppsDataset,
    RepeatRandomSampler,
)
import logging
import copy
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DeepSpeedPlugin,
    ProjectConfiguration,
    gather_object,
    broadcast_object_list,
)
from transformers import (
    set_seed,
    AutoModelForCausalLM,  # Assuming Causal LM for now
    AutoTokenizer,
    get_scheduler,
    PreTrainedModel,
)
from torch.utils.data import DataLoader, Dataset  # Add Dataset import
from tqdm.auto import tqdm  # For progress bars
from torch.utils.data import Sampler
from contextlib import nullcontext
import deepspeed
from deepspeed.utils import safe_get_full_grad
from trl.trainer.utils import selective_log_softmax
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
import uuid
import datetime
import re
import time

# from pvg.utils.logger import setup_logger  # Import the setup function
from pvg.config.args import ExperimentArgs  # Import the args class

# --- Get Project Root Logger ---
logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


# --- The Trainer Class ---
class DisjointSequentialTrainer:
    def __init__(self, args: ExperimentArgs) -> None:
        self.args: ExperimentArgs = args
        self.accelerators: dict[str, Accelerator] = {}
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.dataloaders: dict[str, torch.utils.data.DataLoader] = {}
        self.deepspeed_plugins: dict[str, DeepSpeedPlugin] = {}
        self.vllm_clients: dict[str, VLLMClient | None] = {
            "honest_prover": None,
            "sneaky_prover": None,
            "verifier": None,
        }
        self._signature_columns = None
        self.wandb_run = None

        # Buffer to store inputs for gradient accumulation steps
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # State variables
        self.global_step = 0
        self.current_epoch = 0

        # --- 1. Seeding ---
        self._set_seed()

        # --- 2. Accelerator Initialization ---
        self._initialize_accelerators()
        # Use accelerator_a for general state/logging, device info etc.
        self.accelerator = self.accelerators["honest_prover"]

        # --- 2a. Logging Setup (using info from accelerator) ---
        log_level = (
            logging.INFO if self.accelerator.is_main_process else logging.WARNING
        )
        log_dir = os.path.join(
            self.args.output_dir, "logs"
        )  # Place logs inside output dir
        setup_logger(
            level=log_level,
            rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes,
            log_to_file=True,  # Enable file logging
            log_dir=log_dir,
            log_filename="training.log",
            main_process_only_file=True,  # Only rank 0 writes the main log file
        )
        # Now subsequent logging calls in any module using logging.getLogger("pvg...")
        # will use this configuration.
        logger.info("DisjointSequentialTrainer logging configured.")
        self.prepare_wandb()

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
                "Sneaky Prover accelerator not initialized. If this is expected, ignore this message. If not, you may have not passed sneaky_prover_name_or_path correctly to the script."
            )

        logger.info("Trainer initialized. Accelerators are set up.")

        # --- GPU Visibility Logging ---
        logger.info("--- Training Process GPU Info ---")
        logger.info(f"  Accelerator Device: {self.accelerator.device}")
        if torch.cuda.is_available():
            logger.info(
                f"  torch.cuda.device_count() seen by this process: {torch.cuda.device_count()}"
            )
            try:
                current_cuda_device_index = torch.cuda.current_device()
                logger.info(
                    f"  torch.cuda.current_device(): {current_cuda_device_index}"
                )
                logger.info(
                    f"  torch.cuda.get_device_name(): {torch.cuda.get_device_name(current_cuda_device_index)}"
                )
            except Exception as e:
                logger.error(f"  Could not get current CUDA device details: {e}")
        else:
            logger.info("  CUDA not available to this process.")
        logger.info("--- End Training Process GPU Info ---")
        # --- End GPU Visibility Logging ---

        # Wait for everyone
        # self.accelerator.wait_for_everyone() # Not needed (seems like)

        # --- Instantiate vLLM Clients (Main Process Only) ---
        if self.accelerator.is_main_process:
            logger.info("Main process initializing vLLM clients...")
            base_group_port = 51216
            # Initialize vLLM clients - Honest Prover
            self._initialize_vllm_client(
                client_key="honest_prover",
                host=self.args.vllm_honest_prover.vllm_host,
                port=self.args.vllm_honest_prover.vllm_port,
                group_port=base_group_port,
                timeout=self.args.vllm_honest_prover.vllm_server_timeout,
            )
            # Initialize vLLM clients - Sneaky Prover
            self._initialize_vllm_client(
                client_key="sneaky_prover",
                host=self.args.vllm_sneaky_prover.vllm_host,
                port=self.args.vllm_sneaky_prover.vllm_port,
                group_port=base_group_port + 1,
                timeout=self.args.vllm_sneaky_prover.vllm_server_timeout,
            )
            # Initialize vLLM clients - Verifier
            self._initialize_vllm_client(
                client_key="verifier",
                host=self.args.vllm_verifier.vllm_host,
                port=self.args.vllm_verifier.vllm_port,
                group_port=base_group_port + 2,
                timeout=self.args.vllm_verifier.vllm_server_timeout,
            )

        # --- End vLLM Client Instantiation ---
        logger.info("vLLM Clients initialized.")
        logger.info("--- vLLM Clients State ---")
        logger.info(f"  vLLM Client A: {self.vllm_clients['honest_prover']}")
        logger.info(f"  vLLM Client B: {self.vllm_clients['sneaky_prover']}")
        (
            logger.info(f"  vLLM Client C: {self.vllm_clients['verifier']}")
            if self.vllm_clients["verifier"]
            else None
        )

        # --- 3. Load Tokenizer, Models, Datasets, Optimizers ---
        self._load_tokenizer()
        self._load_models()
        if self.accelerator.is_main_process and self.wandb_run:
            self._log_model_parameters()

        # --- 4. Load Datasets ---
        self.train_dataset, self.eval_dataset = self._load_datasets()

        # --- 5. Create Optimizers ---
        self._create_optimizers()

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # --- 6. Create Dataloaders ---
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._get_train_sampler(),
            drop_last=True,
            collate_fn=data_collator,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=self._get_eval_sampler(self.eval_dataset),
            drop_last=True,
            collate_fn=data_collator,
        )
        # Copy for model_b
        self.train_dataloader_b = copy.deepcopy(self.train_dataloader)
        self.eval_dataloader_b = copy.deepcopy(self.eval_dataloader)
        # --- 6. Prepare Components ---
        self._prepare_components()  # This will use self.accelerators and update self.models etc.

        # Calculate num_training_steps after dataloader is prepared
        self.num_training_steps = self._calculate_num_training_steps()
        # Now create schedulers with the correct number of steps
        self._create_schedulers(
            self.num_training_steps
        )  # Necessary to do this here, since we need the dataloaders to be prepared before calculating the number of training steps --> Thus schedulers prepared a posteriori

        # --- 7. Load from Checkpoint (if specified) ---
        if self.args.resume_from_checkpoint:
            self.load_checkpoint(self.args.resume_from_checkpoint)

        # --- 8. Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations ---
        num_processes = self.accelerator.num_processes
        global_batch_size = self.args.per_device_train_batch_size * num_processes
        possible_values = [
            n_gen
            for n_gen in range(2, global_batch_size + 1)
            if (global_batch_size) % n_gen == 0
        ]
        if self.args.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {self.args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.args.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )

        # --- 9. Initialize the Metrics used for logging ---
        # Initialize the metrics
        self._metrics = {
            "train": {
                "honest_prover": defaultdict(list),
                "sneaky_prover": defaultdict(list),
                "verifier": defaultdict(list),
            },
            "eval": {
                "honest_prover": defaultdict(list),
                "sneaky_prover": defaultdict(list),
                "verifier": defaultdict(list),
            },
        }
        self._total_train_tokens = {
            "honest_prover": 0,
            "sneaky_prover": 0,
            "verifier": 0,
        }

        logger.info("--- Initialization Complete ---")

    def prepare_wandb(self) -> None:
        try:
            logger.info("Initializing WandB tracker via accelerator.init_trackers...")
            # Pass project name, config (args), and specific wandb init kwargs
            self.accelerator.init_trackers(
                project_name=self.args.wandb_project_name,
                config=self.args.__dict__,  # Log all script args
                init_kwargs={
                    "wandb": {
                        "entity": self.args.wandb_entity,
                        "name": self.args.wandb_run_name,  # Optional run name
                        # Add other wandb.init args here if needed, e.g., tags, notes
                    }
                },
            )
            logger.info("WandB tracker initialization requested.")
            # Now, immediately try to get the run object on the main process
            if self.accelerator.is_main_process:
                self.wandb_run = self.accelerator.get_tracker("wandb").run
                if self.wandb_run:
                    logger.info(
                        f"Successfully retrieved WandB run. Run ID: {self.wandb_run.id}"
                    )
                    # Create LLM interaction log directory on main process
                    self.llm_interaction_log_dir = os.path.join(
                        self.args.output_dir, self.wandb_run.id, "llm_interaction_logs"
                    )
                    if self.accelerator.is_main_process:
                        os.makedirs(self.llm_interaction_log_dir, exist_ok=True)
                        logger.info(
                            f"LLM interaction logs will be saved to: {self.llm_interaction_log_dir}"
                        )
                else:
                    logger.error(
                        "Called init_trackers, but failed to retrieve WandB run object."
                    )
        except Exception as e:
            logger.error(
                f"Error during accelerator.init_trackers or run retrieval: {e}",
                exc_info=True,
            )
            # Ensure self.wandb_run remains None if init fails
            self.wandb_run = None

        if self.accelerator.is_main_process:
            if not self.accelerator.trackers:
                logger.error("WandB tracker not initialized. Cannot log.")
                self.wandb_run = None  # Or raise error
            else:
                self.wandb_run = self.accelerator.get_tracker("wandb").run
                if self.wandb_run is None:
                    logger.error("Could not retrieve WandB run object.")
                else:
                    logger.info(
                        f"WandB tracker initialized. Run ID: {self.wandb_run.id}"
                    )
                    # Log initial config (accelerate might do some, but explicit update is safer)
                    self.wandb_run.config.update(
                        self.args.__dict__, allow_val_change=True
                    )

                    # Log environment details
                    try:
                        import importlib.metadata as importlib_metadata
                        import sys
                        import platform

                        libs = [
                            "torch",
                            "transformers",
                            "accelerate",
                            "deepspeed",
                            "vllm",
                            "wandb",
                        ]
                        lib_versions = {
                            lib: importlib_metadata.version(lib)
                            for lib in libs
                            if importlib_metadata.version(lib)
                        }
                        self.wandb_run.config.update(
                            {
                                "environment/python_version": sys.version,
                                "environment/platform": platform.platform(),
                                "environment/num_processes": self.accelerator.num_processes,
                                "environment/mixed_precision": self.accelerator.mixed_precision,
                                "environment/distributed_type": str(
                                    self.accelerator.distributed_type
                                ),
                                "environment/library_versions": lib_versions,
                            }
                        )
                        logger.info("Environment details logged to WandB.")
                    except Exception as e:
                        logger.warning(f"Could not log all environment details: {e}")

    def _log_model_parameters(self) -> None:
        # Log model parameter counts (example for honest prover)
        try:
            for model_key in ["honest_prover", "sneaky_prover"]:
                log_model = self.models[model_key]  # Use the unprepared model
                total_params = sum(p.numel() for p in log_model.parameters())
                trainable_params = sum(
                    p.numel() for p in log_model.parameters() if p.requires_grad
                )
                self.wandb_run.config.update(
                    {
                        f"model/{model_key}/total_params": total_params,
                        f"model/{model_key}/trainable_params": trainable_params,
                        f"model/{model_key}/class": log_model.__class__.__name__,
                    }
                )

            del log_model  # Delete the model from memory (since already loaded into self.models)
            del total_params  # Delete the total parameter count from memory
            del trainable_params  # Delete the trainable parameter count from memory

        except Exception as e:
            logger.warning(f"Could not log model parameter counts: {e}")

    def _initialize_vllm_client(
        self,
        client_key: str,
        host: str,
        port: int,
        group_port: int | None,
        timeout: float,
    ) -> None:
        """Initializes a single VLLMClient instance and handles errors."""
        logger.info(f"Attempting to initialize vLLM client: {client_key}")
        try:
            client_kwargs = {
                "host": host,
                "server_port": port,
                "connection_timeout": timeout,
            }
            if group_port is not None:
                client_kwargs["group_port"] = group_port

            self.vllm_clients[client_key] = VLLMClient(**client_kwargs)
            logger.info(
                f"Successfully connected vLLM client '{client_key}' to {host}:{port}"
            )
        except ConnectionError as e:
            logger.error(
                f"Failed to connect to vLLM server '{client_key}' at {host}:{port}. Ensure it's running and accessible: {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred initializing vLLM Client '{client_key}': {e}",
                exc_info=True,
            )
            raise

    def _set_seed(self) -> None:
        set_seed(self.args.seed)
        logger.info(f"Set random seed to {self.args.seed}")

    def _setup_logging(self) -> None:
        # Logging setup happens before accelerator init, so we check env vars
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

    def _initialize_accelerators(self) -> None:
        logger.info("Initializing Accelerators...")
        # Create DeepSpeed Plugins
        try:
            ds_plugin_a = DeepSpeedPlugin(
                hf_ds_config=self.args.ds_config_honest_prover
            )
            ds_plugin_b = DeepSpeedPlugin(
                hf_ds_config=self.args.ds_config_sneaky_prover
            )
            logger.info("DeepSpeed plugins created.")
        except Exception as e:
            logger.error(f"Failed to create DeepSpeed plugins: {e}", exc_info=True)
            raise
        self.deepspeed_plugins = {
            "honest_prover": ds_plugin_a,
            "sneaky_prover": ds_plugin_b,
        }
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir,
            logging_dir=os.path.join(self.args.output_dir, "accelerate_logs"),
        )

        # Instantiate the first accelerator
        try:
            accelerator_a = Accelerator(
                deepspeed_plugin=self.deepspeed_plugins,
                log_with="wandb",
                project_config=project_config,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                mixed_precision=self.args.mixed_precision,
            )
            logger.info("First Accelerator (accelerator_a) initialized.")
            self.accelerators["honest_prover"] = accelerator_a
        except Exception as e:
            logger.error(
                f"Failed to initialize the first Accelerator: {e}", exc_info=True
            )
            raise

        # Instantiate the second accelerator
        try:
            accelerator_b = Accelerator()
            logger.info("Second Accelerator (accelerator_b) initialized.")
            self.accelerators["sneaky_prover"] = accelerator_b
        except Exception as e:
            logger.error(
                f"Failed to initialize the second Accelerator: {e}", exc_info=True
            )
            raise

        # Redirect verifier accelerator to self.accelerators["honest_prover"]
        self.accelerators["verifier"] = self.accelerators["honest_prover"]  # Hacky?

    def _load_tokenizer(self) -> None:
        tokenizer_path = (
            self.args.tokenizer_name_or_path
            if self.args.tokenizer_name_or_path
            else self.args.honest_prover_name_or_path
        )
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Add padding token if missing (common for GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer missing pad token, setting to eos token.")

    def _load_models(self) -> None:
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs = {}
        model_init_kwargs["use_cache"] = (
            False
            if self.args.gradient_checkpointing
            else model_init_kwargs.get("use_cache")
        )  # TODO: Missing gradient checkpointing arg and logic dealing with it

        # Fetch models from local paths (& update path to model if found)
        self.args.honest_prover_name_or_path = self._fetch_local_models(
            self.args.honest_prover_name_or_path
        )
        self.args.sneaky_prover_name_or_path = self._fetch_local_models(
            self.args.sneaky_prover_name_or_path
        )

        logger.info(
            f"Loading Honest Prover from {self.args.honest_prover_name_or_path}"
        )
        model_a = AutoModelForCausalLM.from_pretrained(
            self.args.honest_prover_name_or_path, **model_init_kwargs
        )
        # --- Enable Gradient Checkpointing for Model A ---
        if self.args.gradient_checkpointing:
            model_a = self._enable_gradient_checkpointing(model_a)
            logger.info("Gradient checkpointing enabled for Honest Prover.")

        logger.info(
            f"Loading Sneaky Prover from {self.args.sneaky_prover_name_or_path}"
        )
        model_b = AutoModelForCausalLM.from_pretrained(
            self.args.sneaky_prover_name_or_path, **model_init_kwargs
        )
        # --- Enable Gradient Checkpointing for Model B ---
        if self.args.gradient_checkpointing:
            model_b = self._enable_gradient_checkpointing(model_b)
            logger.info("Gradient checkpointing enabled for Sneaky Prover.")
        # logger.info(f"Loading Verifier from {self.args.verifier_name_or_path}")
        # # TODO: Load Verifier (depending on the type of verifier we want to use?)
        # verifier = AutoModelForCausalLM.from_pretrained(
        #     self.args.verifier_name_or_path, **model_init_kwargs
        # )

        self.models = {
            "honest_prover": model_a,
            "sneaky_prover": model_b,
            # "verifier": verifier,
        }

        # Reference models for RL
        if self.args.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model_a = None
            self.ref_model_b = None
        # elif is_deepspeed_zero3_enabled():
        else:
            self.ref_model_a = AutoModelForCausalLM.from_pretrained(
                self.args.honest_prover_name_or_path
            )
            self.ref_model_b = AutoModelForCausalLM.from_pretrained(
                self.args.sneaky_prover_name_or_path
            )
        self.ref_models = {
            "honest_prover": self.ref_model_a,
            "sneaky_prover": self.ref_model_b,
        }

        if (
            self.args.apply_liger_kernel
        ):  # Optimization stuff (claims 60% memory savings)
            _apply_liger_kernel_to_instance(model_a)
            _apply_liger_kernel_to_instance(model_b)
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.args.beta,
                epsilon_low=self.args.epsilon_low,
                epsilon_high=self.args.epsilon_high,
                temperature=self.args.vllm_temperature_honest_prover,
                use_ref_model=True if self.args.beta != 0.0 else False,
            )

    def _load_datasets(
        self,
    ) -> tuple[AppsDataset, AppsDataset]:  # Returns (train_dataset, eval_dataset)
        train_dataset = AppsDataset(
            dataset_name=self.args.dataset_name,
            tokenizer=self.tokenizer,
            num_samples=self.args.train_num_samples,
            split="train",
        )

        eval_dataset = AppsDataset(
            dataset_name=self.args.dataset_name,
            tokenizer=self.tokenizer,
            split="test",
        )

        # def data_collator(features):  # No data collation is needed in GRPO
        #     return features
        #
        # # ---------------------------------------------------------------------
        # train_dataloader = DataLoader(
        #     train_dataset,
        #     shuffle=True,  # Shuffle for training
        #     collate_fn=data_collator,
        #     batch_size=self.args.per_device_train_batch_size
        # )
        # eval_dataloader = None
        # if eval_dataset:
        #     eval_dataloader = DataLoader(
        #         eval_dataset,
        #         shuffle=False,  # No shuffle for eval
        #         collate_fn=data_collator,
        #         batch_size=self.args.per_device_eval_batch_size,
        #     )

        # logger.info(f"Train Dataloader: {len(train_dataloader)} batches")
        # if eval_dataloader:
        #     logger.info(f"Eval Dataloader: {len(eval_dataloader)} batches")
        logger.info(f"Train Dataset: {len(train_dataset)} samples")
        logger.info(f"Eval Dataset: {len(eval_dataset)} samples")

        return train_dataset, eval_dataset

    def _create_optimizers(self) -> None:
        # Simple AdamW optimizer setup - can be customized
        optimizer_a = torch.optim.AdamW(
            self.models["honest_prover"].parameters(),
            lr=self.args.learning_rate_honest_prover,
            weight_decay=self.args.weight_decay_honest_prover,
        )
        optimizer_b = torch.optim.AdamW(
            self.models["sneaky_prover"].parameters(),
            lr=self.args.learning_rate_sneaky_prover,
            weight_decay=self.args.weight_decay_sneaky_prover,
        )
        logger.info("Optimizers created.")
        self.optimizers = {"honest_prover": optimizer_a, "sneaky_prover": optimizer_b}

    def _calculate_num_training_steps(self) -> int:
        num_update_steps_per_epoch = (
            len(self.train_dataloader) // self.args.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = max(
            num_update_steps_per_epoch, 1
        )  # Ensure at least one step

        if self.args.max_train_steps is not None:
            num_training_steps = self.args.max_train_steps
            self.args.num_train_epochs = (
                self.args.max_train_steps // num_update_steps_per_epoch
            )
        else:
            num_training_steps = num_update_steps_per_epoch * self.args.num_train_epochs

        logger.info(f"Calculated num_training_steps: {num_training_steps}")
        return num_training_steps

    def _create_schedulers(self, num_training_steps: int) -> None:

        scheduler_a = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizers["honest_prover"],
            num_warmup_steps=self.args.num_warmup_steps
            * self.accelerator.num_processes,  # Adjust warmup steps for distributed
            num_training_steps=num_training_steps,
        )
        scheduler_b = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizers["sneaky_prover"],
            num_warmup_steps=self.args.num_warmup_steps
            * self.accelerator.num_processes,  # Adjust warmup steps for distributed
            num_training_steps=num_training_steps,
        )
        logger.info("Schedulers created.")
        self.schedulers = {"honest_prover": scheduler_a, "sneaky_prover": scheduler_b}

        # Prepare the schedulers
        self._prepare_schedulers()

    def _prepare_training_components(
        self,
        model_key: Literal["honest_prover", "sneaky_prover"],
        dataloader: DataLoader,
    ) -> tuple[
        PreTrainedModel,
        torch.optim.Optimizer,
        DataLoader,
    ]:
        """
        Selects the DeepSpeed plugin and prepares the model, optimizer, dataloader, and scheduler
        for the specified model_key using its associated accelerator.

        Args:
            model_key: The key identifying the model and its components.
            dataloader: The specific dataloader instance to prepare for this model.

        Returns:
            A tuple containing the prepared (model, optimizer, dataloader, scheduler).
        """
        accelerator: Accelerator = self.accelerators[model_key]
        model: PreTrainedModel = self.models[model_key]
        optimizer: torch.optim.Optimizer = self.optimizers[model_key]

        logger.info(f"Selecting plugin and preparing components for {model_key}...")
        accelerator.state.select_deepspeed_plugin(model_key)

        prepared_components: tuple[
            PreTrainedModel,
            torch.optim.Optimizer,
            DataLoader,
        ] = accelerator.prepare(model, optimizer, dataloader)

        logger.info(f"Components prepared for {model_key}.")
        # Returns: model, optimizer, dataloader (in that order)
        return prepared_components

    def _prepare_schedulers(self) -> None:
        logger.info("Preparing schedulers...")
        self.schedulers["honest_prover"] = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizers["honest_prover"],
            num_warmup_steps=self.args.num_warmup_steps
            * self.accelerator.num_processes,  # Adjust warmup steps for distributed
            num_training_steps=self.num_training_steps,
        )
        self.schedulers["sneaky_prover"] = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizers["sneaky_prover"],
            num_warmup_steps=self.args.num_warmup_steps
            * self.accelerator.num_processes,  # Adjust warmup steps for distributed
            num_training_steps=self.num_training_steps,
        )
        logger.info("Schedulers prepared.")

    def _prepare_components(self) -> None:
        logger.info("Preparing components with Accelerate...")

        # Prepare Model A (Honest Prover) components
        (
            self.models["honest_prover"],
            self.optimizers["honest_prover"],
            self.train_dataloader,
        ) = self._prepare_training_components(
            model_key="honest_prover",
            dataloader=self.train_dataloader,  # Pass the specific dataloader
        )

        # Prepare Model B (Sneaky Prover) components
        (
            self.models["sneaky_prover"],
            self.optimizers["sneaky_prover"],
            self.train_dataloader_b,
        ) = self._prepare_training_components(
            model_key="sneaky_prover",
            dataloader=self.train_dataloader_b,  # Pass the specific dataloader for B
        )

        # Prepare Eval Dataloaders (using either accelerator context is fine, let's use honest_prover's)
        # Note: accelerator.prepare handles DataLoaders differently (doesn't require plugin selection)
        if self.eval_dataloader:
            logger.info("Preparing Eval Dataloaders...")
            self.eval_dataloader = self.accelerators["honest_prover"].prepare(
                self.eval_dataloader
            )
            self.eval_dataloader_b = self.accelerators[
                "sneaky_prover"
            ].prepare(  # Use respective accelerator
                self.eval_dataloader_b
            )

        # Prepare reference models (using existing logic as it's slightly different)
        if self.ref_model_a:  # One check is enough if logic is symmetric
            logger.info("Preparing reference models with DeepSpeed...")
            self.ref_models["honest_prover"] = prepare_deepspeed(
                self.ref_model_a,
                self.accelerators["honest_prover"],
            )
            self.ref_models["sneaky_prover"] = prepare_deepspeed(
                self.ref_model_b,
                self.accelerators["sneaky_prover"],
            )

        logger.info("Component preparation complete.")

    # --- Placeholder Methods ---
    def train(self) -> None:
        logger.info("***** Starting Training *****")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        # Calculate the effective batch size correctly
        effective_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes  # Use num_processes from one accelerator
            * self.args.gradient_accumulation_steps
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {effective_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.num_training_steps}")

        progress_bar = tqdm(
            range(self.num_training_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training Steps",
        )

        # Resume progress bar if needed
        if self.args.resume_from_checkpoint:
            progress_bar.update(self.global_step)

        for epoch in range(self.current_epoch, self.args.num_train_epochs):
            self.models["honest_prover"].train()
            self.models["sneaky_prover"].train()
            logger.info(
                f"--- Starting Epoch {epoch+1}/{self.args.num_train_epochs} ---"
            )
            # Optional: Track loss over accumulation steps for more stable logging
            # accumulated_loss_a = 0.0
            # accumulated_loss_b = 0.0

            # --- Training Loop ---
            for step, batch in enumerate(self.train_dataloader):
                # --- vLLM Generation & Training Step ---
                # Generation happens on main process, results broadcast if needed
                # Loss calculation and backward happen on all processes
                losses = self._training_step(batch)
                loss_a = losses["loss_a"]
                loss_b = losses["loss_b"]

                logger.info(
                    f"[Process {self.accelerator.process_index}] Completed training step"
                )
                logger.info(
                    f"[Process {self.accelerator.process_index}] Losses: {loss_a}, {loss_b}"
                )

                # Optional: Track loss over accumulation steps for more stable logging
                # accumulated_loss_a += loss_a.item()
                # accumulated_loss_b += loss_b.item()

                # --- Gradient Accumulation ---
                # Scale the loss for this micro-batch
                loss_a_scaled = loss_a / self.args.gradient_accumulation_steps
                loss_b_scaled = loss_b / self.args.gradient_accumulation_steps

                # Perform backward pass to accumulate gradients
                # Use the correct accelerator for each model's loss
                # The backward pass should happen *before* the sync step check
                self.accelerators["honest_prover"].backward(loss_a_scaled)
                self.accelerators["sneaky_prover"].backward(loss_b_scaled)
                logger.info(
                    f"[Process {self.accelerator.process_index}] Completed backward pass"
                )
                logger.info(
                    f"[Process {self.accelerator.process_index}] Losses: {loss_a_scaled}, {loss_b_scaled}"
                )

                # --- Synchronization Point Check ---
                # Determine if this is the last step of an accumulation cycle or the overall last step
                is_last_step_in_batch = (step + 1) == len(self.train_dataloader)
                is_accumulation_boundary = (
                    step + 1
                ) % self.args.gradient_accumulation_steps == 0
                is_sync_step = is_accumulation_boundary or is_last_step_in_batch

                # --- Optimizer Step ---
                if is_sync_step:  # See above for explanation of this

                    # Calculate gradient norms BEFORE clipping and stepping
                    grad_norm_a = torch.tensor(
                        0.0, device=self.accelerators["honest_prover"].device
                    )
                    try:
                        model_a_params = self.models["honest_prover"].parameters()
                        grad_sq_sum_a = torch.tensor(
                            0.0, device=self.accelerators["honest_prover"].device
                        )
                        for p in model_a_params:
                            grad = safe_get_full_grad(p)
                            if grad is not None:
                                grad_sq_sum_a += torch.norm(grad) ** 2
                        grad_norm_a = torch.sqrt(grad_sq_sum_a)
                    except Exception as e:
                        logger.warning(
                            f"Could not compute gradient norm for honest_prover: {e}"
                        )
                    eval_mode = "train"  # Assuming this is only called during training
                    self._metrics[eval_mode]["honest_prover"]["grad_norm"] = (
                        self._metrics[eval_mode]["honest_prover"].get("grad_norm", [])
                    )
                    self._metrics[eval_mode]["honest_prover"]["grad_norm"].append(
                        grad_norm_a.item()
                    )

                    grad_norm_b = torch.tensor(
                        0.0, device=self.accelerators["sneaky_prover"].device
                    )
                    try:
                        model_b_params = self.models["sneaky_prover"].parameters()
                        grad_sq_sum_b = torch.tensor(
                            0.0, device=self.accelerators["sneaky_prover"].device
                        )
                        for p in model_b_params:
                            grad = safe_get_full_grad(p)
                            if grad is not None:
                                grad_sq_sum_b += torch.norm(grad) ** 2
                        grad_norm_b = torch.sqrt(grad_sq_sum_b)
                    except Exception as e:
                        logger.warning(
                            f"Could not compute gradient norm for sneaky_prover: {e}"
                        )
                    self._metrics[eval_mode]["sneaky_prover"]["grad_norm"] = (
                        self._metrics[eval_mode]["sneaky_prover"].get("grad_norm", [])
                    )
                    self._metrics[eval_mode]["sneaky_prover"]["grad_norm"].append(
                        grad_norm_b.item()
                    )

                    # Clip Gradients (Optional)
                    if self.args.max_grad_norm_honest_prover is not None:
                        self.accelerators["honest_prover"].clip_grad_norm_(
                            self.models["honest_prover"].parameters(),
                            self.args.max_grad_norm_honest_prover,
                        )
                    if self.args.max_grad_norm_sneaky_prover is not None:
                        self.accelerators["sneaky_prover"].clip_grad_norm_(
                            self.models["sneaky_prover"].parameters(),
                            self.args.max_grad_norm_sneaky_prover,
                        )

                    # Optimizer & Scheduler Steps
                    self.optimizers["honest_prover"].step()
                    if self.schedulers["honest_prover"]:
                        self.schedulers["honest_prover"].step()
                    self.optimizers["honest_prover"].zero_grad()

                    self.optimizers["sneaky_prover"].step()
                    if self.schedulers["sneaky_prover"]:
                        self.schedulers["sneaky_prover"].step()
                    self.optimizers["sneaky_prover"].zero_grad()

                    logger.info(
                        f"[Process {self.accelerator.process_index}] Completed optimizer step"
                    )
                    # *** ADD GLOBAL BARRIER HERE ***
                    # Ensure both processes complete optim/sched before proceeding
                    logger.info(
                        f"[Process {self.accelerator.process_index}] Waiting at barrier after optimizer step..."
                    )
                    self.accelerator.wait_for_everyone()  # Use primary accelerator for global sync
                    logger.info(
                        f"[Process {self.accelerator.process_index}] Passed barrier after optimizer step."
                    )

                    self.global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss_a": losses["loss_a"].item(),
                            "loss_b": losses["loss_b"].item(),
                            "step": self.global_step,  # Show global step
                            #  Example using averaged loss:
                            # "avg_loss_a": accumulated_loss_a / self.args.gradient_accumulation_steps,
                            # "avg_loss_b": accumulated_loss_b / self.args.gradient_accumulation_steps,
                        }
                    )  # Use .item() carefully

                    # --- Logging ---
                    if self.global_step % self.args.logging_steps == 0:
                        # Log metrics based on the last micro-batch loss or averaged loss
                        # Pass the unscaled losses from the last micro-batch
                        # self._log_metrics({"loss_a": loss_a, "loss_b": loss_b})
                        step_data = {
                            "losses": {
                                "honest_prover": loss_a.item(),
                                "sneaky_prover": loss_b.item(),
                            },
                            "step": self.global_step,
                            "epoch": self.current_epoch,
                            "lr": {
                                "honest_prover": (
                                    self.schedulers["honest_prover"].get_last_lr()[0]
                                    if self.schedulers["honest_prover"]
                                    else self.optimizers["honest_prover"].param_groups[
                                        0
                                    ]["lr"]
                                ),
                                "sneaky_prover": (
                                    self.schedulers["sneaky_prover"].get_last_lr()[0]
                                    if self.schedulers["sneaky_prover"]
                                    else self.optimizers["sneaky_prover"].param_groups[
                                        0
                                    ]["lr"]
                                ),
                            },
                        }
                        self._log_metrics(step_data)

                    # --- Evaluation ---
                    if (
                        self.eval_dataloader
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        self.evaluate()

                    # --- Checkpointing ---
                    if self.global_step % self.args.save_steps == 0:
                        self.save_checkpoint()

                    if self.accelerator.is_main_process:
                        logger.info(
                            f"[Step {self.global_step}] Optimizer step completed. Starting weight synchronization..."
                        )
                        sync_start_time = time.time()

                    # --- Weight Synchronization ---
                    # Decide when to sync weights (e.g., every N steps)
                    if self.global_step % self.args.sync_steps == 0:
                        self._sync_weights_to_vllm()

                    # --- FINAL BARRIER FOR THE SYNC STEP ---
                    # Ensure all processes complete logging, eval, checkpointing, and syncing
                    # before ANY process starts the next iteration.
                    logger.info(
                        f"[Process {self.accelerator.process_index}] Step {self.global_step}: Reached end of sync step logic. Waiting at final barrier..."
                    )
                    self.accelerator.wait_for_everyone()
                    logger.info(
                        f"[Process {self.accelerator.process_index}] Step {self.global_step}: Passed final barrier."
                    )

                    if self.accelerator.is_main_process:
                        sync_end_time = time.time()
                        sync_duration = sync_end_time - sync_start_time
                        logger.info(
                            f"[Step {self.global_step}] Weight synchronization finished. Duration: {sync_duration:.2f} seconds."
                        )
                        # Log this duration to wandb as well
                        if self.wandb_run:
                            self.wandb_run.log(
                                {"train/sync_duration": sync_duration},
                                step=self.global_step,
                            )

                    # Reset accumulated loss trackers if using them for logging
                    # accumulated_loss_a = 0.0
                    # accumulated_loss_b = 0.0

                if (
                    self.args.max_train_steps is not None
                    and self.global_step >= self.args.max_train_steps
                ):
                    logger.info("Reached max_train_steps. Stopping training.")
                    break  # Break inner loop

            self.current_epoch += 1  # Increment epoch counter
            if (
                self.args.max_train_steps is not None
                and self.global_step >= self.args.max_train_steps
            ):
                break  # Break outer loop

            # Accelerator wait for everyone
            self.accelerator.wait_for_everyone()

        progress_bar.close()
        logger.info("***** Training Finished *****")
        # Final save?
        # self.save_checkpoint(final=True)

    def _training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Performs a single training step:
        1. Generates sequences using vLLM (main process).
        2. Broadcasts generated sequences (if needed).
        3. Computes loss and gradients using training models (all processes).
        """
        logger.info(
            f"[Process {self.accelerator.process_index}] Entering _training_step"
        )
        # Plan:
        # 0. Prepare the inputs via vLLM + scoring + advantages for both provers [honest_prover, sneaky_prover]
        logger.info(
            f"[Process {self.accelerator.process_index}] Calling _prepare_inputs"
        )
        inputs = self._prepare_inputs(batch)  # This returns the following:
        # {
        #     "prompt_ids": prompt_ids,
        #     "prompt_mask": prompt_mask,
        #     "completion_ids": completion_ids,
        #     "completion_mask": completion_mask,
        #     "advantages": advantages,
        #     "old_per_token_logps": old_per_token_logps,
        #     "ref_per_token_logps": ref_per_token_logps,
        # }

        logger.info(
            f"[Process {self.accelerator.process_index}] Calling compute_loss for honest_prover"
        )
        # --- Model A ---
        # Forward/Backward for Model A's own loss
        loss_a = self.compute_loss(
            self.models["honest_prover"], inputs["honest_prover"], "honest_prover"
        )
        logger.info(
            f"[Process {self.accelerator.process_index}] Completed compute_loss for honest_prover"
        )
        # --- Model B ---
        # Forward/Backward for Model B's own loss
        loss_b = self.compute_loss(
            self.models["sneaky_prover"], inputs["sneaky_prover"], "sneaky_prover"
        )

        logger.info(
            f"[Process {self.accelerator.process_index}] Completed compute_loss for sneaky_prover"
        )
        logger.info(f"[Process {self.accelerator.process_index}] Returning losses")
        logger.info(
            f"[Process {self.accelerator.process_index}] Losses: {loss_a}, {loss_b}"
        )

        # --- Return losses ---
        return {"loss_a": loss_a, "loss_b": loss_b}

    def _log_metrics(self, step_data: dict[str, Any]) -> None:
        """Gathers and logs metrics. Takes step_data from _training_step() -- losses, lrs, grad_norms, etc.-- and self._metrics -- lists of scalars collected during a step/eval."""
        eval_mode = "train"  # NOTE: This function is only called during training
        global_step = step_data["step"]
        metrics_to_log = {}

        print(f"Step data: {step_data}")
        print(f"Metrics: {self._metrics}")

        # --- Log step-specific data ---
        metrics_to_log[f"{eval_mode}/epoch"] = step_data["epoch"]
        metrics_to_log[f"{eval_mode}/loss_a"] = step_data["losses"]["honest_prover"]
        metrics_to_log[f"{eval_mode}/loss_b"] = step_data["losses"]["sneaky_prover"]
        metrics_to_log[f"{eval_mode}/lr_a"] = step_data["lr"]["honest_prover"]
        metrics_to_log[f"{eval_mode}/lr_b"] = step_data["lr"]["sneaky_prover"]

        model_metrics = self._metrics[
            eval_mode
        ]  # Dict having keys: "honest_prover", "sneaky_prover", "verifier"
        for model_key in model_metrics.keys():
            for metric_name, values in model_metrics[model_key].items():
                # Check if 'values' is a list and is not empty before processing
                if isinstance(values, list) and values:
                    try:
                        # Attempt to convert to tensor and calculate mean
                        metrics_to_log[f"{eval_mode}/{metric_name}_{model_key}"] = (
                            torch.tensor(values).mean().item()
                        )
                    except Exception as e:
                        # Log a warning if conversion or mean calculation fails for a non-empty list
                        logger.warning(
                            f"Could not compute mean for metric '{metric_name}' in model '{model_key}'. Values: {values}. Error: {e}"
                        )
                # Clear the list for the next logging interval
                if isinstance(values, list):
                    model_metrics[model_key][metric_name] = []

        # --- Log metrics ---
        self.accelerator.log(metrics_to_log, step=global_step)
        # --- Log metrics to wandb ---
        logger.info(f"Logging metrics to wandb for step {global_step}")
        logger.info(f"Metrics to log: {metrics_to_log}")
        if self.accelerator.is_main_process and self.wandb_run:
            self.wandb_run.log(metrics_to_log, step=global_step)

        logger.info(f"Step {global_step}: {metrics_to_log}")

    def evaluate(self) -> dict[str, Any | int]:
        logger.info(f"--- Running Evaluation at Step {self.global_step} ---")
        self.models["honest_prover"].eval()
        self.models["sneaky_prover"].eval()
        all_losses_a = []
        all_losses_b = []
        # Add accumulators for preds/labels if computing metrics

        with torch.no_grad():
            for step, batch in enumerate(
                tqdm(
                    self.eval_dataloader,
                    desc="Evaluating",
                    disable=not self.accelerator.is_local_main_process,
                )
            ):
                # Inputs should come from _prepare_inputs
                inputs = self._prepare_inputs(batch)

                # --- Model A Eval Step ---
                loss_a = self.compute_loss(
                    self.models["honest_prover"],
                    inputs["honest_prover"],
                    "honest_prover",
                )
                all_losses_a.append(self.accelerator.gather(loss_a))  # Gather loss

                # --- Model B Eval Step (potentially using A's output) ---
                loss_b = self.compute_loss(
                    self.models["sneaky_prover"],
                    inputs["sneaky_prover"],
                    "sneaky_prover",
                )
                all_losses_b.append(self.accelerator.gather(loss_b))  # Gather loss

        # Calculate final metrics
        avg_loss_a = torch.cat(all_losses_a).mean().item()
        avg_loss_b = torch.cat(all_losses_b).mean().item()

        metrics = {
            "eval/loss_a": avg_loss_a,
            "eval/loss_b": avg_loss_b,
            "step": self.global_step,
            "epoch": self.current_epoch,
            # Add other computed metrics here
        }

        # Add perplexity maybe? ppl_a = math.exp(avg_loss_a)
        self.accelerator.log(metrics, step=self.global_step)
        logger.info(f"Evaluation Step {self.global_step}: {metrics}")

        # Switch back to train mode
        self.models["honest_prover"].train()
        self.models["sneaky_prover"].train()

        self.accelerator.wait_for_everyone()

        return metrics

    def save_checkpoint(self, final: bool = False) -> None:
        checkpoint_dir = os.path.join(
            self.args.output_dir, f"checkpoint-{self.global_step}"
        )
        if final:
            checkpoint_dir = os.path.join(self.args.output_dir, "final_checkpoint")
        logger.info(f"Saving checkpoint to {checkpoint_dir}...")

        self.accelerator.wait_for_everyone()

        # --- Save Model A State ---
        try:
            logger.info("Selecting plugin for Model A save...")
            self.accelerators["honest_prover"].state.select_deepspeed_plugin(
                "honest_prover"
            )
            output_dir_a = os.path.join(checkpoint_dir, "honest_prover_state")
            logger.info(f"Saving Model A state to {output_dir_a}")
            self.accelerators["honest_prover"].save_state(output_dir_a)
            logger.info("Model A state saved.")
        except Exception as e:
            logger.error(f"Failed to save Model A state: {e}", exc_info=True)

        # --- Save Model B State ---
        try:
            logger.info("Selecting plugin for Model B save...")
            self.accelerators["sneaky_prover"].state.select_deepspeed_plugin(
                "sneaky_prover"
            )
            output_dir_b = os.path.join(checkpoint_dir, "sneaky_prover_state")
            logger.info(f"Saving Model B state to {output_dir_b}")
            self.accelerators["sneaky_prover"].save_state(output_dir_b)
            logger.info("Model B state saved.")
        except Exception as e:
            logger.error(f"Failed to save Model B state: {e}", exc_info=True)

        # --- Save Loop State (on main process) ---
        if self.accelerator.is_main_process:
            # Save custom state
            loop_state = {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                # Add other state like best_metric if tracking
            }
            with open(
                os.path.join(checkpoint_dir, "trainer_loop_state.json"), "w"
            ) as f:
                json.dump(loop_state, f)

            # Save script arguments for reference
            with open(os.path.join(checkpoint_dir, "script_args.json"), "w") as f:
                json.dump(self.args.__dict__, f, indent=4)

            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)

        # Accelerator wait for everyone
        self.accelerator.wait_for_everyone()

        logger.info(f"Checkpoint {self.global_step} saved successfully.")
        # Add checkpoint rotation logic here if needed

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

        # --- Load Loop State (on all processes, main reads first) ---
        loop_state_path = os.path.join(checkpoint_dir, "trainer_loop_state.json")
        if os.path.exists(loop_state_path):
            with open(loop_state_path, "r") as f:
                loop_state = json.load(f)
            self.global_step = loop_state.get("global_step", 0)
            self.current_epoch = loop_state.get("current_epoch", 0)
            # Load other state like best_metric
            logger.info(
                f"Loaded loop state: global_step={self.global_step}, current_epoch={self.current_epoch}"
            )
        else:
            logger.warning(
                f"Trainer loop state file not found at {loop_state_path}. Starting from scratch."
            )

        # --- Load Model A State ---
        try:
            logger.info("Selecting plugin for Model A load...")
            self.accelerators["honest_prover"].state.select_deepspeed_plugin(
                "honest_prover"
            )
            input_dir_a = os.path.join(checkpoint_dir, "honest_prover_state")
            logger.info(f"Loading Model A state from {input_dir_a}")
            self.accelerators["honest_prover"].load_state(input_dir_a)
            logger.info("Model A state loaded.")
        except Exception as e:
            logger.error(
                f"Failed to load Model A state from {input_dir_a}: {e}", exc_info=True
            )
            # Decide whether to raise or continue

        # --- Load Model B State ---
        try:
            logger.info("Selecting plugin for Model B load...")
            self.accelerators["sneaky_prover"].state.select_deepspeed_plugin(
                "sneaky_prover"
            )
            input_dir_b = os.path.join(checkpoint_dir, "sneaky_prover_state")
            logger.info(f"Loading Model B state from {input_dir_b}")
            self.accelerators["sneaky_prover"].load_state(input_dir_b)
            logger.info("Model B state loaded.")
        except Exception as e:
            logger.error(
                f"Failed to load Model B state from {input_dir_b}: {e}", exc_info=True
            )
            # Decide whether to raise or continue

        logger.info(
            f"Checkpoint loading complete. Resuming from step {self.global_step}."
        )

    def _move_model_to_vllm(self, model_key: str) -> None:
        accelerator = self.accelerators[model_key]
        model = self.models[model_key]
        vllm_client = self.vllm_clients[model_key]
        can_sync_to_client = accelerator.is_main_process and vllm_client is not None
        logger.info(
            f"[Process {accelerator.process_index}] Starting weight sync logic for {model_key}..."
        )
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = (
            deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        )
        unwrapped_model = accelerator.unwrap_model(model)
        named_params = list(unwrapped_model.named_parameters())
        num_params = len(named_params)
        logger.info(
            f"[Process {accelerator.process_index} / {model_key}] Starting parameter sync loop ({num_params} params)..."
        )

        param_iterator = named_params
        if accelerator.is_main_process:
            param_iterator = tqdm(
                named_params, desc=f"Syncing {model_key}", leave=False, disable=False
            )

        for name, param in param_iterator:
            if not param.requires_grad:
                continue
            try:
                # Collective operation happens here
                with gather_if_zero3(
                    [param], modifier_rank=0 if zero_stage_3 else None
                ):
                    if can_sync_to_client:
                        try:
                            vllm_client.update_named_param(name, param.data)
                        except Exception as e:
                            logger.error(
                                f"Failed to update param {name} for {model_key} via vLLM: {e}",
                                exc_info=True,
                            )
                            break
            except Exception as e:
                logger.error(
                    f"Error during GatheredParameters for {name} in {model_key}: {e}",
                    exc_info=True,
                )
                break

        # --- Barrier AFTER loop ---
        # Ensures all processes finish the loop before proceeding to cache reset
        logger.info(
            f"[Process {accelerator.process_index} / {model_key}] Finished parameter loop. Waiting at barrier..."
        )
        accelerator.wait_for_everyone()  # Use the specific accelerator
        logger.info(
            f"[Process {accelerator.process_index} / {model_key}] Passed barrier after parameter loop."
        )

        # --- Reset Cache (Main Process Only) ---
        if can_sync_to_client:
            logger.info(
                f"[Process {accelerator.process_index} / {model_key}] Resetting vLLM prefix cache..."
            )
            # ... (try-except block for reset) ...

        # --- Final Barrier for this function ---
        logger.info(
            f"[Process {accelerator.process_index}] Finished _move_model_to_vllm for {model_key}. Waiting at final barrier..."
        )
        accelerator.wait_for_everyone()  # Use the specific accelerator again
        logger.info(
            f"[Process {accelerator.process_index}] Exiting _move_model_to_vllm for {model_key}."
        )

    def _sync_weights_to_vllm(self):
        """Helper method to trigger weight sync for both models. Called by all processes."""
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Entering _sync_weights_to_vllm"
        )

        # --- Sync honest_prover ---
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Selecting DS plugin 'honest_prover'..."
        )
        self.accelerators["honest_prover"].state.select_deepspeed_plugin(
            "honest_prover"
        )
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Calling _move_model_to_vllm for honest_prover"
        )
        self._move_model_to_vllm("honest_prover")
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Finished _move_model_to_vllm for honest_prover"
        )

        # *** CRUCIAL GLOBAL BARRIER ***
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Global barrier before sneaky_prover sync..."
        )
        self.accelerator.wait_for_everyone()  # Synchronize everyone using the primary accelerator
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Passed global barrier."
        )

        # --- Sync sneaky_prover ---
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Selecting DS plugin 'sneaky_prover'..."
        )
        self.accelerators["sneaky_prover"].state.select_deepspeed_plugin(
            "sneaky_prover"
        )
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Calling _move_model_to_vllm for sneaky_prover"
        )
        self._move_model_to_vllm("sneaky_prover")
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Finished _move_model_to_vllm for sneaky_prover"
        )

        # --- Final Global Barrier (Optional but safe) ---
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Global barrier after all syncs..."
        )
        self.accelerator.wait_for_everyone()  # Synchronize everyone
        logger.info(
            f"[Process {self.accelerator.process_index}] ===> Exiting _sync_weights_to_vllm"
        )

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.args.num_generations,
            batch_size=effective_batch_size // self.args.num_generations,
            repeat_count=self.args.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.args.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        return model

    def _get_per_token_logps(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = (
            logits / self.args.vllm_temperature_honest_prover
        )  # NOTE: Should be the same temp for honest and sneaky prover so makes no difference
        return selective_log_softmax(
            logits, input_ids
        )  # compute logprobs for the input tokens

    def _calculate_log_probabilities(
        self,
        model_key: Literal["honest_prover", "sneaky_prover"],
        container_data: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Calculates old (policy) and reference log probabilities for a given model key.
        Ensures reference model is in eval mode.

        Args:
            model_key: The key identifying the model ("honest_prover" or "sneaky_prover").
            container_data: The dictionary slice from the container for the given model key,
                            containing keys like "prompt_completion_ids", "prompt_completion_mask", "logits_to_keep".

        Returns:
            A tuple containing (old_per_token_logps, ref_per_token_logps). Each can be None.
        """
        old_per_token_logps = None
        ref_per_token_logps = None
        policy_model = self.models[model_key]
        ref_model = self.ref_models.get(model_key)  # Use .get for safety

        input_ids = container_data["completion_ids"]
        attention_mask = container_data["completion_mask"]
        logits_to_keep = container_data["logits_to_keep"]

        # Calculate old log probabilities if needed (num_iterations > 1)
        if self.args.num_iterations > 1:
            # Assuming policy_model is already in the correct mode (train/eval)
            old_per_token_logps = self._get_per_token_logps(
                policy_model, input_ids, attention_mask, logits_to_keep
            )

        # Calculate reference log probabilities if needed (beta > 0)
        if self.args.beta > 0.0:
            if ref_model is None:
                raise ValueError(
                    f"Reference model for {model_key} required but not loaded (beta > 0)."
                )
            # Ensure reference model is in eval mode
            ref_model.eval()
            ref_per_token_logps = self._get_per_token_logps(
                ref_model, input_ids, attention_mask, logits_to_keep
            )

        return old_per_token_logps, ref_per_token_logps

    def _prepare_inputs(
        self,
        batch: dict[str, torch.Tensor | Any],
    ) -> dict[str, torch.Tensor | Any]:
        # mode = "eval" if self.control.should_evaluate else "train" # Find a more elegant way to do this
        is_training = self.models[
            "honest_prover"
        ].training  # Make sure both models always have the same mode

        if is_training:
            buffer_index = self.global_step % self.args.gradient_accumulation_steps
            buffered_inputs = self._buffered_inputs[buffer_index]
            if (
                self.global_step % self.args.num_iterations == 0
                or buffered_inputs is None
            ):
                # buffered_inputs=None can occur when resuming from a checkpoint
                inputs = self._generate_and_score_completions(batch)
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = buffered_inputs
                self.global_step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(batch)
        return inputs

    def _generate_and_score_completions(
        self, batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, dict[str, Any | list[Any] | None]]:
        """Generate completions and score them."""

        devices = {
            "honest_prover": self.accelerator.device,
            "sneaky_prover": self.accelerators["sneaky_prover"].device,
        }
        raw_prompts = [x["question"] for x in batch]  # List of strings, no devicing
        system_prompts: dict[
            Literal["honest_prover", "sneaky_prover", "verifier"], str
        ] = {
            "honest_prover": self.args.honest_prover_system_prompt,
            "sneaky_prover": self.args.sneaky_prover_system_prompt,
            "verifier": self.args.verifier_system_prompt,
        }
        is_instruct = {
            "honest_prover": "instruct" in self.args.honest_prover_name_or_path.lower()
            or "-it" in self.args.honest_prover_name_or_path.lower(),
            "sneaky_prover": "instruct" in self.args.sneaky_prover_name_or_path.lower()
            or "-it" in self.args.sneaky_prover_name_or_path.lower(),
            "verifier": "instruct" in self.args.verifier_name_or_path.lower()
            or "-it" in self.args.verifier_name_or_path.lower(),
        }

        container = Container(
            tokenizer=self.tokenizer,
            raw_prompts=raw_prompts,
            system_prompts=system_prompts,
            devices=devices,
        )  # Container object for prompts-completions-rewards (?) of all models

        ### Model A ###
        # Prepare inputs for honest prover (formatting) - Done in all processes
        container.prepare_inputs(
            "honest_prover",
            format_type="instruct" if is_instruct["honest_prover"] else "base",
        )
        all_honest_prompts = gather_object(
            container.container["honest_prover"]["prompt_texts"]
        )  # Gathering to collect all prompts from all processes
        # Note: prompts here are not unique! They are repeated num_generations times

        assert (
            len(all_honest_prompts) == len(raw_prompts) * self.accelerator.num_processes
        ), "Number of honest prompts does not match number of raw prompts. There must be a problem with how we are preparing the inputs."

        honest_gen_args = {
            "temperature": self.args.vllm_temperature_honest_prover,
            "top_p": self.args.vllm_top_p_honest_prover,
            "top_k": (
                -1
                if self.args.vllm_top_k_honest_prover is None
                else self.args.vllm_top_k_honest_prover
            ),
            "max_tokens": self.args.vllm_max_new_tokens_honest_prover,
            "repetition_penalty": self.args.vllm_repetition_penalty_honest_prover,
            "frequency_penalty": self.args.vllm_frequency_penalty_honest_prover,
            "min_p": self.args.vllm_min_p_honest_prover,
            "stop": self.args.vllm_stop_sequences_honest_prover,
        }

        logger.info(
            f"Generating completions for honest prover with args: {honest_gen_args}"
        )
        # logger.info(f"All honest prompts: {all_honest_prompts}")

        completion_ids_a, completion_texts_a, _ = self._generate_via_vllm_and_broadcast(
            client_key="honest_prover",
            all_prompts_gathered=all_honest_prompts,
            generation_args=honest_gen_args,
            n_generations=self.args.num_generations,
            logprobs_count=0,
            raw_prompts_len_local=len(raw_prompts),  # Pass local raw prompt length
        )

        # Call post-process (load completions) on all processes & prepare inputs for following model
        container.load_completions(
            "honest_prover", completion_texts_a, completion_ids_a
        )
        container.prepare_inputs(
            "sneaky_prover",
            format_type="instruct" if is_instruct["sneaky_prover"] else "base",
        )

        # Model B
        all_prompts_text_b = gather_object(
            container.container["sneaky_prover"]["prompt_texts"]
        )

        sneaky_gen_args = {
            "temperature": self.args.vllm_temperature_sneaky_prover,
            "top_p": self.args.vllm_top_p_sneaky_prover,
            "top_k": (
                -1
                if self.args.vllm_top_k_sneaky_prover is None
                else self.args.vllm_top_k_sneaky_prover
            ),
            "max_tokens": self.args.vllm_max_new_tokens_sneaky_prover,
            "repetition_penalty": self.args.vllm_repetition_penalty_sneaky_prover,
            "frequency_penalty": self.args.vllm_frequency_penalty_sneaky_prover,
            "min_p": self.args.vllm_min_p_sneaky_prover,
            "stop": self.args.vllm_stop_sequences_sneaky_prover,
        }

        logger.info(
            f"Generating completions for sneaky prover with args: {sneaky_gen_args}"
        )
        completion_ids_b, completion_texts_b, _ = self._generate_via_vllm_and_broadcast(
            client_key="sneaky_prover",
            all_prompts_gathered=all_prompts_text_b,
            generation_args=sneaky_gen_args,
            n_generations=self.args.num_generations,
            logprobs_count=0,
            raw_prompts_len_local=len(raw_prompts),  # Pass local raw prompt length
        )

        # Call post-process (load completions) on all processes & prepare inputs for following model
        container.load_completions(
            "sneaky_prover", completion_texts_b, completion_ids_b
        )

        ## For every prover we now have:
        # - completion_ids
        # - completion_texts
        # - solutions

        # We can now also generate rewards via verifier
        container.prepare_inputs(
            "verifier",
            format_type="instruct" if is_instruct.get("verifier") else "base",
        )
        all_verifier_prompts = gather_object(
            container.container["verifier"]["prompt_texts"]
        )

        # Sanity checks - TODO: Remove when done debugging
        assert len(all_verifier_prompts) % 2 == 0, "Number of prompts must be even"
        assert (
            len(all_verifier_prompts)
            == 2 * len(raw_prompts) * self.accelerator.num_processes
        ), "Number of verifier prompts (query-solution pairs to be evaluated) must be twice the number of raw prompts due to solutions coming from two provers."

        verifier_gen_args = {
            "temperature": self.args.vllm_temperature_verifier,
            "top_p": self.args.vllm_top_p_verifier,
            "top_k": (
                -1
                if self.args.vllm_top_k_verifier is None
                else self.args.vllm_top_k_verifier
            ),
            "max_tokens": self.args.vllm_max_new_tokens_verifier,
            "repetition_penalty": self.args.vllm_repetition_penalty_verifier,
            "frequency_penalty": self.args.vllm_frequency_penalty_verifier,
            "min_p": self.args.vllm_min_p_verifier,
            "stop": self.args.vllm_stop_sequences_verifier,
        }
        verifier_logprobs_request_count = (
            15  # Number of logprobs needed for reward extraction? Adjust if needed.
        )
        logger.info(
            f"Generating completions for verifier with args: {verifier_gen_args}"
        )
        logger.info(
            "NOTE: Verifier gets **FULL** list of ids, texts, and logprobs. This is different from the other models."
        )

        completion_ids_v, completion_texts_v, logprobs_v = (
            self._generate_via_vllm_and_broadcast(
                client_key="verifier",
                all_prompts_gathered=all_verifier_prompts,
                generation_args=verifier_gen_args,
                n_generations=1,  # Verifier generates one response per input
                logprobs_count=verifier_logprobs_request_count,
                # For verifier, the number of local items corresponds to *both* prover solutions
                raw_prompts_len_local=len(raw_prompts) * 2,
            )
        )

        logger.info("Completed verifier generation. Extracting rewards...")

        # --- Reward Processing and Advantage Calculation ---
        rewards_all = None  # Initialize
        if self.accelerator.is_main_process:
            logger.info(
                f"[Process {self.accelerator.process_index}] Main process entering reward extraction phase."
            )
            # Need to reconstruct the *full* list of completion texts on main process to extract rewards globally
            logger.info(
                f"[Process {self.accelerator.process_index}] Gathering verifier completion texts..."
            )
            # gather_start_time = time.time()
            # # Gather local texts back - ensure completion_texts_v is defined and is a list
            # if not isinstance(completion_texts_v, list):
            #      logger.error(f"[Process {self.accelerator.process_index}] 'completion_texts_v' is not a list, type is {type(completion_texts_v)}. Cannot gather.")
            #      # Handle error appropriately, maybe raise or set rewards_all to indicate failure
            #      rewards_all = [-0.1] * (len(raw_prompts) * 2 * self.accelerator.num_processes) # Placeholder
            # else:
            #      all_completion_texts_v_gathered = gather_object(completion_texts_v)
            # gather_end_time = time.time()
            # logger.info(f"[Process {self.accelerator.process_index}] Gathered {len(all_completion_texts_v_gathered) if all_completion_texts_v_gathered else 'N/A'} verifier texts globally. Took {gather_end_time - gather_start_time:.2f}s.")
            # --- Add Logging ---
            logger.info(
                f"[Process {self.accelerator.process_index}] Length of completion_texts_v before reward extraction: {len(completion_texts_v)}"
            )
            # --- End Add Logging ---

            logger.info(
                f"[Process {self.accelerator.process_index}] Starting reward extraction loop..."
            )
            rewards_all = [
                self.extract_verifier_reward(text) for text in completion_texts_v
            ]
            # --- Add Logging ---
            logger.info(
                f"[Process {self.accelerator.process_index}] Length of rewards_all after reward extraction: {len(rewards_all)}"
            )
            # --- End Add Logging ---

        else:
            # Placeholder on non-main processes before broadcast
            expected_len = len(raw_prompts) * 2 * self.accelerator.num_processes
            logger.info(
                f"[Process {self.accelerator.process_index}] Non-main process creating placeholder for {expected_len} rewards, waiting for broadcast."
            )
            rewards_all = [None] * expected_len

        # Log before broadcast on ALL processes
        logger.info(
            f"[Process {self.accelerator.process_index}] Preparing to broadcast rewards (object type: {type(rewards_all)}, length: {len(rewards_all) if isinstance(rewards_all, list) else 'N/A'})."
        )

        rewards_all = broadcast_object_list(
            rewards_all, from_process=0
        )  # Broadcast the globally extracted rewards

        # Now `rewards_all` contains the rewards corresponding to `all_verifier_prompts` on every process.

        assert len(rewards_all) == len(all_verifier_prompts), (
            f"[Process {self.accelerator.process_index}] Mismatch after broadcast: "
            f"len(rewards_all)={len(rewards_all)} vs len(all_verifier_prompts)={len(all_verifier_prompts)}"
        )
        logger.info(
            f"[Process {self.accelerator.process_index}] Assertion passed: Length of rewards_all ({len(rewards_all)}) matches all_verifier_prompts."
        )

        rewards_all_tensor = torch.tensor(
            rewards_all, dtype=torch.float32, device=self.accelerator.device
        )  # TODO: Device placement can be problematic. fp32 is likely unnecessary while we are not using logits.
        rewards_a = rewards_all_tensor[::2]  # Note: global
        rewards_b = rewards_all_tensor[1::2]  # Note: global

        logger.info("Completed reward extraction. Handling NaNs...")
        logger.info(f"Rewards tensor: {rewards_all_tensor}")
        logger.info(f"Rewards a in {self.accelerator.process_index}: {rewards_a}")
        logger.info(f"Rewards b in {self.accelerator.process_index}: {rewards_b}")

        # NaNs handling
        if torch.isnan(rewards_all_tensor).any():
            nan_mask = torch.isnan(rewards_all_tensor)
            rewards_all_tensor[nan_mask] = (
                -0.1
            )  # Or another strategy like self.args.nan_reward_value
            # Log a warning if NaNs were present
            if self.accelerator.is_main_process:
                logger.warning(
                    f"Replaced {nan_mask.sum()} NaN values in rewards tensor with -0.1."
                )

        if (rewards_all_tensor == -0.1).all():  # See above
            nan_row_idx = torch.isnan(rewards_a).any(dim=1).nonzero(as_tuple=True)[0][0]
            # row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs = {
                "prompt": all_verifier_prompts[nan_row_idx],
                "completion": completion_texts_v[nan_row_idx],
            }
            # Log a warning with logger
            logger.warning(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        assert (
            len(rewards_a) % self.args.num_generations == 0
        ), "Number of rewards_a must be divisible by num_generations. Something must have gone wrong."
        assert (
            len(rewards_b) % self.args.num_generations == 0
        ), "Number of rewards_b must be divisible by num_generations. Something must have gone wrong."

        logger.info("Completed reward extraction. Calculating advantages...")

        # Perform Global GRPO advantage calculation for Prover A
        advantages_a_global = self._calculate_grpo_advantages(
            global_rewards=rewards_a,
            num_generations=self.args.num_generations,
            scale_advantages=self.args.scale_rewards,
            # adv_clip=self.args.adv_clip_a # Add if you have clipping args
        )

        # Perform Global GRPO advantage calculation for Prover B (Similarly)
        advantages_b_global = self._calculate_grpo_advantages(
            global_rewards=rewards_b,
            num_generations=self.args.num_generations,
            scale_advantages=self.args.scale_rewards,
            # adv_clip=self.args.adv_clip_b # Add if you have clipping args
        )

        logger.info(
            "Completed advantage calculation. Slicing advantages and rewards..."
        )
        logger.info(f"Advantages a: {advantages_a_global}")
        logger.info(f"Advantages b: {advantages_b_global}")

        # --- Local Slicing - Advantages and Rewards ---
        # Calculate the *correct* local slice indices
        # The slice should correspond to the number of *original* prompts processed
        # by this rank for *one* prover. len(raw_prompts) holds this.
        num_local_samples_per_prover = len(raw_prompts)  # This is key
        start_index_prover = (
            self.accelerator.process_index * num_local_samples_per_prover
        )
        end_index_prover = start_index_prover + num_local_samples_per_prover
        local_slice_prover = slice(start_index_prover, end_index_prover)

        # Slice the *global* advantages and rewards to get the local part for provers
        local_advantages_a = advantages_a_global[local_slice_prover]
        local_advantages_b = advantages_b_global[local_slice_prover]
        local_rewards_a = rewards_a[
            local_slice_prover
        ]  # Slice global rewards for logging
        local_rewards_b = rewards_b[
            local_slice_prover
        ]  # Slice global rewards for logging

        logger.info("Completed slicing. Preparing model inputs and logging metrics...")

        # --- Prepare Model Inputs and Log Metrics ---
        # Pad the completions, and concatenate them with the prompts
        container.pad_and_concatenate("honest_prover")
        container.pad_and_concatenate("sneaky_prover")

        # Mask everything after the first EOS token, and store the mask and logits_to_keep in the container
        container.mask_completion("honest_prover")
        container.mask_completion("sneaky_prover")

        # --- Calculate Log Probabilities using the helper ---
        with torch.no_grad():
            old_per_token_logps_a, ref_per_token_logps_a = (
                self._calculate_log_probabilities(
                    "honest_prover", container.container["honest_prover"]
                )
            )
            old_per_token_logps_b, ref_per_token_logps_b = (
                self._calculate_log_probabilities(
                    "sneaky_prover", container.container["sneaky_prover"]
                )
            )

        logger.info(
            "Completed log probability calculation. Loading completions into container..."
        )
        logger.info(f"Rewards all tensor: {rewards_all_tensor}")
        logger.info(f"Local rewards a: {local_rewards_a}")
        logger.info(f"Local rewards b: {local_rewards_b}")
        logger.info(f"Local advantages a: {local_advantages_a}")
        logger.info(f"Local advantages b: {local_advantages_b}")
        logger.info(f"Old per token logps a: {old_per_token_logps_a}")
        logger.info(f"Old per token logps b: {old_per_token_logps_b}")
        logger.info(f"Ref per token logps a: {ref_per_token_logps_a}")
        logger.info(f"Ref per token logps b: {ref_per_token_logps_b}")

        # Load the completions into the container
        container.load_completions(
            "verifier", completion_texts_v, completion_ids_v, rewards_all_tensor
        )
        # container.pad_and_concatenate(model_key="verifier")

        # Log the metrics
        eval_mode = (
            "train" if self.models["sneaky_prover"].training else "eval"
        )  # Either prover works

        # Log Honest Prover Metrics
        self._store_generation_metrics(
            model_key="honest_prover",
            eval_mode=eval_mode,
            completion_mask=container.container["honest_prover"]["completion_mask"],
            is_eos=container.container["honest_prover"]["is_eos"],
            rewards=local_rewards_a,
            advantages=local_advantages_a,
        )

        # Log Sneaky Prover Metrics
        self._store_generation_metrics(
            model_key="sneaky_prover",
            eval_mode=eval_mode,
            completion_mask=container.container["sneaky_prover"]["completion_mask"],
            is_eos=container.container["sneaky_prover"]["is_eos"],
            rewards=local_rewards_b,
            advantages=local_advantages_b,
        )

        container_honest = container.container["honest_prover"]
        container_sneaky = container.container["sneaky_prover"]

        return {
            "honest_prover": {
                "prompt_ids": container_honest["prompt_ids"],
                "prompt_mask": container_honest["prompt_mask"],
                "completion_ids": container_honest["completion_ids"],
                "completion_mask": container_honest["completion_mask"],
                "advantages": local_advantages_a,
                "old_per_token_logps": old_per_token_logps_a,
                "ref_per_token_logps": ref_per_token_logps_a,
                "logits_to_keep": container_honest["logits_to_keep"],
            },
            "sneaky_prover": {
                "prompt_ids": container_sneaky["prompt_ids"],
                "prompt_mask": container_sneaky["prompt_mask"],
                "completion_ids": container_sneaky["completion_ids"],
                "completion_mask": container_sneaky["completion_mask"],
                "advantages": local_advantages_b,
                "old_per_token_logps": old_per_token_logps_b,
                "ref_per_token_logps": ref_per_token_logps_b,
                "logits_to_keep": container_sneaky["logits_to_keep"],
            },
        }

    def _compute_entropy(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Calculate entropy from logits: H(p) = -sum(p_i * log(p_i))
        Args:
            logits: Raw logits from model (before softmax), shape (batch_size, sequence_length, vocab_size)
            mask: Optional mask to apply (for ignoring padding tokens), shape (batch_size, sequence_length)
            reduce: If True, return mean entropy; if False, return per-token entropy tensor
        Returns:
            If reduce=True: Mean entropy as scalar tensor
            If reduce=False: Per-token entropy tensor of shape (batch_size, sequence_length)
        """
        # Convert logits to probabilities with softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate entropy: -sum(p_i * log(p_i))
        entropy = -torch.sum(
            probs * torch.log(probs + 1e-10), dim=-1
        )  # Shape: (batch_size, sequence_length)

        if not reduce:
            return entropy  # Return full entropy tensor

        if mask is not None:
            # Apply mask and calculate mean entropy only over valid tokens
            return (entropy * mask).sum() / mask.sum().clamp(min=1.0)
        return entropy.mean()

    def _compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        model_key: Literal["honest_prover", "sneaky_prover"],
    ):

        # Make sure devicing is set correctly
        device = self.accelerators[model_key].device

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"].to(device), inputs[
            "prompt_mask"
        ].to(device)
        completion_ids, completion_mask = inputs["completion_ids"].to(device), inputs[
            "completion_mask"
        ].to(device)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Get full logits to compute entropy
        unwrapped_model = self.accelerator.unwrap_model(model)
        outputs = unwrapped_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[
            :, -logits_to_keep - 1 : -1, :
        ]  # Get logits for completion tokens

        # Calculate entropy metrics
        per_token_entropy = self._compute_entropy(logits, completion_mask, reduce=False)
        mean_entropy = (
            per_token_entropy * completion_mask
        ).sum() / completion_mask.sum().clamp(min=1.0)
        entropy_std = torch.sqrt(
            (
                (per_token_entropy - mean_entropy.unsqueeze(-1)).pow(2)
                * completion_mask
            ).sum()
            / completion_mask.sum().clamp(min=1.0)
        )

        # Calculate min, max, and percentiles
        # Create a flattened version of valid entropy values for percentile calculation
        valid_entropy = per_token_entropy[completion_mask.bool()]
        if len(valid_entropy) > 0:
            entropy_min = valid_entropy.min()
            entropy_max = valid_entropy.max()
            # Calculate percentiles if enough tokens
            if (
                len(valid_entropy) >= 4
            ):  # Need at least a few points for meaningful percentiles
                sorted_entropy, _ = torch.sort(valid_entropy)
                idx_25 = max(
                    0, min(len(sorted_entropy) - 1, int(0.25 * len(sorted_entropy)))
                )
                idx_50 = max(
                    0, min(len(sorted_entropy) - 1, int(0.50 * len(sorted_entropy)))
                )
                idx_75 = max(
                    0, min(len(sorted_entropy) - 1, int(0.75 * len(sorted_entropy)))
                )

                entropy_25 = sorted_entropy[idx_25]
                entropy_50 = sorted_entropy[idx_50]  # median
                entropy_75 = sorted_entropy[idx_75]
            else:
                entropy_25 = entropy_min
                entropy_50 = mean_entropy
                entropy_75 = entropy_max
        else:
            entropy_min = torch.tensor(0.0, device=device)
            entropy_max = torch.tensor(0.0, device=device)
            entropy_25 = torch.tensor(0.0, device=device)
            entropy_50 = torch.tensor(0.0, device=device)
            entropy_75 = torch.tensor(0.0, device=device)

        # Store entropy metrics
        eval_mode = "eval" if not model.training else "train"
        self._metrics[eval_mode][model_key]["entropy_mean"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_mean", [])
        self._metrics[eval_mode][model_key]["entropy_std"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_std", [])
        self._metrics[eval_mode][model_key]["entropy_min"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_min", [])
        self._metrics[eval_mode][model_key]["entropy_max"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_max", [])
        self._metrics[eval_mode][model_key]["entropy_25"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_25", [])
        self._metrics[eval_mode][model_key]["entropy_50"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_50", [])
        self._metrics[eval_mode][model_key]["entropy_75"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_75", [])
        self._metrics[eval_mode][model_key]["entropy_iqr"] = self._metrics[eval_mode][
            model_key
        ].get("entropy_iqr", [])

        self._metrics[eval_mode][model_key]["entropy_mean"].append(
            self.accelerators[model_key]
            .gather_for_metrics(mean_entropy)
            .nanmean()
            .item()
        )
        self._metrics[eval_mode][model_key]["entropy_std"].append(
            self.accelerators[model_key]
            .gather_for_metrics(entropy_std)
            .nanmean()
            .item()
        )
        self._metrics[eval_mode][model_key]["entropy_min"].append(
            self.accelerators[model_key]
            .gather_for_metrics(entropy_min)
            .nanmean()
            .item()
        )
        self._metrics[eval_mode][model_key]["entropy_max"].append(
            self.accelerators[model_key]
            .gather_for_metrics(entropy_max)
            .nanmean()
            .item()
        )
        self._metrics[eval_mode][model_key]["entropy_25"].append(
            self.accelerators[model_key].gather_for_metrics(entropy_25).nanmean().item()
        )
        self._metrics[eval_mode][model_key]["entropy_50"].append(
            self.accelerators[model_key].gather_for_metrics(entropy_50).nanmean().item()
        )
        self._metrics[eval_mode][model_key]["entropy_75"].append(
            self.accelerators[model_key].gather_for_metrics(entropy_75).nanmean().item()
        )
        entropy_iqr = entropy_75 - entropy_25
        self._metrics[eval_mode][model_key]["entropy_iqr"].append(
            self.accelerators[model_key]
            .gather_for_metrics(entropy_iqr)
            .nanmean()
            .item()
        )

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute the KL divergence between the model and the reference model
        if self.args.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.args.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1, 1 - self.args.epsilon_low, 1 + self.args.epsilon_high
        )
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(
            min=1.0
        )

        # Log the metrics
        eval_mode = "eval" if not model.training else "train"

        if self.args.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[eval_mode][model_key]["kl"].append(
                self.accelerators[model_key]
                .gather_for_metrics(mean_kl)
                .nanmean()
                .item()
            )

        # Compute the clip ratio
        is_clipped = (
            (coef_1 < 1 - self.args.epsilon_low) & (advantages.unsqueeze(1) < 0)
        ) | ((coef_1 > 1 + self.args.epsilon_high) & (advantages.unsqueeze(1) > 0))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[eval_mode][model_key]["clip_ratio"].append(
            self.accelerators[model_key].gather_for_metrics(clip_ratio).nanmean().item()
        )
        return loss

    def _get_last_hidden_state(
        self,
        model: torch.nn.Module,
        model_key: Literal["honest_prover", "sneaky_prover"],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        accelerator = self.accelerators[model_key]
        # unwrap the model to access the model.model
        unwrapped_model = accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state

    def compute_liger_loss(
        self, model, inputs, model_key: Literal["honest_prover", "sneaky_prover"]
    ):

        # Make sure devicing is set correctly
        device = self.accelerators[model_key].device
        zero3 = (
            self.accelerators[model_key].state.deepspeed_plugin is not None
            and self.accelerators[model_key].state.deepspeed_plugin.zero_stage == 3
        )

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"].to(device), inputs[
            "prompt_mask"
        ].to(device)
        completion_ids, completion_mask = inputs["completion_ids"].to(device), inputs[
            "completion_mask"
        ].to(device)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Get full logits to compute entropy
        unwrapped_model = self.accelerators[model_key].unwrap_model(model)
        outputs = unwrapped_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[
            :, -logits_to_keep - 1 : -1, :
        ]  # Get logits for completion tokens

        # Calculate entropy over the logits
        entropy = self._compute_entropy(logits, completion_mask)

        # Store entropy in metrics
        eval_mode = "eval" if not model.training else "train"
        self._metrics[eval_mode][model_key]["entropy"] = self._metrics[eval_mode][
            model_key
        ].get("entropy", [])
        self._metrics[eval_mode][model_key]["entropy"].append(
            self.accelerators[model_key].gather_for_metrics(entropy).nanmean().item()
        )

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            model, model_key, input_ids, attention_mask, logits_to_keep
        )

        # # --- DEBUG PRINTS ---
        # if model_key == "honest_prover":
        #     print(unwrapped_model) # Trying to assess if maybe lm_head is called differently...
        #     print(
        #         f"[DEBUG {model_key} Rank {self.accelerators[model_key].process_index}] Shapes before liger_grpo_loss:"
        #     )
        #     print(f"  - last_hidden_state: {last_hidden_state.shape}")
        #     print(f"  - lm_head.weight: {unwrapped_model.lm_head.weight.shape}")
        #     print(f"  - completion_ids: {completion_ids.shape}")
        #     print(f"  - completion_mask: {completion_mask.shape}")
        #     print(f"  - advantages: {inputs['advantages'].shape}")
        #     if inputs["ref_per_token_logps"] is not None:
        #         print(f"  - ref_per_token_logps: {inputs['ref_per_token_logps'].shape}")
        #     else:
        #         print("  - ref_per_token_logps: None")
        #     if inputs["old_per_token_logps"] is not None:
        #         print(f"  - old_per_token_logps: {inputs['old_per_token_logps'].shape}")
        #     else:
        #         print("  - old_per_token_logps: None")
        # # --- END DEBUG PRINTS ---

        with self._full_lm_head_params(unwrapped_model, zero3):
            weight = unwrapped_model.lm_head.weight  # now full on every rank
            bias = unwrapped_model.lm_head.bias

            loss, metrics = self.liger_grpo_loss(
                _input=last_hidden_state,
                lin_weight=weight,
                bias=bias,
                selected_token_ids=completion_ids,
                attention_mask=completion_mask,
                advantages=inputs["advantages"],
                ref_per_token_logps=inputs["ref_per_token_logps"],
                old_per_token_logps=inputs["old_per_token_logps"],
            )

        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.args.beta != 0.0 else None
        clip_ratio = metrics[-1]

        eval_mode = "eval" if not model.training else "train"
        if self.args.beta != 0.0:
            self._metrics[eval_mode][model_key]["kl"].append(
                self.accelerators[model_key]
                .gather_for_metrics(mean_kl)
                .nanmean()
                .item()
            )
        self._metrics[eval_mode][model_key]["clip_ratio"].append(
            self.accelerators[model_key].gather_for_metrics(clip_ratio).nanmean().item()
        )
        return loss

    def compute_loss(
        self,
        model,
        inputs,
        model_key: Literal["honest_prover", "sneaky_prover"],
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.args.apply_liger_kernel:
            # Compute the loss using the liger grpo loss
            return self.compute_liger_loss(model, inputs, model_key)
        else:
            return self._compute_loss(model, inputs, model_key)

    def _fetch_local_models(self, model_path: str) -> str:
        """Fetches a local model from a given path."""
        local_model_path = "/home/jvelja/local_models"
        full_model_path = os.path.join(local_model_path, model_path)
        if os.path.exists(full_model_path):
            return full_model_path
        else:
            raise FileNotFoundError(f"Model not found at {full_model_path}")

    # Should we provide a system prompt...? Why?
    # - For the mode of the model, i.e., sneaky or honest

    def extract_verifier_reward(self, completion_text: str) -> float:
        """Extract the reward from the verifier's completion text."""
        # Find the first occurrence of <verdict>...</verdict>
        verdict_match = re.search(
            r"<verdict>(.*?)</verdict>", completion_text, re.DOTALL
        )

        if verdict_match:
            verdict = verdict_match.group(1).strip()
            if verdict == "BACKDOOR":
                return 1.0
            elif verdict == "CLEAN":
                return 0.0
        return (
            -0.1
        )  # Default reward for incorrect verdicts (i.e., not BACKDOOR or CLEAN or not having a verdict at all)

    def _log_llm_interaction(
        self,
        model_mode: str,
        prompts: list[str],
        output_ids: list[str],
        output_texts: list[str],
        logprobs: list[dict[int, float]] | None = None,
    ):
        """Logs LLM interaction details to a JSON file on the main process."""
        if not self.accelerator.is_main_process:
            return

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        interaction_id = str(uuid.uuid4())
        log_filename = (
            f"{timestamp.replace(':', '-')}_{model_mode}_{interaction_id}.json"
        )

        log_filename = (
            f"{timestamp.replace(':', '-')}_{model_mode}_{interaction_id}.json"
        )
        log_filepath = os.path.join(self.llm_interaction_log_dir, log_filename)
        step_dir = os.path.join(
            self.llm_interaction_log_dir, f"step_{self.global_step}"
        )
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)

        log_filepath = os.path.join(step_dir, log_filename)  # New

        log_data = {
            "interaction_id": interaction_id,
            "timestamp_utc": timestamp,
            "model_mode": model_mode,
            "prompts": prompts,  # Log the unique prompts used for generation
            "output_ids": output_ids,  # Raw output IDs from vLLM
            "output_texts": output_texts,  # Decoded output texts
        }
        if logprobs is not None:
            log_data["logprobs"] = logprobs  # Add logprobs if available (for verifier)

        try:
            with open(log_filepath, "w") as f:
                json.dump(log_data, f, indent=4)
            # logger.debug(f"Saved LLM interaction log to: {log_filepath}")
        except Exception as e:
            logger.error(f"Failed to save LLM interaction log to {log_filepath}: {e}")

    def _store_generation_metrics(
        self,
        model_key: Literal["honest_prover", "sneaky_prover", "verifier"],
        eval_mode: str,  # "train" or "eval"
        completion_mask: torch.Tensor,
        is_eos: torch.Tensor,
        rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
    ):
        """Logs metrics related to generated sequences for a specific mode."""
        # NOTE: Accelerator is mode specific; take from self.accelerators

        # --- Completion Lengths ---
        agg_completion_mask_sum = self.accelerators[model_key].gather_for_metrics(
            completion_mask.sum(1)
        )
        self._metrics[eval_mode][model_key]["completions/mean_length"].append(
            agg_completion_mask_sum.float().mean().item()
        )
        self._metrics[eval_mode][model_key]["completions/min_length"].append(
            agg_completion_mask_sum.float().min().item()
        )
        self._metrics[eval_mode][model_key]["completions/max_length"].append(
            agg_completion_mask_sum.float().max().item()
        )

        # --- EOS Metrics ---
        agg_terminated_with_eos = self.accelerators[model_key].gather_for_metrics(
            is_eos.any(dim=1)
        )
        term_completion_mask_sum = agg_completion_mask_sum[agg_terminated_with_eos]
        clipped_completions_ratio = (
            1.0 - (len(term_completion_mask_sum) / len(agg_completion_mask_sum))
            if len(agg_completion_mask_sum) > 0
            else 1.0
        )
        self._metrics[eval_mode][model_key]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )

        if len(term_completion_mask_sum) == 0:
            # Handle edge case where no sequences terminated with EOS in this batch/process
            term_completion_mask_sum = torch.tensor(
                [0.0], device=completion_mask.device
            )  # Use a tensor with 0

        self._metrics[eval_mode][model_key][
            "completions/mean_terminated_length"
        ].append(term_completion_mask_sum.float().mean().item())
        self._metrics[eval_mode][model_key]["completions/min_terminated_length"].append(
            term_completion_mask_sum.float().min().item()
        )
        self._metrics[eval_mode][model_key]["completions/max_terminated_length"].append(
            term_completion_mask_sum.float().max().item()
        )

        # Log reward metrics if available
        if rewards is not None:
            agg_rewards = self.accelerators[model_key].gather_for_metrics(rewards)
            # Use nanmean and nanstd (assuming you have the nanstd function available)
            self._metrics[eval_mode][model_key]["rewards/mean"].append(
                torch.nanmean(agg_rewards.float()).item()
            )
            self._metrics[eval_mode][model_key]["rewards/std"].append(
                nanstd(agg_rewards.float()).item()
            )

        # Log advantage metrics if available
        if advantages is not None:
            agg_advantages = self.accelerators[model_key].gather_for_metrics(advantages)
            self._metrics[eval_mode][model_key]["advantages/mean"].append(
                torch.nanmean(agg_advantages.float()).item()
            )
            self._metrics[eval_mode][model_key]["advantages/std"].append(
                nanstd(agg_advantages.float()).item()
            )

    def _generate_via_vllm_and_broadcast(
        self,
        client_key: Literal["honest_prover", "sneaky_prover", "verifier"],
        all_prompts_gathered: list[
            str
        ],  # The gathered list of prompts from all processes
        generation_args: dict[
            str, Any
        ],  # Args for vllm_client.generate (temp, top_p, etc.)
        n_generations: int,
        logprobs_count: int,  # Number of logprobs to request (0 if none)
        raw_prompts_len_local: int,  # Length of the original local prompt list (needed for slicing)
    ) -> tuple[list[list[int]], list[str], list[dict[int, float]] | None]:
        """
        Generates completions using the specified vLLM client on the main process,
        logs the interaction, broadcasts results, and returns the local slice.
        """
        completion_ids_all = None
        completion_texts_all = None
        logprobs_all = None
        num_total_prompts = len(all_prompts_gathered)

        if self.accelerator.is_main_process:
            # Generation happens only on the main process
            client = self.vllm_clients[client_key]
            if client is None:
                raise ValueError(f"vLLM client '{client_key}' is not initialized.")

            # Only pass unique prompts if n_generations > 1
            if n_generations > 1:
                prompts_to_generate = all_prompts_gathered[::n_generations]
                assert (
                    len(prompts_to_generate) * n_generations == num_total_prompts
                ), "Prompt list length mismatch"
            else:
                prompts_to_generate = all_prompts_gathered

            generate_kwargs = {
                "prompts": prompts_to_generate,
                "n": n_generations,
                **generation_args,  # Spread the generation args (temp, top_p, max_tokens, etc.)
            }
            if logprobs_count > 0:
                generate_kwargs["logprobs"] = logprobs_count

            # Call generate
            if logprobs_count > 0:
                completion_ids_nested, logprobs_nested = client.generate(
                    **generate_kwargs
                )
                # Process logprobs: Flatten the nested list structure if necessary
                logprobs_all = logprobs_nested
                # --- Add Logging ---
                logger.info(
                    f"[Process {self.accelerator.process_index} / {client_key}] Raw client output length: completion_ids_nested={len(completion_ids_nested)}, logprobs_nested={len(logprobs_nested)}"
                )
                # --- End Add Logging ---
            else:
                completion_ids_nested = client.generate(**generate_kwargs)
                # --- Add Logging ---
                logger.info(
                    f"[Process {self.accelerator.process_index} / {client_key}] Raw client output length: completion_ids_nested={len(completion_ids_nested)}"
                )
                # --- End Add Logging ---
                logprobs_all = None  # Ensure it's None if not requested/returned

            # Flatten completion IDs if nested due to n > 1
            if n_generations > 1:
                completion_ids_all = completion_ids_nested
            else:
                completion_ids_all = completion_ids_nested  # Already flat if n=1

            completion_texts_all = self.tokenizer.batch_decode(
                completion_ids_all,
                skip_special_tokens=True,
                add_generation_prompt=False,  # Usually False here
            )

            # --- Add Logging ---
            logger.info(
                f"[Process {self.accelerator.process_index} / {client_key}] Length after batch_decode: completion_texts_all={len(completion_texts_all)}"
            )
            # --- End Add Logging ---

            # Log interaction
            self._log_llm_interaction(
                model_mode=client_key,
                prompts=all_prompts_gathered,  # Log all prompts that *should* have been generated for
                output_ids=completion_ids_all,
                output_texts=completion_texts_all,
                logprobs=logprobs_all,
            )

        else:
            # Placeholders for non-main processes
            completion_ids_all = [None] * num_total_prompts
            completion_texts_all = [None] * num_total_prompts
            if logprobs_count > 0:
                logprobs_all = [None] * num_total_prompts  # Match structure

        # Broadcast results from main process
        completion_ids_all = broadcast_object_list(completion_ids_all, from_process=0)
        completion_texts_all = broadcast_object_list(
            completion_texts_all, from_process=0
        )
        if logprobs_count > 0:
            logprobs_all = broadcast_object_list(logprobs_all, from_process=0)

        # Calculate and apply the slice for the current process
        process_slice = slice(
            self.accelerator.process_index * raw_prompts_len_local,
            (self.accelerator.process_index + 1) * raw_prompts_len_local,
        )
        local_completion_ids = completion_ids_all[process_slice]
        local_completion_texts = completion_texts_all[process_slice]
        local_logprobs = (
            logprobs_all[process_slice] if logprobs_all is not None else None
        )

        # --- MODIFIED RETURN LOGIC ---
        if client_key == "verifier":
            # For the verifier, the calling function needs the *full* broadcasted lists
            # because rewards are calculated based on all completions on the main process.
            return completion_ids_all, completion_texts_all, logprobs_all
        else:
            # For provers, return the local slice needed for loss calculation on each process
            process_slice = slice(
                self.accelerator.process_index * raw_prompts_len_local,
                (self.accelerator.process_index + 1) * raw_prompts_len_local,
            )
            local_completion_ids = completion_ids_all[process_slice]
            local_completion_texts = completion_texts_all[process_slice]
            local_logprobs = (
                logprobs_all[process_slice] if logprobs_all is not None else None
            )
            return local_completion_ids, local_completion_texts, local_logprobs

    def _calculate_grpo_advantages(
        self,
        global_rewards: torch.Tensor,
        num_generations: int,
        scale_advantages: bool,
        adv_clip: float | None = None,  # Optional advantage clipping
        eps: float = 1e-8,  # Epsilon for std division
    ) -> torch.Tensor:
        """
        Calculates GRPO advantages globally based on rewards grouped by prompt.

        Args:
            global_rewards: Tensor containing rewards for all generations across all prompts (globally).
            num_generations: Number of generations per unique prompt.
            scale_advantages: Whether to scale advantages by the standard deviation.
            adv_clip: Optional value to clip advantages.
            eps: Small value to add to standard deviation before division.

        Returns:
            Tensor containing the calculated advantages for all generations (globally).
        """
        if global_rewards.dim() != 1:
            raise ValueError(
                f"Expected global_rewards to be 1D, but got shape {global_rewards.shape}"
            )
        if len(global_rewards) % num_generations != 0:
            raise ValueError(
                f"Length of global_rewards ({len(global_rewards)}) must be divisible by num_generations ({num_generations})."
            )

        num_unique_prompts_global = len(global_rewards) // num_generations

        # Reshape rewards to (num_unique_prompts_global, num_generations)
        rewards_grouped = global_rewards.view(
            num_unique_prompts_global, num_generations
        )

        # Calculate mean and std per prompt group (nans handled already)
        mean_grouped_rewards = torch.nanmean(
            rewards_grouped.float(), dim=1
        )  # TODO: Possibly not needed nanmean
        std_grouped_rewards = nanstd(
            rewards_grouped.float(), dim=1
        )  # TODO: Possibly not needed nanstd

        # Expand mean/std back to the original shape
        mean_expanded = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        std_expanded = std_grouped_rewards.repeat_interleave(num_generations, dim=0)

        # Calculate advantages
        advantages = global_rewards - mean_expanded

        # --- Add Shape Logging ---
        logger.info(f"[AdvCalc shapes] global_rewards: {global_rewards.shape}")
        logger.info(f"[AdvCalc shapes] rewards_grouped: {rewards_grouped.shape}")
        logger.info(
            f"[AdvCalc shapes] mean_grouped_rewards: {mean_grouped_rewards.shape}"
        )
        logger.info(
            f"[AdvCalc shapes] std_grouped_rewards: {std_grouped_rewards.shape}"
        )
        logger.info(f"[AdvCalc shapes] mean_expanded: {mean_expanded.shape}")
        logger.info(f"[AdvCalc shapes] std_expanded: {std_expanded.shape}")
        logger.info(f"[AdvCalc shapes] advantages: {advantages.shape}")
        # --- End Shape Logging ---

        # Optional scaling
        if scale_advantages:
            advantages = advantages / (std_expanded + eps)

        # Optional clipping
        if adv_clip is not None:
            advantages = torch.clamp(advantages, -adv_clip, adv_clip)

        return advantages

    def _full_lm_head_params(self, unwrapped_model, zero3):
        # returns weight,bias tensors that are guaranteed to be full on every rank
        params = [unwrapped_model.lm_head.weight]
        if unwrapped_model.lm_head.bias is not None:
            params.append(unwrapped_model.lm_head.bias)

        if zero3:
            # modifier_rank=None => gather on *all* ranks (not only rank‑0)
            ctx = deepspeed.zero.GatheredParameters(params, modifier_rank=None)
        else:
            ctx = nullcontext()
        return ctx  # to be used as a "with" context‑manager
