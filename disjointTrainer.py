import logging
import os
import json
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from transformers import (
    set_seed,
    AutoModelForCausalLM,  # Assuming Causal LM for now
    AutoTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling,  # Example collator
    PreTrainedModel,
)
from torch.utils.data import DataLoader, Dataset  # Add Dataset import
from tqdm.auto import tqdm  # For progress bars
from vllmclient import VLLMClient
from torch.utils.data import Sampler
from contextlib import nullcontext
import deepspeed
from rep_sampler import RepeatRandomSampler
from trl.utils import selective_log_softmax
from utils import ScriptArguments, get_args

# Missing:
# Liger kernel (from liger_kernel.transformers import apply_liger_kernel_to_instance)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- The Trainer Class ---
class DisjointSequentialTrainer:
    def __init__(self, args: ScriptArguments):
        self.args: ScriptArguments = args
        self.accelerators: dict[str, Accelerator] = {}
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.dataloaders: dict[str, torch.utils.data.DataLoader] = {}
        self.deepspeed_plugins: dict[str, DeepSpeedPlugin] = {}
        self.vllm_client_a = None  # Initialize to None
        self.vllm_client_b = None  # Initialize to None
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # State variables
        self.global_step = 0
        self.current_epoch = 0

        # --- 1. Logging & Seeding ---
        self._setup_logging()
        self._set_seed()

        # --- 2. Accelerator Initialization ---
        self._initialize_accelerators()
        # Use accelerator_a for general state/logging, device info etc.
        self.accelerator = self.accelerators["honest_prover"]

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

        # --- Instantiate vLLM Clients (Main Process Only) ---
        if self.accelerator.is_main_process:
            logger.info("Main process initializing vLLM clients...")
            try:
                self.vllm_client_a = VLLMClient(
                    host=self.args.vllm_host_a,
                    server_port=self.args.vllm_port_a,
                    connection_timeout=self.args.vllm_server_timeout,
                )
                logger.info(
                    f"vLLM Client A connected to {self.args.vllm_host_a}:{self.args.vllm_port_a}"
                )
            except ConnectionError as e:
                logger.error(
                    f"Failed to connect to vLLM server A: {e}. Ensure it's running and accessible.",
                    exc_info=True,
                )
                raise  # Or handle gracefully depending on requirements
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred initializing vLLM Client A: {e}",
                    exc_info=True,
                )
                raise

            try:
                self.vllm_client_b = VLLMClient(
                    host=self.args.vllm_host_b,
                    server_port=self.args.vllm_port_b,
                    connection_timeout=self.args.vllm_server_timeout,
                )
                logger.info(
                    f"vLLM Client B connected to {self.args.vllm_host_b}:{self.args.vllm_port_b}"
                )
            except ConnectionError as e:
                logger.error(
                    f"Failed to connect to vLLM server B: {e}. Ensure it's running and accessible.",
                    exc_info=True,
                )
                raise
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred initializing vLLM Client B: {e}",
                    exc_info=True,
                )
                raise

            try:
                self.vllm_client_c = VLLMClient(
                    host=self.args.vllm_host_c,
                    server_port=self.args.vllm_port_c,
                    connection_timeout=self.args.vllm_server_timeout,
                )
                logger.info(
                    f"vLLM Client C connected to {self.args.vllm_host_c}:{self.args.vllm_port_c}"
                )
            except ConnectionError as e:
                logger.error(
                    f"Failed to connect to vLLM server C: {e}. Ensure it's running and accessible.",
                    exc_info=True,
                )
                raise
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred initializing vLLM Client C: {e}",
                    exc_info=True,
                )
                raise
        # --- End vLLM Client Instantiation ---
        logger.info("vLLM Clients initialized.")
        logger.info("--- vLLM Clients State ---")
        logger.info(f"  vLLM Client A: {self.vllm_client_a}")
        logger.info(f"  vLLM Client B: {self.vllm_client_b}")
        logger.info(f"  vLLM Client C: {self.vllm_client_c}")

        # --- 3. Load Tokenizer, Models, Datasets, Optimizers ---
        self._load_tokenizer()
        self._load_models()
        self.train_dataloader, self.eval_dataloader = self._load_dataloaders()
        self._create_optimizers()
        # Calculate num_training_steps after dataloader is prepared
        self.num_training_steps = self._calculate_num_training_steps()
        # Now create schedulers with the correct number of steps
        self._create_schedulers(self.num_training_steps)

        # --- 4. Prepare Components ---
        self._prepare_components()  # This will use self.accelerators and update self.models etc.

        # --- 5. Load from Checkpoint (if specified) ---
        if self.args.resume_from_checkpoint:
            self.load_checkpoint(self.args.resume_from_checkpoint)

        # --- 6. Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations ---
        num_processes = self.accelerator.num_processes
        global_batch_size = self.args.train_batch_size * num_processes
        possible_values = [
            n_gen
            for n_gen in range(2, global_batch_size + 1)
            if (global_batch_size) % n_gen == 0
        ]
        if self.args.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {self.args.train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.args.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )

        logger.info("--- Initialization Complete ---")

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

    def _load_tokenizer(self) -> AutoTokenizer:
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
        logger.info(
            f"Loading Honest Prover from {self.args.honest_prover_name_or_path}"
        )
        model_a = AutoModelForCausalLM.from_pretrained(
            self.args.honest_prover_name_or_path, **model_init_kwargs
        )
        logger.info(
            f"Loading Sneaky Prover from {self.args.sneaky_prover_name_or_path}"
        )
        model_b = AutoModelForCausalLM.from_pretrained(
            self.args.sneaky_prover_name_or_path, **model_init_kwargs
        )
        logger.info(f"Loading Verifier from {self.args.verifier_name_or_path}")
        # TODO: Load Verifier
        verifier = AutoModelForCausalLM.from_pretrained(
            self.args.verifier_name_or_path, **model_init_kwargs
        )

        self.models = {
            "honest_prover": model_a,
            "sneaky_prover": model_b,
            "verifier": verifier,
        }

        # Reference models for RL
        if self.args.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        # elif is_deepspeed_zero3_enabled():
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.args.honest_prover_name_or_path
            )

    def _load_dataloaders(self):
        # --- This is a placeholder - Replace with your actual dataset loading ---
        logger.warning(
            "Using placeholder dataset loading. Replace with your actual data loading logic."
        )

        # Example: Simple list dataset
        class PlaceholderDataset(Dataset):
            def __init__(self, tokenizer, num_samples=1000, max_length=50):
                self.num_samples = num_samples
                self.tokenizer = tokenizer
                self.max_length = max_length
                # Generate some dummy text
                self.texts = [
                    f"This is sample text number {i} for training."
                    for i in range(num_samples)
                ]

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                # Remove batch dimension added by tokenizer
                encoding = {k: v.squeeze(0) for k, v in encoding.items()}
                # Create labels (usually shifted input_ids for LM)
                encoding["labels"] = encoding["input_ids"].clone()
                return encoding

        train_dataset = PlaceholderDataset(
            self.tokenizer, num_samples=100
        )  # Small for demo
        eval_dataset = (
            PlaceholderDataset(self.tokenizer, num_samples=50)
            if self.args.validation_file
            else None
        )
        # ---------------------------------------------------------------------

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )  # TODO: # Data collator

        # def data_collator(features):  # No data collation is needed in GRPO
        #     return features

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,  # Shuffle for training
            collate_fn=data_collator,
            batch_size=self.args.train_batch_size,
        )
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                shuffle=False,  # No shuffle for eval
                collate_fn=data_collator,
                batch_size=self.args.eval_batch_size,
            )

        logger.info(f"Train Dataloader: {len(train_dataloader)} batches")
        if eval_dataloader:
            logger.info(f"Eval Dataloader: {len(eval_dataloader)} batches")

        return train_dataloader, eval_dataloader

    def _create_optimizers(self) -> None:
        # Simple AdamW optimizer setup - can be customized
        optimizer_a = torch.optim.AdamW(
            self.models["honest_prover"].parameters(),
            lr=self.args.learning_rate_a,
            weight_decay=self.args.weight_decay_a,
        )
        optimizer_b = torch.optim.AdamW(
            self.models["sneaky_prover"].parameters(),
            lr=self.args.learning_rate_b,
            weight_decay=self.args.weight_decay_b,
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

    def _prepare_components(self) -> None:
        logger.info("Preparing components with Accelerate...")
        # Prepare Model A components
        logger.info("Selecting plugin for Model A...")
        self.accelerators["honest_prover"].state.select_deepspeed_plugin(
            "honest_prover"
        )
        logger.info("Preparing Model A, Optimizer A, Train Dataloader...")
        prepared_a = self.accelerators["honest_prover"].prepare(
            self.models["honest_prover"],
            self.optimizers["honest_prover"],
            self.train_dataloader,
            self.schedulers["honest_prover"],
        )
        (
            self.models["honest_prover"],
            self.optimizers["honest_prover"],
            self.train_dataloader,
        ) = (prepared_a[0], prepared_a[1], prepared_a[2])

        # Prepare Model B components
        logger.info("Selecting plugin for Model B...")
        # Use accelerator_b's state object, though it points to the same shared state
        self.accelerators["sneaky_prover"].state.select_deepspeed_plugin(
            "sneaky_prover"
        )
        logger.info("Preparing Model B, Optimizer B...")
        prepared_b = self.accelerators["sneaky_prover"].prepare(
            self.models["sneaky_prover"],
            self.optimizers["sneaky_prover"],
            self.schedulers["sneaky_prover"],
        )
        self.models["sneaky_prover"], self.optimizers["sneaky_prover"] = (
            prepared_b[0],
            prepared_b[1],
        )

        # Prepare Eval Dataloader (using accelerator_a context is fine)
        if self.eval_dataloader:
            logger.info("Preparing Eval Dataloader...")
            self.eval_dataloader = self.accelerators["honest_prover"].prepare(
                self.eval_dataloader
            )

        logger.info("Component preparation complete.")

    def _prepare_schedulers(self) -> None:
        # Prepare schedulers separately after num_training_steps is known
        if self.schedulers["honest_prover"] is not None:
            logger.info("Preparing Scheduler A...")
            self.schedulers["honest_prover"] = self.accelerators[
                "honest_prover"
            ].prepare(self.schedulers["honest_prover"])
        if self.schedulers["sneaky_prover"] is not None:
            logger.info("Preparing Scheduler B...")
            self.schedulers["sneaky_prover"] = self.accelerators[
                "sneaky_prover"
            ].prepare(self.schedulers["sneaky_prover"])
        logger.info("Scheduler preparation complete.")

    # --- Placeholder Methods ---
    def train(self):
        logger.info("***** Starting Training *****")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps}"
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

            for step, batch in enumerate(self.train_dataloader):
                # --- vLLM Generation & Training Step ---
                # Generation happens on main process, results broadcast if needed
                # Loss calculation and backward happen on all processes
                losses = self._training_step(batch)

                # --- Optimizer Step ---
                is_sync_step = (
                    (step + 1) % self.args.gradient_accumulation_steps == 0
                ) or (step + 1 == len(self.train_dataloader))
                if is_sync_step:
                    # Clip Gradients (Optional)
                    if self.args.max_grad_norm_a is not None:
                        self.accelerators["honest_prover"].clip_grad_norm_(
                            self.models["honest_prover"].parameters(),
                            self.args.max_grad_norm_a,
                        )
                    if self.args.max_grad_norm_b is not None:
                        self.accelerators["sneaky_prover"].clip_grad_norm_(
                            self.models["sneaky_prover"].parameters(),
                            self.args.max_grad_norm_b,
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

                    self.global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss_a": losses["loss_a"].item(),
                            "loss_b": losses["loss_b"].item(),
                        }
                    )  # Use .item() carefully

                    # --- Logging ---
                    if self.global_step % self.args.logging_steps == 0:
                        self._log_metrics(losses)  # Pass detached losses

                    # --- Evaluation ---
                    if (
                        self.eval_dataloader
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        self.evaluate()

                    # --- Checkpointing ---
                    if self.global_step % self.args.save_steps == 0:
                        self.save_checkpoint()

                    # --- Weight Synchronization ---
                    # Decide when to sync weights (e.g., every N steps)
                    if self.global_step % self.args.sync_steps == 0:
                        self._sync_weights_to_vllm()

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
        Returns detached losses and potentially the generated data.
        """
        # --- Model A ---
        # Generate intermediate output (no gradients needed for this part)
        # This needs specific implementation based on how B uses A's output
        # with torch.no_grad():
        #     # Example: Get hidden states or generate text
        #     # intermediate_output_a = self.models["honest_prover"](...) # Or .generate()
        #     # For now, let's assume B doesn't need A's output directly in this step
        #     intermediate_output_a = None  # Placeholder

        # Forward/Backward for Model A's own loss
        outputs_a = self.models["honest_prover"](**batch)
        loss_a = outputs_a.loss
        # Normalize loss for accumulation
        loss_a_scaled = loss_a / self.args.gradient_accumulation_steps
        self.accelerators["honest_prover"].backward(loss_a_scaled)

        # --- Model B ---
        # Construct inputs for B (using batch and potentially intermediate_output_a)
        inputs_b = batch  # Placeholder - modify as needed
        # if intermediate_output_a is not None:
        #     inputs_b['encoder_hidden_states'] = intermediate_output_a # Example

        outputs_b = self.models["sneaky_prover"](**inputs_b)
        loss_b = outputs_b.loss
        # Normalize loss for accumulation
        loss_b_scaled = loss_b / self.args.gradient_accumulation_steps
        self.accelerators["sneaky_prover"].backward(loss_b_scaled)

        # Return detached losses for logging
        return {"loss_a": loss_a.detach(), "loss_b": loss_b.detach()}

    def _log_metrics(self, losses: dict[str, torch.Tensor]) -> None:
        """Gathers and logs metrics."""
        # Gather losses across processes
        gathered_loss_a = self.accelerator.gather(losses["loss_a"]).mean()
        gathered_loss_b = self.accelerator.gather(losses["loss_b"]).mean()

        log_data = {
            "step": self.global_step,
            "epoch": self.current_epoch,
            "train/loss_a": gathered_loss_a.item(),
            "train/loss_b": gathered_loss_b.item(),
        }
        # Add learning rates
        if self.schedulers["honest_prover"]:
            log_data["train/lr_a"] = self.schedulers["honest_prover"].get_last_lr()[0]
        else:
            log_data["train/lr_a"] = self.optimizers["honest_prover"].param_groups[0][
                "lr"
            ]  # Fallback if no scheduler

        if self.schedulers["sneaky_prover"]:
            log_data["train/lr_b"] = self.schedulers["sneaky_prover"].get_last_lr()[0]
        else:
            log_data["train/lr_b"] = self.optimizers["sneaky_prover"].param_groups[0][
                "lr"
            ]  # Fallback

        self.accelerator.log(log_data, step=self.global_step)
        if self.accelerator.is_main_process:
            logger.debug(f"Step {self.global_step}: {log_data}")

    def evaluate(self):
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
                # --- Model A Eval Step ---
                outputs_a = self.models["honest_prover"](**batch)
                loss_a = outputs_a.loss
                all_losses_a.append(self.accelerator.gather(loss_a))  # Gather loss

                # --- Model B Eval Step (potentially using A's output) ---
                # intermediate_output_a = ... # Generate if needed
                inputs_b = batch  # Placeholder
                outputs_b = self.models["sneaky_prover"](**inputs_b)
                loss_b = outputs_b.loss
                all_losses_b.append(self.accelerator.gather(loss_b))  # Gather loss

                # Gather preds/labels here if needed for metrics

        # Calculate final metrics
        avg_loss_a = torch.cat(all_losses_a).mean().item()
        avg_loss_b = torch.cat(all_losses_b).mean().item()

        metrics = {
            "eval/loss_a": avg_loss_a,
            "eval/loss_b": avg_loss_b,
            # Add other computed metrics here
        }
        # Add perplexity maybe? ppl_a = math.exp(avg_loss_a)

        self.accelerator.log(metrics, step=self.global_step)
        logger.info(f"Evaluation Step {self.global_step}: {metrics}")

        # Switch back to train mode
        self.models["honest_prover"].train()
        self.models["sneaky_prover"].train()
        return metrics

    def save_checkpoint(self, final=False):
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
            # Save RNG states
            self.rng_tracker.save_state(os.path.join(checkpoint_dir, "rng_state.pth"))

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

        logger.info(f"Checkpoint {self.global_step} saved successfully.")
        # Add checkpoint rotation logic here if needed

    def load_checkpoint(self, checkpoint_dir):
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

    def _move_model_to_vllm(self, model_key: str):
        """Synchronizes weights from the training model to the corresponding vLLM server."""
        if not self.accelerator.is_main_process:
            return  # Only main process interacts with vLLM client

        logger.info(f"Synchronizing weights for {model_key} to its vLLM server...")
        model_to_sync = self.models[model_key]  # The prepared training model
        vllm_client = (
            self.vllm_client_a if model_key == "honest_prover" else self.vllm_client_b
        )

        if vllm_client is None:
            logger.warning(
                f"vLLM client for {model_key} not initialized. Skipping weight sync."
            )
            raise ValueError(
                f"vLLM client for {model_key} not initialized. Skipping weight sync."
            )

        # Use the provided logic, adapting slightly
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerators[model_key].state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        gather_if_zero3 = (
            deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        )

        # Unwrap the model if necessary (needed for named_parameters)
        # Note: accelerator.unwrap_model might be needed depending on DeepSpeed/FSDP wrapping
        unwrapped_model = self.accelerators[model_key].unwrap_model(model_to_sync)

        logger.info(f"Handling standard model weight sync for {model_key}...")
        # For non-PEFT models, gather and update each parameter individually if ZeRO-3
        for name, param in unwrapped_model.named_parameters():
            with gather_if_zero3(
                [param], modifier_rank=0 if zero_stage_3 else None
            ):  # Pass modifier_rank=0 for DS3
                # Ensure we are on the main process *after* gathering if needed
                if self.accelerator.is_main_process:
                    logger.debug(f"[Weight Sync {model_key}] Updating param: {name}")
                    vllm_client.update_named_param(name, param.data)  # Use param.data

        # Reset cache on main process
        if self.accelerator.is_main_process:
            logger.info(f"Resetting vLLM prefix cache for {model_key}.")
            vllm_client.reset_prefix_cache()

        logger.info(f"Weight synchronization for {model_key} complete.")
        self.accelerator.wait_for_everyone()  # Ensure sync before proceeding

    def _sync_weights_to_vllm(self):
        """Helper method to trigger weight sync for both models."""
        self._move_model_to_vllm("honest_prover")
        self._move_model_to_vllm("sneaky_prover")
        # Verifier is frozen and not updated in this step, so we are good like that

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

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
        logits = logits / self.args.temperature
        return selective_log_softmax(
            logits, input_ids
        )  # compute logprobs for the input tokens

    def _prepare_inputs(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            buffer_index = self.global_step % self.args.gradient_accumulation_steps
            buffered_inputs = self._buffered_inputs[buffer_index]
            if (
                self.state.global_step % self.num_iterations == 0
                or buffered_inputs is None
            ):
                # buffered_inputs=None can occur when resuming from a checkpoint
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = buffered_inputs
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Generate completions and score them."""
        # TODO: Implement this method
        pass


# --- Main Execution Logic ---
def run_training():
    script_args = get_args()

    # --- Basic Sanity Checks ---
    if not os.path.exists(script_args.ds_config_a):
        raise FileNotFoundError(
            f"DeepSpeed config for Model A not found at {script_args.ds_config_a}"
        )
    if not os.path.exists(script_args.ds_config_b):
        raise FileNotFoundError(
            f"DeepSpeed config for Model B not found at {script_args.ds_config_b}"
        )
    os.makedirs(script_args.output_dir, exist_ok=True)

    trainer = DisjointSequentialTrainer(args=script_args)
    trainer.train()


if __name__ == "__main__":
    # Create dummy DeepSpeed config files if they don't exist
    ds_configs_exist = True
    if not os.path.exists("ds_config_zero2.json"):
        try:
            with open("ds_config_zero2.json", "w") as f:
                f.write(
                    '{"zero_optimization": {"stage": 2}, "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": "auto", "gradient_clipping": "auto", "fp16": {"enabled": "auto"}}'
                )  # More realistic dummy
            logger.warning("Created dummy ds_config_zero2.json")
        except OSError:
            logger.error("Failed to create dummy ds_config_zero2.json.")
            ds_configs_exist = False
    if not os.path.exists("ds_config_zero3.json"):
        try:
            with open("ds_config_zero3.json", "w") as f:
                f.write(
                    '{"zero_optimization": {"stage": 3}, "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": "auto", "gradient_clipping": "auto", "fp16": {"enabled": "auto"}}'
                )  # More realistic dummy
            logger.warning("Created dummy ds_config_zero3.json")
        except OSError:
            logger.error("Failed to create dummy ds_config_zero3.json.")
            ds_configs_exist = False

    if ds_configs_exist:
        run_training()
    else:
        logger.error("Cannot proceed without DeepSpeed config files.")
