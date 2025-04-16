Here's my code for adversarial training of two models (`sneaky_prover` and `honest_prover`), scored by a third model (`verifier`). It is pretty much done, still quite experimental but has a functioning training loop. I would like to clean up the completion logging functionality given a quick HTML I have written for completion inspection.

What the completion logger does at the moment is that at each training step, it collects completions from honest, sneaky and verifier and stores the relative jsons in a folder. I like this so far, but it can be a lot cleaner, saving within a `step_i` directory (where `i` is the `global_step`), and changing the HTML source code accordingly so that it can ingest a full outputs folder -- which in turn contains step folders, each containing honest_prover, sneaky_prover and verifier completions at that step. I want to be able to inspect a whole training run this way, else it becomes cumbersome and messy.

A checkpoint about what the codebase does is the following:

<checkpoint>
## Project Checkpoint Report: Disjoint Sequential Model Training with vLLM Acceleration

**Date:** April 15, 2025 *(Updated)*

**Version:** 0.2 (Training Loop Functional)

**Prepared By:** Joan & Gemini 2.5 Pro

---

**1. Introduction & Goal**

This project aims to implement an efficient training pipeline for two distinct Large Language Models (LLMs), designated `honest_prover` and `sneaky_prover`, potentially scored/evaluated by a third model (`verifier`). The key characteristics of this pipeline are:

1.  **Sequential Dependency (Implicit):** While not strictly sequential in the training *step*, the overall goal involves interaction (e.g., `sneaky_prover` might learn based on `honest_prover` outputs, `verifier` scores both). Generation for scoring/rewards involves outputs from both provers.
2.  **Disjoint Training:** The models (`honest_prover`, `sneaky_prover`) are trained with separate optimizers, schedulers, and DeepSpeed ZeRO-3 configurations. Their gradients are not shared during the backward pass for their respective losses.
3.  **Distributed Training:** The training process must scale across multiple GPUs, potentially spanning multiple nodes, leveraging the `accelerate` library and backend frameworks like DeepSpeed. This is done by using `accelerate` and DeepSpeed ZeRO-3.
4.  **Accelerated Inference:** Inference for generation (provers) and scoring (verifier) is offloaded to dedicated vLLM server instances.

The ultimate goal is to create a robust and scalable framework suitable for advanced RL training schemes (like GRPO or PPO variants) or other methods requiring fast generation interleaved with gradient updates on separate models.

**2. Problem Statement**

Standard training frameworks like Hugging Face `Trainer` are primarily designed for single-model training or tightly coupled multi-model scenarios (e.g., encoder-decoder). They do not natively support:

*   Orchestrating separate distributed training configurations (like distinct DeepSpeed engines) for multiple models within the same training script.
*   Integrating external, high-throughput inference servers (like vLLM) for intermediate generation steps within the training loop.
*   Managing the specific GPU resource allocation required for separating inference servers from training workers.

Therefore, a custom solution is required to manage the complex interplay between distributed training, sequential generation, and inference offloading.

**3. Chosen Architecture**

The chosen architecture relies on a **custom training loop built directly on the `accelerate` library**, combined with **multiple standalone vLLM server instances**.

*   **Custom Training Loop (`DisjointSequentialTrainer` Class):** Provides maximum flexibility.
*   **`accelerate` Library:** Handles distributed training setup, mixed precision, gradient accumulation, and DeepSpeed integration.
*   **Two `Accelerator` Instances:** Following `accelerate` documentation for disjoint DeepSpeed models (find more details [here](https://huggingface.co/docs/accelerate/usage_guides/deepspeed_multiple_model)), two `Accelerator` instances (`self.accelerators["honest_prover"]`, `self.accelerators["sneaky_prover"]`) are created, managing separate DeepSpeed engines and communicator groups. This is crucial because the backward passes for `honest_prover` and `sneaky_prover` happen independently based on their respective losses.
*   **Multiple vLLM Servers:** Instead of modifying the vLLM server code, separate instances are launched for `honest_prover` and `sneaky_prover`, each pinned to specific GPUs using `CUDA_VISIBLE_DEVICES`. This aligns with the standard vLLM server design and simplifies resource management and weight synchronization.
*   **`VLLMClient`:** A client class is used *only on the main training process* to communicate with the vLLM servers for generation and weight updates. Each client manages its own communication channel with its respective server.
*   **GPU Allocation Strategy:** Remains flexible, typically dedicating GPUs for vLLM servers and separate GPUs for training workers.
    *   **GPU Allocation Strategy (Example: N=1, G=8):**
        *   GPU 0: Dedicated to vLLM server for `honest_prover`.
        *   GPU 1: Dedicated to vLLM server for `sneaky_prover`.
        *   GPU 2: Dedicated to vLLM server for `verifier` (can be assigned to the same GPU as `sneaky_prover` or `honest_prover` as it is supposed to be a smaller model).
        *   GPUs 2-7: Used by `accelerate` for distributed training of `honest_prover` and `sneaky_prover`.
*   **Weight Synchronization:** A mechanism (`_move_model_to_vllm`, called via `_sync_weights_to_vllm`) updates vLLM server weights using `deepspeed.zero.GatheredParameters` for ZeRO-3 compatibility. **Crucially, synchronization between the disjoint sync operations for each model is enforced using global barriers (`self.accelerator.wait_for_everyone()`) within the main training loop.**

**4. Implementation Details**

The core logic is encapsulated within the `DisjointSequentialTrainer` class.

**4.1. Argument Parsing (`FlatExperimentArgs`)**

*   A **single, flattened `dataclass`** (`FlatExperimentArgs`) defines all command-line arguments for simplicity.
*   Includes paths/names for all three models (`honest_prover`, `sneaky_prover`, `verifier`), tokenizer, DeepSpeed configs, dataset details, output directory.
*   Specifies separate training hyperparameters (LR, WD, Grad Norm) for `honest_prover` and `sneaky_prover`.
*   Includes RL-specific hyperparameters (`num_generations`, `num_iterations`, `beta`, `epsilon_low`, `epsilon_high`, `scale_rewards`).
*   Defines vLLM server connection details and generation parameters for all three models.
*   Contains standard training options (seed, steps, checkpointing, precision).
*   Includes system prompts for each model role.
*   **Added WandB configuration arguments** (`wandb_project_name`, `wandb_entity`, `wandb_run_name`, logging frequencies).


**4.2. Initialization (`__init__`)**

The constructor orchestrates the setup:
1.  Stores arguments (`args`).
2.  Initializes state variables (`global_step`, `current_epoch`).
3.  Sets up logging and random seeds (`_setup_logging`, `set_seed`).
4.  **Initializes Accelerators (`_initialize_accelerators`):**
    *   Creates `DeepSpeedPlugin` instances for `honest_prover` and `sneaky_prover`.
    *   Stores them in a dictionary `deepspeed_plugins`.
    *   Creates `accelerator_a = Accelerator(deepspeed_plugin=deepspeed_plugins, ...)` which initializes the shared state.
    *   Creates `accelerator_b = Accelerator()` which inherits the shared state.
    *   Stores both in `self.accelerators`. `self.accelerator` is set to `accelerator_a` for convenience in accessing shared state properties like device or rank.
    *   Logs GPU visibility info for debugging `CUDA_VISIBLE_DEVICES`.

    ```python
    # Snippet: Accelerator Initialization in _initialize_accelerators
    ds_plugin_a = DeepSpeedPlugin(hf_ds_config=self.args.ds_config_honest_prover)
    ds_plugin_b = DeepSpeedPlugin(hf_ds_config=self.args.ds_config_sneaky_prover)
    self.deepspeed_plugins = {"honest_prover": ds_plugin_a, "sneaky_prover": ds_plugin_b}
    # ... project_config setup ...
    accelerator_a = Accelerator(
        deepspeed_plugin=self.deepspeed_plugins,
        log_with="wandb", # ... other args ...
    )
    accelerator_b = Accelerator() # Inherits state
    self.accelerators = {"honest_prover": accelerator_a, "sneaky_prover": accelerator_b}
    self.accelerator = self.accelerators["honest_prover"] # Primary for global ops
    self.accelerators["verifier"] = self.accelerators["honest_prover"] # Verifier uses primary context
    ```
5.  **Initializes WandB (`prepare_wandb`):** Calls `self.accelerator.init_trackers` with project/entity/run details. Retrieves the `wandb.run` object on the main process. Logs configuration and environment details to `wandb.config`.
6.  Sets up LLM interaction log directory.
7.  Logs Accelerator state and GPU visibility.
8.  **Initializes vLLM Clients:** *Only on the main process* (`self.accelerator.is_main_process`), creates `VLLMClient` instances (`self.vllm_client_a`, `self.vllm_client_b`) connecting to the specified server addresses and ports. Includes connection checks and error handling.
    ```python
        # Snippet: vLLM Client Initialization in __init__
        if self.accelerator.is_main_process:
            self._initialize_vllm_client(client_key="honest_prover", ...)
            self._initialize_vllm_client(client_key="sneaky_prover", ...)
            self._initialize_vllm_client(client_key="verifier", ...)
    ```
9.  Loads tokenizer, training models (`honest_prover`, `sneaky_prover`), datasets, and optimizers using helper methods.
10. **Prepares Components (`_prepare_components`, `_prepare_schedulers`):** Critically uses `accelerator.state.select_deepspeed_plugin()` to set the active context before calling `accelerators[key].prepare()` for each model, its optimizer, the dataloaders, and the schedulers. This ensures correct wrapping (DDP/DeepSpeed/FSDP) and device placement according to the respective configurations. Prepares reference models if `beta > 0`.
    ```python
    # Snippet: Component Preparation in _prepare_components
    # Prepare Model A components
    self.accelerators["honest_prover"].state.select_deepspeed_plugin("honest_prover")
    (
        self.models["honest_prover"],
        self.optimizers["honest_prover"],
        self.train_dataloader,
    ) = self.accelerators["honest_prover"].prepare(
        self.models["honest_prover"], self.optimizers["honest_prover"], self.train_dataloader
    )
    # Prepare Model B components (using accelerator_b)
    self.accelerators["sneaky_prover"].state.select_deepspeed_plugin("sneaky_prover")
    (
        self.models["sneaky_prover"],
        self.optimizers["sneaky_prover"],
        self.train_dataloader_b,
    ) = self.accelerators["sneaky_prover"].prepare(
        self.models["sneaky_prover"], self.optimizers["sneaky_prover"], self.train_dataloader_b
    )
    # Prepare Eval Dataloaders
    if self.eval_dataloader:
        self.eval_dataloader = self.accelerators["honest_prover"].prepare(self.eval_dataloader)
        self.eval_dataloader_b = self.accelerators["sneaky_prover"].prepare(self.eval_dataloader_b)
    ```
11. Calculates `num_training_steps` *after* dataloaders are prepared (using `len(self.train_dataloader)`).
12. Creates and prepares schedulers (`_create_schedulers`, `_prepare_schedulers`) using the calculated `num_training_steps`.
13. Loads training state from a checkpoint if `args.resume_from_checkpoint` is provided (`load_checkpoint`).
14. Performs batch size / num_generations check.
15. Initializes the `_metrics` dictionary for collecting metrics during steps.

**4.3. Training Loop (`train`)**

*   Outer loop iterates through epochs, inner loop iterates through batches from `train_dataloader`.
*   Calls `_training_step` to get losses for both models based on the current batch.
*   Scales losses for gradient accumulation.
*   Calls `accelerator.backward()` separately for each model's loss using its corresponding accelerator instance.
*   **Synchronization Point (`if is_sync_step:`):**
    *   Performs gradient clipping using the respective accelerator for each model.
    *   Executes `optimizer.step()` and `scheduler.step()` for both models.
    *   Calls `optimizer.zero_grad()` for both models.
    *   **Crucially, calls `self.accelerator.wait_for_everyone()` to synchronize all processes after optimizer steps.**
    *   Increments `self.global_step`.
    *   Collects step data (losses, LRs, etc.).
    *   Calls `_log_metrics` if `logging_steps` condition met.
    *   Calls `evaluate` if `eval_steps` condition met.
    *   Calls `save_checkpoint` if `save_steps` condition met.
    *   **Calls `_sync_weights_to_vllm` if `sync_steps` condition met.** This function is now called collectively by all processes after the post-optimizer barrier.
    *   **Calls a final `self.accelerator.wait_for_everyone()` at the end of the `is_sync_step` block** to ensure all conditional operations (logging, eval, save, sync) are finished before the next iteration starts.
*   Includes logic to break loops based on `max_train_steps`.
*   Calls `accelerator.end_training()` after the main loop finishes.

**4.4. Core Step Logic (`_training_step`, `_prepare_inputs`, `_generate_and_score_completions`, `compute_loss`)**

*   `_prepare_inputs`: Handles buffering logic based on `num_iterations`. Calls `_generate_and_score_completions`.
*   `_generate_and_score_completions`:
    *   Uses `Container` class to manage prompt formatting based on model role and `is_instruct`.
    *   Calls `_generate_via_vllm_and_broadcast` sequentially for honest prover, sneaky prover, and verifier.
    *   Extracts rewards using `extract_verifier_reward`.
    *   Broadcasts rewards. Handles potential NaNs.
    *   Calculates global GRPO advantages (`_calculate_grpo_advantages`).
    *   Slices advantages/rewards locally.
    *   Pads/masks completions via `Container`.
    *   Calculates old/ref log probabilities (`_calculate_log_probabilities`).
    *   **Stores** generation-related metrics (rewards, advantages, lengths) in `self._metrics` using `_store_generation_metrics`.
    *   Returns the dictionary `inputs` needed for `compute_loss`.
*   `compute_loss`: Calculates the GRPO loss (or Liger fused loss if enabled) using policy/old log probs, advantages, and KL penalty (if `beta > 0`). **Stores** scalar KL and clip ratio metrics in `self._metrics`. Returns the scalar loss tensor.

**4.5. Weight Synchronization (`_move_model_to_vllm`, `_sync_weights_to_vllm`)**

*   `_move_model_to_vllm`:
    *   Called collectively by all processes for a specific `model_key`.
    *   Uses the correct `accelerator` instance for the `model_key`.
    *   Uses `deepspeed.zero.GatheredParameters` context manager around the parameter iteration loop for ZeRO-3 compatibility.
    *   Inside the context, only the main process calls `vllm_client.update_named_param`.
    *   Uses `accelerator.wait_for_everyone()` barriers *after* the parameter loop and at the *end* of the function, using the specific accelerator instance.
    ```python
    # Snippet: Parameter Gathering in _move_model_to_vllm
    for name, param in param_iterator:
        if not param.requires_grad: continue
        try:
            # Collective operation happens here
            with gather_if_zero3([param], modifier_rank=0 if zero_stage_3 else None):
                if can_sync_to_client:
                    vllm_client.update_named_param(name, param.data)
        except Exception as e:
             logger.error(...)
             break
    # ... barriers ...
    ```
*   `_sync_weights_to_vllm`:
    *   Called collectively by all processes.
    *   Selects the appropriate DeepSpeed plugin.
    *   Calls `_move_model_to_vllm` for "honest_prover".
    *   **Calls a global `self.accelerator.wait_for_everyone()` barrier.**
    *   Selects the appropriate DeepSpeed plugin.
    *   Calls `_move_model_to_vllm` for "sneaky_prover".
    *   Calls a final global `self.accelerator.wait_for_everyone()` barrier.

**4.6. Evaluation & Checkpointing (`evaluate`, `save_checkpoint`, `load_checkpoint`)**

*   `evaluate`: Sets models to eval mode, iterates through eval dataloader, calls `_prepare_inputs` and `compute_loss`. Gathers losses and calculates average loss. Logs basic scalar loss metrics via `accelerator.log`. **Currently lacks logging for detailed RL/generation metrics collected during evaluation.**
*   `save_checkpoint`: Uses `accelerator.save_state` for each model's state into separate subdirectories, ensuring DeepSpeed compatibility. Saves loop state and tokenizer on the main process. Includes barriers.
*   `load_checkpoint`: Uses `accelerator.load_state` for each model. Loads loop state on all processes.


**5. Launch Orchestration (Bash Script)**

A launch script (`run_training_vllm.sh`) does the following:
*   It sets `CUDA_VISIBLE_DEVICES` to isolate GPUs for the vLLM server processes (e.g., 0 for Honest Prover Server, 1 for Sneaky Prover Server, 2 for Verifier Server).
*   It launches the `vllm_serve.py` script for each model in the background on their assigned GPUs and ports.
*   It sets `CUDA_VISIBLE_DEVICES` again to isolate the remaining GPUs (e.g., 2-7) for the training processes.
*   It uses `accelerate launch` to start the main training script (`train_disjoint_sequential.py`), passing necessary arguments, including the vLLM server hosts and ports.
*   Includes basic process management (`&`, `wait`, `trap`) for cleanup.

**6. Current Status & Next Steps**

*   **Core training loop is functional:** Generation, loss calculation, backward pass, optimizer updates, and crucially, **weight synchronization with ZeRO-3 appear to be working correctly** due to proper barrier placement.
*   **WandB Logging is Broken:** The primary outstanding issue. The `_log_metrics` function needs a complete overhaul to correctly use `accelerator.log` for scalars and direct `wandb` calls for non-scalars, processing data stored in `self._metrics`. Evaluation logging also needs implementation.
*   **Performance:** Potential bottlenecks exist in the SWA implementation during training and the `GatheredParameters` usage during weight sync.
*   **Code Clarity:** Refactoring large functions like `_generate_and_score_completions` is still recommended for long-term maintainability.
*   **Evaluation:** The `evaluate` method needs enhancement to log detailed RL/generation metrics.

**Next Steps:**
1.  **Fix WandB Logging:** Implement the detailed logging plan using the hybrid `accelerator.log`/`wandb.log` approach.
2.  **Enhance Evaluation:** Add comprehensive metric calculation and logging to the `evaluate` function.
3.  **Investigate SWA Warning:** Check library versions/config to optimize training attention.
4.  **Profile & Optimize:** Measure step time and sync time; consider reducing sync frequency if needed.
5.  **Refactor:** Improve code structure by breaking down large methods.
```
</checkpoint>

And here's the code:

<codebase>
<structure>
```tree
├── pvg/
│   ├── config.yaml
│   ├── disjointTrainer.py
│   ├── test_args.py
│   ├── cleanups.md
│   ├── application.log
│   ├── .env
│   ├── __init__.py
│   ├── ds_config_zero3.json
│   ├── train.py
│   ├── checkpoint.md
│   ├── inspect_docs.md
├── inference/
│   ├── vllmclient.py
│   ├── __init__.py
│   ├── vllm_serve.py
│   ├── __pycache__/
│   │   ├── vllmclient.cpython-311.pyc
│   │   ├── __init__.cpython-311.pyc
├── utils/
│   ├── __init__.py
│   ├── utils.py
│   ├── logger_config.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-311.pyc
│   │   ├── utils.cpython-311.pyc
│   │   ├── logger_config.cpython-311.pyc
├── data/
│   ├── rep_sampler.py
│   ├── prompts.py
│   ├── __init__.py
│   ├── dataset.py
│   ├── __pycache__/
│   │   ├── rep_sampler.cpython-311.pyc
│   │   ├── prompts.cpython-311.pyc
│   │   ├── dataset.cpython-311.pyc
│   │   ├── __init__.cpython-311.pyc
├── __pycache__/
│   ├── vllmclient.cpython-311.pyc
│   ├── rep_sampler.cpython-311.pyc
│   ├── prompts.cpython-311.pyc
│   ├── dataset.cpython-311.pyc
│   ├── __init__.cpython-311.pyc
│   ├── utils.cpython-311.pyc
│   ├── logger_config.cpython-311.pyc
│   ├── disjointTrainer.cpython-311.pyc
```
</structure>

<code>
```disjointTrainer.py
# disjointTrainer.py

import os
import json
from pvg.data.dataset import AppsDataset
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
from trl.trainer.utils import selective_log_softmax
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
import uuid
import datetime
import re
import time
import wandb
import numpy as np # Possibly not needed, let's see

logger = setup_logger(__name__)


# --- The Trainer Class ---
class DisjointSequentialTrainer:
    def __init__(self, args):
        self.args = args
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

        # --- 1. Logging & Seeding ---
        self._setup_logging()
        self._set_seed()

        # --- 2. Accelerator Initialization ---
        self._initialize_accelerators()
        # Use accelerator_a for general state/logging, device info etc.
        self.accelerator = self.accelerators["honest_prover"]
        self.prepare_wandb()

        # Create LLM interaction log directory on main process
        self.llm_interaction_log_dir = os.path.join(
            self.args.output_dir, "llm_interaction_logs"
        )
        if self.accelerator.is_main_process:
            os.makedirs(self.llm_interaction_log_dir, exist_ok=True)
            logger.info(
                f"LLM interaction logs will be saved to: {self.llm_interaction_log_dir}"
            )


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
                host=self.args.vllm_host_honest_prover,
                port=self.args.vllm_port_honest_prover,
                group_port=base_group_port,
                timeout=self.args.vllm_server_timeout,
            )
            # Initialize vLLM clients - Sneaky Prover
            self._initialize_vllm_client(
                client_key="sneaky_prover",
                host=self.args.vllm_host_sneaky_prover,
                port=self.args.vllm_port_sneaky_prover,
                group_port=base_group_port + 1,
                timeout=self.args.vllm_server_timeout,
            )
            # Initialize vLLM clients - Verifier
            self._initialize_vllm_client(
                client_key="verifier",
                host=self.args.vllm_host_verifier,
                port=self.args.vllm_port_verifier,
                group_port=base_group_port + 2,
                timeout=self.args.vllm_server_timeout,
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

        # Dummy: dataloaders (though we won't use them)
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
        self._create_schedulers(self.num_training_steps) # Necessary to do this here, since we need the dataloaders to be prepared before calculating the number of training steps --> Thus schedulers prepared a posteriori

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

        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        # maxlen = (
        #     self.accelerator.num_processes
        #     * self.args.per_device_train_batch_size
        #     * self.args.gradient_accumulation_steps
        # )
        # self._textual_logs = {
        #     "prompt": deque(maxlen=maxlen),
        #     "completion": deque(maxlen=maxlen),
        #     "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        # }

        logger.info("--- Initialization Complete ---")


    def prepare_wandb(self):
        try:
            logger.info("Initializing WandB tracker via accelerator.init_trackers...")
            # Pass project name, config (args), and specific wandb init kwargs
            self.accelerator.init_trackers(
                project_name=self.args.wandb_project_name,
                config=self.args.__dict__, # Log all script args
                init_kwargs={
                    "wandb": {
                        "entity": self.args.wandb_entity,
                        "name": self.args.wandb_run_name # Optional run name
                        # Add other wandb.init args here if needed, e.g., tags, notes
                    }
                }
            )
            logger.info("WandB tracker initialization requested.")
            # Now, immediately try to get the run object on the main process
            if self.accelerator.is_main_process:
                self.wandb_run = self.accelerator.get_tracker("wandb").run
                if self.wandb_run:
                    logger.info(f"Successfully retrieved WandB run. Run ID: {self.wandb_run.id}")
                else:
                    logger.error("Called init_trackers, but failed to retrieve WandB run object.")
        except Exception as e:
            logger.error(f"Error during accelerator.init_trackers or run retrieval: {e}", exc_info=True)
            # Ensure self.wandb_run remains None if init fails
            self.wandb_run = None


        if self.accelerator.is_main_process:
            if not self.accelerator.trackers:
                logger.error("WandB tracker not initialized. Cannot log.")
                self.wandb_run = None # Or raise error
            else:
                self.wandb_run = self.accelerator.get_tracker("wandb").run
                if self.wandb_run is None:
                    logger.error("Could not retrieve WandB run object.")
                else:
                    logger.info(f"WandB tracker initialized. Run ID: {self.wandb_run.id}")
                    # Log initial config (accelerate might do some, but explicit update is safer)
                    self.wandb_run.config.update(self.args.__dict__, allow_val_change=True)

                    # Log environment details
                    try:
                        import importlib.metadata as importlib_metadata
                        import sys
                        import platform
                        libs = ["torch", "transformers", "accelerate", "deepspeed", "vllm", "wandb"]
                        lib_versions = {lib: importlib_metadata.version(lib) for lib in libs if importlib_metadata.version(lib)}
                        self.wandb_run.config.update({
                            "environment/python_version": sys.version,
                            "environment/platform": platform.platform(),
                            "environment/num_processes": self.accelerator.num_processes,
                            "environment/mixed_precision": self.accelerator.mixed_precision,
                            "environment/distributed_type": str(self.accelerator.distributed_type),
                            "environment/library_versions": lib_versions,
                        })
                        logger.info("Environment details logged to WandB.")
                    except Exception as e:
                        logger.warning(f"Could not log all environment details: {e}")


    def _log_model_parameters(self):
        # Log model parameter counts (example for honest prover)
        try:
            for model_key in ["honest_prover", "sneaky_prover"]:
                log_model = self.models[model_key] # Use the unprepared model
                total_params = sum(p.numel() for p in log_model.parameters())
                trainable_params = sum(p.numel() for p in log_model.parameters() if p.requires_grad)
                self.wandb_run.config.update({
                    f"model/{model_key}/total_params": total_params,
                    f"model/{model_key}/trainable_params": trainable_params,
                    f"model/{model_key}/class": log_model.__class__.__name__,
                })

            del log_model # Delete the model from memory (since already loaded into self.models)
            del total_params # Delete the total parameter count from memory
            del trainable_params # Delete the trainable parameter count from memory

        except Exception as e:
            logger.warning(f"Could not log model parameter counts: {e}")

    def _initialize_vllm_client(
        self,
        client_key: str,
        host: str,
        port: int,
        group_port: int | None,
        timeout: float,
    ):
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
        self.accelerators["verifier"] = self.accelerators["honest_prover"] # Hacky?

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

    def _load_datasets(self) -> tuple[AppsDataset, AppsDataset]:
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
    def train(self):
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
                step_start_time = time.time()
                micro_step_metrics = defaultdict(lambda: defaultdict(list)) # Store metrics per micro-batch if needed
                # --- vLLM Generation & Training Step ---
                # Generation happens on main process, results broadcast if needed
                # Loss calculation and backward happen on all processes
                losses = self._training_step(batch)
                loss_a = losses["loss_a"]
                loss_b = losses["loss_b"]

                logger.info(f"[Process {self.accelerator.process_index}] Completed training step")
                logger.info(f"[Process {self.accelerator.process_index}] Losses: {loss_a}, {loss_b}")

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
                logger.info(f"[Process {self.accelerator.process_index}] Completed backward pass")
                logger.info(f"[Process {self.accelerator.process_index}] Losses: {loss_a_scaled}, {loss_b_scaled}")

                # --- Synchronization Point Check ---
                # Determine if this is the last step of an accumulation cycle or the overall last step
                is_last_step_in_batch = (step + 1) == len(self.train_dataloader)
                is_accumulation_boundary = (
                    step + 1
                ) % self.args.gradient_accumulation_steps == 0
                is_sync_step = is_accumulation_boundary or is_last_step_in_batch

                # --- Optimizer Step ---
                if is_sync_step:  # See above for explanation of this
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

                    logger.info(f"[Process {self.accelerator.process_index}] Completed optimizer step")
                    # *** ADD GLOBAL BARRIER HERE ***
                    # Ensure both processes complete optim/sched before proceeding
                    logger.info(f"[Process {self.accelerator.process_index}] Waiting at barrier after optimizer step...")
                    self.accelerator.wait_for_everyone() # Use primary accelerator for global sync
                    logger.info(f"[Process {self.accelerator.process_index}] Passed barrier after optimizer step.")


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
                            "losses" : {
                                "honest_prover" : loss_a.item(),
                                "sneaky_prover" : loss_b.item(),
                            },
                            "step" : self.global_step,
                            "epoch" : self.current_epoch,
                            "lr" : {
                                "honest_prover" : self.schedulers["honest_prover"].get_last_lr()[0] if self.schedulers["honest_prover"] else self.optimizers["honest_prover"].param_groups[0]["lr"],
                                "sneaky_prover" : self.schedulers["sneaky_prover"].get_last_lr()[0] if self.schedulers["sneaky_prover"] else self.optimizers["sneaky_prover"].param_groups[0]["lr"],
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
                        logger.info(f"[Step {self.global_step}] Optimizer step completed. Starting weight synchronization...")
                        sync_start_time = time.time()

                    # --- Weight Synchronization ---
                    # Decide when to sync weights (e.g., every N steps)
                    if self.global_step % self.args.sync_steps == 0:
                        self._sync_weights_to_vllm()

                     # --- FINAL BARRIER FOR THE SYNC STEP ---
                    # Ensure all processes complete logging, eval, checkpointing, and syncing
                    # before ANY process starts the next iteration.
                    logger.info(f"[Process {self.accelerator.process_index}] Step {self.global_step}: Reached end of sync step logic. Waiting at final barrier...")
                    self.accelerator.wait_for_everyone()
                    logger.info(f"[Process {self.accelerator.process_index}] Step {self.global_step}: Passed final barrier.")

                    if self.accelerator.is_main_process:
                        sync_end_time = time.time()
                        sync_duration = sync_end_time - sync_start_time
                        logger.info(f"[Step {self.global_step}] Weight synchronization finished. Duration: {sync_duration:.2f} seconds.")
                        # Log this duration to wandb as well
                        if self.wandb_run:
                            self.wandb_run.log({"train/sync_duration": sync_duration}, step=self.global_step)

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
        logger.info(f"[Process {self.accelerator.process_index}] Entering _training_step")
        # Plan:
        # 0. Prepare the inputs via vLLM + scoring + advantages for both provers [honest_prover, sneaky_prover]
        logger.info(f"[Process {self.accelerator.process_index}] Calling _prepare_inputs")
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

        logger.info(f"[Process {self.accelerator.process_index}] Calling compute_loss for honest_prover")
        # --- Model A ---
        # Forward/Backward for Model A's own loss
        loss_a = self.compute_loss(
            self.models["honest_prover"], inputs["honest_prover"], "honest_prover"
        )
        logger.info(f"[Process {self.accelerator.process_index}] Completed compute_loss for honest_prover")
        # --- Model B ---
        # Forward/Backward for Model B's own loss
        loss_b = self.compute_loss(
            self.models["sneaky_prover"], inputs["sneaky_prover"], "sneaky_prover"
        )

        logger.info(f"[Process {self.accelerator.process_index}] Completed compute_loss for sneaky_prover")
        logger.info(f"[Process {self.accelerator.process_index}] Returning losses")
        logger.info(f"[Process {self.accelerator.process_index}] Losses: {loss_a}, {loss_b}")

        # --- Return losses ---
        return {"loss_a": loss_a, "loss_b": loss_b}

    def _log_metrics(self, step_data: dict[str, Any]) -> None:
        """Gathers and logs metrics. Takes step_data from _training_step() -- losses, lrs, grad_norms, etc.-- and self._metrics -- lists of scalars collected during a step/eval. """
        eval_mode = "train" # NOTE: This function is only called during training
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

        model_metrics = self._metrics[eval_mode]

        for metric_name, values in model_metrics.items():
            if values: # Only log non-empty metrics
                metrics_to_log[f"{eval_mode}/{metric_name}"] = torch.tensor(values).mean().item()

        # --- Log metrics ---
        self.accelerator.log(metrics_to_log, step=global_step)
        # --- Log metrics to wandb ---
        logger.info(f"Logging metrics to wandb for step {global_step}")
        logger.info(f"Metrics to log: {metrics_to_log}")
        if self.wandb_run:
            self.wandb_run.log(metrics_to_log, step=global_step)

        logger.info(f"Step {global_step}: {metrics_to_log}")


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

    # def _move_model_to_vllm(self, model_key: str):
    #     """Synchronizes weights from the training model to the corresponding vLLM server."""
    #     # logger.info(f"[Process {self.accelerator.process_index}] Starting _move_model_to_vllm for {model_key}...")
    #     # if not self.accelerator.is_main_process:
    #     #     return  # Only main process interacts with vLLM client

    #     # logger.info(f"Synchronizing weights for {model_key} to its vLLM server...")
    #     # model_to_sync = self.models[model_key]  # The prepared training model
    #     # vllm_client = self.vllm_clients[model_key]

    #     # if vllm_client is None:
    #     #     logger.warning(
    #     #         f"vLLM client for {model_key} not initialized. Skipping weight sync."
    #     #     )
    #     #     raise ValueError(
    #     #         f"vLLM client for {model_key} not initialized. Skipping weight sync."
    #     #     )

    #     # # Use the provided logic, adapting slightly
    #     # # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
    #     # deepspeed_plugin = self.accelerators[model_key].state.deepspeed_plugin
    #     # zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

    #     # gather_if_zero3 = (
    #     #     deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
    #     # )

    #     # # Unwrap the model if necessary (needed for named_parameters)
    #     # # Note: accelerator.unwrap_model might be needed depending on DeepSpeed/FSDP wrapping
    #     # unwrapped_model = self.accelerators[model_key].unwrap_model(model_to_sync)

    #     # logger.info(f"Handling standard model weight sync for {model_key}...")
    #     # # For non-PEFT models, gather and update each parameter individually if ZeRO-3
    #     # for name, param in unwrapped_model.named_parameters():
    #     #     with gather_if_zero3(
    #     #         [param], modifier_rank=0 if zero_stage_3 else None
    #     #     ):  # Pass modifier_rank=0 for DS3
    #     #         # Ensure we are on the main process *after* gathering if needed
    #     #         if self.accelerator.is_main_process:
    #     #             logger.debug(f"[Weight Sync {model_key}] Updating param: {name}")
    #     #             vllm_client.update_named_param(name, param.data)  # Use param.data

    #     # # Reset cache on main process
    #     # if self.accelerator.is_main_process:
    #     #     logger.info(f"Resetting vLLM prefix cache for {model_key}.")
    #     #     vllm_client.reset_prefix_cache()

    #     # logger.info(f"Weight synchronization for {model_key} complete.")
    #     # self.accelerator.wait_for_everyone()  # Ensure sync before proceeding

    #     model = self.models[model_key]
    #     vllm_client = self.vllm_clients[model_key] # Get the correct client
    #     accelerator = self.accelerators[model_key] # Get the correct accelerator
    #     # accelerator = self.accelerator

    #     if vllm_client is None:
    #         logger.warning(f"No vLLM client configured for {model_key}, skipping weight sync.")
    #         return

    #     # Unwrap the model if necessary (e.g., if wrapped by Accelerate/DeepSpeed)
    #     unwrapped_model = accelerator.unwrap_model(model)

    #     # Determine if using DeepSpeed ZeRO Stage 3
    #     deepspeed_plugin = accelerator.state.deepspeed_plugin
    #     zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
    #     # Use GatheredParameters context only if ZeRO-3 is active
    #     gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext


    #     logger.info(f"[Process {accelerator.process_index} / {model_key}] Non-PEFT model detected.")
    #     named_params = list(unwrapped_model.named_parameters())

    #     for i, (name, param) in enumerate(tqdm(named_params, desc=f"Syncing {model_key}", disable=not accelerator.is_main_process)):
    #         # logger.debug(f"[Sync {model_key}] Gathering param {i+1}/{num_params}: {name}")
    #         with gather_if_zero3([param]): # Pass modifier_rank=0 if needed? Check Accelerate/DS docs
    #             # logger.debug(f"[Sync {model_key}] Param {name} gathered (took {param_gather_end - param_gather_start:.2f}s).")
    #             # Ensure we are on the main process *after* gathering if needed
    #             if accelerator.is_main_process:
    #                 # logger.debug(f"[Sync {model_key}] Updating param: {name}")
    #                 vllm_client.update_named_param(name, param.data)
    #             # Parameter is automatically re-partitioned if needed when exiting context

    #     # Reset cache on main process only
    #     if accelerator.is_main_process:
    #         logger.info(f"[Process {accelerator.process_index} / {model_key}] Resetting vLLM prefix cache...")
    #         reset_start = time.time()
    #         vllm_client.reset_prefix_cache()
    #         reset_end = time.time()
    #         logger.info(f"[Process {accelerator.process_index} / {model_key}] vLLM prefix cache reset (took {reset_end - reset_start:.2f}s).")

    #     logger.info(f"[Process {accelerator.process_index}] Finished _move_model_to_vllm for {model_key}.")

    def _move_model_to_vllm(self, model_key: str):
        accelerator = self.accelerators[model_key]
        model = self.models[model_key]
        vllm_client = self.vllm_clients[model_key]
        can_sync_to_client = accelerator.is_main_process and vllm_client is not None
        logger.info(f"[Process {accelerator.process_index}] Starting weight sync logic for {model_key}...")
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        unwrapped_model = accelerator.unwrap_model(model)
        named_params = list(unwrapped_model.named_parameters())
        num_params = len(named_params)
        logger.info(f"[Process {accelerator.process_index} / {model_key}] Starting parameter sync loop ({num_params} params)...")

        param_iterator = named_params
        if accelerator.is_main_process:
             param_iterator = tqdm(named_params, desc=f"Syncing {model_key}", leave=False, disable=False)

        for name, param in param_iterator:
            if not param.requires_grad: continue
            try:
                # Collective operation happens here
                with gather_if_zero3([param], modifier_rank=0 if zero_stage_3 else None):
                    if can_sync_to_client:
                        try:
                            vllm_client.update_named_param(name, param.data)
                        except Exception as e:
                            logger.error(f"Failed to update param {name} for {model_key} via vLLM: {e}", exc_info=True)
                            break
            except Exception as e:
                 logger.error(f"Error during GatheredParameters for {name} in {model_key}: {e}", exc_info=True)
                 break

        # --- Barrier AFTER loop ---
        # Ensures all processes finish the loop before proceeding to cache reset
        logger.info(f"[Process {accelerator.process_index} / {model_key}] Finished parameter loop. Waiting at barrier...")
        accelerator.wait_for_everyone() # Use the specific accelerator
        logger.info(f"[Process {accelerator.process_index} / {model_key}] Passed barrier after parameter loop.")

        # --- Reset Cache (Main Process Only) ---
        if can_sync_to_client:
            logger.info(f"[Process {accelerator.process_index} / {model_key}] Resetting vLLM prefix cache...")
            # ... (try-except block for reset) ...

        # --- Final Barrier for this function ---
        logger.info(f"[Process {accelerator.process_index}] Finished _move_model_to_vllm for {model_key}. Waiting at final barrier...")
        accelerator.wait_for_everyone() # Use the specific accelerator again
        logger.info(f"[Process {accelerator.process_index}] Exiting _move_model_to_vllm for {model_key}.")

    # def _sync_weights_to_vllm(self):
    #     """Helper method to trigger weight sync for both models."""
    #     # Sync honest_prover
    #     logger.info(f"[Process {self.accelerator.process_index}] ====> Preparing to call _move_model_to_vllm for honest_prover")
    #     self._move_model_to_vllm("honest_prover")
    #     logger.info(f"[Process {self.accelerator.process_index}] ====> Finished call to _move_model_to_vllm for honest_prover")

    #     logger.info(f"[Process {self.accelerator.process_index}] ====> Waiting for all processes after honest_prover sync...")
    #     # self.accelerators["honest_prover"].wait_for_everyone() # Use the main accelerator instance here
    #     logger.info(f"[Process {self.accelerator.process_index}] ====> Wait finished.")

    #     # Sync sneaky_prover
    #     logger.info(f"[Process {self.accelerator.process_index}] ====> Preparing to call _move_model_to_vllm for sneaky_prover")
    #     self._move_model_to_vllm("sneaky_prover")
    #     logger.info(f"[Process {self.accelerator.process_index}] ====> Finished call to _move_model_to_vllm for sneaky_prover")
    #     # self.accelerators["sneaky_prover"].wait_for_everyone() # Use the main accelerator instance here
    #     logger.info(f"[Process {self.accelerator.process_index}] ====> Wait finished.")
    #     # Verifier is frozen and not updated in this step, so we are good like that

    def _sync_weights_to_vllm(self):
        """Helper method to trigger weight sync for both models. Called by all processes."""
        logger.info(f"[Process {self.accelerator.process_index}] ===> Entering _sync_weights_to_vllm")

        # --- Sync honest_prover ---
        logger.info(f"[Process {self.accelerator.process_index}] ===> Selecting DS plugin 'honest_prover'...")
        self.accelerators["honest_prover"].state.select_deepspeed_plugin("honest_prover")
        logger.info(f"[Process {self.accelerator.process_index}] ===> Calling _move_model_to_vllm for honest_prover")
        self._move_model_to_vllm("honest_prover")
        logger.info(f"[Process {self.accelerator.process_index}] ===> Finished _move_model_to_vllm for honest_prover")

        # *** CRUCIAL GLOBAL BARRIER ***
        logger.info(f"[Process {self.accelerator.process_index}] ===> Global barrier before sneaky_prover sync...")
        self.accelerator.wait_for_everyone() # Synchronize everyone using the primary accelerator
        logger.info(f"[Process {self.accelerator.process_index}] ===> Passed global barrier.")

        # --- Sync sneaky_prover ---
        logger.info(f"[Process {self.accelerator.process_index}] ===> Selecting DS plugin 'sneaky_prover'...")
        self.accelerators["sneaky_prover"].state.select_deepspeed_plugin("sneaky_prover")
        logger.info(f"[Process {self.accelerator.process_index}] ===> Calling _move_model_to_vllm for sneaky_prover")
        self._move_model_to_vllm("sneaky_prover")
        logger.info(f"[Process {self.accelerator.process_index}] ===> Finished _move_model_to_vllm for sneaky_prover")

        # --- Final Global Barrier (Optional but safe) ---
        logger.info(f"[Process {self.accelerator.process_index}] ===> Global barrier after all syncs...")
        self.accelerator.wait_for_everyone() # Synchronize everyone
        logger.info(f"[Process {self.accelerator.process_index}] ===> Exiting _sync_weights_to_vllm")

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
        self, batch: dict[str, torch.Tensor | Any],
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

        # TODO: assert claim made above?
        # In principle we should just slide a num_generations window over all_honest_prompts and check if the window (with stride = num_generations) contains identical prompts. If not, I might be missing something.
        # TODO: implement this

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
        }

        logger.info(f"Generating completions for honest prover with args: {honest_gen_args}")
        # logger.info(f"All honest prompts: {all_honest_prompts}")


        completion_ids_a, completion_texts_a, _ = self._generate_via_vllm_and_broadcast(
            client_key="honest_prover",
            all_prompts_gathered=all_honest_prompts,
            generation_args=honest_gen_args,
            n_generations=self.args.num_generations,
            logprobs_count=0,
            raw_prompts_len_local=len(raw_prompts),  # Pass local raw prompt length
        )

        # logger.info(f"Completion texts: {completion_texts_a}")

        # Map prompt to completion text
        prompt_to_completion_text = {prompt: completion_text for prompt, completion_text in zip(all_honest_prompts, completion_texts_a)}
        # logger.info(f"Prompt to completion text: {prompt_to_completion_text}")

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
        }

        logger.info(f"Generating completions for sneaky prover with args: {sneaky_gen_args}")
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
        }
        verifier_logprobs_request_count = (
            15  # Number of logprobs needed for reward extraction? Adjust if needed.
        )
        logger.info(f"Generating completions for verifier with args: {verifier_gen_args}")
        logger.info("NOTE: Verifier gets **FULL** list of ids, texts, and logprobs. This is different from the other models.")

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

        # # --- Reward Processing and Advantage Calculation ---
        # rewards_all = None
        # all_completion_texts_v_gathered = None
        # if self.accelerator.is_main_process:
        #     logger.info(f"[Process {self.accelerator.process_index}] Main process entering reward extraction phase.")
        #     # Need to reconstruct the *full* list of completion texts on main process to extract rewards globally
        #     logger.info(f"[Process {self.accelerator.process_index}] Gathering verifier completion texts...")
        #     gather_start_time = time.time()
        #     # Need to reconstruct the *full* list of completion texts on main process to extract rewards globally
        #     all_completion_texts_v = gather_object(
        #         completion_texts_v
        #     )  # Gather local texts back
        #     rewards_all = [
        #         self.extract_verifier_reward(text) for text in all_completion_texts_v
        #     ]
        # else:
        #     rewards_all = [None] * (
        #         len(raw_prompts) * 2 * self.accelerator.num_processes
        #     )  # TODO: Check if multiplying by the number of processes is correct. I have a feeling it's not.

        # --- Reward Processing and Advantage Calculation ---
        rewards_all = None # Initialize
        all_completion_texts_v_gathered = None # Initialize for logging
        if self.accelerator.is_main_process:
            logger.info(f"[Process {self.accelerator.process_index}] Main process entering reward extraction phase.")
            # Need to reconstruct the *full* list of completion texts on main process to extract rewards globally
            logger.info(f"[Process {self.accelerator.process_index}] Gathering verifier completion texts...")
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
            logger.info(f"[Process {self.accelerator.process_index}] Length of completion_texts_v before reward extraction: {len(completion_texts_v)}")
            # --- End Add Logging ---

            logger.info(f"[Process {self.accelerator.process_index}] Starting reward extraction loop...")
            rewards_all = [
                self.extract_verifier_reward(text) for text in completion_texts_v
            ]
            # --- Add Logging ---
            logger.info(f"[Process {self.accelerator.process_index}] Length of rewards_all after reward extraction: {len(rewards_all)}")
            # --- End Add Logging ---

        else:
            # Placeholder on non-main processes before broadcast
            expected_len = len(raw_prompts) * 2 * self.accelerator.num_processes
            logger.info(f"[Process {self.accelerator.process_index}] Non-main process creating placeholder for {expected_len} rewards, waiting for broadcast.")
            rewards_all = [None] * expected_len

        # Log before broadcast on ALL processes
        logger.info(f"[Process {self.accelerator.process_index}] Preparing to broadcast rewards (object type: {type(rewards_all)}, length: {len(rewards_all) if isinstance(rewards_all, list) else 'N/A'}).")

        rewards_all = broadcast_object_list(
            rewards_all, from_process=0
        )  # Broadcast the globally extracted rewards

        # Now `rewards_all` contains the rewards corresponding to `all_verifier_prompts` on every process.

        assert len(rewards_all) == len(all_verifier_prompts), \
            f"[Process {self.accelerator.process_index}] Mismatch after broadcast: " \
            f"len(rewards_all)={len(rewards_all)} vs len(all_verifier_prompts)={len(all_verifier_prompts)}"
        logger.info(f"[Process {self.accelerator.process_index}] Assertion passed: Length of rewards_all ({len(rewards_all)}) matches all_verifier_prompts.")



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

        logger.info("Completed advantage calculation. Slicing advantages and rewards...")
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

        # # Calculate local slice indices for *verifier* rewards
        # num_local_samples_verifier = (
        #     len(raw_prompts) * 2
        # )  # Verifier handles both prover outputs locally
        # start_index_verifier = (
        #     self.accelerator.process_index * num_local_samples_verifier
        # )
        # end_index_verifier = start_index_verifier + num_local_samples_verifier
        # local_slice_verifier = slice(start_index_verifier, end_index_verifier)

        # # Slice the *global* verifier rewards to get the local part for logging
        # local_rewards_v = rewards_all_tensor[local_slice_verifier]

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

        logger.info("Completed log probability calculation. Loading completions into container...")
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

        # Log Verifier Metrics (Rewards only, no advantages)
        # We need a completion mask just for length calculation
        # verifier_completion_mask = (
        #     container.container["verifier"]["completion_ids"]
        #     != self.tokenizer.pad_token_id
        # ) # Bool - fine as is
        # self._store_generation_metrics(
        #     model_key="verifier",
        #     eval_mode="eval", # Verifier is frozen during provers training
        #     completion_mask=verifier_completion_mask,  # Use derived mask
        #     is_eos=container.container["verifier"]["is_eos"],
        #     rewards=None,  # Verifier doesn't have rewards
        #     advantages=None,  # Verifier doesn't have advantages
        # )

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
        self, model, input_ids, attention_mask, logits_to_keep=None
    ):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
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

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            model, input_ids, attention_mask, logits_to_keep
        )
        unwrapped_model = self.accelerator.unwrap_model(model)

        # --- DEBUG PRINTS ---
        if model_key == "honest_prover":
            print(f"[DEBUG {model_key} Rank {self.accelerators[model_key].process_index}] Shapes before liger_grpo_loss:")
            print(f"  - last_hidden_state: {last_hidden_state.shape}")
            print(f"  - lm_head.weight: {unwrapped_model.lm_head.weight.shape}")
            print(f"  - completion_ids: {completion_ids.shape}")
            print(f"  - completion_mask: {completion_mask.shape}")
            print(f"  - advantages: {inputs['advantages'].shape}")
            if inputs["ref_per_token_logps"] is not None:
                 print(f"  - ref_per_token_logps: {inputs['ref_per_token_logps'].shape}")
            else:
                 print("  - ref_per_token_logps: None")
            if inputs["old_per_token_logps"] is not None:
                print(f"  - old_per_token_logps: {inputs['old_per_token_logps'].shape}")
            else:
                 print("  - old_per_token_logps: None")
        # --- END DEBUG PRINTS ---

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
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
                self.accelerators[model_key].gather_for_metrics(mean_kl).nanmean().item()
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
        log_filepath = os.path.join(self.llm_interaction_log_dir, log_filename)

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

        self._metrics[eval_mode][model_key]["completions/mean_terminated_length"].append(
            term_completion_mask_sum.float().mean().item()
        )
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
                logger.info(f"[Process {self.accelerator.process_index} / {client_key}] Raw client output length: completion_ids_nested={len(completion_ids_nested)}, logprobs_nested={len(logprobs_nested)}")
                # --- End Add Logging ---
            else:
                completion_ids_nested = client.generate(**generate_kwargs)
                 # --- Add Logging ---
                logger.info(f"[Process {self.accelerator.process_index} / {client_key}] Raw client output length: completion_ids_nested={len(completion_ids_nested)}")
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
            logger.info(f"[Process {self.accelerator.process_index} / {client_key}] Length after batch_decode: completion_texts_all={len(completion_texts_all)}")
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
        logger.info(f"[AdvCalc shapes] mean_grouped_rewards: {mean_grouped_rewards.shape}")
        logger.info(f"[AdvCalc shapes] std_grouped_rewards: {std_grouped_rewards.shape}")
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
```

```data/dataset.py
# dataset.py

import logging
import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# --- Dataset Class ---
class AppsDataset(Dataset):
    """
    Dataset for efficient handling of APPS with tokenization optimization.
    Loads data, tokenizes only the question column without padding, and handles caching.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        split: str = "train",  # Specify split during initialization
        num_samples: int | None = None,  # Use None for all samples
        max_length: int | None = None,
        tokenize_column: str = "question",  # Column to tokenize
        keep_columns: list[str] = [
            "question",
            "solutions",
            "input_output",
        ],  # Columns to keep
        cache_dir: str | None = None,
        preprocessing_num_workers: int | None = None,
        min_length: int | None = None,
        truncation_strategy: str = "longest_first",
    ) -> None:
        """
        Initialize the AppsDataset.

        Args:
            dataset_name: Name of the dataset in HuggingFace hub (e.g., "codeparrot/apps").
            tokenizer: Tokenizer to use.
            split: Dataset split to load ('train', 'validation', 'test').
            num_samples: Number of samples to load (None for all).
            max_length: Maximum sequence length for truncation (None means no truncation during initial tokenization).
            tokenize_column: Name of the column to tokenize.
            keep_columns: List of columns to keep in the final dataset.
            cache_dir: Directory to cache tokenized datasets.
            preprocessing_num_workers: Number of workers for preprocessing.
            min_length: Minimum sequence length (filters shorter sequences after tokenization).
            truncation_strategy: Strategy for truncation if max_length is applied during tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenize_column = tokenize_column
        self.keep_columns = keep_columns
        self.min_length = min_length
        self.truncation_strategy = truncation_strategy
        self.split = split

        logging.info(f"Loading raw dataset '{dataset_name}' split '{split}'...")
        raw_dataset = load_dataset(
            dataset_name, split=split, cache_dir=cache_dir, trust_remote_code=True
        )

        # Select subset if num_samples is specified
        if (
            num_samples is not None
            and num_samples > 0
            and num_samples < len(raw_dataset)
        ):
            logging.info(f"Selecting {num_samples} samples from the dataset.")
            self.raw_dataset = raw_dataset.select(range(num_samples))
        else:
            logging.info(f"Using all {len(raw_dataset)} samples from the dataset.")
            self.raw_dataset = raw_dataset

        # Create tokenizer-specific cache path
        tokenizer_name = tokenizer.name_or_path.replace("/", "_")
        cache_file_name = f"{dataset_name.replace('/', '_')}_{split}_{tokenizer_name}_tokenized.hf"  # Use HF dataset cache format
        cache_file_path = None
        if cache_dir:
            cache_file_path = os.path.join(cache_dir, cache_file_name)
            os.makedirs(cache_dir, exist_ok=True)

        # Check if valid cached dataset exists (using datasets library's caching)
        try:
            # Use map's caching mechanism - it's more robust
            logging.info(
                "Attempting to load tokenized dataset from cache (if available)..."
            )
            self.tokenized_dataset = self.raw_dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                # Don't remove the keep_columns as we need them in the final dataset
                load_from_cache_file=True,  # Enable caching
                cache_file_name=cache_file_path,  # Specify cache file hint
                desc=f"Tokenizing {split} dataset",
            )
            logging.info(
                "Tokenized dataset loaded successfully (from cache or newly processed)."
            )

        except Exception as e:
            logging.error(
                f"Error during tokenization or cache loading: {e}", exc_info=True
            )
            logging.warning(
                "Proceeding without caching or retrying tokenization without explicit cache file path."
            )
            # Fallback: Tokenize without explicit cache file path if loading failed
            self.tokenized_dataset = self.raw_dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                # Don't remove the keep_columns
                load_from_cache_file=True,  # Still try to use implicit caching
                desc=f"Tokenizing {split} dataset (fallback)",
            )

        # Filter by length if min_length is specified
        if self.min_length is not None and self.min_length > 0:
            original_size = len(self.tokenized_dataset)
            logging.info(
                f"Filtering sequences shorter than {self.min_length} tokens..."
            )
            self.tokenized_dataset = self.tokenized_dataset.filter(
                lambda example: len(example["input_ids"]) >= self.min_length,
                num_proc=preprocessing_num_workers,
                desc="Filtering short sequences",
            )
            logging.info(
                f"Filtered dataset from {original_size} to {len(self.tokenized_dataset)} samples."
            )

        logging.info(
            f"Dataset initialization complete for split '{split}'. Size: {len(self.tokenized_dataset)}"
        )

    def _tokenize_function(self, examples):
        """Tokenization logic applied only to the question column."""
        # Tokenize without padding (padding done dynamically in collator)
        # Only truncate if max_length is specified during dataset init
        truncation = bool(self.max_length)

        # Only tokenize the specified column
        tokenized_output = self.tokenizer(
            examples[self.tokenize_column],
            truncation=truncation,
            max_length=self.max_length,
            # No padding here!
            return_attention_mask=True,  # Keep attention mask
            return_token_type_ids=False,  # Not needed for decoder-only models
        )

        # Copy the tokenization results to the output
        result = {}

        # Add tokenization outputs (input_ids, attention_mask)
        for key, value in tokenized_output.items():
            result[key] = value

        # Add all the columns we want to keep
        for column in self.keep_columns:
            if column in examples:
                result[column] = examples[column]

        return result

    def __len__(self) -> int:
        """Return the number of examples in the tokenized dataset."""
        return len(self.tokenized_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a tokenized example by index - without padding.
        Padding will be applied by the data collator at batch time.
        """
        item = self.tokenized_dataset[idx]

        # Convert token lists to tensors
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

        return {
            "question": item["question"],
            "solutions": item["solutions"],
            "input_output": item["input_output"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
```

```data/rep_sampler.py
# rep_sampler.py

import torch
from torch.utils.data import Sampler
from typing import Sized


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: int | None = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count
```

```inference/vllm_serve.py
import argparse
import logging
import os
from dataclasses import dataclass, field
from collections.abc import Sequence

import torch
import math
import json

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_available,
)


if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.worker.worker import Worker
else:
    Worker = object

logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorker(Worker):
    """
    A vLLM worker that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    def __init__(self, *args, **kwargs):
        if not is_vllm_available():
            raise ImportError(
                "vLLM is required to use the WeightSyncWorker. Please install it using `pip install vllm`."
            )

        super().__init__(*args, **kwargs)

        # The following attributes are initialized when `init_communicator` method is called.
        self.pynccl_comm = None  # Communicator for weight updates
        self.client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. Call close_communicator first."
            )

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(
        self, name: str, dtype: torch.dtype, shape: Sequence[int]
    ) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.pynccl_comm is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(
            weight, src=self.client_rank, stream=torch.cuda.current_stream()
        )
        self.pynccl_comm.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: str | None = field(
        default=None,
        metadata={
            "help": "Revision to use for the model. If not specified, the default branch will be used."
        },
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: int | None = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError(
            "vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`."
        )

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        dtype=script_args.dtype,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_cls=WeightSyncWorker,
    )

    app = FastAPI()

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():
        """
        Retrieves the tensor parallel size from the LLM engine.

        Returns:
            `dict`:
                A dictionary containing the tensor parallel size.

        Example response:
        ```json
        {"tensor_parallel_size": 8}
        ```
        """
        return {
            "tensor_parallel_size": llm.llm_engine.parallel_config.tensor_parallel_size
        }

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: str | None = None
        logprobs: int | None = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]
        logprobs: list[list[dict[int, float | None]]] | None = None

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=request.guided_decoding_regex
            )
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
            logprobs=request.logprobs,
        )
        all_outputs = llm.generate(request.prompts, sampling_params=sampling_params)
        # completion_ids = [
        #     list(output.token_ids)
        #     for outputs in all_outputs
        #     for output in outputs.outputs
        # ]
        # return {"completion_ids": completion_ids}
        # --- Extract completion_ids AND logprobs ---
        completion_ids = []
        logprobs_data = [] if request.logprobs is not None else None

        # --- Add Detailed Logging ---
        raw_logprobs_for_logging = []

        for i, request_output in enumerate(all_outputs):
            for j, output in enumerate(request_output.outputs):
                completion_ids.append(list(output.token_ids))
                if request.logprobs is not None and output.logprobs is not None:
                    token_logprobs_list = []
                    raw_step_logprobs_list = []  # For logging

                    for k, step_logprobs in enumerate(output.logprobs):
                        raw_step_data = {}  # For logging
                        if step_logprobs:
                            current_step_dict: dict[int, float | None] = {}
                            for token_id, logprob_obj in step_logprobs.items():
                                # --- Log the raw value ---
                                raw_val = logprob_obj.logprob
                                raw_step_data[int(token_id)] = (
                                    raw_val  # Store raw value for logging
                                )
                                # --------------------------

                                # Replace non-finite values with None for JSON compatibility
                                if not math.isfinite(raw_val):
                                    current_step_dict[int(token_id)] = None
                                else:
                                    # Check if it's exactly 0.0 - might indicate an issue or high prob
                                    if raw_val == 0.0:
                                        logger.debug(
                                            f"Logprob is exactly 0.0 for token {token_id} at step {k}, output {j}, request {i}"
                                        )
                                    current_step_dict[int(token_id)] = raw_val

                            token_logprobs_list.append(current_step_dict)
                            raw_step_logprobs_list.append(
                                raw_step_data
                            )  # Add raw data for this step
                        else:
                            token_logprobs_list.append({})
                            raw_step_logprobs_list.append({})  # Add empty raw data

                    logprobs_data.append(token_logprobs_list)
                    raw_logprobs_for_logging.append(
                        raw_step_logprobs_list
                    )  # Add raw data for this output

                elif request.logprobs is not None:
                    logprobs_data.append([])
                    raw_logprobs_for_logging.append([])  # Add empty raw data

        # --- Log the raw data before returning ---
        # Use json.dumps for potentially cleaner multi-line output if needed
        logger.info(
            f"Raw logprobs extracted (before None conversion): {json.dumps(raw_logprobs_for_logging, indent=2)}"
        )
        logger.info(
            f"Processed logprobs for response: {json.dumps(logprobs_data, indent=2)}"
        )

        # --- Return both completion_ids and logprobs ---
        return (
            {"completion_ids": completion_ids, "logprobs": logprobs_data}
            if request.logprobs is not None
            else {"completion_ids": completion_ids}
        )

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(
        request: InitCommunicatorRequest, background_tasks: BackgroundTasks
    ):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        background_tasks.add_task(
            llm.collective_rpc,
            "init_communicator",
            args=(request.host, request.port, script_args.tensor_parallel_size + 1),
        )
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(
        request: UpdateWeightsRequest, background_tasks: BackgroundTasks
    ):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function is called this way: update_named_param(name="name", dtype=torch.float32, shape=(10, 10))
        # So with collect_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        # And with background_tasks.add_task we need to call it this way:
        # background_tasks.add_task(llm.collective_rpc, "update_named_param", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(
            llm.collective_rpc,
            "update_named_param",
            args=(request.name, dtype, request.shape),
        )

        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        success = llm.llm_engine.reset_prefix_cache()
        return {
            "message": "Request received, resetting prefix cache status: "
            + str(success)
        }

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        llm.collective_rpc("close_communicator")
        return {"message": "Request received, closing communicator"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "vllm-serve",
            help="Run the vLLM serve script",
            dataclass_types=ScriptArguments,
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
```

```inference/vllmclient.py
import atexit
import logging
import time

import torch
from torch import nn

from trl.import_utils import is_requests_available, is_vllm_available


if is_requests_available():
    import requests
    from requests import ConnectionError


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup


logger = logging.getLogger(__name__)


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError(
                "requests is not installed. Please install it with `pip install requests`."
            )
        if not is_vllm_available():
            raise ImportError(
                "vLLM is not installed. Please install it with `pip install vllm`."
            )

        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout
        self.init_communicator()
        atexit.register(
            self.close_communicator
        )  # when the client object is deleted, close the weight update group

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(
                f"Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: str | None = None,
        logprobs: int | None = None,
    ) -> list[list[str]] | tuple[list[list[str]], list[list[dict[int, float]]]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        url = f"http://{self.host}:{self.server_port}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
                "logprobs": logprobs,
            },
        )
        if response.status_code == 200:
            return (
                response.json()["completion_ids"]
                if logprobs is None
                else (response.json()["completion_ids"], response.json()["logprobs"])
            )
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        # Get the tensor parallel size from the server
        url = f"http://{self.host}:{self.server_port}/get_tensor_parallel_size/"
        response = requests.get(url)
        if response.status_code == 200:
            tensor_parallel_size = response.json()["tensor_parallel_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = tensor_parallel_size + 1
        self.rank = tensor_parallel_size  # The client's rank is the last process

        # Initialize weight update group
        url = f"http://{self.host}:{self.server_port}/init_communicator/"
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(
            url,
            json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size},
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(
            host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
        )
        client_device = f"cuda:{torch.cuda.current_device()}"
        self.pynccl_comm = PyNcclCommunicator(pg, device=client_device)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        response = self.session.post(
            url, json={"name": name, "dtype": dtype, "shape": shape}
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(
            weights, src=self.rank, stream=torch.cuda.current_stream()
        )
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"http://{self.host}:{self.server_port}/close_communicator/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
```
</structure>
</codebase>

And the HTML processor is the following:

<processor>

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Experiment Inspector v3</title>
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Highlight.js for code syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- Optional: Load specific languages if needed, but python is usually included -->
    <!-- <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script> -->

    <style>
        /* --- Keep previous styles from v2 --- */
        :root {
            --bg-color: #f8f9fa;
            --container-bg: #ffffff;
            --text-color: #212529;
            --heading-color: #343a40;
            --border-color: #dee2e6;
            --border-light: #e9ecef;
            --primary-color: #007bff;
            --primary-light: #cce5ff;
            --honest-bg: #e6f7ff; /* Light blue */
            --sneaky-bg: #fff0f0; /* Light red */
            --verifier-bg: #f0fff0; /* Light green */
            --prompt-bg: #f1f3f5; /* Light grey for prompts */
            --code-bg: #282c34; /* hljs github-dark background */
            --code-text: #abb2bf; /* hljs github-dark text */
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .container {
            max-width: 1400px; /* Wider for comparison */
            margin: auto;
            background: var(--container-bg);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        h1, h2 {
            color: var(--heading-color);
            border-bottom: 1px solid var(--border-light);
            padding-bottom: 10px;
        }
        h2 {
            margin-top: 40px;
        }
        .file-input-area {
            margin-bottom: 25px;
            padding: 20px;
            border: 2px dashed var(--border-color);
            border-radius: 5px;
            background-color: #fdfdfd;
            text-align: center;
            transition: border-color 0.3s ease, background-color 0.3s ease;
        }
        .file-input-area.dragover {
             border-color: var(--primary-color);
             background-color: var(--primary-light);
        }
        .file-input-area label {
            cursor: pointer;
            color: var(--primary-color);
            font-weight: bold;
        }
        .file-input-area input[type="file"] {
            display: none; /* Hide default input */
        }
        #fileList {
            margin-top: 10px;
            font-size: 0.9em;
            color: #6c757d;
        }
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 25px;
        }
        .tab-button {
            padding: 12px 18px;
            cursor: pointer;
            border: none;
            background: none;
            border-bottom: 3px solid transparent;
            font-size: 1.05em;
            margin-right: 8px;
            color: #495057;
            transition: border-color 0.3s ease, color 0.3s ease;
        }
        .tab-button:hover {
            color: var(--primary-color);
        }
        .tab-button.active {
            border-bottom-color: var(--primary-color);
            font-weight: 600;
            color: var(--primary-color);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .file-section {
            border: 1px solid var(--border-color);
            margin-bottom: 25px;
            border-radius: 6px;
            background-color: var(--container-bg);
            box-shadow: 0 2px 4px rgba(0,0,0,0.03);
            overflow: hidden; /* Contain header background */
        }
        .file-header {
            background-color: var(--border-light);
            padding: 12px 18px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.95em;
            color: var(--heading-color);
            word-wrap: break-word;
        }
        .interaction {
            border: 1px solid var(--border-light);
            border-radius: 5px;
            margin: 15px;
            padding: 15px;
            background-color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .interaction:last-child {
             margin-bottom: 5px;
        }

        .prompt-section, .output-section, .ids-section {
            margin-bottom: 15px;
        }
         .prompt-section h4, .output-section h4, .ids-section h4 {
            margin-top: 0;
            margin-bottom: 8px;
            color: #495057;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }

        .prompt-content {
            background-color: var(--prompt-bg);
            padding: 12px;
            border-radius: 4px;
            border: 1px solid var(--border-light);
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.95em;
        }
        .output-content {
            padding: 12px;
            border-radius: 4px;
            border: 1px solid var(--border-light);
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.95em;
        }
        #content-honest .output-content { background-color: var(--honest-bg); border-color: #b8e0f8;}
        #content-sneaky .output-content { background-color: var(--sneaky-bg); border-color: #f8d7da;}
        #content-verifier .output-content { background-color: var(--verifier-bg); border-color: #c3e6cb;}

        .output-ids {
            font-family: monospace;
            font-size: 0.8em;
            color: #6c757d;
            background-color: #e9ecef;
            padding: 5px 10px;
            border-radius: 3px;
            word-break: break-all;
            display: none; /* Hidden by default */
            margin-top: 5px;
        }
        .show-ids .ids-section {
             display: block !important;
        }
        .ids-section {
            display: none;
        }

        .toggle-area {
            margin-bottom: 20px;
            text-align: right;
        }
        .toggle-area label {
            margin-right: 8px;
            font-size: 0.9em;
            color: #495057;
            cursor: pointer;
        }
        .toggle-area input[type="checkbox"] {
            vertical-align: middle;
            cursor: pointer;
        }

        /* --- Comparison View Styles --- */
        .comparison-block { /* Renamed from comparison-grid */
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 30px;
            background-color: #fff;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .comparison-verifier-prompt { /* Section for the verifier prompt */
             margin-bottom: 20px;
             padding-bottom: 15px;
             border-bottom: 1px dashed var(--border-light);
        }
        .comparison-verifier-prompt h4 { /* Style prompt header */
             margin-top: 0;
             margin-bottom: 8px;
             color: #495057;
             font-size: 0.9em;
             text-transform: uppercase;
             letter-spacing: 0.8px;
             font-weight: 600;
        }
        .comparison-outputs-grid { /* Grid for the output tiles */
            display: grid;
            /* Adjust columns: maybe 2x2 or flexible */
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }
        .comparison-output-tile { /* Individual tile for each output */
            padding: 15px;
            border-radius: 4px;
            border: 1px solid var(--border-light);
        }
        .comparison-output-tile h5 { /* Header within the tile */
             margin-top: 0;
             margin-bottom: 10px;
             font-size: 0.9em;
             color: var(--heading-color);
             text-transform: uppercase;
             letter-spacing: 0.5px;
             border-bottom: 1px solid var(--border-light);
             padding-bottom: 5px;
             font-weight: 600;
        }
        /* Apply specific backgrounds to output tiles */
        .comparison-output-tile.honest { background-color: var(--honest-bg); border-color: #b8e0f8; }
        .comparison-output-tile.sneaky { background-color: var(--sneaky-bg); border-color: #f8d7da; }
        .comparison-output-tile.verifier { background-color: var(--verifier-bg); border-color: #c3e6cb; }

        .comparison-output-tile .output-content { /* Reuse output content style */
            padding: 0;
            border: none;
            background: none;
            font-size: 0.95em; /* Match other output content */
        }
         .comparison-output-tile .output-ids { margin-top: 10px; }
         .comparison-output-tile .ids-section h4 { /* Adjust ID header style in tiles */
             font-size: 0.8em;
             margin-bottom: 5px;
             border-bottom: none;
             padding-bottom: 0;
         }

        .comparison-placeholder {
            font-style: italic;
            color: #6c757d;
            font-size: 0.9em;
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }


        /* --- Markdown and Code Styling (Keep from v2) --- */
        .prompt-content p, .output-content p { margin-top: 0; margin-bottom: 0.75em; }
        .prompt-content ul, .output-content ul, .prompt-content ol, .output-content ol { margin-top: 0; margin-bottom: 0.75em; padding-left: 30px; }
        .prompt-content pre, .output-content pre { background-color: var(--code-bg); color: var(--code-text); padding: 1em; border-radius: 5px; overflow-x: auto; margin: 0.75em 0; }
        .prompt-content code:not(pre > code), .output-content code:not(pre > code) { background-color: #e9ecef; padding: 0.2em 0.4em; margin: 0 0.1em; border-radius: 3px; font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9em; color: #c7254e; }
        .prompt-content pre code.hljs, .output-content pre code.hljs { background: none; padding: 0; border-radius: 0; color: inherit; font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9em; } /* Ensure font-family */
        .prompt-content blockquote, .output-content blockquote { border-left: 4px solid var(--border-color); padding-left: 15px; margin-left: 0; color: #6c757d; font-style: italic; }
        .prompt-content h1, .output-content h1, .prompt-content h2, .output-content h2, .prompt-content h3, .output-content h3, .prompt-content h4, .output-content h4, .prompt-content h5, .output-content h5, .prompt-content h6, .output-content h6 { margin-top: 1.2em; margin-bottom: 0.6em; border-bottom: none; padding-bottom: 0; font-weight: 600; color: var(--heading-color); }
        .prompt-content hr, .output-content hr { border: 0; border-top: 1px solid var(--border-light); margin: 1.5em 0; }

    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Experiment Inspector v3</h1>

        <div class="file-input-area" id="dropArea">
            <label for="jsonFilesInput">Click here or drag & drop JSON files (Honest, Sneaky, Verifier)</label>
            <input type="file" id="jsonFilesInput" multiple accept=".json">
            <div id="fileList"></div>
        </div>

        <div class="toggle-area">
            <label for="showIdsToggle">Show Output IDs:</label>
            <input type="checkbox" id="showIdsToggle">
        </div>

        <div id="tabs" class="tabs"></div>
        <div id="content"></div>
    </div>

    <script>
        // Configure marked to use highlight.js
        marked.setOptions({
            highlight: function(code, lang) {
                // Log the language detected by marked
                console.log(`Highlighting code block. Detected lang: '${lang}'`);
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                if (language !== 'plaintext') {
                     console.log(`Using language: '${language}' for highlight.js`);
                } else if (lang) {
                     console.warn(`Language '${lang}' not recognized by highlight.js, falling back to plaintext.`);
                }
                try {
                    // Use ignoreIllegals: true for robustness
                    return hljs.highlight(code, { language, ignoreIllegals: true }).value;
                } catch (e) {
                    console.error("Highlight.js error:", e);
                    // Fallback to plaintext highlighting on error
                    return hljs.highlight(code, { language: 'plaintext', ignoreIllegals: true }).value;
                }
            },
            langPrefix: 'hljs language-', // crucial for CSS targeting
            pedantic: false,
            gfm: true, // Enable GitHub Flavored Markdown (needed for ```lang)
            breaks: true, // Render GFM line breaks
            sanitize: false, // We handle escaping manually
            smartLists: true,
            smartypants: false,
            xhtml: false
        });

        const jsonFilesInput = document.getElementById('jsonFilesInput');
        const fileListDisplay = document.getElementById('fileList');
        const tabsContainer = document.getElementById('tabs');
        const contentContainer = document.getElementById('content');
        const showIdsToggle = document.getElementById('showIdsToggle');
        const dropArea = document.getElementById('dropArea');

        let fileMap = { honest: [], sneaky: [], verifier: [] };

        // --- Event Listeners ---
        jsonFilesInput.addEventListener('change', handleFiles);
        showIdsToggle.addEventListener('change', toggleIdsVisibility);
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
        });
        dropArea.addEventListener('drop', handleDrop, false);

        // --- Functions ---
        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            jsonFilesInput.files = files;
            handleFiles({ target: { files: files } });
        }

        function escapeHtml(unsafe) {
            if (typeof unsafe !== 'string') return unsafe;
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        // --- File Handling (Keep corrected version from previous step) ---
        function handleFiles(event) {
            console.log("handleFiles triggered.");
            const files = event.target.files;
            if (!files || files.length === 0) {
                console.log("No files selected or event target has no files.");
                fileListDisplay.innerHTML = '<i>No files selected.</i>';
                return;
            }
            console.log(`Processing ${files.length} file(s).`);

            fileMap = { honest: [], sneaky: [], verifier: [] };
            tabsContainer.innerHTML = '';
            contentContainer.innerHTML = '<p><i>Loading files...</i></p>';
            fileListDisplay.innerHTML = '<strong>Processing Files:</strong><br>';

            let fileNames = [];
            const filePromises = [];

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                if (file) {
                    fileNames.push(escapeHtml(file.name));
                    filePromises.push(readFile(file));
                } else {
                    console.warn("Encountered an undefined file object at index:", i);
                }
            }

            fileListDisplay.innerHTML = `<strong>Selected Files:</strong><br>${fileNames.join('<br>')}`;

            Promise.all(filePromises)
                .then(() => {
                    console.log("Promise.all resolved. Final fileMap:", fileMap);
                    const hasData = Object.values(fileMap).some(arr => arr.length > 0);

                    if (hasData) {
                        console.log("Data found, building UI.");
                        buildUI();
                    } else {
                        console.log("No data loaded into fileMap.");
                        contentContainer.innerHTML = `<p style="color: orange;">Finished processing, but no valid data (Honest/Sneaky/Verifier) was found in the selected files. Please check file contents and naming conventions.</p>`;
                        tabsContainer.innerHTML = '';
                    }
                })
                .catch(error => {
                    console.error("Unexpected error during file processing:", error);
                    contentContainer.innerHTML = `<p style="color: red;">An unexpected error occurred during file processing. Check console for details. Message: ${escapeHtml(error.message || String(error))}</p>`;
                    tabsContainer.innerHTML = '';
                });
        }

        function readFile(file) {
            console.log(`readFile attempting for: ${file.name}`);
            return new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    console.log(`reader.onload triggered for: ${file.name}`);
                    try {
                        const json = JSON.parse(e.target.result);
                        const fileName = file.name;
                        console.log(`Successfully parsed JSON for: ${fileName}`);

                        let type = 'unknown';
                        const lowerFileName = fileName.toLowerCase();
                        if (lowerFileName.includes('honest')) type = 'honest';
                        else if (lowerFileName.includes('sneaky')) type = 'sneaky';
                        else if (lowerFileName.includes('verifier')) type = 'verifier';
                        else if (json.model_mode) {
                            const mode = json.model_mode.toLowerCase();
                            if (mode.includes('honest')) type = 'honest';
                            else if (mode.includes('sneaky')) type = 'sneaky';
                            else if (mode.includes('verifier')) type = 'verifier';
                        }
                        console.log(`Determined type '${type}' for file: ${fileName}`);

                        if (fileMap[type]) {
                            if (!json.prompts) json.prompts = [];
                            if (!json.output_texts) json.output_texts = [];
                            if (!json.output_ids) json.output_ids = [];
                            fileMap[type].push({ fileName: fileName, data: json });
                            console.log(`Added data from ${fileName} to fileMap.${type}`);
                        } else {
                            console.warn(`File ${fileName} has unknown or unhandled type '${type}'. Mode: ${json.model_mode}. Skipping.`);
                        }
                        resolve();
                    } catch (error) {
                        console.error(`Error parsing JSON from file ${file.name}:`, error);
                        fileListDisplay.innerHTML += `<br><span style="color:red;">Error parsing ${escapeHtml(file.name)}!</span>`;
                        resolve();
                    }
                };
                reader.onerror = (e) => {
                    console.error(`Error reading file ${file.name}:`, e);
                    fileListDisplay.innerHTML += `<br><span style="color:red;">Error reading ${escapeHtml(file.name)}!</span>`;
                    resolve();
                };
                if (file && typeof file === 'object') {
                   console.log(`Calling reader.readAsText for ${file.name}`);
                   reader.readAsText(file);
                } else {
                   console.error("Invalid file object passed to readFile:", file);
                   resolve();
                }
            });
        }
        // --- End File Handling ---


        // --- UI Building ---
        function buildUI() {
            tabsContainer.innerHTML = '';
            contentContainer.innerHTML = '';
            let firstTab = null;
            const types = ['honest', 'sneaky', 'verifier'];

            // Create Tabs for individual views
            types.forEach(type => {
                if (fileMap[type]?.length > 0) {
                    createTab(type, type.charAt(0).toUpperCase() + type.slice(1));
                    if (!firstTab) firstTab = type;
                    buildIndividualTabContent(type); // Build content immediately
                }
            });

            // Create Comparison Tab if verifier exists (and ideally others)
            if (fileMap.verifier?.length > 0) {
                 createTab('comparison', 'Comparison (Verifier Driven)');
                 buildComparisonTabContent(); // Build comparison content
                 if (!firstTab) firstTab = 'comparison';
            } else {
                 // Optionally add a disabled comparison tab or message
                 console.log("Verifier file missing, cannot build comparison view.");
            }


            // Activate the first available tab
            if (firstTab) {
                showTab(firstTab);
            } else {
                 contentContainer.innerHTML = '<p><i>No valid files loaded or data found. Please select JSON files matching the expected structure (honest/sneaky/verifier).</i></p>';
            }

            // Apply initial ID visibility
            toggleIdsVisibility();

            // Apply syntax highlighting AFTER all content is in the DOM
            console.log("Calling hljs.highlightAll()");
            try {
                 hljs.highlightAll();
            } catch (e) {
                console.error("Error during highlightAll:", e);
            }
        }

        function createTab(tabId, tabLabel) {
             const button = document.createElement('button');
             button.className = 'tab-button';
             button.textContent = tabLabel;
             button.dataset.tab = tabId;
             button.onclick = () => showTab(tabId);
             tabsContainer.appendChild(button);

             const contentDiv = document.createElement('div');
             contentDiv.id = `content-${tabId}`;
             contentDiv.className = 'tab-content';
             contentContainer.appendChild(contentDiv);
        }

        function buildIndividualTabContent(type) {
            // --- This function remains largely the same as in v2 ---
            // --- Ensure it uses escapeHtml() before marked.parse() ---
            const contentDiv = document.getElementById(`content-${type}`);
            if (!contentDiv) return;
            contentDiv.innerHTML = ''; // Clear previous content if rebuilding

            fileMap[type].forEach(fileInfo => {
                const fileSection = document.createElement('div');
                fileSection.className = 'file-section';

                const fileHeader = document.createElement('div');
                fileHeader.className = 'file-header';
                fileHeader.textContent = `File: ${escapeHtml(fileInfo.fileName)} (ID: ${escapeHtml(fileInfo.data.interaction_id || 'N/A')})`;
                fileSection.appendChild(fileHeader);

                const data = fileInfo.data;
                const maxInteractions = Math.max(data.prompts?.length || 0, data.output_texts?.length || 0);

                if (maxInteractions > 0) {
                    for (let i = 0; i < maxInteractions; i++) {
                        const interactionDiv = document.createElement('div');
                        interactionDiv.className = 'interaction';

                        // Prompt
                        if (data.prompts?.[i] !== undefined) {
                            interactionDiv.appendChild(createContentSection('Prompt', i + 1, data.prompts[i], 'prompt-content'));
                        } else {
                             interactionDiv.innerHTML += `<div class="prompt-section"><h4>Prompt ${i + 1}</h4><div class="prompt-content"><em>Prompt data missing</em></div></div>`;
                        }

                        // Output Text
                        if (data.output_texts?.[i] !== undefined) {
                             interactionDiv.appendChild(createContentSection('Output Text', i + 1, data.output_texts[i], 'output-content'));
                        } else {
                             interactionDiv.innerHTML += `<div class="output-section"><h4>Output Text ${i + 1}</h4><div class="output-content"><em>Output text missing</em></div></div>`;
                        }

                        // Output IDs (Optional)
                        if (data.output_ids?.[i] !== undefined) {
                            interactionDiv.appendChild(createIdsSection(i + 1, data.output_ids[i]));
                        }

                        fileSection.appendChild(interactionDiv);
                    }
                } else {
                    fileSection.innerHTML += '<p style="padding: 15px;"><em>No prompts or outputs found in this file.</em></p>';
                }
                contentDiv.appendChild(fileSection);
            });
        }

        // Helper for creating content sections (Prompt/Output)
        function createContentSection(titlePrefix, index, text, contentClass) {
            const section = document.createElement('div');
            section.className = `${contentClass.split('-')[0]}-section`; // e.g., prompt-section
            section.innerHTML = `<h4>${titlePrefix} ${index}</h4>`;
            const content = document.createElement('div');
            content.className = contentClass;
            // Escape HTML *before* Markdown parsing
            const escapedText = escapeHtml(text || '');
            // console.log(`Parsing for ${contentClass} ${index}:`, escapedText.substring(0, 100) + "..."); // Log snippet
            content.innerHTML = marked.parse(escapedText);
            section.appendChild(content);
            return section;
        }

        // Helper for creating IDs section
        function createIdsSection(index, idsData) {
            const idsSection = document.createElement('div');
            idsSection.className = 'ids-section'; // Initially hidden by CSS
            idsSection.innerHTML = `<h4>Output IDs ${index}</h4>`;
            const idsContent = document.createElement('div');
            idsContent.className = 'output-ids';
            idsContent.textContent = escapeHtml(Array.isArray(idsData) ? idsData.join(', ') : String(idsData));
            idsSection.appendChild(idsContent);
            return idsSection;
        }


        // --- NEW Comparison Logic ---
        function buildComparisonTabContent() {
            const contentDiv = document.getElementById('content-comparison');
            if (!contentDiv) return;
            contentDiv.innerHTML = ''; // Clear previous content

            // Use first file of each type for comparison (simplification)
            const verifierFile = fileMap.verifier?.[0];
            const honestFile = fileMap.honest?.[0];
            const sneakyFile = fileMap.sneaky?.[0];

            if (!verifierFile) {
                contentDiv.innerHTML = '<p><i>Verifier file is required for comparison view.</i></p>';
                return;
            }
            if (!honestFile && !sneakyFile) {
                 contentDiv.innerHTML = '<p><i>Verifier file loaded, but no Honest or Sneaky file found for comparison.</i></p>';
                 // Optionally still show verifier prompts/outputs? For now, require at least one other.
                 // return;
            }


            const verifierData = verifierFile.data;
            const honestData = honestFile?.data; // Use optional chaining
            const sneakyData = sneakyFile?.data; // Use optional chaining
            const numVerifierInteractions = verifierData.prompts?.length || 0;

            console.log(`Building comparison view with ${numVerifierInteractions} verifier interactions.`);

            if (numVerifierInteractions === 0) {
                 contentDiv.innerHTML = '<p><i>No interactions found in the verifier file to compare.</i></p>';
                 return;
             }

            // Iterate through verifier interactions, taking 2 at a time
            for (let i = 0; i < numVerifierInteractions; i += 2) {
                const comparisonBlock = document.createElement('div');
                comparisonBlock.className = 'comparison-block';

                const verifierPromptIndex = i;
                const honestSneakyIndex = i / 2; // Corresponding index in honest/sneaky

                // --- Verifier Prompt (Top Section) ---
                const promptSection = document.createElement('div');
                promptSection.className = 'comparison-verifier-prompt';
                const promptText = verifierData.prompts?.[verifierPromptIndex];
                promptSection.innerHTML = `<h4>Verifier Prompt ${verifierPromptIndex + 1} (Corresponds to H/S Index ${honestSneakyIndex + 1})</h4>`;
                if (promptText !== undefined) {
                    promptSection.appendChild(createContentSection('', '', promptText, 'prompt-content')); // Use helper, empty titles
                } else {
                    promptSection.innerHTML += '<div class="prompt-content"><em>Verifier prompt data missing for this index.</em></div>';
                }
                comparisonBlock.appendChild(promptSection);

                // --- Outputs Grid ---
                const outputsGrid = document.createElement('div');
                outputsGrid.className = 'comparison-outputs-grid';

                // --- Honest Output Tile ---
                outputsGrid.appendChild(
                    createComparisonOutputTile(
                        'Honest',
                        honestSneakyIndex,
                        honestData?.output_texts, // Pass array
                        honestData?.output_ids,   // Pass array
                        'honest'
                    )
                );

                // --- Sneaky Output Tile ---
                 outputsGrid.appendChild(
                    createComparisonOutputTile(
                        'Sneaky',
                        honestSneakyIndex,
                        sneakyData?.output_texts,
                        sneakyData?.output_ids,
                        'sneaky'
                    )
                );

                // --- Verifier Output Tile (First of Pair - relates to Honest) ---
                outputsGrid.appendChild(
                    createComparisonOutputTile(
                        `Verifier (Eval Honest ${honestSneakyIndex + 1})`,
                        verifierPromptIndex, // Use verifier index i
                        verifierData.output_texts,
                        verifierData.output_ids,
                        'verifier'
                    )
                );

                // --- Verifier Output Tile (Second of Pair - relates to Sneaky) ---
                // Check if the second verifier interaction (i+1) exists
                if (i + 1 < numVerifierInteractions) {
                     outputsGrid.appendChild(
                        createComparisonOutputTile(
                            `Verifier (Eval Sneaky ${honestSneakyIndex + 1})`,
                            verifierPromptIndex + 1, // Use verifier index i+1
                            verifierData.output_texts,
                            verifierData.output_ids,
                            'verifier'
                        )
                    );
                } else {
                    // Add placeholder if the second verifier interaction is missing
                     const placeholderTile = document.createElement('div');
                     placeholderTile.className = 'comparison-output-tile verifier'; // Style as verifier
                     placeholderTile.innerHTML = `<h5>Verifier (Eval Sneaky ${honestSneakyIndex + 1})</h5>`;
                     placeholderTile.appendChild(createPlaceholder('Second verifier output missing for this pair.'));
                     outputsGrid.appendChild(placeholderTile);
                }


                comparisonBlock.appendChild(outputsGrid);
                contentDiv.appendChild(comparisonBlock);
            }
        }

        // Helper to create a tile in the comparison grid
        function createComparisonOutputTile(titlePrefix, index, outputTexts, outputIds, typeClass) {
            const tile = document.createElement('div');
            tile.className = `comparison-output-tile ${typeClass}`;
            tile.innerHTML = `<h5>${titlePrefix} Output ${index + 1}</h5>`; // Use h5 for tile headers

            const outputText = outputTexts?.[index]; // Safely access text
            if (outputText !== undefined) {
                 tile.appendChild(createContentSection('', '', outputText, 'output-content')); // Use helper

                 // Add IDs if present
                 const idsData = outputIds?.[index];
                 if (idsData !== undefined) {
                     tile.appendChild(createIdsSection(index + 1, idsData));
                 }
            } else {
                 tile.appendChild(createPlaceholder(`No ${typeClass} output data for index ${index + 1}.`));
            }
            return tile;
        }

        // Helper for placeholder content
        function createPlaceholder(message) {
            const placeholder = document.createElement('div');
            placeholder.className = 'comparison-placeholder';
            placeholder.textContent = message;
            return placeholder;
        }
        // --- End UI Building ---


        // --- Tab/Visibility Logic (Keep from v2) ---
        function showTab(tabName) {
            document.querySelectorAll('.tab-button').forEach(button => button.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            const activeButton = document.querySelector(`.tab-button[data-tab="${tabName}"]`);
            const activeContent = document.getElementById(`content-${tabName}`);

            if (activeButton) activeButton.classList.add('active');
            if (activeContent) activeContent.classList.add('active');

            // It might be beneficial to re-run highlightAll *only* when switching to a tab
            // that hasn't been highlighted yet, but for simplicity, running it after buildUI is often sufficient.
            // If performance becomes an issue with many large files, optimize this.
            // console.log(`Switched to tab: ${tabName}. Re-highlighting (optional).`);
            // try { hljs.highlightAll(); } catch(e) { console.error("Highlight error on tab switch:", e); }
        }

        function toggleIdsVisibility() {
            const show = showIdsToggle.checked;
            const rootElement = document.getElementById('content');
            if (!rootElement) return; // Guard against calls before content exists
            if (show) {
                rootElement.classList.add('show-ids');
            } else {
                rootElement.classList.remove('show-ids');
            }
             // Log the state change
             console.log(`Toggled IDs visibility: ${show ? 'ON' : 'OFF'}`);
        }
        // --- End Tab/Visibility Logic ---

    </script>
</body>
</html>
```

</processor>

Plan moves, without implementing anything quite yet. I want to brainstorm initially.