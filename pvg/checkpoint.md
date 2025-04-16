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