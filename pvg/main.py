# adversarial_training_core/train.py

# The prover π ∗ used for sampling solutions for verifier training is a mixture of an initial base prover and previous round provers, each balanced to have equal number of correct and incorrect solutions. Each of the previous rounds has equal ratio in the mixture, which is tuned as a hyperparameter. In round 0, the solutions are from the base prover sampled via a few-shot prompt (App. [H)](#page-31-0). In later rounds of training, we replace part of the solutions with those from the new provers. Hence we always use the same number of solutions per problem. The amount of verifier optimization is constant over the rounds.

import logging
import os
import torch
from jsonargparse import auto_cli
from transformers import set_seed


# Configuration
from pvg.config.args import ExperimentArgs
from typing import Literal

# Core Components (Managers)
from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.data_manager import DataManager
from pvg.components.model_manager import ModelManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.components.state_tracker import StateTracker

# RL Components (Example - adjust imports as needed)
# from adversarial_training_core.rl.step_logic import RLStepLogic
# from adversarial_training_core.rl.loss import LossCalculator

# Top-Level Orchestrator
from pvg.orchestrator import TrainingPhaseOrchestrator

# Utilities
from pvg.utils.logger import setup_logger

# --- Minimal Initial Logging Setup ---
# Configure basic logging BEFORE parsing args or initializing accelerate
# This captures early messages. It will be reconfigured after accelerator init.
initial_logger = setup_logger(name="pvg", level=logging.INFO, log_to_file=False)
initial_logger.info("Starting main training script...")

# Define a placeholder type for the potentially complex resumed state
ResumedState = tuple[int, str, int] | None  # e.g., (round, phase_name, phase_step)


def main():
    # --- Log GPU Availability, Architecture, and Initial Memory ---
    # Move import here to satisfy linters and ensure torch is available for the block
    try:
        # Attempt to use pynvml for more detailed architecture info if available
        pynvml_available = False
        nvml_arch_map = {}
        try:
            from pynvml.smi import nvidia_smi  # noqa: F401
            import pynvml  # noqa: F401

            pynvml.nvmlInit()
            pynvml_available = True
            # Map NVML architecture enums to names (add more as needed based on pynvml constants)
            nvml_arch_map = {
                pynvml.NVML_DEVICE_ARCH_KEPLER: "Kepler",
                pynvml.NVML_DEVICE_ARCH_MAXWELL: "Maxwell",
                pynvml.NVML_DEVICE_ARCH_PASCAL: "Pascal",
                pynvml.NVML_DEVICE_ARCH_VOLTA: "Volta",
                pynvml.NVML_DEVICE_ARCH_TURING: "Turing",
                pynvml.NVML_DEVICE_ARCH_AMPERE: "Ampere",
                pynvml.NVML_DEVICE_ARCH_HOPPER: "Hopper",
                pynvml.NVML_DEVICE_ARCH_UNKNOWN: "Unknown",
            }
            initial_logger.info(
                "Initialized pynvml for detailed GPU architecture reporting."
            )
        except ImportError:
            initial_logger.warning(
                "pynvml not found. Will fall back to using PyTorch compute capability for architecture estimation. Install nvidia-ml-py (`pip install nvidia-ml-py`) for more detailed info."
            )
        except pynvml.NVMLError as e:
            initial_logger.warning(
                f"Failed to initialize pynvml: {e}. Will fall back to PyTorch compute capability."
            )
            pynvml_available = False  # Ensure flag is false if init fails

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            initial_logger.info(
                f"Found {num_gpus} CUDA-enabled GPU(s) visible to PyTorch."
            )

            # Fallback compute capability map (used if pynvml fails or isn't installed)
            cc_arch_map = {
                3: "Kepler",
                5: "Maxwell",
                6: "Pascal",
                7: "Volta/Turing",  # Needs refinement based on minor version
                8: "Ampere",
                9: "Hopper",
                # Add future major versions here
            }

            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                total_memory_gb = props.total_memory / (1024**3)
                compute_capability = f"{props.major}.{props.minor}"
                arch_name = "Unknown"

                # 1. Try getting architecture from pynvml
                nvml_arch_success = False
                if pynvml_available:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        arch_enum = pynvml.nvmlDeviceGetArchitecture(handle)
                        arch_name = nvml_arch_map.get(
                            arch_enum, f"Unknown NVML Arch ({arch_enum})"
                        )
                        nvml_arch_success = True
                    except pynvml.NVMLError as e:
                        initial_logger.warning(
                            f"pynvml failed to get architecture for GPU {i}: {e}. Falling back to compute capability."
                        )
                    except Exception as e:  # Catch potential unexpected errors
                        initial_logger.warning(
                            f"Unexpected error using pynvml for GPU {i}: {e}. Falling back to compute capability."
                        )

                # 2. Fallback to compute capability map if pynvml failed or wasn't available
                if not nvml_arch_success:
                    arch_name = cc_arch_map.get(props.major, "Unknown/Future CC")
                    # Refine Volta/Turing based on minor version if major is 7
                    if props.major == 7:
                        if props.minor == 0:
                            arch_name = "Volta"
                        elif props.minor == 5:
                            arch_name = "Turing"
                        else:
                            arch_name = f"Volta/Turing (CC {compute_capability})"  # Unknown minor version for 7.x

                # Get current memory stats for this device *for this process*
                allocated_memory_gb = 0.0
                reserved_memory_gb = 0.0
                memory_info_str = "Memory stats unavailable"
                try:
                    # Ensure context is on the correct device for memory query
                    with torch.cuda.device(i):
                        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                        reserved_memory_gb = torch.cuda.memory_reserved() / (
                            1024**3
                        )  # Includes allocated + cached by allocator
                    memory_info_str = f"Current Process Memory - Allocated: {allocated_memory_gb:.2f} GB, Reserved (Cached): {reserved_memory_gb:.2f} GB"
                except RuntimeError as e:
                    memory_info_str = f"Could not query memory stats: {e}"
                    initial_logger.warning(
                        f"  Warning querying memory for GPU {i}: {e}"
                    )
                except Exception as e:  # Catch potential unexpected errors
                    memory_info_str = f"Unexpected error querying memory stats: {e}"
                    initial_logger.warning(
                        f"  Unexpected error querying memory for GPU {i}: {e}"
                    )

                initial_logger.info(
                    f"  GPU {i}: {props.name}, Arch: {arch_name} (CC {compute_capability}), "
                    f"Total Memory: {total_memory_gb:.2f} GB. {memory_info_str}"
                )

            # Note about distributed training allocation
            initial_logger.info(
                "Note: The above list shows GPUs visible *before* framework allocation. "
                "Memory figures reflect the state *at this moment* for the current process. "
                "The distributed training framework (Accelerate/DeepSpeed) will manage final GPU allocation and memory usage based on its configuration and environment variables (e.g., CUDA_VISIBLE_DEVICES)."
            )

            # Shutdown pynvml if it was initialized
            if pynvml_available:
                try:
                    pynvml.nvmlShutdown()
                    initial_logger.info("Shut down pynvml.")
                except pynvml.NVMLError as e:
                    initial_logger.warning(f"Failed to shut down pynvml: {e}")

        else:
            initial_logger.warning(
                "torch.cuda.is_available() returned False. No CUDA-enabled GPUs detected by PyTorch. Training will likely proceed on CPU."
            )

    except ImportError:
        initial_logger.warning(
            "PyTorch ('torch') is not installed or could not be imported. Cannot check for GPU availability."
        )
    except Exception as e:
        initial_logger.error(
            f"An unexpected error occurred during GPU availability check: {e}",
            exc_info=True,
        )
        # Ensure pynvml is shut down if an error occurred after its init but before normal shutdown
        if "pynvml_available" in locals() and pynvml_available:
            try:
                pynvml.nvmlShutdown()
                initial_logger.info(
                    "Shut down pynvml after encountering an error during GPU check."
                )
            except NameError:  # pynvml might not be defined if import failed
                pass
            except pynvml.NVMLError as nvml_err:
                initial_logger.warning(
                    f"Failed to shut down pynvml during error handling: {nvml_err}"
                )

    # --- 1. Parse Arguments ---
    args: ExperimentArgs = auto_cli(
        ExperimentArgs,
        as_positional=False,
        description="PVG Experiment Runner Configuration",
    )
    initial_logger.info("Parsed arguments from command line.")
    # initial_logger.info(f"Parsed ExperimentArgs:\n{json.dumps(args.__dict__, indent=2, default=lambda o: o.__dict__)}") # TODO: Remove this.

    # Optionally log the full parsed args (can be verbose)
    verifier_mode: Literal[
        "regressor", "classifier", "inference_classifier", "inference_regressor"
    ] = (
        args.training_verifier.verifier_mode
    )  # Contains Literal["regressor", "classifier", "inference_classifier", "inference_regressor"], needed in several components
    state_tracker = StateTracker(verifier_mode=verifier_mode)
    global_step_callback = state_tracker.get_step
    global_round_callback = state_tracker.get_round
    global_phase_callback = state_tracker.get_phase

    # --- 2. Initialize Shared Foundational Components ---
    initial_logger.info("Initializing foundational components...")

    # a. AcceleratorManager (Initializes distributed environment)
    accelerator_manager = AcceleratorManager(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        ds_config_honest_prover=args.training_honest_prover.ds_config,
        ds_config_sneaky_prover=args.training_sneaky_prover.ds_config,
        ds_config_verifier=args.training_verifier.ds_config,
        wandb_config=args.wandb,
        global_step_callback=global_step_callback,
    )

    # b. Logging Reconfiguration (Now with rank/world_size)
    log_level = (
        logging.INFO
        if accelerator_manager.get_state_property("is_main_process")
        else logging.WARNING
    )
    log_dir = os.path.join(args.output_dir, "logs")
    # The logger instance is retrieved later using logging.getLogger("pvg")
    setup_logger(
        name="pvg",  # Ensure consistent root name
        level=log_level,
        rank=accelerator_manager.get_state_property("process_index"),
        world_size=accelerator_manager.get_state_property("num_processes"),
        log_to_file=True,  # Example: Enable file logging
        log_dir=log_dir,
        log_filename="training.log",
        main_process_only_file=True,  # Example: Only rank 0 writes main log
    )
    logger = logging.getLogger("pvg")  # Get the configured root logger
    logger.info("Logging reconfigured after accelerator initialization.")

    # c. Seeding (after accelerator init for potential distributed seeding needs)
    # Assuming a shared seed for simplicity, adjust if phase-specific seeds are needed
    seed = args.training_honest_prover.seed
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    sampler_args = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_generations": args.rl.num_generations,
        "num_iterations": args.rl.num_iterations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    # d. DataManager
    logger.info("Initializing DataManager...")
    data_manager = DataManager(
        accelerator_manager=accelerator_manager,
        dataset_config=args.dataset,
        sampler_args=sampler_args,
        seed=seed,
        global_phase_callback=global_phase_callback,
        verifier_mode=verifier_mode,
    )
    # Load datasets; create dataloaders; prepare dataloaders
    data_manager.load_datasets()  # Will create the datasets as attributes (self.prover_train_dataset, self.prover_eval_dataset, self.verifier_train_dataset, self.verifier_eval_dataset)
    data_manager.create_dataloaders()  # NOTE: Very wonky.
    data_manager.prepare_dataloaders()
    logger.info("DataManager initialized.")

    # e. ModelManager
    logger.info("Initializing ModelManager...")
    model_manager = ModelManager(
        accelerator_manager=accelerator_manager,
        honest_config=args.honest_prover,
        sneaky_config=args.sneaky_prover,
        verifier_config=args.verifier,
        honest_training_config=args.training_honest_prover,
        sneaky_training_config=args.training_sneaky_prover,
        verifier_training_config=args.training_verifier,
        rl_config=args.rl,  # For beta/ref models
        global_phase_callback=global_phase_callback,
        global_round_callback=global_round_callback,
        global_step_callback=global_step_callback,
    )
    # model_manager.initialize_model_manager() # Loads phase-specific models, prepares them
    logger.info("ModelManager initialized.")

    # f. OptimizerSchedulerManager
    logger.info("Initializing OptimizerSchedulerManager...")
    # Pass only relevant shared args for schedulers and optimizers
    shared_training_config = {
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_warmup_steps": args.num_warmup_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
    }
    optimizer_scheduler_manager = OptimizerSchedulerManager(
        honest_training_config=args.training_honest_prover,
        sneaky_training_config=args.training_sneaky_prover,
        verifier_training_config=args.training_verifier,  # Add verifier training args
        shared_training_config=shared_training_config,
        model_manager=model_manager,  # Needs prepared models for params
        data_manager=data_manager,  # Needs dataloader lengths for step calculation
        accelerator_manager=accelerator_manager,
        global_step_callback=global_step_callback,
        global_phase_callback=global_phase_callback,
        global_round_callback=global_round_callback,
    )
    # NOTE: Init components is called when the phase changes or phase-specific training starts. Acts as a reset for the model manager. (--> memory efficient, as we only init optimizer/scheduler when we train a phase)
    # optimizer_scheduler_manager.create_optimizers()
    # assert data_manager.dataloaders is not None
    # assert hasattr(data_manager.dataloaders["provers"]["train_dataloader"], "_accelerator_device"), "Dataloaders have not been prepared by the accelerator"
    # assert hasattr(data_manager.dataloaders["verifier"]["train_dataloader"], "_accelerator_device"), "Dataloaders have not been prepared by the accelerator"

    logger.info("OptimizerSchedulerManager initialized.")

    # h. MetricsLogger
    logger.info("Initializing MetricsLogger...")
    # The callback will be provided by the orchestrator/phase trainers when logging
    metrics_logger = MetricsLogger(
        accelerator_manager=accelerator_manager,
        wandb_config=args.wandb,
        global_step_callback=global_step_callback,
        global_phase_callback=global_phase_callback,
    )
    if args.wandb.use_wandb:
        # Pass the full config dict for initial logging
        metrics_logger.setup_wandb(config=args.__dict__)
        # TODO: Log model parameters if needed (might require helper in ModelManager or here)
        # if accelerator_manager.get_state_property('is_main_process'):
        #     log_model_parameters_to_wandb(model_manager, metrics_logger)
    logger.info("MetricsLogger initialized.")

    # i. VLLMOrchestrator
    logger.info("Initializing VLLMOrchestrator...")
    llm_log_dir = os.path.join(args.output_dir, "llm_interaction_logs")
    # Create dir only on main process to avoid race conditions
    if accelerator_manager.get_state_property("is_main_process"):
        os.makedirs(llm_log_dir, exist_ok=True)
        logger.info(f"LLM interaction logs will be saved to: {llm_log_dir}")
    # Callback provided later by orchestrator/phase trainer
    vllm_orchestrator = VLLMOrchestrator(
        accelerator_manager=accelerator_manager,
        vllm_config_honest=args.vllm_honest_prover,
        vllm_config_sneaky=args.vllm_sneaky_prover,
        vllm_config_verifier=args.vllm_verifier,
        tokenizer_callback=data_manager.get_tokenizer,
        llm_interaction_log_dir=llm_log_dir,
        global_step_callback=global_step_callback,
    )
    logger.info("VLLMOrchestrator initialized.")

    # j. CheckpointManager
    # logger.info("Initializing CheckpointManager...")
    # checkpoint_manager = CheckpointManager(
    #      accelerator_manager=accelerator_manager,
    #      output_dir=args.output_dir,
    #      save_steps=args.save_steps, # Or phase-specific save steps?
    #      args_to_save=args,
    #      tokenizer_to_save=data_manager.get_tokenizer()
    # )
    # logger.info("CheckpointManager initialized.")

    # # k. DataGenerator
    # if accelerator_manager.get_state_property('is_main_process'):
    #     logger.info("Initializing DataGenerator...")
    #     data_generator = DataGenerator(
    #         args=args,
    #         model_manager=model_manager,
    #         optimizer_scheduler_manager=optimizer_scheduler_manager,
    #         data_manager=data_manager,
    #         metrics_logger=metrics_logger,
    #         vllm_orchestrator=vllm_orchestrator,
    #         state_tracker=state_tracker,
    #     )
    #     logger.info("DataGenerator init done. Find the splits on HuggingFace Hub.")

    # wait here for everyone
    # accelerator_manager.wait_for_everyone()

    # # k. RL Components (Instantiate if needed by any phase)
    # logger.info("Initializing RL components (if applicable)...")
    # # rl_step_logic = RLStepLogic(...) # Pass managers, args.rl
    # # loss_calculator = LossCalculator(...) # Pass args.rl
    # # logger.info("RL components initialized.")
    # # Assign placeholder None if not used, or handle conditionally
    # rl_step_logic = None
    # loss_calculator = None

    # --- 3. Initialize Top-Level Orchestrator ---
    logger.info("Initializing TrainingPhaseOrchestrator...")
    orchestrator = TrainingPhaseOrchestrator(
        args=args,
        accelerator_manager=accelerator_manager,
        model_manager=model_manager,
        data_manager=data_manager,
        optimizer_scheduler_manager=optimizer_scheduler_manager,
        metrics_logger=metrics_logger,
        vllm_orchestrator=vllm_orchestrator,
        state_tracker=state_tracker,
        # checkpoint_manager=checkpoint_manager,
        # Pass RL components if they exist
        # rl_step_logic=rl_step_logic,
        # loss_calculator=loss_calculator,
    )
    logger.info("TrainingPhaseOrchestrator initialized.")

    # # --- 4. Load Checkpoint (Optional) ---
    # resumed_state: ResumedState = None
    # if args.resume_from_checkpoint:
    #     logger.warning(f"Attempting to resume training from checkpoint: {args.resume_from_checkpoint}")
    #     # CheckpointManager loads state into the existing manager objects
    #     # We need to get back the loop state (round, phase, step) to pass to the orchestrator
    #     try:
    #         # TODO: CheckpointManager load needs to return loop state
    #         # loaded_state = checkpoint_manager.load_checkpoint(
    #         #     args.resume_from_checkpoint,
    #         #     model_manager,
    #         #     optimizer_scheduler_manager
    #         # )
    #         # resumed_state = (loaded_state['round'], loaded_state['phase'], loaded_state['step'])
    #         # logger.info(f"Resuming from Round: {resumed_state[0]}, Phase: {resumed_state[1]}, Step: {resumed_state[2]}")
    #         logger.error("Checkpoint loading logic needs implementation in CheckpointManager to return loop state.")
    #         # For now, we cannot properly resume loop state.
    #         # checkpoint_manager.load_checkpoint(
    #         #      args.resume_from_checkpoint, model_manager, optimizer_scheduler_manager
    #         # )

    #     except FileNotFoundError:
    #         logger.error(f"Resume checkpoint directory not found: {args.resume_from_checkpoint}. Starting from scratch.")
    #     except Exception as e:
    #         logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.", exc_info=True)
    #         resumed_state = None # Ensure we start fresh on error

    # --- 5. Start Training Rounds ---
    logger.info("Starting training rounds...")
    try:
        orchestrator.run_training_rounds()
        logger.info("Training rounds completed successfully.")
    except Exception:
        logger.error("Training failed with an exception.", exc_info=True)
        # Optionally perform cleanup or save emergency checkpoint
        raise  # Re-raise the exception after logging
    finally:
        # --- 6. End Training ---
        logger.info("Ending training run.")
        # Accelerator might handle this, but explicit cleanup can be good
        # e.g., close WandB run if MetricsLogger doesn't do it automatically
        # wandb.finish() ?


if __name__ == "__main__":
    main()
