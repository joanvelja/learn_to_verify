# pvg/main.py

# The prover π ∗ used for sampling solutions for verifier training is a mixture of an initial base prover and previous round provers, each balanced to have equal number of correct and incorrect solutions. Each of the previous rounds has equal ratio in the mixture, which is tuned as a hyperparameter. In round 0, the solutions are from the base prover sampled via a few-shot prompt (App. [H)](#page-31-0). In later rounds of training, we replace part of the solutions with those from the new provers. Hence we always use the same number of solutions per problem. The amount of verifier optimization is constant over the rounds.

import logging
import os
from typing import Literal, cast

from jsonargparse import auto_cli
from transformers import set_seed

# Core Components (Managers)
from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.data_manager import DataManager
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.model_manager import ModelManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.state_tracker import StateTracker
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.components.formatter import Formatter
from pvg.components.code_evaluator import BatchEvaluator, EvaluationConfig

# Configuration
from pvg.config.args import ExperimentArgs

# Top-Level Orchestrator
from pvg.orchestrator import TrainingPhaseOrchestrator

# RL Components
from pvg.rl.grpo import GRPO
from pvg.utils.gpu_info import gpu_info

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
    gpu_info(initial_logger=initial_logger)
    # --- 1. Parse Arguments ---
    args: ExperimentArgs = auto_cli(
        ExperimentArgs,
        as_positional=False,
        description="PVG Experiment Runner Configuration",
    )
    initial_logger.info("Parsed arguments from command line.")

    verifier_mode: Literal[
        "regressor", "classifier", "inference_classifier", "inference_regressor"
    ] = cast(
        Literal[
            "regressor", "classifier", "inference_classifier", "inference_regressor"
        ],
        args.training_verifier.verifier_mode,
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
    seed = args.training_sneaky_prover.seed
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
    data_manager.create_dataloaders()
    logger.info("DataManager initialized.")

    # e. ModelManager
    logger.info("Initializing ModelManager...")
    model_manager = ModelManager(
        accelerator_manager=accelerator_manager,
        sneaky_config=args.sneaky_prover,
        verifier_config=args.verifier,
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
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
    }
    optimizer_scheduler_manager = OptimizerSchedulerManager(
        sneaky_training_config=args.training_sneaky_prover,
        verifier_training_config=args.training_verifier,
        shared_training_config=shared_training_config,
        model_manager=model_manager,
        data_manager=data_manager,
        accelerator_manager=accelerator_manager,
        global_step_callback=global_step_callback,
        global_phase_callback=global_phase_callback,
        global_round_callback=global_round_callback,
    )
    logger.info("OptimizerSchedulerManager initialized.")

    # h. MetricsLogger
    logger.info("Initializing MetricsLogger...")
    # The callback will be provided by the orchestrator/phase trainers when logging
    metrics_logger = MetricsLogger(
        accelerator_manager=accelerator_manager,
        global_step=global_step_callback,
    )
    accelerator_manager.setup_wandb(config=args.__dict__)
    logger.info("MetricsLogger initialized.")
    accelerator_manager.wait_for_everyone()  # To ensure that all is set up before setting up vllm orchestrator

    # i. VLLMOrchestrator
    logger.info("Initializing VLLMOrchestrator...")
    llm_log_dir = accelerator_manager.get_llm_interaction_log_dir()
    logger.info(f"LLM interaction logs will be saved to: {llm_log_dir}")
    # Callback provided later by orchestrator/phase trainer
    vllm_orchestrator = VLLMOrchestrator(
        accelerator_manager=accelerator_manager,
        vllm_config_sneaky=args.vllm_sneaky_prover,
        vllm_config_verifier=args.vllm_verifier,
        tokenizer_callback=data_manager.get_tokenizer,
        llm_interaction_log_dir=llm_log_dir,
        global_step_callback=global_step_callback,
    )
    logger.info("VLLMOrchestrator initialized.")

    # k. RL Components
    grpo = GRPO(args.rl)

    # --- 3. Initialize Top-Level Orchestrator ---
    # j. Formatter
    formatter = Formatter(
        tokenizer=data_manager.get_tokenizer(),
    )

    # l. BatchEvaluator
    config = EvaluationConfig(
        step_timeouts={"exec": 2, "test_gen": 15, "verify": 15},
        total_timeout=35,
        success_threshold=0.85,
    )
    batch_evaluator = BatchEvaluator(config=config)

    # m. TrainingPhaseOrchestrator
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
        grpo=grpo,
        formatter=formatter,
        batch_evaluator=batch_evaluator,
        # checkpoint_manager=checkpoint_manager,
    )
    logger.info("TrainingPhaseOrchestrator initialized.")

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


if __name__ == "__main__":
    main()
