# pvg/config/args.py

from dataclasses import dataclass, field
from typing import Literal
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    """Arguments pertaining to model loading."""

    name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (branch name, tag name or commit id)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."
        },
    )
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Hugging Face cache."},
    )
    torch_dtype: str = field(
        default="auto",
        metadata={
            "help": "The data type of the model. Choose between 'auto', 'float16', 'float32', 'bfloat16', or 'int8'."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False, metadata={"help": "Whether or not to use low CPU memory usage."}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention for faster training."},
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={
            "help": "Attention implementation to use: 'eager', 'flash_attention_2', or 'sdpa'."
        },
    )
    device_map: str | None = field(
        default=None,
        metadata={
            "help": "Device map for model distribution. 'auto' for automatic mapping, None for no mapping."
        },
    )
    model_max_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum sequence length the model can handle. Overrides model's default if set."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8-bit precision."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4-bit precision."},
    )


@dataclass
class TrainingArgs:
    """Arguments pertaining to the training loop itself."""

    # --- DeepSpeed Config Paths ---
    ds_config: str | None = field(
        default=None, metadata={"help": "Path to the DeepSpeed config file."}
    )
    # --- Training Hyperparameters ---
    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    learning_rate: float = field(
        default=5e-6, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    max_grad_norm: float = field(
        default=0.1, metadata={"help": "Max gradient norm for clipping."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory."},
    )
    # --- Liger Kernel Optimization ---
    apply_liger_kernel: bool = field(
        default=False,
        metadata={
            "help": "Apply Liger kernel optimization. Yields better memory usage and training speed for large models (up to 60% memory savings)."
        },
    )
    verifier_mode: (
        Literal[
            "regressor", "classifier", "inference_classifier", "inference_regressor"
        ]
        | None
    ) = field(default=None, metadata={"help": "The mode of the verifier."})
    # Note: gradient_accumulation_steps, epochs, max_steps, lr_scheduler, warmup_steps,
    # logging_steps, save_steps, eval_steps, mixed_precision are handled by the top-level ExperimentArgs
    # or inferred, as they often need coordination between models or the overall loop.


@dataclass
class DatasetArgs:
    """Arguments pertaining to dataset loading and processing."""

    dataset_name: str | None = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path. If None, uses model's path."
        },
    )
    train_num_samples: int | None = field(
        default=None,
        metadata={
            "help": "Number of training samples to use (for debugging). None means use all."
        },
    )
    eval_num_samples: int | None = field(
        default=None,
        metadata={
            "help": "Number of evaluation samples to use (for debugging). None means use all."
        },
    )
    cache_dir: str | None = field(
        default=None, metadata={"help": "Path to cache directory for dataset files."}
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    min_length: int | None = field(
        default=None,
        metadata={"help": "Minimum sequence length to keep after tokenization."},
    )


@dataclass
class VLLMServerArgs:
    """Arguments pertaining to the vLLM server connection and generation parameters for a specific model."""

    host: str = field(
        default="127.0.0.1", metadata={"help": "Host address of the vLLM server."}
    )
    port: int = field(default=8000, metadata={"help": "Port of the vLLM server."})
    timeout: float = field(
        default=60.0, metadata={"help": "Connection timeout for the vLLM server."}
    )
    # Generation parameters specific to this model's server/role
    temperature: float = field(
        default=1.0, metadata={"help": "Temperature for sampling."}
    )
    top_p: float = field(default=1.0, metadata={"help": "Top-p for sampling."})
    top_k: int = field(
        default=-1,
        metadata={"help": "Top-k for sampling. -1 means no top-k filtering."},
    )
    max_tokens: int = field(
        default=512, metadata={"help": "Maximum number of new tokens to generate."}
    )
    repetition_penalty: float = field(
        default=1.0, metadata={"help": "Repetition penalty. 1.0 means no penalty."}
    )
    frequency_penalty: float = field(
        default=0.0, metadata={"help": "Frequency penalty."}
    )
    min_p: float = field(
        default=0.0, metadata={"help": "Minimum probability for nucleus sampling."}
    )
    stop_sequences: list[str] | None = field(
        default=None, metadata={"help": "List of sequences to stop generation at."}
    )
    logprobs: int | None = field(
        default=None,
        metadata={
            "help": "Request logprobs for the top N tokens at each step (e.g., for verifier)."
        },
    )
    # vLLM Server launch args (used in shell script, but good to document here)
    # gpu_memory_utilization: float = 0.9
    # tensor_parallel_size: int = 1
    # dtype: str = "auto"
    # max_model_len: Optional[int] = None
    # enable_prefix_caching: Optional[bool] = None


@dataclass
class RLArgs:
    """Arguments specific to the Reinforcement Learning algorithm (GRPO)."""

    num_generations: int = field(
        default=2, metadata={"help": "Number of completions to generate per prompt."}
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of times to reuse generations for gradient updates."},
    )
    beta: float = field(
        default=0.0, metadata={"help": "Coefficient for the KL penalty term."}
    )
    epsilon_low: float = field(
        default=0.2, metadata={"help": "Lower bound for the GRPO clipping ratio."}
    )
    epsilon_high: float = field(
        default=0.28, metadata={"help": "Upper bound for the GRPO clipping ratio."}
    )
    scale_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to standardize advantages (divide by std)."},
    )
    adv_clip: float | None = field(
        default=None, metadata={"help": "Clip advantages for provers."}
    )
    nan_reward_value: float = field(
        default=-0.1,
        metadata={
            "help": "Value to assign if reward extraction fails (e.g., returns NaN)."
        },
    )
    # NEW: Reward normalization parameters
    normalize_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply batch normalization to rewards using running statistics."
        },
    )
    reward_norm_momentum: float = field(
        default=0.99,
        metadata={
            "help": "Momentum factor for running mean update: μₜ = α·μₜ₋₁ + (1-α)·mean(batch)."
        },
    )
    normalize_reward_std: bool = field(
        default=False,
        metadata={
            "help": "Whether to normalize by running standard deviation in addition to mean."
        },
    )
    reward_norm_eps: float = field(
        default=1e-8,
        metadata={
            "help": "Small epsilon value added to running std before division to prevent numerical issues."
        },
    )


@dataclass
class WandbArgs:
    """Arguments for Weights & Biases logging."""

    use_wandb: bool = field(
        default=True, metadata={"help": "Whether to use Weights & Biases for logging."}
    )
    wandb_project_name: str | None = field(
        default="disjoint_sequential_training", metadata={"help": "W&B project name."}
    )
    wandb_entity: str | None = field(
        default=None, metadata={"help": "W&B entity (username or team name)."}
    )
    wandb_run_name: str | None = field(
        default=None, metadata={"help": "W&B run name. Defaults to a generated name."}
    )
    wandb_hist_freq_multiplier: int = field(
        default=50, metadata={"help": "Log histograms every N * logging_steps steps."}
    )
    wandb_table_freq_multiplier: int = field(
        default=50, metadata={"help": "Log tables every N * logging_steps steps."}
    )
    wandb_table_samples: int = field(
        default=64, metadata={"help": "Number of samples in training table."}
    )
    wandb_eval_table_samples: int = field(
        default=32, metadata={"help": "Number of samples in eval table."}
    )
    wandb_log_system_freq_multiplier: int = field(
        default=50,
        metadata={"help": "Log system metrics every N * logging_steps steps."},
    )
    output_dir: str | None = field(
        default=None, metadata={"help": "Output directory, copied from ExperimentArgs."}
    )


@dataclass
class ExperimentArgs:
    """Top-level arguments coordinating the entire experiment."""

    # --- Model Definitions ---
    honest_prover: ModelArgs = field(
        default_factory=lambda: ModelArgs(),
        metadata={"help": "Configuration for the honest prover model."},
    )
    sneaky_prover: ModelArgs = field(
        default_factory=lambda: ModelArgs(),
        metadata={"help": "Configuration for the sneaky prover model."},
    )
    verifier: ModelArgs = field(
        default_factory=lambda: ModelArgs(),
        metadata={"help": "Configuration for the verifier model."},
    )

    # --- Dataset ---
    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(),
        metadata={"help": "Dataset configuration."},
    )

    # --- RL Algorithm ---
    rl: RLArgs = field(
        default_factory=RLArgs,
        metadata={"help": "Reinforcement learning algorithm configuration."},
    )

    # --- Logging ---
    wandb: WandbArgs = field(
        default_factory=WandbArgs,
        metadata={"help": "Weights & Biases logging configuration."},
    )

    # --- vLLM Server Connections ---
    vllm_honest_prover: VLLMServerArgs = field(
        default_factory=VLLMServerArgs,
        metadata={"help": "vLLM server configuration for the honest prover."},
    )
    vllm_sneaky_prover: VLLMServerArgs = field(
        default_factory=VLLMServerArgs,
        metadata={"help": "vLLM server configuration for the sneaky prover."},
    )
    vllm_verifier: VLLMServerArgs = field(
        default_factory=VLLMServerArgs,
        metadata={"help": "vLLM server configuration for the verifier."},
    )

    # --- Training Hyperparameters (Can be potentially different per model) ---
    training_honest_prover: TrainingArgs = field(
        default_factory=TrainingArgs,
        metadata={"help": "Training configuration for the honest prover."},
    )
    training_sneaky_prover: TrainingArgs = field(
        default_factory=TrainingArgs,
        metadata={"help": "Training configuration for the sneaky prover."},
    )
    training_verifier: TrainingArgs = field(
        default_factory=TrainingArgs,
        metadata={"help": "Training configuration for the verifier."},
    )

    # --- Datamix Generation ---
    generation_batch_size: int = field(
        default=512, metadata={"help": "Batch size for data generation."}
    )
    num_rounds_to_keep: int = field(
        default=10, metadata={"help": "Number of rounds to keep for data generation."}
    )
    new_sample_weight_target: float = field(
        default=0.8,
        metadata={"help": "Target proportion of samples from the latest round."},
    )

    # --- Shared Training Loop Arguments ---
    output_dir: str = field(
        default="",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Total number of training epochs to perform."}
    )
    max_train_steps: int | None = field(
        default=None,
        metadata={
            "help": "If set, overrides num_train_epochs. Total number of training steps to perform."
        },
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = field(default="linear", metadata={"help": "The scheduler type to use."})
    num_warmup_steps: int = field(
        default=0, metadata={"help": "Number of steps for the linear warmup phase."}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=100, metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(
        default=100, metadata={"help": "Run evaluation every X updates steps."}
    )
    mixed_precision: Literal["no", "fp16", "bf16"] | None = field(
        default="bf16",
        metadata={
            "help": "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."
        },
    )
    sync_steps: int = field(
        default=1,
        metadata={
            "help": "Frequency (in global steps) to synchronize weights to vLLM servers."
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={
            "help": "The path to a folder containing a checkpoint to resume training from."
        },
    )

    # --- Checkability Training ---
    num_rounds: int = field(
        default=8, metadata={"help": "Number of training rounds to perform."}
    )

    # --- Hugging Face Hub ---
    hf_token: str | None = field(
        default=None,
        metadata={"help": "Hugging Face token for pushing datasets to the hub."},
    )

    # --- System Prompts ---
    honest_prover_system_prompt: str = field(
        default="", metadata={"help": "System prompt for the honest prover."}
    )
    sneaky_prover_system_prompt: str = field(
        default="", metadata={"help": "System prompt for the sneaky prover."}
    )
    verifier_system_prompt: str = field(
        default="", metadata={"help": "System prompt for the verifier."}
    )

    def __post_init__(self):

        # 0. Check for instantiations of ModelArgs name_or_path
        if self.honest_prover.name_or_path is None:
            raise ValueError("honest_prover.name_or_path is not set.")
        if self.sneaky_prover.name_or_path is None:
            raise ValueError("sneaky_prover.name_or_path is not set.")
        if self.verifier.name_or_path is None:
            raise ValueError("verifier.name_or_path is not set.")

        # 1. Tokenizer Path Default
        if self.dataset.tokenizer_name_or_path is None:
            if self.honest_prover.name_or_path:
                self.dataset.tokenizer_name_or_path = self.honest_prover.name_or_path
                logger.info(
                    f"Tokenizer path not specified, using honest prover path: {self.dataset.tokenizer_name_or_path}"
                )
            else:
                raise ValueError(
                    "Tokenizer path not specified and honest_prover.name_or_path is empty. Tokenizer may not be loaded correctly."
                )

        # Iterate through models for checks
        models_to_check = {
            "honest_prover": self.honest_prover,
            "sneaky_prover": self.sneaky_prover,
            "verifier": self.verifier,
        }

        for model_name, model_args in models_to_check.items():
            # 2. Quantization Conflicts
            if model_args.load_in_8bit and model_args.load_in_4bit:
                raise ValueError(
                    f"Model '{model_name}' cannot have both load_in_8bit and load_in_4bit set to True."
                )

            # 3. Attention Implementation Consistency
            if (
                model_args.attn_implementation == "flash_attention_2"
                and not model_args.use_flash_attention
            ):
                logger.warning(
                    f"WARNING: Model '{model_name}' has attn_implementation='flash_attention_2' but use_flash_attention=False. Consider setting use_flash_attention=True for Flash Attention 2."
                )
                # Optionally force it: self.honest_prover.use_flash_attention = True
            elif (
                model_args.use_flash_attention
                and model_args.attn_implementation != "flash_attention_2"
            ):
                logger.warning(
                    f"WARNING: Model '{model_name}' has use_flash_attention=True but attn_implementation='{model_args.attn_implementation}'. Consider setting attn_implementation='flash_attention_2' to utilize Flash Attention."
                )

        # 4. Training Steps vs. Epochs
        if self.max_train_steps is not None and self.max_train_steps > 0:
            logger.info(
                f"max_train_steps ({self.max_train_steps}) is set, overriding num_train_epochs ({self.num_train_epochs})."
            )

        # 5. Batch Size and Accumulation
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be a positive integer.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")

        # 6. RL Beta vs. Iterations Warning
        if self.rl.num_iterations > 1 and self.rl.beta == 0.0:
            logger.warning(
                "rl.num_iterations > 1 is typically used with rl.beta > 0 (KL penalty) to benefit from reusing reference model logps."
            )
        elif self.rl.num_iterations <= 0:
            raise ValueError("rl.num_iterations must be a positive integer.")

        # 7. RL Epsilon Clipping
        if self.rl.epsilon_low <= 0:
            raise ValueError("rl.epsilon_low must be positive.")
        if self.rl.epsilon_high <= self.rl.epsilon_low:
            raise ValueError(
                "rl.epsilon_high must be strictly greater than rl.epsilon_low."
            )

        # 8. Verifier Logprobs Requirement
        if self.vllm_verifier.logprobs is None or self.vllm_verifier.logprobs <= 0:
            logger.warning(
                "vllm_verifier.logprobs is not set or is non-positive. Verifier might require logprobs for reward calculation."
            )

        # 9. WandB Configuration
        if self.wandb.use_wandb:
            if not self.wandb.wandb_project_name:
                logger.warning(
                    "wandb.use_wandb is True, but wandb.wandb_project_name is not set."
                )
            if not self.wandb.wandb_entity:
                logger.warning(
                    "wandb.use_wandb is True, but wandb.wandb_entity is not set."
                )
            # Check multipliers are >= 1
            if self.wandb.wandb_hist_freq_multiplier < 1:
                raise ValueError("wandb.wandb_hist_freq_multiplier must be >= 1.")
            if self.wandb.wandb_table_freq_multiplier < 1:
                raise ValueError("wandb.wandb_table_freq_multiplier must be >= 1.")
            if self.wandb.wandb_log_system_freq_multiplier < 1:
                raise ValueError("wandb.wandb_log_system_freq_multiplier must be >= 1.")

        # 10. DeepSpeed Configuration Paths
        # Consider adding checks if DeepSpeed is actually intended/enabled
        if not self.training_honest_prover.ds_config:
            logger.warning("training_honest_prover.ds_config path is empty.")
        # Optional: elif not os.path.exists(self.ds_config_honest_prover):
        #    raise FileNotFoundError(f"DeepSpeed config for honest prover not found: {self.ds_config_honest_prover}")

        if not self.training_sneaky_prover.ds_config:
            logger.warning("training_sneaky_prover.ds_config path is empty.")
        # Optional: elif not os.path.exists(self.ds_config_sneaky_prover):
        #    raise FileNotFoundError(f"DeepSpeed config for sneaky prover not found: {self.ds_config_sneaky_prover}")

        # 11. Save/Eval/Logging Steps
        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be positive.")
        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive.")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive.")

        # 12. Resume Path (Optional Check)
        if self.resume_from_checkpoint and not os.path.isdir(
            self.resume_from_checkpoint
        ):
            # Check if it's a file instead? Depends on how checkpoint loading works.
            # For now, assume it should be a directory.
            raise FileNotFoundError(
                f"Resume checkpoint directory not found: {self.resume_from_checkpoint}"
            )

        # 13. Send temperature values for each model to the TrainingArgs if liger kernel is applied
        if (
            self.training_honest_prover.apply_liger_kernel
        ):  # Needed for LinearGRPOLoss object
            self.training_honest_prover.temperature = (
                self.vllm_honest_prover.temperature
            )
        if (
            self.training_sneaky_prover.apply_liger_kernel
        ):  # Needed for LinearGRPOLoss object
            self.training_sneaky_prover.temperature = (
                self.vllm_sneaky_prover.temperature
            )
        if self.training_verifier.apply_liger_kernel:
            self.training_verifier.temperature = self.vllm_verifier.temperature

        # 14. copy the output_dir to the wandb args
        self.wandb.output_dir = self.output_dir
