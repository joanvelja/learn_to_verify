import torch
from copy import deepcopy
import deepspeed
import re
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoTokenizer
from typing import Literal
from pvg.data.prompts import (
    BASE_HONEST,
    BASE_SNEAKY,
    BASE_VERIFIER,
    INSTRUCT_HONEST,
    INSTRUCT_SNEAKY,
    INSTRUCT_VERIFIER,
)
from trl.trainer.utils import pad

logger = logging.getLogger(__name__)


# torch.nanstd doesn't exist, so we define it here
# def nanstd(tensor: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

#     Args:
#         tensor (`torch.Tensor`):
#             Input tensor of shape `(N,)`.

#     Returns:
#         `torch.Tensor`:
#             Standard deviation of the tensor, ignoring NaNs.
#     """
#     variance = torch.nanmean(
#         (tensor - torch.nanmean(tensor, keepdim=True)) ** 2
#     )  # Compute variance ignoring NaNs
#     count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
#     variance *= count / (count - 1)  # Bessel's correction
#     return torch.sqrt(variance)


# Edited version to handle 2D tensors
def nanstd(
    tensor: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    correction: int = 1,
) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor along a given dimension, ignoring NaNs.

    Args:
        tensor (`torch.Tensor`): Input tensor.
        dim (`int`, *optional*): Dimension along which to compute the standard deviation.
                                  If None, compute over the entire tensor.
        keepdim (`bool`): Whether the output tensor has `dim` retained or not.
        correction (`int`): Difference between the sample size and sample degrees of freedom.
                             Defaults to 1 (Bessel's correction).

    Returns:
        `torch.Tensor`: Standard deviation of the tensor, ignoring NaNs.
    """
    # Calculate mean, keeping dim for broadcasting
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)

    # Calculate squared deviations, propagating NaNs correctly
    # Where tensor is NaN, deviation should be NaN
    # Where tensor is not NaN, deviation is (tensor - mean)**2
    squared_dev = torch.where(torch.isnan(tensor), torch.nan, (tensor - mean) ** 2)

    # Calculate variance (mean of squared deviations), keeping dim temporarily for correction calculation
    variance = torch.nanmean(
        squared_dev, dim=dim, keepdim=True
    )  # Always keepdim=True here

    # Adjust for Bessel's correction if needed
    if correction != 0:
        # Count non-NaN elements along the dimension
        # Need keepdim=True for broadcasting with variance
        count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=True)
        # Ensure we don't divide by zero or negative numbers
        n = count.clamp(min=correction)
        factor = n / (n - correction).clamp(min=1e-8)  # Shape will have dim kept
        variance = (
            variance * factor
        )  # Both variance and factor have dim kept, broadcasting works

    # Apply final sqrt
    std_dev = torch.sqrt(variance.clamp(min=0))

    # Squeeze dimension only at the end if keepdim was False
    if dim is not None and not keepdim:
        std_dev = std_dev.squeeze(dim)

    return std_dev


def prepare_deepspeed(model, accelerator):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10
                    * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9
                    * hidden_size
                    * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


# @dataclass
# class ModelConfig:
#     """Arguments related to model paths and identifiers."""
#     honest_prover_name_or_path: str = field(default="", metadata={"help": "Path/ID for Honest Prover (training & vLLM)."})
#     sneaky_prover_name_or_path: str = field(default="", metadata={"help": "Path/ID for Sneaky Prover (training & vLLM)."})
#     verifier_name_or_path: str = field(default="", metadata={"help": "Path/ID for Verifier (training & vLLM)."})
#     tokenizer_name_or_path: str | None = field(
#         default=None, metadata={"help": "Path/ID for Tokenizer. If None, uses model_a_name_or_path."}
#     )
#     apply_liger_kernel: bool = field(default=True, metadata={"help": "Apply Liger kernel to the model. See liger_kernel on github."})

# @dataclass
# class DataConfig:
#     """Arguments related to data paths and processing."""
#     dataset_name: str = field(default="", metadata={"help": "The name of the dataset to use (via the datasets library)."})
#     train_num_samples: int | None = field(default=None, metadata={"help": "Number of training samples to use. If None, uses all samples."})


# @dataclass
# class TrainingConfig:
#     """Core training hyperparameters."""
#     learning_rate_a: float = field(default=5e-5, metadata={"help": "Initial learning rate for Honest Prover."})
#     learning_rate_b: float = field(default=5e-5, metadata={"help": "Initial learning rate for Sneaky Prover."})
#     weight_decay_a: float = field(default=0.0, metadata={"help": "Weight decay for Honest Prover optimizer."})
#     weight_decay_b: float = field(default=0.0, metadata={"help": "Weight decay for Sneaky Prover optimizer."})
#     per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/core for training."})
#     per_device_eval_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/core for evaluation."})
#     gradient_accumulation_steps: int = field(default=1, metadata={"help": "Steps for gradient accumulation."})
#     gradient_checkpointing: bool = field(default=False, metadata={"help": "Use gradient checkpointing."})
#     lr_scheduler_type: str = field(default="linear", metadata={"help": "Learning rate scheduler type."})
#     num_warmup_steps: int = field(default=0, metadata={"help": "Number of warmup steps for the scheduler."})
#     num_train_epochs: int = field(default=1, metadata={"help": "Total number of training epochs."}) # Default to 1 epoch
#     max_train_steps: int | None = field(default=None, metadata={"help": "Override num_train_epochs, train for specific steps."})
#     max_grad_norm_a: float | None = field(default=1.0, metadata={"help": "Max gradient norm for Honest Prover."}) # Default to 1.0
#     max_grad_norm_b: float | None = field(default=1.0, metadata={"help": "Max gradient norm for Sneaky Prover."}) # Default to 1.0
#     # RL specific args
#     sync_steps: int = field(default=1, metadata={"help": "Number of steps to sync weights."})
#     num_generations: int = field(default=2, metadata={"help": "Number of generations per prompt (for Advantage Estimation in GRPO)."})
#     num_iterations: int = field(
#     default=1,
#     metadata={
#         "help": (
#             "The number of optimization epochs (policy updates) to perform using the same batch of generated "
#             "completions (rollout data) before generating a new batch. Corresponds to 'μ' in the GRPO paper. "
#             "Setting this > 1 reuses the generated data for multiple gradient steps, potentially improving sample "
#             "efficiency but requires the PPO clipped objective (automatically used by the trainer when > 1) "
#             "to maintain stability as the policy drifts from the one that generated the data. Default is 1."
#             )
#         }
#     )
#     beta: float = field(default=0.0, metadata={"help": "Beta for KL-penalty in GRPO. If 0, no KL-penalty is applied (and accordingly, no reference policy is loaded.)"})
#     # Missing: further RL args! (TODO: Check Will Brown's repo for them)


# @dataclass
# class DistributedConfig:
#     """Configurations for distributed training and backends."""
#     ds_config_honest_prover: str = field(default="", metadata={"help": "Path to the DeepSpeed config file for Honest Prover."})
#     ds_config_sneaky_prover: str = field(default="", metadata={"help": "Path to the DeepSpeed config file for Sneaky Prover."})
#     mixed_precision: str | None = field(
#         default='bf16', metadata={"help": "Mixed precision ('no', 'fp16', 'bf16')."}
#     )

# @dataclass
# class InferenceConfig:
#     """Configurations for vLLM inference servers."""
#     vllm_host_a: str = field(default="127.0.0.1", metadata={"help": "Host IP for vLLM server A."}) # Default to localhost
#     vllm_port_a: int = field(default=8000, metadata={"help": "Port for vLLM server A."})
#     vllm_host_b: str = field(default="127.0.0.1", metadata={"help": "Host IP for vLLM server B."}) # Default to localhost
#     vllm_port_b: int = field(default=8001, metadata={"help": "Port for vLLM server B."})
#     vllm_host_c: str = field(default="127.0.0.1", metadata={"help": "Host IP for vLLM server C."}) # Default to localhost
#     vllm_port_c: int = field(default=8002, metadata={"help": "Port for vLLM server C."})
#     vllm_server_timeout: float = field(default=60.0, metadata={"help": "Timeout in seconds to wait for vLLM servers."})
#     # Generation parameters
#     vllm_max_new_tokens_a: int = field(default=64, metadata={"help": "Max new tokens for vLLM generation."})
#     vllm_temperature_a: float = field(default=0.7, metadata={"help": "Temperature for vLLM generation."})
#     vllm_max_new_tokens_b: int = field(default=64, metadata={"help": "Max new tokens for vLLM generation."})
#     vllm_temperature_b: float = field(default=0.7, metadata={"help": "Temperature for vLLM generation."})
#     vllm_max_new_tokens_c: int = field(default=64, metadata={"help": "Max new tokens for vLLM generation."})
#     vllm_temperature_c: float = field(default=0.7, metadata={"help": "Temperature for vLLM generation."})
#     vllm_top_p_a: float = field(default=1.0, metadata={"help": "Top-p for vLLM generation."})
#     vllm_top_k_a: int = field(default=-1, metadata={"help": "Top-k for vLLM generation (-1 disables)."})
#     vllm_top_p_b: float = field(default=1.0, metadata={"help": "Top-p for vLLM generation."})
#     vllm_top_k_b: int = field(default=-1, metadata={"help": "Top-k for vLLM generation (-1 disables)."})
#     vllm_top_p_c: float = field(default=1.0, metadata={"help": "Top-p for vLLM generation."})
#     vllm_top_k_c: int = field(default=-1, metadata={"help": "Top-k for vLLM generation (-1 disables)."})
#     # Add other sampling params as needed

# @dataclass
# class LoggingSavingConfig:
#     """Configurations for logging, saving, and reproducibility."""
#     output_dir: str = field(default="", metadata={"help": "Output directory for checkpoints and logs."})
#     logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."}) # Lower default
#     save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."}) # Lower default
#     eval_steps: int = field(default=100, metadata={"help": "Run evaluation every X updates steps."}) # Lower default
#     seed: int = field(default=42, metadata={"help": "Random seed."})
#     resume_from_checkpoint: str | None = field(default=None, metadata={"help": "Path to checkpoint to resume from."})
#     # Add save_total_limit, etc.

# # --- Main Experiment Arguments Container ---
# @dataclass
# class ExperimentArgs:
#     """Container for all experiment configurations."""
#     model: ModelConfig = field(default_factory=ModelConfig)
#     data: DataConfig = field(default_factory=DataConfig)
#     training: TrainingConfig = field(default_factory=TrainingConfig)
#     distributed: DistributedConfig = field(default_factory=DistributedConfig)
#     inference: InferenceConfig = field(default_factory=InferenceConfig)
#     logging_saving: LoggingSavingConfig = field(default_factory=LoggingSavingConfig)


# def get_args():
#     parser = HfArgumentParser((ExperimentArgs,))
#     experiment_args = parser.parse_args_into_dataclasses()[0]
#     return experiment_args


# --- Define the SINGLE, FLAT arguments dataclass ---
@dataclass
class FlatExperimentArgs:
    """Container for ALL experiment configurations (flattened)."""

    # --- Fields from ModelConfig ---
    honest_prover_name_or_path: str = field(
        default="", metadata={"help": "Path/ID for Honest Prover."}
    )
    sneaky_prover_name_or_path: str = field(
        default="", metadata={"help": "Path/ID for Sneaky Prover."}
    )
    verifier_name_or_path: str = field(
        default="", metadata={"help": "Path/ID for Verifier."}
    )
    tokenizer_name_or_path: str | None = field(
        default=None, metadata={"help": "Path/ID for Tokenizer."}
    )
    apply_liger_kernel: bool = field(
        default=True, metadata={"help": "Apply Liger kernel."}
    )

    # --- Fields from DataConfig ---
    dataset_name: str = field(default="", metadata={"help": "Dataset name."})
    train_num_samples: int | None = field(
        default=None, metadata={"help": "Number of training samples."}
    )

    # --- Fields from TrainingConfig ---
    learning_rate_honest_prover: float = field(
        default=5e-5, metadata={"help": "LR for Honest Prover."}
    )
    learning_rate_sneaky_prover: float = field(
        default=5e-5, metadata={"help": "LR for Sneaky Prover."}
    )
    weight_decay_honest_prover: float = field(
        default=0.0, metadata={"help": "Weight decay Honest Prover."}
    )
    weight_decay_sneaky_prover: float = field(
        default=0.0, metadata={"help": "Weight decay Sneaky Prover."}
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Train batch size per device."}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Eval batch size per device."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation."}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Gradient checkpointing."}
    )
    lr_scheduler_type: str = field(default="linear", metadata={"help": "LR scheduler."})
    num_warmup_steps: int = field(default=0, metadata={"help": "Warmup steps."})
    num_train_epochs: int = field(default=1, metadata={"help": "Training epochs."})
    max_train_steps: int | None = field(
        default=None, metadata={"help": "Max train steps."}
    )
    max_grad_norm_honest_prover: float | None = field(
        default=0.1, metadata={"help": "Max grad norm Honest Prover."}
    )
    max_grad_norm_sneaky_prover: float | None = field(
        default=0.1, metadata={"help": "Max grad norm Sneaky Prover."}
    )
    sync_steps: int = field(default=1, metadata={"help": "Sync steps."})
    num_generations: int = field(default=2, metadata={"help": "Num generations."})
    num_iterations: int = field(
        default=1, metadata={"help": "Num iterations (GRPO mu)."}
    )
    beta: float = field(default=0.0, metadata={"help": "Beta for KL-penalty."})
    epsilon_low: float = field(default=0.2, metadata={"help": "Epsilon low."})
    epsilon_high: float = field(default=0.28, metadata={"help": "Epsilon high."})
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Scale rewards (if True, uses vanilla GRPO scaling, if not uses method from Dr. GRPO paper, which claims a length bias when scaling)."
        },
    )

    # --- Fields from DistributedConfig ---
    ds_config_honest_prover: str = field(
        default="", metadata={"help": "DS config path Honest Prover."}
    )
    ds_config_sneaky_prover: str = field(
        default="", metadata={"help": "DS config path Sneaky Prover."}
    )
    mixed_precision: str | None = field(
        default="bf16", metadata={"help": "Mixed precision."}
    )

    # --- Fields from InferenceConfig ---
    vllm_host_honest_prover: str = field(
        default="127.0.0.1", metadata={"help": "vLLM host Honest Prover."}
    )
    vllm_port_honest_prover: int = field(
        default=8000, metadata={"help": "vLLM port Honest Prover."}
    )
    vllm_host_sneaky_prover: str = field(
        default="127.0.0.1", metadata={"help": "vLLM host Sneaky Prover."}
    )
    vllm_port_sneaky_prover: int = field(
        default=8001, metadata={"help": "vLLM port Sneaky Prover."}
    )
    vllm_host_verifier: str = field(
        default="127.0.0.1", metadata={"help": "vLLM host Verifier."}
    )
    vllm_port_verifier: int = field(
        default=8002, metadata={"help": "vLLM port Verifier."}
    )
    vllm_server_timeout: float = field(
        default=60.0, metadata={"help": "vLLM server timeout."}
    )
    vllm_max_new_tokens_honest_prover: int = field(
        default=64, metadata={"help": "Max new tokens Honest Prover."}
    )
    vllm_temperature_honest_prover: float = field(
        default=1.0, metadata={"help": "Temperature Honest Prover."}
    )
    vllm_max_new_tokens_sneaky_prover: int = field(
        default=64, metadata={"help": "Max new tokens Sneaky Prover."}
    )
    vllm_temperature_sneaky_prover: float = field(
        default=1.0, metadata={"help": "Temperature Sneaky Prover."}
    )
    vllm_max_new_tokens_verifier: int = field(
        default=64, metadata={"help": "Max new tokens Verifier."}
    )
    vllm_temperature_verifier: float = field(
        default=0.7, metadata={"help": "Temperature Verifier."}
    )
    vllm_repetition_penalty_honest_prover: float = field(
        default=1.0, metadata={"help": "Repetition penalty Honest Prover."}
    )
    vllm_repetition_penalty_sneaky_prover: float = field(
        default=1.0, metadata={"help": "Repetition penalty Sneaky Prover."}
    )
    vllm_repetition_penalty_verifier: float = field(
        default=1.0, metadata={"help": "Repetition penalty Verifier."}
    )
    vllm_frequency_penalty_honest_prover: float = field(
        default=0.0, metadata={"help": "Frequency penalty Honest Prover."}
    )
    vllm_frequency_penalty_sneaky_prover: float = field(
        default=0.0, metadata={"help": "Frequency penalty Sneaky Prover."}
    )
    vllm_frequency_penalty_verifier: float = field(
        default=0.0, metadata={"help": "Frequency penalty Verifier."}
    )

    vllm_stop_sequences_honest_prover: list[str] | None = field(
        default=None, metadata={"help": "Stop sequences Honest Prover."}
    )
    vllm_stop_sequences_sneaky_prover: list[str] | None = field(
        default=None, metadata={"help": "Stop sequences Sneaky Prover."}
    )
    vllm_stop_sequences_verifier: list[str] | None = field(
        default=None, metadata={"help": "Stop sequences Verifier."}
    )

    vllm_top_p_honest_prover: float = field(
        default=1.0, metadata={"help": "Top-p Honest Prover."}
    )
    vllm_top_k_honest_prover: int = field(
        default=-1, metadata={"help": "Top-k Honest Prover."}
    )
    vllm_min_p_honest_prover: float = field(
        default=0.0, metadata={"help": "Min-p Honest Prover."}
    )
    vllm_top_p_sneaky_prover: float = field(
        default=1.0, metadata={"help": "Top-p Sneaky Prover."}
    )
    vllm_top_k_sneaky_prover: int = field(
        default=-1, metadata={"help": "Top-k Sneaky Prover."}
    )
    vllm_min_p_sneaky_prover: float = field(
        default=0.0, metadata={"help": "Min-p Sneaky Prover."}
    )
    vllm_top_p_verifier: float = field(
        default=1.0, metadata={"help": "Top-p Verifier."}
    )
    vllm_top_k_verifier: int = field(default=-1, metadata={"help": "Top-k Verifier."})
    vllm_min_p_verifier: float = field(
        default=0.0, metadata={"help": "Min-p Verifier."}
    )

    # --- Fields from LoggingSavingConfig ---
    output_dir: str = field(default="", metadata={"help": "Output directory."})
    logging_steps: int = field(default=1, metadata={"help": "Log steps."})
    save_steps: int = field(default=100, metadata={"help": "Save steps."})
    eval_steps: int = field(default=100, metadata={"help": "Eval steps."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "Resume checkpoint path."}
    )

    # --- Fields from InstructionConfig ---
    honest_prover_system_prompt: str = field(
        default="", metadata={"help": "System prompt for Honest Prover."}
    )
    sneaky_prover_system_prompt: str = field(
        default="", metadata={"help": "System prompt for Sneaky Prover."}
    )
    verifier_system_prompt: str = field(
        default="", metadata={"help": "System prompt for Verifier."}
    )

    # --- WandB Logging Config ---
    wandb_project_name: str = field(
        default="pvg", metadata={"help": "WandB project name."}
    )
    wandb_entity: str = field(
        default="jvelja-private", metadata={"help": "WandB entity (username or team)."}
    )
    wandb_run_name: str | None = field(
        default=None, metadata={"help": "Optional WandB run name."}
    )
    wandb_log_freq: int = field(
        default=1,
        metadata={"help": "Base frequency for logging scalars (in global steps)."},
    )
    wandb_hist_freq_multiplier: int = field(
        default=50, metadata={"help": "Log histograms every N * wandb_log_freq steps."}
    )
    wandb_table_freq_multiplier: int = field(
        default=50, metadata={"help": "Log tables every N * wandb_log_freq steps."}
    )
    wandb_table_samples: int = field(
        default=64, metadata={"help": "Number of samples in training table."}
    )
    wandb_eval_table_samples: int = field(
        default=32, metadata={"help": "Number of samples in eval table."}
    )
    wandb_log_system_freq_multiplier: int = field(
        default=50,
        metadata={"help": "Log system metrics every N * wandb_log_freq steps."},
    )


# --- Update get_args function ---
def get_args():
    # Initialize parser with the single flat dataclass type
    parser = HfArgumentParser((FlatExperimentArgs,))
    # parse_args_into_dataclasses returns a tuple, get the first element which is our args instance
    script_args = parser.parse_args_into_dataclasses()[0]
    return script_args  # Returns an instance of FlatExperimentArgs


# Storage classes for _generate_and_score_completions
class Container:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        raw_prompts: list[str],
        system_prompts: dict[
            Literal["honest_prover", "sneaky_prover", "verifier"], str
        ],
        devices: dict[str, torch.device],
    ) -> None:
        self.tokenizer = tokenizer
        self.raw_prompts = raw_prompts  # List of strings (problem queries), not preprocessed for generation (i.e., not formatted with system prompts, etc.) --> PROBLEM HERE: Only questions, not solutions! Handling input preparation for sneaky & verifier is a pain
        self.system_prompts = system_prompts  # Dict of system prompts for all models
        self.devices = devices  # Dict of devices for all models

        self.container = {  # Container for all the relevant stuff
            "honest_prover": {
                "prompt_texts": [],
                "prompt_ids": [],
                "prompt_mask": [],
                "completion_texts": [],
                "completion_ids": [],
                "completion_mask": [],
                "prompt_completion_ids": [],
                "prompt_completion_mask": [],
                "logits_to_keep": -1,
                "is_eos": [],
            },
            "sneaky_prover": {
                "prompt_texts": [],
                "prompt_ids": [],
                "prompt_mask": [],
                "completion_texts": [],
                "completion_ids": [],
                "completion_mask": [],
                "prompt_completion_ids": [],
                "prompt_completion_mask": [],
                "logits_to_keep": -1,
                "is_eos": [],
            },
            "verifier": {  # TODO: Check if we need to store verifiers ids and masks here (we shouldnt?)
                "prompt_texts": [],
                "prompt_ids": [],
                "prompt_mask": [],
                "completion_texts": [],
                "completion_ids": [],
                "completion_mask": [],
                "prompt_completion_ids": [],
                "prompt_completion_mask": [],
                "logits_to_keep": -1,
                "is_eos": [],
            },
        }

        self.prompt_bank = (
            {  # Bank of prompts for all models (callable by model_key and format_type)
                "honest_prover": {
                    "base": BASE_HONEST,
                    "instruct": INSTRUCT_HONEST,
                    "args": ["problem"],
                },
                "sneaky_prover": {
                    "base": BASE_SNEAKY,
                    "instruct": INSTRUCT_SNEAKY,
                    "args": ["problem", "honest_solution"],
                },
                "verifier": {
                    "base": BASE_VERIFIER,
                    "instruct": INSTRUCT_VERIFIER,
                    "args": ["problem", "solution"],
                },
            }
        )
        self.honest_solutions: list[str] | None = (
            None  # Solutions from the honest prover
        )
        self.sneaky_solutions: list[str] | None = (
            None  # Solutions from the sneaky prover
        )
        self.verifier_rewards: list[float] | None = None  # Rewards from the verifier

    # TODO: Check completions_ids type!!!!!!
    def load_completions(
        self,
        model_key: Literal["honest_prover", "sneaky_prover", "verifier"],
        completions_str: list[str],
        completions_ids,
        rewards: list[float] | None = None,
    ):
        """Load completions from a model."""
        if model_key == "honest_prover":
            self.container[model_key]["completion_texts"] = completions_str
            self.container[model_key]["completion_ids"] = completions_ids
            self.honest_solutions = [
                self.extract_solution(completion) for completion in completions_str
            ]
        elif model_key == "sneaky_prover":
            self.container[model_key]["completion_texts"] = completions_str
            self.container[model_key]["completion_ids"] = completions_ids
            self.sneaky_solutions = [
                self.extract_solution(completion) for completion in completions_str
            ]
        elif model_key == "verifier":
            self.container[model_key]["completion_texts"] = completions_str
            self.container[model_key]["completion_ids"] = completions_ids
            self.verifier_rewards = rewards

    def extract_solution(self, completion_text: str) -> str:
        """
        Extracts the code block enclosed in ```<language> ... ``` from a completion text.
        It finds the first occurrence of such a block, regardless of the specified language.
        Provides fallbacks if the delimiters are not found as expected.
        """
        # Regex to find ``` followed by any language identifier (optional), newline, content, and closing ```
        # - ```(?:[a-zA-Z0-9]+)? : Matches ``` optionally followed by a language identifier (alphanumeric)
        # - \s*\n                 : Matches optional whitespace and a newline
        # - (.*?)                : Captures the content non-greedily (the actual code/solution)
        # - \n?```               : Matches an optional newline and the closing ```        # re.DOTALL makes '.' match newlines as well
        # 1. Try ```<language> ... ```
        match = re.search(
            r"```(?:[a-zA-Z0-9]*)?\s*\n(.*?)\n?```", completion_text, re.DOTALL
        )
        if match:
            return match.group(1).strip()

        # 2. Try <solution> ... </solution>
        match = re.search(
            r"<solution>\s*(.*?)\s*</solution>", completion_text, re.DOTALL
        )
        if match:
            # Check if the content inside <solution> itself contains ``` block
            inner_content = match.group(1).strip()
            inner_match = re.search(
                r"```(?:[a-zA-Z0-9]*)?\s*\n(.*?)\n?```", inner_content, re.DOTALL
            )
            if inner_match:
                return inner_match.group(1).strip()
            # If no inner ``` block, return the content of <solution>
            return inner_content

        # 3. Try <injected_solution> ... </injected_solution>
        match = re.search(
            r"<injected_solution>\s*(.*?)\s*</injected_solution>",
            completion_text,
            re.DOTALL,
        )
        if match:
            # Check if the content inside <injected_solution> itself contains ``` block
            inner_content = match.group(1).strip()
            inner_match = re.search(
                r"```(?:[a-zA-Z0-9]*)?\s*\n(.*?)\n?```", inner_content, re.DOTALL
            )
            if inner_match:
                return inner_match.group(1).strip()
            # If no inner ``` block, return the content of <injected_solution>
            return inner_content

        # 4. Fallback
        logger.warning(
            f"Could not find ```...```, <solution>...</solution>, or <injected_solution>...</injected_solution> block in completion text. Returning raw text. Text was:\n---\n{completion_text}\n---"
        )
        # Fallback: Return the original text (TODO: Find a better fallback, maybe a penalty for the model?)
        return completion_text

    def make_query_prompt(
        self,
        model_key: Literal["honest_prover", "sneaky_prover", "verifier"],
        format_type: Literal["base", "instruct"],
    ) -> str:
        """Make an instruction prompt from a question. Does not .format() and fill in the prompt."""
        if format_type == "base":
            return self.prompt_bank[model_key]["base"]
        elif format_type == "instruct":
            chat = [
                {"role": "system", "content": self.system_prompts[model_key]},
                {"role": "user", "content": self.prompt_bank[model_key]["instruct"]},
            ]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

    def prepare_inputs(
        self,
        model_key: Literal["honest_prover", "sneaky_prover", "verifier"],
        format_type: Literal["base", "instruct"],
    ) -> None:
        """Prepare inputs for a model. This implies the str prompt filled in, the tokenized prompt, and the mask."""
        if model_key == "sneaky_prover" and self.honest_solutions is None:
            raise ValueError(
                "Honest solutions are required for sneaky prover prompt preparation."
            )

        if model_key == "verifier" and (
            self.sneaky_solutions is None or self.honest_solutions is None
        ):
            raise ValueError(
                "Sneaky and honest solutions are required for verifier prompt preparation."
            )

        # TODO: This is the wrong way to do this. .format() should, conditionally on model_key, fill in either:
        # - honest_prover: problem --> raw_prompts[i] --> .format(problem=raw_prompts[i]) for all problems in raw_prompts
        # - sneaky_prover: problem, honest_solution --> raw_prompts[i], honest_solutions[i] --> .format(problem=raw_prompts[i], honest_solution=honest_solutions[i]) for all i in range(len(raw_prompts)) # Assumes raw_prompts and honest_solutions are of same length
        # - verifier: problem, solution --> raw_prompts[i], solutions[i] (where solutions is joint sneaky & honest solutions -- concat) --> .format(problem=raw_prompts[i], solution=solutions[i]) for all i in range(len(raw_prompts))

        # if format_type == "base":
        #     formatted_prompts = [self.make_query_prompt(model_key, format_type).format(**{arg: self.raw_prompts[i] for arg in self.prompt_bank[model_key]["args"]}) for i in range(len(self.raw_prompts))]
        # elif format_type == "instruct":
        #     formatted_prompts = [self.make_query_prompt(model_key, format_type).format(**{arg: self.raw_prompts[i] for arg in self.prompt_bank[model_key]["args"]}) for i in range(len(self.raw_prompts))]

        if model_key == "honest_prover":
            formatted_prompts = [
                self.make_query_prompt(model_key, format_type).format(
                    problem=self.raw_prompts[i]
                )
                for i in range(len(self.raw_prompts))
            ]
        elif model_key == "sneaky_prover":
            formatted_prompts = [
                self.make_query_prompt(model_key, format_type).format(
                    problem=self.raw_prompts[i],
                    honest_solution=self.honest_solutions[i],
                )
                for i in range(len(self.raw_prompts))
            ]
        elif model_key == "verifier":
            formatted_prompts = [
                self.make_query_prompt(model_key, format_type).format(
                    problem=self.raw_prompts[i], solution=self.honest_solutions[i]
                )
                for i in range(len(self.raw_prompts))
            ]
            formatted_prompts += [
                self.make_query_prompt(model_key, format_type).format(
                    problem=self.raw_prompts[i], solution=self.sneaky_solutions[i]
                )
                for i in range(len(self.raw_prompts))
            ]
            # Shuffle?
            # random.shuffle(formatted_prompts)

        tokenized_prompts = (
            self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            ).to(self.devices[model_key])
            if model_key != "verifier"
            else None
        )

        self.container[model_key]["prompt_texts"] = formatted_prompts
        self.container[model_key]["prompt_ids"] = (
            tokenized_prompts["input_ids"] if model_key != "verifier" else None
        )
        self.container[model_key]["prompt_mask"] = (
            tokenized_prompts["attention_mask"] if model_key != "verifier" else None
        )

    def pad_and_concatenate(
        self, model_key: Literal["honest_prover", "sneaky_prover", "verifier"]
    ) -> None:
        """Pad and concatenate the completion ids for a model."""
        # See below
        # completion_ids_a = [torch.tensor(ids, device=device_a) for ids in completion_ids_a]
        # completion_ids_a = pad(completion_ids_a, padding_value=self.tokenizer.pad_token_id)
        # prompt_completion_ids_a = torch.cat([prompt_ids_a, completion_ids_a], dim=1)

        # completion_ids_b = [torch.tensor(ids, device=device_a) for ids in completion_ids_b]
        # completion_ids_b = pad(completion_ids_b, padding_value=self.tokenizer.pad_token_id)
        # prompt_completion_ids_b = torch.cat([prompt_ids_b, completion_ids_b], dim=1)

        # completion_ids (for every model_key) is a list of list of ints (each inner list is a completion), but it must be tensorized and padded
        # Why list[list[int]]?
        # vLLM client returns a json.response object, which is handled easily via the above data structure. We need to do an extra step to tensorize first and then pad : (list[torch.Tensor], padding_value: int = 0, padding_side: str = "right")
        device = self.devices[model_key] if model_key != "verifier" else "cuda:0"
        self.container[model_key]["completion_ids"] = [
            torch.tensor(ids, device=device)
            for ids in self.container[model_key]["completion_ids"]
        ]  # Needed to pass to pad ()
        self.container[model_key]["completion_ids"] = (
            pad(
                self.container[model_key]["completion_ids"],
                padding_value=self.tokenizer.pad_token_id,
            )
            if model_key != "verifier"
            else None
        )
        self.container[model_key]["prompt_completion_ids"] = (
            torch.cat(
                [
                    self.container[model_key]["prompt_ids"],
                    self.container[model_key]["completion_ids"],
                ],
                dim=1,
            )
            if model_key != "verifier"
            else None
        )

    def mask_completion(
        self, model_key: Literal["honest_prover", "sneaky_prover", "verifier"]
    ) -> None:
        """Mask the completion ids for a model."""
        # See below
        # is_eos_a = completion_ids_a == self.tokenizer.eos_token_id
        # eos_idx_a = torch.full((is_eos_a.size(0),), is_eos_a.size(1), dtype=torch.long, device=device_a)
        # eos_idx_a[is_eos_a.any(dim=1)] = is_eos_a.int().argmax(dim=1)[is_eos_a.any(dim=1)]
        # sequence_indices_a = torch.arange(is_eos_a.size(1), device=device_a).expand(is_eos_a.size(0), -1)
        # completion_mask_a = (sequence_indices_a <= eos_idx_a.unsqueeze(1)).int()
        device = self.devices[model_key] if model_key != "verifier" else "cuda:0"
        is_eos = (
            self.container[model_key]["completion_ids"] == self.tokenizer.eos_token_id
        )
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        self.container[model_key]["completion_mask"] = completion_mask
        self.container[model_key]["prompt_completion_mask"] = torch.cat(
            [self.container[model_key]["prompt_mask"], completion_mask], dim=1
        )
        self.container[model_key]["logits_to_keep"] = self.container[model_key][
            "completion_ids"
        ].size(1)
        self.container[model_key]["is_eos"] = is_eos
