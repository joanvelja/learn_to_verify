import torch
from copy import deepcopy
import deepspeed
from dataclasses import dataclass, field
from transformers import HfArgumentParser


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean(
        (tensor - torch.nanmean(tensor, keepdim=True)) ** 2
    )  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


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
    learning_rate_a: float = field(
        default=5e-5, metadata={"help": "LR for Honest Prover."}
    )
    learning_rate_b: float = field(
        default=5e-5, metadata={"help": "LR for Sneaky Prover."}
    )
    weight_decay_a: float = field(default=0.0, metadata={"help": "Weight decay A."})
    weight_decay_b: float = field(default=0.0, metadata={"help": "Weight decay B."})
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
    max_grad_norm_a: float | None = field(
        default=1.0, metadata={"help": "Max grad norm A."}
    )
    max_grad_norm_b: float | None = field(
        default=1.0, metadata={"help": "Max grad norm B."}
    )
    sync_steps: int = field(default=1, metadata={"help": "Sync steps."})
    num_generations: int = field(default=2, metadata={"help": "Num generations."})
    num_iterations: int = field(
        default=1, metadata={"help": "Num iterations (GRPO mu)."}
    )
    beta: float = field(default=0.0, metadata={"help": "Beta for KL-penalty."})

    # --- Fields from DistributedConfig ---
    ds_config_honest_prover: str = field(
        default="", metadata={"help": "DS config path A."}
    )
    ds_config_sneaky_prover: str = field(
        default="", metadata={"help": "DS config path B."}
    )
    mixed_precision: str | None = field(
        default="bf16", metadata={"help": "Mixed precision."}
    )

    # --- Fields from InferenceConfig ---
    vllm_host_a: str = field(default="127.0.0.1", metadata={"help": "vLLM host A."})
    vllm_port_a: int = field(default=8000, metadata={"help": "vLLM port A."})
    vllm_host_b: str = field(default="127.0.0.1", metadata={"help": "vLLM host B."})
    vllm_port_b: int = field(default=8001, metadata={"help": "vLLM port B."})
    vllm_host_c: str = field(default="127.0.0.1", metadata={"help": "vLLM host C."})
    vllm_port_c: int = field(default=8002, metadata={"help": "vLLM port C."})
    vllm_server_timeout: float = field(
        default=60.0, metadata={"help": "vLLM server timeout."}
    )
    vllm_max_new_tokens_a: int = field(
        default=64, metadata={"help": "Max new tokens A."}
    )
    vllm_temperature_a: float = field(default=0.7, metadata={"help": "Temperature A."})
    vllm_max_new_tokens_b: int = field(
        default=64, metadata={"help": "Max new tokens B."}
    )
    vllm_temperature_b: float = field(default=0.7, metadata={"help": "Temperature B."})
    vllm_max_new_tokens_c: int = field(
        default=64, metadata={"help": "Max new tokens C."}
    )
    vllm_temperature_c: float = field(default=0.7, metadata={"help": "Temperature C."})
    vllm_top_p_a: float = field(default=1.0, metadata={"help": "Top-p A."})
    vllm_top_k_a: int = field(default=-1, metadata={"help": "Top-k A."})
    vllm_top_p_b: float = field(default=1.0, metadata={"help": "Top-p B."})
    vllm_top_k_b: int = field(default=-1, metadata={"help": "Top-k B."})
    vllm_top_p_c: float = field(default=1.0, metadata={"help": "Top-p C."})
    vllm_top_k_c: int = field(default=-1, metadata={"help": "Top-k C."})

    # --- Fields from LoggingSavingConfig ---
    output_dir: str = field(default="", metadata={"help": "Output directory."})
    logging_steps: int = field(default=10, metadata={"help": "Log steps."})
    save_steps: int = field(default=100, metadata={"help": "Save steps."})
    eval_steps: int = field(default=100, metadata={"help": "Eval steps."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "Resume checkpoint path."}
    )


# --- Update get_args function ---
def get_args():
    # Initialize parser with the single flat dataclass type
    parser = HfArgumentParser((FlatExperimentArgs,))
    # parse_args_into_dataclasses returns a tuple, get the first element which is our args instance
    script_args = parser.parse_args_into_dataclasses()[0]
    return script_args  # Returns an instance of FlatExperimentArgs
