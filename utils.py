import torch
from copy import deepcopy
import deepspeed
from dataclasses import dataclass, field
from transformers.hf_argument_parser import HfArgumentParser


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


@dataclass
class ScriptArguments:
    """
    Arguments pertaining to the script setup, model configurations, and paths.
    """

    honest_prover_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier for Honest Prover"
        }
    )
    sneaky_prover_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier for Model B"}
    )
    verifier_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier for Verifier"}
    )
    ds_config_honest_prover: str = field(
        metadata={"help": "Path to the DeepSpeed config file for Honest Prover."}
    )
    ds_config_sneaky_prover: str = field(
        metadata={"help": "Path to the DeepSpeed config file for Sneaky Prover."}
    )
    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        }
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to tokenizer if different from model path. If None, uses honest_prover_name_or_path."
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    validation_file: str | None = field(
        default=None, metadata={"help": "Path to the validation data file."}
    )
    dataset_name: str | None = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
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
    # Add specific learning rates if they differ, otherwise one might suffice
    learning_rate_a: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Model A's AdamW optimizer."},
    )
    learning_rate_b: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Model B's AdamW optimizer."},
    )
    # ... add other relevant training args (weight decay, warmup, etc.)
    num_warmup_steps: int = field(
        default=0, metadata={"help": "Number of steps for the linear warmup."}
    )
    weight_decay_a: float = field(
        default=0.0, metadata={"help": "Weight decay for Model A optimizer."}
    )
    weight_decay_b: float = field(
        default=0.0, metadata={"help": "Weight decay for Model B optimizer."}
    )
    max_grad_norm_a: float | None = field(
        default=None, metadata={"help": "Max gradient norm for Model A."}
    )
    max_grad_norm_b: float | None = field(
        default=None, metadata={"help": "Max gradient norm for Model B."}
    )
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "Path to checkpoint to resume training from."}
    )
    mixed_precision: str | None = field(
        default=None,
        metadata={
            "help": "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU."
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of worker processes to use for data loading."},
    )
    lr_scheduler_type: str = field(
        default="linear", metadata={"help": "The scheduler type to use."}
    )
    # vllm specific args
    vllm_host_a: str = field(
        default="0.0.0.0", metadata={"help": "Host IP for vLLM server A."}
    )  # For Honest Prover
    vllm_port_a: int = field(
        default=8000, metadata={"help": "Port for vLLM server A."}
    )  # For Honest Prover
    vllm_host_b: str = field(
        default="0.0.0.0", metadata={"help": "Host IP for vLLM server B."}
    )  # For Sneaky Prover
    vllm_port_b: int = field(
        default=8001, metadata={"help": "Port for vLLM server B."}
    )  # For Sneaky Prover
    vllm_host_c: str = field(
        default="0.0.0.0", metadata={"help": "Host IP for vLLM server C."}
    )  # For Verifier
    vllm_port_c: int = field(
        default=8002, metadata={"help": "Port for vLLM server C."}
    )  # For Verifier
    vllm_server_timeout: float = field(
        default=60.0, metadata={"help": "Timeout in seconds to wait for vLLM servers."}
    )
    # vllm sampling args
    vllm_max_new_tokens_a: int = field(
        default=64, metadata={"help": "Max new tokens for vLLM generation."}
    )  # For Honest Prover
    vllm_temperature_a: float = field(
        default=0.7, metadata={"help": "Temperature for vLLM generation."}
    )  # For Honest Prover
    vllm_max_new_tokens_b: int = field(
        default=64, metadata={"help": "Max new tokens for vLLM generation."}
    )  # For Sneaky Prover
    vllm_temperature_b: float = field(
        default=0.7, metadata={"help": "Temperature for vLLM generation."}
    )  # For Sneaky Prover
    vllm_max_new_tokens_c: int = field(
        default=64, metadata={"help": "Max new tokens for vLLM generation."}
    )  # For Verifier
    vllm_temperature_c: float = field(
        default=0.7, metadata={"help": "Temperature for vLLM generation."}
    )  # For Verifier
    # More args to be added in the future
    sync_steps: int = field(
        default=1, metadata={"help": "Number of steps to sync weights."}
    )
    # Missing: RL args! (TODO: Check Will Brown's repo for them)


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    return parser.parse_args()
