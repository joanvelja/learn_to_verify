import os
from contextlib import contextmanager
from typing import Callable, Iterable

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint import save as dcp_save, load as dcp_load, state_dict as dcp_state_dict


def init_dist(backend: str = "nccl", timeout_seconds: int = 7200) -> None:
    """Initialize torch.distributed with sensible defaults for single-node H100."""
    if dist.is_initialized():
        return
    os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
    os.environ.setdefault("NCCL_MIN_NRINGS", "8")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    dist.init_process_group(backend=backend, timeout=torch.distributed.timedelta(seconds=timeout_seconds))


def set_hopper_defaults() -> None:
    """Enable BF16 and TF32-friendly toggles and SDPA/FlashAttention where available."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # SDPA/FlashAttention flags are model-config driven in HF; leave global as-is.


def wrap_fsdp(model: torch.nn.Module, auto_wrap_policy: Callable | None = None) -> torch.nn.Module:
    """Wrap a module with FSDP2 fully_shard using recommended flags."""
    return fully_shard(
        model,
        use_orig_params=True,
        forward_prefetch=True,
        # let policy shard per transformer block if provided
        auto_wrap_policy=auto_wrap_policy,
    )


def dcp_save_sharded(model: torch.nn.Module, optim: torch.optim.Optimizer | None, out_dir: str, tag: str) -> None:
    """Save FSDP2 sharded state via torch.distributed.checkpoint."""
    state = {"model": model}
    if optim is not None:
        state["optim"] = optim
    dcp_save(dcp_state_dict(state), checkpoint_id=os.path.join(out_dir, tag))


def dcp_load_sharded(model: torch.nn.Module, optim: torch.optim.Optimizer | None, path: str) -> None:
    """Load FSDP2 sharded state via torch.distributed.checkpoint."""
    state = {"model": model}
    if optim is not None:
        state["optim"] = optim
    dcp_load(dcp_state_dict(state), checkpoint_id=path)


@contextmanager
def autocast_bf16(enabled: bool = True):
    with torch.cuda.amp.autocast(enabled=enabled, dtype=torch.bfloat16):
        yield


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    out = t.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def all_gather_variable_batch(t_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """All-gather a list of tensors with potentially different first-dim sizes.

    Pads to max length across ranks, gathers, then unpads per-rank.
    """
    if not dist.is_initialized():
        return t_list
    local_n = torch.tensor([t_list[0].shape[0]], device=t_list[0].device, dtype=torch.int64)
    all_n = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
    dist.all_gather(all_n, local_n)
    max_n = int(torch.stack(all_n).max().item())

    def _pad(t: torch.Tensor, n: int) -> torch.Tensor:
        if t.shape[0] == n:
            return t
        pad = (0, 0) * (t.dim() - 1) + (0, n - t.shape[0])
        return torch.nn.functional.pad(t, pad)

    padded = [_pad(x, max_n) for x in t_list]
    stacked = torch.stack(padded, dim=0)  # [B, ...]
    gather_buf = [torch.zeros_like(stacked) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_buf, stacked)
    # Unpad per rank
    result = []
    for rank, buf in enumerate(gather_buf):
        n = int(all_n[rank].item())
        result.append(buf[:n])
    return result

