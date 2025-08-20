from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.parallel.fsdp2_utils import wrap_fsdp, autocast_bf16


class ParallelBackend:
    """Abstract parallel backend with a small, stable interface."""

    def wrap_model(self, model: torch.nn.Module, key: str) -> torch.nn.Module:
        return model

    def prepare_dataloader(self, dataloader, key: str):
        return dataloader

    def prepare_optimizer(self, optimizer, key: str):
        return optimizer

    def prepare_scheduler(self, scheduler, key: str):
        return scheduler

    def backward(self, loss: torch.Tensor, key: str) -> None:
        loss.backward()

    def global_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        out = tensor.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def move_to_device(self, t: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        return t.to(next(model.parameters()).device, non_blocking=True)

    @contextmanager
    def autocast(self) -> Iterator[None]:
        with autocast_bf16(True):
            yield


class FSDP2Backend(ParallelBackend):
    """FSDP2 backend using native torch.distributed + BF16 autocast."""

    def wrap_model(self, model: torch.nn.Module, key: str) -> torch.nn.Module:
        # gradient checkpointing should be enabled by caller if desired
        m = wrap_fsdp(model)
        # Try compile for Hopper path if available
        try:
            m = torch.compile(m, mode="reduce-overhead", dynamic=True)  # type: ignore[attr-defined]
        except Exception:
            print("[WARNING] Failed to compile model with torch.compile. Using native mode.")
        return m


class AccelerateBackend(ParallelBackend):
    """Adapter over AcceleratorManager to provide the same interface without if/else at call sites."""

    def __init__(self, accel: AcceleratorManager) -> None:
        self.accel = accel

    def wrap_model(self, model: torch.nn.Module, key: str) -> torch.nn.Module:
        self.accel.accelerators[key].state.select_deepspeed_plugin(key)
        return self.accel.accelerators[key].prepare(model)

    def prepare_dataloader(self, dataloader, key: str):
        return self.accel.prepare_dataloader(dataloader, key)

    def prepare_optimizer(self, optimizer, key: str):
        self.accel.accelerators[key].state.select_deepspeed_plugin(key)
        return self.accel.accelerators[key].prepare(optimizer)

    def prepare_scheduler(self, scheduler, key: str):
        return self.accel.prepare_scheduler(key, scheduler)

    def backward(self, loss: torch.Tensor, key: str) -> None:
        self.accel.backward(loss, key)

    def global_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.accel.gather_for_metrics(tensor, key="verifier").sum()

    def move_to_device(self, t: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        # Accelerator moves are handled inside prepare; identity here
        return t

    @contextmanager
    def autocast(self) -> Iterator[None]:
        # Rely on accelerator’s mixed precision; no extra context
        yield


def make_backend(args, accelerator_manager: AcceleratorManager | None) -> ParallelBackend:
    backend_name = getattr(args.parallel, "parallel_backend", "accelerate")
    if backend_name == "fsdp2":
        return FSDP2Backend()
    else:
        if accelerator_manager is None:
            raise RuntimeError("Accelerate backend selected but AcceleratorManager is None")
        return AccelerateBackend(accelerator_manager)

