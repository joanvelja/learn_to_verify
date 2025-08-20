# Parallelism Transition: DeepSpeed ZeRO-3 → PyTorch FSDP2

This document tracks the rationale, differences, and the concrete steps to transition the repository from Accelerate+DeepSpeed ZeRO-3 to a PyTorch-native FSDP2 stack. It also lists expectations (performance/behavior) and the migration checklist.

## Why FSDP2
- PyTorch-native sharding (DTensor/fully_shard) reduces integration surface and simplifies checkpointing.
- Better fit for single-node Hopper (H100) where DP-style sharding + BF16 + FlashAttention-3 hit high MFU.
- Modern features: forward/backward prefetch, use_orig_params, efficient sharded checkpointing (DCP/TorchSnapshot).

## ZeRO-3 vs FSDP2: What changes
- Parameter sharding
  - ZeRO-3: shards params, grads, optimizer states across ranks via DeepSpeed engine; often requires plugin selection per object and explicit gathers.
  - FSDP2: wraps modules with `fully_shard(...)` and uses per-param DTensor layouts; reshards on-the-fly with prefetch hints.
- Checkpointing
  - ZeRO-3: engine-coupled (DeepSpeed state) or full state dicts (large, slow).
  - FSDP2: Distributed Checkpoint (DCP) or TorchSnapshot shards; fast parallel save/reshard on load.
- API surface
  - ZeRO-3: Accelerate abstractions (`accelerator.prepare`, engine/optimizer wrapping).
  - FSDP2: torch.distributed + FSDP wrappers; dataloader stays standard; optimizer is vanilla `AdamW(fused=True)`.
- Failure modes
  - ZeRO-3: hanging on full-state gathers, mismatched plugin selection.
  - FSDP2: requires stable shapes for compile/graphs; caution with multiple forwards before backward.

## Expected impacts
- Training step latency: lower (no engine indirections), especially with `torch.compile(mode="reduce-overhead")` and FA-3.
- Memory: comparable or better; activation checkpointing still recommended.
- Checkpoints: faster, smaller sharded saves; full export only when needed.
- Orchestration: fewer barriers; remove param-by-param vLLM syncs (separate task).

## Migration plan (repo-scoped)
1) Add FSDP2 utilities (this change)
- `pvg/parallel/fsdp2_utils.py`: init_process_group, FSDP wrap helpers, DCP save/load, common env toggles for H100.
- Keep DeepSpeed path intact for fallback while wiring new backend.

2) Configuration switch
- Add `parallel_backend: Literal["accelerate","fsdp2"] = "accelerate"` to `ExperimentArgs`.
- Short term: default remains Accelerate; medium term: flip to `fsdp2` in training scripts.

3) Trainers/Managers surface compatibility
- Introduce an FSDP-aware code path that:
  - wraps models via `wrap_fsdp(model, policy)`;
  - uses native `optimizer.step()/zero_grad()`;
  - replaces `accelerator.backward(loss)` with `loss.backward()`;
  - removes ZeRO plugin selection calls.
- Dataloaders: set `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor` and keep standard `DataLoader`.

4) Checkpointing
- Replace DeepSpeed state saves with DCP sharded saves at training checkpoints; provide full export at phase end.

5) Logging/Expectations
- Document differences in wall-clock per step (compile warmup, graph capture), memory footprint, and resume behavior.

## Done in this patch
- Added FSDP2 utilities module and config flag scaffold.
- No functional change to existing DS path yet (safe, non-breaking).

### Update: Verifier wired to FSDP2 (initial)
- Verifier phase now supports `parallel_backend=fsdp2`:
  - Wraps verifier with FSDP2 (`fully_shard`) in `VerifierPhaseStrategy`.
  - Uses native backward in `VerifierRegressorTrainer` when FSDP2 is selected.
  - Keeps existing Accelerate path unchanged when `parallel_backend=accelerate`.
  - Optimizer/scheduler created on wrapped model; dataloaders remain standard (no accelerator.prepare).
  - Checkpointing remains as before; next step is switching to DCP sharded saves.

## Next steps (implementation)
- Gate `AcceleratorManager` usage behind `parallel_backend` and route to FSDP2 helpers in trainers.
- Migrate Verifier training path first (lowest risk), then Prover.
- Remove DS-specific sync calls and barriers from hot loops.

### Performance-oriented updates (current)
- Verifier FSDP2 path now:
  - Enables gradient checkpointing if configured and attempts `torch.compile(mode="reduce-overhead")`.
  - Moves inputs to device with non-blocking copies; wraps forward in BF16 autocast.
  - Uses `zero_grad(set_to_none=True)` to reduce allocator pressure.
  - Dataloaders use workers, pin_memory, persistent_workers, and prefetch_factor.
  - Pure torch.distributed init for FSDP2 path; no accelerator.prepare for verifier.
  - Length-bucketed epoch plan in `VerifierDataset` to stabilize shapes and improve packing.
  - Sharded checkpoints via Torch Distributed Checkpoint (DCP) saved at epoch end; optional resume support.

### Data pipeline cleanup
- Removed HF Hub pushes in `DataManager.load_datasets()`; run entirely local for high throughput.
- Kept AppsDataset wrappers through splits instead of raw HF datasets to maintain metadata and lengths.
- AppsDataset now precomputes `lengths` and refreshes them on `shuffle()`/`select()` for length-aware sampling.
- `RepeatRandomSampler` accepts optional `lengths` and performs sortish length bucketing per batch for better packing.

## Operational guidance (H100 80GB)
- Set: `torch.backends.cuda.matmul.allow_tf32=True`, BF16 autocast, FlashAttention via SDPA.
- FSDP wrap: `fully_shard(..., use_orig_params=True, forward_prefetch=True)`, activation checkpointing on attn+MLP.
- Compile: `torch.compile(model, mode="reduce-overhead")` after FSDP wrap; stabilize sequence shapes via bucketing.
- NCCL: enable NVLS (`NCCL_NVLS_ENABLE=1`), keep `NCCL_MIN_NRINGS=8`, `TORCH_NCCL_BLOCKING_WAIT=1`.
