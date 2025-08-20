# Refactor Plan: Colocated vLLM + FSDP2 for PVG

This document captures how a round unfolds today, the critical bottlenecks slowing the system, and a concrete refactor/tuning plan to maximize MFU and throughput on 4×/8× H100 with 3B/7B/14B models.

## 1) Current Round Flow (as implemented)
- Phase 0 – Data mix build (verifier_dataset_round_i):
  - `TrainingPhaseOrchestrator` instantiates `DataGenerator` which loads a split, constructs prompts, generates sneaky completions via vLLM, parses/tests them, optionally pushes to Hub.
- Phase 1 – Verifier training:
  - `VerifierPhaseStrategy` builds datasets/dataloaders; `VerifierRegressor/Classifier` trains a scoring head (BT objective for regressor). Checkpoints saved.
- Phase 2 – Prover training (GRPO):
  - `ProverTrainer` generates completions (vLLM), computes rewards via trained verifier, computes GRPO loss, applies optimizer step(s). After steps, weights are pushed to vLLM (today: param-by-param sync).
- Next round:
  - Mix prior rounds + new sneaky completions to produce `verifier_dataset_round_{i+1}` and repeat.

Observed cost centers:
- vLLM is driven from rank‑0 (HTTP/RPC), results broadcast as Python objects; frequent global barriers.
- Per‑param weight streaming to vLLM after optimizer steps.
- Dataloaders lack workers/pinning/prefetch; excessive per‑batch logging and cache thrash.
- Hub pushes/checks in hot path.

## 2) Target Design (single node colocation)
- Keep sequential phases, but time‑slice vLLM and training on the same 8 GPUs:
  - Gen burst (vLLM) → sleep(level=2) → Train burst (FSDP2) → repeat.
- Replace per‑param streaming with batched updates at rollout boundaries (either runtime reload or re-init), leveraging latest vLLM startup/reload improvements.
- Replace Python object broadcasts with tensor collectives only.
- Remove network I/O (Hub) from hot loops.

## 3) vLLM Engine Tuning (H100, BF16)
- Core knobs:
  - `tensor_parallel_size`: 2–4 for 3B/7B, 4–8 for 14B (keep TP intra‑node).
  - `gpu_memory_utilization`: 0.60–0.80 in colocation; raise if gen is the bottleneck (ensure sleep(level=2) before training).
  - `max_model_len`: cap at 4k–8k (longer prompts rely on chunked prefill).
  - `max_seq_len_to_capture`: ~2048 to keep typical prompts on CUDA‑graph fast path.
  - `max_num_batched_tokens`: push as high as KV allows; see budget below.
  - Enable: prefix caching, chunked prefill.
- Token‑budgeting and “sequence packing” for vLLM:
  - vLLM batches by a token budget per micro‑batch. Set a high `max_num_batched_tokens` and bucket prompts by length (no literal concatenation) so each micro‑batch fills the budget densely.
  - Suggested length buckets from your stats (prompt only):
    - B1: 600–1.1k (≈ p50)
    - B2: 1.1–1.5k (≈ p50–p90)
    - B3: 1.5–1.9k (≈ p90–p95)
    - B4: 1.9–2.5k (≈ p95–p99)
    - B5: >2.5k (tail; smaller batch counts)
- KV cache budget approximation (per token, BF16):
  - KV bytes/token ≈ 2 (K,V) × layers × heads × head_dim × 2 bytes.
  - Rough guide: 3B ≈ 0.4–0.5 MB/token; 7B ≈ ~0.5 MB/token; 14B ≈ ~1.0–1.3 MB/token.
  - On H100 80GB with TP=T and `gpu_memory_utilization=μ`, usable KV per GPU ≈ 80GB × μ ÷ T (minus model/aux). Choose `max_num_batched_tokens` so that (tokens_in_flight × bytes/token) fits comfortably.
- Starter `max_num_batched_tokens` (per engine, adjust upward until plateau):
  - 3B: 16k–32k tokens; 7B: 12k–24k; 14B: 8k–16k.
- Throughput cadence (update rule):
  - Generate rollouts for K optimizer steps, then reload weights. Start with K=4–8 steps or ~60–120s between reloads. Increase K if reload shows up in profiles; decrease if rollouts feel stale.

## 4) Training (FSDP2 + FlashAttn‑3 + compile)
- Wrap with `fully_shard(use_orig_params=True, forward_prefetch=True)` and auto‑wrap per block.
- BF16 autocast; gradient checkpointing on attn+MLP; matmul TF32 enabled.
- `torch.compile(mode="reduce-overhead")` after FSDP wrap (keep shapes stable via length bucketing to help capture).
- Optimizer: `AdamW(fused=True)`. Sharded saves via DCP; one full export at phase end.
- Dataloaders: `num_workers=8–16`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`; non‑blocking `.to(device)`; length‑bucketed batches.

## 5) Prompt Stats → Concrete Adjustments
- Your prompt p50 ≈ 1,043; p90 ≈ 1,467; p95 ≈ 1,635; tail > 2k.
- Set `max_seq_len_to_capture≈2048` so the median/p90/p95 prefills ride the CUDA‑graph path; cap `max_model_len` at 4k–8k; rely on chunked prefill for the long tail.
- Use the 5 buckets above to keep micro‑batches dense. For each bucket, size `max_num_batched_tokens` so that micro‑batch has 8–32 sequences (depends on `TP`, model size, and output length target). Measure tokens/s and adjust.
- If typical generation length (L_gen) is 256–512, the per‑request token cost ≈ L_prompt + L_gen; use that sum when budgeting tokens per micro‑batch.

## 6) Batch Size Planning (3B/7B/14B; 4×/8× H100)
- Training micro‑batch per GPU (guideline, BF16, 2k sequence, checkpointing):
  - 3B: 8–16 seq/GPU; 7B: 4–8 seq/GPU; 14B: 2–4 seq/GPU.
- Use grad‑accum M so that global effective batch = (seq/GPU × GPUs × M) matches GRPO constraints (`effective_train_batch_size % num_generations == 0`).
- vLLM gen budget:
  - Start with the `max_num_batched_tokens` ranges above; for buckets near p50/p90, aim to keep micro‑batch count at least 8–16 sequences to amortize overheads; reduce for the tail bucket to stay within KV.

## 7) Update Cadence Options (colocation)
- Time‑sliced cycle (recommended):
  - Gen burst (fill buffer for K steps) → sleep(level=2) → Train K steps → runtime reload weights → repeat.
- K selection:
  - Start at K=4–8 steps (or ~60–120s), then tune: K ≈ clamp(round(T_gen/T_train), 4, 16).
- With latest vLLM runtime reloading and faster startup, keep updates at **burst boundaries**; avoid per‑step updates even if RPC exists.

## 8) Critical Code Changes (surgical)
- Remove param‑by‑param vLLM syncing; add a rank‑0 colocated engine that supports: `reload(model_path)`, `generate(prompts, sampling)`, `sleep(level=2)`.
- Replace Python object broadcasts with padded tensor collectives (ids/lengths on GPU).
- Move Hub pushes/existence checks out of the hot path (phase/round boundaries only).
- Tidy loops: collapse inner `wait_for_everyone()`, drop `empty_cache()`/`gc.collect()` from step loop, throttle logs.

## 9) Open Items / Inputs Needed
- Typical generation length target(s) for prover/verifier.
- Exact 3B/7B/14B architectures (layers/heads/head_dim) to refine KV/token budgets.
- Desired global batch and GRPO `num_generations` to lock grad‑accum.

## 10) Next Steps
- Implement colocated engine (rank‑0) and refactor prover loop to gen/train bursts with runtime reload.
- Add length‑bucketed batching on both gen and train paths; set initial engine presets per model size/node config; benchmark tokens/s and MFU.
- Migrate verifier training to FSDP2 + compile; keep verifier inference as compiled HF forward during prover phase (avoid second vLLM).
