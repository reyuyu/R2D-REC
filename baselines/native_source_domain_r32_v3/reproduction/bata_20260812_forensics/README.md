# BATA 2026-08-12 forensic replay harness

This directory contains diagnostic-only tooling used to investigate whether
the historical BATA SFT trajectory can be resumed reproducibly from the real
`checkpoint-553`. It is not a new training recipe and must not be used as a
replacement for the recovered 06:33 source.

The harness records hashes and scalar diagnostics only. It never records raw
prompts, dataset rows, token values, model weights, optimizer states, or RNG
state bytes.

## Components

- `replay_forensics.py`: frozen batch-contract checks, RNG, DDP bucket
  pre/post-allreduce gradients, pre/post-clip LoRA gradients, LoRA, effective
  B@A, optimizer, scheduler, loss, and environment fingerprints.
- `frozen_step554_batch_contract.json`: public-safe per-rank ordered input
  fingerprints established by the exact-input DET-A/DET-B runs. Gradient
  localization verifies runtime field shapes/order against this frozen
  contract instead of copying eight complete CUDA tensors to CPU after every
  microbatch.
- `prepare_replays.py`: verifies the SHA-pinned historical checkpoint and
  creates independent replay directories.
- `prepare_kernel_matrix.py`: configures already-prepared, evidence-empty
  replay directories for the diagnostic-only FA2/Liger 2x2 matrix.
- `mount_historical_packed_cache.py`: exposes the historical Arrow shards as a
  read-only `load_from_disk` dataset without retokenizing or repacking.
- `trainer_integration.patch`: the minimal patch applied only to a copied
  recovered-source runtime.
- `launch_replay.sh`: fail-closed 4-GPU launcher. `--deterministic` sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts and requests strict
  `torch.use_deterministic_algorithms(True, warn_only=False)` before CUDA is
  initialized.
- `case_c_step554.json`: public-safe raw fingerprint/scalar evidence for the
  first observed A/B divergence.
- `summarize_replay_pair.py`: compares two four-rank evidence directories,
  fails closed on any checkpoint/input/RNG/layout mismatch, and localizes the
  first divergence to local backward, DDP/NCCL, clipping, or optimizer update.
- `deterministic_step554_result.json` and `deterministic_step554_report.md`:
  strict deterministic result and the bounded FA2/Liger isolation outcome.

## Isolation contract

1. Never patch the historical checkpoint or the pristine recovered source.
2. Copy the recovered source to an isolated runtime directory.
3. Copy `replay_forensics.py` into that runtime's `scripts/` directory and
   apply `trainer_integration.patch`.
4. Keep the training config's original `max_steps=1106`; stop using
   `BATA_REPLAY_STOP_AFTER_STEP` only after the requested optimizer step.
5. Prepare every replay from a new byte-identical copy of historical
   checkpoint-553.
6. Refuse to reuse a run directory or append to existing rank evidence.

## Local-loss capture position

The recorded value is a pre-backward, rank-local forward/loss value:

1. `CustomSeq2SeqTrainer.compute_task_microbatches()` calls
   `self.compute_loss(model, model_inputs)`.
2. The recovered `_compute_source_weighted_loss()` performs exactly one
   `model(**inputs)` forward, computes `compute_native_sid8_loss()`, and
   returns its scalar loss.
3. The forensic `compute_loss()` wrapper calls the original method and then
   `record_loss(..., value.detach())` before returning.
4. `compute_task_microbatches()` subsequently applies task scaling and calls
   `self.accelerator.backward(scaled_loss)`.
5. DDP gradient hooks and gradient all-reduce run during that backward call,
   after the scalar was captured.

Therefore the step554 rank-local loss divergence predates backward and DDP
gradient all-reduce. DDP/NCCL cannot be the sole first source of that loss
divergence. DDP forward-side behavior is not excluded by this observation,
and the current evidence does not distinguish model-forward kernels from the
local SID8 loss aggregation.

The relevant recovered-source locations are:

- `scripts/train_native_source_domain_r32_v3.py`:
  `_compute_source_weighted_loss()`.
- `rec_pu/sid8_rec_pu_integration.py`:
  `compute_native_sid8_loss()` and its `torch.unique(sorted=False)` plus
  `scatter_add_` aggregation.
- LLaMAFactory `src/llamafactory/train/sft/trainer.py`:
  `compute_task_microbatches()` and `accelerator.backward()`.
- This directory's `replay_forensics.py`:
  `install_replay_instrumentation()`.

## Deterministic one-step diagnostic

Prepare two labels from the real checkpoint:

```bash
python prepare_replays.py \
  --root /root/bata_sft_deterministic_20260902/runs \
  --checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-553 \
  --source-config /root/bata_sft_deterministic_20260902/source_runtime/config/train_bata_baseline_4gpu_gc04_2epoch.yaml \
  --runtime-source /root/bata_sft_deterministic_20260902/source_runtime \
  --llamafactory-source /data/reference/llamafactory-01398eb \
  --tokenized-path /root/bata_sft_micro_replay_20260902/historical_packed_cache_mount \
  --label DET-A --label DET-B
```

Run each sequentially only while all four GPUs are empty:

```bash
bash launch_replay.sh /root/bata_sft_deterministic_20260902 DET-A --deterministic
bash launch_replay.sh /root/bata_sft_deterministic_20260902 DET-B --deterministic
```

Both runs retain `max_steps=1106` but execute only the 553-to-554 update. A
strict deterministic `RuntimeError` is evidence and must not trigger an
automatic fallback or another experiment.

If DET-A/DET-B complete with identical rank-local losses but different
post-backward state, prepare F2/F3/F4 A/B directories with
`prepare_replays.py`, then run:

```bash
python prepare_kernel_matrix.py \
  --runs-root /root/bata_sft_deterministic_20260902/runs \
  --case F2 --case F3 --case F4
```

F1 is the original FA2-on/Liger-on DET-A/DET-B pair. Every additional cell is
strict deterministic, executes only step 553 to 554, and is compared only to
its same-cell repeat. These variants are diagnostic-only and are not a new
historical training contract.

## Compatibility bypass

Transformers 5.3.0 blocks checkpoint restore under PyTorch 2.5.1. The harness
first verifies every checkpoint file against the pinned SHA table, then only
disables the Transformers policy gate. `torch.load(weights_only=True)` remains
active, and NumPy safe globals are allowlisted solely for historical RNG-state
deserialization. Any SHA mismatch fails before the bypass is installed.

## Gradient-localization diagnostic

The strict deterministic DET-A/DET-B pair established `CASE_C`: checkpoint,
ordered inputs, restored RNG, all 16 per-rank microbatch losses, and rank-local
mean loss match exactly, while the post-backward/update state differs. The
bounded GRAD-A/GRAD-B diagnostic keeps FA2 and Liger enabled and executes only
the 553-to-554 update. It wraps PyTorch 2.5.1's official default all-reduce
hook, preserving its divide-by-world-size and asynchronous sum semantics.

The classifier is fail closed:

1. Any initial checkpoint/LoRA/optimizer, input, RNG, microbatch order/count,
   or DDP bucket-layout mismatch is `CONTRACT_MISMATCH`.
2. Matching contracts with different local losses are `D1_PRE_BACKWARD`.
3. Matching local losses compare pre-allreduce, post-allreduce, pre-clip,
   post-clip, and optimizer state in that order.
4. Full equality is `STEP554_FULLY_REPEATABLE`; missing gradient evidence is
   never treated as repeatability.
