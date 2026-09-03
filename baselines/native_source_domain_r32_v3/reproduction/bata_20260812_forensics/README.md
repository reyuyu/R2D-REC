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
- `launch_fadet_replay.sh` and `fadet_step554_contract.json`: the bounded
  FADET-A/B launcher and frozen diagnostic contract. The launcher exports
  `FLASH_ATTENTION_DETERMINISTIC=1` before Python starts; the runtime probe
  fails closed unless an actual flash-attn call receives `deterministic=True`.
- `case_c_step554.json`: public-safe raw fingerprint/scalar evidence for the
  first observed A/B divergence.
- `summarize_replay_pair.py`: compares two four-rank evidence directories,
  fails closed on any checkpoint/input/RNG/layout mismatch, and localizes the
  first divergence to local backward, DDP/NCCL, clipping, or optimizer update.
- `deterministic_step554_result.json` and `deterministic_step554_report.md`:
  strict deterministic result and the bounded FA2/Liger isolation outcome.
- `gradient_step554_result.json` and `gradient_step554_report.md`:
  public-safe DDP bucket, clipping, and update evidence localizing the first
  repeat divergence to local backward.
- `fadet_step554_result.json` and `fadet_step554_report.md`: the successful
  strict FA2 deterministic-backward repeat. The result is fully repeatable
  through step554 and supports the bounded material-source conclusion in the
  report.

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

## FA2 deterministic-backward diagnostic

FADET-A and FADET-B are fresh copies of the historical checkpoint-553 and
execute exactly one optimizer update. For frozen-contract runs, all eight
training tensors are hashed in the collator while they are still on CPU. Only
the hashes and tensor metadata travel with the consumed batch; `compute_loss`
removes that diagnostic key before the model forward. This verifies the 16
actual rank-local microbatches and their consumption order without a CUDA to
CPU copy or an extra forward.

Run sequentially after preparing fresh `FADET-A` and `FADET-B` directories:

```bash
bash launch_fadet_replay.sh /root/bata_sft_fadet_20260902 FADET-A
bash launch_fadet_replay.sh /root/bata_sft_fadet_20260902 FADET-B
```

FADET-A keeps the raw PRE_ALLREDUCE bucket only in its isolated server-side
diagnostic directory. FADET-B compares matching parameter slices and emits
public-safe per-LoRA statistics; raw gradient values are never included in
the summary or committed. A stable verdict additionally requires confirmed
`deterministic=True` calls on every rank. Missing confirmation is
`UNRESOLVED`, never stable.

## BATA-STABLE-V0 production-like pilot

`BATA-STABLE-V0` turns the confirmed FA2 deterministic-backward controls into
a low-interference candidate recipe. It keeps the recovered 06:33 training
source, historical packed cache, original optimizer/schedule, four A800 ranks,
GA16, 8K neat packing, FA2, Liger, bf16, and the original 1106-step scheduler
horizon. It adds only strict PyTorch determinism,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`FLASH_ATTENTION_DETERMINISTIC=1`, a startup contract snapshot, and a callback
that saves checkpoint-560 before stopping.

The stable runtime does not register a DDP communication hook, hash gradient
tensors, retain a local gradient reference, monkey-patch FlashAttention, or
execute extra forward/backward calls. Prepare three fresh checkpoint-553
copies with `prepare_stable_pilot.py`, then launch `STABLE560-A`,
`STABLE560-B`, and `STABLE560-C` sequentially with
`launch_stable_pilot.sh`. Each launch fails if any GPU is already occupied or
if source/checkpoint/config contracts changed.

After all three runs stop at 560, `compare_stable_pilots.py` compares raw LoRA,
full effective B@A geometry via low-rank trace identities, canonical optimizer,
scheduler, and RNG state. It also projects each seven-step effective update
onto the historical 553-to-1106 update globally, by attention/MLP projection,
and by layer. The historical-direction metrics are observations only and do
not affect the stability verdict or any checkpoint.

The completed public-safe evidence is in `stable560_result.json` and
`stable560_report.md`. The private server-side checkpoint paths and artifacts
are intentionally excluded.

## BATA-STABLE-V0 first-epoch repeatability

The `stable553` tools extend the validated deterministic runtime to two
independent fresh-base runs. `prepare_stable553.py` pins the recovered source,
base-model files, 222,001-row BATA dataset, 35,380-row historical packed cache,
generated configs, and LLaMAFactory runtime before creating `STABLE553-A` and
`STABLE553-B`. Both configs explicitly disable checkpoint resume.

`launch_stable553.sh` requires all four A800 GPUs to be empty, sets the same
strict deterministic controls as STABLE560, preserves the original 1106-step
scheduler horizon, and uses `stable553_runtime.py` only to save and stop at
optimizer step553. It does not install gradient hashes, DDP hooks, FlashAttention
patches, or any extra forward/backward operation.

After both sequential runs finish, `compare_stable553.py` compares raw LoRA,
effective B@A, optimizer, scheduler, four-rank RNG, and shared scalar logs. It
also compares STABLE553-A with historical checkpoint-553 by projection and
layer, but historical similarity is excluded from the repeatability verdict.
`publish_stable553.py` emits only path-free JSON and Markdown evidence.

## Controlled epoch-2 continuations

`prepare_stable_epoch2.py`, `stable_epoch2_runtime.py`, and
`launch_stable_epoch2.sh` provide an isolated checkpoint-553 to
checkpoint-1106 continuation. The source checkpoint is always STABLE553-A;
its adapter, optimizer, scheduler, trainer state, and four rank RNG files are
SHA-pinned before launch. Every run has a new seed-labelled directory and the
launcher refuses occupied GPUs or reused output.

The 8892 monitor exposes the same fixed workflow through a local-only button.
Its API accepts only an integer seed and cannot accept a path or shell command.
The dashboard shows continuation progress and curves, provides adapter-only
downloads after checkpoint-1106, and stores optional manually entered external
scores and notes in `manual_scores.json`. Changing the seed is an explicit
random-seed ablation; the original continuation uses seed `20260806`.
