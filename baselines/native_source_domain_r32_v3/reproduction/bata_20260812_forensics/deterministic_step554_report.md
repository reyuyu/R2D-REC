# BATA checkpoint-553 deterministic step554 diagnosis

## Result

The strict deterministic replay result is **D3**. DET-A and DET-B completed
one optimizer update from the real historical checkpoint-553. Every rank had
the same ordered batch, explicit RNG fingerprint, all 16 rank-local
microbatch losses, and rank-local mean loss across the two runs. The global
gradient norm and all post-update fingerprints nevertheless differed.

This does not identify a specific kernel. The narrow defensible root-cause
label remains `UNRESOLVED`, bounded to execution after the captured local-loss
scalar and before or during the optimizer update. Local backward and DDP/NCCL
reduction remain live possibilities. Equal scalar losses do not prove that
all hidden activations or logits were bit-identical.

## Deterministic contract

- Real BATA `checkpoint-553`, including historical optimizer, scheduler, and
  four rank RNG states.
- Recovered 06:33 source and the same read-only packed Arrow cache.
- Four A800 GPUs, gradient accumulation 16, FA2 on, Liger on.
- Configured training horizon remained 1106 steps; the diagnostic stop hook
  stopped after the single 553-to-554 update.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` was set before Python/CUDA startup.
- `torch.use_deterministic_algorithms(True, warn_only=False)` was applied
  before CUDA initialization and verified in rank evidence.

## Four-rank comparison

| Rank | Batch SHA (A=B) | RNG SHA (A=B) | Local loss A | Local loss B | Grad A | Grad B |
|---:|---|---|---:|---:|---:|---:|
| 0 | `acec704d...46095` | `b3cfd074...d9e2f` | 2.0197828598320484 | 2.0197828598320484 | 0.7419519424 | 0.7419999242 |
| 1 | `045f5ddb...0a034` | `2d745ebb...1b323` | 1.5963932350277900 | 1.5963932350277900 | 0.7419519424 | 0.7419999242 |
| 2 | `b3d47427...a05ad` | `754dad79...ac17f` | 2.5609864667057990 | 2.5609864667057990 | 0.7419519424 | 0.7419999242 |
| 3 | `365d252d...5f3f` | `4e147f71...f8b` | 1.3222751496359706 | 1.3222751496359706 | 0.7419519424 | 0.7419999242 |

For every rank, the complete ordered list of 16 microbatch fingerprints and
the complete list of 16 loss scalars also matched exactly.

At step554, DET-A/DET-B had different canonical LoRA, effective B@A, and
optimizer SHA256 fingerprints. Their initial553 LoRA and optimizer
fingerprints were equal.

## Capture position

The wrapper in `replay_forensics.py` calls the recovered trainer's original
`compute_loss`, then records its detached result. The original function first
runs `model(**inputs)` and `compute_native_sid8_loss()`. LLaMAFactory then
scales that returned loss and calls `accelerator.backward()`. DDP gradient
hooks and all-reduce occur during backward, after the recorded scalar.

The reported rank-local losses are therefore captured after forward and local
SID8 loss aggregation, but before backward and DDP gradient all-reduce.

## FA2/Liger isolation

The fixed-contract 2x2 matrix could not be completed without changing memory
behavior:

- F1, FA2 on/Liger on: A/B completed and produced D3.
- F2, FA2 off/Liger on: F2-A failed in eager-attention forward softmax with a
  CUDA OOM while requesting 8.00 GiB. It did not reach step554.
- F3, FA2 on/Liger off: F3-A failed inside `accelerator.backward()` with a
  CUDA OOM while requesting 5.38 GiB. It did not reach step554.
- F4, both off: not run because each constituent disable path already exceeded
  the fixed 80 GiB contract. No batch size, sequence length, GC fraction, or
  training parameter was changed to force these cells to run.

Raw JSONL, checkpoint copies, and full tracebacks remain local-only evidence
because they are runtime artifacts or large logs. The public JSON contains
only hashes, scalar values, environment contract, and sanitized error
summaries.

## Next minimum diagnostic

On the unchanged F1 path, run one more independent A/B pair with lightweight
gradient fingerprints immediately before and after DDP reduction. It must not
add another model forward or alter the optimizer. This is the minimum check
that can distinguish local backward divergence from DDP/NCCL reduction before
any attempt to recover the full 553-to-1106 trajectory.
