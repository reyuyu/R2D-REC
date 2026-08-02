# OneReason multi-task SFT extension

This repository is a private, reproducible LLaMA-Factory-based implementation for the OneReason multi-task SFT workflow.

## What is included

- `multitask_macro_training`: task-aware macro-step training with four top-level tasks.
- Same-subtask packing with segment-level `position_ids` reset. With FlashAttention 2 and Transformers v5.6+, the reset positions are converted to varlen `cu_seqlens`, preventing cross-segment attention.
- Configurable fixed or `balanced_40` macro-step super-cycle.
- DDP-consistent task ordering, sampler state, macro-loader state, and checkpoint recovery at macro-step boundaries.
- Per-task loss and packing statistics under `output_dir/monitor/metrics.jsonl`.
- Dataset split, prediction, checkpoint watching, cutoff audit, metric plotting, SID/action evaluation utilities, and unit tests.

## Important training semantics

When `multitask_macro_training: true`:

- `max_steps` counts macro-steps / optimizer updates.
- `gradient_accumulation_steps` must be `1`.
- A macro-step contains the configured number of packed task microbatches.
- Native `packing` and `neat_packing` must remain `false`; macro mode owns packing.

See `configs/onereason/` for the two-GPU LoRA rank 16 and rank 64 experiment configurations.

## Not included

Model weights, checkpoints, datasets, raw predictions, logs, monitoring records, credentials, and local environment files are intentionally excluded.

## Upstream

This project is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) commit `01398eb` and retains its upstream license.
