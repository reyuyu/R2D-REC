# NSD-R32-V3-2E-GC04-2GPU

## Goal

This is an isolated Native SFT reproduction baseline adapted to the current OneReason data. It deliberately excludes the current macro-step trainer, GradNorm, Action auxiliary losses, recommendation Trie losses, and BFD coverage scheduling. The aim is an attributable comparison against the peer material-domain route rather than another multitask-loss ablation.

## Frozen Training Method

- Reference LLaMA-Factory source: `01398eb18dd475a6e27c36f15b970aeacf0d4a60`
- Model: `/data/models/onereason-8b-pretrain-competition`
- LoRA: target `all`, rank 32, alpha 64, dropout 0.05
- Sequence path: BF16, FlashAttention-2, Liger 0.8.1, 8,192-token neat packing
- Optimization: AdamW SFT, learning rate `2e-4`, weight decay `0.01`, cosine scheduler, warmup ratio `0.03`, seed `20260806`
- Two GPUs: physical GPU 2 and 3, per-device batch 1, accumulation 32, global batch 64
- Epochs: 2; audited dataset has 33,515 packed sequences, so 524 optimizer steps/epoch and 1,048 optimizer steps total
- Partial gradient checkpointing: `NATIVE_GC_FRACTION=0.4`, which checkpoints a fixed prefix of 14 out of 36 Qwen3 decoder blocks. It does not change model weights, loss, optimizer, or schedule.

## Data

The active dataset version is `native_source_domain_r32_v3`, with 219,370 records and no world data. It combines material bucket canonical/no-think data, reversed material data, sampled material data, V1 user data, and V3 recommendation data. The manifest records source counts, material-domain balancing values, input files, and SHA256.

Recommendation V3 multi-positive metadata is retained in `aux_metadata_json` outside `model.forward`. The baseline loss intentionally leaves it unused. This is the extension interface for a future recommendation loss.

## Monitoring And Artifacts

- Trainer metrics: `loss`, `grad_norm`, `learning_rate`, `epoch`, every 5 optimizer steps.
- Complete metric log: `train.log`; each checkpoint persists full `log_history` in `trainer_state.json`. TensorBoard is not installed in the isolated reference environment, so no dependency is added.
- Resume checkpoints: steps 262, 524, 786, and 1,048; `save_total_limit=4`, including optimizer and scheduler state.
- Full stdout/stderr: `/data/logs/baselines/native_source_domain_r32_v3/NSD-R32-V3-2E-GC04-2GPU-20260810/train.log`.
- Output root: `/data/outputs/baselines/native_source_domain_r32_v3/NSD-R32-V3-2E-GC04-2GPU-20260810`.

No task-specific metrics are fabricated here: using GradNorm or recommendation metrics would no longer be a Native SFT reproduction.

## Preflight

The exact dual-GPU 8K path was smoke-tested for 5 optimizer steps under all checkpointing options:

| Mode | Result | Time / step | Peak memory / GPU |
|---|---|---:|---:|
| full GC | pass | 87.85 s | 41.3 GiB |
| 0.5 GC, 18/36 blocks | pass | 75.64 s | 70.1 GiB |
| 0.4 GC, 14/36 blocks | pass | 73.01 s | 76.3 GiB |
| no GC | OOM at first step | - | requires more than 80 GiB |

0.4GC is selected at the user's request. It is a narrow-memory configuration: the smoke passed, but long training still carries more OOM risk than 0.5GC. At the measured rate, 2 epochs are approximately 21.3 hours excluding initialization and checkpoint I/O.

## Launch

```bash
cd /data/baselines/native_source_domain_r32_v3
bash scripts/launch_2gpu_gc04_2epoch.sh
```
