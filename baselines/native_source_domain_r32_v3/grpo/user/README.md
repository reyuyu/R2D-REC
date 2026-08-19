# GR_USER_v1 data preparation

This is the CPU-only first phase of the independent User GRPO project. It builds
an evaluation-aligned Action/Chain NoCoT dataset and does not contain a trainer,
reward implementation, GPU generation, or formal launch path.

Runtime layout:

```text
/data/GRPO_USER/
├── scripts/
├── config/
├── tests/
├── docs/
├── data/gr_user_v1/
├── runs/
├── outputs/
└── cache/
```

Build:

```bash
cd /data/GRPO_USER
python3 scripts/build_gr_user_v1.py \
  --source-dir '/data/lf_data_versions/task_pools/懂用户/chian异常清洗' \
  --output-dir /data/GRPO_USER/data/gr_user_v1 \
  --parent-checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106 \
  --eval-examples /data/GRPO_USER/data/gr_user_v1/evidence/eval_examples.json
```

Test:

```bash
python3 tests/test_gr_user_v1.py \
  --data-dir /data/GRPO_USER/data/gr_user_v1 \
  --parent-checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106
```

The selected parent is beta-baseline Epoch 2 (`checkpoint-1106`). The adapter
operates on raw user content and never adds chat special tokens. The parent
checkpoint tokenizer owns the chat wrapper and NoCoT assistant prefix.
The 5/5 Action and 5/5 Chain examples in the supplied evaluation log contain a
cross-user format demonstration, so both V1 routes include a fixed demonstration
that is independent of the current sample and its Gold.

## GPU policy

Phase 1 is CPU-only even when GPUs are idle. In later phases, tests may use one
to four GPUs, including multi-GPU execution, only after checking the intended
devices are unoccupied. An occupied GPU must not be used or preempted.
