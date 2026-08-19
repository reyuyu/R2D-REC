# GR_USER_v1

GR_USER_v1 is an independent User GRPO project. Phase 1 freezes the
evaluation-aligned Action/Chain NoCoT dataset. Phase 2 adds a CPU-only reward,
constraint, and token-attribution layer. It still contains no trainer, GPU
generation, or formal launch path.

The selected parent is beta-baseline Epoch 2 (`checkpoint-1106`). Frozen data
must retain the SHA256 values recorded in `data/gr_user_v1/manifest.json`.

## Data build

```bash
cd /data/GRPO_USER
python3 scripts/build_gr_user_v1.py \
  --source-dir '/data/lf_data_versions/task_pools/懂用户/chian异常清洗' \
  --output-dir /data/GRPO_USER/data/gr_user_v1 \
  --parent-checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106 \
  --eval-examples /data/GRPO_USER/data/gr_user_v1/evidence/eval_examples.json
```

The prompt adapter operates on raw user content and never adds chat special
tokens. The parent tokenizer owns the chat wrapper and NoCoT assistant prefix.
Both routes use a fixed cross-user format demonstration that is independent of
the current sample and its Gold.

## Data tests

```bash
python3 tests/test_gr_user_v1.py \
  --data-dir /data/GRPO_USER/data/gr_user_v1 \
  --parent-checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106
```

## Reward tests

```bash
cd /data/GRPO_USER
python3 -m unittest discover -s tests -p 'test_user_rewards.py' -v
```

## Controlled audit

```bash
python3 scripts/audit_user_rewards.py \
  --data-dir /data/GRPO_USER/data/gr_user_v1 \
  --parent-checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106 \
  --output /data/GRPO_USER/data/gr_user_v1/reward_audit_v1.json \
  --summary-output /data/GRPO_USER/results/reward_audit_v1_summary.json
```

The audit verifies frozen data before and after reading it, runs controlled
perturbations over probe and pilot data, audits 100 real SIDs with the parent
tokenizer, checks 50 hallucination and 50 duplicate spans per route, and
benchmarks 300 samples per route on CPU.

## Penalty mask compiler

Phase 3A compiles the whitelisted Phase 2 violations into per-kind and union
token masks. It does not set penalty weights or connect to training.

```bash
python3 tests/test_user_penalty_mask.py \
  --parent-checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106 \
  -v

python3 scripts/audit_user_penalty_mask.py \
  --data-dir /data/GRPO_USER/data/gr_user_v1 \
  --parent-checkpoint /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106 \
  --output /data/GRPO_USER/results/penalty_mask_audit_v1.json \
  --summary-output /data/GRPO_USER/results/penalty_mask_audit_v1_summary.json
```

The locality audit uses only real four-token SIDs from pilot history and checks
50 examples for each of six Action/Chain violation categories.

## GPU policy

Phases 1 and 2 are CPU-only even when GPUs are idle. The scripts do not import
Torch, generate completions, initialize a trainer, or select a CUDA device.
Future GPU work may use only devices confirmed to be unoccupied and must not
preempt another process.
