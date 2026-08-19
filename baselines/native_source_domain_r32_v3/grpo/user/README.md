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

## G=4 rollout-only audit

Phase 3B loads the frozen Beta baseline Epoch 2 parent in `eval()` mode with
all parameters set to `requires_grad=False`. It performs NoCoT sampling and
reward/mask auditing only; the runner contains no optimizer, backward, trainer,
or checkpoint-save path.

The formal audit deterministically selects 50 Action and 50 Chain prompts from
`pilot_600.jsonl`, generates four candidates per prompt with temperature 0.9,
top-p 0.95, seed 20260819, and `max_new_tokens=512`, then verifies LoRA tensor
checksums and frozen-data SHA values before publishing results. Independent
`audit-shard` workers may bind to four already-idle GPUs; CPU `aggregate` mode
accepts only 100 unique prompts, 400 unique candidates, and 100 groups.

The completed run `GR-USER-G4-AUDIT-4GPU-20260819-181843` used GPUs 0-3.
Action and Chain zero-std rates were 4% and 0%, respectively, so G=4 is healthy
under the predeclared `<5%` criterion. See
`results/rollout_audit_g4_v1_summary.json` and
`docs/rollout_audit_g4_v1.md` for the full static report.

## Token advantage offline simulation

Phase 4A reads the frozen Phase 3B candidates/groups and simulates token
advantages without generation, GPU use, model loading, or training. It compares
fixed-token and square-root-normalized spans at lambda 0.10, 0.25, 0.50, and
1.00. Zero-std groups retain task advantage 0 while eligible masked tokens can
still receive local negative signal; overlapping violations use the strongest
effective lambda and are never summed.

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/simulate_user_token_advantage.py \
  --run-dir /data/GRPO_USER/runs/GR-USER-G4-AUDIT-4GPU-20260819-181843 \
  --summary-output /data/GRPO_USER/results/token_advantage_sim_v1_summary.json \
  --docs-output /data/GRPO_USER/docs/token_advantage_sim_v1.md
```

The static evidence recommends square-root normalization with initial lambda
0.50. This preserves meaningful correction on short SID spans while keeping
roughly 80-token duplicate events from receiving linear per-token amplification.
See `results/token_advantage_sim_v1_summary.json` and
`docs/token_advantage_sim_v1.md` for all eight schemes and representative
high-reward violations.

## GPU policy

Phases 1-3A are CPU-only even when GPUs are idle. Phase 3B may use only devices
confirmed to be unoccupied and must not preempt another process. It generates
audit candidates but does not train or mutate model parameters.

## Passive monitor adapter

`scripts/user_monitor_adapter.py` converts already-computed User GRPO trainer
statistics into the shared append-only monitor schema. It performs no reward
recomputation, model call, distributed collective, or training control. A User
run manifest must use `run_kind: user_grpo`; legacy Recommendation manifests
remain unchanged and default to `recommendation_grpo` in the server.

The adapter accepts common policy diagnostics plus route-specific Action,
Chain, token-advantage, violation-count, per-kind penalty-mass, rollout, and
sampled-trace fields. Missing optional diagnostics are omitted. This keeps the
monitor fail-open and prevents monitoring from changing optimization behavior.

## Fixed Probe monitoring

`scripts/user_fixed_probe.py` evaluates the frozen `probe_v1.jsonl` set with
12 Action and 8 Chain prompts at Step 0, every 10 optimizer steps, and the
final step. Sampling settings and seed stay fixed across checkpoints. Probe
generation runs under inference mode and does not contribute reward,
advantage, loss, gradients, or optimizer state.

The User `Prob / 固定探针` view compares Action F1/precision/recall and Chain
total/action/logic alignment over training time. Candidate text marks proven
correct SID or exact event spans in green and local penalty spans in red;
diagnostic-only violations such as `wrong_selection_sid` remain neutral.
`scripts/backfill_user_fixed_probe.py` can populate an older run using frozen
checkpoints, but only as rollout-only inference on four confirmed-idle GPUs.

## Chain degradation diagnosis

Phase 5D performs matched, inference-only Chain diagnosis over C0, C20, and
C40. It first analyzes the existing eight fixed Chain probes, then builds a
40-sample held-out `probe_chain_v2.jsonl` directly from the audited native
NoCoT and converted CoT source pools. Probe v2 has zero sample overlap with
`train_3000`, `pilot_600`, `probe_v1`, and cumulative Pilot300; its event-count,
source, and prompt-length strata track the real Chain training distribution.

The completed diagnosis classified the original eight-sample decline as
`SMALL_PROBE_NOISE`: on Probe v2, C40-C0 Chain Total was +0.004985 with median
+0.003492 and bootstrap 95% CI [-0.006283, +0.016295]. Of 40 matched samples,
18 improved, 7 were approximately unchanged, and 15 degraded. The Pilot300
Chain mix was nevertheless distribution-shifted versus train: 2-event and
5-event samples were each represented at 2.0x, while 3-event samples were at
0.55x.

See `docs/chain_diagnosis_v1.md`, `results/chain_diagnosis_v1.json`, and
`results/diagnosis/` for per-sample comparisons, reward/constraint conflict,
token-local attribution, and five diagnostic figures. The runner contains no
optimizer, backward, training, or checkpoint-save path.
