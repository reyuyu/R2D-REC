# GR_REC_ThinkExactClamp_Ablation_v1

## Status and history

**GPU AUDIT HARNESS READY / EXECUTION PENDING AUTHORIZATION**

**GPU NOT USED / TRAINING NOT STARTED**

- Phase 1 `917d3d3a9e53db2e80bf425b435c597bb210b804`: Think-only Centered Exact-Clamp.
- Phase 2 `ba813321f3d31158f293e67a5729669e78d42ca9`: added dead-zero Gold-A and
  A-collapse A+B teacher branches.
- Phase 3 `ab47d3141dc08445f6bb4deddb565b710075ff61`: removes the
  A-collapse branch and replaces sequence-wide NoThink credit with Conditional
  Hierarchical Token Credit.
- Phase 4 current: adds a standalone zero-step GPU gradient audit harness; the
  formal trainer, objectives and runner contract are unchanged.

The branch remains `ablation/gr-rec-think-exact-clamp-v1`. Initialization remains
the fresh original BATA adapter; Git phases are code history, not checkpoint
resume sources.

## Think

Think Centered Exact-Clamp is unchanged:

```text
raw_i = (reward_i - group_mean) / 8
advantage_i = 0 if reward_i >= 8 and raw_i < 0 else raw_i
```

All ten Phase 1 archetypes and G4 isolation remain regression tested.

## NoThink hierarchy state

`q_reward` and its values `-1/-0.25/0/.5/2/8` are unchanged. Each final SID is
classified as `valid`, `domain_correct`, `a_correct`, `ab_correct`, and `exact`,
with monotonic implication from exact down to valid.

For one NoThink G8, fixed scale is 8:

```text
A_adv = .5 * (a_correct - mean(a_correct | domain_correct)) / 8
B_adv = 1.5 * (ab_correct - mean(ab_correct | a_correct)) / 8
C_adv = 6 * (exact - mean(exact | ab_correct)) / 8
```

Ineligible candidates receive zero for that stage. A stage with no indicator
variance is exactly zero.

For `[0,.5,2,8,0,.5,2,8]`, each repeated cycle is:

| Reward | A credit | B credit | C credit |
|---:|---:|---:|---:|
| 0 | `-.046875` | 0 | 0 |
| .5 | `.015625` | `-.125` | 0 |
| 2 | `.015625` | `.0625` | `-.375` |
| 8 | `.015625` | `.0625` | `.375` |

Thus an already-correct A is not penalized for a wrong suffix, and an
already-correct AB is not penalized at B for a wrong C.

## Final SID token assignment

The implementation finds the last contiguous token block matching the parsed
final SID:

```text
[domain_token, a_token, b_token, c_token]
```

It writes A/B/C credit only to the corresponding A/B/C completion-token
positions. Every other token has zero advantage. A parsed final SID without a
matching contiguous block fails closed.

The parent population-std scalar advantage is still computed and retained for
debugging, but it is not consumed by the NoThink loss.

## Token-level PPO

NoThink retains the same current/old log-probability, PPO ratio, clipping,
completion mask, route multiplier and gradient-accumulation division. Formal
GRPO uses a credited-token sum per sample, avoiding implicit dilution by the
number of unrelated completion tokens:

```text
per_sample_loss = sum(clipped_PPO_loss * token_advantages * completion_mask)
```

The old sequence-wide loss is not added, so there is no double counting. BNPO
retains its existing token-mask normalization and is not redesigned here.

## Dead-zero bridge

Only exact rewards `[0]*8` activate the retained bridge. Conditional hierarchy
credit is then all zero, so the bridge supplies a small uniform-mean Gold-A
teacher loss. It teaches no B or C and keeps `lambda_bridge=0.02`.

The former `a_collapse_ab_bridge`, missing Gold-A teacher, current-path Gold-B
teacher, B query and B teacher forward are absent from current code.

## CPU verification

Tests cover the required mixed archetype; A-only, B-only and C-only variance;
uniform `.5`, `2`, `8`, and `0` groups; correct-prefix invariants; hierarchy
state derivation; global G8-to-rank row alignment; last contiguous final-SID
matching and fail-close; token-only tensor writes; sequence-advantage exclusion;
dead-zero bridge gating; and uniform Gold-A CE.

Think ExactClamp regression and the baseline TRL structure suite must continue
to pass before any future GPU validation.

## Zero-step GPU gradient audit

`ablations/gr_rec_think_exact_clamp_v1/audit_gpu_gradient_scale.py` compares,
for the same real NoThink G8 rollout, old log-probabilities and model parameters:

1. Legacy population-std sequence-level GRPO gradient.
2. Current Conditional Hierarchical Token Credit gradient.
3. Raw and `0.02`-weighted Gold-A bridge gradient, only when the generated
   reward vector is exactly `[0]*8`.

Each objective starts with `model.zero_grad(set_to_none=True)`, performs a fresh
forward/backward, copies only `requires_grad=True` LoRA gradients to CPU for the
global L2 norm/cosine calculation, and clears gradients again. No optimizer or
scheduler is constructed and the harness contains no `.step()` call.

The default audit is eight independently generated G8 groups and can be set
from one to sixteen. Per-group output includes reward topology, norms, ratio,
cosine, active hierarchy stages, credited A/B/C token counts, bridge activity,
and the rollout fingerprint. Aggregate output includes median/mean/min/max. A
real all-zero bridge is compared with the median nonzero hierarchical norm from
other audited groups. Ratios outside `0.25x..4x` and bridge ratios above `20%`
are flagged for review only; the harness never changes coefficients.

The execution flag `--execute-zero-step-gpu-audit` is mandatory. Its presence
only enables generation plus isolated forward/backward; it does not authorize
training, checkpoint writes, or any parameter update.

## Frozen runner contract

The runner is not restructured. Prefix remains `GR-REC-CLAMP-BRIDGE-V1-`,
formal max steps remain 1500, fresh BATA and same-run-only resume remain fixed,
and the existing seeds, route schedule, G4/G8, `num_iterations=2`, PPO
hyperparameters, sampling and Beam32 contracts remain unchanged.

## Final status

**GPU AUDIT HARNESS READY / EXECUTION PENDING AUTHORIZATION**

**GPU NOT USED / TRAINING NOT STARTED**
