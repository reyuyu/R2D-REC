# GR_REC_ThinkExactClamp_Ablation_v1

## Status and history

**TEXT-DOMAIN DECISION-TOKEN PLACEMENT CPU VERIFIED**

**GPU RE-AUDIT NOT RUN / ZERO PARAMETER UPDATE / TRAINING NOT STARTED**

- Phase 1 `917d3d3a9e53db2e80bf425b435c597bb210b804`: Think-only Centered Exact-Clamp.
- Phase 2 `ba813321f3d31158f293e67a5729669e78d42ca9`: added dead-zero Gold-A and
  A-collapse A+B teacher branches.
- Phase 3 `ab47d3141dc08445f6bb4deddb565b710075ff61`: removes the
  A-collapse branch and replaces sequence-wide NoThink credit with Conditional
  Hierarchical Token Credit.
- Phase 4 `e7980753232f46e6aacfad9a8c2da293cade67ad`: adds and hardens the
  standalone zero-step GPU gradient audit harness; the
  formal trainer, objectives and runner contract are unchanged.
- Phase 5 `70b73560caba8f892a6dbbebacba8c7fe85f79b9`: records the authorized
  eight-group zero-step GPU audit and
  its unchanged-parameter checksum evidence.
- Phase 6 `9a5bbb2c989cb0a805b641e843698fc0fe0f135e`: adds the GPU-evidenced
  Domain token-credit stage before the unchanged A/B/C stages.
- Phase 7 `1752ac7069b5b715911274339ecfe9915a71a665`: records the strictly
  paired eight-group SID-Domain-stage zero-step GPU re-audit and comparison.
- Phase 8 `efe7dc02ba833b9b19519e2a01eab19d8857d8f1`: audits the real NoThink
  SFT and historical rollout Domain decision span without GPU execution.
- Phase 9 current: moves the unchanged Domain advantage from final SID
  `<|domain_begin|>` serialization to the preceding natural-language Domain
  decision token; A/B/C and the dead-zero bridge remain unchanged.

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
Domain_adv = .25 * (domain_correct - mean(domain_correct | valid)) / 8
A_adv = .5 * (a_correct - mean(a_correct | domain_correct)) / 8
B_adv = 1.5 * (ab_correct - mean(ab_correct | a_correct)) / 8
C_adv = 6 * (exact - mean(exact | ab_correct)) / 8
```

Ineligible candidates receive zero for that stage. A stage with no indicator
variance is exactly zero.

The Domain formula is unchanged, but its placement is gated by token-native
text/SID alignment. For every valid candidate, the last `视频` / `商品` / `广告` /
`主播` token after `</think>` and before the final SID must exist and match the
final SID domain. Any missing or mismatched valid candidate zeroes only the
entire G8 Domain column. There is no fallback to `<|domain_begin|>`; A/B/C are
still computed normally.

For `[-.25,0,-.25,-.25,-.25,0,-.25,-.25]`, with all candidates
valid, the two correct-domain candidates receive Domain credit `.0234375` and
the six wrong-domain candidates receive `-.0078125`. A/B/C are all zero.

For `[0,.5,2,8,0,.5,2,8]`, Domain is uniformly zero and each repeated
A/B/C cycle remains:

| Reward | Domain | A credit | B credit | C credit |
|---:|---:|---:|---:|---:|
| 0 | 0 | `-.046875` | 0 | 0 |
| .5 | 0 | `.015625` | `-.125` | 0 |
| 2 | 0 | `.015625` | `.0625` | `-.375` |
| 8 | 0 | `.015625` | `.0625` | `.375` |

Thus an already-correct Domain is not penalized for an A/B/C suffix failure,
an already-correct A is not penalized for a wrong B/C suffix, and an
already-correct AB is not penalized at B for a wrong C. No validity stage is
introduced. Uniform `[-.25]*8` remains the recorded
`ALL_WRONG_DOMAIN_ZERO_SIGNAL` boundary.

## Decision-token assignment

The implementation first finds the last contiguous token block matching the
parsed final SID:

```text
[domain_token, a_token, b_token, c_token]
```

It then searches `completion_ids`, not string offsets, after the last
`</think>` and before that SID block. The final recognized natural-language
Domain token is the Domain position. Credit placement is therefore:

```text
[text_domain_token, sid_a_token, sid_b_token, sid_c_token]
```

The final SID `<|domain_begin|>` position always has zero Domain advantage.
Every other token also has zero advantage. A parsed final SID without a
matching contiguous block fails closed; missing or mismatched text Domain uses
the group-level Domain-only gate described above.

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
matching; real-tokenizer four-domain text placement; SID-Domain zero placement;
G8 mismatch/missing Domain gating; unchanged A/B/C values; token-only tensor
writes; sequence-advantage exclusion; dead-zero bridge gating; and uniform
Gold-A CE.

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
cosine, active hierarchy stages, credited Domain/A/B/C token counts, bridge activity,
text/SID alignment diagnostics, and the rollout fingerprint. Aggregate output
includes median/mean/min/max. A
real all-zero bridge is compared with the median nonzero hierarchical norm from
other audited groups. Ratios outside `0.25x..4x` and bridge ratios above `20%`
are flagged for review only; the harness never changes coefficients.

The execution flag `--execute-zero-step-gpu-audit` is mandatory. Its presence
only enables generation plus isolated forward/backward; it does not authorize
training, checkpoint writes, or any parameter update.

## Pre-Domain G8 GPU audit result (2026-08-21)

The authorized audit used one A800 (`cuda:0`), seed `20260816`, the original
8B base and the fresh original BATA adapter. It audited eight real NoThink G8
groups and then stopped. The first launch reached no completed group and failed
closed on activation-memory OOM; the completed run used non-reentrant gradient
checkpointing and shared RNG state for the old/legacy/hierarchical forwards.
Neither run constructed an optimizer or performed a parameter update.

Result artifact:
[`../results/gpu_gradient_scale_audit_g8_seed20260816_20260821.json`](../results/gpu_gradient_scale_audit_g8_seed20260816_20260821.json)

Key aggregate results:

| Metric | Result |
|---|---:|
| Audited real G8 groups | 8 |
| Median `hier_over_legacy` (7 defined) | `0.0` |
| Mean `hier_over_legacy` | `0.1104712968` |
| Median legacy/hier cosine (2 defined) | `0.3411243334` |
| Median active hierarchical gradient norm | `0.1886301152` |
| Real `[0]*8` groups | 0 |

Under the pre-Domain formula, six groups had no active hierarchical stage, one
activated A+B, and one activated A only. Five inactive groups nevertheless had
real `-.25` versus `0` reward variance; that evidence motivated Phase 6's
Domain stage. The historical audit consequently reported
`HIER_GRAD_TOO_SMALL_REVIEW`. This is a preflight review result, not an
automatic coefficient change. No real `[0]*8` group appeared, so the bridge
conclusion is `BRIDGE SCALE NOT OBSERVED`; no rewards were synthesized and the
audit was not extended to sixteen groups.

The trainable-LoRA SHA256 checksum was identical before and after:

```text
b1cfbe7b048ca6c7de8a906ea4419cbe8e339e9974f58d5ed1371f1a471066ad
```

Thus `PARAMETER CHANGE = ZERO`, `optimizer.step = NO`, and
`scheduler.step = NO`.

## Domain-stage paired G8 re-audit (2026-08-21)

This section is historical evidence for the Phase 6/7 SID-Domain placement.
The current Phase 9 harness now places Domain credit on the natural-language
decision token. No GPU re-audit of Phase 9 has been run.

The re-audit used the same one-A800 execution contract, seed `20260816`, G8
sampling parameters, original 8B base, fresh original BATA adapter,
non-reentrant gradient checkpointing and shared forward RNG as the pre-Domain
audit. All eight `group_id`, rollout fingerprint and reward-vector triples
matched exactly, so `paired_audit_valid=true` and Domain-stage attribution is
valid.

Artifacts:

- [`../results/gpu_gradient_scale_audit_domain_g8_seed20260816_20260821.json`](../results/gpu_gradient_scale_audit_domain_g8_seed20260816_20260821.json)
- [`../results/gpu_gradient_scale_audit_domain_paired_comparison_g8_seed20260816_20260821.json`](../results/gpu_gradient_scale_audit_domain_paired_comparison_g8_seed20260816_20260821.json)

| Metric | Pre-Domain | Domain stage |
|---|---:|---:|
| Active hierarchical groups | 2 | 7 |
| Median `hier_over_legacy` (7 defined) | `0.0` | `0.00005477798` |
| Mean `hier_over_legacy` | `0.1104712968` | `0.1220513189` |
| Median legacy/hier cosine | `0.3411243334` (2 defined) | `0.02377167344` (7 defined) |
| Median active hierarchical gradient norm | `0.1886301152` | `0.00006368184` |

Groups 0/1/3/4/5 are Domain-only: all changed from zero hierarchical gradient
to positive gradient, with new norms `0.3667742312`, `0.000008133517`,
`0.000018333494`, `0.000063681837`, and `0.000002085016`. The corresponding
`hier_over_legacy` values are `0.08060434529`, `0.0000102956524`,
`0.0000036167795`, `0.0000547779839`, and `0.00000273857159`. Their cosines are
`0.6929594874`, `0.02377167344`, `0.00457752822`, `-0.005322964862`, and
`-0.01470425259`. The large spread is recorded without changing coefficients.

Group 2 (`[-.25]*8`) remains all-stage zero. Group 6 retains A+B activity and
has Domain off. Group 7 retains A activity and also activates Domain because
its paired rewards contain both `-.25` and correct-domain values; an expectation
that Group 7 Domain would be zero would contradict the formal Domain formula.
No real `[0]*8` appeared, so `BRIDGE SCALE NOT OBSERVED` remains the only valid
bridge conclusion.

The new audit's trainable-LoRA checksum is identical before and after:

```text
b1cfbe7b048ca6c7de8a906ea4419cbe8e339e9974f58d5ed1371f1a471066ad
```

No optimizer or scheduler step occurred, no parameter changed, and no training
or checkpoint write was started.

## Frozen runner contract

The runner is not restructured. Prefix remains `GR-REC-CLAMP-BRIDGE-V1-`,
formal max steps remain 1500, fresh BATA and same-run-only resume remain fixed,
and the existing seeds, route schedule, G4/G8, `num_iterations=2`, PPO
hyperparameters, sampling and Beam32 contracts remain unchanged.

## Final status

**TEXT-DOMAIN DECISION-TOKEN PLACEMENT CPU VERIFIED**

**GPU RE-AUDIT NOT RUN / ZERO PARAMETER UPDATE / TRAINING NOT STARTED**
