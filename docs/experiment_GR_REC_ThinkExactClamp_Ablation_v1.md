# GR_REC_ThinkExactClamp_Ablation_v1

## Status and history

**EARLIEST DOMAIN BRANCH POINT GPU PAIRED DIAGNOSTIC COMPLETE**

**ZERO PARAMETER UPDATE / TRAINING NOT STARTED**

## R2 48-step optimizer smoke (2026-08-21)

Commit `e35f487be55b551f8d0b00f3cd18ce42ef470faa` fixed the detailed-monitor
advantage tail replacement for both `deque` and `list` logs without changing
the training formula. All three focused CPU suites and the fast commitment
preflight passed before launch.

`GR-REC-CLAMP-BRIDGE-V1-SMOKE48-R2-20260821` then ran from original 8B plus
fresh original BATA on four A800 GPUs, with no resume, and stopped at exactly
48 optimizer steps. It produced 32 Think G4 groups and 32 NoThink G8 groups.
The result is `SMOKE_PASS`: loss range was
`[-0.02894183062016964, 0.00028963969089090824]`, gradient-norm range was
`[0.023588674142956734, 0.2531088888645172]`, and no NaN, Inf, OOM, traceback,
or save failure occurred.

Think reward mean was `2.981781005859375`; 14/32 Think groups were zero-std.
NoThink reward mean was `0.078125`; 1/32 NoThink groups was zero-std. NoThink
stage-active counts were Domain `22/32`, A `17/32`, B `4/32`, and C `0/32`.
Commitment accounting was branch `252`, direct-SID fallback `4`, unresolved
`0`, out of 256 eligible candidates. The sole zero-std NoThink group was an
all-wrong-Domain group; no dead-zero group occurred, so the bridge correctly
did not activate.

Optimizer steps did execute. Trainable LoRA changed by
`0.00027950378729713066` on every rank while frozen base parameters had exactly
zero delta. The step-48 checkpoint was written only to the smoke output tree;
no pilot, formal run, or benchmark was started.

Artifact:

- [`../results/optimizer_smoke48_r2_20260821.json`](../results/optimizer_smoke48_r2_20260821.json)

**R2 48-STEP OPTIMIZER SMOKE PASS / STOPPED AT STEP 48**

## G8-anchored milestone baseline fix and 12-step smoke (2026-08-21)

Commit `6c5f41254e3d8969fa9c71489c134310756ddc3c` fixes the NoThink
baseline-scope bug. Domain/A/B/C milestone means are now computed over the
complete G8, while token credit remains prefix-gated and Domain placement
remains candidate-level earliest commitment with direct-SID fallback. All
increments, scale, route multiplier, credited-token SUM reduction, bridge,
PPO settings, and Think ExactClamp are unchanged.

Exact CPU regressions A-J passed, including positive B credit for singleton AB
success and positive B/C credit for singleton Exact success. Monitor events now
record Domain/A/B/C stage-active booleans plus singleton B/C success-group
booleans.

The fresh four-A800 run
`GR-REC-CLAMP-BRIDGE-V1-SMOKE12-G8BASE-20260821` completed exactly 12/12
optimizer steps from original 8B plus fresh original BATA. It executed 8 Think
G4 groups and 8 NoThink G8 groups. Domain/A/B/C active counts were `6/2/0/0`
(`75%/25%/0%/0%`). No real singleton B or C group occurred in this short
sample, so both observed singleton counts were zero; their behavior is covered
by the exact CPU regressions.

Loss range was `[-0.006651192903518677, 0.00007984426338225603]` and gradient
norm range was `[0.07880621403455734, 0.17861153185367584]`. No NaN, Inf, OOM,
or runtime exception occurred. Trainable LoRA delta was
`0.000033886617558209764` on every rank and frozen base delta was exactly zero.
The process stopped after step 12; formal 1500-step training was not started.

Artifact:

- [`../results/optimizer_smoke12_g8_baseline_20260821.json`](../results/optimizer_smoke12_g8_baseline_20260821.json)

**G8-ANCHORED BASELINE 12-STEP SMOKE PASS / FORMAL TRAINING NOT STARTED**

## Phase 12: formal earliest Domain commitment credit (2026-08-21)

The formal NoThink Domain stage now credits the earliest exact Domain
declaration branch token (`video/prod/ad/living`). If a candidate has no exact
declaration but does have a valid contiguous final SID, that candidate alone
falls back to the SID Domain token. Unresolvable candidates receive no Domain
credit; they no longer disable Domain credit for the rest of the G8. Domain
advantages are centered over the eligible candidate subset and retain the
frozen `0.25 * centered / 8` formula. A/B/C credit, bridge lambda, Think
ExactClamp, rewards, runner and sampling contracts are unchanged.

CPU regression tests passed for the formal locator, seven-branch plus one-SID
fallback topology, eligible-subset centering, frozen A/B/C values, bridge
lambda, audit harness and Think ExactClamp.

One GPU zero-step fast preflight used original 8B plus fresh original BATA,
seed `20260816`, and the same eight immutable G8 rollouts. Fingerprint and
reward parity were both `8/8`.

| Group | Eligible / branch / fallback / unresolved | Hier norm | Hier / legacy | Matched cosine | A/B/C parity |
|---:|:---:|---:|---:|---:|:---:|
| 0 | `8 / 7 / 1 / 0` | `0.392624` | `0.0863845` | `0.999925` | true |
| 1 | `8 / 8 / 0 / 0` | `0.147466` | `0.186649` | `0.999949` | true |
| 2 | `8 / 8 / 0 / 0` | n/a (constant reward) | n/a | n/a | true |
| 3 | `8 / 8 / 0 / 0` | `0.156317` | `0.0308458` | `0.999954` | true |
| 4 | `8 / 8 / 0 / 0` | `0.253557` | `0.218078` | `0.999982` | true |
| 5 | `8 / 8 / 0 / 0` | `0.0794926` | `0.104427` | `0.999919` | true |
| 6 | `8 / 8 / 0 / 0` | n/a | n/a | n/a | true (`A+B`, `8/5/0`) |
| 7 | `8 / 8 / 0 / 0` | n/a | n/a | n/a | true (`A`, `5/0/0`) |

Artifact:

- [`../results/gpu_fast_commitment_preflight_g8_seed20260816_20260821.json`](../results/gpu_fast_commitment_preflight_g8_seed20260816_20260821.json)

The formal candidate-level fallback restores Group 0 without weakening the
earlier semantic credit for its seven aligned candidates. All five historical
Domain-only groups now have nonzero real LoRA gradient and matched-support
cosine above `0.9999`. Groups 6/7 preserve A/B/C stage activation and credited
token counts exactly. The trainable-LoRA checksum remained
`b1cfbe7b048ca6c7de8a906ea4419cbe8e339e9974f58d5ed1371f1a471066ad`
before and after. Result: **FAST_PREFLIGHT_PASS**.

No optimizer/scheduler step, parameter update, checkpoint, smoke, pilot, or
formal training occurred.

## 4x A800 optimizer smoke attempt (2026-08-21)

Run `GR-REC-CLAMP-BRIDGE-V1-SMOKE48-20260821` was launched from commit
`6bd86f0f7962f12c73fbee67b1edc6c084c210d2` on four A800 GPUs with the
original 8B base and fresh original BATA adapter. The bounded contract was 48
optimizer steps, `num_iterations=2`, `lr=1e-6`, the frozen `T,T,N,N,N,N`
route schedule, and a single permitted checkpoint at step 48.

The attempt stopped at `0/48` with **SMOKE_FAIL** during the first Think
rollout, before loss, backward, optimizer step, or parameter update. All four
ranks raised:

```text
TypeError: sequence index must be integer, not 'slice'
think_exact_clamp_trainer.py:242
```

The immediate cause is slice assignment into the current TRL advantages log
container, which is a `deque`. This is a training-chain compatibility defect,
not a reward/loss numerical failure. Per the smoke stop contract, no automatic
fix or retry was attempted.

The pre-failure monitor captured four real Think G4 groups (16 candidates):
reward mean `3.0625`, reward distribution `{0: 8, 0.5: 2, 8: 6}`, zero-std
group rate `2/4 = 50%`, invalid rate `0/16`, and completion length
`574..945` (mean `759.5`). ExactClamp high-quality negative clamp count was
zero. No NoThink rollout was reached, so Domain/A/B/C activity, commitment
counts, zero-signal taxonomy, bridge activation, loss, and grad norm are not
observed rather than zero-rate conclusions.

There was no NaN/Inf, OOM, checkpoint, optimizer update, benchmark, or formal
training. Logs remain at
`/data/GRPO/logs/GR-REC-CLAMP-BRIDGE-V1-SMOKE48-20260821.log`; monitor data
remain at `/data/GRPO/runs/GR-REC-CLAMP-BRIDGE-V1-SMOKE48-20260821`.

Structured summary:

- [`../baselines/native_source_domain_r32_v3/grpo/results/optimizer_smoke48_failure_20260821.json`](../baselines/native_source_domain_r32_v3/grpo/results/optimizer_smoke48_failure_20260821.json)

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
- Phase 9 `40d0a447699e1e36547365545400a260ba085d05`: moves the unchanged Domain advantage from final SID
  `<|domain_begin|>` serialization to the preceding natural-language Domain
  decision token; A/B/C and the dead-zero bridge remain unchanged.
- Phase 10 current: records the strictly paired Text-Domain zero-step GPU
  re-audit, including matched-support diagnostics and unchanged-parameter proof.
- Phase 11 current: adds and records the diagnostic-only earliest Domain branch
  point tokenizer and paired zero-step GPU audit. Formal placement is unchanged.

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
3. For Domain-only groups, diagnostic-only legacy population-std advantage on
   exactly the natural-language Domain-token support.
4. Raw and `0.02`-weighted Gold-A bridge gradient, only when the generated
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
The Phase 9/10 result below supersedes its placement conclusion while retaining
this artifact as the immutable paired reference.

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

## Text-Domain paired G8 re-audit (2026-08-21)

The authorized Phase 10 audit reused GPU 0, seed `20260816`, eight real NoThink
G8 groups, temperature/top-p `1.0`, the original 8B base, fresh original BATA
adapter, and the historical per-group generation/backward RNG order. All eight
`group_id`, rollout fingerprint and reward-vector triples matched the saved
SID-Domain audit exactly. Therefore `paired_audit_valid=true`.

Artifacts:

- [`../results/gpu_gradient_scale_audit_text_domain_g8_seed20260816_20260821.json`](../results/gpu_gradient_scale_audit_text_domain_g8_seed20260816_20260821.json)
- [`../results/gpu_gradient_scale_audit_text_domain_paired_comparison_g8_seed20260816_20260821.json`](../results/gpu_gradient_scale_audit_text_domain_paired_comparison_g8_seed20260816_20260821.json)

Domain-only comparison:

| Group | Alignment | Legacy norm | Old SID-Domain norm | New Text-Domain norm | New / legacy | New / old | Matched-support cosine |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | false | `4.5559769` | `0.36677423` | `0` | `0` | `0` | n/a |
| 1 | true | `0.78997797` | `8.13352e-6` | `2.10854e-5` | `2.66912e-5` | `2.59241` | `1.01047` |
| 3 | true | `5.0683770` | `1.83335e-5` | `6.21136e-6` | `1.22551e-6` | `0.338799` | `1.01155` |
| 4 | true | `1.1627312` | `6.36818e-5` | `8.39188e-6` | `7.21739e-6` | `0.131778` | `1.01358` |
| 5 | true | `0.76123321` | `2.08502e-6` | `1.12524e-7` | `1.47818e-7` | `0.0539679` | `1.00857` |

Group 0 contains one real candidate with no natural-language Domain token
before its final SID. The formal group gate therefore disables Domain credit
for the whole G8; no fallback or synthetic alignment was used. Alignment is
valid for the other seven groups.

The proposed desaturation hypothesis is not supported. Across the 39 defined
Text-Domain observations in the five historical Domain-only groups, probability
has median `0.9999998808`, mean `0.9999956964`, and range
`0.9999465971..1.0`. The corresponding SID-Domain aggregate is distorted by
Group 0's malformed candidate (`0.02297536`); for aligned Domain-only Groups
1/3/4/5, both positions remain overwhelmingly saturated. Only Group 1's new
gradient is larger than its old SID-Domain gradient. Groups 3/4/5 are smaller,
and the median new/old ratio over all five Domain-only groups is `0.1317783`.
The audit retains `HIER_GRAD_TOO_SMALL_REVIEW`; no coefficient was changed.

For aligned pure `-.25/0` groups, the diagnostic legacy-on-Text-Domain and
hierarchical Text-Domain gradients are analytically positive scalar multiples.
Measured BF16 cosine values are `1.0086..1.0136`; values slightly above the
mathematical bound are reduction-rounding error and mean approximately `+1`,
not super-alignment. Group 0 has no defined matched-support comparison because
its formal Domain stage is gated off.

Groups 6/7 preserve A/B/C stage activation and credited-token counts exactly:
Group 6 remains A+B with counts `8/5/0`, and Group 7 remains A-only with counts
`5/0/0`. Group 6's total hierarchical norm is also stable
(`0.28579360 -> 0.28578994`). Group 7 retains A credit but also has a valid
Domain stage, so moving Domain support legitimately changes its combined norm
(`0.09160201 -> 0.11375201`); this is not an A/B/C formula change.

The trainable-LoRA checksum was identical before and after:

```text
b1cfbe7b048ca6c7de8a906ea4419cbe8e339e9974f58d5ed1371f1a471066ad
```

No optimizer or scheduler step occurred, no parameter changed, no checkpoint
was written, and no smoke, pilot, or formal training was started.

## Earliest Domain branch-point paired audit (2026-08-21)

The real tokenizer gives the following complete declaration sequences:

| Domain | Declaration token ids |
|---|---|
| video | `75882 20002 104044 99729 9370 87140 18830 25 220` |
| prod | `75882 20002 104044 72651 34187 45943 25 220` |
| ad | `75882 20002 104044 112429 101927 18830 25 220` |
| living | `75882 20002 104044 104181 75437 100541 34187 107206 25 220` |

Their longest common token prefix is `[75882, 20002, 104044]`, decoded as
`[该, 用户, 最近]`. The first divergent declaration-token index is `3`:
video `喜欢/99729`, prod `点击/72651`, ad `感兴趣的/112429`, and living
`首次/104181`. Candidate alignment requires one exact full declaration between
the final `</think>` and final contiguous SID, with declaration and SID domains
equal. There is no substring or SID fallback.

Artifacts:

- [`../results/earliest_domain_branch_tokenizer_audit_20260821.json`](../results/earliest_domain_branch_tokenizer_audit_20260821.json)
- [`../results/gpu_earliest_domain_branch_audit_g8_seed20260816_20260821.json`](../results/gpu_earliest_domain_branch_audit_g8_seed20260816_20260821.json)
- [`../results/gpu_earliest_domain_branch_paired_comparison_g8_seed20260816_20260821.json`](../results/gpu_earliest_domain_branch_paired_comparison_g8_seed20260816_20260821.json)

The completed audit used physical GPU 1, original 8B plus fresh original BATA,
seed `20260816`, and the same eight immutable G8 rollouts. Fingerprint and
reward parity are both `8/8`. An initial pre-result harness attempt was stopped
after one group when a reference-field adapter bug prevented Domain-only
gradient execution; it wrote no result JSON and the corrected CPU test proves
the expected historical indices `[0,1,3,4,5]` before the completed run.

Across the five historical Domain-only groups, 39 defined branch probabilities
have median `0.4584907`, mean `0.4633573`, and range
`0.0177775..0.7987047`. The same candidates' noun probabilities have median
`0.99999988`; SID probability median is `0.99998623`. The branch is therefore
materially less saturated.

| Group | Alignment | SID norm | Noun norm | Branch norm | Branch / legacy | Matched cosine |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | false (7/8) | `0.366774` | `0` | `0` | `0` | n/a |
| 1 | true | `8.13352e-6` | `2.10854e-5` | `0.147565` | `0.186797` | `0.999948` |
| 3 | true | `1.83335e-5` | `6.21136e-6` | `0.156431` | `0.0308641` | `0.999952` |
| 4 | true | `6.36818e-5` | `8.39188e-6` | `0.253551` | `0.218065` | `0.999983` |
| 5 | true | `2.08502e-6` | `1.12524e-7` | `0.0794869` | `0.104419` | `0.999920` |

Groups 1/3/4/5 provide strong evidence that the earliest branch point recovers
real LoRA gradient with the expected matched-support direction. Group 0 still
contains one direct-SID candidate with no exact declaration; the conservative
group gate therefore disables all branch credit and does not improve over the
noun placement for that group. Because only four of five Domain-only groups
have reliable full-G8 alignment, the required three-way verdict is
`BRANCH_POINT_AMBIGUOUS`, not a formal placement recommendation.

The trainable-LoRA checksum is identical before and after:

```text
b1cfbe7b048ca6c7de8a906ea4419cbe8e339e9974f58d5ed1371f1a471066ad
```

No optimizer/scheduler step, parameter update, checkpoint, smoke, pilot, or
formal training occurred. Formal Domain placement and all coefficients remain
unchanged.

## Frozen runner contract

The runner is not restructured. Prefix remains `GR-REC-CLAMP-BRIDGE-V1-`,
formal max steps remain 1500, fresh BATA and same-run-only resume remain fixed,
and the existing seeds, route schedule, G4/G8, `num_iterations=2`, PPO
hyperparameters, sampling and Beam32 contracts remain unchanged.

## Final status

**EARLIEST DOMAIN BRANCH POINT GPU PAIRED DIAGNOSTIC COMPLETE**

**ZERO PARAMETER UPDATE / TRAINING NOT STARTED**
