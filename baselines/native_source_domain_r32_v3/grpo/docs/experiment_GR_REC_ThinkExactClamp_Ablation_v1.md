# GR_REC_ThinkExactClamp_Ablation_v1

## Status

**CPU IMPLEMENTATION READY / GPU VALIDATION PENDING**

**GPU NOT USED / TRAINING NOT STARTED**

This implementation was created from the final GitHub `main` observed during
the work, `9a5b2ea93a2633898dad14ff90bba69ba4bd2641`. Its change after the initially
specified `750102a` is confined to `user_grpo` runtime-benchmark artifacts; the
recommendation GR_REC path is unchanged. No model was loaded and no CUDA
smoke, `torchrun`, pilot, or formal training was started.

## Isolation

- Parent: original BATA adapter and GR_REC_v1 training contract.
- New sibling: `ablations/gr_rec_think_exact_clamp_v1/`.
- The only training variable is the **Think advantage rule**.
- NoThink is the unchanged GR_REC_v1 population-std rule.
- Production GR_REC, DSR, DSR-Simple, ExactFloor and its formula-review history
  are not modified.

The research goal is to preserve existing correct modes, not create artificial
diversity. The experiment asks whether Current GRPO causes good-mode
cannibalization and overweights weak A-prefix winners.

## Formula

Think G4 only:

```text
raw_i = (reward_i - group_mean) / 8.0
advantage_i = 0                    if reward_i >= 8 and raw_i < 0
advantage_i = raw_i                otherwise
```

NoThink G8 remains exactly:

```text
(reward_i - group_mean) / (group_population_std + 1e-4)
```

NoThink output is returned directly from `RecGRPOTrainer`; it is not recomputed
or replaced by the child trainer. Population std continues to use
`correction=0`.

## Frozen contracts

Think remains G4, temperature .9, top-p .95 and Beam32. NoThink remains G8,
temperature 1 and top-p 1. Primary rewards, geometric multi-positive reward,
LR `1e-6`, beta 0, epsilon .2, `loss_type=grpo`, `num_iterations=2`, route
weight, sampler, seed, data order, Probe, CoT stopping, PPO clipping and old
log-prob behavior remain inherited from GR_REC_v1.

No S_N, Grounded-N/Coverage reward, D_A, entropy, prefix/length/diversity
auxiliary, zero-gradient rescue, NoThink rescue, resampling, or curriculum is
added. Raw N, Grounded N and Coverage are diagnostics only and never enter loss.

## Exact Think contracts

| Rewards | Think ExactClamp advantages |
|---|---|
| `[0,0,0,0]` | `[0,0,0,0]` |
| `[0,0,0,.5]` | `[-.015625,-.015625,-.015625,.046875]` |
| `[0,0,0,2]` | `[-.0625,-.0625,-.0625,.1875]` |
| `[0,0,0,8]` | `[-.25,-.25,-.25,.75]` |
| `[8,8,8,9]` | `[0,0,0,.09375]` |
| `[8,8,8,12]` | `[0,0,0,.375]` |
| `[8,2,3,9]` | `[.3125,-.4375,-.3125,.4375]` |
| `[8,8,8,8]` | `[0,0,0,0]` |
| `[12,12,12,12]` | `[0,0,0,0]` |
| `[12,12,12,14]` | `[0,0,0,.1875]` |

## CPU verification

Tests cover all archetypes, two independent Think groups, exact G4 enforcement,
dtype/device, finite output, equal-high zero, and zero negative rate for reward
at least 8. NoThink parity uses several G8 reward groups, including
`[0,0,0,.5,0,0,0,0]`, and asserts that the child returns the same parent output
object and the same advantage tensor object. Parent reward output is likewise
returned unchanged for NoThink.

## Historical Think-only counterfactual

The audit uses saved rewards only: 52 GR_REC fixed-Probe Think groups plus 1,544
DSR-Simple full-forensic Think groups, totaling 1,596 groups and 6,384
candidates. DSR-Simple auxiliary rewards are excluded.

| Reward | N | Current negative | Clamp negative | Current mean positive A | Clamp mean positive A |
|---|---:|---:|---:|---:|---:|
| 0 | 1,706 | 37.16% | 37.16% | n/a | n/a |
| .5 | 1,098 | 25.32% | 25.32% | .9160 | .0265 |
| 2 | 391 | 24.81% | 24.81% | .8918 | .0911 |
| 8 | 607 | **14.99%** | **0%** | .9027 | .3852 |
| >8 | 1,323 | **20.48%** | **0%** | .8751 | .2322 |

The Current positive winner scale is nearly flat across `.5`, `2`, `8` and
`>8`, supporting the hierarchy-compression hypothesis. Clamp restores a clear
separation between weak A-only and high-quality rewards without giving losing
high-quality modes negative advantages.

Advantage mass is a proxy only, not a gradient norm. Real gradients also depend
on token length, policy-logprob gradients, PPO ratio and clipping.

| Metric | Current | Think ExactClamp |
|---|---:|---:|
| MULTI_EXACT mean mass / A_ONLY | 1.56x | 15.12x |
| video mass share | 48.21% | 35.38% |
| gold_count 5+ mass share | 52.43% | 41.04% |

Pattern counts include 268 ALL_ZERO, 369 A_ONLY, 127 AB_SIGNAL, 38
SINGLE_EXACT, 78 EXACT_SATURATED, 88 MULTI_EXACT and 403 MIXED_HIGH groups.
Full reward, pattern, domain and density statistics are stored in
`think_counterfactual.json`; `think_counterfactual_groups.csv` retains every
group and both advantage vectors.

## Passive monitor

The trainer writes a dedicated per-rank `rankN-think-exact-clamp.jsonl` using
only objects already produced by the parent rollout. Each local G4 group records:

- four rewards, Exact (`R>=8`) and multi-positive (`R>8`) candidate counts;
- high-quality negative rate and mean absolute advantage;
- existing Beam Exact/AB/A hit counts;
- distinct Exact gold SIDs already covered across local G4, derived from saved
  Beam SIDs without another Beam or collective;
- completion length, Raw N, Grounded N, Coverage and parser success per candidate.

These fields are diagnostic-only. The monitor adds no generation, Beam,
forward, collective or synchronization.

## Formal runner and future plan

`run_think_exact_clamp_train.py` requires:

- run ID prefix `GR-REC-THINK-EXACT-CLAMP-V1-`;
- explicit `--max-steps` between 1 and 1500;
- initial use of the original BATA adapter;
- resume only from a checkpoint whose parent directory is the same run ID.

If GPU execution is separately authorized, retain checkpoints at 600, 800,
1000, 1200, 1400 and 1500, with external evaluation prioritizing 1000 and
1500. This is a plan only; none were created in this CPU implementation turn.

## Known limits

- ALL_ZERO groups remain zero-gradient.
- `[8,8,8,8]` Exact-saturated groups still have no exploration signal.
- Clamp protects correct modes that already exist; it does not actively find a
  new Gold mode or guarantee broader correct-mode diversity.
- Cross-group reward buckets remain contextualized by their own means, so the
  formula is not a global ranking guarantee.

## Final status

**CPU IMPLEMENTATION READY / GPU VALIDATION PENDING**
