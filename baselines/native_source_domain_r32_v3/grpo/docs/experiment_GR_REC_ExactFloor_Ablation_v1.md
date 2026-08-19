# GR_REC_ExactFloor_Ablation_v1

## Status

**CPU IMPLEMENTATION READY / GPU VALIDATION PENDING**

No GPU smoke, model load, `torchrun`, training, or pilot was run during this
implementation. This branch is not approved as `READY FOR TRAINING`.

## Isolation and hypothesis

- Parent model: original BATA adapter.
- Algorithm baseline: `GR_REC_v1` from GitHub `main` at `646f5119e9a7dcb2cb3da79633f5f8250cec34c2`.
- The implementation lives only in `ablations/gr_rec_exact_floor_v1/`.
- It does not inherit DSR v1, DSR-Simple, or any GRPO checkpoint.
- Existing GR_REC, DSR and production files are unchanged.

The hypothesis is that per-group standard deviation normalization removes the
absolute reward hierarchy. For example, current GRPO makes `[0,0,0,.5]` and
`[8,8,8,12]` nearly identical after normalization. It also gives the three
Exact candidates in `[8,8,8,9]` negative advantages.

ExactFloor tests only two proposed corrections:

1. Preserve the magnitude ordering between A-only, AB, Exact, and rewards above Exact.
2. Do not punish an `R=8` Exact candidate because another candidate in its group scores higher.

It does not address all-zero groups, `[8,8,8,8]` exploration, CoT format,
grounding, or any other zero-gradient condition.

## Exact math

For both Think `G=4` and NoThink `G=8`:

```text
baseline_g = min(mean(group_rewards), 8.0)
advantage_i = (reward_i - baseline_g) / 8.0
```

There is no group std, batch std, or other dynamic scale. The pure function is
`exact_floor_advantages(rewards, group_size, exact_floor=8.0, fixed_scale=8.0)`.
The trainer subclasses `RecGRPOTrainer`, delegates reward/generation/Beam/sampler
work to the parent, and replaces only the returned advantage tensor. It adds no
forward, generate, Beam, collective, or CUDA synchronization.

## Frozen contracts

Think remains G4, temperature 0.9, top-p 0.95 and Beam32. NoThink remains G8,
temperature 1.0 and top-p 1.0. Reward values, multi-positive geometric decay,
route weight, sampler, dataset, seed, Probe, CoT stopping, LR `1e-6`, beta `0`,
epsilon `0.2`, `loss_type=grpo`, `num_iterations=2`, PPO ratio and clipping are unchanged.

No S_N, Grounded-N/Coverage, D_A, prefix/entropy/length/diversity reward,
unlikelihood rescue, dynamic resampling, curriculum, or zero-gradient rescue is added.

## A-I exact contracts

| Case | Rewards | ExactFloor advantages |
|---|---|---|
| A | `[0,0,0,0]` | `[0,0,0,0]` |
| B | `[0,0,0,.5]` | `[-.015625,-.015625,-.015625,.046875]` |
| C | `[0,0,0,2]` | `[-.0625,-.0625,-.0625,.1875]` |
| D | `[0,0,0,8]` | `[-.25,-.25,-.25,.75]` |
| E | `[8,8,8,9]` | `[0,0,0,.125]` |
| F | `[8,8,8,12]` | `[0,0,0,.5]` |
| G | `[8,2,3,9]` | `[.3125,-.4375,-.3125,.4375]` |
| H | `[8,8,8,8]` | `[0,0,0,0]` |
| I | `[12,12,12,12]` | `[.5,.5,.5,.5]` |

The positive mean in case I is intentional. It is also the main design risk.

## Historical CPU counterfactual

The audit uses saved rewards only. It covers 3,192 reconstructed groups:

- `GR_REC_v1_probe`: 104 G4/G8 fixed-Probe groups. This is not full training coverage.
- `DSR_Simple_training`: 3,088 groups from the complete saved
  `simple_forensic.jsonl` (1,544 Think and 1,544 NoThink). Only `primary_reward`
  is used; DSR-Simple auxiliary values are excluded.

Exact comparison for the motivating patterns (`1e-4` is retained in the current rule):

| Rewards | Current GRPO | ExactFloor |
|---|---|---|
| `[0,0,0,.5]` | `[-.577084,-.577084,-.577084,1.731251]` | `[-.015625,-.015625,-.015625,.046875]` |
| `[0,0,0,2]` | `[-.577284,-.577284,-.577284,1.731851]` | `[-.0625,-.0625,-.0625,.1875]` |
| `[0,0,0,8]` | `[-.577334,-.577334,-.577334,1.732001]` | `[-.25,-.25,-.25,.75]` |
| `[8,8,8,9]` | `[-.577217,-.577217,-.577217,1.731651]` | `[0,0,0,.125]` |
| `[8,8,8,12]` | `[-.577317,-.577317,-.577317,1.731951]` | `[0,0,0,.5]` |

High-quality negative rates:

| Source | Metric | Current | ExactFloor |
|---|---|---:|---:|
| GR_REC Probe | `P(A<0 | R=8)` | 47.83% | 0% |
| GR_REC Probe | `P(A<0 | R>8)` | 23.08% | 0% |
| DSR-Simple training | `P(A<0 | R=8)` | 9.60% | 0% |
| DSR-Simple training | `P(A<0 | R>8)` | 20.35% | 0% |

The complete reward-level statistics, pattern counts and strata are committed as
`ablations/gr_rec_exact_floor_v1/historical_audit.json` and can be regenerated
with `audit_historical_rewards.py`.

## Prevalence and stratification

| Source | Groups | mean > 8 | all candidates > 8 |
|---|---:|---:|---:|
| GR_REC fixed Probe | 104 | 15.38% | 6.73% |
| DSR-Simple full training forensic | 3,088 | 8.61% | 6.90% |

In the full training forensic, the all-`>8` rate is 11.31% for video versus
2.36% for prod, 3.97% for living and 6.57% for ad. By target density it is
2.77% for gold count 1-2, 6.90% for 3-4, and 12.00% for 5+.
The event is confined to Think: 213/1,544 Think groups (13.80%) have every
candidate above 8, while NoThink has 0/1,544 because its reward ceiling is 8.

Therefore the all-positive behavior is not negligible. ExactFloor may give
easy/dense groups extra effective training weight, especially video and 5+
gold groups. This is a pilot monitoring requirement, not evidence to alter the
authorized formula in this implementation.

## CPU verification

`test_exact_floor.py` covers all A-I outputs, G4/G8, independent multi-group
baselines, finite results, dtype/device preservation, trainer inheritance and
frozen route constants. Existing reward, trainer, formal runner and monitor CPU
tests are also run before commit. The historical audit asserts
`P(A<0 | R=8) = 0` for ExactFloor.

Future GPU validation should monitor reward patterns, per-level advantages,
high-quality negative rate, zero std and diagnostic-only Raw N, Grounded N and
CoT length. It must use a separately authorized run from the original BATA
adapter, never a GRPO/DSR checkpoint.
