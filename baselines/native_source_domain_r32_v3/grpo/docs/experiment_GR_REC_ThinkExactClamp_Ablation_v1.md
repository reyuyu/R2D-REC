# GR_REC_ThinkExactClamp_Ablation_v1

## Status and phase history

**CPU IMPLEMENTATION READY / GPU GRADIENT AUDIT PENDING**

**GPU NOT USED / TRAINING NOT STARTED**

- Phase 1: commit `917d3d3a9e53db2e80bf425b435c597bb210b804`, CPU-only
  Think Centered Exact-Clamp implementation.
- Phase 2: the current commit, Think Centered Exact-Clamp plus NoThink Minimal
  Hierarchical Teacher Bridge. This is the formal future GPU candidate.

The branch is `ablation/gr-rec-think-exact-clamp-v1`. Phase 2 continues the
same implementation directory; no combined sibling exists. The parent model is
always the **fresh original BATA adapter**. Phase 1 is a code history point, not
a model checkpoint and not a resume source.

## Isolation and research questions

The experiment retains all GR_REC_v1 primary reward, sampling, PPO and schedule
contracts. It changes only:

1. Think advantages, to avoid giving an already Exact candidate a negative
   advantage merely because a stronger candidate is in the same G4.
2. NoThink loss, by adding a small, strictly gated token-local teacher loss when
   primary G8 has no A signal or has collapsed onto one correct A.

The Think question is preservation of existing correct modes. The NoThink
question is bootstrap of zero-signal and strict A-collapse hard groups. The
bridge is **not a GRPO rollout**: no Gold SID is injected as a sampled candidate.

## Think: Centered Exact-Clamp

For each Think G4:

```text
mean_g = mean(reward)
raw_i = (reward_i - mean_g) / 8.0
advantage_i = 0       if reward_i >= 8 and raw_i < 0
advantage_i = raw_i   otherwise
```

The ten exact regressions remain:

| Rewards | Advantages |
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

Thus `P(advantage < 0 | reward >= 8) = 0` by construction. Think G4 isolation,
dtype/device behavior and all archetypes remain CPU-regression tested.

## NoThink primary parity

NoThink primary advantage remains exactly:

```text
(reward_i - group_mean) / (group_population_std(correction=0) + 1e-4)
```

Primary reward, advantage, per-token log-probability, PPO ratio, clipping,
approximate KL, route multiplier and GRPO reduction all execute in the unchanged
parent `RecGRPOTrainer._compute_loss`. The child first receives that primary loss.
With `lambda_bridge=0`, it returns the exact same tensor object before any teacher
forward; the CPU test also verifies identical primary gradients. The bridge does
not modify the sampled completion batch or primary advantage.

## NoThink bridge planner

`plan_nothink_bridge()` is deterministic, CPU-only and model/tokenizer-free.
Branches are mutually exclusive in this order:

1. `dead_zero_a_bridge`: only exact rewards `[0]*8`. It teaches the uniform set
   of unique target-domain Gold A tokens. It never teaches B or C.
2. `a_collapse_ab_bridge`: unique Gold A at least 3; all eight predicted SIDs
   valid and target-domain; one predicted A; that A is Gold; max reward at most
   `.5`; and at least one reward equals `.5`. It teaches missing Gold A plus the
   unique Gold B tokens under the current A.
3. `off`.

Wrong A plus exact `[0]*8` takes Branch A by priority. Any invalid SID,
wrong-domain candidate, multiple predicted A values, reward 2, or reward 8 turns
Branch B off. Planner cases A-J and the priority rule are CPU-tested.

## Teacher loss

For a target set `T`, the bridge uses deduplicated uniform mean CE:

```text
L(T) = mean_t[-log P(t)]
```

It does not use `-log sum P(t)` and target count does not multiply group loss.
Synthetic CPU tests cover single/multiple A, duplicate A, multiple B, exact
mean-not-sum behavior, and verify that a gradient step from the test state raises
every teacher target probability.

Queries read next-token logits only:

- A query: rendered prompt + correct target-domain token.
- B query: rendered prompt + correct target-domain token + current A token.

Branch A uses `L_bridge=L_A`. Branch B uses
`L_bridge=.5*L_A_missing + .5*L_B_current`. No C or full-SID teacher exists.
The NoThink total is `L_primary + .02*L_bridge`; `.02` and `.5/.5` are frozen v1
contract values pending gradient calibration.

## Tokenizer contract

The CPU tokenizer-only audit loaded no model weights. It verified all four
domain tokens and every `<s_a_0>` through `<s_a_8191>` and `<s_b_0>` through
`<s_b_8191>`: 16,388 tokens checked, zero failures, all exactly one token.
Runtime revalidates every concrete domain/A/B teacher token and fails closed on
any violation. Results are in `bridge_tokenizer_audit.json`.

## DDP and `num_iterations=2`

The planner runs after one G8 rollout and its immutable plan is cached. The same
plan is reused for both optimizer steps; teacher logits are recomputed from the
current policy on each step. There is no second generation, sample, or Beam.

If no global group is active, all ranks skip the extra forward. If any group is
active, every rank executes one compatible two-row forward (A row and B row).
Inactive rows/groups have zero loss weight. The per-rank factor
`world_size / (global_group_count * ranks_for_group)` makes the DDP mean equal a
mean over global groups. The configured 4-rank/2-group layout has factor 1; a
teacher is not repeated eight times for the eight candidates.

## CPU historical gate audit

The audit replayed 101 complete NoThink G8 observations: Simple full fixed Probe
(52), Simple full global traces (38), DSR pilot fixed Probe (8), and DSR pilot
global traces (3). `simple_forensic.jsonl` is explicitly **UNAVAILABLE** for this
purpose because it contains Think candidates only.

| Stratum | N | Dead-zero A | A-collapse AB | Off |
|---|---:|---:|---:|---:|
| Overall | 101 | 33 (32.67%) | 0 (0%) | 68 (67.33%) |
| ad | 33 | 18 (54.55%) | 0 | 15 |
| prod | 22 | 9 (40.91%) | 0 | 13 |
| living | 19 | 6 (31.58%) | 0 | 13 |
| video | 27 | 0 | 0 | 27 |
| Gold count 1-2 | 65 | 31 (47.69%) | 0 | 34 |
| Gold count 3-4 | 9 | 2 (22.22%) | 0 | 7 |
| Gold count 5+ | 27 | 0 | 0 | 27 |

Dead-zero is common and concentrated in ad/prod/living and sparse Gold sets;
video is lower in this sample (0/27). No historical observation passes the very
strict A-collapse gate, so its empirical rate and exposure are zero rather than
estimated. These are repeated observational checkpoints, not independent user
samples, and should not be interpreted as population prevalence.

Branch A exposure is mean 1.91 unique A targets, median 2, p90 2, max 2
(`n=33`). Branch B missing-A and current-B exposure are unavailable (`n=0`)
because it never fired. Full strata and source rates are stored in
`nothink_bridge_history_audit.json`; per-observation replay is retained in
`nothink_bridge_history_observations.jsonl`.

## Monitoring

The dedicated per-rank bridge JSONL records rollout plans and per-policy-step
losses without completion text. Fields include branch/active/lambda, unique Gold
and predicted A counts, current A and Gold membership, target counts, raw A/B and
total bridge loss, weighted bridge loss, primary/total loss, primary zero-std and
all-zero flags, Gold-A/Gold-AB/Exact candidate hit rates, wrong-domain rate and
valid-SID rate. Think retains its Phase 1 ExactClamp monitor.

## Runner and frozen schedule

The same `run_think_exact_clamp_train.py` now requires prefix
`GR-REC-CLAMP-BRIDGE-V1-`, `--max-steps` in `[1,1500]`, fresh original BATA, and
same-run-ID-only resume. Frozen future training values remain: seed 20260816,
Probe seed 20260818, the same four fixed Probes and order, Think G4, NoThink G8,
`num_iterations=2`, `steps_per_generation=1`, LR `1e-6`, beta 0, epsilon `.2`,
`loss_type=grpo`, route temperatures/top-p and Beam32.

The preregistered future checkpoints are 600/800/1000/1200/1400/1500; external
benchmark priority is 1000 and 1500. No checkpoint was created in this turn.

## Required GPU gradient audit

Before any smoke or training, run a real 8B + original BATA zero-optimizer-step
audit on: normal primary-signal NoThink, dead-zero A bridge, and A-collapse A+B
bridge groups. Compare `||lambda_bridge*g_bridge||` with median normal NoThink
primary gradient norm. The provisional target is 5%-15%; above 20% blocks
training and requires lambda review. This is a calibration target, not a theory
guarantee.

## Known risks

1. The teacher bridge introduces supervised bias.
2. Excessive lambda could cause teacher takeover; real gradient scale is unknown.
3. Dead-zero A support does not guarantee B or C learning.
4. The strict A-collapse gate may miss A plateaus and fired 0/101 historically.
5. C plateaus are not treated.
6. High-quality zero-gradient `[8]*8` is not treated.
7. Think all-zero is not treated.
8. The bridge can reduce the proportion of purely on-policy exploration.
9. Uniform CE assumes all unique Gold A targets are equally appropriate hints.
10. Real 8B+BATA gradient scale and memory behavior remain unverified.

## Final status

**CPU IMPLEMENTATION READY / GPU GRADIENT AUDIT PENDING**

**GPU NOT USED / TRAINING NOT STARTED**
