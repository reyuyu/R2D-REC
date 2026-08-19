# GR_REC advantage formula review

## Scope and decision

**Recommendation: PREFER CENTERED EXACT-CLAMP FOR GPU**

This is a CPU-only counterfactual over saved rewards. GPU was not used, no 8B
model was loaded, and training/smoke was not started. The existing ExactFloor
v1 implementation and commit `98050bd3ebe9ea41a4ebbaed208d5a1dbb10dead`
remain unchanged.

The audit reconstructs 3,192 groups and 19,152 candidates:

- 104 GR_REC_v1 fixed-Probe groups. This is repeated fixed-Probe evidence, not
  full GR_REC training coverage.
- 3,088 DSR-Simple full-forensic groups. Only saved `primary_reward` is used;
  DSR auxiliary rewards are excluded.

`sum(abs(advantage))` is an **advantage-mass proxy only**, not a gradient norm.
Real updates also depend on token length, policy-logprob gradients, PPO ratio
and clipping.

## Formulas

```text
Current:     (R - group_mean) / (group_population_std + 1e-4)
ExactFloor: (R - min(group_mean, 8)) / 8
ExactClamp: raw = (R - group_mean) / 8
            raw < 0 is clamped to 0 only when R >= 8
```

## Exact archetypes

| Rewards | Current | ExactFloor | Centered Exact-Clamp |
|---|---|---|---|
| `[0,0,0,0]` | `[0,0,0,0]` | `[0,0,0,0]` | `[0,0,0,0]` |
| `[0,0,0,.5]` | `[-.577084,-.577084,-.577084,1.731251]` | `[-.015625,-.015625,-.015625,.046875]` | same as ExactFloor |
| `[0,0,0,2]` | `[-.577284,-.577284,-.577284,1.731851]` | `[-.0625,-.0625,-.0625,.1875]` | same as ExactFloor |
| `[0,0,0,8]` | `[-.577334,-.577334,-.577334,1.732001]` | `[-.25,-.25,-.25,.75]` | same as ExactFloor |
| `[8,8,8,9]` | `[-.577217,-.577217,-.577217,1.731651]` | `[0,0,0,.125]` | `[0,0,0,.09375]` |
| `[8,8,8,12]` | `[-.577317,-.577317,-.577317,1.731951]` | `[0,0,0,.5]` | `[0,0,0,.375]` |
| `[8,2,3,9]` | `[.821968,-1.150755,-.821968,1.150755]` | `[.3125,-.4375,-.3125,.4375]` | same as ExactFloor |
| `[8,8,8,8]` | `[0,0,0,0]` | `[0,0,0,0]` | `[0,0,0,0]` |
| `[12,12,12,12]` | `[0,0,0,0]` | `[.5,.5,.5,.5]` | `[0,0,0,0]` |
| `[12,12,12,14]` | `[-.577284,-.577284,-.577284,1.731851]` | `[.5,.5,.5,.75]` | `[0,0,0,.1875]` |

## Reward-level evidence

Rates and means are candidate weighted across both saved sources.

| Rule / reward | N | mean A | mean abs(A) | positive | negative | zero |
|---|---:|---:|---:|---:|---:|---:|
| Current / 0 | 10,918 | -0.3172 | 0.3487 | 2.61% | 53.38% | 44.01% |
| Current / .5 | 3,695 | 0.5877 | 0.8331 | 60.11% | 21.62% | 18.27% |
| Current / 2 | 748 | 0.7823 | 1.0094 | 58.02% | 14.71% | 27.27% |
| Current / 8 | 856 | 0.6436 | 0.8287 | 52.57% | 10.63% | 36.80% |
| Current / >8 | 1,323 | 0.2762 | 0.6286 | 51.70% | 20.48% | 27.82% |
| ExactFloor / 0 | 10,918 | -0.0294 | 0.0299 | 2.61% | 53.38% | 44.01% |
| ExactFloor / .5 | 3,695 | -0.0098 | 0.0492 | 60.11% | 21.62% | 18.27% |
| ExactFloor / 2 | 748 | 0.0586 | 0.1062 | 58.02% | 14.71% | 27.27% |
| ExactFloor / 8 | 856 | 0.2841 | 0.2841 | 52.57% | **0%** | 47.43% |
| ExactFloor / >8 | 1,323 | 0.3391 | 0.3391 | 100% | **0%** | 0% |
| ExactClamp / 0 | 10,918 | -0.0295 | 0.0299 | 2.61% | 53.38% | 44.01% |
| ExactClamp / .5 | 3,695 | -0.0098 | 0.0492 | 60.11% | 21.62% | 18.27% |
| ExactClamp / 2 | 748 | 0.0585 | 0.1062 | 58.02% | 14.71% | 27.27% |
| ExactClamp / 8 | 856 | 0.2841 | 0.2841 | 52.57% | **0%** | 47.43% |
| ExactClamp / >8 | 1,323 | 0.1200 | 0.1200 | 51.70% | **0%** | 48.30% |

Current therefore has both quality-hierarchy compression and high-quality
negative advantages. Its mean positive advantages by level are `.5=1.1818`,
`2=1.5440`, `8=1.4003`, `>8=.8751`. ExactClamp changes these to `.0328`,
`.1420`, `.5404`, `.2322`: high rewards are much stronger than A-only, but
cross-group reward buckets are not globally monotonic because every value is
still contextualized by its own group mean.

## Advantage-mass proxy

| Metric | Current | ExactFloor | ExactClamp |
|---|---:|---:|---:|
| Total `sum abs(A)` | 10,303.83 | 1,381.84 | 1,093.88 |
| R>=8 share | 14.96% | **50.06%** | **36.75%** |
| R>8 share | 8.07% | **32.46%** | **14.52%** |
| R>8 positive mean / R=.5 positive mean | 0.74x | 10.34x | 7.08x |
| MULTI_EXACT mean mass / A_ONLY mean mass | 0.97x | 13.41x | 13.26x |

Raw totals must not be interpreted as gradient magnitudes across formulas.
Shares and within-formula ratios show that ExactFloor heavily concentrates its
proxy mass on `R>=8`; ExactClamp materially reduces, but does not eliminate,
that concentration.

## Route, domain and density

| Mass share | Current | ExactFloor | ExactClamp |
|---|---:|---:|---:|
| Think | 31.11% | 58.66% | 47.78% |
| video | 45.08% | 39.87% | 35.06% |
| gold_count 5+ | 49.30% | 49.21% | 41.87% |

Relative to ExactFloor, ExactClamp reduces Think by 10.88 points, video by 4.81
points, and gold-count 5+ by 7.33 points. ExactFloor does not increase every
domain/density share relative to Current, so the evidence supports a narrower
claim: it creates strong high-reward absolute reinforcement, concentrated in
Think, while ExactClamp reduces the resulting easy/dense exposure.

## Equal-high behavior

| Group condition | Count | Current mean sum abs(A) | ExactFloor | ExactClamp |
|---|---:|---:|---:|---:|
| all reward >8 | 220 | 2.0515 | 1.3456 | 0.0887 |
| all reward >=8 | 346 | 1.8013 | 0.9045 | 0.0815 |
| group mean >8 | 282 | 2.3749 | 1.2149 | 0.1938 |
| all equal and reward >=8 | 170 | **0** | **0.5968** | **0** |

The 170 equal-high groups prove the ExactFloor risk is realized historically,
not merely theoretical. ExactFloor assigns them 101.4512 total proxy mass;
Current and ExactClamp assign exactly zero.

## Pattern-level evidence

| Pattern | Groups | Current mean mass | ExactFloor | ExactClamp |
|---|---:|---:|---:|---:|
| ALL_ZERO | 731 | 0 | 0 | 0 |
| A_ONLY | 1,087 | 5.0804 | 0.1424 | 0.1424 |
| AB_SIGNAL | 334 | 4.4487 | 0.3877 | 0.3877 |
| SINGLE_EXACT | 142 | 4.8394 | 1.6368 | 1.6368 |
| EXACT_SATURATED | 78 | 0 | 0 | 0 |
| MULTI_EXACT | 137 | 4.9296 | 1.9097 | 1.8884 |
| MIXED_HIGH | 403 | 2.7442 | 1.4213 | 0.7140 |
| OTHER | 280 | 2.9540 | 0.1096 | 0.1096 |

## Risk judgment

1. **Current GRPO:** not preferred. It compresses absolute quality and gives
   negative advantage to 10.63% of `R=8` and 20.48% of `R>8` candidates.
2. **ExactFloor:** fixes high-quality negatives and restores strong hierarchy,
   but gives all 170 equal-high groups positive mass and moves 50.06% of proxy
   mass to `R>=8`. This is meaningful easy/high-quality absolute reinforcement.
3. **Centered Exact-Clamp:** also makes both high-quality negative rates exactly
   zero, keeps equal-high mass exactly zero, retains a 7.08x high-vs-A positive
   signal and a 13.26x MULTI_EXACT-vs-A_ONLY group signal, while reducing the
   high/Think/video/dense concentration relative to ExactFloor.

Centered Exact-Clamp is therefore the better isolated next GPU ablation. The
residual caveat is that it is not a global cross-group monotonic ranking rule:
group context can make aggregate `R>8` positive advantages smaller than `R=8`.
That limitation should be monitored, not silently treated as solved.

## Final status

**PREFER CENTERED EXACT-CLAMP FOR GPU**

**GPU NOT USED / TRAINING NOT STARTED**
