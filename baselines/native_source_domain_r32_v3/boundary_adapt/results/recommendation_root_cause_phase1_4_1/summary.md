# Recommendation Root-Cause Phase 1.4.1

CPU-only adjudication of existing Phase1.2 and Phase1.4 predictions. No model inference or generation was run.

## Paired No-Think result

| Metric | Beta | Step900 | Paired delta | 95% bootstrap CI | P(delta>0) |
|---|---:|---:|---:|---:|---:|
| mrr | 0.138056 | 0.117838 | 0.020218 | [-0.01337243221478769, 0.061737302986764185] | 0.8582 |
| hit32 | 0.425000 | 0.350000 | 0.075000 | [0.0, 0.175] | 0.9565 |
| ab32 | 0.475000 | 0.400000 | 0.075000 | [0.0, 0.175] | 0.9565 |
| a32 | 0.650000 | 0.625000 | 0.025000 | [0.0, 0.075] | 0.6417 |
| history_fraction | 0.326562 | 0.269531 | 0.057031 | [0.0359375, 0.08046875] | 1.0000 |

## Natural within-domain weighting

| Domain | Beta MRR | Step900 MRR | Delta | CI | Coverage | Uncertainty |
|---|---:|---:|---:|---:|---:|---|
| video | 0.107663 | 0.107663 | 0.000000 | [0.0, 0.0] | 0.983 | DESCRIPTIVE_BOOTSTRAP |
| prod | 0.016143 | 0.010762 | 0.005381 | [-0.0005979073243647238, 0.01674140508221226] | 1.000 | UNDERRESOLVED_STRATA |
| ad | 0.059917 | 0.045692 | 0.014225 | [-0.005762167125803498, 0.04368686868686869] | 1.000 | UNDERRESOLVED_STRATA |
| living | 0.015278 | 0.010227 | 0.005050 | [-0.0017910797687575985, 0.012832220253509618] | 1.000 | UNDERRESOLVED_STRATA |

## Decision

- Signal strength: **MODERATE**
- Root class: **NOTHINK_SIGNAL_HISTORY_CONFOUNDED**
- Primary signal: paired No-Think delta MRR=0.020218; equal-domain natural-weighted delta MRR=0.006164; supporting domains=2/4; NonHistory delta MRR=-0.002546; route DiD=0.016172
- Main confound: domain directions are mixed; GoldNotInHistory does not support Beta advantage; one or more domain strata are underresolved
- Next experiment: **OFFICIAL_REWARD_OR_TEST_DISTRIBUTION_AUDIT** (not started)

The equal-domain macro is diagnostic and is not an official aggregate score. Bootstrap intervals with underresolved strata are descriptive, not formal significance claims.
