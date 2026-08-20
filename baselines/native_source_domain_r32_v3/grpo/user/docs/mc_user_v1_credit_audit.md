# MC_USER_v1 Real-data Marginal Credit Audit

Status: **MC_CREDIT_AUDIT_PASS**

## Scope

- Seed: `20260821`
- Frozen train SHA256: `5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801`
- Selection: 100 Action + 100 Chain from train_3000.jsonl
- Execution: CPU-only; no generation, training, checkpoint, or GR_USER_v1 changes
- Credit source: leave-one-out calls to the existing Action/Chain scorers only

## Action

| Metric | Result |
|---|---:|
| TP positive rate | 100.000000% (9480/9480) |
| FP negative rate | 100.000000% (200/200) |
| Duplicate-zero rate | 100.000000% (100/100) |
| History-in/out FP consistency | 100.000000% |
| Span errors | 0 / 9780 |
| Anomalies | 0 |

## Chain

| Metric | Result |
|---|---:|
| Positive / negative / zero | 1395 / 200 / 0 |
| delta_total mean / min / max | 0.158965841782 / -0.200000000000 / 0.666666666667 |
| delta_action mean | 0.165191367323 |
| delta_logic mean | 0.152740316242 |
| Reward-delta max error | 1.110e-16 |
| Span errors | 0 / 1595 |
| Invalid variants | 0 |

## Interpretation

Marginal credit is determined only by the existing evaluator-aligned reward change.
Constraint and grounding violations are diagnostics and are not used to alter any credit.
Complete anomaly records are retained in the JSON artifact when present.
