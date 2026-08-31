# GR_REC Think Sample8 FullSID Positive A0 V3

Independent second-stage experiment built on the recorded best GRPO-TK model:

- parent: V3 formal `checkpoint-250`
- recorded parent external score: `1.3579`
- parent adapter SHA256: `64e1a85500b68f48d1996e6a215ea7a0bfe0dfdb925d6d53a86579ea23650be5`
- dataset: `grpo_tk_positive_groups_1946_20260829/train.jsonl`
- dataset SHA256: `e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693`
- topology: 611 Think-only unique groups, zero Probe4 overlap
- optimizer: fresh AdamW, lr `1e-6`, weight decay `0`, constant scheduler
- schedule: 611 fresh rollouts, `num_iterations=2`, 1222 optimizer steps

All V3 rollout, parsing, normalization, caching, old-logp and loss code is reused
unchanged. The sole mathematical change is the SID reward table:

| Match | V3 | Stage 2 |
|---|---:|---:|
| invalid | -1 | -1 |
| wrong domain | -0.25 | -0.25 |
| no hit | 0 | 0 |
| A-only | 0.5 | 0 |
| AB | 2 | 2 |
| Exact | 8 | 8 |

The shaped values drive both each independent SID G8 advantage and the V3 CoT
reward sum. AB and Exact are not treated as A-only and retain rewards 2 and 8.

This directory only configures the experiment. It does not start training.
Resume is deliberately rejected so a second-stage run cannot accidentally load
an optimizer or RNG state from V3 or another experiment.
