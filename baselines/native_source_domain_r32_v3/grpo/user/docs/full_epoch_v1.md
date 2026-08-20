# GR_USER_v1 Full Epoch

- Run: `GR-USER-FULL-EPOCH-20260820-052019`
- Resume: step 40 / 300 prompts
- Final: step 378 / 3000 unique prompts
- Final checkpoint: `/data/GRPO_USER/runs/GR-USER-FULL-EPOCH-20260820-052019/full-epoch-final`
- Internal status: **INTERNAL_READY_FOR_EXTERNAL_EVAL**

## Runtime

- Training wall: 25184.79s
- Probe wall: 739.01s (2.85%)
- Checkpoint wall: 9.80s
- Total runner wall: 25974.38s
- Seconds/new prompt: 9.3277
- Peak VRAM: 41558.8 MiB

## Fixed light probe

| Step | Action F1 | Precision | Recall | Chain Total | Chain Action | Chain Logic |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.606597 | 0.700724 | 0.617951 | 0.404108 | 0.590907 | 0.217310 |
| 65 | 0.616433 | 0.716435 | 0.627506 | 0.370402 | 0.543129 | 0.197675 |
| 90 | 0.617214 | 0.717437 | 0.627506 | 0.444597 | 0.640748 | 0.248445 |
| 115 | 0.617590 | 0.719874 | 0.627506 | 0.400331 | 0.592166 | 0.208496 |
| 140 | 0.614932 | 0.714553 | 0.627506 | 0.422136 | 0.619745 | 0.224528 |
| 165 | 0.626837 | 0.714553 | 0.648340 | 0.461362 | 0.652999 | 0.269724 |
| 190 | 0.625028 | 0.705372 | 0.644737 | 0.447495 | 0.646939 | 0.248052 |
| 215 | 0.633456 | 0.692877 | 0.656328 | 0.418651 | 0.603197 | 0.234105 |
| 240 | 0.646640 | 0.734438 | 0.657112 | 0.422061 | 0.618764 | 0.225358 |
| 265 | 0.639955 | 0.742374 | 0.652726 | 0.453016 | 0.648806 | 0.257226 |
| 290 | 0.651246 | 0.761538 | 0.662281 | 0.434286 | 0.622018 | 0.246553 |
| 315 | 0.650511 | 0.748327 | 0.659305 | 0.468502 | 0.673605 | 0.263399 |
| 340 | 0.645599 | 0.733876 | 0.663690 | 0.453881 | 0.651610 | 0.256152 |
| 365 | 0.647865 | 0.729661 | 0.659305 | 0.433520 | 0.627732 | 0.239307 |
| 378 | 0.655567 | 0.750891 | 0.663690 | 0.437132 | 0.629048 | 0.245216 |

Final 40-sample Chain Probe v2 is run after training and recorded separately before external evaluation.

## Final Chain Probe v2

Matched inference-only probe: 40 Chain samples, G=4 (160 candidates).

| Checkpoint | Total | Action | Logic |
| --- | ---: | ---: | ---: |
| C0 | 0.432861 | 0.635907 | 0.229815 |
| C40 | 0.437846 | 0.645696 | 0.229997 |
| Final | 0.449262 | 0.667042 | 0.231482 |

| Comparison | Metric | Mean delta | Bootstrap 95% CI |
| --- | --- | ---: | ---: |
| C40-C0 | total_reward | 0.004985 | [-0.006283, 0.016295] |
| C40-C0 | action_alignment | 0.009788 | [-0.006110, 0.026083] |
| C40-C0 | logic_alignment | 0.000182 | [-0.008665, 0.008782] |
| Final-C0 | total_reward | 0.016401 | [0.001667, 0.030570] |
| Final-C0 | action_alignment | 0.031135 | [0.010172, 0.051805] |
| Final-C0 | logic_alignment | 0.001667 | [-0.009326, 0.012385] |
| Final-C40 | total_reward | 0.011416 | [-0.003748, 0.025679] |
| Final-C40 | action_alignment | 0.021346 | [-0.000224, 0.042260] |
| Final-C40 | logic_alignment | 0.001485 | [-0.009031, 0.011490] |

Internal status: `INTERNAL_READY_FOR_EXTERNAL_EVAL`.

## External evaluation: step 240

The external v3.1 evaluator result reported for `checkpoint-step240` is:

```text
aggregate:      1.3103
material:       0.0504, 0.0357, 0.0500, 0.0419
user:           0.1583, 0.0994
recommendation: 0.1195, 0.1462, 0.2044, 0.1674
world:          0.2372
```

The evaluator log started at `2026-08-20 12:22:33`, completed all 11 tasks with zero failed tasks, wrote both result files, and successfully reported the result. It evaluated a merged model at `/tmp/eval_model/merged`; the log contains no source checkpoint path or model hash, so the step-240 attribution is supplied by the evaluation submission rather than independently proven by the log.

| Group | BETA `1.3313` | Step 240 | Delta |
| --- | ---: | ---: | ---: |
| Material | 0.1807 | 0.1780 | -0.0027 |
| User | 0.2545 | 0.2577 | +0.0032 |
| Recommendation | 0.6594 | 0.6375 | -0.0219 |
| World | 0.2368 | 0.2372 | +0.0004 |
| Aggregate | 1.3313 | 1.3103 | -0.0210 |

The rounded group values sum to `1.3104`; the official aggregate is `1.3103` because aggregation uses unrounded metrics. The total decrease exceeds the repository's approximate `+/-0.01` single-run variation band and is concentrated in Recommendation, especially product (`0.1598 -> 0.1462`, `-0.0136`). User itself moved slightly upward (`+0.0032`), consistent in direction with the internal probe but much smaller than the internal probe improvement.

This is therefore evidence of cross-task interference, not evidence that the GR_USER optimization failed on its own objective. The run used only User prompts and `beta=0.0`, so it had no explicit reference-model KL term protecting Recommendation behavior. Small LoRA movement can still change SID ranking on a different task.

## External evaluation: final step 378

The external result reported for `full-epoch-final` took `2h40m32s` and is:

```text
aggregate:      1.3146
material:       0.0520, 0.0351, 0.0514, 0.0418
user:           0.1576, 0.0996
recommendation: 0.1223, 0.1462, 0.2058, 0.1638
world:          0.2390
```

| Group | BETA `1.3313` | Step 240 | Final 378 | Final vs BETA | Final vs Step 240 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Material | 0.1807 | 0.1780 | 0.1803 | -0.0004 | +0.0023 |
| User | 0.2545 | 0.2577 | 0.2572 | +0.0027 | -0.0005 |
| Recommendation | 0.6594 | 0.6375 | 0.6381 | -0.0213 | +0.0006 |
| World | 0.2368 | 0.2372 | 0.2390 | +0.0022 | +0.0018 |
| Aggregate | 1.3313 | 1.3103 | 1.3146 | -0.0167 | +0.0043 |

The second half of training recovered `0.0043` aggregate points but did not recover Recommendation. User remained slightly above BETA, while Recommendation remained lower by `0.0213`; product stayed at `0.1462` and live decreased further to `0.1638`. This confirms that the step-240 result was not merely a bad intermediate checkpoint. It also shows no continued catastrophic drift from step 240 to final: the final result is modestly better overall, with nearly unchanged User and Recommendation group sums.

The final scores and duration are user-provided in this record; unlike step 240, no evaluator log was supplied for an independent task-completion or checkpoint-identity audit.
