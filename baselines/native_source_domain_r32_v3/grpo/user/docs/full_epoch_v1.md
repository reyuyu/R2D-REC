# GR_USER_v1 Full Epoch

- Run: `GR-USER-FULL-EPOCH-20260820-052019`
- Resume: step 40 / 300 prompts
- Final: step 378 / 3000 unique prompts
- Final checkpoint: `/data/GRPO_USER/runs/GR-USER-FULL-EPOCH-20260820-052019/full-epoch-final`
- Internal status: **PENDING_FINAL_CHAIN_PROBE_V2**

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
