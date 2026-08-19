# GR_USER_v1 150-Prompt Directional Pilot

Run: `GR-USER-PILOT150-20260820-004130`

Decision: **HEALTHY_TO_300**

## Runtime

- Prompts: 150
- Optimizer steps: 20
- Training wall: 1506.51 seconds
- Seconds per prompt: 10.043
- Peak VRAM: 41196 MiB

## Action

| Segment | F1 | Precision | Recall | Wrong selection | Hallucination | Duplicate |
|---|---:|---:|---:|---:|---:|---:|
| early | 0.600021 | 0.713259 | 0.601485 | 0.763158 | 0.144737 | 0.105263 |
| middle | 0.551381 | 0.677015 | 0.557811 | 0.702703 | 0.020270 | 0.000000 |
| late | 0.662646 | 0.750360 | 0.646514 | 0.750000 | 0.052632 | 0.039474 |

## Chain

| Segment | Reward | Action Alignment | Logic Alignment | Date mismatch | Action mismatch |
|---|---:|---:|---:|---:|---:|
| early | 0.364682 | 0.536361 | 0.193003 | 0.407895 | 0.197368 |
| middle | 0.391102 | 0.582705 | 0.199499 | 0.351351 | 0.135135 |
| late | 0.402223 | 0.595265 | 0.209180 | 0.381579 | 0.302632 |

## Global

- Mean group reward std: 0.109318
- Zero-std ratio: 1.3333%
- Grad norm mean/max: 0.459249 / 0.703764
- Clip fraction mean/max: 0.0000% / 0.0000%
- Masked-token rate: 2.6226%
- Base delta: 0.0
- LoRA delta L2: 0.049535247
- NaN/Inf: False

Ten-prompt rolling means are stored in the JSON summary. Historical Phase 3B
baseline is recorded for context only; it is not treated as an exact matched
comparison because the prompt set differs.

No 300-prompt continuation, full epoch, or external evaluation was run.
