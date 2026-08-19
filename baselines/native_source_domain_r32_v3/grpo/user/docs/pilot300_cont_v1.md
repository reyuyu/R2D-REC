# GR_USER_v1 Pilot300 continuation

- Run: `GR-USER-PILOT300-CONT-20260820-031716`
- Resume: `/data/GRPO_USER/runs/GR-USER-PILOT150-20260820-004130/pilot150-final`
- Optimizer: step 20 -> 40
- Samples: 150 new, 0 overlap, 300 cumulative unique
- Decision: **STOP_AND_ANALYZE**

## Fixed Probe

| Step | Action F1 | Precision | Recall | Chain Total | Chain Action | Chain Logic |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.605866 | 0.807167 | 0.593404 | 0.419327 | 0.613821 | 0.224833 |
| 20 | 0.600360 | 0.802604 | 0.583231 | 0.383608 | 0.561802 | 0.205413 |
| 30 | 0.601392 | 0.808022 | 0.578838 | 0.416167 | 0.606698 | 0.225637 |
| 40 | 0.607999 | 0.803286 | 0.593560 | 0.387233 | 0.566351 | 0.208115 |

## Second150

- Action F1/P/R: 0.626557 / 0.729036 / 0.644783
- Action wrong-selection/hallucination/duplicate: 68.6667% / 5.6667% / 0.6667%
- Chain reward/action/logic: 0.421263 / 0.622206 / 0.220320
- Chain date/action mismatch: 39.6667% / 10.3333%

## Cumulative300

- Action F1/P/R: 0.609224 / 0.716907 / 0.618065
- Chain reward/action/logic: 0.404244 / 0.598176 / 0.210313
- Probe overhead: 20.4464%

This phase did not run a full epoch or external evaluation.
