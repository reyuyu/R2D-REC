# GR_REC Clamp-Bridge G8 baseline formal-1500 forensic summary

This is a CPU-only post-run summary. No benchmark or additional generation was run.

## Run integrity

- Final optimizer step: 1500
- Records: metrics=1500, rollouts=750, probes=36, traces=750
- Checkpoints: 8/8 complete
- Runtime: 35273.7s; train loss=0.015012
- Parameter deltas: LoRA=0.013679, base=0.0
- Fatal errors=0; non-finite metric values=0

## Policy stability

| Route | |loss| mean | grad mean / p95 / max | KL mean / p95 / max | clip mean / p95 / max | |ratio-1| mean |
|---|---:|---:|---:|---:|---:|
| Think | 0.004209 | 0.1006 / 0.2507 / 0.4954 | 0.000853 / 0.001909 / 0.002136 | 0.009771 / 0.024234 / 0.030857 | 0.000928 |
| NoThink | 0.036219 | 0.3550 / 1.0469 / 6.1630 | 0.000470 / 0.001714 / 0.017980 | 0.004688 / 0.029412 / 0.111111 | 0.001899 |

## Window trends

| Window | Think reward | Think len | Raw N | Grounded N | Coverage | Adv active | NoThink reward | All-zero | Bridge active | Wrong domain | Exact hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| steps_1_250 | 2.8912 | 699.1 | 3.2098 | 2.7485 | 0.8608 | 0.5893 | 0.2223 | 0.1673 | 0.1673 | 0.1711 | 0.0179 |
| steps_251_600 | 3.1405 | 678.6 | 3.1552 | 2.6875 | 0.8543 | 0.6810 | 0.2897 | 0.2364 | 0.2364 | 0.0188 | 0.0172 |
| steps_601_1000 | 3.6511 | 572.7 | 2.7923 | 2.5074 | 0.8913 | 0.5882 | 0.4835 | 0.3191 | 0.3191 | 0.0090 | 0.0363 |
| steps_1001_1500 | 3.4339 | 409.4 | 2.2988 | 1.7180 | 0.7422 | 0.4939 | 0.5100 | 0.3476 | 0.3476 | 0.0056 | 0.0347 |

## Fixed probes

| Step | Think reward | Think exact rate | Think length | NoThink reward | NoThink exact rate | NoThink positive rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.7305 | 0.4375 | 747.6 | 0.0000 | 0.0000 | 0.1562 |
| 200 | 4.2188 | 0.5000 | 670.1 | 0.4141 | 0.0312 | 0.2812 |
| 400 | 4.1738 | 0.5000 | 693.1 | 0.7891 | 0.0938 | 0.2500 |
| 600 | 2.6406 | 0.2500 | 676.4 | 0.4453 | 0.0312 | 0.2812 |
| 800 | 2.9297 | 0.3125 | 590.2 | 0.3906 | 0.0312 | 0.2188 |
| 1000 | 2.9023 | 0.3125 | 522.6 | 0.5469 | 0.0312 | 0.2500 |
| 1200 | 2.6816 | 0.2500 | 461.6 | 0.3125 | 0.0000 | 0.2500 |
| 1400 | 3.2539 | 0.3125 | 310.1 | 0.6094 | 0.0625 | 0.1875 |
| 1500 | 2.2305 | 0.1875 | 415.4 | 0.7188 | 0.0625 | 0.3125 |

## Initial interpretation

- Optimization stayed numerically stable: finite recorded metrics, bounded KL/clip statistics, and no fatal runtime or save error.
- NoThink learned useful hierarchy: wrong-domain predictions fell while A/AB/exact hit rates improved, but late all-zero and bridge exposure rose and remain a watch item.
- Think did not show a healthy monotonic trajectory. Late reward, answer length, Raw N, Grounded N, grounding coverage, and effective-advantage exposure weakened together, consistent with a real structure/coverage collapse risk rather than only noisier SID citation.
- The user-reported checkpoint-250 external score is 1.310. It is below the referenced baseline, so the early checkpoint is not evidence of benchmark improvement; checkpoint selection must be decided by the same external evaluation across later saved checkpoints.

## Evidence boundary

The external score is user-reported and was not reproduced in this post-run audit. This report does not claim final benchmark quality and did not run any benchmark.
