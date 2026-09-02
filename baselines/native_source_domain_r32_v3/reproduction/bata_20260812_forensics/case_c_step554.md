# CASE C raw step554 evidence

Two independent recovered-source replays used byte-identical historical
checkpoint-553 inputs. At step554, every rank received the same ordered batch
and retained the same explicit RNG state across Replay A and Replay B.

| Rank | Batch SHA (A=B) | RNG SHA (A=B) | Loss A | Loss B | B-A |
|---:|---|---|---:|---:|---:|
| 0 | `acec704d...46095` | `b3cfd074...d9e2f` | 2.0197827555 | 2.0197827704 | +1.49e-08 |
| 1 | `045f5ddb...0a034` | `2d745ebb...1b323` | 1.5963932090 | 1.5963931940 | -1.49e-08 |
| 2 | `b3d47427...a05ad` | `754dad79...ac17f` | 2.5609864555 | 2.5609864257 | -2.98e-08 |
| 3 | `365d252d...5f3f` | `4e147f71...f8b` | 1.3222748777 | 1.3222748926 | +1.49e-08 |

Each rank processed 16 microbatches. Global grad norm was `0.7419703007` in A
and `0.7419010401` in B.

The loss scalars were captured after the single model forward and local SID8
loss computation, but before `accelerator.backward()` and DDP gradient
all-reduce. This excludes DDP/NCCL gradient reduction as the sole source of the
first loss divergence. It does not distinguish local model-forward kernels
from local loss aggregation, so the current root-cause level is `UNRESOLVED`.
