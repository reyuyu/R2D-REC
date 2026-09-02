# BATA-STABLE-V0 checkpoint-553 to 560

## Verdict

`STABLE560_EXACT`

Three independent four-A800 continuations produced exact raw LoRA, full
effective B@A, optimizer, scheduler, and four-rank RNG state at step560.
The pilot retained the original 1106-step scheduler horizon and stopped
through a save-and-stop callback. No external evaluation was executed.

## Stable recipe

- Recovered 06:33 BATA source and original packed cache.
- Historical checkpoint-553, four A800 ranks, GA16, 8K neat packing.
- FA2 and Liger enabled; bf16 and pure_bf16 enabled.
- Original AdamW, cosine schedule, LR, warmup, and weight decay; horizon 1106.
- Strict deterministic algorithms, CUBLAS workspace contract, and
  deterministic FlashAttention backward.
- No DDP hook, gradient hashing, gradient reference, FA2 monkey patch, or
  extra forward/backward.

## Runs

| Run | Step | Loss | Grad norm | LR | Runtime sec |
|---|---:|---:|---:|---:|---:|
| STABLE560-A | 560 | 26.741113281250 | 0.933081269264 | 0.000103223090875715 | 358.316 |
| STABLE560-B | 560 | 26.741113281250 | 0.933081269264 | 0.000103223090875715 | 327.876 |
| STABLE560-C | 560 | 26.741113281250 | 0.933081269264 | 0.000103223090875715 | 328.135 |

## Pairwise stability

Every A/B, A/C, and B/C comparison has raw-LoRA cosine 1, relative L2
0, effective-B@A cosine 1, relative L2 0, and exact optimizer, scheduler,
and RNG fingerprints. Adam step is 560 for every state entry.

## Historical trajectory observation

| Scope | Update cosine | Progress | Residual |
|---|---:|---:|---:|
| All modules | 0.250077188 | 0.026318278 | 0.101896695 |
| Attention | 0.244607783 | 0.030178993 | 0.119629142 |
| MLP | 0.251675404 | 0.025761318 | 0.099064529 |

Stable step560 loss/grad norm are 26.741113281250 / 0.933081269264; historical values are 26.738397216797 / 0.954276442528.
These direction and scalar comparisons are observations only, not the
repeatability gate and not evidence of external competition score.

## Scope

This result authorizes review of a possible checkpoint-553 to 1106
continuation. It does not itself authorize or execute that continuation.
No dataset, token value, weight, checkpoint, optimizer/RNG byte, server
path, or credential is included in the public artifacts.
