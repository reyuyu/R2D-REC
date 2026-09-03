# BATA-STABLE-V0 base to checkpoint-553

## Verdict

`STABLE553_EXACT`

`FIRST_EPOCH_REPEATABILITY = CONFIRMED`

Two independent four-A800 runs started from the same SHA-pinned base model,
retained the original 1106-step scheduler horizon, and used a callback to
save and stop at optimizer step553. No second-epoch continuation or external
evaluation was executed.

## Recipe

- Recovered 2026-08-12 06:33 BATA source and historical packed cache.
- 222,001 source segments and 35,380 packed rows; four A800 ranks.
- Micro batch 1, GA16, global batch 64, 8K neat packing.
- FA2 and Liger enabled; bf16 and pure_bf16 enabled.
- LoRA r32/alpha64/dropout0.05/all; AdamW LR 2e-4, cosine, warmup 0.03.
- Seed 20260806, weight decay 0.01, fractional GC 0.4.
- Strict PyTorch determinism, deterministic FA2 backward, and fixed CUBLAS workspace.
- No resume checkpoint and no forensic gradient/DDP instrumentation.

## Runs

| Run | Step | Latest logged step | Loss | Grad norm | LR | Epoch | Runtime sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| STABLE553-A | 553 | 550 | 31.284515380859 | 0.832315027714 | 0.000106150355246772 | 0.994912379876 | 23277.332 |
| STABLE553-B | 553 | 550 | 31.284515380859 | 0.832315027714 | 0.000106150355246772 | 0.994912379876 | 23263.464 |

## Repeatability

| Evidence | Result |
|---|---|
| Adapter file SHA | exact |
| Canonical LoRA SHA | exact |
| Raw LoRA cosine / relative L2 | 1 / 0 |
| Effective B@A cosine / relative L2 | 1 / 0 |
| Optimizer | exact |
| Scheduler | exact |
| Four-rank RNG | exact |
| Shared scalar logs | exact |

## Historical checkpoint-553 comparison

Raw LoRA cosine / relative L2: 0.884313333 / 0.480408541.

Effective B@A cosine / relative L2: 0.655698650 / 0.826195052.

| Projection | B@A cosine | Relative L2 |
|---|---:|---:|
| q | 0.492441646 | 1.006106254 |
| k | 0.491638872 | 1.003757731 |
| v | 0.475436702 | 1.015680747 |
| o | 0.520439396 | 0.978304582 |
| gate | 0.721084002 | 0.743200086 |
| up | 0.702011155 | 0.769039124 |
| down | 0.534397981 | 0.956807053 |

Layerwise B@A cosine min/mean/max: 0.230693013 / 0.652073614 / 0.951943928.

Historical scalar comparison uses 110 common logging steps. Mean/max absolute loss delta: 0.110123652 / 0.962811279; mean/max grad-norm delta: 0.084651455 / 0.654540539. First clearly divergent step: 5.

Historical similarity is observational and does not affect the repeatability verdict.

## Evaluation

`NOT_EXECUTED`

No reliable historical checkpoint-553 result under a separately proven identical
external-evaluator contract was used in this phase.

## Scope

The runs stopped at checkpoint-553. They did not continue to step1106 and did
not start GR_REC, GRPO-TK, or MC_USER. Public artifacts contain no dataset rows,
tokens, model/optimizer/RNG bytes, raw examples, credentials, or server paths.
