# GR_USER_v1 External Regression Evidence Package

## Question

Why did a numerically healthy, user-only GRPO run improve its internal User probes while the official 11-task score fell below the BETA parent, mainly through Recommendation regression?

This document separates observed facts from interpretations. It is intended as the primary context file for an independent GPT review.

## Experiment identity

- Run: `GR-USER-FULL-EPOCH-20260820-052019`
- Parent: BETA-baseline Epoch 2, `checkpoint-1106`
- Resume point: GR_USER pilot300 at optimizer step 40
- Final point: optimizer step 378, 3000 unique prompts
- Training data: 1500 Action + 1500 Chain, duplicate 0, missing 0
- Base model: frozen throughout; only 504 LoRA tensors were trainable
- Formal objective: G=4, temperature 0.9, top-p 0.95, max-new-tokens 512, LR `1e-6`, PPO epsilon 0.2, `beta=0.0`, sqrt token penalty with lambda 0.5
- The run did not train Material, Recommendation, or World examples.

The exact contract is enforced in `scripts/run_user_full_epoch.py`; the complete training summary is `results/full_epoch_v1_summary.json`.

## Official external scores

Task order is Material 4, User 2, Recommendation 4, World 1.

| Model | Aggregate | Material | User | Recommendation | World |
| --- | ---: | ---: | ---: | ---: | ---: |
| BETA parent | 1.3313 | 0.1807 | 0.2545 | 0.6594 | 0.2368 |
| GR_USER step 240 | 1.3103 | 0.1780 | 0.2577 | 0.6375 | 0.2372 |
| GR_USER final 378 | 1.3146 | 0.1803 | 0.2572 | 0.6381 | 0.2390 |

Final 378 versus BETA:

| Task | BETA | Final | Delta |
| --- | ---: | ---: | ---: |
| Material video | 0.0519 | 0.0520 | +0.0001 |
| Material product | 0.0363 | 0.0351 | -0.0012 |
| Material ad | 0.0503 | 0.0514 | +0.0011 |
| Material live | 0.0422 | 0.0418 | -0.0004 |
| User Action Select | 0.1573 | 0.1576 | +0.0003 |
| User Topic/Chain | 0.0972 | 0.0996 | +0.0024 |
| Recommendation video | 0.1223 | 0.1223 | 0.0000 |
| Recommendation product | 0.1598 | 0.1462 | -0.0136 |
| Recommendation ad | 0.2072 | 0.2058 | -0.0014 |
| Recommendation live | 0.1701 | 0.1638 | -0.0063 |
| World | 0.2368 | 0.2390 | +0.0022 |

The final aggregate delta is `-0.0167`. Recommendation contributes `-0.0213`, while User contributes `+0.0027`. The product and live Recommendation domains explain `0.0199` of the `0.0213` Recommendation decrease.

Final versus step 240:

- Aggregate: `+0.0043`
- Material: `+0.0023`
- User: `-0.0005`
- Recommendation: `+0.0006`
- World: `+0.0018`

The second half of training did not continue a broad decline, but it also did not restore Recommendation.

## Internal training and probe evidence

Training was numerically stable:

- Mean/max gradient norm: `0.4215 / 0.9854`
- Mean/max clip fraction: `0 / 0`
- Reward population std: `0.09193`
- Zero-std group ratio: `1.67%`
- Masked-token rate: `2.27%`
- NaN/Inf: none
- Base parameter delta: exactly 0

The fixed light probe contains only 3 Action and 3 Chain prompts and was evaluated every 25 steps:

| Step | Action F1 | Chain Total | Chain Action | Chain Logic |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 0.606597 | 0.404108 | 0.590907 | 0.217310 |
| 240 | 0.646640 | 0.422061 | 0.618764 | 0.225358 |
| 378 | 0.655567 | 0.437132 | 0.629048 | 0.245216 |

The independent matched 40-sample Chain Probe v2, G=4, also improved from BETA/C0 to Final:

| Checkpoint | Total | Action | Logic |
| --- | ---: | ---: | ---: |
| C0 | 0.432861 | 0.635907 | 0.229815 |
| Final | 0.449262 | 0.667042 | 0.231482 |

Matched Final-C0 deltas and bootstrap 95% CIs:

- Total: `+0.016401`, CI `[+0.001667, +0.030570]`
- Action: `+0.031135`, CI `[+0.010172, +0.051805]`
- Logic: `+0.001667`, CI `[-0.009326, +0.012385]`

Thus the internal User improvement is real for the sampled User objective, especially Action alignment. It does not establish preservation of other tasks.

## Parameter movement evidence

The formal summary reports step40-to-final LoRA delta L2 `0.227619` and max absolute tensor-element delta `0.0002193`.

A read-only CPU safetensors comparison gave:

| Comparison | Relative LoRA L2 delta | Weight cosine |
| --- | ---: | ---: |
| BETA parent to Final | 0.3595% | 0.99999354 |
| Pilot step40 to Final | 0.3223% | 0.99999481 |

This is small in parameter space, but it is not a logit KL measurement. Small LoRA movement can change close SID rankings. Also, per-step clip fraction 0 only says each local PPO update stayed inside its clipping range; it does not bound cumulative policy drift across 378 steps.

## Evaluator evidence and boundaries

The supplied step-240 evaluator log:

- SHA256: `097041c669cb26a7035cc2634739b006263b33654e38a9f0e6b405442f36e828`
- 55,058 lines
- Evaluator version v3.1, seed 42, thinking disabled
- All 11 tasks completed, failed tasks 0
- Results and report were written and successfully submitted
- Runtime: 9446.9 seconds
- It identifies the evaluated model only as `/tmp/eval_model/merged`; it contains no source checkpoint path or model hash
- It emits a tokenizer regex warning, but no task failure or traceback

Therefore the log supports evaluation completeness, but cannot independently prove that the merged model came from step 240 or that adapter merging was correct. The final score and `2h40m32s` duration are user-provided; no final evaluator log was supplied.

The repository's BETA reference has observed run-to-run variation of about `+/-0.01`. A final aggregate delta of `-0.0167` is only moderately outside that band, but the same Recommendation deficit appears at step 240 (`-0.0219`) and final (`-0.0213`). This repeated, task-concentrated pattern is harder to explain as aggregate noise alone. There is no paired repeated evaluation or confidence interval for the official external scores.

`results/full_epoch_v1_summary.json` contains `external_evaluation_run: false` because that integrity summary was finalized before either external evaluation. It must not be interpreted as claiming the later evaluations did not happen.

## Evidence-based hypothesis status

| Hypothesis | Current status | Reason |
| --- | --- | --- |
| User-only GRPO caused cross-task interference in Recommendation | Supported | User improved slightly while Recommendation repeatedly lost about 0.021; no Recommendation data or KL protection was present |
| The GR_USER objective itself failed | Not supported | Both internal User probes improved, and official User sum increased slightly |
| Broad catastrophic forgetting occurred | Not supported | Material and World stayed near BETA; regression is concentrated in Recommendation; base was frozen and LoRA movement was small |
| Official evaluation noise fully explains the result | Insufficient and unlikely alone | No repeated paired eval, but two checkpoints show nearly identical Recommendation deficits larger than the usual aggregate variation |
| Fixed light probe predicted external generalization | Not supported | It has only six prompts and covers only User routes |
| Checkpoint merge or evaluator-tokenizer issue caused the score | Evidence insufficient | Logs lack checkpoint hash and merge audit; warning exists but evaluation completed normally |
| Absence of reference KL contributed to interference | Plausible, not isolated | `beta=0.0` is confirmed, but there is no otherwise-identical KL ablation |

## Prompt for independent GPT review

Please inspect this repository and analyze the GR_USER_v1 result using the evidence files listed below. Do not propose a new training run until you have reconciled all existing evidence.

Primary files:

- `baselines/native_source_domain_r32_v3/grpo/user/docs/gr_user_v1_external_regression_evidence.md`
- `baselines/native_source_domain_r32_v3/grpo/user/docs/full_epoch_v1.md`
- `baselines/native_source_domain_r32_v3/grpo/user/results/full_epoch_v1_summary.json`
- `baselines/native_source_domain_r32_v3/grpo/user/results/full_epoch_v1_external_eval_step240.json`
- `baselines/native_source_domain_r32_v3/grpo/user/results/full_epoch_v1_external_eval_final.json`
- `baselines/native_source_domain_r32_v3/grpo/user/scripts/run_user_full_epoch.py`
- `baselines/native_source_domain_r32_v3/grpo/user/scripts/user_grpo_trainer.py`
- `baselines/native_source_domain_r32_v3/docs/BATA_BASELINE.md`

Answer these questions:

1. Is “User improved internally while Recommendation regressed externally” a genuine contradiction, reward/evaluator mismatch, or expected multi-task interference?
2. Rank the hypotheses in the table above by explanatory strength. For each, separate direct evidence, inference, and missing evidence.
3. Explain why a 0.36% relative LoRA L2 change with cosine 0.9999935 can still produce a 0.0213 Recommendation decrease. Do not treat weight-space cosine as output-space KL.
4. Explain what `clip_fraction=0`, `beta=0`, the six-prompt fixed probe, and the 40-sample matched Chain probe do and do not establish.
5. Assess whether the repeated step240/final Recommendation pattern is likely larger than evaluation noise, while respecting the lack of repeated paired official evaluations.
6. Identify the smallest read-only audits that existing checkpoints and evaluator outputs could answer, especially merge identity, output-format drift, SID validity/ranking, output length, and product/live Recommendation changes.
7. Give a concise final judgment using: supported, not supported, or evidence insufficient. Do not invent unavailable metrics or assume the final evaluator log exists.
