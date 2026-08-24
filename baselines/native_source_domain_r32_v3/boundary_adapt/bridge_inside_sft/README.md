# Bridge-Inside Transition SFT V1

This package implements a narrowly scoped transition migration from Fresh BATA:

`CoT -> </think> -> Bridge -> Domain -> ABC`

to:

`CoT -> Bridge -> </think> -> Domain -> ABC`

Only the exact source Bridge tokens and the single atomic `</think>` token are supervised. Prompt, CoT, Domain, and ABC remain masked. Each source group contributes one row, and each row first averages CE over its own transition tokens so different Bridge token lengths retain equal group weight.

The workflow creates a deterministic domain-stratified adaptation holdout, persists fixed-seed Self-CoT token IDs, performs strict Beam32 evaluation, trains only the existing Fresh BATA LoRA for 50 optimizer steps, and stops for manual review. It never invokes an external evaluator or automatically continues to 200 steps.
