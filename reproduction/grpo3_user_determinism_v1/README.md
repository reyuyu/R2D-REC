# GRPO-3 User GRPO deterministic smoke

This package runs two independent five-optimizer-step repetitions of the existing
`MC_USER Hybrid K4` trainer. It does not start formal GRPO-3 training.

Frozen training semantics:

- Parent: cumulative GRPO-2 continued-single-adapter checkpoint 300.
- Registered dataset: `user_grpo`, train split, 3,000 rows.
- Dataset SHA256: `5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801`.
- Strict alternating Action/Chain route schedule, candidate-parallel K=4.
- Temperature 0.9, top-p 0.95, 512 maximum new tokens.
- AdamW, learning rate `3e-7`, weight decay 0, sequence/local weights 1.0/0.3.
- The existing reward, prompt, sampler, objective and LoRA architecture are unchanged.

The only additions are deterministic runtime seeding, evidence logging, saving each
smoke's final adapter, and an A/B comparator. Formal training is allowed only when
all five step records and the final adapter are byte-exact across the two runs.
