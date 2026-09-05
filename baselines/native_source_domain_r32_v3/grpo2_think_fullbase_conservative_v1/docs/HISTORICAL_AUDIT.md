# Historical GRPO-2 audit

The repository contains a dedicated second-stage implementation at
`gr_rec_think_sample8_fullsid_positive_a0_v3`; GRPO-2 is therefore not modeled
as the Think route of GRPO-1.

Frozen historical semantics:

- 611 unique Think-only recommendation groups.
- Domain counts: ad 160, living 73, prod 118, video 260.
- CoT generation: global G4, temperature 0.9, top-p 0.95, closure required.
- Per-CoT SID generation: eight independent sampled continuations, temperature
  1.0, top-p 1.0, top-k 0, no fixed-domain prefix.
- Four SID groups are normalized independently as G8, never as G32.
- CoT reward is the sum of its eight shaped SID rewards.
- Reward levels are invalid -1, wrong-domain -0.25, no-hit 0, A-only 0,
  AB 2, exact 8.
- Loss is `L_cot + L_sid`; `num_iterations=2` reuses accepted rollout,
  rewards, advantages, and detached old log-probabilities.
- Historical optimizer was fresh AdamW, learning rate 1e-6, weight decay 0,
  constant schedule, beta 0.

Current pilot-only differences are the standalone merged test parent, a fresh
R32 LoRA, deterministic evidence, lineage gates, and learning rate 2e-7.

The BF16 parent-export parity gate requires exact prompt IDs and greedy generated
IDs on four fixed data rows. Corresponding logits are compared on the source
model's fixed Top-32 token IDs; each reloaded model Top-32 set must overlap at
least 31/32 tokens. The gate additionally
requires selected-logit max absolute difference at most 0.5, mean absolute
difference at most 0.2, and max relative difference at most 2%. These bounds
cover observed high-logit BF16 ULP accumulation while remaining fail-closed on
behavioral, ranking, or material relative drift.
