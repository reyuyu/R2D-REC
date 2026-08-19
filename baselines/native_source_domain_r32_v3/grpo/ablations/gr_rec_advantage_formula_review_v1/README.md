# GR_REC advantage formula review v1

CPU-only sibling analysis comparing:

1. Current GRPO group-std normalization.
2. ExactFloor v1.
3. Centered Exact-Clamp.

It does not modify a production trainer and does not invoke a model. Run from
the GRPO project root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_advantage_formula_review_v1" \
  python3 ablations/gr_rec_advantage_formula_review_v1/test_advantage_formula_review.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_advantage_formula_review_v1" \
  python3 ablations/gr_rec_advantage_formula_review_v1/audit_advantage_formulas.py
```

`sum(abs(advantage))` is an advantage-mass proxy only. It is not a gradient
norm; token length, policy logprob gradients, PPO ratio and clipping still
affect the real update.
