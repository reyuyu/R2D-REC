# GR_REC_ThinkExactClamp_Ablation_v1

Current experiment components:

- Think G4: Centered Exact-Clamp.
- NoThink G8: Conditional Hierarchical Token Credit on the final SID A/B/C
  token positions.
- NoThink exact `[0]*8`: Gold-A dead-zero teacher bridge with lambda `0.02`.

The former `a_collapse_ab_bridge`, missing-A teacher and current-path Gold-B
teacher were removed after commit
`ba813321f3d31158f293e67a5729669e78d42ca9`. They remain recoverable from Git
history only.

CPU checks from the GRPO root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/test_think_exact_clamp.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/test_nothink_hierarchical_credit.py
```

The runner prefix remains `GR-REC-CLAMP-BRIDGE-V1-`, same-run resume only, with
`--max-steps <= 1500`. GPU execution and training remain unauthorized.
