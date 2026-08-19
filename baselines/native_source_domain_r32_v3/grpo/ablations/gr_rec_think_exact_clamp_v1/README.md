# GR_REC_ThinkExactClamp_Ablation_v1

Independent GR_REC_v1 ablation from the original BATA adapter. Only Think G4
advantages use Centered Exact-Clamp. NoThink G8 returns the untouched parent
GR_REC_v1 advantages.

CPU checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/test_think_exact_clamp.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/audit_think_counterfactual.py
```

The formal runner requires the prefix `GR-REC-THINK-EXACT-CLAMP-V1-`, permits
resume only inside the same run ID, and requires `--max-steps` no greater than 1500. It must not
be launched without separate GPU authorization.
