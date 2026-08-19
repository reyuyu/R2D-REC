# GR_REC_ThinkExactClamp_Ablation_v1

Phase 2 combines two isolated changes over GR_REC_v1 and the fresh original
BATA adapter:

- Think G4: Centered Exact-Clamp advantages.
- NoThink G8: unchanged GR_REC_v1 primary GRPO plus a gated Minimal
  Hierarchical Teacher Bridge (`lambda_bridge=0.02`).

Phase 1, the pure Think-only implementation, is permanently available at
`917d3d3a9e53db2e80bf425b435c597bb210b804`. Phase 2 is the formal future GPU
candidate. This is code inheritance, not checkpoint continuation.

CPU checks from the GRPO root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/test_think_exact_clamp.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/test_nothink_bridge.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/audit_bridge_tokenizer.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/audit_nothink_bridge_history.py
```

The runner prefix is `GR-REC-CLAMP-BRIDGE-V1-`, resume is limited to the same
run ID, and `--max-steps` is capped at 1500. Do not launch it until the required
real 8B + original BATA zero-step gradient audit and separate GPU authorization.

Status: **CPU IMPLEMENTATION READY / GPU GRADIENT AUDIT PENDING**.
