# GR_REC_ThinkExactClamp_Ablation_v1

Current experiment components:

- Think G4: Centered Exact-Clamp.
- NoThink G8: Conditional Hierarchical Token Credit on the sampled natural-language
  Domain decision token followed by the final SID A/B/C token positions.
- NoThink exact `[0]*8`: Gold-A dead-zero teacher bridge with lambda `0.02`.

NoThink Domain credit is written to the last `视频` / `商品` / `广告` / `主播`
token after `</think>` and before the final SID. The final SID
`<|domain_begin|>` token receives zero Domain credit. If any valid G8 candidate
is missing this text token or its text/SID domains disagree, Domain credit is
zeroed for the entire G8 while A/B/C continue unchanged.

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
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1" \
  python3 ablations/gr_rec_think_exact_clamp_v1/test_gpu_gradient_scale_audit.py
```

The runner prefix remains `GR-REC-CLAMP-BRIDGE-V1-`, same-run resume only, with
`--max-steps <= 1500`. GPU execution and training remain unauthorized.

## Zero-step gradient audit harness

`audit_gpu_gradient_scale.py` is a standalone, zero-update NoThink audit. It
generates each real G8 once, computes old log-probabilities once, and compares
the legacy sequence-level objective with the current hierarchical token-credit
objective using isolated forward/backward passes over the same immutable
rollout and model parameters. Exact real `[0]*8` groups additionally measure
raw and lambda-weighted Gold-A bridge gradients.

The harness has no optimizer or scheduler and contains no `.step()` call. It
requires the explicit `--execute-zero-step-gpu-audit` flag, defaults to eight
groups, accepts up to sixteen, and records only trainable LoRA gradients. Do not
run it until GPU execution is separately authorized.

The paired Domain-stage G8 re-audit is recorded in `../../results/` with full
fingerprint parity and unchanged trainable-parameter checksums. Those historical
gradients used SID-Domain placement. The harness now uses text-Domain placement,
but no new GPU audit has been executed.

**TEXT-DOMAIN PLACEMENT CPU VERIFIED / GPU RE-AUDIT NOT RUN / TRAINING NOT STARTED**
