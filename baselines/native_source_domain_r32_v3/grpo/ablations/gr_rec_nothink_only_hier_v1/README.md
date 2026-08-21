# GR_REC_NoThinkOnly_Hier_v1

NoThink-only RL attribution ablation from parent code point
`e0d8fcd471dedaf615eb36a94434b0298f8b6d5d`.

The runner filters the joint route dataset to `route=no_think`, retains the
same four-group fixed-probe exclusion, and reuses
`ThinkExactClampRecGRPOTrainer` without copying hierarchy or bridge math.
Thin runtime guards reject any Think reward or loss batch. Fixed Think probes
remain inference-only and do not enter reward training, backward, or optimizer
updates.

The parent joint sampler selected 1545 groups after the four probes and
audited one deterministic tail group as dropped by its fixed chunk topology.
This runner explicitly applies the same recorded tail exclusion, yielding the
same 1544-group training cohort instead of duplicating a group or changing the
two-G8 global rollout shape.

Frozen contracts:

- fresh original BATA parent, no RL checkpoint parent;
- NoThink G8, two unique groups per global rollout;
- `num_iterations=2`, temperature/top-p `1/1`;
- route multiplier `0.5`;
- full-G8 milestone baselines with prefix-gated Domain/A/B/C token credit;
- credited-token SUM reduction;
- exact `[0]*8` Gold-A bridge only, lambda `0.02`.

CPU checks are run from the GRPO root with the production environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1:ablations/gr_rec_nothink_only_hier_v1" \
  python3 ablations/gr_rec_nothink_only_hier_v1/test_nothink_only_runner.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts:ablations/gr_rec_think_exact_clamp_v1:ablations/gr_rec_nothink_only_hier_v1" \
  python3 ablations/gr_rec_nothink_only_hier_v1/audit_nothink_only_sampler.py
```
