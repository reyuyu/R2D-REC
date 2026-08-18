# GR_REC_DSR_Simple_Ablation_v1

Status: **IMPLEMENTED / SMOKE ONLY**

- Parent: BATA
- Baseline: GR_REC_v1
- Reference ablation: GR_REC_DSR_Ablation_v1
- Runner prefix: `GR-REC-DSR-SIMPLE-V1-`

DSR-Simple intentionally removes grounding reward, evidence diversity reward,
prefix-support reward, entropy exploration, and always-on Think auxiliary.
Think retains only raw-interest-count anti-collapse and dead-zero target-domain
Beam A diversity. NoThink directly reuses DSR v1.

Engineering validation is complete. No pilot decision is implied by this smoke.

No pilot has been authorized or launched.

## Development-machine validation (2026-08-19)

- CPU suite: 49/49 PASS, including the complete DSR v1 regression suite.
- Real model: OneReason 8B + original BATA adapter, four A800 GPUs.
- Accepted smoke: GR-REC-DSR-SIMPLE-V1-SMOKE-20260819-R2.
- Route sequence: Think, Think, NoThink, NoThink, NoThink, NoThink.
- Optimizer steps: 12/12; LoRA delta 0.000040; base delta exactly 0.
- Monitor: 6 rollout summaries and 12 step summaries; training_signal and diagnostic_only are separated.
- Think branches observed: primary_only, zero_std_count_rescue, and dead_zero_count_diversity_rescue.
- Gradient audit: optimizer_created=false and optimizer_steps=0.
- Primary-signal groups: 9; Simple auxiliary gradient exact-zero: true.
- Active rescue groups: 4; weighted rescue gradient norm mean 0.0852467593, max 0.0871258684.
- NoThink S_A-only gradient proof: true; wrong-A descent direction proof: true.
- Audit artifact: /data/GRPO/outputs/GR-REC-DSR-SIMPLE-V1-GRADIENT-AUDIT.json
- Smoke monitor: /data/GRPO/runs/GR-REC-DSR-SIMPLE-V1-SMOKE-20260819-R2
- The first launch attempt stopped before optimizer step 1 because NCCL selected an unusable interface. The accepted R2 launch used NCCL_SOCKET_IFNAME=lo, GLOO_SOCKET_IFNAME=lo, and NCCL_IB_DISABLE=1.

## Performance

- Accepted smoke runtime: 285 seconds for 12 optimizer steps.
- Mean Think rollout wall time: 117.1 seconds.
- Mean NoThink rollout wall time: about 3.0 seconds.
- Mean policy-update phase: about 0.62 seconds.
- Frozen full-schedule estimate from the smoke: about 13.6 hours. This is an estimate, not a launched run.

## Dashboard validation

- The development-machine dashboard detects `DSR-Simple` independently from old DSR runs.
- Core Think cards and curves show Raw N, S_N, target-domain unique A, D_A, Simple rescue rate, auxiliary magnitude, and the three branch rates.
- Grounded N, Coverage, D_cot, S_prefix, and primary Exact/AB/A are labeled diagnostic-only.
- Candidate details merge passive DSR traces by rollout, group, and candidate index.
- Existing DSR v1 pilot200 cards, curves, and checkpoint downloads still render correctly.
- Chart hover uses nearest-step snapping and reports all series at the aligned sampling step.

## Isolation proof

- Old DSR directory SHA-256 aggregate: 3662b1b31eec73a4181e48fd3de6ef998acea00ef448ed66e0df8cd55c1baaf7
- Production runner SHA-256: 66d5135385c0093f03023a38cc852045fa85b32cd130dece7be172eb40d98c79
- Both hashes match the pre-deployment values.
