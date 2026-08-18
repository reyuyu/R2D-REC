# GR_REC_DSR_Simple_Ablation_v1

Status: **FULL RUN IN PROGRESS**

- Parent: BATA
- Baseline: GR_REC_v1
- Reference ablation: GR_REC_DSR_Ablation_v1
- Runner prefix: `GR-REC-DSR-SIMPLE-V1-`

DSR-Simple intentionally removes grounding reward, evidence diversity reward,
prefix-support reward, entropy exploration, and always-on Think auxiliary.
Think retains only raw-interest-count anti-collapse and dead-zero target-domain
Beam A diversity. NoThink directly reuses DSR v1.

Engineering validation is complete. One continuous 2316-step formal run is
authorized with an automatic step-200 PASS/WARN/STOP safety gate. PASS and WARN
continue in the same Trainer, optimizer, scheduler, and RNG process; only STOP
ends training after persisting checkpoint-200.

## Formal run contract (2026-08-19)

- RUN_ID: `GR-REC-DSR-SIMPLE-V1-FULL-E1-GATE200-20260819`
- Parent: original BATA adapter; resume is forbidden.
- Optimizer steps: 2316, one epoch only.
- Optimizer schedule SHA-256: `ac86490e5660e365fe4adedb6ac9321cffb95db1636b2f4bafad99841ba508b7`.
- Fixed probes: step 0, every 200 steps, and final; seed 20260818.
- Checkpoints: gate 200 plus normal 500-step intervals; retention limit 6.
- Compact evidence: every Think candidate and every NoThink G=8 group in
  `simple_forensic.jsonl`, written on rank0 from the existing gathered records.
- Gate report: `gate200_report.json`; no extra inference, decode, CUDA sync, or
  distributed collective is introduced.
- The exact logging-only source commit and deployment hashes are recorded in the
  run deployment manifest before launch.

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
- Frozen full-schedule estimate from the smoke: about 13.6 hours.

## Dashboard validation

- The development-machine dashboard detects `DSR-Simple` independently from old DSR runs.
- Core Think cards and curves show Raw N, S_N, target-domain unique A, D_A, Simple rescue rate, auxiliary magnitude, and the three branch rates.
- Grounded N, Coverage, D_cot, S_prefix, and primary Exact/AB/A are labeled diagnostic-only.
- Candidate details merge passive DSR traces by rollout, group, and candidate index.
- Existing DSR v1 pilot200 cards, curves, and checkpoint downloads still render correctly.
- Chart hover uses nearest-step snapping and reports all series at the aligned sampling step.
- The DSR-Simple page shows a Chinese Gate200 panel with current step, decision,
  core threshold metrics, timestamp, warnings, and stop reasons.

## Isolation proof

- Old DSR directory SHA-256 aggregate: 3662b1b31eec73a4181e48fd3de6ef998acea00ef448ed66e0df8cd55c1baaf7
- Production runner SHA-256: 66d5135385c0093f03023a38cc852045fa85b32cd130dece7be172eb40d98c79
- Both hashes match the pre-deployment values.
