# GR_REC_DSR_Simple_Ablation_v1

Status: **FULL RUN COMPLETE - ENGINEERING PASS, RESEARCH RESULT MIXED**

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

## Final CPU-only forensic audit (2026-08-19)

The bounded full schedule completed normally at optimizer step 2316. No second
run was started. The final Trainer state records epoch 1.0, train loss
0.0004440254, LoRA delta 0.0085789816, and base delta exactly 0. The wall-clock
runtime was 46,316 seconds (12 h 51 m 56 s).

### Completion and integrity

- No training process remains for this RUN_ID.
- `metrics.jsonl` contains exactly 2316 unique steps spanning 1 through 2316,
  with no missing or duplicate step.
- All rows in metrics, rollouts, DSR steps, DSR metrics, compact forensic,
  probes, and traces parse as strict JSON. There are no non-finite stored values.
- The final fixed probe exists at step 2316; the probe schedule contains steps
  0, every 200 through 2200, and 2316 for all four domains.
- The log contains no OOM, Traceback, NCCL error/timeout, save error, or STOP.
  PyTorch emitted one process-group-not-destroyed warning during otherwise
  normal process exit; it did not affect checkpoint persistence.
- `checkpoint-2316` contains the adapter and full Trainer recovery state.
  Adapter config SHA-256: `3593affa880ae584d6837eceffd5c47155645db0f4e6a7f040bd3300fe71e187`.
  Adapter model SHA-256: `6200af822dc9f24b8f6453c28e6be01dafd8165b7a53de81668d0bd0cffae67d`.

### Optimization health

First-200 versus last-200 policy means remained bounded: approximate KL
0.000962 -> 0.000393, clip fraction 0.006870 -> 0.004282, ratio mean
1.001021 -> 1.000321, and gradient norm 0.837804 -> 0.471814. Reward mean
increased 1.056519 -> 1.407625, while zero-std ratio also increased
0.282500 -> 0.360000. These values show stable, conservative policy updates;
the higher zero-std rate is a signal-quality limitation rather than optimizer
instability.

### Think findings

- The first-200 to last-200 training windows show Raw N 3.140 -> 2.041,
  Grounded N 2.780 -> 1.232, and grounding coverage 0.884 -> 0.624.
- Fixed probes confirm the direction: mean Raw N 3.8125 -> 2.1875 and mean
  grounding coverage 0.9375 -> 0.6146 from step 0 to 2316.
- This is material interest-count compression, but not a catastrophic collapse:
  last-window Raw N remains above the preregistered 1.5 STOP threshold.
- Target-domain Beam diversity improved in the training windows: D_A
  0.743 -> 0.858 and unique valid target A 7.184 -> 8.805.
- Think reward increased on sampled training windows (2.910 -> 3.311) but the
  fixed-probe mean declined 3.730 -> 2.902. The reward gain therefore is not
  accepted as robust generalization evidence.
- Parser success reached 1.0 and closure remained 1.0 in the last window.
  Think Beam invalid rate decreased from 0.00580 to 0.00360.

### NoThink findings

The last window retained perfect valid-SID rate and zero invalid candidates.
Gold-A-or-better candidate rate improved 0.1698 -> 0.2276, wrong-domain rate
fell 0.2024 -> 0.0009, and the fixed-probe mean reward improved 0.000 ->
0.203. However, zero-std/all-zero group rate rose from 0.1866/0.1567 to
0.2910/0.2910, so signal sparsity remains material. Rescue activation tracked
the all-zero groups as designed.

### Verdict

The run is an engineering PASS: the schedule, isolation contract, persistence,
monitoring, and optimizer behavior are valid. The research result is MIXED and
does not support replacing full DSR with DSR-Simple. Removing grounding-aware
Think objectives preserved target-domain Beam diversity but coincided with a
large Raw N reduction, a sharper grounding-coverage reduction, and lower fixed
Think probe reward. Treat `checkpoint-2316` as a completed ablation artifact,
not as a promoted production winner.
