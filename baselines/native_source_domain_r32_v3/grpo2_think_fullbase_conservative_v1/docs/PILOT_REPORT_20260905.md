# GRPO-2 Think-only deterministic pilot report

## Outcome

- Final state: `READY_FOR_GRPO2_PARENT_FINALIZATION`
- Pilot status: `PASS`
- Formal GRPO-2 training started: no
- GRPO-3 started: no
- Parent classification: isolated test parent only; not a canonical GRPO-1 parent

## Historical contract

- Recipe: Positive-A0 V3, Think-only global G4 CoT sampling plus four independent G8 FullSID normalizations
- Think-only dataset SHA256: `e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693`
- Historical learning rate: `1e-6`
- Pilot learning rate: `2e-7`
- Training seed: `20260816`
- LoRA initialization seed: `20260905`
- Optimizer: fresh LoRA-only AdamW, constant schedule, weight decay 0, beta 0

## Test parent

- Source SFT model SHA256: `8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59`
- Source GRPO-1 checkpoint-300 adapter SHA256: `fbae37f3892c414a7c86f2285a4568f2a5bb526c8d249616f0445bd9b90f6c19`
- Canonical merged-model identity: `2648d40e375fe01725d58a974175df352d2156367fa9a50861edd6c98805b45b`
- Standalone reload: `PASS`
- Functional parity: exact tokenization and greedy IDs; Top-32 overlap per probe was 32/32, 31/32, 31/32, and 32/32
- BF16 selected-logit drift: max absolute `0.5`, mean absolute `0.126953125`, max relative `0.017167381974248927`
- Parent identity remained unchanged through the pilot

## Determinism gate

Two independent 5-step executions were compared after excluding telemetry-only wall-clock fields.

- Result: `BYTE_EXACT`
- Smoke A adapter SHA256: `31da488c6b72c380e1e7913b032e48ceda06b2774588e0d50eea2e5efd9b8cac`
- Smoke B adapter SHA256: `31da488c6b72c380e1e7913b032e48ceda06b2774588e0d50eea2e5efd9b8cac`
- All four ranks matched for step evidence and trainer evidence
- Sample8 rollout events matched
- Probe4 Beam32 events matched
- First deterministic divergence: none

## Pilot

- Steps: 20
- Train loss: `-0.28578320145606995`
- In-memory base delta: `0.0`
- LoRA delta: `0.00022990919105581042`
- Checkpoint 10 adapter SHA256: `4be2118b946fb1d954a85aa8278e44dd2bf14c173fdf7dbd248364ee85e3fadd`
- Checkpoint 20 adapter SHA256: `4ba669ae2c8304355c570be799918cf83da78ca7f7436ff23acfce27a0ab93c6`
- Both checkpoints are adapter-only and include optimizer, scheduler, trainer state, training arguments, four RNG states, and lineage

## Retention probe

The fixed four-group probe is a smoke-level retention signal, not an external model evaluation.

| Route | Metric | Step 0 | Step 20 |
| --- | --- | ---: | ---: |
| Think | success@32 | 0.75 | 0.75 |
| Think | mean reward | 2.861328125 | 2.861328125 |
| Think | closure rate | 1.0 | 1.0 |
| NoThink | success@k | 0.25 | 0.25 |
| NoThink | mean reward | -0.078125 | 0.0078125 |
| NoThink | positive candidate rate | 0.125 | 0.1875 |

## Remaining blocker

The parent is deliberately marked `canonical=false` and `test_parent_only=true`. The pilot supports the recipe, implementation, fixed-environment repeatability, and retention behavior, but a canonical GRPO-1 parent must be finalized before any formal GRPO-2 training is authorized.
