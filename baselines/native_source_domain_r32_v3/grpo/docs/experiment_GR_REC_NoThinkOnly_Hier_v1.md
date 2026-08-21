# GR_REC_NoThinkOnly_Hier_v1

## Purpose

This is an independent NoThink-only RL attribution ablation branched from
`ablation/gr-rec-think-exact-clamp-v1` at
`e0d8fcd471dedaf615eb36a94434b0298f8b6d5d`.

It asks whether the frozen NoThink hierarchical token objective improves
Recommendation without Think optimizer updates, and whether NoThink-only
updates to the shared LoRA preserve fixed-probe CoT structure better than the
joint run. It does not introduce a new reward, teacher, hierarchy coefficient,
adapter split, or RL objective.

## Frozen parent and training contract

- Base: `/data/models/onereason-8b-pretrain-competition`.
- Adapter: fresh original
  `BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333`.
- RL checkpoint parent: none.
- Training route: NoThink only; Think rollout/reward/loss/optimizer updates: 0.
- Fixed Think probe: inference-only, using the joint run's group selection,
  seed and generation/evaluation contract.
- G8, temperature `1`, top-p `1`, LR `1e-6`, beta `0`, epsilon `0.2`, GRPO,
  `num_iterations=2`, credited-token SUM reduction, route multiplier `0.5`.
- Hierarchy: Earliest Domain Commitment then A/B/C, full-G8 milestone
  baselines, prefix-gated token credit, unchanged coefficients
  `0.25/0.5/1.5/6`, scale `8`.
- Domain commitment uses the declaration branch when matched and per-candidate
  direct-SID fallback otherwise.
- Dead-zero bridge: exact `[0]*8` to Gold-A only, lambda `0.02`.

## Formal run

Run ID: `GR-REC-NOTHINK-ONLY-HIER-G8BASE-E1-20260821`.

The full-epoch step is derived by the CPU sampler audit, not hardcoded.
Requested checkpoints are `250/500/666/750/1000/1250/final_epoch`; every
checkpoint must lie on a legal two-iteration rollout boundary. The formal run
uses four A800 GPUs. Step 50 and 100 record elapsed time, mean seconds per
optimizer step, and ETA without changing training.

The raw split contains 1549 groups. The shared four-group probe holdout leaves
1545 candidates; the parent joint sampler audit records one deterministic tail
group as dropped by its fixed chunk topology. This ablation explicitly reuses
that parent-audited exclusion so its training cohort is the same 1544 groups,
with no repeated group and no change to the two-unique-G8 rollout topology.

No automatic external benchmark is permitted. Post-run work is CPU-only and
must compare BATA, joint step 1000 vs NoThink-only approximately 666, and joint
step 1500 vs NoThink-only step 1000 using online metrics and fixed probes.

## Status

Implementation complete. CPU sampler audit: `1549` raw groups, `4` fixed
probes, `1` parent-parity tail exclusion, `1544` NoThink training groups,
`772` fresh rollouts, `num_iterations=2`, and `1544` optimizer steps. Derived
checkpoints are `250/500/666/750/1000/1250/1544`. GPU training has not yet
started at this code point.
