# GR_REC_ThinkDualBeam8_v2 — Change Record

## Formal-ready implementation (pending GPU validation)

- Implemented isolated CoT G4 and four independent SID G8 PPO-style branches.
- Added strict one-Beam8-per-CoT ABC3 validation, detached full-forward old-logp
  guards, and exact two-iteration rollout reuse.
- Pinned old Probe4 IDs and asserted 1549 raw / 1545 train / 3090-step topology.
- Added dual rollout/optimization monitoring without changing shared GRPO math.
- Added explicit final-step checkpoint, 29 CPU contracts and a 4-GPU zero-update
  preflight harness. Formal training remains prohibited pending review.

## 2026-08-28

### Motivation

Stopped the previous `gr_rec_think_suffix_sid_v1` direction before formal training.
The suffix-only design could optimize the answer continuation while leaving the sampled
CoT itself without a direct quality objective.

### New experiment design

Created isolated ablation:

`baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_dual_beam8_v2/`

Design changes:

1. Retain the existing held-out Probe4 evaluation set and exclude it from training.
2. Train Think route only.
3. Each business group produces one global stochastic G4 of CoTs.
4. On 4 GPUs, use one CoT per rank.
5. Each CoT is continued once with deterministic fixed-domain Beam8 strict ABC3.
6. Reuse the same Beam8 candidate set for both objectives.
7. CoT objective:
   - score each CoT with existing Beam aggregate `think_reward` over its Beam8 SIDs;
   - normalize four CoT rewards as one population-std G4;
   - apply PPO loss only to sampled CoT action tokens.
8. SID objective:
   - score each of the eight ABC3 candidates with existing NoThink-v1 `q_reward`;
   - normalize independently inside each CoT's own G8;
   - apply PPO-style loss only to the three ABC action tokens.
9. Never flatten the four SID groups into one G32 advantage group.
10. Default combined objective is `L_cot + L_sid` with 1:1 weights.
11. No zero-std reroll in this v2 design; zero population std gives zero advantages for
    that branch/group.
12. Keep `num_iterations=2`; the second policy iteration must reuse the same sampled
    CoTs, Beam8 candidates, rewards, advantages, and old log-probs.
13. Training Beam is Beam8, but the held-out fixed Probe remains on its existing Beam32
    evaluation contract.
14. Parent remains the immutable recorded-1.3510 `checkpoint-1500` with SHA guard.

### Reference scaffold commits

A lightweight reference scaffold was added to `main` while defining the design. It is
not formal-training-approved. Relevant commits include:

- `add6bcd1a5c2fc69b9922c3509ef599173052434` — create isolated ablation package;
- `982f34b0265521482d6b2914f7c7a817ae762fe0` — add dual Beam8 trainer scaffold;
- `3d963a80373583606b391deab28e5c4ed40955b0` — add formal runner scaffold;
- `793463aff930cf28a17d2298aaae940530cb69b9` — add launcher scaffold;
- `c96eb9ac09787030246230eb7376e9660b141ab5` — complete audit fields / document
  Beam-selected SID branch;
- `56de144d0f0cdddb021af6eb771ade81768c597e` — add Codex implementation handoff.

### Important interpretation

The CoT branch is stochastic-policy GRPO/PPO-style optimization based on G4 relative
CoT quality.

The SID branch uses deterministic Beam-selected candidates. It must be described as
**Beam-selected PPO-style / group-relative optimization**, not strict on-policy GRPO.

### Formal-training status

`FORMAL TRAINING APPROVED = NO`

Implementation validation must cover the following contracts:

- exact Probe4 retention / zero overlap;
- CPU contract tests;
- dual-objective monitoring;
- correct rollout/Beam reuse across `num_iterations=2`;
- full-forward old-logp semantics for both branches;
- branch-isolated gradient-boundary checks;
- 4-GPU zero-update preflight;
- explicit final checkpoint save.

No formal training should be launched as part of the implementation phase.
