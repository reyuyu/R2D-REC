# Recommendation Bridge x Training-Memory Interaction Audit

## Contract

- Phase: Recommendation Root-Cause Phase 1.5.5
- Final implementation commit: `b0226acc75a557340b40b8a06b0291be9ba05258`
- Result commit: recorded by the commit containing this report
- Execution: CPU-only analysis of immutable Phase 1.5 through Phase 1.5.4 artifacts
- Canonical input SHA256: `c5c72a0d780046d75cd692c70c2267740c11117b2a44d69d0e6640af6f0de86a`
- Primary contract: `OFFICIAL_DOMAIN_SEMANTIC_FROZEN_BATA_COT_ABC3`
- Primary cohort: GoldNotInHistory with same-group exact-ABC supervision
- Resampling unit: unique group carrying MiniFix/Gamma x Bare/ExactBridge records

The first implementation commit treated an old manifest field as a bridge-text hash. The historical producer defines it as a token-id JSON hash. Commit `b0226acc...` restores the original token hash contract and separately records a UTF-8 bridge-bytes hash for parity. The failed preflight stopped before analysis outputs were generated.

## Context and cohort audit

- Primary eligible groups: 11
- Four-way complete groups: 11
- Dropped incomplete groups: 0
- Bridge context parity: PASS
- Protected raw artifacts unchanged: YES

For each group, MiniFix and Gamma use the same official prompt, Frozen BATA CoT, exact old bridge bytes, domain token IDs, and gold set. Native-prefix secondary records are excluded.

## Core 2 x 2

| Model | Condition | N | Hit@32 | MRR | AHit@32 | ABHit@32 | History candidate fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| MiniFix | Bare | 11 | 18.18% | 0.010193 | 90.91% | 54.55% | 54.83% |
| MiniFix | ExactBridge | 11 | 27.27% | 0.018398 | 81.82% | 63.64% | 21.88% |
| Gamma | Bare | 11 | 9.09% | 0.012987 | 81.82% | 45.45% | 23.30% |
| Gamma | ExactBridge | 11 | 36.36% | 0.024900 | 90.91% | 54.55% | 20.74% |

MiniFix MRR gain is `+0.008205`; Gamma MRR gain is `+0.011913`. The difference-in-differences is `-0.003707`. Hit@32 gains are `+9.09 pp` and `+27.27 pp`, producing a `-18.18 pp` interaction.

The deterministic 10,000-repeat paired group bootstrap gives:

- MRR interaction 95% percentile CI: `[-0.017894, 0.012196]`
- `P(interaction > 0) = 0.3009`
- Interpretation: descriptive only; selected diagnostic cohort

## Acquisition and hierarchy

| Model | New gold | Lost gold | Rerank up | Rerank down | Unchanged miss |
|---|---:|---:|---:|---:|---:|
| MiniFix | 2 | 1 | 1 | 0 | 7 |
| Gamma | 3 | 0 | 0 | 1 | 7 |

Both MiniFix NEW_GOLD cases start from a Bare AB hit. Gamma has two NEW_GOLD cases from Bare AB hit and one from Bare A-only. No acquisition starts from a complete hierarchy miss. This supports a fine-C ranking/state adjustment interpretation more than a fresh reconstruction of user interest.

MiniFix's history-candidate fraction falls by `-0.3295` after Bridge; Gamma falls by `-0.0256`. The local gold improvements therefore are not accompanied by increased exact history copying.

## Controls

The 10 uncovered NonHistory groups show MiniFix/Gamma MRR gains of `+0.003333/+0.006250`, with interaction `-0.002917`. The direction is similar to the covered cohort rather than specific to same-group memory opportunity.

Only one group is `OTHER_GROUP_ONLY`; it is retained as a group-level listing and classified underpowered. No inferential claim is made from it.

## Decision

- `MINIFIX_SPECIFIC_MEMORY_KEY_SUPPORT=UNSUPPORTED`
- `SHARED_BRIDGE_CONDITIONER_SUPPORT=SUPPORTED`
- `BRIDGE_CONDITIONED_FINE_C_RANKING_SUPPORT=SUPPORTED`
- `BRIDGE_MEMORY_INTERACTION=NO_MINIFIX_SPECIFIC_ADVANTAGE`

Existing local Beam records do not support a MiniFix-specific old-bridge memory advantage. Both models respond to ExactBridge, and Gamma responds at least as strongly. The covered and uncovered controls have similar interaction directions, which is more consistent with a shared decoder-state conditioner than a MiniFix-only parameter-memory key.

This is local mechanism evidence only. `OFFICIAL_GAP_EXPLAINED=NO` because no official per-example History/coverage decomposition exists.

No future GPU experiment is required for this CPU conclusion. If revisited, the bounded experiment is a coverage-matched teacher-forced gold-logprob four-way comparison; it was not started.

## Integrity

- Required outputs: 12/12
- Runtime/GitHub implementation SHA parity: PASS
- Protected raw artifacts unchanged: YES
- GPU inference started: NO
- Model forward started: NO
- Training started: NO
- Optimizer steps: 0
- Self-CoT generation started: NO
- External evaluation started: NO
- Next experiment started: NO
