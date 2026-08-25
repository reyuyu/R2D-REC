# Recommendation Coverage-Conditioned Retrospective Beam Audit

## Contract

- Phase: Recommendation Root-Cause Phase 1.5.4
- Implementation commit: `847eda086c29424237f431cb5d0fe02928119b07`
- Result commit: recorded by the commit containing this report
- Execution: CPU-only retrospective analysis of immutable Beam32 records
- GPU inference, model forward, generation, training, optimizer, Self-CoT, and external evaluation: not started
- Primary sample contract: official-domain semantic prompt, Frozen BATA CoT, Bare domain, Beam32 ABC3
- Secondary contract: MiniFix native-prefix records, kept separate from the root decision

## Phase 1.5.3 reproducibility repair

The Phase 1.5.3 implementation commit was read directly from Git and its decision rule was rerun against immutable raw JSON artifacts.

| Field | Published | Recomputed |
|---|---:|---:|
| SN same-group exact ABC present rate | 1/4 (25%) | 4/4 (100%) |
| Decoder support | MODERATE | MODERATE |
| Root class | COVERAGE_PLUS_RANKING_BOTTLENECK | COVERAGE_PLUS_RANKING_BOTTLENECK |

`PHASE153_DECISION_REPRO_PASS=NO` because the published explanatory report encoded the SN rate incorrectly. The old implementation, raw coverage rows, decoder decision, and root decision are mutually consistent with 4/4. The correction changes only the derived Phase 1.5.3 report; raw training coverage, frequency, branching, and Beam artifacts retain their original SHA256 values.

## Record inventory and deduplication

| Item | Count |
|---|---:|
| Raw Beam records | 160 |
| Canonical records | 156 |
| Unique groups | 31 |
| Identical duplicates removed | 4 |
| Conflicting duplicates excluded | 0 |
| Primary homogeneous records | 124 |
| Secondary native-prefix records | 32 |

Deduplication uses `model + group_id + condition + sample_contract`. Bare and ExactBridge remain paired conditions and are never counted as independent groups.

## Covered NonHistory performance

The primary question is whether a gold ABC that Mini training actually supervised can be retrieved without a History shortcut.

| Model | NonHistory N | ABC covered N | ABC uncovered N | Covered Hit@32 | Covered MRR | Covered miss rate |
|---|---:|---:|---:|---:|---:|---:|
| MiniFix | 22 | 12 | 10 | 16.67% | 0.009343 | 83.33% |
| Gamma | 22 | 12 | 10 | 8.33% | 0.011905 | 91.67% |

For same-group-covered cases, miss rates are 81.82% for MiniFix and 90.91% for Gamma. Each model has two other-group-covered cases, and all four model-records miss at Beam32. That other-group-only subset is too small for a separate causal claim.

The 21 pooled covered misses decompose as:

- A-only: 38.10%
- AB-but-not-ABC: 38.10%
- No hierarchy hit: 23.81%
- AHit@32 overall: 76.19%
- ABHit@32 overall: 38.10%

This is evidence of a residual fine-item/C-ranking problem after coverage is present, but the cohort remains a selected diagnostic sample.

## Frequency, model, and Bridge sensitivity

Other-group ABC frequency does not show an improving Hit@32 trend in these records: frequency-zero Hit@32 is 7.50%, while all populated positive-frequency buckets are 0%. The bucket sizes are small, so the result is classified `MIXED`, not monotonic evidence.

On the same 12 covered groups, MiniFix MRR is `0.009343` and Gamma MRR is `0.011905`; the difference is `-0.002561`. This provides no descriptive advantage for MiniFix serialization on this cohort.

ExactBridge adds five covered gold acquisitions, reranks one already retrieved covered gold upward, and loses one covered gold. It also adds two uncovered gold acquisitions, so `BRIDGE_HELPS_RETRIEVE_LEARNED_ITEMS=MIXED`: Bridge is not exclusively retrieving learned item knowledge.

## Root-cause decision

- `DATA_COVERAGE_LIMIT_SUPPORT=STRONG`
- `RANKING_RESIDUAL_SUPPORT=MODERATE`
- `ROOT_CLASS=FINE_ITEM_DATA_COVERAGE_BOTTLENECK_WITH_RANKING_RESIDUAL`

The natural 1,595-group result remains primary: 97.09% of Mini NonHistory gold ABCs are unseen in other training groups. Existing Beam records provide 12 covered NonHistory cases per model with 83-92% miss rates, satisfying the predeclared MODERATE threshold (`N>=8`) but not the STRONG threshold (`N>=16`). Therefore coverage is the primary bottleneck and ranking is a residual, not a co-equal confirmed cause.

The bounded future recommendation is `TEACHER_FORCED_GOLD_LOGPROB_COVERED_HIT_VS_COVERED_MISS_VS_UNCOVERED_MISS`. It was not started.

## Integrity

- Required outputs: 14/14
- Phase 1.5.3 raw artifacts unchanged: YES
- Duplicate Beam conflicts: 0
- Runtime/GitHub implementation SHA parity: PASS
- GPU inference started: NO
- Model forward started: NO
- Training started: NO
- Optimizer steps: 0
- Self-CoT generation started: NO
- External evaluation started: NO
- Next experiment started: NO

