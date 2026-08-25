# Recommendation Training Coverage x SID Hierarchy Audit

## Audit contract

- Phase: Recommendation Root-Cause Phase 1.5.3
- Scope: CPU-only corpus/statistics audit; no model loading, generation, Beam32, training, or external evaluation
- Implementation commit: `2b193376f6dd8943a2fd59c349533639bd1c5e9a`
- Result commit: recorded by the commit containing this report
- Script SHA256: `406b2dcf2684551387eca1a83fb715aa24236be18822d7c650b27153ce7a08bd`
- Runtime/source/GitHub script parity: PASS at execution time
- Natural target pool: the fixed 1,595-group Phase 1.5.2 pool

## Source corpus inventory

| Corpus | Recommendation rows | Groups | NoCoT rows | CoT rows | SHA256 |
|---|---:|---:|---:|---:|---|
| BATA baseline | 48,269 | 20,531 | 16,090 | 32,179 | `f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca` |
| Mini-Fix | 11,192 | 1,549 | n/a | n/a | `6dc7660417f093b923c93d4f19734ed9240d5fba4c9beaaa01b6a164ccd94f7f` |
| Mini-Gamma | 11,192 | 1,549 | n/a | n/a | `216857d8c5d0049a3e9642279acd89051c40f5bcddb1d4afe513e36080b086cf` |

Mini-Fix and Mini-Gamma have identical group, all-gold, and current-gold contracts. Their answer suffix match rates are all `1.0`. The BATA corpus contains every group in the natural pool, so this pool is not a BATA held-out split. Mini training contains 159/1,595 target groups (`9.9687%`).

## Natural-pool coverage

The main measure is whether any gold SID from the target group appeared in another training group. This excludes trivial same-group exposure.

| Corpus | Domain | A coverage | AB coverage | ABC coverage |
|---|---|---:|---:|---:|
| Mini | video | 97.72% | 39.98% | 1.95% |
| Mini | prod | 46.64% | 1.35% | 0.45% |
| Mini | ad | 83.88% | 34.71% | 7.44% |
| Mini | living | 76.81% | 37.68% | 8.70% |
| BATA | video | 99.78% | 63.27% | 8.45% |
| BATA | prod | 80.27% | 5.38% | 1.35% |
| BATA | ad | 94.63% | 50.83% | 24.38% |
| BATA | living | 97.10% | 59.42% | 21.26% |

Natural-distribution weighted Mini coverage is `85.77%` at A, `33.48%` at AB, and only `3.45%` at ABC. This is direct evidence that coarse interest-family supervision exists while exact item-family supervision is sparse.

For Mini non-history golds, the fraction with no other-group ABC exposure is:

| Domain | Non-history ABC unseen |
|---|---:|
| video | 98.13% |
| prod | 99.45% |
| ad | 94.88% |
| living | 92.05% |
| weighted overall | 97.09% |

## Frequency and hierarchy branching

Mini exact-ABC reuse is concentrated near zero. The `ABC=0` target counts are 905/923 for video, 222/223 for prod, 224/242 for ad, and 189/207 for living. By contrast, AB prefixes are reused substantially more often, especially outside prod.

| Corpus | Domain | C-per-AB median | C-per-AB p90 | max | Weighted entropy |
|---|---|---:|---:|---:|---:|
| Mini | video | 1 | 1 | 16 | 0.176 |
| Mini | prod | 1 | 1 | 3 | 0.052 |
| Mini | ad | 1 | 1 | 11 | 0.283 |
| Mini | living | 1 | 2 | 20 | 0.969 |
| BATA | video | 1 | 2 | 51 | 0.476 |
| BATA | prod | 1 | 1 | 3 | 0.042 |
| BATA | ad | 1 | 2 | 44 | 0.755 |
| BATA | living | 1 | 2 | 60 | 1.319 |

BATA provides more other-group ABC opportunity than Mini by `+6.50 pp` video, `+0.90 pp` prod, `+16.94 pp` ad, and `+12.56 pp` living. This describes data opportunity only; it is not evidence that a particular trained model can exploit it.

## Phase 1.5.1 join

The existing Phase 1.5.1 SN/UN Beam records were joined without rerunning inference.

| Split | Groups | Same-group exact ABC seen | Other-group A | Other-group AB | Other-group ABC | Existing Beam GoldABC Hit@32 |
|---|---:|---:|---:|---:|---:|---:|
| SN | 4 | 4/4 | 3/4 | 3/4 | 0/4 | 0/4 for every model/route |
| UN | 4 | 0/4 | 3/4 | 1/4 | 0/4 | 0/4 for every model/route |

SN is the key counterexample to a coverage-only explanation: all four groups had at least one exact gold ABC supervised in the same group, yet the existing Beam32 diagnostic still failed to place any gold ABC in the top 32. The sample is small, so this supports a decoder-ranking limitation at moderate rather than strong confidence.

## Root-cause decision

- `DATA_COVERAGE_LIMIT_SUPPORT=STRONG`
- `DECODER_RANKING_LIMIT_SUPPORT=MODERATE`
- `COARSE_INTEREST_SIGNAL=PRESENT`
- `FINE_ITEM_SIGNAL=WEAK`
- `ROOT_CLASS=COVERAGE_PLUS_RANKING_BOTTLENECK`
- `BETA_TRUE_RECOMMENDATION_IDENTIFIABLE=NO`
- `BATA_HAS_MORE_FINE_ITEM_TRAINING_OPPORTUNITY=YES`

`VIDEO_FINE_ITEM_SPACE_IS_HARDER=NO`. Video is difficult, but it is not uniquely hardest under the audited hierarchy statistics: prod has lower exact-ABC coverage, while living has higher C-per-AB p90, higher entropy, and higher seen-target surprisal.

The bounded follow-up justified by this audit is a coverage-controlled, teacher-forced comparison of seen-ABC versus unseen-ABC targets. It is a recommendation only and was not started.

## Integrity and preservation

- Required output files: 15/15 present
- Phase 1.5.2 natural-pool/history-overlap parity: PASS
- Mini-Fix/Mini-Gamma group/gold contract parity: PASS
- Protected Phase 1.4.1, 1.5, 1.5.1, and 1.5.2 result hashes: unchanged
- `torch` imported: NO
- CUDA visible to audit: empty
- GPU inference started: NO
- Beam32 started: NO
- Training started: NO
- External evaluation started: NO
- Next experiment started: NO
