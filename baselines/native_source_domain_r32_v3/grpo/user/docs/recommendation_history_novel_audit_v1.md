# Recommendation History-vs-Novel CPU Audit v1

## Scope

This is a CPU-only task-structure audit. It does not test whether a trained model actually copies History SIDs.

- Dataset: `/data/GRPO/data/rec_mp_grpo_v2/train.jsonl`
- SHA256: `791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc`
- Raw rows / unique groups: 3098 / 1549
- Gold: authoritative `all_gold_sids` group metadata
- History: complete SIDs in `prompt`, extracted with existing `grpo_sid.all_sids`
- Think/NoThink route parity: PASS
- GR_USER Gold subset of History: 3000/3000 (100.00%)

## Domain results

| Domain | Groups | Gold mean | Gold median | History only | Mixed | Novel only | Mean history rate | Mean novel rate | Ceiling median | Ceiling=0 | Ceiling<0.25 | Ceiling<0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | 550 | 15.540 | 15.000 | 0.00% | 9.45% | 90.55% | 0.71% | 99.29% | 0.00% | 90.55% | 100.00% | 100.00% |
| Product | 382 | 2.429 | 2.000 | 3.40% | 18.59% | 78.01% | 11.90% | 88.10% | 0.00% | 78.01% | 78.80% | 82.46% |
| Ad | 427 | 2.513 | 2.000 | 2.81% | 20.84% | 76.35% | 12.34% | 87.66% | 0.00% | 76.35% | 77.28% | 83.84% |
| Live | 190 | 2.484 | 2.000 | 0.00% | 12.11% | 87.89% | 5.15% | 94.85% | 0.00% | 87.89% | 88.95% | 91.58% |

## Product / Live versus Video

- Product minus Video: mean novel Gold -0.1119; novel-only rate -0.1253; mean History-copy ceiling +0.1119.
- Live minus Video: mean novel Gold -0.0444; novel-only rate -0.0265; mean History-copy ceiling +0.0444.

## Interpretation

Across all Recommendation groups, mean novel Gold rate is 92.78%, novel-only rate is 83.21%, and mean History-copy recall ceiling is 7.22%.
GR_USER is structurally extractive in this dataset: every audited Gold SID is contained in its sample History. Recommendation is structurally predictive: most groups require at least one SID absent from History, and most are entirely novel.
Product novel rate higher than Video: NO. Live novel rate higher than Video: NO. The domain-specific Product/Live premise is therefore not supported; Video is the most novel domain in this dataset.
The overall task-structure conflict can still be supported even though the proposed domain ordering is false. This audit cannot show that GR_USER training actually increased History copying; that requires model-output evidence.

## Direct answers

1. Product / Live mean novel Gold rate higher than Video: **NO**. Both are lower than Video, by 11.19 and 4.44 percentage points respectively.
2. Product / Live have many novel-only groups: **YES** (78.01% / 87.89%).
3. Product / Live History-copy recall ceilings lower than Video: **NO**. Their mean ceilings are higher than Video, although all three are low in absolute terms.
4. User extraction versus Recommendation prediction task-structure conflict: **SUPPORTED overall**. GR_USER Gold is 100% contained in History, while Recommendation mean novel Gold is 92.78% and 83.21% of groups are novel-only.

Decision: `TASK_CONFLICT_SUPPORTED`

## Protection confirmation

- CPU-only
- No training
- No generation
- No model or checkpoint access
