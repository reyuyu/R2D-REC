# GR_REC_Think_CompositeInterest_v1

## Status

CPU-only provenance, parser, cohort, lexical calibration, synthetic metric audit, and tests completed on 2026-08-22.

DATA_PROVENANCE_READY = YES under the eligible-cohort policy.
LEXICAL_METRIC_READY = YES.
Formal runner construction and all GPU work remain intentionally out of scope.

No GPU, generation, optimizer step, smoke run, or training was executed.

## Motivation and scope

This Think-only ablation tests whether a balanced downstream Beam32 and structured-interest reward can resist the information collapse observed in earlier Think training. ExactClamp, NoThink, bridge, teacher, explicit diversity reward, and length reward are excluded.

Gold CoT is reward-side reference data only. It is never appended to generation prompts, Beam prompts, model inputs, or teacher-forcing targets.

## Gold provenance and eligible cohort

The GRPO source is /data/GRPO/data/rec_mp_grpo_v2/train.jsonl. Gold references are read from /data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl where source_segment equals recommendation_cot.

The join is exact equality on aux_metadata_json.recommendation_group_id. No fuzzy matching is used. Repeated rows are resolved only when their CoT prefix is byte-identical.

Counts:
- total Think groups: 1549
- exact safe Gold joins: 1458
- missing Gold: 91
- ambiguous or unresolved joins: 0
- parser-valid non-empty Gold: 1446
- excluded parser-invalid Gold: 12
- eligible Think groups: 1446

Eligibility is per group: route Think, exact safe Gold join, shared parser success, and at least one parsed interest. Missing or invalid groups are excluded; references are never generated or substituted.

## Parser calibration

Before the safe extension, the shared parser accepted 1360 of 1458 joined Gold CoTs. The 98 failures were:
- 86 missing_interest_heading results caused by observed safe title variants
- 12 empty_interest_section results caused by inline prose without line-delimited bullet boundaries

Supported title variants were extended narrowly:
- canonical bracketed Interest Summary with an optional terminal colon
- Markdown emphasis inside the brackets
- plain Markdown heading Interest Summary

All 86 title variants retain the existing numbered line bullets and were recovered. The remaining 12 inline-prose references have no unambiguous interest boundary and remain excluded. No whole-CoT or whole-paragraph fallback was added. Canonical parsing semantics are unchanged and regression-tested.

## Interest parser and evidence semantics

Gold and candidate CoTs use the same extract_interest_units function. Each unit includes order, raw text, normalized text, evidence SIDs, and evidence SIDs grounded in the current user prompt.

Normalization removes bullet numbers, SID serialization, Markdown emphasis, template markers, and redundant whitespace without deleting semantic words.

If a Gold interest has no grounded evidence, evidence is N/A for that pair and S_pair equals S_text. This makes every parser-valid Gold identity pair exactly 1. If Gold has evidence, S_pair remains 0.7 times S_text plus 0.3 times grounded evidence-set F1. Candidate-missing evidence is therefore penalized when the reference contains evidence.

## Lexical calibration protocol

Historical candidates came from GR-REC-CLAMP-BRIDGE-V1-G8BASE-FORMAL1500-20260821. Only exact-safe Gold groups with parser-valid Gold and parser-valid candidates were used.

For every candidate interest, the audit compared all same-group Gold interests and deterministically sampled five real Gold interests from other groups in the same target domain plus five from other domains. Seed was 20260822.

Metrics:
- M1: character bigram multiset F1
- M2: character unigram multiset F1
- M3: pooled namespaced character unigram and bigram multiset F1
- M4: current 8B tokenizer, special-token-free subtoken multiset F1
- M5: exact character LCS / ROUGE-L style F1

All metrics use the same normalized text and the same conditional evidence rule.

### Distribution summary

Values below are P50 / P75 / P90 / P95 for maximum pair score per candidate interest.

| Metric | Same group | Same-domain wrong group | Cross-domain wrong group | Same-vs-same-domain AUC |
|---|---|---|---|---:|
| M1 | .1806 / .2509 / .3235 / .3650 | .1183 / .1520 / .1895 / .2190 | .1132 / .1440 / .1800 / .2035 | .7202 |
| M2 | .3813 / .4640 / .5294 / .5656 | .3250 / .3722 / .4188 / .4485 | .3191 / .3656 / .4107 / .4412 | .6518 |
| M3 | .2813 / .3551 / .4250 / .4591 | .2193 / .2576 / .2994 / .3259 | .2136 / .2504 / .2889 / .3168 | .6925 |
| M4 | .3093 / .3838 / .4529 / .4942 | .2513 / .2927 / .3364 / .3682 | .2446 / .2857 / .3272 / .3580 | .6751 |
| M5 | .2430 / .3091 / .3764 / .4217 | .1923 / .2250 / .2642 / .2936 | .1887 / .2213 / .2564 / .2815 | .6747 |

M1 has the strongest rank separation.

### Threshold sweep

Each row lists threshold .20/.25/.30/.35/.40/.45/.50/.55/.60.

Historical candidate nonzero match rates:
- M1: .7424/.5236/.3125/.1647/.0642/.0236/.0034/.0008/.0000
- M2: .9907/.9713/.9181/.8269/.7035/.5481/.3640/.1740/.0625
- M3: .9493/.8649/.7247/.5380/.3226/.1503/.0532/.0152/.0034
- M4: .9696/.9037/.7973/.6453/.4417/.2551/.1208/.0372/.0118
- M5: .9130/.7838/.5701/.3302/.1731/.0718/.0245/.0059/.0008

Same-domain wrong-group any false-match rates over five sampled references:
- M1: .0788/.0212/.0056/.0012/.0000/.0000/.0000/.0000/.0000
- M2: .9801/.8929/.6541/.3602/.1485/.0483/.0168/.0068/.0012
- M3: .6606/.2867/.0996/.0283/.0081/.0028/.0003/.0000/.0000
- M4: .8291/.5125/.2179/.0747/.0246/.0087/.0022/.0003/.0000
- M5: .4371/.1463/.0411/.0112/.0022/.0000/.0000/.0000/.0000

Cross-domain any false-match rates:
- M1: .0567/.0128/.0028/.0006/.0000/.0000/.0000/.0000/.0000
- M2: .9785/.8776/.6152/.3257/.1242/.0427/.0137/.0037/.0009
- M3: .6205/.2534/.0766/.0202/.0050/.0009/.0003/.0000/.0000
- M4: .8138/.4630/.1787/.0576/.0190/.0059/.0019/.0003/.0000
- M5: .4094/.1227/.0293/.0075/.0009/.0000/.0000/.0000/.0000

The complete JSON also includes mean matched counts, pair-level false-match rates, matched-count histograms, P01-P99 distributions, and review examples.

## Selected metric

The selected S_text remains M1 character bigram multiset F1. MATCH_THRESHOLD is 0.30.

At 0.30:
- historical candidate nonzero match rate: 31.25%
- mean matched count: 0.3606
- same-domain sampled any false-match rate: 0.56%
- cross-domain sampled any false-match rate: 0.28%

Threshold 0.25 gives more signal but raises same-domain any false matches to 2.12%; 0.35 is more precise but reduces nonzero candidates to 16.47%. The conservative 0.30 point was selected from real positive and negative distributions, not from match rate alone.

The match-quality formula remains frozen at Q = clip((mean similarity - 0.60) / 0.40, 0, 1). Lowering the matching threshold does not lower the Q quality floor.

## Matching and reward

Matching is maximum-cardinality thresholded one-to-one matching with total similarity as deterministic tie-break. A candidate and Gold unit are valid only when S_pair is at least 0.30. One Gold unit cannot be consumed more than once.

K is valid match count. Coverage is K / N_gold and monitor-only precision is K / N_pred.

Coverage tier T:
- K = 0: 0
- 0 < coverage <= .25: .20
- .25 < coverage <= .50: .45
- .50 < coverage < 1: .70
- coverage = 1: 1.00

U_cot = .8 * T + .2 * Q.

U_beam = clip(log(1 + max(R_beam, 0)) / log(17), 0, 1), with invalid and non-finite inputs mapped to zero.

R_total = .60 * U_beam + .40 * U_cot.

Standard G4 advantage remains (R_total - mean) / (population_std + 1e-4), with exactly equal rewards producing exactly zero advantages.

## Historical selected-metric replay

Among 1188 historical candidates in eligible groups, 1184 were parser-valid (99.6633%).

At M1 threshold .30:
- nonzero match rate: .3125
- mean matched count: .3606
- matched counts: 0=814, 1=319, 2=45, 3=6
- U_cot mean: .0909
- U_cot P50/P75/P90/P95/P99/max: 0/.16/.36/.36/.80/.80

## Synthetic audit

On 128 parser-valid, domain/interest-count-stratified real Gold groups, mean U_cot was:
- identity: 1.0000
- reorder: 1.0000
- drop one: .5837
- drop half: .4463
- one interest: .5541
- duplicate one: .5543
- add one to three real Chinese Gold interests from other groups: 1.0000
- remove SID: .9029
- replace with wrong SID: .9029
- light lexical edit: .9542

Identity and reorder are highest; drop-half is below drop-one; duplicate does not improve coverage; real extra interests do not improve K; missing or wrong evidence lowers the score.

CPU runtime was about 2.24 ms per candidate and 8.98 ms per G4.

## Cohort topology

The original four fixed probes, deterministically recomputed with seed 20260818 in video/living/prod/ad order, are all eligible.

Static eligible-only topology:
- eligible groups: 1446
- fixed probes: 4
- post-probe groups: 1442
- sampler drop: 2
- training groups: 1440
- fresh G4 rollouts: 360
- optimizer steps with num_iterations 2: 720

This is static audit only. No formal runner was created.

## Data adapter and leakage guard

The read-only adapter now requires an explicit eligible_group_ids set. Non-eligible records are excluded. Every emitted record is asserted to have exact Gold, parser-valid non-empty interests, and a byte-identical original prompt. Eligible groups missing from source records or Gold fail closed.

Tests verify prompt parity and confirm Gold text is absent from model input prompts.

## Artifacts

- gr_rec_think_composite_interest_v1_gold_cot_provenance_20260822.json
- gr_rec_think_composite_interest_v1_similarity_calibration_20260822.json
- gr_rec_think_composite_interest_v1_offline_metric_audit_20260822.json

## Remaining gate

LEXICAL_METRIC_READY is YES for M1 at threshold .30. This does not authorize GPU work. The next phase still requires explicit code review and a separately authorized runner implementation.
