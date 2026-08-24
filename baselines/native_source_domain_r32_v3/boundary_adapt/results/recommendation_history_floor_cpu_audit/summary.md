# Recommendation Root-Cause Phase 1.5.2

## Contract

This was a CPU-only audit over versioned source data and persisted Phase 1.0-1.5.1 records. It did not load a model, import torch, use CUDA, generate Beam outputs, run external evaluation, train, or create a checkpoint.

- Final implementation commit: `f3b374b3e645a02aee84e9a27da622ddaf5c8a6a`
- Script SHA256: `326406af71b46c20a06c127b757bd945798b3713e713f55969d16e6a52df731a`
- Natural proxy: 1595 adaptation-heldout recommendation groups
- Phase 1.5.1 hierarchy input: 64 persisted cases, with original Beam32 order retained
- Existing Bare/Bridge copy audit: 184 persisted pairs across Phase 1.0, Phase 1.5-Quick, and Phase 1.5.1
- This is not an official-test distribution or official-score decomposition.

## Natural history overlap

All matches are strict target-domain SID matches. A and AB rates ask whether any Gold prefix exists in target-domain history; ABC is the complete SID match.

| Domain | N | Gold A in history | Gold AB in history | Gold ABC in history | H-K1 | H-K2+ | NH-K1 | NH-K2+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| video | 923 | 58.2882% | 16.7931% | 1.7335% | 2 | 14 | 256 | 651 |
| prod | 223 | 57.8475% | 24.2152% | 18.8341% | 38 | 4 | 147 | 34 |
| ad | 242 | 66.5289% | 28.5124% | 11.1570% | 22 | 5 | 183 | 32 |
| living | 207 | 44.4444% | 26.5700% | 14.9758% | 28 | 3 | 162 | 14 |

Video has by far the lowest exact-history rate, but its A-level overlap remains high. Thus video is not history-free at the interest-family level; it is mainly low in exact item repetition.

## Official artifacts

The bounded repository search found four versioned score records/artifacts: the README overview, Beta score record, MiniFix score record, and one unrelated GRPO external checkpoint score artifact. None contains official per-example prediction, Gold, history, and evaluator-supported score fields.

| Model | Video | Prod | Ad | Living | Rec sum | Project-file status |
|---|---:|---:|---:|---:|---:|---|
| Beta | 0.1223 | 0.1598 | 0.2072 | 0.1701 | 0.6594 | Available |
| MiniFix | 0.1297 | 0.1598 | 0.2030 | 0.1719 | 0.6644 | Available |
| Gamma | - | - | - | - | - | Missing |
| Sol | - | - | - | - | - | Missing |
| Step900 | - | - | - | - | - | Missing |

The project files do not explicitly confirm per-task maximum scores, so normalized scores were not calculated. MiniFix-to-Gamma, MiniFix-to-Sol, and Beta-to-Step900 domain-drop/history associations are `INSUFFICIENT`; no values were inferred from chat history or invented.

## NonHistory hierarchy

The table reports persisted Beam32 Hit@32. All ABC values are zero, but A-level signal remains, especially in SeenNonHistory.

| Model | Cell | Route | A@32 | AB@32 | ABC@32 | A MRR | AB MRR |
|---|---|---|---:|---:|---:|---:|---:|
| MiniFix | SN | Bare | 1.00 | 0.25 | 0.00 | 0.20972 | 0.01389 |
| MiniFix | SN | Bridge | 0.75 | 0.50 | 0.00 | 0.31250 | 0.02976 |
| MiniFix | UN | Bare | 0.50 | 0.00 | 0.00 | 0.25000 | 0.00000 |
| MiniFix | UN | Bridge | 0.50 | 0.00 | 0.00 | 0.07566 | 0.00000 |
| Gamma | SN | Bare | 0.75 | 0.25 | 0.00 | 0.14773 | 0.03571 |
| Gamma | SN | Bridge | 1.00 | 0.25 | 0.00 | 0.30955 | 0.01471 |
| Gamma | UN | Bare | 0.25 | 0.00 | 0.00 | 0.05000 | 0.00000 |
| Gamma | UN | Bridge | 0.25 | 0.00 | 0.00 | 0.04167 | 0.00000 |

This supports `A_PRESENT_AB_LIMITED_ABC_ZERO_IN_NONHISTORY`: the models are not completely devoid of recommendation signal, but fine SID/item discrimination is weak within Beam32. Nothing in this audit establishes deeper-than-32 global ranks.

## History retrieval and Bridge

For the eight exact-history SH/UH groups:

| Model | New Gold acquisition | Existing Gold rerank | Lost Gold | Unchanged |
|---|---:|---:|---:|---:|
| MiniFix | 1 | 2 | 1 | 4 |
| Gamma | 1 | 1 | 1 | 5 |

Across all 184 eligible persisted Bare/Bridge pairs, HistoryCandidateFraction changed by a mean of `-0.191916`: 133 decreased, 27 increased, and 24 were unchanged. Therefore `BRIDGE_IS_NOT_A_SIMPLE_COPY_SWITCH=SUPPORTED`. Bridge can acquire, lose, or rerank Gold while usually reducing the count of exact-history candidates.

## Root-cause decision

- `HISTORY_REPEAT_FLOOR_SUPPORT=SUPPORTED_LOCAL`
- `NOVEL_RECOMMENDATION_ABILITY=WEAK_WITHIN_BEAM32_LOCAL_PROXY`
- `HIERARCHICAL_RECOMMENDATION_SIGNAL=A_PRESENT_AB_LIMITED_ABC_ZERO_IN_NONHISTORY`
- `VIDEO_SUPPORTS_NOVEL_DIFFICULTY_HYPOTHESIS=PARTIAL`
- `MINI_DATA_SUPPORTS_REPEAT_NOT_NOVEL_HYPOTHESIS=PARTIAL`
- `BETA_TRUE_NOVEL_ABILITY_IDENTIFIABLE=NO`
- `ROOT_CLASS=LOCAL_HISTORY_FLOOR_SUPPORTED_OFFICIAL_ATTRIBUTION_UNRESOLVED`

The local evidence supports a meaningful exact-history floor and weak novel fine-item prediction. It does not establish that history cases explain the official MiniFix/Gamma or Beta/Step900 gap because official per-example artifacts and three models' domain score files are absent.

## Preservation and stop state

Phase 1.4.1, Phase 1.5-Quick, and Phase 1.5.1 protected hashes were identical before and after the audit. In particular, Phase 1.5-Quick retained `records.jsonl` SHA256 `1d74f95727bbb29a38e49c312efae473742f9cbc1ec4f079222e7bfce6d57731` and `seen_unseen_summary.json` SHA256 `342bde81ca26d5748597ec4328cfd14137efb0005423248486e0e378812acab0`.

- GPU inference started: NO
- Training started: NO
- Optimizer steps: 0
- Self-CoT generation started: NO
- External evaluation started: NO
- Next experiment started: NO

Future experiment is recommendation only, not launched: teacher-forced Gold ABC log-prob on SeenNonHistory versus UnseenNonHistory.
