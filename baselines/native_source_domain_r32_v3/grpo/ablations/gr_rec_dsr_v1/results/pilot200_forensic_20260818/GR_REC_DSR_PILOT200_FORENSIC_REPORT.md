# GR_REC_DSR PILOT200 Forensic Report

## Scope and verdict

This is a CPU-only, read-only forensic audit. No GPU, generation, training, resume, checkpoint execution, parser/reward/lambda change, 400-step run, or external benchmark was performed.

**Final research judgment: INSUFFICIENT EVIDENCE**

The logs support exact full-flow aggregate diagnostics but do not persist the baseline candidate/group records needed for the requested matched coverage, zero-variance transitions, affected Beam-task rate, or NoThink rotation analysis.

Fixed Probe has n=4 groups and is used only for fixed-prompt/fixed-seed longitudinal behavior. It is not used as population evidence.

## Data provenance and availability

| Evidence | Availability |
|---|---|
| DSR Think full flow | 34 rollouts, 136 groups, 544 candidates; aggregate Raw/Grounded histograms and coverage mean observed directly |
| DSR candidate CoT | 68-candidate systematic sample: first 2/4 candidates from first group of every Think rollout |
| Baseline candidate CoT, Step 0-200 | 2 Think groups from `GRPO_TRACE_EVERY=20` |
| Beam outputs | Exact aggregate invalid counts for all 17,408 outputs/run; task detail for only 8/544 tasks/run |
| Raw invalid Beam text | Unavailable |
| NoThink DSR plans | Concentration/rescue plans for 132 groups |
| NoThink matched baseline candidates | 3/132 groups |
| Full group-level primary outcomes | Unavailable; rollout aggregates only |

Values labeled unavailable were not reconstructed or guessed.

## Matched schedule

- Exact first-200 rollout alignment: `True`.
- Train seed: `20260816` in both runs.
- Optimizer schedule SHA-256: `ac86490e5660e365fe4adedb6ac9321cffb95db1636b2f4bafad99841ba508b7`.
- Matched schedule records: 268 route-group occurrences across 136 unique groups.
- Probe exclusions, sampler audit, route, group order, and occurrence order match.

## A. Coverage

**Conclusion: PROBE-LOCAL**

Full DSR training-stream aggregates:

| Window | Candidates | Raw N | Grounded N | Coverage | Defined |
|---|---:|---:|---:|---:|---:|
| 0-50 | 144 | 3.2083 | 2.7847 | 86.538% | 90.278% |
| 50-100 | 144 | 3.5208 | 3.1667 | 89.362% | 97.917% |
| 100-150 | 128 | 3.1328 | 2.7891 | 89.601% | 91.406% |
| 150-200 | 128 | 3.2656 | 2.8203 | 86.667% | 93.750% |
| Overall | 544 | 3.2868 | 2.8952 | 88.058% | 93.382% |

Coverage is 86.54% in 0-50 and 86.67% in 150-200; it rises in the middle windows rather than showing a monotonic collapse. The Fixed Probe 93.8%→76.0% warning is therefore not reproduced as a full-flow temporal trend.

Full-flow Raw/Grounded quantiles are available from exact histograms. Candidate-level Coverage and Gap quantiles are unavailable because their joint values were not persisted; the 68-candidate systematic sample is reported in `summary.json` and `think_candidate_coverage.csv`.

Baseline coverage can be recomputed with the same parser for only 2 matched Think groups. Their group mean Coverage deltas are 0.0000 and -0.0417. This is not enough for a population paired conclusion; the 5000-resample group bootstrap is included only to document the available sample, not to manufacture power.

## B. Evidence Coverage Loophole

**Conclusion: UNCERTAIN**

Systematic DSR candidate sample (68/544):

| Coverage bucket | Candidates | Groups | S_cot | A_aux>0 | A_aux<0 |
|---|---:|---:|---:|---:|---:|
| 1. Coverage = 1.0 | 48 | 30 | 0.9974 | 60.417% | 12.500% |
| 2. 0.75 <= Coverage < 1 | 4 | 4 | 1.0000 | 75.000% | 25.000% |
| 3. 0.50 <= Coverage < 0.75 | 6 | 6 | 1.0000 | 33.333% | 16.667% |
| 4. 0 < Coverage < 0.50 | 4 | 4 | 0.0000 | 0.000% | 100.000% |
| 5. Coverage = 0 | 1 | 1 | 0.0000 | 0.000% | 100.000% |
| undefined | 5 | 5 | 0.0000 | 0.000% | 100.000% |

A direct loophole-shaped case exists: Raw N=4, Grounded N=2, Coverage=.5, S_cot=1.0, A_aux=+1.73165. However, only 68 candidates have jointly persisted Coverage and A_aux, and only 6 fall in the 0.50-0.75 bucket. The prevalence claim 'low coverage candidates frequently receive positive advantage' is therefore uncertain.

### Low-Coverage / Positive-Aux real cases

#### Case 1: step 36 / d05fa0bdabcdbfdf757af0e76860078cfe741d3f8aaf30d12263d4238eaec977

Raw=4, Grounded=2, Coverage=0.5, D_cot=1.0000, S_cot=1.0000, S_aux=1.0000, A_aux=1.731651.

Titles: `["休闲零食", "游戏娱乐", "搞笑娱乐与情感共鸣", "生活与时尚"]`

Grounded flags: `[false, true, false, true]`

Evidence: `[[], ["<|prod_begin|><s_a_513><s_b_2632><s_c_7041>"], ["<|living_begin|><s_a_6640><s_b_7052><s_c_2856>"], ["<|prod_begin|><s_a_225><s_b_5934><s_c_2196>", "<|prod_begin|><s_a_1815><s_b_4026><s_c_4605>"]]`

Grounded evidence: `[[], ["<|prod_begin|><s_a_513><s_b_2632><s_c_7041>"], [], ["<|prod_begin|><s_a_225><s_b_5934><s_c_2196>", "<|prod_begin|><s_a_1815><s_b_4026><s_c_4605>"]]`

```text
<think>好的，收到您的指令。作为推荐系统分析助手，我将对该用户的行为历史进行分析，并提供我的洞察与预测。

【兴趣归纳】
根据用户的多域交互历史，可以归纳出以下几个核心兴趣类目：
1. **休闲零食**：此为用户最核心的兴趣，表现出极高的交互深度。用户不仅进行了多次搜索，还点击了多种零食商品，覆盖了膨化、辣味、休闲肉类等多个子类目。该兴趣贯穿了从内容探索到商品浏览的全链路，表明用户对这一品类有持续且强烈的消费需求。
2. **游戏娱乐**：用户对此类目表现出浓厚兴趣。交互行为包括搜索特定游戏、游戏相关商品<|prod_begin|><s_a_513><s_b_2632><s_c_7041>、游戏周边以及浏览游戏主播的直播内容。这表明用户是游戏内容的积极消费者，关注点可能涉及游戏攻略、主播动态及游戏文化。
3. **搞笑娱乐与情感共鸣**：用户有明确的搞笑、情感类内容消费偏好。交互行为包括多次搜索搞笑视频或文案，点击相关主题的搞笑直播<|living_begin|><s_a_6640><s_b_7052><s_c_2856>，并对带有幽默、温情或怀旧元素的商品产生兴趣。这反映了用户在娱乐消遣和情感连接方面的需求。
4. **生活与时尚**：该兴趣较为宽泛，包括对个人护理<|prod_begin|><s_a_225><s_b_5934><s_c_2196>、家居装饰<|prod_begin|><s_a_1815><s_b_4026><s_c_4605>、服饰<|prod_begin|><s_a_225><s_b_5934><s_c_2196>等商品的点击行为，以及对穿搭、舞蹈、美食等生活化内容的探索，表明用户对提升生活品质和个人形象有一定关注度。

【行为模式】
用户的消费偏好表现出“即时满足”和“怀旧”两个显著特征。其电商行为中频繁出现“限时”、“秒杀”、“爆款”、“怀旧”、“老味道”等关键词，显示出用户对高性价比、时效性商品和具有情感价值的童年回忆类商品有较强偏好。
跨域协同信号十分明确：用户在内容域(搜索)产生的兴趣，会直接或间接地导向电商域(点击商品)的行为。例如，用户在搜索游戏或搞笑视频后，会点击相关商品或观看主播直播，表现出“即看即搜即买”的转化路径。同时，用户的娱乐偏好(如游戏、搞笑)也与其对商品内容的关注点高度一致，形成了“兴趣-搜索-消费”的闭环。

【预测总结】
综合分析，用户当前处于对多种零食类目进行广泛浏览的阶段。尽管其兴趣广泛，但休闲零食是贯穿所有行为的核心。考虑到近期的高频点击和搜索行为，用户对“怀旧”和“限时”属性的商品接受度非常高。

因此，预测用户接下来最可能感兴趣并点击的内容，**有较大几率是带有“怀旧”、“童年记忆”或“老味道”标签的零食商品或短视频**。这类内容完美契合了用户在消费偏好上对情感价值和性价比的双重追求。此外，考虑到用户对游戏和搞笑内容的持续关注，一个**风格幽默的零食评测主播进行的直播**，也极有可能吸引用户的点击，因为它融合了用户的核心兴趣点(零食、搞笑)。</think>
```

Primary reward and Beam hits are unavailable for this task because it was not one of the `GRPO_TRACE_EVERY=20` generic traces.

## Counterfactual structure-score audit

S0=S_cot; S1=S_cot×Coverage; S2=S_cot×sqrt(Coverage); S3=0 below Coverage .5, otherwise S_cot. Raw N=0 maps to zero. This is an offline ranking audit, not a proposed algorithm.

Only 2 complete training groups were persisted. Eight Fixed Probe group-step observations are reported separately and are not pooled with training evidence.

| Scope | Groups | S1 top change | S2 top change | S3 top change |
|---|---:|---:|---:|---:|
| training_trace | 2 | 0.000% | 0.000% | 0.000% |
| fixed_probe | 8 | 0.000% | 0.000% | 0.000% |

Complete candidate scores, advantages, ranks, pairwise flips, positive-membership changes, low-coverage downgrades, high-coverage upgrades, and primary-high downranking are in `counterfactual_structure_scores.csv` and `summary.json`.

## C. Think Beam invalid

Correct denominator: 34 Think rollouts × 16 CoT tasks/rollout × 32 Beam outputs/task = 17,408 Beam outputs per run.

| Run | Invalid outputs | Output rate | Affected rollouts | Affected task rate |
|---|---:|---:|---:|---|
| DSR | 75 / 17408 | 0.431% | 8 / 34 | unavailable; bound 1.654%-7.904% |
| Baseline | 21 / 17408 | 0.121% | 6 / 34 | unavailable; bound 1.103%-3.860% |

Absolute delta: +0.310 percentage points. Relative ratio: 3.571x.

The full logs do not retain per-task invalid counts, so affected-task mean/median/p90/p95/max, top task concentration, top-20 tasks, and invalid text categories are unavailable. Mathematical bounds are: DSR severe tasks 0-7, critical 0-3; baseline severe 0-1, critical 0-0. The 8 directly traced tasks in each run all have zero invalid outputs. Raw invalid Beam strings were not persisted.

At rollout level, DSR invalids are concentrated: top 2/34 rollouts contain 85.333% of all DSR invalid outputs. This is a rollout proxy, not task concentration.

## D. Think matched zero variance

Baseline primary-zero groups: 61/136 (44.853%). DSR: 58/136 (42.647%). Aggregate delta: -2.206 percentage points.

DSR auxiliary rescue is directly aggregated for 52/136 groups (38.235%).

The group identities behind baseline and DSR zero counts were not persisted. Therefore the requested transition matrix, Primary Improvement, and baseline-zero∩DSR-aux-rescue count cannot be point-estimated. Exact compatible bounds are in `summary.json`. Only 2 matched trace groups are direct: one signal→signal and one signal→zero.

Conclusion: the evidence mainly demonstrates auxiliary rescue. It does not demonstrate a material matched primary-variance improvement.

## E. NoThink

Full matched candidate-level aggregate across 1,056 candidates:

| Metric | Baseline | DSR |
|---|---:|---:|
| Gold-A or better | 17.992% | 18.087% |
| Gold-AB or better | 3.030% | 3.977% |
| Exact | 1.610% | 1.989% |
| Wrong domain | 21.780% | 19.129% |
| Valid SID | 99.905% | 100.000% |
| Reward mean | 0.1766 | 0.2216 |

Gold-A-or-better is effectively flat (+0.095 percentage points), while wrong-domain falls 2.652 points and AB/Exact rise modestly. This is useful aggregate evidence but cannot distinguish productive exploration from wrong-A rotation at group level.

Stage-wise diagnostics:

| Window | Base zero-std | DSR zero-std | DSR all-zero | Rescue | Concentration | Gold-A group hit | Wrong domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-50 | 21.875% | 9.375% | 6.250% | 6.250% | 0.2695 | 40.625% | 30.078% |
| 50-100 | 21.875% | 6.250% | 6.250% | 6.250% | 0.2930 | 75.000% | 16.406% |
| 100-150 | 20.588% | 20.588% | 20.588% | 20.588% | 0.3199 | 64.706% | 20.588% |
| 150-200 | 23.529% | 41.176% | 38.235% | 38.235% | 0.3235 | 55.882% | 9.926% |

Only 3/132 groups retain candidate outputs in both runs. In that sample there are no productive-exploration or rotation-without-progress cases under the requested definitions; two groups lose a baseline Gold-A hit. The sample is too small and selection-fixed to estimate either rate.

Stage-wise zero-std, all-zero, rescue, concentration, Gold-A/AB/Exact group rates, six reward levels, validity, wrong-domain, domain/gold-count/unique-A plan strata, and direct sample classifications are in `summary.json` and `nothink_matched_groups.csv`.

## Required answers

- A. Coverage: **PROBE-LOCAL**.
- B. S_cot loophole prevalence: **UNCERTAIN**. A real loophole-shaped case exists, but prevalence is not identifiable.
- C. Beam invalid: DSR **0.431%**, baseline **0.121%**. Affected-task rate unavailable; strict bounds reported.
- D. Think: primary-zero aggregate improves slightly; most direct evidence is auxiliary rescue, not matched primary improvement.
- E. NoThink: productive exploration versus wrong-A rotation is not identifiable from 3/132 matched direct groups.

## Final research judgment

**INSUFFICIENT EVIDENCE**

The logs support exact full-flow aggregate diagnostics but do not persist the baseline candidate/group records needed for the requested matched coverage, zero-variance transitions, affected Beam-task rate, or NoThink rotation analysis.

No new training should be launched from this audit. A future matched forensic run would need candidate/group-level records for every rollout: group IDs, primary rewards, parsed CoTs, DSR scores/advantages, per-task Beam invalid counts/raw invalid text, and NoThink parsed SIDs for both baseline and DSR.

## Deliverables

- `GR_REC_DSR_PILOT200_FORENSIC_REPORT.md`
- `matched_groups.csv`
- `think_candidate_coverage.csv`
- `beam_invalid_tasks.csv`
- `nothink_matched_groups.csv`
- `counterfactual_structure_scores.csv`
- `summary.json`
