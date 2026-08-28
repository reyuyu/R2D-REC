# OneReason 多任务 SFT

本仓库记录 OneReason-8B 的数据版本、训练代码、可复现实验配置和评测结果。当前重点进入 **GRPO 系列**：以 BETA-baseline 为策略初始点，针对懂推荐进行 outcome 优化；Alpha、Mini 和 BETA-baseline 作为前置 SFT 实验与统一参考线继续保留。

仓库不提交模型权重、原始 JSONL、tokenized cache、日志、checkpoint 或任何凭据；这些内容保留在开发机，仓库只保留代码、配置、测试和可审计的版本元数据。

## 背景

模型需要同时学习三类主要能力：

- 懂物料：理解商品、视频、直播、广告等域的 SID 与文本描述；
- 懂用户：完成 Action Select 和用户行为链推理；
- 懂推荐：根据用户历史和画像生成推荐 SID 或推荐解释。

实验的核心原则是：每次只改变一个明确记录的变量，并把数据版本、loss route、packing、训练步数和评测结果一起记录。训练 loss 主要用于观察优化过程，最终效果以固定评测器输出的总分和分项分数为准。

## 统一参考线：BETA-baseline

当前所有 Alpha 和 Mini 实验都以 **BETA-baseline 纯净版** 为参考。新版 baseline 文档文件名仍沿用了历史的 `BATA_BASELINE.md` 命名。

| 项目 | 固定设置 |
| --- | --- |
| 数据 | `bata_baseline_v1`，共 222,001 条；不含 world |
| 模型 | Native Source-Domain R32，LoRA r32 / alpha64 / dropout 0.05 |
| 训练 | 4 GPU、8K neat packing、global batch 64、2 epoch |
| 优化 | AdamW、学习率 `2e-4`、cosine、warmup `0.03`、GC `0.4` |
| 损失 | 原生 one-hot SID8 CE；普通 token=1、非 canonical SID/domain=8、canonical response=4 |
| Epoch 1 | 总分 `1.3090` |
| Epoch 2 | **总分 `1.3246`，约 1.33 分参考线** |
| GRPO 比较基线 | **多次评测中较高的一次 `1.3313`** |

外部评测自身约有 `±0.01` 的正常波动。`1.3246` 是历史 Epoch 2 首次记录；GRPO 使用同一模型多次评测中较高的 `1.3313` 作为保守对照，二者差异不代表模型发生变化。新实验必须同时报告总分和分项分数，不能仅凭 raw task loss 或 `0.01` 内的单次差异判断优劣。

详细合同和结果：[BETA-baseline 纯净版](./baselines/native_source_domain_r32_v3/docs/BATA_BASELINE.md)。

## GRPO 系列（当前重点）

### 系列动机与实验逻辑

SFT 系列已把总分稳定推到约 1.33，但 Mini 实验表明，单纯扩充懂用户数据或延长 epoch 不能稳定换取收益：Mini-Fix-U3K 和 Mini-Fix-E3 均为负结果。GRPO 系列因此不再同时改数据、训练轮数和 loss，而从 BETA-baseline 出发，只针对懂推荐的实际生成结果进行 outcome 优化，并用外部 11 项评测检查收益是否外溢或损害其他能力。

### 方案总览

当前 GRPO 不是单一算法，而是四条严格隔离的研究路线。它们共享 OneReason-8B、LoRA-only 更新、group-relative advantage、PPO ratio/clipping 和被动监控，但训练路由、credit placement 与研究问题不同。

| 路线 | 代表实验 | 参与优化的路由 | 主要信号 | 研究问题 |
| --- | --- | --- | --- | --- |
| **混合懂推荐 GRPO** | `GR_REC_v1`、`GR_REC_DSR_Ablation_v1`、`GR_REC_ThinkExactClamp_Ablation_v1` | Think + NoThink | Beam outcome；SID outcome 或分层 token credit | 两条推荐路径联合训练能否提高外部懂推荐分数 |
| **NoThink-only GRPO** | `GR_REC_NoThinkOnly_Hier_v1`、`GR_REC_NoThinkOnly_Frontier_v1` | 仅 NoThink；Think 只做 probe | Domain/A/B/C credit、dead-zero bridge、首错 frontier | 直接 SID 路径能否独立提升 |
| **Think-only GRPO** | `GR_REC_Think_CompositeInterest_v1`、`GRPO-TK` | 仅 Think | Beam/兴趣结构，或 CoT 后独立 Sample8 FullSID outcome | 如何把最终 SID 质量反向归因给 CoT，同时保持两级动作边界 |
| **懂用户 GRPO** | `GR_USER_v1`、`MC_USER_v1`、`MC_USER Hybrid K4` | Action + Chain | Set-F1、Action/Logic alignment、局部 token credit | 懂用户提升及共享 LoRA 的跨任务干扰 |

不同路线会改变 loss 尺度，不能横向比较训练 loss 的绝对值。最终判断统一依赖固定 probe、结构监控与外部 11 项评测。

### 共同底座与基础数学

`GR_REC_v1` 是懂推荐实验的母合同，也是当前外部最高分方案的来源：

- Think：`G=4`、temperature `0.9`、top-p `0.95`、route multiplier `1.0`；每条 sampled CoT 再执行 Beam32。基础 reward 为 Exact=`8`、AB=`2`、A=`0.5`，同一 Gold 前缀去重并按排名几何衰减。
- NoThink：`G=8`、temperature `1.0`、top-p `1.0`、route multiplier `0.5`；按最终 SID 使用互斥六档 reward `-1/-0.25/0/0.5/2/8`。
- 组内 advantage 为 `A=(R-mean)/(std_population+1e-4)`；严格同分组 advantage 为 0。
- PPO/GRPO 固定为 `beta=0`、`epsilon=0.2`、`loss_type=grpo`；一个 rollout 复用两个 optimizer steps。base 冻结，只更新 LoRA。
- `G4 x 1.0` 与 `G8 x 0.5` 的 trick 是让两条路线每个 group 的总权重大致可比，不是修改 NoThink reward 数值。

基础版 Think Beam32 会生成较长 continuation 再解析最终 SID。后来的 Fixed-domain Strict ABC3 是新的 Think-only 语义，不能反向重解释 `GR_REC_v1` 的历史结果。

### 混合懂推荐 GRPO

#### GR_REC_v1：Outcome 混合母版

Think 用 Beam32 outcome，NoThink 用六档 SID outcome，两条路线交替训练并共享 LoRA。关键 trick 是动态 `G4/G8`、route weight 对齐、精确 `</think>` stopping、从真实 completion IDs 重建 Beam context，以及只在完整 rollout 边界保存/恢复。

后期风险是 Think CoT 变短、Raw N/Grounded N 下降及 zero-std 上升，因此 checkpoint 不能按“越晚越好”选择。

#### GR_REC_DSR_Ablation_v1：Diversity & Signal Rescue

DSR 保持 `GR_REC_v1` primary reward 不变，只增加隔离辅助项：

- Think 仅认可包含原 prompt 完整真实 SID 的 grounded interest bullet；用兴趣数量、evidence Jaccard diversity、Beam prefix/exploration 构造独立组归一化 advantage，再以 `0.10` 加入同一 PPO surrogate。
- NoThink 仅在真实 `[0]*8` 组触发 sampled-A token unlikelihood；按重复频率加权，`lambda_A` 根据 Gold-A 多样性取 `0.10/0.20`。
- 不增加 forward、generation、Beam、collective 或 CUDA sync。

该方案研究“结构坍缩”和“dead group 无梯度”能否分别获救。Pilot200 forensic 已落盘，但不是新的外部最佳分数结论。

#### Think ExactClamp + NoThink Hierarchical Bridge

尽管目录名是 `GR_REC_ThinkExactClamp_Ablation_v1`，正式组合版同时训练两条路线：

- Think：`raw=(reward-group_mean)/8`；当 `reward>=8` 且 `raw<0` 时 clamp 为 0，避免 Exact candidate 收到负更新。
- NoThink：sequence-wide scalar advantage 改为稀疏 token credit。Domain/A/B/C 系数为 `0.25/0.5/1.5/6`，统一除以 8，并按前缀条件逐级 gating。
- Domain credit 位于 `</think>` 后、最终 SID 前的自然语言 decision token；A/B/C 位于最终 SID component token。无关 token advantage 为 0，credited-token loss 使用 SUM，避免 completion 长度稀释。
- 仅精确 `[0]*8` 启用 Gold-A teacher bridge，`lambda_bridge=0.02`；不教 B/C，不与旧 sequence objective 双计数。

该组合版 checkpoint 1000 外部得分 `1.3383`，仍在约 `+/-0.01` 波动参考内；checkpoint 1500 回落到 `1.3064`。工程稳定不等于有效，晚期 Think 结构收缩仍是主要风险。

### NoThink-only GRPO

#### NoThinkOnly Hier

从 fresh BETA/BATA parent 只训练 NoThink，没有 Think rollout/reward/loss/optimizer update；Think 仅做固定 inference probe。算法沿用 earliest Domain commitment + A/B/C hierarchical credit、`0.25/0.5/1.5/6`、scale 8、route multiplier `0.5` 与 `[0]*8` Gold-A bridge `0.02`。

它用于严格归因：NoThink 改善而 Think probe 下降时，说明共享 LoRA 仍会产生跨路由干扰。仓库保留 runner、sampler audit 和正式合同，效果以对应结果记录为准。

#### NoThinkOnly Frontier

Frontier 在 Hier 上只改两点：严格格式 gate，以及“第一个失败层级”的绝对负 credit。

- 合法输出只能是 SFT domain declaration + 完整 SID，或 direct-SID fallback；非空 think、额外 prose、非法 SID、模板错误触发固定总质量 sequence penalty。
- 合法 candidate 保留正向 milestone credit；Domain/A/B/C 首个失败点分别给 `-0.03125/-0.0625/-0.1875/-0.75`，后续层级 gated。
- `[0]*8` 时 Frontier A-negative 与独立 Gold-A bridge 可同时存在；bridge lambda 仍为 `0.02`。

正式 full epoch 已完成 `1544` steps，工程稳定，但晚期 zero-std 上升，未训练 Think probe 的 Exact/Beam 明显退化。未运行外部 benchmark，因此只能判定“训练链成立、效果未定”。

### Think-only GRPO：Composite Interest

该路线只训练 Think G4。它把 sampled CoT 的 `【兴趣归纳】` 与 reward-only Gold CoT 做一对一兴趣匹配：

```text
K = matched interest count
coverage = K / N_gold
Q = clip((mean_similarity - 0.60) / 0.40, 0, 1)
U_cot = 0.8 * coverage_tier + 0.2 * Q
U_beam = clip(log(1 + max(R_beam, 0)) / log(17), 0, 1)
R_total = 0.60 * U_beam + 0.40 * U_cot
```

核心 trick：

- 字符 bigram multiset F1，阈值 `0.30`；maximum-cardinality 一对一 matching，total similarity 仅作确定性 tie-break，防止重复消费一个 Gold interest。
- Gold CoT 只进入 reward，绝不进入 prompt、Beam context 或 generation。
- 当前 production Beam 是 Fixed-domain Strict ABC3：`Prompt + sampled CoT + fixed target-domain prefix` 后 Beam32，严格生成 3 个 raw token（A/B/C），`min_new_tokens=max_new_tokens=3`，reward 直接解析 raw IDs。
- 旧 free-domain/128-token/generic `final_sid` 路径出现过 long-continuation parse artifact，只能作工程历史。
- Monitor 展示 Beam raw、`U_beam`、`U_cot`、Composite、K、Raw N、Grounded N、grounding coverage 和 lazy-loaded 32 条真实 Beam；前端不重算 reward。

Composite 已完成 CPU 校准、4-GPU zero-update preflight 与 Smoke12 合同验证，但尚无可替代 `1.3510` 的完整外部评测。若 parent 使用 plus-gamma epoch1/epoch2，现有 12 个固定 probe 中有 `8/12` 的 group ID 与训练数据重合，不能作为干净泛化集，正式比较需重建 group-disjoint probe。

### Think-only GRPO：GRPO-TK（Sample8 FullSID）

`GRPO-TK` 的工程名为 `GR_REC_ThinkSample8_FullSID_v3`。它不再用 fixed-domain Beam8 评价 CoT，而让每条 CoT 后的模型自行生成完整 SID：

```text
1 business group
  -> stochastic G4 CoT
  -> each CoT independently samples 8 continuations
  -> scan the first complete domain+A+B+C SID
  -> 4 independent SID G8 advantages
  -> CoT reward = sum(its 8 SID rewards)
  -> global CoT G4 advantage
  -> L_total = L_cot + L_sid
```

核心 trick：

- SID 上下文只有 `prompt + sampled CoT through </think>`，不固定 target domain，也不插入自然语言 bridge。
- continuation 内允许先出现自然语言；解析器扫描第一个完整 raw-ID SID。多 SID 只用第一个打分并标记，未找到完整 SID记 `-1`。
- SID 六档 reward 为 `-1/-0.25/0/0.5/2/8`。每条 CoT 的 G8 单独 population-normalize，四组不能 flatten 成 G32。
- CoT reward 是对应 8 个 SID raw reward 的算术和，四条 CoT 再做全局 G4 population advantage。
- CoT loss 只更新 CoT action tokens；SID loss 只更新第一个完整 SID 的 4 个 tokens。Base 冻结，只训练 LoRA。
- `num_iterations=2` 的第二次 policy pass 完整复用第一次 rollout 和 detached full-forward old logp；当前正式版本不做 zero-std reroll。

当前 parent 三次外部评测均值为 `1.3481`，GRPO-TK checkpoint-250 单次为 **`1.3579`**：相对 parent 均值 `+0.0098`，相对 parent 三次最好值 `1.3510` 为 `+0.0069`。这是当前仓库最高的已记录单次总分，但尚未完成同 checkpoint 多次复测，因此记录为早期正向信号，而不是统计复现后的最终结论。

完整公式、11 项分数和分项分析见 [GRPO-TK 实验记录](./docs/experiment_GRPO_TK.md)；从私有数据契约到 4-GPU 启动、恢复与监控的步骤见 [GRPO-TK 复现指南](./docs/reproduce_GRPO_TK.md)。

### 懂用户系列 GRPO

懂用户项目与懂推荐代码隔离，包含 Action 与 Chain 两条 NoCoT 路由：

- Action reward 是合法 SID 集合的 Set-F1。
- Chain reward 是 `0.5 * ActionAlignmentF1 + 0.5 * LogicAlignmentF1`，使用保序一对一 event matching。
- `G=4`、temperature `0.9`、top-p `0.95`、population-std advantage、`beta=0`、clip `0.2`。
- 对 hallucinated SID、date/action mismatch、duplicate 等白名单局部 span，使用 `lambda_eff=0.50/sqrt(token_count)`；token advantage 取 `min(sequence_advantage,-lambda_eff)`，重叠处罚只取最强值。
- `wrong_selection_sid` 与 format/schema violation 保持诊断语义，不被隐式加入局部 penalty。

`GR_USER_v1` full epoch 的内部 Action/Chain probe 改善，但外部总分为 `1.3146`，主要因为 Recommendation 相对 BETA 下降 `0.0213`。`MC_USER Hybrid K4` 从 `GR_REC_v1 step1500` 加入懂用户目标后，User 上升而 Recommendation 持续下降，说明共享 LoRA 的跨任务干扰不能只靠目标任务 probe 判断。

当前实验链路如下：

1. **BETA-baseline（GRPO 基线）**：不是新的训练实验，而是 GRPO policy 的初始 Adapter。使用多次评测中较高的 `1.3313` 作保守对照，动机是避免以偏低测次夸大 GRPO 收益。
2. **GR_REC_v1（首个正式 GRPO 实验）**：在冻结 BETA SFT Adapter 合同的基础上，只训练 recommendation group。Think 路由用 G=4 与 Beam32 outcome reward，NoThink 路由用 G=8 SID 分档 reward；通过 route weight 保持两条路由每 group 权重可比。它要回答的核心问题是：在不修改 reward、采样、Beam 和 PPO/GRPO 数学的情况下，生成侧 outcome 信号能否提高懂推荐外部得分。
3. **Checkpoint 选择**：不按“越晚越好”选择，也不只看训练 reward。先比较 Step 1000/1500/2000/final 的外部 11 项，再结合 CoT 长度、多样性、zero-std、Beam invalid 和固定四域 Probe。当前 Step 1500 是优先复测候选。
4. **后续实验隔离原则**：CoT 后期单一兴趣收缩与 NoThink/Think 零方差是两类不同问题。后续若分别测试多样性约束或稀疏 reward 改进，必须单变量立项，不在同一实验中同时改 G、reward、sampler 或数据顺序。

### 当前最高单次记录：GRPO-TK Step 250 = 1.3579

GRPO-TK 使用 `GR_REC_v1 checkpoint-1500` parent。该 parent 三次同口径外部评测分别为 `1.3436 / 1.3497 / 1.3510`，均值 `1.3481`；GRPO-TK step 250 得到 `1.3579`。

| 模型 / Step | 总分 | 懂物料合计 | 懂用户合计 | 懂推荐合计 | 懂世界 | 相对 parent 三次均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent 三次均值 | `1.3481` | `0.1814` | `0.2563` | `0.6773` | `0.2331` | - |
| **GRPO-TK Step 250** | **`1.3579`** | `0.1822` | `0.2575` | **`0.6840`** | `0.2342` | **`+0.0098`** |

净增量主要来自懂推荐合计 `+0.0067`；懂物料、懂用户和懂世界分别约为 `+0.0008/+0.0012/+0.0011`。由于 GRPO-TK 目前只有一次外部评测，该结果需要补齐重复测量并结合后续 checkpoint 才能判断稳定性。

### 此前基础混合高光：GR_REC_v1 Step 1500 = 1.3510

> **`1.3510` 是基础混合路线 `GR_REC_v1 checkpoint-1500` 的高光结果，也是 GRPO-TK 的 parent 三次评测最好值。** 它相对保守 BETA 基线 `1.3313` 提升 `+0.0197`，其中懂推荐合计从 `0.6594` 提升到 `0.6800`。这是优先复测和模型选择候选，不是已完成统计复现的最终结论。

| 模型 / Step | 总分 | 懂物料合计 | 懂用户合计 | 懂推荐合计 | 懂世界 | 相对 `1.3313` | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BETA-baseline | `1.3313` | `0.1807` | `0.2545` | `0.6594` | `0.2368` | - | 保守比较基线 |
| GR_REC_v1 Step 1000 | `1.3270` | `0.1817` | `0.2550` | `0.6563` | `0.2338` | `-0.0043` | 波动范围内，视为持平 |
| **GR_REC_v1 Step 1500** | **`1.3510`** | `0.1816` | `0.2563` | **`0.6800`** | `0.2331` | **`+0.0197`** | 当前最佳，需重复评测 |
| GR_REC_v1 Step 2000 | `1.3326` | `0.1806` | `0.2581` | `0.6607` | `0.2331` | `+0.0013` | 波动范围内，视为持平 |

Step 1500 的总分增量 `+0.0197` 和懂推荐合计增量 `+0.0206` 超过单次 `0.01` 波动带，是目前唯一明确值得复测的信号；Step 1000 和 Step 2000 都只能判为与基线同一水平。Step 1500 之后的回落与监控中 CoT 变短、兴趣覆盖收缩和 zero-std 上升方向一致，但当前只是相关证据，不作因果结论。`checkpoint-2316` 尚待外部评测。

### GR_USER_v1 外部评测

| 模型 / Step | 总分 | 懂物料合计 | 懂用户合计 | 懂推荐合计 | 懂世界 | 相对 `1.3313` | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BETA-baseline | `1.3313` | `0.1807` | `0.2545` | `0.6594` | `0.2368` | - | 保守比较基线 |
| GR_USER_v1 Step 240 | `1.3103` | `0.1780` | `0.2577` | `0.6375` | `0.2372` | `-0.0210` | 懂用户小幅上升，但推荐下降 `0.0219`，支持跨任务干扰 |
| GR_USER_v1 Final 378 | `1.3146` | `0.1803` | `0.2572` | `0.6381` | `0.2390` | `-0.0167` | 相比 step240 回升 `0.0043`，推荐仍下降 `0.0213` |

Step 240 和 final 378 的训练内固定 Probe 均比 step 40 更好，但外部总分下降。两者并不矛盾：固定 Probe 只覆盖 GR_USER 目标分布，外部评测同时检查 11 个任务；两次外部评测的损失都集中在未训练的 Recommendation。Final 没有继续恶化，相对 step240 回升 `0.0043`，但也没有恢复推荐能力，因此完整 epoch 结果进一步支持跨任务干扰判断。详细结果见 [GR_USER full-epoch 文档](./baselines/native_source_domain_r32_v3/grpo/user/docs/full_epoch_v1.md)。

供独立复核的完整证据、假设分级和网页版 GPT 提示词见 [GR_USER 外部分数回退证据包](./baselines/native_source_domain_r32_v3/grpo/user/docs/gr_user_v1_external_regression_evidence.md)。

Recommendation Gold 相对 History 的纯 CPU 任务结构审计见 [Recommendation History-vs-Novel Audit](./baselines/native_source_domain_r32_v3/grpo/user/docs/recommendation_history_novel_audit_v1.md)。

### MC_USER Hybrid K4 外部评测

| 模型 / Step | 总分 | 懂物料合计 | 懂用户合计 | 懂推荐合计 | 懂世界 | 相对 Parent `1.3510` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent GR_REC_v1 Step 1500 | `1.3510` | `0.1816` | `0.2563` | `0.6800` | `0.2331` | - |
| MC_USER Hybrid Step 256 | `1.3472` | `0.1842` | `0.2597` | `0.6690` | `0.2342` | `-0.0038` |
| MC_USER Hybrid Step 512 | `1.3402` | `0.1830` | `0.2623` | `0.6588` | `0.2361` | `-0.0108` |

Hybrid 训练健康完成，但外部任务呈现随 step 增强的 User/Recommendation 权衡：相对 Parent，User 从 `+0.0034` 增至 `+0.0060`，Recommendation 从 `-0.0110` 扩大到 `-0.0212`。这更支持 User-only objective 的跨任务干扰，而不是“学习率低到没有学到”；`1e-6` 已经产生稳定方向性变化，提高学习率本身不能保证保留 Recommendation。完整原始 11 项、逐项 delta 和保留策略见 [MC_USER Hybrid 外部评测记录](./baselines/native_source_domain_r32_v3/grpo/user/docs/mc_user_hybrid_external_eval_v1.md)。

### 代码、Runner、文档和结果入口

| 路线 / 实验 | 正式入口 | 核心实现 | 说明与结果 |
| --- | --- | --- | --- |
| GR_REC 基础混合 | [`run_grpo_trl_train.py`](./baselines/native_source_domain_r32_v3/grpo/scripts/run_grpo_trl_train.py) | [`grpo_trl_trainer.py`](./baselines/native_source_domain_r32_v3/grpo/scripts/grpo_trl_trainer.py) | [`GR_REC_v1` 完整记录](./baselines/native_source_domain_r32_v3/docs/experiment_GR_REC_v1.md) |
| DSR 混合消融 | [`run_dsr_train.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/run_dsr_train.py) | [`dsr_trainer.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/dsr_trainer.py) | [实验文档](./baselines/native_source_domain_r32_v3/docs/experiment_GR_REC_DSR_Ablation_v1.md) / [目录说明](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/README.md) / [Pilot200 forensic](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/results/pilot200_forensic_20260818/GR_REC_DSR_PILOT200_FORENSIC_REPORT.md) |
| ExactClamp + Hier Bridge 混合 | [`run_think_exact_clamp_train.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_clamp_v1/run_think_exact_clamp_train.py) | [`think_exact_clamp_trainer.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_clamp_v1/think_exact_clamp_trainer.py) / [`nothink_hierarchical_credit.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_clamp_v1/nothink_hierarchical_credit.py) | [实验文档](./baselines/native_source_domain_r32_v3/grpo/docs/experiment_GR_REC_ThinkExactClamp_Ablation_v1.md) / [Formal1500 forensic](./baselines/native_source_domain_r32_v3/grpo/results/gr_rec_clamp_bridge_v1_formal1500_forensic_20260821.md) |
| NoThink-only Hier | [`run_nothink_only_hier_train.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_nothink_only_hier_v1/run_nothink_only_hier_train.py) | 复用 ExactClamp 目录中的 hierarchy/bridge | [实验文档](./baselines/native_source_domain_r32_v3/grpo/docs/experiment_GR_REC_NoThinkOnly_Hier_v1.md) / [目录说明](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_nothink_only_hier_v1/README.md) |
| NoThink-only Frontier | [`run_nothink_only_frontier_train.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_nothink_only_frontier_v1/run_nothink_only_frontier_train.py) | [`frontier_trainer.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_nothink_only_frontier_v1/frontier_trainer.py) / [`frontier_credit.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_nothink_only_frontier_v1/frontier_credit.py) | [实验文档](./baselines/native_source_domain_r32_v3/grpo/docs/experiment_GR_REC_NoThinkOnly_Frontier_v1.md) / [Formal forensic](./baselines/native_source_domain_r32_v3/grpo/results/gr_rec_nothink_frontier_v1_formal_e1_forensic_20260822.md) |
| Think-only Composite | [`run_gr_rec_think_composite_interest_v1.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_composite_interest_v1/run_gr_rec_think_composite_interest_v1.py) | [`composite_trainer.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_composite_interest_v1/composite_trainer.py) / [`interest_metric.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_composite_interest_v1/interest_metric.py) | [实验文档](./baselines/native_source_domain_r32_v3/grpo/docs/experiment_GR_REC_Think_CompositeInterest_v1.md) / [目录说明](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_composite_interest_v1/README.md) |
| **GRPO-TK / Sample8 FullSID** | [`run_sample8_fullsid_train.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/run_sample8_fullsid_train.py) | [`sample8_fullsid_trainer.py`](./baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/sample8_fullsid_trainer.py) | [实验记录](./docs/experiment_GRPO_TK.md) / [复现指南](./docs/reproduce_GRPO_TK.md) / [Preflight 证据](./baselines/native_source_domain_r32_v3/grpo/results/gr_rec_think_sample8_fullsid_v3_gpu_preflight_scan_20260828.json) |
| GR_USER_v1 | [`run_user_full_epoch.py`](./baselines/native_source_domain_r32_v3/grpo/user/scripts/run_user_full_epoch.py) | [`user_grpo_trainer.py`](./baselines/native_source_domain_r32_v3/grpo/user/scripts/user_grpo_trainer.py) | [项目入口](./baselines/native_source_domain_r32_v3/grpo/user/README.md) / [Reward 合同](./baselines/native_source_domain_r32_v3/grpo/user/docs/reward_contract_v1.md) / [Trainer 合同](./baselines/native_source_domain_r32_v3/grpo/user/docs/trainer_objective_contract_v1.md) / [Full epoch](./baselines/native_source_domain_r32_v3/grpo/user/docs/full_epoch_v1.md) |
| MC_USER Hybrid K4 | [`run_mc_user_formal_hybrid_k4_ddp_v1.py`](./baselines/native_source_domain_r32_v3/grpo/user/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py) | [`user_mc_hybrid_objective.py`](./baselines/native_source_domain_r32_v3/grpo/user/scripts/user_mc_hybrid_objective.py) | [外部评测与跨任务权衡](./baselines/native_source_domain_r32_v3/grpo/user/docs/mc_user_hybrid_external_eval_v1.md) |

总入口：[GRPO 工程 README](./baselines/native_source_domain_r32_v3/grpo/README.md) / [实验记录总索引](./实验记录/README.md)。

## Alpha 系列

Alpha 系列基于 BETA-baseline，重点研究数据提示、数据清洗、Action Select 约束、loss 目标与训练监控。除各实验记录中明确说明的变量外，模型（OneReason-8B + LoRA r32/a64）、优化器、8K neat packing 与外部评测方式保持一致。

### 实验脉络

Alpha 系列的第一条主线是建立**可复现的正式母版 `alpha_jiankong`**：从监控验证体系搭建开始，经历了 SID/domain 权重合同修复等调试阶段，最终收敛为统一正式实验。此前的「Alpha-监控优化」「Alpha-SID8 cache fix」并非独立实验，而是 `alpha_jiankong` 形成过程中的中间阶段（监控验证体系 + 权重合同 bug 修复），最终以 `alpha_jiankong` 为准。

| 实验 | 主线 | 主要改动 | 外部评测得分 |
| --- | --- | --- | --- |
| **Alpha-Jiankong** | **正式母版** | 98/2 leak-safe dev（train98/dev2/probe）+ 训练阶段验证 + monitor；修复静态 tokenized cache 的 SID/domain 权重合同（普通 token=1、canonical=4、非 canonical SID/domain=8） | **Epoch 1 总分 `1.2605`，Epoch 2 总分 `1.2992`** |
| Alpha-CoT | 权重侧 | Recommendation CoT 重复归一化，CoT body 使用 `0.5/N` | Epoch 1 总分 `1.2263` |
| **Alpha-Smooth** | **loss 目标侧** | **第二 epoch 起开启标签平滑 ε=0.05**（`alpha_smooth_start_epoch: 1.0`） | **Epoch 2 总分 `1.3185`，较 Alpha-Jiankong 有提升** |
| Alpha-V2 | 数据重建 | 推荐全量 CoT 重建 + NoCoT 恢复至 1/3 + 与 dev2 零交叉 | Epoch 1 `1.2584` / **Epoch 2 `1.2856`** |

> **Alpha-Smooth 的动机**：监控发现**懂推荐**在第二轮训练出现"训练集损失继续下降（尤其 NoCoT 样本）、验证集损失反而上升"的阶梯式过拟合——模型在第二轮主要是在记忆第一轮见过的训练结果。因此对推荐最终 A/B/C 位置施加标签平滑 ε=0.05 作为正则，并采用"先拟合、后平滑"的分阶段 schedule：第一轮原生 CE 充分拟合结构，第二轮起平滑目标抑制过度自信（`alpha_smooth_start_epoch: 1.0`）。Mini 系列的 Mini-Smooth / Mini-Whole-Smooth 是同一机制在小型数据版本上的复刻与 start-epoch 消融。

### Alpha-Jiankong 正式母版得分（外部评测器）

`alpha_jiankong` 是 Alpha 系列的对照基准（train98 / dev2 / probe + SID8 权重合同修复后）。分项顺序与 BETA-baseline 一致：懂物料 4 项 → 懂用户 2 项 → 懂推荐 4 项 → 懂世界 1 项（懂物料分项按历史约定不具备横向参考性，比较时以其余分项加和为准）。

| Epoch | 总分 | 分项 |
| --- | ---: | --- |
| Epoch 1 | `1.2605` | 物料 `0.0465, 0.0379, 0.0441, 0.0426`；用户 `0.1502, 0.0915`；推荐 `0.1204, 0.1394, 0.2016, 0.1521`；世界 `0.2342` |
| Epoch 2 | **`1.2992`** | 物料 `0.0490, 0.0369, 0.0516, 0.0420`；用户 `0.1556, 0.0955`；推荐 `0.1241, 0.1394, 0.2002, 0.1683`；世界 `0.2364` |

Epoch 1 → Epoch 2 提升 `+0.0387`，主要来自用户与推荐分项改善；世界分项保持稳定。

### Alpha 系列得分一览（外部评测器，总分）

| 实验 | Epoch 1 | Epoch 2 |
| --- | ---: | ---: |
| BETA-baseline（统一参考线） | `1.3090` | `1.3246` |
| Alpha-Jiankong（正式母版） | `1.2605` | **`1.2992`** |
| Alpha-CoT | `1.2263` | — |
| **Alpha-Smooth** | — | **`1.3185`** |
| **Alpha-V2** | `1.2584` | **`1.2856`** |

Alpha 记录入口：

- [Alpha-Jiankong 正式母版（含监控优化与 SID8 权重合同修复历程）](./baselines/native_source_domain_r32_v3/docs/experiment_alpha_monitor.md)
- [Alpha 监控结果分析](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_监控优化_结果分析.md)
- [Alpha SID8 cache 权重合同修复记录](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_SID8_cache_fix.md)
- [Alpha-CoT 重复归一化](./baselines/native_source_domain_r32_v3/docs/experiment_alpha_cot_repeat05n.md)
- [Alpha-Smooth 标签平滑](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHASMOOTH.md)
- [Alpha-V2 推荐重建](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_V2.md)
- [Alpha Mini R32 两 epoch](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_Mini_R32_2E.md)

Alpha-CoT 与 Alpha-Smooth 的正式运行配置和代码位于服务器的 Native baseline 工作目录。核心对比必须使用 raw CoT body CE、CoT Gold SID CE、No-think Gold SID CE、验证集指标以及外部评测总分与分项，不能直接比较改变权重或损失目标后的 recommendation task loss。

## Mini 系列

Mini 系列是 Alpha 正式实验的轻量复现、排查和数据消融版本：不启动完整全量训练，用缩小的任务池验证数据、loss route、梯度和监控，同时每个 Mini 版本本身也会跑完 2 epoch 并给出外部评测分数，用于回答"数据怎么改会带来什么影响"。

Mini 系列以 **`alpha_mini`（49,490 行组合任务池）为大基线**，后续 Mini 版本以数据构造改动为主（用户/推荐/物料比例与形态），Mini-Smooth / Mini-Whole-Smooth 则在同一数据版本上验证 loss 目标（标签平滑）改动；模型、LoRA、优化器、packing 与评测方式保持不变。

### 实验脉络

| 实验 | 数据改动 | 外部评测得分（2 epoch） |
| --- | --- | --- |
| **Alpha-Mini（大基线）** | 懂用户 2,000 + 懂推荐 11,192（多正 group，CoT/NoCoT 混合）+ 懂物料 36,298 = **49,490** | **总分 `1.2861`** |
| **Mini-V2** | 懂用户扩到 3,000；懂推荐 NoCoT 比例恢复至约 1/3（其余恢复为 CoT）；懂物料不变 = **50,490** | **总分 `1.2671`** |
| **Mini-V3** | 懂推荐全部恢复为 CoT（NoCoT=0）；懂用户 3,000、懂物料 36,298 不变 = **50,490** | **总分 `1.2178`** |
| Mini-CoT | ~~Alpha-mini 基础上对 recommendation CoT think span 加 `0.5/N` 重复归一化权重~~（**已放弃该 trick**） | CPU 验收通过（未启动正式训练） |
| **Mini-Smooth** | 在 Mini 基线上开启推荐最终 A/B/C 标签平滑 ε=0.05，**第二 epoch 起生效**（`alpha_smooth_start_epoch: 1.0`），缓解第二轮过拟合 | **已完成，总分 `1.2681`** |
| **Mini-Whole-Smooth** | 与 Mini-Smooth 同源，但平滑**全程生效**（`alpha_smooth_start_epoch: 0.0`），两个 epoch 都平滑，回答"平滑应该从哪个 epoch 开始" | **已完成，总分 `1.2859`（全程优于第二 epoch 开启，但仍略低于无平滑基线）** |
| **Mini-Fix** | alpha_mini 基础上：推荐短 think 行从 NoCoT **归还 cot**（恢复短思考教学）+ NoCoT 纯度提升至 100%（占比 44.3% 不变） | **总分 `1.3093`，推荐分项 `0.6644`（历史最高）** |
| **Mini-Short-CoT** | 以 Mini-Fix 为父实验；6,235 条 Recommendation-CoT 的 think span 只保留 `【兴趣归纳】`，最终答案、多正 metadata、NoThink、懂用户和懂物料不变 | **Epoch 2 总分 `1.2989`，推荐分项合计 `0.6615`** |
| **Mini-Fix-U3K** | 保留 Mini-Fix 推荐样本与训练预算；懂用户扩至 3,000 条；移除 1,000 条物料反向补充冗余行以守恒总行数 | **已完成，总分 `1.2846`（负结果：用户扩展未带来提升）** |
| **Mini-Fix-E3** | Mini-Fix 数据与训练合同完全不变，仅从 2 epoch 延长至 3 epoch；保留每轮 checkpoint | **已完成，总分 `1.2495`（负结果：第三轮过拟合恶化）** |
| **Mini-Fix-U4K** | Mini-Fix 推荐契约不变；懂用户扩至 4,000 条（Mini-V2 3,000 + chian 补充 1,000）；**物料完整保留**（51,490 行） | **已完成，总分 `1.3002`（用户分 +0.0044 但推荐 -0.0147，净降）** |
| **Mini-Fix-Whole-Smooth** | Mini-Fix 数据（nocot 纯度 100%）+ 全程标签平滑 ε0.05（start_epoch 0.0），排除性验证 | **已完成，总分 `1.2884`（负收益确认：数据越干净 smooth 副作用越大）** |

### Mini 系列得分一览（外部评测器，总分与分项）

分项顺序与 BETA-baseline 一致：懂物料 4 项 → 懂用户 2 项 → 懂推荐 4 项 → 懂世界 1 项（懂物料分项按历史约定不具备横向参考性）。

| 实验 | 总分 | 懂物料 | 懂用户 | 懂推荐 | 懂世界 |
| --- | ---: | --- | --- | --- | ---: |
| **Alpha-Mini（大基线）** | **`1.2861`** | `0.0621, 0.0357, 0.0516, 0.0426` | `0.1395, 0.0708` | `0.1335, 0.1496, 0.1960, 0.1746` | `0.2301` |
| **Mini-V2** | **`1.2671`** | `0.0607, 0.0353, 0.0507, 0.0431` | `0.1333, 0.0809` | `0.1185, 0.1496, 0.1988, 0.1656` | `0.2305` |
| **Mini-V3** | **`1.2178`** | `0.0619, 0.0345, 0.0512, 0.0432` | `0.1513, 0.0772` | `0.0980, 0.1292, 0.1722, 0.1611` | `0.2379` |
| **Mini-Smooth** | **`1.2681`** | `0.0624, 0.0378, 0.0516, 0.0420` | `0.1359, 0.0717` | `0.1157, 0.1598, 0.1834, 0.1764` | `0.2312` |
| **Mini-Whole-Smooth** | **`1.2859`** | `0.0627, 0.0371, 0.0519, 0.0429` | `0.1388, 0.0723` | `0.1251, 0.1530, 0.1960, 0.1737` | `0.2323` |
| **Mini-Fix** | **`1.3093`** | `0.0631, 0.0363, 0.0517, 0.0426` | `0.1479, 0.0720` | `0.1297, 0.1598, 0.2030, 0.1719` | `0.2312` |
| **Mini-Short-CoT** | **`1.2989`** | `0.0613, 0.0369, 0.0514, 0.0424` | `0.1451, 0.0723` | `0.1260, 0.1598, 0.2002, 0.1755` | `0.2279` |
| **Mini-Fix-U3K** | **`1.2846`** | `0.0620, 0.0359, 0.0529, 0.0418` | `0.1410, 0.0731` | `0.1400, 0.1564, 0.1918, 0.1674` | `0.2223` |
| **Mini-Fix-E3** | **`1.2495`** | `0.0617, 0.0367, 0.0520, 0.0428` | `0.1315, 0.0808` | `0.1120, 0.1224, 0.2030, 0.1746` | `0.2320` |
| **Mini-Fix-U4K** | **`1.3002`** | `0.0625, 0.0358, 0.0522, 0.0426` | `0.1463, 0.0780` | `0.1279, 0.1564, 0.2016, 0.1638` | `0.2331` |
| **Mini-Fix-Whole-Smooth** | **`1.2884`** | `0.0615, 0.0357, 0.0529, 0.0430` | `0.1468, 0.0700` | `0.1213, 0.1564, 0.1988, 0.1719` | `0.2301` |

观察：NoCoT 比例从基线（约 42%）降到 1/3（Mini-V2）再降到 0（Mini-V3），推荐分项逐步下降（0.6537 → 0.6325 → 0.5605），世界分项小幅上升（0.2301 → 0.2305 → 0.2379）——说明全 CoT 化对推荐任务本身未必有利，推荐分项随 NoCoT 减少而单调下降。Mini-Smooth（Mini 基线 + 第二 epoch 推荐 A/B/C 标签平滑）总分 1.2681、推荐分项 0.6353，介于 Mini-V2 与基线之间——平滑正则对 Mini 基线有一定正则代价，其与 Mini-Whole-Smooth 的 start-epoch 消融结论以外部评分为准。**Mini-Fix（NoCoT 高纯度 100% + cot 恢复短思考教学）总分 1.3093、推荐分项 0.6644，为全系列最高（超过 BETA 0.6594）**。Mini-Short-CoT 将 6,235 条 CoT 压缩到只保留兴趣归纳后，推荐合计为 `0.6615`，与 Mini-Fix 接近但总分下降 `0.0104`；该结果没有证明进一步删去行为模式和预测总结能带来净收益。Mini-Fix 总分仍低于 BETA（1.3313），主要差异来自用户分项（mini 用户占比仅 4% vs BETA 14.8%）。

Mini 版本不作为最终排行榜结果，不覆盖 Alpha 正式 output，也不改变正式训练的 scheduler horizon。当前 mini 复现包不包含模型权重，原始数据仍需根据 manifest 从服务器或本地数据源恢复。

Mini 记录：

- [Alpha-Mini 大基线（R32 两 epoch）](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_Mini_R32_2E.md)
- [Mini-V2（NoCoT 1/3 + 用户 3,000）](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_Mini_V2.md)
- [Mini-V3（推荐全 CoT）](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_Mini_V3.md)
- [Mini-CoT（think span 0.5/N，**已放弃**）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_COT_ALPHA_Mini_R32_2E.md)
- [Mini-Smooth（第二 epoch 标签平滑，总分 `1.2681`）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_SMOOTH.md)
- [Mini-Whole-Smooth（全程标签平滑，总分 `1.2859`）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_WHOLE_SMOOTH.md)
- [Mini-Fix（NoCoT 高纯度 + cot 短思考教学，总分 `1.3093` / 推荐 `0.6644`）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX.md)
- [Mini-Short-CoT（CoT 只保留兴趣归纳，Epoch 2 总分 `1.2989`）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_SHORT_COT.md)
- [Mini-Fix-U3K（用户扩到 3,000，总分 `1.2846`，负结果）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX_U3K.md)
- [Mini-Fix-E3（3 epoch，总分 `1.2495`，第三轮过拟合）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX_E3.md)
- [Mini-Fix-U4K（用户扩到 4,000，物料完整，总分 `1.3002`）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX_U4K.md)
- [Mini-Fix-Whole-Smooth（全程平滑排除性验证，总分 `1.2884`）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX_WHOLE_SMOOTH.md)
- [Mini-Fix-U3K（保留推荐优势、懂用户 3,000 条，已配置待训练）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX_U3K.md)
- [Mini-Fix-E3（Mini-Fix 仅延长至 3 epoch，U3K 成功后自动启动）](./baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX_E3.md)

## 数据版本

GitHub 只保存数据版本注册信息、manifest、数量和 SHA-256 摘要；实际 JSONL 和 tokenized cache 保存在服务器。

| 版本 | 作用 |
| --- | --- |
| `raw_all` | 八个原始全量子数据集 |
| `v1_thought_prompt_all` | 为样本补充 `think` / `no_think` 标志 |
| `v2_recommendation_cot_complete_all` | 过滤懂推荐不完整 CoT |
| `v3_material_clean` | 只替换懂物料，其他子集从父版本继承 |
| `alpha-jiankong` | Alpha 监控的 train98、dev2 和固定 probe |
| `bata_baseline_v1` | BETA-baseline 纯净版全量训练数据 |

相关文件：

- [数据版本说明](./ONEREASON_DATASET_VERSIONS.md)
- [数据版本清单](./data/onereason_dataset_versions.json)
- [数据集注册表](./data/dataset_info.json)

服务器常用目录：

```text
/data/lf_data_versions/alltrain
/data/lf_data_versions/task_pools
/data/lf_data_versions/task_pools_abnormal
```

数据版本采用父版本继承，清洗单个任务时只替换对应子数据集，不复制或覆盖其他任务。启动训练前必须检查 manifest、字段、数量守恒、哈希和实际 loss route。

## 指标说明

### 最终评测

首先看总分是否接近 BETA-baseline 约 1.33 的水平；GRPO 系列统一使用较高复测值 `1.3313` 作保守基线。单次差异在 `±0.01` 内默认按正常评测波动处理，再看懂物料、懂用户和懂推荐的分项变化。不同版本若更改数据、SID 权重或输出格式，必须结合实验记录解释，不能只比较一个 checkpoint 的总分。

### 训练过程

- `total_loss`：当前训练实际反向的总损失；改变 token 权重或 CoT 权重后，绝对值会改变。
- `task_loss_material`、`task_loss_recommendation`、`task_loss_user_action`、`task_loss_user_chain`：各任务 loss，用于观察同一实验中的趋势。
- `grad_norm`：更新前梯度范数；长期高于 clipping 阈值时，实际更新会受到裁剪。
- `learning_rate`：当前 scheduler 学习率；cosine 后期下降不代表模型已经完全收敛。

### 推荐和 Action

- `CoT body CE`、`CoT Gold SID CE`、`No-think Gold SID CE`：Alpha-CoT 的核心 raw CE，未乘 CoT 重复降权，用于跨版本比较。
- `candidate_hit`、`coverage`、`chain`：teacher-forcing 下的候选命中、集合覆盖和链路完整性。
- `gold_top5_rate`、`top1_illegal_hit_rate`、`topk_illegal_rate`：Action Select 合法性和 gold 命中情况。
- `cot_body_numerator_share`、`cot_final_answer_numerator_share`、`nocot_numerator_share`：加权训练 numerator 的组成，只解释 loss 尺度，不等价于生成质量。

### 验证集

Alpha 监控使用 leak-safe dev 和固定 probe。验证指标用于判断训练损失是否真正转化为泛化能力，不参与训练 backward；若验证集指标反弹而训练 loss 继续下降，优先考虑过拟合、数据路线差异或评测格式问题。

## 常用入口

- Native baseline 代码和配置：[baselines/native_source_domain_r32_v3](./baselines/native_source_domain_r32_v3)
- 多任务设计：[ONEREASON_MULTITASK.md](./ONEREASON_MULTITASK.md)
- 新版实验记录：[Native baseline docs](./baselines/native_source_domain_r32_v3/docs)

启动正式训练前，应确认 GPU、数据 manifest、实际 loss route、输出目录和实验记录一致。Mini smoke 通过后再启动完整 epoch。

## 工程边界

- 不上传原始数据、模型权重、checkpoint、日志和密钥。
- 新 loss 必须有独立开关、数学回归和最小 smoke；关闭开关时恢复 baseline 行为。
- 数据清洗、数据版本、训练配置和实验记录分离管理，保证单个子任务可以独立替换和回退。

代码沿用 LLaMA-Factory 的 Apache-2.0 许可证；模型和竞赛数据须遵守各自许可证及赛事规则。
