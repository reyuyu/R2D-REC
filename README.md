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

当前实验链路如下：

1. **BETA-baseline（GRPO 基线）**：不是新的训练实验，而是 GRPO policy 的初始 Adapter。使用多次评测中较高的 `1.3313` 作保守对照，动机是避免以偏低测次夸大 GRPO 收益。
2. **GR_REC_v1（首个正式 GRPO 实验）**：在冻结 BETA SFT Adapter 合同的基础上，只训练 recommendation group。Think 路由用 G=4 与 Beam32 outcome reward，NoThink 路由用 G=8 SID 分档 reward；通过 route weight 保持两条路由每 group 权重可比。它要回答的核心问题是：在不修改 reward、采样、Beam 和 PPO/GRPO 数学的情况下，生成侧 outcome 信号能否提高懂推荐外部得分。
3. **Checkpoint 选择**：不按“越晚越好”选择，也不只看训练 reward。先比较 Step 1000/1500/2000/final 的外部 11 项，再结合 CoT 长度、多样性、zero-std、Beam invalid 和固定四域 Probe。当前 Step 1500 是优先复测候选。
4. **后续实验隔离原则**：CoT 后期单一兴趣收缩与 NoThink/Think 零方差是两类不同问题。后续若分别测试多样性约束或稀疏 reward 改进，必须单变量立项，不在同一实验中同时改 G、reward、sampler 或数据顺序。

### GR_REC_v1 外部评测

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

GRPO 记录入口：

- [GR_REC_v1：动机、冻结合同、训练完成记录、问题侧证据与完整分项](./baselines/native_source_domain_r32_v3/docs/experiment_GR_REC_v1.md)
- [GRPO 代码、监控与测试入口](./baselines/native_source_domain_r32_v3/grpo/README.md)
- [实验记录总索引](./实验记录/README.md)

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
