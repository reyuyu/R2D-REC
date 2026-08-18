# 实验 GR_REC_v1：Recommendation Multi-Positive GRPO

状态：**正式训练中**。本文保留 Step 1680 的运行快照，并追加截至 Step 2280 的问题侧分析；Probe 趋势和结论均为中期结果，不代表最终评测。

## 实验标识

- 实验名：`GR_REC_v1`
- 正式 Run：`REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818`
- 启动时间：`2026-08-18 02:49:22`（Asia/Shanghai）
- 代码版本：`a044c9d386173976c5112b2b7a4576e7515a14ae`
- 基座：`/data/models/onereason-8b-pretrain-competition`
- 初始 Adapter：`BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333`
- 数据：`rec_mp_grpo_v2`，原始 1,549 个 recommendation group
- 输出：`/data/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/`
- 监控：`/data/GRPO/runs/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/`

## 目标与假说

本实验在 BATA SFT Adapter 上进行 recommendation-only GRPO，保留 Think 的物料检索与 Beam32 优势，同时用 group-relative outcome reward 强化用户兴趣推断和最终 SID 命中。Think 与 NoThink 使用各自的生产采样形状，但通过 route loss multiplier 保持单 group 总权重一致。

实验不改动 reward 定义、数据顺序、动态 G、Beam32 语义、采样温度、GRPO/PPO 数学或 old policy log-prob。正式训练前的性能改动均要求 token/reward/loss/gradient parity。

## 冻结训练合同

| 项目 | 配置 |
| --- | --- |
| GPU | 4 x NVIDIA A800-SXM4-80GB |
| LoRA | r32 / alpha64 / dropout 0.05，base 冻结 |
| Optimizer steps | 2,316 |
| Learning rate | `1e-6`，constant |
| Seed | `20260816` |
| Think | G=4，temperature=0.9，top_p=0.95，route weight=1.0 |
| NoThink | G=8，temperature=1.0，top_p=1.0，route weight=0.5 |
| GRPO | beta=0，epsilon=0.2，loss_type=grpo，group scaling |
| Iterations | 每个 rollout 复用 2 个 optimizer steps |
| Length | max prompt 8192，max completion 2048 |
| Think stopping | `</think>` 单 token、per-sample StoppingCriteria、生成后防御性截断 |
| Beam32 | context batch=1，num_beams=32，returns=32，max_new_tokens=128 |
| Checkpoint | 每 500 step；最多保留 4 个 |

Sampler 审计结果：四个 Probe group 排除后剩 1,545 个训练候选 group；为满足完整 batch，固定丢弃 `c399e01d...`，实际 Think/NoThink 均覆盖 1,544 个 group。预计 Think rollout 386 次、NoThink rollout 772 次，路线循环为 `T,T,N,N,N,N`，共 2,316 optimizer steps。

## Reward 合同

- NoThink：SID 六档互斥 reward `-1 / -0.25 / 0 / 0.5 / 2 / 8`。
- Think：每条 CoT 进入 Beam32；Exact=8、AB=2、A=0.5，按命中等级与几何衰减聚合。
- Think 只计算 Think reward；NoThink 只计算 NoThink reward。
- reward、advantage、old_per_token_logps 和 GRPO loss 数学保持冻结。

## 四域固定 Probe

固定 Probe 使用 seed `20260818`，在 Step 0、每 200 step 和最终 step 执行。四个 group 永久从训练集排除，评估前后恢复 Python、CPU 和 CUDA RNG。Think 保持 4 groups x G=4；NoThink 保持两批 2 groups x G=8。

| 顺序 | 域 | Group ID |
| ---: | --- | --- |
| 1 | video | `fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e` |
| 2 | living | `6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f` |
| 3 | prod | `281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700` |
| 4 | ad | `2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8` |

早期错误 Run `REC-MP-GRPO-FULL-E1-PROBE4-20260818` 的 Probe 为 `prod/living/living/ad`，缺少 video，已在 Step 36 停止并保留审计记录；不得与本实验曲线混合。

## 中期运行快照

截至 2026-08-18 本次记录时：

- 训练进程正常，最新 Step `1680 / 2316`。
- 已保存 `checkpoint-500`、`checkpoint-1000`、`checkpoint-1500`。
- 已产生 Step 0 至 Step 1600 的 9 轮 Probe，共 36 行。
- 未观察到训练进程退出、OOM、NaN 或 Inf。

四域平均 Probe：

| Step | Think reward | NoThink reward | Think closure | Beam invalid |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3.7305 | 0.0000 | 100% | 0 |
| 200 | 2.5742 | 0.0234 | 100% | 0 |
| 400 | 4.0879 | 0.2266 | 100% | 25 |
| 600 | 3.7617 | 0.2031 | 100% | 1 |
| 800 | 3.6719 | 0.4219 | 100% | 29 |
| 1000 | 4.2012 | 0.1094 | 100% | 0 |
| 1200 | 4.1992 | 0.0625 | 100% | 0 |
| 1400 | 3.7754 | 0.2188 | 100% | 0 |
| 1600 | 2.9492 | 0.4688 | 100% | 0 |

阶段性观察：NoThink 相对 Step 0 整体抬升，Step 1600 为当前最高均值；Think 波动较大，Step 1000/1200 高于基线，但 Step 1600 回落，尚不能判定稳定改善。closure 始终为 100%。Step 400 的 living 和 ad、Step 800 的 ad 出现 Beam invalid，之后 Step 1000-1600 恢复为 0；该异常必须在最终报告中结合原始 Beam SID 复核。

## 问题侧中期分析（截至 Step 2280）

本节只记录已观察到的风险和支持证据，不作为最终实验判定。分析时严格区分三类问题：CoT 多样性坍缩、组内 reward 零方差，以及 Beam SID 解析无效。后两者不能混称为同一种“无效样本”。

### 1. CoT 出现奖励兼容型模式坍缩

固定 video Probe 使用相同 prompt 和 seed，只有模型 checkpoint 随 step 变化。早期 4 条 CoT 通常包含 3--4 个兴趣类目、行为模式和预测总结；Step 1800/2200 的 4 条 CoT 基本都只保留第一个“服饰穿搭”兴趣，随后立即输出 `</think>`。

| Step | CoT 平均长度 | 最短/最长 | 组内字符 4-gram Jaccard | Think 最高 reward |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 714.8 | 659 / 793 | 0.162 | 9.19 |
| 200 | 718.2 | 669 / 794 | 0.134 | 12.19 |
| 400 | 770.0 | 680 / 859 | 0.121 | 14.09 |
| 1400 | 539.8 | 123 / 715 | 0.119 | 14.09 |
| 1800 | 90.5 | 71 / 110 | 0.473 | 9.19 |
| 2000 | 132.2 | 100 / 202 | 0.374 | 9.19 |
| 2200 | 90.5 | 77 / 106 | 0.511 | 8.44 |

该现象不只是单个 Probe 的长度波动。全训练 Think rollout 的平均 completion 长度从 Step 0--399 的 `703.6` 降至 Step 1600--1999 的 `267.0`，Step 2000+ 进一步降至 `195.9`；同期 zero-std group 比例由 `40.8%` 升至 `60.2%` 和 `63.0%`。

短 CoT 仍可获得较高 reward，说明模型并非单纯丧失 SID 命中能力，而是在学习一种更短、更相似、集中于单一主兴趣的高收益模式。当前 Think reward 只评价后续 Beam32 的 SID 命中，不直接约束兴趣覆盖、论据数量、CoT 长度或组内语义多样性，因此该退化属于 **reward-compatible CoT mode collapse**。这一结论对 video 固定 Probe 有直接纵向证据；是否在所有 group 上都表现为“单一兴趣”，仍需更多文本级 trace 统计。

### 2. NoThink reward 稀疏且后期区分性下降

截至 Step 2280，共记录 764 个 NoThink rollout、1,528 个 group 和 12,224 个候选：

- 446/1,528 个 group 为组内 reward 零方差，占 `29.19%`。
- `8,573/12,224 = 70.13%` 的候选得分恰好为 0。
- `73.12%` 的候选得分小于等于 0；正 reward 候选仅 `26.88%`。
- Exact=8 只有 `276/12,224 = 2.258%`。
- NoThink 平均 reward 从 Step 0--399 的 `0.224` 升至 Step 2000+ 的 `0.498`，但 zero-std group 同时从 `22.0%` 升至 `40.8%`。

因此，平均 reward 的提高并不等于信号覆盖面改善。当前更符合“少数容易样本或高分候选推高均值，但越来越多 group 的 8 个候选仍完全同分”的情形；这些 group 产生零 advantage，消耗 rollout 计算但不贡献组内相对学习信号。

数据集 gold 数量显示该问题具有结构性域差异：

| 目标域 | Group 数 | Gold 中位数 | Gold 均值 | Gold 数不超过 2 的比例 |
| --- | ---: | ---: | ---: | ---: |
| video | 550 | 15 | 15.54 | 0% |
| living | 190 | 2 | 2.48 | 77.37% |
| prod | 382 | 2 | 2.43 | 74.08% |
| ad | 427 | 2 | 2.51 | 73.54% |

living/prod/ad 约四分之三的 group 只有两个 gold。在巨大 SID 空间中使用 G=8 采样时，候选很容易集中于 0 或相邻低分档位。固定 video Probe 有 20 个 gold，而其余三个 Probe 均只有 2 个，因此四域 Probe 平均 reward 也不能被视为等难度域平均。

### 3. Think 同样存在难样本和无区分样本

全训练 382 个 Think rollout、1,528 个 group 中，714 个 group 为 zero-std，占 `46.73%`，高于 NoThink 的 `29.19%`。Step 2000+ Think zero-std 达到 `63.0%`。问题不能只归因于 NoThink。

固定 Probe 的 12 轮结果进一步显示样本难度差异：

| 域 | Think 全零轮次 | NoThink 全零轮次 | Probe gold 数 |
| --- | ---: | ---: | ---: |
| video | 0 / 12 | 0 / 12 | 20 |
| living | 9 / 12 | 5 / 12 | 2 |
| prod | 0 / 12（另有 1 次同分） | 7 / 12 | 2 |
| ad | 8 / 12 | 8 / 12 | 2 |

这里的“无区分样本”定义为组内 reward 标准差为 0。它与 Beam invalid 是不同问题：Beam invalid 曾在 Step 400 出现 living=19、ad=6，Step 600 出现 living=1，Step 800 出现 ad=29；Step 1000--2200 又恢复为 0。前者是奖励稀疏或候选同质导致的训练信号缺失，后者是 Beam 输出 SID 无法解析的生成/解析异常。

### 问题侧结论

GR_REC_v1 并非完全没有学习：NoThink 平均 reward 上升，部分 Think checkpoint 仍能取得较高 Beam reward。但现有证据同时表明，训练后期出现了两种不利趋势：一是模型找到更短、更单一但仍可获得高 Think reward 的捷径；二是 reward 改善集中于部分容易样本，大量低-gold难样本仍缺少组内区分信号。

因此，最终 checkpoint 不应仅按训练平均 reward 选择。最终验收至少应横向比较 `checkpoint-1000`、`checkpoint-1500`、`checkpoint-2000` 和 final 的外部 11 项得分，并同步报告 CoT 长度、组内文本多样性、Think/NoThink zero-std、四域分项和 Beam invalid。当前问题分析只支持诊断，不预注册任何 reward、G、采样或数据修改。

## Monitoring Phase 1

正式运行启用 `GRPO_MONITOR=1`、`DETAILED=0`、`GENERATION_PROFILE=0`、`BEAM_RANK_BALANCE=1`、`TRACE_EVERY=20`。前端每 3 秒拉取 metrics、rollouts、rank、trace、Probe 和 checkpoint API；监控只读取已有训练结果，不进入 loss 或 optimizer。

监控入口：

```text
http://127.0.0.1:8877/?run=REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818
```

下载 Adapter 时必须保持标准 PEFT 文件名并放在同一目录：

```text
adapter_config.json
adapter_model.safetensors
```

当前 checkpoint 权重已核验为真正 LoRA：配置 `peft_type=LORA`、r=32；权重 504 个 tensor 均为 LoRA A/B，非 LoRA tensor 为 0。带 Run 前缀的浏览器下载文件名会使部分平台误判为 full，上传前必须恢复上述标准名称。

## 验证记录

- Monitor、fixed Probe、formal runner、NoThink trace、prompt cache 与 correctness v2 CPU 套件全部通过。
- 固定 Probe 域覆盖测试强制顺序为 `video/living/prod/ad`；错误数量或重复域会 fail-fast。
- 4GPU 真实 smoke 验证 reward/loss/ratio/clip/KL 有限、LoRA 更新且 base 保持冻结。
- 相关提交：`cc3da7d`（固定 Probe）、`a044c9d`（四域分层 Probe）。

## 最终验收

训练完成后必须补充：最终 Step、四个 checkpoint 与最终 Adapter 状态、四域 Step 0/最终 Probe 对照、外部 11 项评测、Think/NoThink reward 与有效信号密度、KL/clip/ratio、Beam invalid 复核，以及 LoRA/base 参数变化。只有固定 Probe 与外部评测共同支持提升，才能将 `GR_REC_v1` 判定为正向实验。
