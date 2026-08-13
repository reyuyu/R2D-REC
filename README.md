# OneReason 多任务 SFT

本仓库用于 OneReason-8B 竞赛模型的多任务监督微调、数据版本管理和实验对比。工程基于 LLaMA-Factory，覆盖懂物料、懂用户、懂推荐等任务，并提供原生 SFT、GradNorm、SID 加权、推荐多正例以及训练监控等可回退实现。

仓库只保存代码、配置、测试、实验记录和数据版本元信息，不保存模型权重、原始 JSONL、日志、checkpoint 或凭据。

## 项目背景

训练目标不是只降低一个总 loss，而是在固定数据合同和训练配方下，同时提升四类能力：

- 懂物料：从 SID 与文本描述中学习商品、视频、直播和广告域的语义映射；
- 懂用户：学习 Action Select 以及用户行为链；
- 懂推荐：根据用户历史和画像完成多域推荐；
- 训练可解释性：记录任务损失、候选质量、梯度范数和验证集变化，区分真实效果与训练尺度变化。

当前所有消融实验都以 **BATA-baseline 纯净版** 为主要参考基线。基线的最新完整结果为：

| 基线 | 训练进度 | 总分 |
| --- | ---: | ---: |
| BATA-baseline 纯净版 | Epoch 1 | 1.3090 |
| BATA-baseline 纯净版 | Epoch 2 | **1.3246，约 1.33 分参考线** |

1.33 不是新的硬性验收阈值，而是当前数据、模型和评测器组合下的横向比较基准。任何新实验都应同时报告总分和分项分数，不能只看单一指标或 raw task loss。

详细基线合同和结果见 [BATA-baseline 纯净版](./baselines/native_source_domain_r32_v3/docs/BATA_BASELINE.md) 与 [实验记录](./实验记录/实验BATA-baseline纯净版.md)。

## 实验系列

所有系列均从 BATA-baseline 的模型、数据合同或训练入口出发，每次只改变明确记录的消融变量。Alpha 和 Beta 是当前两大实验板块。

### Alpha 系列：数据、提示和训练监控

Alpha 系列关注数据清洗、think/no-think 提示、Action Select 约束以及训练过程监控。它们沿用 baseline 的主体训练配方，变化集中在数据版本或可关闭的辅助目标。

| 实验 | 核心变量 | 已记录结果或状态 |
| --- | --- | --- |
| Alpha / 实验 A | GradNorm-lite 与梯度监控 | 历史多任务 GradNorm 对照 |
| Alpha-A0 | 学习率、dropout 和数据版本消融 | `lr=1e-4`、`dropout=0.01` 对照 |
| Alpha-A1 | think/no-think 提示补充 | 使用 V1/V2 提示版本对照 |
| Alpha-监控优化 | 98/2 leak-safe dev、固定 probe、训练中验证指标 | 重点观察四任务 loss 与 dev/probe 指标 |
| Alpha-C | Action Select 历史 Trie、完整 SID 去重、Continue/Stop | 辅助损失版本，已发现部分消融存在负优化风险 |
| Alpha-CoT | Recommendation CoT 重复归一化，CoT body 使用 `0.5/N` 权重 | 正式 2 epoch 实验，raw CE 与加权 task loss 分开解读 |

相关记录：

- [实验 A：GradNorm-lite](./实验记录/实验A_GradNorm-lite.md)
- [实验 A0](./实验记录/实验A0_GradNorm低学习率低Dropout.md)
- [实验 A1](./实验记录/实验A1_think_prompt-GradNorm.md)
- Alpha 监控优化：98/2 leak-safe dev、固定 probe 和训练中验证指标；具体运行文档以服务器当前版本为准。
- [实验 C：Action Select 约束](./实验记录/实验C_Action-Select历史约束.md)
- Alpha-CoT：Recommendation CoT 重复归一化，CoT body 使用 `0.5/N` 权重；代码和运行记录随 Alpha-CoT 分支维护。

### Beta 系列：稳定基线与推荐目标消融

Beta 系列以 BATA-baseline 的物料对齐合同为锚点，逐步验证 SID8、推荐 Set-PU、PackRatio 和候选质量监控。Beta 实验之间必须明确区分数据版本、loss route 和 packing 调度，不能直接用 task loss 的绝对值横比。

| 实验 | 核心变量 | 结果或用途 |
| --- | --- | --- |
| BATA-baseline 纯净版 | 原生 source/domain SFT；SID/domain=8；canonical=4；不启用 REC-PU | Epoch 2 总分 **1.3246，约 1.33** |
| BETA-MATERIAL-ALIGNED-SID8 | 锁定 `BETA_material_aligned_v1` 的三路物料合同 | 为后续 Beta 消融提供统一母版 |
| BETA-SETloss | Recommendation final SID 使用 Set-PU scalar objective，`alpha=0.05` | 真实 autograd；替代旧 surrogate backward |
| BETA-fenpei | Set-PU + `20/45/20/15` PackRatio + 候选质量指标 | 独立 4 GPU、2 epoch 训练路线 |
| REC-PU | Recommendation positive-unlabeled 研究阶段 | Phase 1-4 数学和 smoke 验证记录 |

相关记录：

- [BATA-baseline 纯净版](./baselines/native_source_domain_r32_v3/docs/BATA_BASELINE.md)
- [BETA-fenpei](./baselines/native_source_domain_r32_v3/docs/experiment_BETA-fenpei.md)
- [BETA-SETloss](./baselines/native_source_domain_r32_v3/docs/实验BETA-SETloss.md)
- [REC-PU](./baselines/native_source_domain_r32_v3/docs/实验REC-PU.md)
- [Native Source-Domain R32 V3 baseline 说明](./baselines/native_source_domain_r32_v3/README.md)

旧的 REC-C/REC-F、GradNorm-Ortho 和 SID-weight8 记录仍保留，作为历史消融索引，不作为当前 1.33 baseline 的直接替代品：[实验记录目录](./实验记录/README.md)。

## 数据版本

数据文件只保存在服务器，GitHub 保存注册信息、manifest、样本数和 SHA-256 摘要。版本采用父版本继承，因此可以只替换一个子任务，而不复制其他任务数据。

当前主要版本：

| 版本 | 用途 | 内容 |
| --- | --- | --- |
| `raw_all` | 原始母版 | 服务器上的全量原始子数据集 |
| `v1_thought_prompt_all` | 提示补充 | 为样本补充 `think` / `no_think` 标记 |
| `v2_recommendation_cot_complete_all` | 推荐清洗 | 保留完整的兴趣归纳、行为模式、预测总结样本 |
| `v3_material_clean` | 物料消融示例 | 只替换懂物料，其余任务从父版本继承 |
| `BETA_material_aligned_v1` | Beta 主线 | 三路物料合同：`material_sample`、`canonical`、`reverse` |
| `alpha-jiankong` | Alpha 监控 | 训练 98% + leak-safe dev/probe |
| `BETA_material_aligned_v1` / `bata_baseline_v1` | 当前 baseline | 不含 world，使用 BATA-baseline 纯净版合同 |

数据版本注册表：

- [数据版本说明](./ONEREASON_DATASET_VERSIONS.md)
- [数据版本清单](./data/onereason_dataset_versions.json)
- [数据集注册表](./data/dataset_info.json)

常用服务器路径示例：

```text
/data/lf_data_versions/alltrain/BETA_material_aligned_v1
/data/lf_data_versions/task_pools
/data/lf_data_versions/task_pools_abnormal
```

更新数据版本前，先执行 manifest、字段、数量守恒和 SHA-256 审计；不要覆盖已注册版本，也不要把服务器原始数据直接提交到 GitHub。

## 指标与结果解读

### 评测分数

评测器输出的总分和分项分数是最终效果判断依据。当前首先看总分是否接近或超过 **BATA-baseline 的 1.3246（约 1.33）**，再看懂物料、懂用户、懂推荐和懂世界的分项变化。历史记录中的懂物料分数曾存在不可参考阶段，必须以具体实验记录中的说明为准。

### 训练损失

- `total_loss`：训练实际反向的总 loss；不同实验若改变 SID 权重、CoT 权重、任务配额或 loss route，绝对值不可直接横比。
- `task_loss_material`、`task_loss_recommendation`、`task_loss_user_action`、`task_loss_user_chain`：各任务 raw loss 或路由后的任务 loss，应在同一实验内看时间趋势。
- `grad_norm`：当前更新前的梯度范数；若长期高于裁剪阈值，实际更新会受到 clipping 影响。
- `learning_rate`：学习率调度位置；cosine 后期降低不代表模型一定已经收敛。

### GradNorm 与冲突监控

旧多任务路线还记录任务权重、梯度范数、任务间梯度余弦、residual 和 DDP 一致性。权重触底只能说明控制器在当前 raw loss/梯度尺度下持续压低该任务，不等价于任务已经学好；应结合任务质量和梯度趋势判断。

### Recommendation / Action 指标

- `a/b/c_rec_*`：推荐 SID 各层级的原始 CE；用于比较 CoT、No-think 及 a/b/c 学习难度。
- `gold_sid_ce`：正确 SID token 的原始 CE；不是加权 task loss。
- `candidate_hit`、`coverage`、`chain`：teacher-forcing 下的候选命中、集合覆盖和链路完整性。
- `positive_mass`、`U_mass`、`P-vs-U margin`：Set-PU 或多正例实验中的正例集合质量。
- `topk_illegal_rate`、`top1_illegal_hit_rate`、`gold_top5_rate`：Action Select 非法 SID 与 gold 命中情况。
- `rec_cot_body_numerator_share`、`rec_cot_final_answer_numerator_share`、`rec_nocot_numerator_share`：Alpha-CoT 加权 CE numerator 的组成，只用于解释 loss 尺度，不直接代表生成质量。

### 验证集与固定 probe

Alpha 监控路线使用 98/2 leak-safe dev 和固定 probe；验证只复用已有 logits 或单独的 teacher-forcing 流程，不改变训练梯度。`tf_chain` 表示 teacher-forcing 下的行为链评估，`pathnull` 表示候选路径为空或未形成有效路径的统计，不等价于训练失败。

## 常用配置与测试

核心 Native 配置位于 [`baselines/native_source_domain_r32_v3/config`](./baselines/native_source_domain_r32_v3/config)。旧 rank16 配置仍位于 [`configs/onereason`](./configs/onereason)。

运行 CPU 回归：

```bash
cd /app/LLaMA-Factory
PYTHONPATH=src python3 tests/test_multitask_macro.py
PYTHONPATH=src python3 tests/test_multitask_gradient_controller.py
PYTHONPATH=src python3 tests/test_sid_token_weighting.py
PYTHONPATH=src python3 tests/test_onereason_dataset_versions.py
```

启动训练前必须确认：数据版本 manifest、实际 loss route、SID/domain 权重、输出目录和 GPU 资源均与实验记录一致。正式训练命令以对应 YAML 或启动脚本为准，不要直接复制其他实验的 output_dir。

## 工程边界

- 不上传原始数据、模型权重、checkpoint、日志和密钥。
- 不在未记录的情况下改变 LoRA、模型 forward、optimizer、scheduler、packing 或任务采样比例。
- 新 loss 必须有独立开关、数学回归和最小 smoke；关闭开关时应恢复 baseline 行为。
- 数据清洗、数据版本和实验配置分离管理，保证任何单个子任务都可以独立替换和回退。

## 许可证

代码沿用 LLaMA-Factory 的 Apache-2.0 许可证；模型和竞赛数据须遵守各自许可证及赛事规则。
