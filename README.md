# OneReason 多任务 SFT

本仓库记录 OneReason-8B 的数据版本、训练代码、可复现实验配置和评测结果。当前 README 只展示三条主线：**Alpha 系列**、**Mini 系列**，以及作为统一参考线的 **BETA-baseline 纯净版**。

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

1.33 是当前横向比较的参考分数，不是额外的硬性验收阈值。新实验必须同时报告总分和分项分数，不能仅凭 raw task loss 判断优劣。

详细合同和结果：[BETA-baseline 纯净版](./baselines/native_source_domain_r32_v3/docs/BATA_BASELINE.md)。

## Alpha 系列

Alpha 系列基于 baseline，重点研究数据提示、数据清洗、Action Select 约束和训练监控。除记录中明确说明的变量外，模型、LoRA、优化器和评测方式保持一致。

| 实验 | 主要改动 | 关注点 |
| --- | --- | --- |
| Alpha-监控优化 | 98/2 leak-safe dev、固定 probe、训练阶段验证 | 训练损失和验证指标是否同步，避免数据泄漏 |
| Alpha-SID8 cache fix | 修复静态 tokenized cache 的 SID/domain 权重合同 | 普通 token=1、canonical=4、非 canonical SID/domain=8 |
| Alpha-CoT | Recommendation CoT 重复归一化，CoT body 使用 `0.5/N` | 降低重复 CoT 对训练 numerator 的主导，同时保持 Gold SID=8 |

Alpha 记录入口：

- [Alpha 监控优化](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_监控优化.md)
- [Alpha 监控结果分析](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_监控优化_结果分析.md)
- [Alpha SID8 cache 修复](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_SID8_cache_fix.md)
- [Alpha-CoT 重复归一化](./baselines/native_source_domain_r32_v3/docs/experiment_alpha_cot_repeat05n.md)

Alpha-CoT 的正式运行配置和代码位于服务器的 Native baseline 工作目录；其核心对比必须使用 raw CoT body CE、CoT Gold SID CE、No-think Gold SID CE 和验证集指标，不能直接比较改变权重后的 recommendation task loss。

## Mini 系列

Mini 系列是 Alpha 正式实验的轻量复现和排查版本，用于在不启动完整 2 epoch 的前提下验证数据、loss route、梯度和监控。它服务于“先证明代码和指标正确，再启动正式训练”的流程。

| Mini 阶段 | 用途 | 典型验证 |
| --- | --- | --- |
| Mini-CPU | 不加载大模型的数学和数据回归 | loss_weights、SID8、CoT `0.5/N`、字段和 manifest |
| Mini-单步/短程 | 从 base model 运行 1-30 optimizer steps | forward/backward、raw CE parity、无 NaN/OOM |
| Mini-监控 | 使用固定 probe 和少量 dev 样本 | 四任务 loss、teacher-forcing 命中、验证指标 |
| Mini-复现包 | 打包代码、配置、数据 manifest 和复现说明 | 本地恢复数据并重建 tokenized cache |

Mini 版本不作为最终排行榜结果，不覆盖 Alpha 正式 output，也不改变正式训练的 scheduler horizon。当前 mini 复现包不包含模型权重，原始数据仍需根据 manifest 从服务器或本地数据源恢复。

Mini 记录：[Alpha Mini R32 两 epoch](./baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_Mini_R32_2E.md)。

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

首先看总分是否接近 BETA-baseline 的 `1.3246`，再看懂物料、懂用户和懂推荐的分项变化。不同版本若更改数据、SID 权重或输出格式，必须结合实验记录解释，不能只比较一个 checkpoint 的总分。

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
