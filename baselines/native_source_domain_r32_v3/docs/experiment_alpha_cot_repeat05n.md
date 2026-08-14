# 实验 Alpha-CoT：Recommendation CoT Repeat-Normalized Weighting

状态：已完成 2 epoch 训练；epoch 1 外部评测结果已记录，epoch 2 结果待补充。

母版为 corrected Alpha SID8：四卡、LoRA r32/alpha64/dropout 0.05、8K neat packing、GA16、GC0.4、LR 2e-4 cosine、SID/domain 权重 8、canonical 全 response 权重 4，REC-PU 与 PackRatio 均关闭。开发集/probe 沿用 Alpha 的固定 corrected SID8 cache。

唯一变量：仅对 `source_segment == recommendation_cot` 的监督 `<think>...</think>` span（两个边界 token 也包含在内）使用 `0.5 / N_cot(group_id)`。`N_cot` 是当前 train98 中同一 `recommendation_group_id` 的实际 CoT row 数，来自冻结 manifest；绝不回退到 `recommendation_group_size` 或 1。

`</think>` 后保持 baseline：自然语言答案前缀和 `<|im_end|>\n` 权重 1；final domain、`s_a`、`s_b`、`s_c` 各为 8。NoThink、物料、user action、user chain、canonical 均保持原 cache 不变。

实现开关：`alpha_cot_repeat_weighting: true`，默认关闭。新 cache 位于本机大容量 cache 卷 `/app/lf_tokenized/alpha-jiankong-split-v1/tokenized_train98_8k_sid8w8_cot05n`，不覆盖 `/data/.../tokenized_train98_8k_sid8w8`。

安全约束：缺失 group_id、manifest 中不存在 N、N<=0、或监督 think span 非唯一时 fail closed。必须验证新旧 cache 除 `loss_weights` 外逐列一致、pack 数仍为 33,810；所有 changed positions 必须且仅能处于 CoT body。

## Epoch 1 外部评测

本次提供的 epoch 1 结果如下（总分及分项顺序沿用外部评测器）：

```text
aggregate = 1.2263
material  = 0.0446, 0.0375, 0.0448, 0.0434
user      = 0.1461, 0.0923
recommendation = 0.1148, 0.1292, 0.1778, 0.1620
world/last = 0.2338
```

当前观察是：外部评测中的 recommendation 分项低于 Alpha 基线，但 Alpha-CoT 的固定 probe / dev teacher-forced 指标并未同步恶化，`prob` 类监控在 epoch 1 反而较好。该现象目前不能判定为验证集错误，原因是两类指标测量对象不同：

1. teacher-forced CE、gold probability 和 hit 指标在真实 gold prefix 下预测最终 SID，主要衡量条件概率和局部 token 质量；它们不要求模型在自由生成时正确决定何时停止、如何从多个候选中选完整集合。
2. 外部 recommendation 分数通常包含自由生成后的完整 SID 集合、去重、格式、漏选/多选及域级聚合；一个模型可以在 gold prefix 下概率更高，但自由生成时仍因候选排序、停止或集合组合误差拿到更低分。
3. 本实验把同一 recommendation group 的 CoT think span 权重改为 `0.5/N_cot`，并没有直接加强最终 SID。因而训练 loss 的尺度下降是预期的；它还可能改变共享表示和 CoT/NoThink 两路之间的相对校准，使 teacher-forced 指标改善而生成式集合指标下降。

因此当前结论是“验证/概率指标与外部推荐分数不一致，需继续做生成侧误差拆分”，而不是“验证集无效”。后续比较必须固定 checkpoint、评测脚本、数据版本和分项顺序，并至少同时报告：gold SID CE、TF hit/chain、自由生成的历史外 SID、重复 SID、漏选、多选和停止错误。

## Alpha-CoT 与 Alpha-Jiankong 横向对比（epoch 1）

两组均使用 4 GPU、8K neat packing、固定 733-pack full dev；Alpha-CoT 使用
`ALPHA-COT-REPEAT05N-R32-2E-GC04-4GPU/checkpoint-529`，Alpha-Jiankong 使用
`ALPHA-JIANKONG-SID8FIX-R32-2E-GC04-4GPU-20260813-045032/checkpoint-529`。

| 指标 | Alpha-Jiankong | Alpha-CoT | CoT - Jiankong |
|---|---:|---:|---:|
| CoT body CE | 1.2438 | 1.3735 | +0.1298 |
| CoT Gold SID CE | 4.6503 | 4.6410 | -0.0093 |
| NoThink Gold SID CE | 4.6345 | 4.6311 | -0.0034 |
| Gold a CE | 4.9483 | 4.9405 | -0.0078 |
| Gold b CE | 4.5233 | 4.5274 | +0.0041 |
| Gold c CE | 4.4597 | 4.4429 | -0.0164 |
| Gold probability a/b/c | 0.04647 / 0.08232 / 0.11012 | 0.04661 / 0.08309 / 0.11269 | 均略升 |
| TF a-hit32 | 0.5759 | 0.5806 | +0.0047 |
| TF b-hit8 | 0.4204 | 0.4166 | -0.0038 |
| TF c-hit8 | 0.4826 | 0.4826 | 0 |
| TF chain 32/8/8 | 0.1329 | 0.1310 | -0.0019 |
| CoT TF a-hit32 | 0.5668 | 0.5684 | +0.0016 |
| NoThink TF a-hit32 | 0.5886 | 0.5977 | +0.0091 |

外部最终评分对比：

| 分项 | Alpha-Jiankong | Alpha-CoT | 差值 |
|---|---:|---:|---:|
| 总分 | 1.2605 | 1.2263 | -0.0342 (-2.71%) |
| 推荐 video | 0.1204 | 0.1148 | -0.0056 |
| 推荐 prod | 0.1394 | 0.1292 | -0.0102 |
| 推荐 ad | 0.2016 | 0.1778 | -0.0238 |
| 推荐 living | 0.1521 | 0.1620 | +0.0099 |
| 推荐四域平均 | 0.1534 | 0.1460 | -0.0074 (-4.84%) |

结论：Alpha-CoT 没有表现为 teacher-forced 推荐能力全面变差。它在 epoch1 的 CoT/NoThink Gold CE、a/c CE、gold probability 和 a-hit32 略好；但外部自由生成评分在 video、prod、ad 三域下降，导致推荐四域平均下降约 4.84%。因此问题更像是从 gold prefix 到完整集合生成的误差（候选排序、重复/漏选、停止或域偏置），而不是验证集本身损坏。`CoT body CE` 不应与 Alpha-Jiankong 直接按绝对值比较，因为 Alpha-CoT 的训练权重只改变训练目标，验证侧该值仍会受模型表示变化影响。
