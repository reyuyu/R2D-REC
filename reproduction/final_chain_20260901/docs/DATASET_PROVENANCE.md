# 最终复现数据来源与处理说明

最终四阶段复现只从 `/root/reproduce_datasets/onereason_final_chain_20260901` 读取冻结数据。全部样本来源于赛方官方数据，不包含第三方数据或教师模型生成数据。

| 数据集 | 来源与处理 | 对应脚本 |
| --- | --- | --- |
| BETA SFT | 汇总赛方官方懂物料、懂用户和懂推荐 SFT 数据。 | `baselines/native_source_domain_r32_v3/scripts/build_bata_baseline_dataset.py` |
| SFT 懂推荐 | 官方 Recommendation SFT。每行额外写入 `aux_metadata_json`，其中包含按“去掉末尾思考标记后的完整用户历史 + 目标域”聚合得到的正样本 SID 集合；普通 SFT 仍使用原有单答案 CE，这些元数据不改变监督目标。 | `baselines/native_source_domain_r32_v3/scripts/build_bata_baseline_dataset.py` |
| 懂推荐 GRPO | 从 SFT Recommendation 子集聚合出的 `1,549` 个多正样本组，每组保留 Think/NoThink 两路，共 `3,098` 条。v2 只把通用任务指令改成 video/prod/ad/living 域特定指令；历史 SID、路由、group ID 和 Gold 集合均不变。 | `baselines/native_source_domain_r32_v3/grpo/scripts/build_rec_mp_grpo_v2.py` |
| 懂用户 GRPO | 赛方官方懂用户训练集的冻结子集。 | 正式 runner 的固定 seed 与选择合同 |

懂推荐 GRPO 记录没有独立 `system` 字段。SFT 中的 system/任务指令在构造 GRPO 记录时合并进 `prompt`；因此 v1 到 v2 的逐行字段审计结果是仅 `prompt` 变化，而不是另有一个 `system` 字段发生变化。

SFT Recommendation 元数据字段为：

- `recommendation_group_id`
- `recommendation_group_size`
- `recommendation_all_gold_sids`
- `recommendation_current_gold_sid`

懂推荐 GRPO v1 到 v2 的真实数据审计结果：`3,098 / 3,098` 行仅 `prompt` 字段变化，`recommendation_group_id` 不匹配数为 `0`，历史 SID 行和 `/think`、`/no_think` 标记保持不变。
