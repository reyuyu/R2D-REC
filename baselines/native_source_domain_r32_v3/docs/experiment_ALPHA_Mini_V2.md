# Mini-V2：NoCoT 恢复至 1/3 + 懂用户扩至 3,000

## 实验定位

Mini-V2 是 Mini 系列在 Alpha-Mini 大基线（`alpha_mini`，49,490 行）基础上的第一个数据消融：目标是验证「推荐 NoCoT 比例恢复至约 1/3、懂用户补充 1,000 条」对最终得分的影响。除数据构造外，模型、LoRA、优化器、packing 与评测方式与 Alpha-Mini 完全一致。

## 数据构造

训练数据集：`onereason_mini_v2`（`/data/lf_data_versions/alltrain/mini_v2`），总计 **50,490** 行。

| 任务池 | 与 Alpha-Mini 基线对比 | 行数 |
|---|---|---|
| 懂用户 `mini_v2` | 从 2,000 扩到 3,000（Action 1,800、Chain-CoT 300、Chain-NoThink 900），保留原 2,000 条并等比例补充 | 3,000 |
| 懂推荐 `mini_v2` | 把 Alpha-Mini 选中 group 中一部分 NoThink 行恢复为同 group 的原始 CoT（母集可唯一匹配），最终 NoCoT 占比控制在约 30%（≈1/3）；多正 group 结构与 metadata 不变 | 11,192 |
| 懂物料 `alpha_mini` | 不变（material_sample 10,000 + canonical 11,298 + reverse 15,000） | 36,298 |
| **合计** | | **50,490** |

推荐部分：输入沿用「不完整过滤」后的 48,269 行，保留 Alpha-Mini 选中的 1,549 个多正 group；NoThink 恢复 CoT 时按同 group 唯一匹配的 CoT body，不伪造推理内容。推荐多正 metadata 完整保留：`recommendation_group_id`、`recommendation_group_size`、`recommendation_current_gold_sid`、`recommendation_all_gold_sids`。

## 训练合同

- 配置：`config/train_alpha_mini_v2_4gpu_gc04_2epoch.yaml`
- 数据：`onereason_mini_v2` / `tokenized_mini_v2_train_8k_sid8w8`
- 训练 2 epoch；Native SID8 CE；`GLOBAL_ITEM_WEIGHT=8`；Alpha monitor/TF/validation 只读监控，不改变训练梯度
- 其余与 Alpha-Mini 训练合同一致（LoRA r32/a64、4×A800、GA16、8K neat packing、lr 2e-4 cosine、warmup 0.03、GC 0.4、seed 20260806）

## 评测结果（外部评测器，2 epoch）

```text
aggregate: 1.2671

0.0607, 0.0353, 0.0507, 0.0431
0.1333, 0.0809
0.1185, 0.1496, 0.1988, 0.1656
0.2305
```

分项顺序与 Alpha-Mini 记录一致（4/2/4/1）。与基线对比：总分 `1.2861 → 1.2671`（-0.0190）；推荐分项加和 `0.6537 → 0.6325`，用户分项 `0.2103 → 0.2142`，世界 `0.2301 → 0.2305`。NoCoT 降到 1/3 后推荐分项下降、用户分项略升。

## 机器可读来源

- 懂用户：`/data/lf_data_versions/task_pools/懂用户/mini_v2/manifest.json`
- 懂推荐：`/data/lf_data_versions/task_pools/懂推荐/mini_v2/manifest.json`
- 组合：`/data/lf_data_versions/alltrain/mini_v2/manifest.json` + `dataset_info.json`
- 构建脚本：`build_user_alpha_mini_v2.py`、`build_recommendation_mini_v2.py`、`build_mini_v2_dataset.py`
