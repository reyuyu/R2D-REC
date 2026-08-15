# Mini-V3：推荐全部恢复为 CoT（NoCoT=0）

## 实验定位

Mini-V3 是 Mini 系列在 Mini-V2 基础上的第二个数据消融：把懂推荐样本**全部恢复为 CoT**（NoCoT=0），验证「推荐任务全 CoT 化」对最终得分的影响。除推荐数据形态外，模型、LoRA、优化器、packing 与评测方式与 Mini-V2 / Alpha-Mini 一致。

## 数据构造

训练数据集：`onereason_mini_v3`（`/data/lf_data_versions/alltrain/mini_v3`），总计 **50,490** 行。

| 任务池 | 与 Mini-V2 对比 | 行数 |
|---|---|---|
| 懂用户 `mini_v2` | 不变 | 3,000 |
| 懂推荐 `mini_v3` | 把 Mini-V2 中剩余 NoThink 行全部恢复为 CoT（同 group 唯一匹配的原始 CoT body），**NoCoT=0、全 CoT** | 11,192 |
| 懂物料 `alpha_mini` | 不变 | 36,298 |
| **合计** | | **50,490** |

推荐部分：`recommendation_mini_v3_all_cot.jsonl` 全部为 CoT 样本。恢复过程以 Mini-V2 的 11,192 行为基础，对每一条 NoThink 行在母集/历史 CoT 源中按同 group 唯一匹配非空 CoT body；无法唯一匹配的行在构建审计中单独记录，不伪造推理内容。推荐多正 metadata 完整保留。

构建时的补充审计包括：`mini_v3_cot_source_audit.json`（CoT 来源唯一性）与 `mini_v3_legacy_hdom_audit.json`（历史 domain 兼容性），见 `task_pools/懂推荐/mini_v2/` 与 `task_pools/懂推荐/mini_v3/`。

## 训练合同

- 配置：`config/train_alpha_mini_v3_4gpu_gc04_2epoch.yaml`
- 数据：`onereason_mini_v3` / `tokenized_mini_v3_train_8k_sid8w8`
- 训练 2 epoch；Native SID8 CE；`GLOBAL_ITEM_WEIGHT=8`；Alpha monitor/TF/validation 只读监控，不改变训练梯度
- 其余与 Alpha-Mini / Mini-V2 训练合同一致

## 评测结果（外部评测器，2 epoch）

```text
aggregate: 1.2178

0.0619, 0.0345, 0.0512, 0.0432
0.1513, 0.0772
0.0980, 0.1292, 0.1722, 0.1611
0.2379
```

分项顺序与 Alpha-Mini 记录一致（4/2/4/1）。与 Mini-V2 对比：总分 `1.2671 → 1.2178`（-0.0493）；推荐分项加和 `0.6325 → 0.5605` 继续下降，用户分项 `0.2142 → 0.2285` 上升，世界 `0.2305 → 0.2379` 上升。

## Mini 系列小结

| 实验 | 推荐 NoCoT 比例 | 总分 | 推荐分项加和 |
| --- | ---: | ---: | ---: |
| Alpha-Mini（大基线） | 约 42%（继承不完整过滤降级） | `1.2861` | `0.6537` |
| Mini-V2 | 约 1/3 | `1.2671` | `0.6325` |
| Mini-V3 | 0（全 CoT） | `1.2178` | `0.5605` |

结论：推荐分项随 NoCoT 比例下降而单调下降，全 CoT 化对推荐任务得分不利；用户与世界分项随 NoCoT 减少小幅上升。

## 机器可读来源

- 懂推荐：`/data/lf_data_versions/task_pools/懂推荐/mini_v3/manifest.json` + `recommendation_mini_v3_all_cot.jsonl`
- 组合：`/data/lf_data_versions/alltrain/mini_v3/manifest.json` + `dataset_info.json`
- 构建脚本：`build_recommendation_mini_v3_all_cot.py`、`build_mini_v3_combined_dataset.py`
