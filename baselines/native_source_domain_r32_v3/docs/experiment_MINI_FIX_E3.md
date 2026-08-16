# 候选实验 Mini-Fix-E3：Mini-Fix 延长至 3 epoch

状态：**已配置，排队等待 Mini-Fix-U3K 成功结束后自动启动**（2026-08-16）。

## 动机与假说

Mini-Fix 在 2 epoch 的外部评测达到总分 `1.3093`、推荐分项 `0.6644`。该训练已完成 138 个优化步，终点的训练与固定验证监控均正常。E3 是一个严格单变量候选：保持 Mini-Fix 的全部数据、推荐 CoT/NoCoT 教学合同、模型、packing、优化器、权重与固定验证路线，只把训练上限从 `2` 改为 `3` epoch。

**待验证假说**：Mini-Fix 的高纯度 NoCoT 与短思考 CoT 教学可能仍有可利用的第三轮拟合空间；反面风险是推荐在第三轮出现过拟合。因此 E3 的价值不在于预设第三轮必然更高，而在于把 epoch 2 与 epoch 3 放在相同合同下比较。

## 唯一变量

| 项目 | Mini-Fix | Mini-Fix-E3 |
| --- | ---: | ---: |
| 数据集 | `onereason_mini_fix`，49,490 行 | **完全相同** |
| 推荐合同 | NoCoT 44.3%、纯度 100%；cot 保留短思考教学 | **完全相同** |
| 训练轮数 | 2 | **3** |
| 其余训练/验证合同 | R32/a64、8K neat packing、GA16、GC0.4、LR `2e-4` cosine、seed `20260806`、固定 Mini-disjoint dev/probe | **完全相同** |

配置通过 `save_strategy: epoch` 和 `save_total_limit: 4` 保留 3 个 epoch checkpoint 与最终导出，避免第三轮结果覆盖第二轮可复核模型。

## 自动串行启动

上游 Run：`MINI-FIX-U3K-R32-2E-GC04-4GPU-20260816-174640`。守护脚本：`scripts/wait_for_mini_fix_u3k_then_launch_mini_fix_e3.sh`。

守护条件：上游 torchrun PID 退出后，必须同时存在根目录 `train_results.json` 和 `trainer_state.json`，日志必须含最终 `train metrics` 标记，且 `trainer_state.epoch >= 1.999`。任一条件不满足，脚本记录失败并**不会**启动 E3；目录锁也会阻止重复启动。

## 评测与判定

Mini-Fix 的已知锚点为总分 `1.3093`、推荐 `0.6644`。E3 应分别评测 epoch 2 checkpoint、epoch 3 checkpoint和最终导出，并报告总分、物料 4 项、用户 2 项、推荐 4 项和 world。

第三轮仅在推荐分项不低于 `0.6600` 且总分不低于 `1.3093` 时视为可保留；若第三轮推荐下降，则以 E3 的 epoch 2 checkpoint 作为同合同参照，记录第三轮为过拟合信号，而不是覆盖 Mini-Fix 的 2 epoch 结论。

## 相关文件

- 配置：`config/train_mini_fix_e3_4gpu_gc04.yaml`
- 启动：`scripts/launch_mini_fix_e3_4gpu_gc04.sh`
- 守护：`scripts/wait_for_mini_fix_u3k_then_launch_mini_fix_e3.sh`
- 上游数据：`/data/lf_data_versions/alltrain/mini_fix/`
- E3 输出：`/data/outputs/baselines/native_source_domain_r32_v3/MINI-FIX-E3-R32-3E-GC04-4GPU-<timestamp>/`
