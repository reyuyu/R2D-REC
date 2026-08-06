# 实验 D：SID 加权 SFT

## 目标

对所有监督答案中的 `<s_a_*>`、`<s_b_*>`、`<s_c_*>` token 使用归一化加权交叉熵，SID 权重为 8，普通文本权重为 1。该实验不改变模型前向、packing、GradNorm、Ortho、优化器或调度器。

## 配置

`configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_sid_weight8.yaml`

## 损失

```text
loss = sum(per_token_ce * token_weight * valid_mask) / sum(token_weight * valid_mask)
```

只加权 `labels != -100` 的监督 SID token，因此 prompt/history 中的 SID 不参与。

## 正式 checkpoint 评测

以下按物料四域、用户两项、推荐四域、world 的固定顺序记录。

| Checkpoint | 总分 | 物料 4 域 | 用户 2 项 | 推荐 4 域 | World |
| ---: | ---: | --- | --- | --- | ---: |
| 1000 | 1.1828 | 0.0442, 0.0373, 0.0388, 0.0425 | 0.1321, 0.0840 | 0.1055, 0.1428, 0.1862, 0.1449 | 0.2245 |
| 5200 | 1.2433 | 0.0460, 0.0378, 0.0442, 0.0418 | 0.1551, 0.0948 | 0.1092, 0.1530, 0.1806, 0.1521 | 0.2286 |

实验 D 已完成 5200 步训练。最终总分为 1.2433；后续比较时仍需结合 SID 加权质量占比与自由生成质量，避免只根据总分归因。
