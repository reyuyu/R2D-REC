# rec_A 与 recB：四任务调度对照

更新时间：2026-08-07

## 目的

在相同的模型、数据版本、GradNorm、SID 加权和显存配置下，对照原有 `rec_A` 调度与 `recB` 的 `1:3:3:1` 四任务调度。

四个顶层任务为：

```text
material / user_action / user_chain / recommendation
```

## 共同配置

- 模型：`/data/models/onereason-8b-pretrain-competition`
- LoRA：rank 16，alpha 32，dropout 0.05
- 数据版本：`v2_recommendation_dual`
- 序列长度：8192；`packing: false`
- 学习率：`2e-4`，cosine，warmup ratio `0.03`
- batch：每卡 1，gradient accumulation 1，双卡 DDP
- bf16：开启
- seed：42
- SID token weighting：开启，`sid_token_weight=8`
- Action auxiliary loss：关闭
- GradNorm：四任务开启，更新间隔 10，参考最后 4 层
- Gradient checkpointing：reentrant，覆盖比例 0.75
- 训练上限：5200 macro-step

## rec_A

配置：`configs/onereason/onereason_lora_2gpu_rec_A_r16_gradnorm_sid_weight8_v2_dual_gc075.yaml`

调度：

```text
multitask_supercycle_mode: balanced_40
material: 4
user_action: 1
user_chain: 1
recommendation: 2
```

截至 2026-08-07 17:33 左右，`rec_A` 已运行到约 `3568/5200` macro-step，稳定速度约 `12.4-12.6 秒/macro-step`，GPU 2/3 持续满载，未观察到新的 OOM 或 traceback。输出目录已有 `checkpoint-3500`。

### rec_A checkpoint 评测结果

| Checkpoint | 总分 | 物料 4 域 | 用户 2 项 | 推荐 4 域 | World |
| ---: | ---: | --- | --- | --- | ---: |
| 1000 | 1.1702 | 0.0430, 0.0371, 0.0403, 0.0425 | 0.1405, 0.0831 | 0.1017, 0.1530, 0.1610, 0.1323 | 0.2357 |
| 3500 | 1.2273 | 0.0440, 0.0370, 0.0434, 0.0418 | 0.1498, 0.0967 | 0.1083, 0.1258, 0.1792, 0.1611 | 0.2401 |

## recB

配置：`configs/onereason/onereason_lora_2gpu_recB_r16_gradnorm_sid_weight8_v2_dual_ratio1311_gc075.yaml`

调度：

```text
multitask_supercycle_mode: fixed
material: 1
user_action: 3
user_chain: 3
recommendation: 1
```

`recB` 已将优化器与 `rec_A` 对齐为普通 `adamw_torch`，并使用独立输出目录。

### 第一次运行：fused AdamW

- 日志：`/data/logs/recB_ratio1311.log`
- 训练开始：16:34:30
- 退出：16:39:24
- 进度：`22/5200`
- 错误：rank 1 / GPU 1 在 backward 阶段 `torch.OutOfMemoryError`
- 额外申请：约 `5.27 GiB`
- 当时可用显存：约 `4.41 GiB`
- PyTorch reserved-but-unallocated：约 `17.03 GiB`

### 第二次运行：普通 AdamW

- 日志：`/data/logs/recB_ratio1311_adamw.log`
- 训练开始：17:19:37
- 退出：17:24:21
- 进度：`22/5200`
- 错误：同样是 rank 1 / GPU 1 backward 阶段 OOM
- 失败位置和显存特征与第一次运行基本一致

## 当前结论

1. `1:3:3:1` 的调度配置已经生效，日志中的每个 macro 都显示该 allocation。
2. 将 `adamw_torch_fused` 改回 `adamw_torch` 没有解决 step 22 的 OOM，因此优化器不是主要根因。
3. 当前主要风险是 8192 长序列下 rank 1 的显存峰值，以及显存碎片；`user_chain=3 + recommendation=1` 的 rank 分配可能更容易触发峰值。
4. `recB` 两次都在 `save_steps=500` 之前退出，没有生成可恢复 checkpoint。
5. 若继续运行 `recB`，应优先测试完整梯度检查点、较短 `cutoff_len` 或显存分配器配置，并先用短 smoke 验证，而不是继续重复完整启动。

## 后续 REC 阶段

recC、recD、recE 和 REC_F 的 BFD、coverage/deficit、cost-aware、自适应长度与最终 8K 配置，集中记录在 [REC 自适应 Packing 实验记录](./实验recC_recD_recE_REC_F_自适应Packing.md)。
