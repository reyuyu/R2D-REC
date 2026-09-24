# 实验 Baseline：NSD-R32-V3-2E-GC04-2GPU

## 目的

本实验将 material-domain 训练路线作为 OneReason 数据上的独立原生 SFT 对照。它不启用 macro trainer、GradNorm、Action 辅助损失、推荐 Trie、多任务 coverage 调度或 BFD packing，用于区分基础训练配置与各项扩展的影响。

## 训练配置

| 项目 | 配置 |
|---|---|
| 参考源码 | LLaMA-Factory `01398eb18dd475a6e27c36f15b970aeacf0d4a60` |
| 模型 | `/data/models/onereason-8b-pretrain-competition` |
| LoRA | target `all`，rank 32，alpha 64，dropout 0.05 |
| 序列 | cutoff 8192，neat packing，BF16，FlashAttention-2，Liger 0.8.1 |
| 优化 | 原生 AdamW SFT，LR `2e-4`，weight decay `0.01`，cosine，warmup ratio `0.03` |
| 双卡 | GPU 2/3，每卡 batch 1，accumulation 32，全局 batch 64 |
| 训练量 | 2 epoch；33,515 个 packed sequence，524 step/epoch，合计 1,048 step |
| GC | `NATIVE_GC_FRACTION=0.4`，固定 checkpoint Qwen3 前 14/36 层 |
| 数据版本 | `native_source_domain_r32_v3`，219,370 条，无懂世界 |
| checkpoint | 每 262 step，保留 4 个，包含 optimizer/scheduler 状态 |

0.4GC 只改变激活重算范围，不修改模型参数、损失公式、优化器或学习率调度。

## 数据与分类

活跃数据使用统一 JSONL schema：`instruction`、`input`、`output`、`history`、`data_source`、`source_segment`、`aux_metadata_json`。

- 物料：`sid_bucket_canonical_no_think`、`sid_bucket_reverse`、`material_sample`
- 懂用户 Action：`source_segment=user_action`
- 懂用户 Chain：`source_segment=user_chain_cot` 或 `user_chain_nocot`
- 懂推荐：`data_source=recommend`，并保留 Recommendation V3 的全量多正例 metadata

`aux_metadata_json` 不传给模型；当前基线不使用它。后续可据此构造多正例推荐 loss，不需要重新解析 prompt，也不会破坏现有数据版本。

## 中间指标

每 5 个 optimizer step 记录：

- `loss`：当前训练真实总损失；
- `grad_norm`；
- `learning_rate`；
- `epoch`；
- `task_loss_material`；
- `task_loss_recommendation`；
- `task_loss_user_action`；
- `task_loss_user_chain`。

四个 `task_loss_*` 仅为观测：复用同一次 forward 的 token CE，先按 pack segment 聚合，再进行跨卡求和和任务内样本均值。物料项包含其原有域权重，因此与真实总损失口径一致。它们不参与反传、不会引入新的 loss，也不会改变基线数学。

## 目录和日志

```text
/data/baselines/native_source_domain_r32_v3/
  config/    固化 YAML
  dataset/   当前数据快照、manifest、versions.json
  scripts/   构造、审计、训练、启动脚本
  docs/      实验记录与运行规范

/data/outputs/baselines/native_source_domain_r32_v3/NSD-R32-V3-2E-GC04-2GPU-20260810/
  checkpoint-*/

/data/logs/baselines/native_source_domain_r32_v3/NSD-R32-V3-2E-GC04-2GPU-20260810/train.log
```

隔离参考环境未安装 TensorBoard，因此不额外引入依赖；完整指标写入 `train.log`，并在每次 checkpoint 的 `trainer_state.json` 中持久化完整 `log_history`。

## 预检

| GC 档位 | 5-step 结果 | 每 step | 峰值显存/卡 |
|---|---|---:|---:|
| full GC | 通过 | 87.85 秒 | 41.3 GiB |
| 0.5GC，18/36 层 | 通过 | 75.64 秒 | 70.1 GiB |
| 0.4GC，14/36 层 | 通过 | 73.01 秒 | 76.3 GiB |
| no-GC | 首步 OOM | - | 超过 80 GiB |

0.4GC 烟测通过但只有约 4 GiB 显存余量，是吞吐优先的激进档位。按 73.01 秒/step 估算，2 epoch 约 21.3 小时，未含初始化与 checkpoint I/O。
