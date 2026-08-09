# 实验 Baseline：NSD-R32-V3-2E-GC04-4GPU

这是 `NSD-R32-V3-2E-GC04-2GPU` 的四卡正式运行版。训练目标、数据、模型、LoRA、SID 权重、0.4GC 和优化器完全相同，仅把并行度改为 4 卡并把梯度累积从 32 调为 16，因此全局 batch 仍为 64，保证可归因比较。

## 配置

- GPU：0、1、2、3；每卡 batch 1；`gradient_accumulation_steps=16`；全局 batch 64。
- 训练 2 epoch；33,515 packed sequence；524 optimizer step/epoch；总 1,048 step。
- 8K neat packing、BF16、FA2、Liger 0.8.1、LoRA rank 32/alpha 64/dropout 0.05。
- learning rate `2e-4`、cosine、warmup `0.03`、weight decay `0.01`、seed `20260806`。
- `NATIVE_GC_FRACTION=0.4`：Qwen3 36 层中固定 checkpoint 前 14 层。
- 数据版本 `native_source_domain_r32_v3`，219,370 条，不含懂世界。

## 监控与保存

每 5 step 记录 `loss`、`grad_norm`、`learning_rate`、`epoch` 及四项观测损失：`task_loss_material`、`task_loss_recommendation`、`task_loss_user_action`、`task_loss_user_chain`。四项只观察、不参加反传。每 262 step 保存完整 optimizer/scheduler checkpoint，最多保留 4 个；TensorBoard 未安装，指标写入 `train.log` 与 checkpoint 的 `trainer_state.json`。

输出：

```text
/data/outputs/baselines/native_source_domain_r32_v3/NSD-R32-V3-2E-GC04-4GPU-20260810/
/data/logs/baselines/native_source_domain_r32_v3/NSD-R32-V3-2E-GC04-4GPU-20260810/train.log
```

启动脚本：`scripts/launch_4gpu_gc04_2epoch.sh`。启动前已停止服务器上的其它训练进程，四张 A800 均空闲。
