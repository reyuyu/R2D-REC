# 实验一：GradNorm-lite 多任务梯度平衡

## 目标

验证多任务训练中，仅通过动态调整任务梯度尺度，是否能够改善任务间的不平衡，并为后续 Ortho-LoRA 提供稳定梯度基础。

## 核心思想

GradNorm 不直接修改模型结构，而是动态调整任务 loss 权重：

```
L = w_material * L_material
  + w_user * L_user
  + w_recommendation * L_recommendation
```

其中权重根据：

- 当前任务梯度范数；
- 当前任务学习速度；
- 历史 EMA 平滑统计；

动态更新。

## 任务范围

第一阶段只处理三个主要任务：

- material
- user
- recommendation

world 保持固定权重：

```
w_world = 1.0
```

原因：world 在 balanced_40 中出现频率较低，不适合参与标准 GradNorm。

## 实验流程

### Warmup 阶段

前 200 个 macro-step：

- 不修改任务权重；
- 记录 loss EMA；
- 记录梯度范数 EMA；
- 记录任务间梯度 cosine。

### GradNorm 阶段

200 step 后开启：

- 每 10 个 macro-step 更新一次任务权重；
- 当前步统计结果用于下一 macro-step；
- 使用 EMA 避免单 batch 噪声导致权重震荡。

## 实际配置

```yaml
multitask_gradient_control_enabled: true
multitask_gradient_monitor_enabled: true
multitask_gradnorm_enabled: true
multitask_gradnorm_tasks: [material, user, recommendation]
multitask_world_loss_weight: 1.0
multitask_gradnorm_warmup_steps: 200
multitask_gradnorm_update_interval: 10
multitask_gradnorm_alpha: 0.5
multitask_gradnorm_update_rate: 0.10
multitask_gradnorm_loss_ema_beta: 0.90
multitask_gradnorm_grad_ema_beta: 0.90
multitask_gradnorm_weight_min: 0.5
multitask_gradnorm_weight_max: 2.0
multitask_gradnorm_step_ratio_min: 0.9
multitask_gradnorm_step_ratio_max: 1.1
multitask_grad_reference_last_n_layers: 4
multitask_grad_reference_modules: [q_proj, v_proj, o_proj, down_proj]
multitask_grad_reference_lora_matrix: B
```

完整配置位于：

```text
configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm.yaml
```

原 rank16 配置保持不变，作为关闭梯度控制的基线。

## 参考梯度参数

为了控制计算成本：

- 不计算全部模型梯度；
- 选择最后四层 LoRA-B；
- 模块包括 q_proj、v_proj、o_proj、down_proj。

实现根据参数名中的 Transformer layer index 选择最后 N 层，只接收
`requires_grad=True` 且匹配 `lora_B` 的张量。如果没有匹配项，启动会报错并列出候选 LoRA 参数名。

## 代码入口与缩放语义

- 参数入口：`src/llamafactory/hparams/data_args.py`
- 可选控制器：`src/llamafactory/train/sft/multitask_gradient_controller.py`
- macro-step 接入：`src/llamafactory/train/sft/trainer.py`

控制器只在总开关开启时创建。测量 step 中，参数 Hook 捕获当前正常 backward 写入的局部、已加权、已按
loss divisor 缩放的任务贡献；Hook 原样返回梯度，不修改 DDP backward。

每个任务只同步一个连续 FP32 flat vector。同步后的“实际加权总贡献”与最终参数 `.grad` 使用相同的
DDP 平均语义。任务平均原始梯度按下式还原：

```text
raw_mean_gradient = averaged_captured_contribution
                  * loss_divisor * ddp_world_size
                  / (task_weight * global_task_microbatch_count)
```

这里的 divisor、world size 和 count 均从当前训练状态取得，不硬编码 4 或 8。

## Lagged 更新

每个 macro-step 开始时先固定本步权重。测量结束后计算出的候选权重经过单步比例限制、全局上下界和
`sum(weights)=3` 归一化后写入 `pending_weights`。本步 backward 仍使用旧权重，pending 只在下一
macro-step 开始时应用。

## 启动与消融

GradNorm-lite：

```bash
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 \
llamafactory-cli train configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm.yaml
```

只监控时，在新配置中改为：

```yaml
multitask_gradient_control_enabled: true
multitask_gradient_monitor_enabled: true
multitask_gradnorm_enabled: false
```

此时三个主任务和 world 默认权重均保持 1.0，仍记录 norm/cosine，模型梯度不被改变。完全关闭时设置
`multitask_gradient_control_enabled: false`；不会创建控制器、Hook、buffer 或新增 collective。

## 观察指标

指标写入现有 `output_dir/monitor/metrics.jsonl`，并合并进 Trainer 同一步的常规训练日志供平台回调使用：

- `a_gn_w_*`：本 macro-step 实际使用的任务权重；
- `b_gn_loss_ema_*`、`b_gn_loss_ratio_*`：未加权原始 loss EMA 及相对 baseline 比值；
- `c_gn_raw_norm_*`：任务单 microbatch 平均原始梯度 norm；
- `c_gn_effective_norm_*`：当前权重乘以 raw norm；
- `d_gn_cos_*`：三个主任务两两 cosine；
- `e_gn_measure_active`、`e_gn_weight_update_active`、`e_gn_max_weight_delta`：测量/更新状态；
- `f_gn_capture_ms`、`f_gn_sync_ms`、`f_gn_update_ms`：新增功能耗时。

CUDA capture timing 使用 Event 并在测量结束同步，sync/update timing 在边界同步，避免把异步 kernel
提交时间误当成实际耗时。

## Checkpoint

rank 0 在每个 Trainer checkpoint 内写入：

```text
multitask_gradient_controller.json
```

其中保存当前权重、pending、loss/gradient EMA、冻结 baseline、计数和最近 cosine。恢复时由 rank 0
读取并广播到所有 rank，因此下一 macro-step 的 pending 应用和更新周期与连续训练一致。

## 竞赛评分记录

由于“懂物料”评分目前存在问题，本节保留其原始结果用于追溯，但不将其用于实验间横向比较。本文定义：

```text
可比较分数 = 懂用户 2 项之和 + 懂推荐 4 项之和 + 懂世界
```

各项分数均为四位小数，分项加和与平台总分可能有 `0.0001` 的舍入差异。

rank16 基础实验未启用实验 A 的动态梯度控制，用作相同 rank16 设置下的参考基线。所有原始分项和
排除懂物料后的横向结果统一列在下表中：

| 实验 | checkpoint | 平台总分 | 懂物料（4 项，不作比较） | 懂用户（2 项） | 懂推荐（4 项） | 懂世界 | 可比较分数 | 相对同 checkpoint 基础实验 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| rank16 基础实验 | 1000 | 1.0268 | 0.0035, 0.0156, 0.0008, 0.0000 | 0.1309, 0.0874 | 0.0896, 0.1564, 0.1708, 0.1458 | 0.2260 | 1.0069 | - |
| **实验 A** | **1000** | **1.0448** | 0.0037, 0.0164, 0.0021, 0.0000 | 0.1310, 0.0846 | 0.0915, 0.1598, 0.1820, 0.1440 | 0.2297 | **1.0226** | **+0.0157** |
| rank16 基础实验 | 2500 | 1.0458 | 0.0038, 0.0051, 0.0038, 0.0000 | 0.1442, 0.0951 | 0.0756, 0.1734, 0.1722, 0.1413 | 0.2312 | 1.0330 | - |
| **实验 A** | **2500** | **1.0401** | 0.0032, 0.0000, 0.0047, 0.0000 | 0.1476, 0.0922 | 0.0691, 0.1802, 0.1694, 0.1422 | 0.2316 | **1.0323** | **-0.0007** |
| rank16 基础实验 | 5200（最终） | 1.0637 | 0.0013, 0.0041, 0.0026, 0.0000 | 0.1483, 0.1003 | 0.0821, 0.1768, 0.1708, 0.1431 | 0.2342 | 1.0556 | - |

> 备注：表中所有懂物料评分目前均不具备参考意义。实验 A 的 checkpoint-5200 尚待评测。

当前结果表明，实验 A 在 checkpoint-1000 的可比较分数高于基础实验，但到 checkpoint-2500 时两者基本持平。
在获得实验 A 的最终 checkpoint-5200 评分前，不据此判断最终收益。

## 已验证测试

```bash
PYTHONPATH=src python3 tests/test_multitask_gradient_controller.py

NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
torchrun --standalone --nproc_per_node=2 tests/test_multitask_gradient_controller_ddp.py
```

单卡测试覆盖关闭态、monitor-only、warmup、EMA、baseline、Lagged、边界归一化、Hook 分解、原始梯度
还原、cosine、checkpoint、参数选择和 Hook 生命周期。双卡 smoke test 使用极小随机模型，不下载 8B
权重，并让一个 rank 缺少 recommendation，以验证不会死锁、整向量同步、DDP Hook 缩放和权重一致性。

## 实验边界

实验 A 只调整 loss 权重并监控冲突，其配置不启用梯度投影，也不改变 LoRA 前向结构。局部 PCGrad
投影作为独立的实验 B 实现和消融，不应计入实验 A 的结果。

## 成功标准

如果实验有效，应观察到：

1. 任务权重稳定变化，而不是剧烈震荡；
2. 不同任务梯度范数差异降低；
3. 总体评测提升，尤其减少任务之间互相牺牲。
