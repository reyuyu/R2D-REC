# 实验 B：GradNorm-lite + 局部 LoRA 梯度冲突投影

## 实现状态

实验 B 已实现并通过单卡算法测试、实验 A 回归测试和双卡 DDP 全链路 smoke test。正式 rank16 训练曾启动，但在当前双阈值门控下长期没有形成有意义的实际投影，已于 2026-08-03 主动终止；该运行不作为完整 5200-step 结果。

本项目中的 Ortho-LoRA 精确定义为：复用实验 A 正常 backward Hook 捕获的任务梯度，只对最后 N 层指定模块的 LoRA-B 梯度做 PCGrad 风格的局部冲突投影。它不新增 adapter，不改变 LoRA 矩阵、模型 forward 或推理结构，也不增加模型 forward。

## 配置

完整配置：

```text
configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_ortho.yaml
```

实验 B 与实验 A 保持相同的数据、seed、rank16、balanced_40、max_steps 和 GradNorm 参数，只增加以下字段并使用不同的 `output_dir`：

```yaml
multitask_ortho_enabled: true
multitask_ortho_monitor_only: false
multitask_ortho_start_step: 600
multitask_ortho_interval: 2
multitask_ortho_current_cosine_threshold: -0.05
multitask_ortho_ema_cosine_threshold: 0.0
multitask_ortho_cosine_ema_beta: 0.90
multitask_ortho_use_ema_gate: true
multitask_ortho_norm_ratio_min: 0.7
multitask_ortho_norm_ratio_max: 1.3
multitask_ortho_rotate_order: true
```

消融模式：

- GradNorm-only：`multitask_ortho_enabled: false`，行为与实验 A 一致。
- Ortho monitor-only：Ortho 开启且 `multitask_ortho_monitor_only: true`，计算投影指标但不写回 `.grad`。
- GradNorm + Ortho：GradNorm 和 Ortho 均开启，执行实际投影。
- Ortho-only：关闭 `multitask_gradnorm_enabled`，三个任务权重固定为 1，仍执行冲突监控和投影。

## 代码路径

- 参数与校验：`src/llamafactory/hparams/data_args.py`
- Hook、flat buffer、DDP 同步、GradNorm 和 Ortho：`src/llamafactory/train/sft/multitask_gradient_controller.py`
- macro-step 接入：`src/llamafactory/train/sft/trainer.py`

Trainer 原有顺序保持为：全部 microbatch backward，控制器同步任务向量并投影写回，原有 gradient clipping，`optimizer.step()`，`scheduler.step()`。投影没有重写 Trainer 或优化器。

## 统一采集与梯度分解

实验 B 不注册第二套 Hook，也不建立第二套任务采集 buffer。统一条件为：

```text
capture_due = gradnorm_measure_due OR monitor_due OR ortho_step_due
ortho_step_due = step >= start_step AND step % interval == 0
```

每个任务仍只同步一个连续 FP32 flat vector。同步后保留两种语义：

- `raw_average_gradients`：还原后的任务单 microbatch 平均原始梯度，用于 norm、cosine 和 GradNorm。
- `actual_weighted_contributions`：与最终 DDP `.grad` 同缩放语义的实际加权总贡献，用于投影和重建。

对选中参数：

```text
tracked_sum = material + user + recommendation
residual = original_total_grad - tracked_sum
new_total_grad = residual + projected_tracked_sum
```

`world` 未被 Hook 归入三个主任务，因此自动保留在 residual。未参与控制的 loss 和数值残差也原样保留。只有选中的 LoRA-B 参数使用 `param.grad.copy_()` 写回，其他 LoRA 参数和模型参数不变。

## 冲突门控与投影

启用 EMA gate 时，一对任务必须同时满足：

```text
current_cosine < 0
current_cosine < multitask_ortho_current_cosine_threshold
and cosine_ema < multitask_ortho_ema_cosine_threshold
```

关闭 EMA gate 时使用 `current_cosine < threshold`。任务缺失、零范数或非有限值时，该 pair 不触发。

确定性轮换顺序为：

```text
step % 3 == 0: material -> user -> recommendation
step % 3 == 1: user -> recommendation -> material
step % 3 == 2: recommendation -> material -> user
```

每个待投影向量从该任务的实际加权贡献复制，参考向量始终使用本 macro-step 同步后的原始任务梯度快照。一个任务可以连续消除多个冲突分量，但参考向量不随循环变化。

投影后只对三个任务的合并梯度做整体尺度保护，使
`||projected_tracked|| / ||original_tracked||` 落入 `[0.7, 1.3]`；residual 不参与缩放。

## 指标

实验 A 的权重、loss EMA、norm 和 cosine 指标全部保留。实验 B 新指标采用统一层级名称，避免继续扩展按字母排序的前缀：

- `mtg/ortho/active`、`mtg/ortho/monitor_only`
- `mtg/ortho/pair_count`
- `mtg/ortho/pair/material_user`
- `mtg/ortho/pair/material_recommendation`
- `mtg/ortho/pair/user_recommendation`
- `mtg/ortho/removed_ratio`
- `mtg/ortho/tracked_norm_ratio`
- `mtg/ortho/total_norm_ratio`
- `mtg/ortho/time/capture_ms`
- `mtg/ortho/time/sync_ms`
- `mtg/ortho/time/project_ms`
- `mtg/ortho/time/writeback_ms`

inactive step 的 `active=0`，Ortho 专属比例回到无变化值，专属时间为 0。monitor-only 中 writeback 时间固定为 0。

## Checkpoint

继续使用同一个：

```text
multitask_gradient_controller.json
```

实验 A 状态之外还保存 cosine EMA、Ortho 激活状态、执行次数、累计 pair 次数和最近的 removed/tracked/total norm ratio。rank 0 写入，恢复时广播到全部 rank。调度仍由恢复后的 Trainer `global_step` 决定，因此 interval 和轮换顺序连续。

## 启动与测试

完整实验 B 的启动命令为：

```bash
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 \
llamafactory-cli train configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_ortho.yaml
```

本次验收未运行该 5200-step 命令。已运行：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python3 tests/test_multitask_gradient_controller.py
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python3 tests/test_multitask_ortho.py

NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src \
torchrun --standalone --nproc_per_node=2 tests/test_multitask_ortho_ddp.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
python3 tests/benchmark_multitask_ortho.py --steps 20 --warmup 4
```

双卡 smoke 覆盖 `DDP capture -> all_reduce -> cosine -> projection -> writeback -> optimizer.step`，并让一个 rank 缺少 recommendation、另有 world residual。

## 实测开销

在服务器 GPU 2 上以实际参考元素数 851,968 做控制器路径微基准，20 个计时 step 的结果为：

- GradNorm-only 平均 macro 路径：0.801 ms
- GradNorm + Ortho 平均 macro 路径：1.610 ms
- Ortho active step：2.434 ms
- Ortho inactive step：0.785 ms
- active step 内投影计算：0.817 ms；梯度写回：0.047 ms
- Ortho 额外常驻控制器显存：24,379,392 bytes，约 23.25 MiB
- 每次 capture 的三个任务向量通信载荷：10,223,616 bytes，约 9.75 MiB/rank

该微基准包含 Hook、flat buffer、统计、投影和写回，但不包含 8B 模型 forward、数据加载和 optimizer，因此不能据此声称完整训练的固定百分比开销。完整端到端开销需在正式实验 B 运行后从真实 macro-step 日志比较。

## 已知限制

- 投影只覆盖最后四层的 `q_proj/v_proj/o_proj/down_proj` LoRA-B，不代表全模型梯度无冲突。
- PCGrad 顺序是确定性轮换，不搜索全局最优联合方向。
- BF16/FP32 写回和极小梯度可能存在正常浮点误差。
- 尚未执行 8B、5200-step 实验 B，也未获得端到端吞吐与最终评测结果。
