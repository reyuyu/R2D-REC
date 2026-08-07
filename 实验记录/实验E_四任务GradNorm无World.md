# 实验 E：四任务 GradNorm（无 World）

## 状态

- 状态：已实现，代码与配置已完成单元测试、双卡 smoke 与配置校验，尚未启动完整训练。
- 配置：[onereason_lora_2gpu_balanced40_r16_gradnorm_expE_user_split_no_world.yaml](../configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_expE_user_split_no_world.yaml)
- 输出目录：`/data/outputs/onereason_lora_2gpu_balanced40_r16_gradnorm_expE_user_split_no_world_lr2e4_len8k`

## 动机

实验 A 将 `user` 作为一个整体参与三任务 GradNorm，内部 Action Select 与 Chain 的梯度差异被合并；同时 `world` 参与训练但不参与 GradNorm。实验 E 的目标是：

1. 把 `user` 拆成 `user_action` 与 `user_chain` 两个独立 GradNorm 任务，让 Action Select 的梯度尺度与 Chain 分离；
2. 删除懂世界（world）数据集，避免其对四个主任务梯度造成干扰；
3. 使用固定 `{2,2,2,2}` 调度，让每个 macro-step 四个任务全部出现、GradNorm 全程可见所有任务；
4. 保留实验 A 的训练参数与 GradNorm-lite 语义，新增基于最新 C 系列的四项 Action 诊断监控，用于观察拆分后 Action 侧行为，但不改变训练损失。

## 布局与数据

新增消融项：

```yaml
multitask_task_layout: user_split_no_world
```

该布局的任务与子任务映射：

| 任务 | 子任务 | 数据集 |
| --- | --- | --- |
| material | cot / nocot | onereason_material_cot / onereason_material_nocot |
| user_action | action_nocot | onereason_user_action_nocot |
| user_chain | cot / nocot | onereason_user_chain_cot / onereason_user_chain_nocot |
| recommendation | cot | onereason_recommendation_cot |

训练数据集只包含以上 6 个 `_train98` jsonl，world 的两个数据集（`onereason_world_cot_train98`、`onereason_world_nocot_train98`）不再注册。

默认 `legacy` 布局保持不变：`material/user/recommendation/world` 四个顶层任务、`TASK_IDS` 与 `SUBTASK_RATIOS` 语义与实验 A 完全一致，因此该消融可随时回退。

## 调度：方案 A

`user_split_no_world` 布局下 `Balanced40SuperCycle` 每个 macro-step 固定返回：

```python
{"material": 2, "user_action": 2, "user_chain": 2, "recommendation": 2}
```

40 步周期内四个任务各出现 80 个 pack（`80/80/80/80`），且每个 macro-step 都覆盖全部四个任务，GradNorm 不会出现某一步缺少任务的情况。

## GradNorm 配置

保持实验 A 的 Lagged GradNorm-lite 语义：

```yaml
multitask_gradnorm_tasks:
  - material
  - user_action
  - user_chain
  - recommendation
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

与实验 A 的唯一区别是任务数由 3 变为 4：权重归一化从 `sum(weights)=3` 变为 `sum(weights)=4`，任务轮换与两两冲突检测覆盖全部四任务。

## 训练参数

训练参数与实验 A 保持一致：

```yaml
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0
max_steps: 5200
bf16: true
seed: 42
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_checkpointing_layer_ratio: 0.75
```

0.75 层梯度检查点沿用 C1/C3 验证过的显存/速度折衷。Action 辅助损失与 SID 加权 CE 均关闭，实验 E 是干净的“四任务 GradNorm + 诊断监控”对比组。

## 监控指标

### 1. 四任务 GradNorm 指标

沿用实验 A 的命名，`<task>` 为 `material`、`user_action`、`user_chain`、`recommendation`：

- `a_gn_w_<task>`：本 macro-step 实际使用的任务权重；
- `b_gn_loss_ema_<task>` / `b_gn_loss_ratio_<task>`：未加权原始 loss EMA 与相对 baseline 比值；
- `c_gn_raw_norm_<task>`：任务单 microbatch 平均原始梯度 norm；
- `c_gn_effective_norm_<task>`：当前权重乘以 raw norm；
- `e_gn_measure_active` / `e_gn_weight_update_active` / `e_gn_max_weight_delta`：测量/更新状态；
- `f_gn_capture_ms` / `f_gn_sync_ms` / `f_gn_update_ms`：新增功能耗时。

### 2. 冲突检测

四任务两两 cosine（共 6 对）：

- `d_gn_cos_material_user_action`
- `d_gn_cos_material_user_chain`
- `d_gn_cos_material_recommendation`
- `d_gn_cos_user_action_user_chain`
- `d_gn_cos_user_action_recommendation`
- `d_gn_cos_user_chain_recommendation`

### 3. Action Select 诊断（zero-weight，不改变损失）

复用最新 C 系列的四项核心 Action 指标，通过 `user_action_aux_enabled: true` + 全部权重/上限为 0 的纯诊断模式计算：

- `a_act_legal_sid_mass`：SID 位置模型分给历史合法完整路径候选的概率质量；
- `a_act_topk_illegal_mass`：Top-5 中非法 token 的集中概率质量；
- `a_act_top1_illegal_hit_rate`：Top-1 命中非法 token 的比例；
- `a_act_gold_top5_rate`：gold SID token 进入 Top-5 的比例。

这些指标只统计，不产生任何辅助 loss，因此不会改变训练语义；用于判断拆分 `user_action` 后 Action Select 的幻觉/复读/选择质量是否改善。

### 4. 任务 raw loss 与总 loss

- `loss_raw_material` / `loss_raw_user_action` / `loss_raw_user_chain` / `loss_raw_recommendation`：各任务平均原始（未加权）loss；
- `loss_gradnorm_total`：四任务按 GradNorm 权重加权后的宏步总 loss（与优化器实际使用的标量一致）；
- 常规 `loss` / `grad_norm` / `learning_rate` / `epoch` 照常输出。

## 代码修改

| 文件 | 修改 |
| --- | --- |
| `src/llamafactory/data/multitask.py` | 新增 `TASK_LAYOUT_USER_SPLIT_NO_WORLD`、按布局的任务/子任务/比例/ID 映射、`resolve_task_layout` / `get_task_ids` / `get_task_datasets` / `get_subtask_ratios` / `get_action_task_name`；`TokenizedSubDataset`、`Balanced40SuperCycle`、`MultiTaskMacroStepLoader`、`build_multitask_datasets` 支持布局 |
| `src/llamafactory/hparams/data_args.py` | 新增 `multitask_task_layout`，按布局校验 allocation 与 GradNorm 任务集合 |
| `src/llamafactory/train/sft/trainer.py` | 使用布局映射替代硬编码 `TASK_IDS` / `SUBTASK_RATIOS`；Action 分支按 `action_task_name` 动态识别；监控记录按 `multitask_task_ids` 写入 |
| `src/llamafactory/train/sft/multitask_gradient_controller.py` | 轮换与冲突检测泛化到任意任务集合，支持四任务 |
| `tests/test_multitask_macro.py` | 新增布局映射、2222 调度、双 rank loader、data_args 校验、拒绝 world 等测试 |
| `tests/test_multitask_gradient_controller.py` | 新增四任务 GradNorm 归一化与四任务轮换测试 |

## 测试结果

```bash
cd /app/LLaMA-Factory
PYTHONPATH=src python3 tests/test_multitask_macro.py                 # 全部 PASS
PYTHONPATH=src python3 tests/test_multitask_gradient_controller.py  # 15 项 PASS
PYTHONPATH=src python3 tests/test_sid_token_weighting.py            # 13 项 PASS

# 双卡 smoke（CPU/GLOO，不占 GPU）
GLOO_SOCKET_IFNAME=lo PYTHONPATH=src torchrun --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29555 tests/smoke_expE_4task_ddp.py
# 输出：SMOKE PASS weights=[1.0,1.0,1.0,1.0] global_totals={material:8, user_action:8, user_chain:8, recommendation:8}
```

配置校验（不启动训练）：

```bash
GLOO_SOCKET_IFNAME=lo PYTHONPATH=src torchrun --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port=29556 tests/validate_expE_config.py \
  configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_expE_user_split_no_world.yaml
# 输出：CONFIG OK，layout=user_split_no_world，allocation={2,2,2,2}，gradnorm_tasks=四任务
```

## 启动命令

```bash
cd /app/LLaMA-Factory
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 \
llamafactory-cli train configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_expE_user_split_no_world.yaml
```

## 与实验 A 的区别

| 项 | 实验 A | 实验 E |
| --- | --- | --- |
| 顶层任务 | material / user / recommendation / world | material / user_action / user_chain / recommendation |
| GradNorm 任务 | 3 个 | 4 个 |
| world | 参与训练，固定权重 1 | 完全删除 |
| 调度 | balanced_40 周期内动态分配 | 每步固定 {2,2,2,2} |
| Action 监控 | 无 | 四项 zero-weight 诊断指标 |
| 训练参数 | lr 2e-4 / dropout 0.05 / GC 默认 | 同 lr/dropout，GC 0.75 |

## 未验证事项

- 已在 GPU 0/1 启动 rank16 双卡正式训练；C3 同时占用 GPU 2/3。
- 真实 8B 双卡训练已运行；最终吞吐、显存与完整训练结果在结束后统一归档。
- 四任务 GradNorm 权重归一化目标 `sum(weights)=4` 已在单元测试与双卡 smoke 验证，真实训练的权重稳定性待观察。

## 正式 checkpoint 评测

实验 E 已记录 1000 和 5200 checkpoint 的评测结果；评分仍按物料四域、用户两项、推荐四域、world 的固定顺序输出。world 已从训练任务移除，但仍保留在统一外部评测中。

| Checkpoint | 总分 | 物料 4 域 | 用户 2 项 | 推荐 4 域 | World |
| ---: | ---: | --- | --- | --- | ---: |
| 1000 | 1.1999 | 0.0423, 0.0379, 0.0379, 0.0412 | 0.1336, 0.0959 | 0.0961, 0.1598, 0.1834, 0.1314 | 0.2405 |
| 5200 | 1.2123 | 0.0469, 0.0383, 0.0459, 0.0424 | 0.1499, 0.0996 | 0.0812, 0.1700, 0.1652, 0.1341 | 0.2387 |

已补充 5200 checkpoint；相同评测口径下，5200 总分为 1.2123，高于 1000 checkpoint 的 1.1999。
