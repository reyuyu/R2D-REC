# 实验 Mini-Smooth：Mini 基线 + 第二 Epoch 标签平滑（Label Smoothing）

状态：**训练中**。2 epoch 训练于 2026-08-15 15:25 启动（RUN_ID `MINI-SMOOTH-R32-2E-GC04-4GPU-20260815-152559`），共 140 步，~38s/步，预计 ~16:55 完成。

母版为 **alpha_mini_v1（Mini 大基线）**：OneReason-8B、LoRA r32/a64/dropout 0.05、8K neat packing、GA16（global batch 64）、FA2 + Liger、bf16、GC0.4、LR 2e-4 cosine、warmup 0.03、seed 20260806、SID/domain 权重 8（GLOBAL_ITEM_WEIGHT=8）；`rec_pu_enabled` / `multitask_pack_ratio_enabled` / `rec_candidate_metrics_enabled` 关闭；`alpha_monitor_enabled` / `alpha_validation_enabled` 打开，TF 每 10 步、probe 每 20 步 + epoch 末 full-dev。数据集 `onereason_alpha_mini_v1`（49,490 行）。

**唯一差异**：推荐路由标签平滑（label smoothing），`alpha_smooth_enabled: true`、`alpha_smooth_epsilon: 0.05`、**`alpha_smooth_start_epoch: 1.0`（第一 epoch 用原生 CE，第二 epoch 起训练目标切换为平滑标签）**。其余全部字段（验证路线 dev/probe、tokenized cache、loss 权重）与 Mini 基线完全一致。

## 动机

- **现象（与正式 Alpha-Smooth 相同）**：训练监控发现，尤其是**懂推荐**任务在第二轮训练时出现明显的"阶梯式过拟合"——**训练集损失继续下降（尤其 NoCoT 样本），但验证集损失不降反升**，第二轮验证损失相对第一轮呈明显阶梯状，说明模型在第二轮主要是在**记忆第一轮已经见过的训练结果**，而非学到可泛化的规律；
- **对策**：为缓解"懂推荐第二轮训练记忆第一轮结果"的问题，引入标签平滑（label smoothing）作为正则：将推荐最终 A/B/C 位置的 one-hot 目标向同层其他 SID 分散 ε=0.05 的概率质量，抑制模型对训练样本的过度自信与机械记忆；
- **为何第二 epoch 才开启**：第一轮先用原生 CE 让模型充分拟合任务结构，第二轮再切换平滑目标做正则化微调，采用"先拟合、后平滑"的分阶段 schedule 对抗第二轮过拟合——这也是最初所有 smooth 实验（正式 Alpha-Smooth、本实验）都设 `alpha_smooth_start_epoch: 1.0` 的原因；
- **Mini 版定位**：把正式 Alpha-Smooth 的收益（epoch 2 总分 1.3185，较母版 +4.60%）搬到 Mini 系列，在小型数据版本（49,490 行）上验证同一机制；并与 Mini-Whole-Smooth（全程平滑）构成 start-epoch 消融对。

## 机制

- 复用 `AlphaSmoothConfig`：`active_for_epoch(epoch) = enabled and epoch >= start_epoch`；
- 只作用于推荐最终 A/B/C 位置的 CE：`smoothed = base_CE + ε·(gold_logits − mean_other_same_level)`；plain token、物料、用户路由不受影响；
- 训练侧指标：`as_0_active`（是否激活）、`as_1_epsilon`、`as_2_sid_ls`（平滑后 SID 损失）、`as_3_ls_delta`（平滑前后损失差）。

## 配置 Diff（相对 Mini 基线 alpha_mini_v1）

| 字段 | Mini 基线 | Mini-Smooth |
| --- | --- | --- |
| `alpha_smooth_enabled` | 无（默认 false） | `true` |
| `alpha_smooth_epsilon` | — | `0.05` |
| `alpha_smooth_start_epoch` | — | `1.0` |
| `output_dir` | `MINI-*` | `MINI-SMOOTH-R32-2E-GC04-4GPU-*` |
| 其余全部字段 | 相同 | 相同 |

## 训练过程（进行中）

- steps 0/140 → 运行中；当前 epoch ~0.14，`as_0_active: '0'`（epoch 1 平滑未激活，符合预期）；
- **关键观测点**：step ~70（epoch ≥ 1.0）时 `as_0_active` 应从 0 翻转为 1，`as_2_sid_ls` / `as_3_ls_delta` 开始输出非零值；预计 ~16:10；
- `rec_monitor_missing_gold = 0`、`rec_monitor_invalid_route = 0`（截至最新日志）。

## 配套消融

**Mini-Whole-Smooth**（见 `experiment_MINI_WHOLE_SMOOTH.md`）：同一基线上唯一差异为 `alpha_smooth_start_epoch: 0.0`（两个 epoch 全部平滑）。两者合起来回答"平滑应该从哪个 epoch 开始"。

## 复现方法

1. 配置：`/data/baselines/native_source_domain_r32_v3/config/train_mini_smooth_4gpu_gc04_2epoch.yaml`
2. 启动：`/data/baselines/native_source_domain_r32_v3/scripts/launch_mini_smooth_4gpu_gc04_2epoch.sh`（RUN_ID 自动带时间戳；`ALPHA_FORMAL_PREFLIGHT_ONLY=1` 只做 preflight）
3. 输出：`/data/outputs/baselines/native_source_domain_r32_v3/MINI-SMOOTH-R32-2E-GC04-4GPU-20260815-152559/`
4. 验证指标：`/data/logs/baselines/native_source_domain_r32_v3/MINI-SMOOTH-R32-2E-GC04-4GPU-20260815-152559/alpha_validation_metrics.jsonl`
5. 外部评估：epoch 2 完成后按固定顺序（物料4 / 用户2 / 推荐4 / 世界1）跑评估器。

## 相关代码

- `rec_pu/sid8_rec_pu_integration.py`：`AlphaSmoothConfig` + 平滑逻辑；
- `tests/test_alpha_smooth.py`：平滑机制回归与互斥约束（8/8 通过）。
