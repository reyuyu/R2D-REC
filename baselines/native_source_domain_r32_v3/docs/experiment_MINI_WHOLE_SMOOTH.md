# 实验 Mini-Whole-Smooth：全程（两个 Epoch）标签平滑

状态：**已完成**。2 epoch 训练于 2026-08-15 17:06 启动（RUN_ID `MINI-WHOLE-SMOOTH-R32-2E-GC04-4GPU-20260815-170650`），18:41 完成，共 140 步，耗时 1h35m；外部评测待补充。

母版为 **Mini-Smooth**（见 `experiment_MINI_SMOOTH.md`），即 Mini 大基线（alpha_mini_v1，49,490 行）+ 推荐路由标签平滑 ε=0.05。**唯一差异**：`alpha_smooth_start_epoch: 0.0`（从 epoch 0 起平滑，两个 epoch 全程生效），替换 Mini-Smooth 的 `1.0`。

## 动机

- **背景**：smooth 正则的初衷来自正式 Alpha-Smooth 的发现——尤其是**懂推荐**任务在第二轮训练时出现"阶梯式过拟合"：**训练集损失继续下降（尤其 NoCoT 样本），但验证集损失不降反升**，第二轮验证损失相对第一轮呈明显阶梯状，模型在第二轮主要是在**记忆第一轮已经见过的训练结果**。为此引入标签平滑（ε=0.05，作用于推荐最终 A/B/C 位置）作为正则，并在**所有既有 smooth 实验（正式 Alpha-Smooth、Mini-Smooth）中设为 `alpha_smooth_start_epoch: 1.0`，即第二个 epoch 才开启**；
- **本实验要回答的问题**：平滑应该从哪个 epoch 开始？Mini-Whole-Smooth 将 `start_epoch` 改为 `0.0`，验证"从第一轮起就施加正则"（经典全程 label smoothing 路线）是否优于"第二轮才开启"（先拟合、后平滑）——如果全程正则能更早压制过拟合趋势，第二轮阶梯现象可能更轻，最终分数可能更好；反之若第一轮强拟合被平滑拖慢，分数可能更差；
- **与 Mini-Smooth 构成单变量消融**：同一基线上唯一差异是 `alpha_smooth_start_epoch`（1.0 vs 0.0），结果能直接对比两种 schedule。

## 风险与预期观测

- **风险**：epoch 1 起平滑会削弱对 SID 结构的强拟合，可能拖慢早期收敛、压低 epoch 1 分数；最终 epoch 2 分数可能更好（更抗过拟合）也可能略差（欠拟合）；
- 影响面有限：平滑只作用于推荐最终 A/B/C 三个位置，plain token、物料、用户路由不受影响；
- **关键观测点**：`as_0_active` 从 step 1 起就应为 `'1'`（与 Mini-Smooth 的 epoch 1 为 `'0'` 形成直接对照）；`as_2_sid_ls` / `as_3_ls_delta` 全程非零；同时对比两实验第二轮的验证损失阶梯是否被压低。

## 门控验证（已通过）

`smooth: True 0.05 0.0`；`AlphaSmoothConfig.active_for_epoch` 在 epoch 0.0 / 0.5 / 1.0 / 2.0 全部为 True → step 1 即激活，140 步全程生效。

## 配置 Diff（相对 Mini-Smooth）

| 字段 | Mini-Smooth | Mini-Whole-Smooth |
| --- | --- | --- |
| `alpha_smooth_start_epoch` | `1.0` | `0.0` |
| `output_dir` | `MINI-SMOOTH-R32-2E-GC04-4GPU-*` | `MINI-WHOLE-SMOOTH-R32-2E-GC04-4GPU-*` |
| `alpha_validation_metrics_path` | MINI-SMOOTH 日志目录 | MINI-WHOLE-SMOOTH 日志目录 |
| 其余全部字段 | 相同 | 相同 |

## 训练结果

- steps **140/140**、epoch 2.0、100% 完成；总耗时 1h35m（17:06 → 18:41）；
- 关键观测点验证：`as_0_active` 从 step 1 起即为 `'1'`（与 Mini-Smooth 的 epoch 1 为 `'0'` 形成直接对照），`as_2_sid_ls` / `as_3_ls_delta` 全程非零——平滑全程生效，符合设计；
- `rec_monitor_missing_gold = 0`、`rec_monitor_invalid_route = 0`，无 NaN/Inf。

**epoch 末 full-dev 验证**（`onereason_alpha_mini_v1_dev_filtered`，与 Mini-Smooth 同口径）：

| 指标 | Mini-Smooth e2 | Mini-Whole-Smooth e1 | Mini-Whole-Smooth e2 |
| --- | ---: | ---: | ---: |
| va_rec_cot_body_ce | 1.4135 | 1.4649 | **1.4127** |
| vb_rec_cot_gold_sid_ce | 4.7878 | 4.7899 | 4.8363 |
| vc_rec_nocot_gold_sid_ce | 4.7534 | 4.7536 | 4.7964 |
| vh_rec_tf_a_hit32 | 0.5542 | 0.5515 | **0.5583** |
| vk_rec_tf_chain_32_8_8 | 0.1260 | 0.1247 | **0.1301** |

全程平滑（start_epoch 0.0）的 epoch 2 验证指标与"第二轮才开启"（start_epoch 1.0）基本持平（va 略优、vb/vc 略差、hit32 与 chain 略优），外部总分待补——两实验的最终对比以外部评测为准。

## 复现方法

1. 配置：`/data/baselines/native_source_domain_r32_v3/config/train_mini_whole_smooth_4gpu_gc04_2epoch.yaml`
2. 启动：`/data/baselines/native_source_domain_r32_v3/scripts/launch_mini_whole_smooth_4gpu_gc04_2epoch.sh`
3. 输出：`/data/outputs/baselines/native_source_domain_r32_v3/MINI-WHOLE-SMOOTH-R32-2E-GC04-4GPU-*`
4. 验证指标：日志目录下 `alpha_validation_metrics.jsonl`
5. 外部评估：epoch 2 完成后按固定顺序（物料4 / 用户2 / 推荐4 / 世界1）。

## 相关代码

- `rec_pu/sid8_rec_pu_integration.py`：`AlphaSmoothConfig`（start_epoch 参数）；
- `tests/test_alpha_smooth.py`：平滑机制回归（8/8 通过）。
