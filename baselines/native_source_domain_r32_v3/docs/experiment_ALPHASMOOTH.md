# 实验 Alpha-Smooth：第二 Epoch 标签平滑（Label Smoothing）

状态：已完成 2 epoch 训练（2026-08-14 11:36 → 22:49，耗时 11h11m）；epoch 2 外部评测结果已记录。

母版为 **Alpha-Jiankong（monitor + validation 正式版）**：四卡、LoRA r32/alpha64/dropout 0.05、8K neat packing、GA16（global batch 64）、GC0.4、LR 2e-4 cosine、warmup 0.03、SID/domain 权重 8、canonical response 权重 4；`rec_pu_enabled` / `multitask_pack_ratio_enabled` / `mini_topk_enabled` 均关闭；`alpha_monitor_enabled` 与 `alpha_validation_enabled` 开启（probe 每 100 步 + epoch 末 full-dev）。

**唯一变量**：开启标签平滑——`alpha_smooth_enabled: true`、`alpha_smooth_epsilon: 0.05`、**`alpha_smooth_start_epoch: 1.0`，即第一个 epoch 不启用，从第二个 epoch（epoch ≥ 1.0）起对训练目标应用 label smoothing（ε=0.05）**。其余配置（数据 train98、dev2/probe 验证链路、tokenized cache、loss 权重）与 Alpha-Jiankong 完全一致。

## 动机

- **现象**：训练监控发现，尤其是**懂推荐**任务在第二轮训练时出现明显的"阶梯式过拟合"——**训练集损失继续下降（尤其 NoCoT 样本），但验证集损失不降反升**，第二轮验证损失相对第一轮呈明显阶梯状，说明模型在第二轮主要是在**记忆第一轮已经见过的训练结果**，而非学到可泛化的规律；
- **对策**：为缓解"懂推荐第二轮训练记忆第一轮结果"的问题，引入标签平滑（label smoothing）作为正则：将推荐最终 A/B/C 位置的 one-hot 目标向同层其他 SID 分散 ε=0.05 的概率质量，抑制模型对训练样本的过度自信与机械记忆；
- **为何第二 epoch 才开启**：第一轮先用原生 CE 让模型充分拟合任务结构，第二轮再切换平滑目标做正则化微调，采用"先拟合、后平滑"的分阶段 schedule 对抗第二轮过拟合——这也是最初所有 smooth 实验都设 `alpha_smooth_start_epoch: 1.0` 的原因；
- 属于 Alpha 系列"只改一个变量"原则下的 loss 目标侧实验，与 Alpha-CoT（权重侧）、REC-PU / Mini-TopK（损失函数替换）互为对照；Mini 系列中的 Mini-Smooth / Mini-Whole-Smooth（见 `experiment_MINI_SMOOTH.md` / `experiment_MINI_WHOLE_SMOOTH.md`）为同机制在小型数据版本上的复刻与 start-epoch 消融。

## 实现

- 配置开关（默认关闭，legacy 运行不受影响）：`alpha_smooth_enabled` / `alpha_smooth_epsilon` / `alpha_smooth_start_epoch`；
- 生效判定：`ALPHA_SMOOTH_CONFIG.active_for_epoch(current_epoch)`，`current_epoch < start_epoch` 时不激活；
- 与 `rec_pu_enabled`、`mini_topk_enabled` 三者互斥（同时开启会报错）；
- 训练侧指标：`as_0_active`（是否激活）、`as_1_epsilon`、`as_2_sid_ls`（平滑后的 SID 损失）、`as_3_ls_delta`（平滑前后损失差）。

## 配置 Diff（相对 Alpha-Jiankong 母版）

| 字段 | Alpha-Jiankong | Alpha-Smooth |
| --- | --- | --- |
| `alpha_smooth_enabled` | （无，默认 false） | `true` |
| `alpha_smooth_epsilon` | — | `0.05` |
| `alpha_smooth_start_epoch` | — | `1.0` |
| `output_dir` | `ALPHA-JIANKONG-*` | `ALPHASMOOTH-SIDLS05-R32-2E-GC04-4GPU-*` |
| 其余全部字段 | 相同 | 相同 |

## 训练结果

- steps **1058/1058**、epoch 2.0、100% 完成；总耗时 11:11:13；checkpoint-529（epoch 1）/ checkpoint-1058（epoch 2）及最终 adapter；
- 最终 full-dev 验证（733 packs，type=full，epoch 2.0）：

| 指标 | 值 |
| --- | ---: |
| va_rec_cot_body_ce | 1.2149 |
| vb_rec_cot_gold_sid_ce | 4.8111 |
| vc_rec_nocot_gold_sid_ce | 4.8201 |
| vd_rec_gold_a_ce | 5.000 |
| ve_rec_gold_b_ce | 4.671 |
| vf_rec_gold_c_ce | 4.773 |
| vg_rec_tf_a_hit8 | 0.3516 |
| vh_rec_tf_a_hit32 | 0.5721 |
| vi_rec_tf_b_hit8 | 0.3911 |
| vj_rec_tf_c_hit8 | 0.4505 |
| vk_rec_tf_chain_32_8_8 | 0.1112 |
| as_2_sid_ls / as_3_ls_delta | 5.1321 / 0.4589 |

- 训练过程无 NaN/Inf，`rec_monitor_missing_gold = 0`、`rec_monitor_invalid_route = 0`，峰值显存约 74.7 GiB/卡。

## 外部评测（epoch 2）

外部评测器输出的总分与分项（顺序沿用固定评测器）：

```text
aggregate = 1.3185
material  = 0.0491, 0.0366, 0.0524, 0.0417
user      = 0.1520, 0.0952
recommendation = 0.1363, 0.1496, 0.2044, 0.1674
world/last = 0.2338
```

### 与 Alpha-Jiankong（baseline）对比

Alpha-Jiankong 的官方记录得分为 epoch 1 总分 **1.2605**（记录于 `experiment_alpha_cot_repeat05n.md`），其推荐分项 video/prod/ad/living = 0.1204 / 0.1394 / 0.2016 / 0.1521。

| 分项 | Alpha-Jiankong (e1) | Alpha-Smooth (e2) | 差值 |
| --- | ---: | ---: | ---: |
| 总分 | 1.2605 | **1.3185** | **+0.0580（+4.60%）** |
| 推荐 video | 0.1204 | 0.1363 | +0.0159 |
| 推荐 prod | 0.1394 | 0.1496 | +0.0102 |
| 推荐 ad | 0.2016 | 0.2044 | +0.0028 |
| 推荐 living | 0.1521 | 0.1674 | +0.0153 |

**结论：Alpha-Smooth（epoch 2，第二 epoch 开启标签平滑 ε=0.05）总分 1.3185，相比 Alpha-Jiankong baseline 有提升**；推荐四域分项全部上涨，且总分已逼近 BETA-baseline 纯净版参考线（1.3246）。

## 复刻办法

1. 配置：`/data/baselines/native_source_domain_r32_v3/config/train_alphasmooth_jiankong_monitor_validation_4gpu_gc04_2epoch.yaml`；
2. 启动：`/data/baselines/native_source_domain_r32_v3/scripts/launch_alphasmooth_jiankong_monitor_validation_4gpu_gc04_2epoch.sh`（RUN_ID 自动带时间戳；`ALPHA_FORMAL_PREFLIGHT_ONLY=1` 可只跑 preflight）；
3. 数据与验证链路与 Alpha-Jiankong 相同：`onereason_alpha_jiankong_train98`（216,732 行 / 33,810 packs）+ `onereason_alpha_jiankong_dev2`（4,520 行 / 733 packs），tokenized cache `tokenized_train98_8k_sid8w8`；
4. 输出：`/data/outputs/baselines/native_source_domain_r32_v3/ALPHASMOOTH-SIDLS05-R32-2E-GC04-4GPU-20260814-113635/`（checkpoint-529 / checkpoint-1058）；
5. 验证指标：`/data/logs/baselines/native_source_domain_r32_v3/ALPHASMOOTH-SIDLS05-R32-2E-GC04-4GPU-20260814-113635/alpha_validation_metrics.jsonl`（probe 每 100 步 + epoch 末 full-dev）；
6. preflight 合同：`preflight.log` 中 `alpha_smooth=ON`、`steps_per_epoch=529`、`total_steps=1058`、`warmup_steps=32`。

## 相关测试

- `tests/test_alpha_smooth.py`：标签平滑开关的数学回归与互斥约束。
