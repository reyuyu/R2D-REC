# 实验 C1：Trie 优先的 Action Select 辅助损失

## 动机

实验 C-fast 在 `step 510–1000` 的平均 Action 指标为：

```text
0.06 × trie_loss = 0.00124
continue_loss    = 0.23123
stop_loss        = 0.19738
action_ce        = 0.27660
```

Trie 在原始辅助损失中只占约 `0.3%`，Continue/Stop 占约 `99.7%`。联合辅助损失在全部日志点触发
`8% × Action CE` 上限，导致长度目标决定统一 cap scale，并进一步缩小历史合法性与动态完整 SID 去重梯度。

C1 不增加新 forward、backward、参数或 collective。它保留 C-fast 的向量化动态 unused-history Trie，只重新分配辅助预算。

## 方法

```text
trie_raw   = trie_weight × trie_loss
length_raw = continue_loss + stop_loss

trie_aux   = cap(trie_raw,   trie_cap_ratio × detach(action_ce))
length_aux = cap(length_raw, length_cap_ratio × detach(action_ce))

action_aux = warmup × cap(
    trie_aux + length_aux,
    total_cap_ratio × detach(action_ce)
)
```

所有 cap scale 均由 detach 后的值计算，因此不会对 Action CE 建立额外反向耦合。分项 cap 后仍保留总 `8%` 安全上限。

预算设置：

```yaml
user_action_aux_cap_ratio: 0.08
user_action_aux_split_cap_enabled: true
user_action_trie_cap_ratio: 0.06
user_action_length_cap_ratio: 0.02
```

Trie 获得最多 `6% × Action CE`，长度控制最多获得 `2% × Action CE`。两者上限和不超过总上限，因此总 cap 正常情况下不应再频繁触发。

## C-fast 实测校准

C1 使用：

```yaml
user_action_history_trie_weight: 0.60
user_action_continue_domain_extra: 0.01
user_action_continue_separator_extra: 0.005
user_action_no_early_stop_weight: 0.002
user_action_stop_domain_weight: 0.005
user_action_stop_tail_extra: 0.01
```

Trie 权重相对 C-fast 提高 10 倍。按 C-fast `step 510–1000` 的均值估算，Trie 原始贡献约为
`0.60 × 0.02072 = 0.01243`，即 Action CE 的约 `4.5%`，处于独立 `6%` 预算内。长度额外 CE 系数降低到原来的约 `1%–10%`，目标是使其接近而不是长期撞击独立 `2%` 上限。

该校准只用于构造可检验的消融假设，不把训练日志中的 loss 数值比例等同于梯度范数比例。

## 指标

保留实验 C 的所有指标，并增加：

```text
d_act_effective_trie_loss
d_act_effective_length_loss
d_act_trie_cap_active
d_act_length_cap_active
d_act_total_cap_scale
```

`effective_*` 是经过分项 cap、总 cap 和 warmup 后实际加入 Action raw loss 的分量。`d_act_cap_active` 继续表示最终总安全 cap，而不是分项 cap。

## 配置与启动

独立配置：

```text
configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_c1_trie_priority.yaml
```

启动命令预留为：

```bash
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=2,3 FORCE_TORCHRUN=1 \
llamafactory-cli train \
  configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_c1_trie_priority.yaml
```

本次只实现和测试消融选项，不启动正式训练。

## 已验证测试

```bash
PYTHONPATH=src:tests pytest -q tests/test_user_action_auxiliary.py
PYTHONPATH=src:tests pytest -q tests/test_multitask_macro.py
PYTHONPATH=src:tests pytest -q tests/test_multitask_gradient_controller.py
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:tests \
  python3 tests/test_user_action_auxiliary_qwen3.py
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src:tests \
  torchrun --standalone --nproc_per_node=2 tests/test_user_action_auxiliary_ddp.py
```

结果：Action 辅助损失 `50 passed`，macro/packing `9 passed`，GradNorm 控制器
`13 passed`；随机初始化 Qwen3 + LoRA 单卡 smoke 与双卡 DDP
forward/backward/all-reduce/optimizer smoke 均通过。C1 YAML 也已通过当前参数 dataclass 的字段解析与校验。

这些测试只证明实现和训练链路成立。是否降低自由生成时的历史外 SID 与完整 SID
重复率，仍需后续短跑和 checkpoint 评测确认。

## Gradient checkpointing 双卡短跑

在 GPU 2/3 上以真实 8B 模型、`cutoff_len=8192`、每卡 batch 1 和相同 C1
配置执行 20-step A/B，关闭 checkpoint 保存和外部指标上报：

| 设置 | 完成步数 | Trainer runtime | 平均步耗时 | 训练显存/结果 |
| --- | ---: | ---: | ---: | --- |
| gradient checkpointing 开启 | 20 | 267.82 秒 | 13.39 秒 | 成功完成 |
| gradient checkpointing 关闭 | 0 | - | - | 首个 macro-step OOM；两卡进程分别达到约 79.25/79.31 GiB |

GC-off 在首次 backward 尚需申请 64/128 MiB 时显存已经基本耗尽，因此
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 不能提供足够的可靠余量。由于没有完成
任何 optimizer step，不能从本测试声称 GC-off 具有实际吞吐提升。

C1 正式配置显式保留：

```yaml
disable_gradient_checkpointing: false
```

若将来要测试 GC-off，需要降低 `cutoff_len`、缩短实际序列或使用更强的显存优化；这些都会
改变当前实验的序列/训练语义，不作为 C1 的等价加速方案。

## 选择性 Gradient Checkpointing 加速

Transformer 5.6 将 checkpoint 粒度放在每个 Qwen3 decoder layer。C1 新增模型参数：

```yaml
gradient_checkpointing_layer_ratio: 0.75
```

默认值为 `1.0`，即保持原有所有层 checkpoint。小于 `1.0` 时仅最后
`round(layer_ratio * num_layers)` 个 decoder layer 保持 checkpoint，前面的层保存激活、
避免 backward 重算。该机制不改变模型前向、loss、任务调度、GradNorm、Ortho 或 optimizer。

随机初始化 4 层 Qwen3 的等价测试已验证：全层 checkpoint 与半层 checkpoint 的 loss 和每个
参数梯度在浮点误差内一致。

真实 8B、双卡、`cutoff_len=8192`、相同 C1 配置的 20-step 结果：

| checkpoint 设置 | 完成步数 | Trainer runtime | 平均 macro-step | 结果 |
| --- | ---: | ---: | ---: | --- |
| 全部 36 层、重入 | 20 | 267.82 秒 | 13.39 秒 | 基线通过 |
| 全部 36 层、非重入 | 20 | 267.31 秒 | 13.37 秒 | 仅快 0.19%，不采用 |
| 后 18/36 层（0.50） | 1 | - | - | 第二步 OOM，尚需 5.32 GiB |
| 后 27/36 层（0.75） | 20 | 251.07 秒 | 12.55 秒 | 通过，较基线快 6.25% |

0.75 方案在训练中的 GPU 采样显存约为 `66.4/70.9 GiB`，低于每卡 79.33 GiB 容量；这些是
采样值而非 CUDA allocator 的严格峰值，因此保留约 8 GiB 的观测余量，而不继续提高非 checkpoint
层数。C1 正式 YAML 已采用该值。

## 边界

- 历史外 SID 与完整 SID 去重仍共享动态 Trie；C1 调整其有效预算，不新增额外模型路径。
- teacher forcing 下的 allowed mass 不能替代自由生成的幻觉率与重复率。
- C1 不实现 Illegal Top-K、额外 unlikelihood forward 或约束解码。
- 原实验 C YAML、checkpoint、输出目录和正在运行的进程不修改。
\n### C1 分项预算监控\n\nC1 在 C/C-fast 原有 Action 数据、Trie 合法质量、去重、Continue/Stop 和总辅助 loss 指标之外，新增：\n\n- `d_act_effective_trie_loss`：Trie 原始项经 6% 分项 cap 后的实际贡献。\n- `d_act_effective_length_loss`：Continue/Stop 项经 2% 分项 cap 后的实际贡献。\n- `d_act_trie_cap_active`、`d_act_length_cap_active`：各分项当步是否被上限截断。\n- `d_act_total_cap_scale`：8% 总上限仍需生效时的缩放系数；稳定 C1 中通常应接近 1。\n
## 正式 checkpoint 评测与结论

C1 的 rank16 双卡正式训练已完成。以下为 V1 评测结果；物料四域、用户两项、推荐四域和 world 按固定顺序记录。

| Checkpoint | 评测任务 | 完成时间 | 评测耗时 | 总分 | 物料 4 域（和） | 用户 2 项（和） | 推荐 4 域（和） | World |
| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |
| 1000 | C1-1000_V1_eval_20260804214643 | 2026-08-04 21:46:49 | 5h 00m 00s | 1.1824 | 0.0426, 0.0381, 0.0418, 0.0425 (0.1650) | 0.1339, 0.0872 (0.2211) | 0.0971, 0.1462, 0.1778, 0.1467 (0.5678) | 0.2286 |
| 2500 | C1-2500_V1_eval_20260805032121 | 2026-08-05 03:21:24 | 4h 50m 44s | 1.1346 | 0.0462, 0.0379, 0.0453, 0.0416 (0.1710) | 0.0983, 0.0956 (0.1939) | 0.0709, 0.1632, 0.1708, 0.1323 (0.5372) | 0.2323 |

评测平台标识：SLF7C21943504。

### 结论：C1 为失效优化

- 总分从 1.1824 降至 1.1346，变化为 -0.0478。
- 物料子项和从 0.1650 升至 0.1710；但 user 从 0.2211 降至 0.1939（-0.0272），recommendation 从 0.5678 降至 0.5372（-0.0306）。
- world 仅从 0.2286 升至 0.2323，不能抵消 user 与 recommendation 的下降。

因此，C1 的 Trie 优先加分项 cap 在当前数据、rank16、学习率和训练协议下没有带来有效优化，且是负向消融。后续不将 C1 作为候选训练基线，也不根据它的代理训练指标继续调大 Trie 权重；相关 checkpoint 仅保留作失败消融对照。
