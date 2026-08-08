# 实验 REC_G2：Recommendation No-think Multi-Positive Prefix-Trie Loss

## 状态

REC_G2 基于 REC_G1 和 `v3_recommendation_multi_positive` 数据版本实现。CPU 单元测试、既有回归测试、配置校验和 Python 编译均通过。GPU 0/1 的全量 V3、50-step、2-GPU smoke 已通过：每步 8 个 global pack、每卡 4 个 pack，运行 `666.01 s`，无 NaN、OOM、Traceback 或 DDP hang；`checkpoint-25` 与 `checkpoint-50` 已保存。按要求未做 checkpoint 恢复测试，也未启动正式 5718-step 训练。

## 唯一实验变量

REC_G2 只改变 `recommendation/nocot` 最终 SID 的 `s_a/s_b/s_c` 三个 teacher-forced prediction position：

```text
REC_G1 single-gold weighted CE
        ->
REC_G2 all-gold prefix-trie set NLL
```

以下内容保持不变：

- SID weight 仍为 8，normalized denominator 保持原值；
- recommendation/cot 完全沿用 REC_G1：think SID weight 1，final SID component weight 8；
- material、user_action、user_chain 和 recommendation/cot 不受 Trie objective 影响；
- 四任务 GradNorm、packing、coverage/deficit scheduler、LoRA、模型结构、optimizer、scheduler、forward/backward 次数不变；
- 不做 1/K group normalization、curriculum、history negative、ranking 或 level weighting。

## Trie 与 loss replacement

V3 每个 No-think segment 的 `recommendation_multi_positive` 元数据提供完整 `all_gold_sids`。controller 按 `group_id` 缓存并严格校验完整 SID、domain、group size、current gold 和重复路径。

对于当前 row 的 teacher-forced path `(a*, b*, c*)`：

```text
A = {a | (a,b,c) in all_gold}
B = {b | (a*,b,c) in all_gold}
C = {c | (a*,b*,c) in all_gold}
```

集合只来自真实完整路径，不生成 Cartesian product。每个 label position `p` 使用已有 forward 的 `logits[..., p-1, :]`：

```text
set_nll(p) = logsumexp(logits[p-1]) - logsumexp(logits[p-1, allowed])
```

使用 FP32 logsumexp。实现采用 exact loss delta，而不是增加辅助 loss：

```text
L_G2 = L_G1 + sum_r w_r * (set_nll_r - single_ce_r) / D
```

其中 `w_r=8`，`D` 是当前 SidTokenWeightingController 的原 normalized denominator。singleton group 的三个集合都是单元素，严格退化为 REC_G1（CPU 测试误差小于 `1e-6`）。

## 元数据与 packed 接入

controller 从 segment-level `sample_metadata` 读取 V3 元数据，使用 `segment_offsets` 在各 segment 内唯一定位当前完整 SID，再将 component label position 映射到 `p-1` 的 logits。不会在整个 packed labels 中全局搜索，也不会把不同 segment 的 group 混合。

元数据不在 `_MODEL_INPUT_KEYS` 中，不进入 `model.forward`。一个 pack 仍是一次 forward、一次 backward；不会为 CoT-only gold branch 额外 forward。

## 监控指标

新增精简指标：

```text
rec_mp_segments
rec_mp_group_size
rec_mp_single_ce
rec_mp_trie_nll
rec_mp_delta
rec_mp_allowed_a
rec_mp_allowed_b
rec_mp_allowed_c
rec_mp_multi_a_ratio
rec_mp_multi_b_ratio
rec_mp_multi_c_ratio
rec_mp_invariant_violation_max
```

`rec_mp_delta` 是 `trie_nll - single_ce` 的位置平均值，理论上不应大于数值误差；`rec_mp_invariant_violation_max` 只记录正向违反量。

## 配置

独立 smoke 配置：

```text
configs/onereason/onereason_lora_2gpu_recG2_multi_positive_trie_smoke.yaml
```

关键字段：

```yaml
multitask_dataset_version: v3_recommendation_multi_positive
sid_token_weighting_enabled: true
sid_token_weight: 8.0
recommendation_role_aware_sid_weighting_enabled: true
recommendation_final_sid_weight: 8.0
recommendation_multi_positive_trie_enabled: true
max_steps: 50
```

其余参数继承 REC_G1：8K、GC 0.75、LoRA rank16、学习率 `2e-4`、same-subtask BFD、coverage/deficit、四任务 GradNorm、8 global packs/macro-step。

## CPU 测试

```bash
PYTHONPATH=src python3 tests/test_recommendation_multi_positive.py
PYTHONPATH=src python3 tests/test_recommendation_v3_multi_positive.py
PYTHONPATH=src python3 tests/test_rec_g1_role_aware_sid.py
PYTHONPATH=src python3 tests/test_sid_token_weighting.py
PYTHONPATH=src python3 tests/test_multitask_macro.py
PYTHONPATH=src python3 tests/test_multitask_gradient_controller.py
```

已通过：singleton equivalence、A/B/C branch filtering、禁止 Cartesian product、metadata 损坏 fail-fast、packed 多 segment 隔离、causal shift、梯度方向、disabled noop、V3 透传、REC_G1 role-aware、SID weighting、四任务 GradNorm 和 macro loader 回归。

## GPU smoke 与已知限制

已通过的 2-GPU full-V3 smoke 使用真实 8K、LoRA、GC 0.75、四任务 GradNorm 与 DDP 接线，50 个 macro-step 共耗时 `666.01 s`（训练期约 `13.3 s/step`，不含全量 BFD plan 构建）。每步均为全局 8 pack、两 rank 各 4 pack，四任务 allocation 均为正。GPU 峰值约为 GPU0 `74,477 MiB`、GPU1 `74,451 MiB`。含 No-think recommendation 的步骤中，示例指标为 `rec_mp_segments=14`、`rec_mp_group_size=4.643`、`rec_mp_single_ce=4.156`、`rec_mp_trie_nll=3.567`、`rec_mp_delta=-0.589`，`rec_mp_invariant_violation_max=9.537e-7`，符合集合 NLL 不高于 single-gold CE 的不变量。GradNorm measurement 正常发生；因 warmup=200，50-step smoke 不应进行权重更新。

`checkpoint-25` 和 `checkpoint-50` 保存成功；根据要求未做恢复验证。50-step smoke 仅验证工程稳定性，不能据此判断正式 recommendation 评测收益。

V3 中部分合法 gold 被分配到 CoT route。REC_G2 仍使用每个 No-think row 的完整 `all_gold_sids` 构建 allowed set，但只对当前 No-think teacher-forced path 计算三个位置，不为 CoT-only branch 增加 forward；因此它是当前路径上的完整 prefix-positive objective，而不是多 forward 的完整 Trie 遍历。\n