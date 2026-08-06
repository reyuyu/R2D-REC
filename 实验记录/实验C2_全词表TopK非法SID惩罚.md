# 实验 C2：全词表 Top-K 非法 SID 惩罚

## 状态

已完成 rank16 双卡正式训练与 checkpoint 评测。C2 是在实验 C/C-fast/C1 的已有 Action Select 元数据、task-wise packing、macro-step Trainer 与 GradNorm 接线上的独立消融，不改变数据调度、LoRA 前向、模型前向次数、backward 次数或 DDP collective。

## 动机

C1 的动态 Trie 损失是“语义类型内部”的合法概率质量：在预测 `domain/a/b/c` 位置时，只比较对应语义 token 集合。它能改善已进入 SID 语义空间后的合法路径选择，却不会直接压低分数更高的普通词表 token。因此，生成时仍可能在 SID 位置插入自然语言、格式 token 或其他非语义 token。

C2 把监督重点移到当前 logits 的全词表竞争项上，直接处理这类“未进入 SID 空间”的错误候选。

## C2 损失

只对 `task_name=user` 且 `subtask_name=action_nocot` 生效。对每个 teacher-forcing SID token 位置，仍从历史完整四元组集合构造：

```text
available = history_sids - 已在当前答案前出现的完整 SID
```

gold 自身含重复时不移除；gold 不在历史或解析失败时跳过该 SID 的辅助项，只保留普通 CE。

给定该位置的动态合法 token 集合 (A_t)，令 (z_g) 为 gold token logit，`J_t` 为全词表中排除 (A_t) 后最高的 K=5 个 logit。C2 使用：

```text
L_topk(t) = mean_j softplus(z_j - z_g + margin), j in J_t
```

因此 Top-K 中的普通中文 token、格式 token、错误 SID 分支和已用完整 SID 分支都会被惩罚。gold 的普通 CE 仍负责选择正确 token。

每个 SID 先平均四个位置，再平均 segment 和 microbatch。辅助项在 user GradNorm 权重之前加入 raw loss：

```text
raw_user_loss = base_loss + action_aux_loss
weighted_loss = raw_user_loss * current_user_gradnorm_weight
```

## C2 配置

配置文件：

```text
configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_c2_topk_illegal.yaml
```

关键字段：

```yaml
user_action_history_trie_enabled: false
user_action_length_guard_enabled: false
user_action_topk_illegal_enabled: true
user_action_topk_illegal_k: 5
user_action_topk_illegal_margin: 0.0
user_action_topk_illegal_weight: 0.02
user_action_aux_cap_ratio: 0.02
user_action_topk_illegal_cap_ratio: 0.02
user_action_aux_warmup_steps: 100
```

这里关闭的是 C/C1 的 allowed-mass Trie 损失、Continue/Stop 以及其显式损失项；完整 SID 的 `history - used` 仍保留为每个位置合法集合的构造依据。初始辅助预算仅为 Action CE 的 2%，用于避免 C/C1 中长期触顶 8% 上限的现象。

## 指标

训练监控只保留以下四项，且都按当前 teacher-forcing SID 目标位置，以全词表概率计算：

- `a_act_legal_sid_mass`：动态合法 SID token 集合的总概率质量。越高越好。
- `a_act_topk_illegal_mass`：动态非法 token 中 Top-K（默认 K=5）的总概率质量。越低越好。
- `a_act_top1_illegal_hit_rate`：全词表 Top-1 token 不在动态合法集合中的比例。越低越好。
- `a_act_gold_top5_rate`：gold token 位于全词表 Top-5 的比例。越高越好。

不再输出解析率、历史条数、长度 guard、各项辅助 loss、cap 和 warmup 等 Action 专用中间指标。它们不影响 C2 训练；GradNorm、任务总 loss、梯度范数和梯度冲突指标继续由原监控保留。

## 统计开销

在空闲 A800 上以 151,936 vocab、10 个 SID（40 个目标位置）的后处理基准实测：

| 路径 | 中位耗时 |
| --- | ---: |
| C2 Top-K 损失加旧诊断 | 2.41 ms |
| C2 Top-K 损失加上述四项指标 | 6.55 ms |

新增约 4.14 ms，来自全词表 logsumexp 与合法集合质量归约。该工作只发生在 user/action_nocot microbatch，不发生在其他三个顶层任务或 user chain；没有增加模型 forward、backward、DDP 通信、参数或显存常驻 buffer。它相对真实 8B 模型训练 microbatch 很小，但不应被称为零成本。

## 已验证

- 普通非 SID 词表 token 成为高分竞争项时，会被全词表 Top-K 损失选中；反向传播对其产生正梯度（梯度下降会降低该 logit），对 gold 产生负梯度。
- 历史 SID 使用后只移除完整路径，共享前缀的其他 SID 仍可用。
- gold 不在历史时 C2 安全跳过辅助项。
- 不支持的非单 packed-sequence logits 形状会明确报错，不会静默退回而漏掉 C2 损失。
- Action 辅助模块 54 项无夹具测试全部通过。

## 边界

C2 不是约束解码，不保证推理时绝不出现非法 SID 或自然语言；它只用当前一次 teacher-forcing forward 的 logits 调整训练排序。Illegal Top-K unlikelihood、beam/search 训练和解码期约束仍未实现。

## 正式 checkpoint 评测

以下按物料四域、用户两项、推荐四域、world 的固定顺序记录。

| Checkpoint | 总分 | 物料 4 域 | 用户 2 项 | 推荐 4 域 | World |
| ---: | ---: | --- | --- | --- | ---: |
| 1000 | 1.1963 | 0.0425, 0.0383, 0.0402, 0.0430 | 0.1341, 0.0843 | 0.0961, 0.1734, 0.1708, 0.1458 | 0.2279 |
| 5200 | 1.2044 | 0.0459, 0.0376, 0.0456, 0.0418 | 0.1528, 0.0997 | 0.0821, 0.1598, 0.1694, 0.1359 | 0.2338 |

C2 在既定协议下完成。其训练目标仍只约束 teacher-forcing 下的全词表非法竞争项，不能单独证明自由生成的 SID 幻觉率已经改善。
