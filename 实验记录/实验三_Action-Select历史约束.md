# 实验三：Action Select 历史约束与长度平衡

## 问题与边界

`user/action_nocot` 当前主要错误包括：输出历史中不存在的完整 SID、重复输出同一完整 SID、答案尚未完成就停止，以及正确答案结束后继续生成。本实验只增加一个可关闭的训练辅助目标，不拆出第五个顶层任务。因此 balanced_40、user 内部采样比例、GradNorm/Ortho 任务列表、LoRA 前向、optimizer 和 DDP 分配均不变。

第一版不包含 Illegal Top-K、约束解码、beam search 训练、选择题加权或中文短语运行时识别。Teacher forcing 只能改善训练位置上的概率分布，不能像约束解码一样从机制上保证生成结果始终合法。

## 为什么使用完整 SID Trie

一个 Action SID 是：

```text
domain + s_a + s_b + s_c
```

历史保存四元组而不是四列 token 集合。若把 domain/a/b/c 分开，模型可能把不同历史 SID 的分量交叉拼成从未出现过的非法 SID。动态 Trie 在 domain、domain+a、domain+a+b 三层前缀下分别保留全部合法后继。

第 k 个答案 SID 前，若 gold 自身没有重复：

```text
available_k = history_sids - previously_used_complete_sids
```

只删除已经使用的完整四元组。共享 domain/a/b 前缀的其他完整路径仍保留，因此这不是 token 级 no-repeat。若 gold 自身包含重复 SID，则保留历史合法性 Trie，但对该 segment 关闭动态移除，避免与主 CE 冲突。

## 数据元数据与 packing

启用总开关后，`TokenizedSubDataset` 只对 `task_name=user, subtask_name=action_nocot` 使用当前 tokenizer 的 added vocabulary 解析 tokenized `input_ids/labels`。结果按 sample index 缓存，同一样本不会在每个 epoch 或 forward 重扫。普通样本仍为 `sample_metadata={}`。

每个 Action sample 保存：历史完整 SID、gold SID 与四个本地 token 位置、非最终继续边界、最终 tail 位置、解析状态及 gold 重复标记。`TaskPackCollator` 按 `packed_position=segment_start+local_position` 生成独立的 `action_aux_metadata`。每个 segment 使用自己的 Python 元数据和 available set，不会跨 segment 污染。该字段不属于 Trainer 的模型输入白名单，不会传入 `model.forward`。

Gold 不在截断后历史中时继续使用普通 CE，但该 SID 跳过 Trie loss。解析失败时整个 segment 只使用 base loss，并记录失败率，不中断训练。

## Loss 公式

Trie 的每个层级位置使用 FP32 logits：

```text
allowed_loss = logsumexp(logits[type_ids]) - logsumexp(logits[allowed_ids])
```

它只把当前语义类型内的概率质量拉回历史合法分支，普通 CE 继续负责从合法候选中选 gold。因果位置严格使用 `logits[t-1]` 预测 `labels[t]`。归一化顺序为 SID 内四位置平均、segment 内 SID 平均、microbatch 内有效 segment 平均。

Length Guard 同时处理两个方向：

- 非最终 SID：加强 next-domain CE；若可可靠区分 separator 和最终 tail，则加强 separator CE，并抑制 stop-specific token。
- 最终 SID：加强最多四个真实 tail token 的 CE，并抑制 tail 位置重新分配给任一 domain token的概率质量。

最终辅助损失为：

```text
aux_raw = trie_weight * trie_loss + continue_loss + stop_loss
max_aux = cap_ratio * detach(action_ce)
cap_scale = min(1, max_aux / (detach(aux_raw) + eps))
action_aux = aux_raw * cap_scale * warmup_factor

user_raw_loss = base_loss + action_aux
weighted_loss = user_raw_loss * current_user_gradnorm_weight / original_loss_divisor
```

warmup 使用已完成 macro-step 数，step 0/50/100 对应 0/0.5/1.0。cap 的分子和分母都 detach，辅助损失默认不超过 Action CE 的 8%。Action total raw loss 在 GradNorm user 权重之前统计，因此它参与 user 的真实梯度和 loss EMA。

## 配置

完整配置：

```text
configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_v1.yaml
```

相对实验 A 只更换 `output_dir` 并新增：

```yaml
user_action_aux_enabled: true
user_action_history_trie_enabled: true
user_action_history_trie_weight: 0.06
user_action_length_guard_enabled: true
user_action_continue_domain_extra: 0.75
user_action_continue_separator_extra: 0.20
user_action_no_early_stop_weight: 0.02
user_action_stop_domain_weight: 0.05
user_action_stop_tail_extra: 1.0
user_action_max_stop_tail_positions: 4
user_action_aux_cap_ratio: 0.08
user_action_aux_warmup_steps: 100
```

总开关关闭时不创建元数据解析器或辅助控制器，Trainer 仍走原来的单 loss 返回路径。仅 Trie 消融可关闭 `user_action_length_guard_enabled`。

## 代码入口

- 一次性解析：`src/llamafactory/data/action_select.py`
- dataset 缓存与 packing 偏移：`src/llamafactory/data/multitask.py`
- 辅助目标：`src/llamafactory/train/sft/user_action_auxiliary.py`
- 单 forward 接入：`src/llamafactory/train/sft/trainer.py`
- 参数：`src/llamafactory/hparams/data_args.py`
- 独立评测：`scripts/evaluate_sid_action_predictions.py`

Action microbatch 调用一次 `compute_loss(..., return_outputs=True)`，直接复用 `outputs.logits`，没有第二次 model forward、backward、optimizer、trainable parameter 或 DDP collective。

## 指标

- `a_act_*`：segment 数、解析成功率、历史/gold SID 数、gold 在历史率、gold 重复率。
- `b_act_allowed_*_mass`：各 Trie 层级在同语义类型中的合法概率质量。
- `c_act_*`：完整 SID 移除数、继续边界数、提前停止概率与最终重新开始 domain 概率。
- `d_act_*`：Trie/Continue/Stop、最终 aux、Action CE、cap 与 warmup。
- `e_act_user_*`：Action microbatch 的 user base/aux/total raw loss 拆分。

这些统计附加在现有 `task_statistics` flat tensor 中，复用原有一次 all-reduce。评测脚本另外输出完整四元组的历史合法率、幻觉率、重复率、unique rate、数量误差和 set precision/recall/F1，并区分历史外、重复、合法但选错、漏选、多选、格式错误与显式 length 截断。

## 启动与测试

正式训练命令（本次验收不执行 5200 step）：

```bash
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=2,3 FORCE_TORCHRUN=1 \
llamafactory-cli train configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_v1.yaml
```

单元测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 tests/test_user_action_auxiliary.py
PYTHONPATH=src python3 tests/test_action_select_evaluation.py
```

当前核心单元测试覆盖解析、缓存、packing 偏移、完整 SID Trie、off-by-one、Continue/Stop、warmup/cap、有限标量、关闭等价、任务隔离、一次 forward/backward、GradNorm user raw loss 和原 divisor。

验收结果：

```text
Action auxiliary 单元测试：41/41
Action 独立评测测试：4/4
原 multitask macro/packing 调度：9/9
实验 A gradient controller 回归：12/12
实验 B Ortho 回归：17/17
packing attention eager/FlashAttention-2：2/2
随机小 Qwen3+LoRA 单卡 smoke：通过
GradNorm/Ortho/Action Aux 三组双卡 DDP smoke：通过
```

双卡 Action smoke 覆盖 `forward -> aux -> backward -> statistics all_reduce -> optimizer.step`，并验证两个 rank 更新后参数一致。

## 真实数据审计与开销

对 `onereason_user_action_nocot_train98` 全量 16,309 条样本使用服务器实际 tokenizer 审计：

- parse success：16,309/16,309，100%；
- 平均历史完整 SID：199.8178；
- 平均 gold SID：13.9118；
- gold in history：226,826/226,887，99.9731%；
- gold duplicate segment：0；
- parser 自身平均 0.987 ms/样本；全量数据读取与审计 132.75 秒。

解析使用一次性 semantic token type lookup 和 NumPy 向量匹配；`TokenizedSubDataset` 按 sample index 缓存。Packing 只复制需要偏移的位置字段，历史 SID 以只读引用复用。Trainer 在 `_prepare_inputs` 前过滤模型字段，因此不会每 step 递归复制元数据。

GPU 微基准使用服务器本地 tokenizer、`vocab_size=176255`、随机初始化两层小 Qwen3、28 token、20 个计时 step：

- Action Aux 关闭：8.024 ms/microbatch；
- Action Aux 开启：18.218 ms/microbatch；
- 差值：10.194 ms/microbatch；
- 新增峰值显存：197,632 bytes，约 193 KiB；
- 每个 microbatch 仍为一次 forward 和一次 backward。

小模型的 forward 很短，因此不能把该相对百分比外推到 8B。正式 8B 端到端时间与显存增量仍需短跑采样后确认；本次没有启动 5200-step 训练。

## 尚未实现

- Illegal Top-K unlikelihood；
- constrained decoding / beam search 训练；
- 运行时中文模板识别；
- Action Select 独立 GradNorm 顶层任务；
- 选择题答案加权。
