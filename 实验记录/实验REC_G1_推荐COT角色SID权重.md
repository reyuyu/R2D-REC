# 实验 REC_G1：推荐 CoT 角色感知 SID 权重

## 目标

REC_G1 基于 REC_F 独立创建，仅改变 recommendation/cot 的 SID 加权：`<think>...</think>` 内的 SID 组件使用 `1.0`，最终答案 SID 使用 `8.0`；recommendation/nocot 以及 material、user_action、user_chain 保持 SID weight `8.0`。目的是避免推荐 CoT 解释中的历史 SID 被高权重 CE 过度强化，同时保留最终推荐答案的监督强度。

没有修改数据、模型、LoRA、packing、GradNorm/Ortho、optimizer、scheduler，也没有增加 forward 或 backward。关闭 `recommendation_role_aware_sid_weighting_enabled` 后恢复 REC_F 的统一 SID weight 行为。

## 配置

独立 smoke 配置：

`configs/onereason/onereason_lora_2gpu_recG1_roleaware_sid_smoke.yaml`

新增字段：

```yaml
recommendation_role_aware_sid_weighting_enabled: true
recommendation_think_sid_weight: 1.0
recommendation_final_sid_weight: 8.0
```

其余参数沿用 REC_F：四任务、8K BFD、coverage/deficit、每个 macro-step 8 个 global microbatch、rank16、学习率 `2e-4`、dropout `0.05`、0.75 selective GC、SID weight `8.0`。

## 实现

`sid_token_weighting.py` 复用当前 tokenizer 的 `<think>`/`</think>` added-vocabulary token ID，在每个监督 label 有效连续段内定位角色边界。pack 中每个 segment 的 `labels=-100` 间隔天然隔离，缺少完整边界时保守按 final 处理。普通 SID mask、causal shift 和归一化分母保持原实现：

```text
loss = sum(CE * token_weight * valid) / sum(token_weight * valid)
```

Trainer 仍然只执行一次 forward/backward；加权后的 recommendation raw loss 继续进入四任务 GradNorm。

新增监控：

`rec_think_sid_token_count`、`rec_final_sid_token_count`、`rec_think_sid_ce`、`rec_final_sid_ce`、`rec_think_sid_weighted_mass`、`rec_final_sid_weighted_mass`。

## 数据审计

脚本：`scripts/audit_recommendation_role_sid.py`；结果保存在服务器 `/data/lf_data_versions/alltrain/v2_recommendation_dual/REC_G1_ROLE_AUDIT.json`。

| 路由 | 样本数 | 含 think SID | think SID 组件 | final SID 组件 | think SID 与历史重叠 |
| --- | ---: | ---: | ---: | ---: | ---: |
| recommendation/cot | 20,664 | 17,270（83.58%） | 458,781 | 61,992 | 148,985 / 148,985（100%） |
| recommendation/nocot | 20,666 | 0 | 0 | 61,998 | 不适用 |

CoT 中完整 think SID 均来自历史并不意味着最终 gold 一律不在历史；审计显示 gold-in-history 样本约为 CoT 3.67%、No-think 3.55%。实现按 token 位置分角色，不按 SID 是否出现过历史决定权重。

## 历史复制诊断

脚本：`scripts/measure_recommendation_history_copy.py`。输入预测 JSONL，支持 `prediction`/`candidates`、`reference`/`gold` 和 prompt/history，输出 `history_copy@1/8/32`、`nongold_history_copy@1/8/32`、think/no-think/overall 路由统计，以及 beam 唯一 SID 和历史 SID数量。合成测试覆盖在 `tests/test_recommendation_history_copy.py`。

## 验证

- `tests/test_rec_g1_role_aware_sid.py`：5 项通过，覆盖 think/final 权重、gold 在历史、no-think、其他任务、pack segment 隔离和开关回退。
- `tests/test_sid_token_weighting.py`：18 项通过，原有 SID weighted CE、shift、一次 forward/backward 和 Action 接入均回归通过。
- `tests/test_recommendation_history_copy.py`：通过。
- `tests/test_multitask_macro.py`：14 项通过；`tests/test_multitask_gradient_controller.py`：15 项通过。
- 双卡 2/3 50-step smoke：通过；耗时约 10 分 52 秒，约 12.9 秒/macro-step，GPU 峰值约 71.7/69.1 GiB；checkpoint-25/50 均生成；无 OOM、NaN、Traceback。四任务 GradNorm、冲突余弦、raw loss 和角色指标均有记录。
- checkpoint 恢复测试按用户指示未继续；一次准备性的恢复启动因当前环境 torch 版本低于 Transformers 对 `torch.load` 的安全要求而退出，未影响 REC_F 或 REC_G1 smoke 输出。

REC_G1 仅完成 50-step smoke，不启动正式 5718-step 训练。

## 评测结果（2026-08-08）

实验 RECG1（checkpoint-2000）：

```text
总分：1.1795
物料四域：0.0449, 0.0371, 0.0378, 0.0424
用户两项：0.1526, 0.0953
推荐四域：0.1111, 0.1428, 0.1666, 0.1206
world：0.2283
```
