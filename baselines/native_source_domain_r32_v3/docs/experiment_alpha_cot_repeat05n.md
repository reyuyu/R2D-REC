# 实验 Alpha-CoT Repeat-Normalized Weighting

## 当前正式运行

- Run: `ALPHA-COT-REPEAT05N-R32-2E-GC04-4GPU`
- 数据：Alpha corrected SID8 cache，CoT body 使用 `0.5/N`，No-think 和最终 Gold SID 保持原权重
- 模型与训练：LoRA r32/alpha64/dropout 0.05，4 GPU，GA16，8K neat packing，GC0.4，bf16、FA2、Liger，LR 2e-4 cosine，2 epoch / 1058 optimizer steps
- 输出：`/data/outputs/baselines/native_source_domain_r32_v3/ALPHA-COT-REPEAT05N-R32-2E-GC04-4GPU`
- 日志：`/data/logs/baselines/native_source_domain_r32_v3/ALPHA-COT-REPEAT05N-R32-2E-GC04-4GPU/train.log`

## 目标与边界

同一 recommendation group 内重复出现的 CoT 样本，其 `<think>...</think>` body 监督权重为 `0.5/N`。最终答案部分、最终 Gold SID、No-think response、其它任务和模型 forward 均不改变。该改动只改变 loss weight，不改变输入 token、labels、packing 或数据顺序。

## 30-step smoke

Smoke 从 base model step 0 开始，已完成 30 个 optimizer steps。step 0 cache parity 通过；`missing_gold=0`、`invalid_route=0`，无 NaN、OOM、DDP mismatch。CoT body 原始 CE 从 2.468 降到 2.097，CoT Gold SID CE 约 4.75，No-think Gold SID CE 约 5.11。加权推荐分子中 CoT body 约 62--64%，No-think 约 15--18%。

## 解释训练 loss

`task_loss_recommendation` 是每个 recommendation segment 的加权 CE 样本均值，分母是有效监督 token 数，不是 loss weight 之和。因此 CoT body 权重从 1 降到 `0.5/N` 后，task loss 数值必然低于旧 Alpha/BATA；跨实验应比较未加权 raw CE（CoT body、CoT Gold SID、No-think Gold SID、Gold a/b/c）和固定 dev/probe 指标，不能直接比较 task loss 绝对值。

## 验证与监控

正式配置保留 Alpha monitor、训练 teacher-forcing、固定 probe 与 epoch-end full dev。日志每 5 step 记录总 loss、四任务 loss、grad norm、LR 和推荐 raw CE；训练验证指标写入对应 run 的 `alpha_validation_metrics.jsonl`。

本实验不是正式评测结论；需等待两轮训练和固定 dev/full dev 结果后再比较。
