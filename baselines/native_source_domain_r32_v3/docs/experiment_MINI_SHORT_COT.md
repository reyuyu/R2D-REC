# 实验 MINI-SHORT-COT

## 目标

以正式 `Mini-Fix` 为父实验，只改变懂推荐 CoT 的监督内容：每条 `recommendation_cot` 的
`<think>...</think>` 只保留 `【兴趣归纳】` 标题及其正文，删除前置套话、`【行为模式】`、
`【预测总结】` 和其余后续分析。`</think>` 后的最终推荐答案逐字保持不变。

动机来自强化学习阶段的观察：beam 输出偏向只保留兴趣归纳时结果更好。本实验验证这种短 CoT
监督是否也能在 SFT 阶段降低无关推理负担，并改善最终推荐生成。

## 单变量合同

- Parent：`Mini-Fix`，49,490 rows。
- 修改：6,235 条 `recommendation_cot` 的 think span。
- 不修改：4,957 条 `recommendation_nocot`、2,000 条 user、36,298 条 material。
- 不修改：最终答案、`aux_metadata_json`、多正例 group、样本顺序、source segment。
- 不修改：Native SID8 CE、LoRA、optimizer、LR、packing、GC、batch、validation。
- `REC-PU=false`、`PackRatio=false`、`MiniTopK=false`、Smooth=false。

## 训练合同

- LoRA r32 / alpha64 / dropout0.05，target all。
- 4 GPU，global batch 64，GA16，GC0.4。
- 8K neat packing，FA2，Liger，bf16 / pure_bf16。
- LR 2e-4 cosine，warmup 0.03，weight decay 0.01，seed 20260806。
- 2 epochs；每个 epoch 保存 checkpoint；不 resume。

## 文件

- Dataset：`/data/lf_data_versions/alltrain/mini_short_cot/onereason_mini_short_cot.jsonl`
- Manifest：`/data/lf_data_versions/alltrain/mini_short_cot/manifest.json`
- Cache：`/data/lf_data_versions/alltrain/mini_short_cot/tokenized_train_8k_sid8w8`
- Config：`config/train_mini_short_cot_4gpu_gc04_2epoch.yaml`

## 状态

- 数据结构审计：PASS。49,490 rows 和原顺序不变；6,235 条 recommendation CoT 全部转换；
  43,255 条非 CoT 行保持原始 JSONL bytes；最终答案、aux metadata 和所有非 output 字段 parity PASS。
- 标题审计：6,234 条标准 `【兴趣归纳】`，1 条编号标题 `#### 1. 兴趣归纳`；均被确定性归一成
  `【兴趣归纳】`。原 think 共 7,865,198 chars，新 think 共 4,151,114 chars，减少 47.22%。
- Cache / SID8 审计：PASS。4,106 packs；material/recommendation/user_action/user_chain segments
  为 36,298 / 11,192 / 1,200 / 800；普通 token=1、canonical=4、SID/domain=8，违规为 0。
- 相比 Mini-Fix 的 4,365 packs 减少 5.93%；65 steps/epoch，2 epochs 共 130 steps，warmup 4 steps。
- 正式训练：已完成 2 epochs（130/130 optimizer steps）。

## 正式运行

- Run ID：`MINI-SHORT-COT-R32-2E-GC04-4GPU-20260818-163948`
- Output：`/data/outputs/baselines/native_source_domain_r32_v3/MINI-SHORT-COT-R32-2E-GC04-4GPU-20260818-163948`
- Log：`/data/logs/baselines/native_source_domain_r32_v3/MINI-SHORT-COT-R32-2E-GC04-4GPU-20260818-163948/train.log`
- Epoch 1 checkpoint：`checkpoint-65`
- Epoch 2 checkpoint：`checkpoint-130`
- 总运行时间：`1:27:01.73`
- 平均速度：`0.025 optimizer step/s`，约 40 秒/step。
- 最终 `train_loss=55.7857`；训练正常结束，未出现 NaN、OOM、Traceback 或 DDP mismatch。

最终 recommendation teacher-forcing 监控：

| 指标 | Epoch 2 结束值 |
| --- | ---: |
| CoT body CE | 1.7419 |
| CoT Gold SID CE | 4.7994 |
| No-think Gold SID CE | 4.7444 |
| Gold a / b / c CE | 5.0755 / 4.5769 / 4.6807 |
| a hit@8 / a hit@32 | 0.3428 / 0.5623 |
| b hit@8 / c hit@8 | 0.4201 / 0.4580 |
| chain 32/8/8 | 0.1165 |

## 外部评测

用户确认以下结果对应 **Epoch 2 / checkpoint-130**。指标名称与顺序沿用外部评测器原始输出；
本记录不对未公开正式名称的分项擅自重命名。

```text
aggregate: 1.2989

0.0613, 0.0369, 0.0514, 0.0424
0.1451, 0.0723
0.1260, 0.1598, 0.2002, 0.1755
0.2279
```

相对统一参考线 BETA-baseline Epoch 2 的 `1.3246`，总分低 `0.0257`。因此缩短 Recommendation
CoT 将训练 pack 数和计算量明显降低，并取得 `1.2989`，但尚未追平完整 BETA-baseline。该结论只针对
当前固定的 Mini-Fix 数据规模和两轮训练合同；不能仅根据更低的 CoT body CE 推断外部生成质量更高。
