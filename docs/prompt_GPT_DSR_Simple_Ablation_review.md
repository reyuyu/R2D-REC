# 请审查 GR_REC_DSR_Simple_Ablation_v1 的实验结论与下一步设计

你现在是这个 GRPO 推荐实验的独立研究审查者。请先阅读下面列出的 GitHub
文件，再结合本提示词中的补充证据给出判断。不要只复述实验记录，也不要默认
当前作者的结论正确；请主动寻找反证、混杂因素和不能由现有证据支持的因果推断。

## 必读文件

1. [DSR-Simple 完整实验记录](experiment_GR_REC_DSR_Simple_Ablation_v1.md)
2. [DSR-Simple 公式与冻结合同](../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_simple_v1/README.md)
3. [GR_REC_v1 基线及外部评测记录](../baselines/native_source_domain_r32_v3/docs/experiment_GR_REC_v1.md)
4. [完整 DSR v1 的设计与验收记录](../baselines/native_source_domain_r32_v3/docs/experiment_GR_REC_DSR_Ablation_v1.md)
5. [完整 DSR v1 Pilot200 forensic report](../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/results/pilot200_forensic_20260818/GR_REC_DSR_PILOT200_FORENSIC_REPORT.md)
6. [Pilot200 结构化 forensic summary](../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/results/pilot200_forensic_20260818/summary.json)

如果某项结论无法从这些文件或下述补充证据中验证，请明确标成“证据不足”，
不要补造日志、显著性或机制解释。

## 实验关系

- Parent checkpoint 是原始 BATA adapter。
- `GR_REC_v1` 是冻结基线。
- `GR_REC_DSR_Ablation_v1` 是完整 DSR 参考消融，只完成 200-step pilot，
  没有完整 epoch 外部得分。
- `GR_REC_DSR_Simple_Ablation_v1` 是独立 sibling，不从 GR_REC_v1 或完整
  DSR 续训；它完成了 2316-step 单 epoch。
- Simple 的 Think 只保留 Raw interest count 和 dead-zero target-domain A
  diversity；Grounded N、Coverage、证据多样性、prefix support、entropy
  exploration 等均为 diagnostic-only，不参与训练。
- Simple 的 `S_N = 1` 当且仅当 `2 <= Raw N <= 4`，区间内不区分 2、3、4。
- Simple 的 `D_A` correctness-agnostic，且只在 Think primary all-zero 分支参与。
- NoThink 完整复用 DSR v1 的 all-zero repetition-aware rescue。

## 外部评测证据

评测顺序固定为：总分；素材 video/prod/ad/living；用户 action/chain；
推荐 video/prod/ad/living；世界。记录只保留四位小数，分项和 aggregate 最多
存在 0.0002 的舍入差。

| 模型 | aggregate | 素材合计 | 用户合计 | 推荐合计 | 世界 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BETA baseline | 1.3313 | 0.1807 | 0.2545 | 0.6594 | 0.2368 |
| GR_REC_v1 step 1000 | 1.3270 | 0.1817 | 0.2550 | 0.6563 | 0.2338 |
| GR_REC_v1 step 1500 | **1.3510** | 0.1816 | 0.2563 | **0.6800** | 0.2331 |
| GR_REC_v1 step 2000 | 1.3326 | 0.1806 | 0.2581 | 0.6607 | 0.2331 |
| DSR-Simple step 1000 | **1.3440** | 0.1807 | 0.2560 | **0.6722** | 0.2349 |
| DSR-Simple step 1500 | 1.3254 | 0.1813 | 0.2552 | 0.6535 | 0.2353 |
| DSR-Simple step 2316 | 1.3203 | **0.1829** | 0.2553 | 0.6446 | **0.2375** |

已有记录认为同一模型单次外部评测约有 `0.01` 波动，因此：

- Simple step 1000 比 BETA 高 0.0127，但只有一次评测。
- Simple step 1000 比 GR_REC_v1 最佳 step 1500 低 0.0070，不能据此稳定排序。
- Simple step 1000 到完整 epoch 下降 0.0237，超过记录中的典型波动带。
- 这 0.0237 回落中，推荐合计下降 0.0276；素材和世界分别上升
  0.0022、0.0026，用户基本不变（-0.0007）。
- 推荐四项从 Simple step 1000 到完整 epoch 的变化是：video +0.0028、
  prod -0.0204、ad -0.0028、living -0.0072。

## 开发机完整记录的 checkpoint 前 200-step 窗口统计

以下值由完成后的 append-only JSONL 在 CPU 上计算。数据完整性审计确认 metrics
严格覆盖 1..2316，无缺步、重复、坏 JSON 或非有限存储值。原始大体积 run
artifact 未提交 GitHub，因此请把这些数字视为本提示词提供的审计证据，而不是
声称可以从 GitHub 重新计算。

| 指标 | step 1000 | step 1500 | step 2316 |
| --- | ---: | ---: | ---: |
| policy reward mean | 1.5311 | 1.4706 | 1.4076 |
| zero-std ratio | 0.3625 | 0.3575 | 0.3600 |
| approx KL | 0.000415 | 0.000468 | 0.000393 |
| clip fraction | 0.004061 | 0.004224 | 0.004282 |
| ratio mean | 1.000341 | 1.000015 | 1.000321 |
| grad norm | 0.4189 | 0.4947 | 0.4718 |
| Think Raw N | 2.9853 | 2.5322 | 2.0410 |
| Think Grounded N | 2.5864 | 1.9886 | 1.2324 |
| Think Coverage | 0.8579 | 0.7928 | 0.6243 |
| Think S_N | 0.8272 | 0.7064 | 0.5312 |
| Think D_A | 0.8086 | 0.8075 | 0.8582 |
| Think unique valid target A | 8.2040 | 8.3939 | 8.8047 |
| Think primary zero-std rate | 0.3897 | 0.4091 | 0.5156 |
| NoThink Gold-A-or-better candidate rate | 0.2415 | 0.2808 | 0.2276 |
| NoThink wrong-domain rate | 0.0057 | 0.0112 | 0.0009 |

注意：step 1000 前窗口的 Think Beam invalid output rate 约 2.39%，随后在
step 1500 前窗口降至约 0.018%，最终窗口约 0.36%。请判断这是否会影响对
checkpoint-1000 的选择，以及还需要什么样的原始异常分布证据。

## 固定 Probe 证据

固定 Probe 只有 4 个 domain group，应视为纵向诊断，不是总体显著性证据。

| Probe step | Think reward | NoThink reward | Raw N | Grounded N | Coverage | D_A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.7305 | 0.0000 | 3.8125 | 3.5625 | 0.9375 | 0.7969 |
| 1000 | **4.2656** | 0.1562 | 2.7500 | 2.3750 | **0.8854** | 0.8125 |
| 1400 | 3.1211 | 0.1094 | 2.1875 | 1.6250 | 0.7292 | 0.8438 |
| 1600 | 3.7656 | 0.4062 | 2.1875 | 1.4375 | 0.6510 | 0.8125 |
| 2200 | 3.3047 | 0.5625 | 1.8750 | 1.0000 | 0.5833 | 0.8438 |
| 2316 | 2.9023 | 0.2031 | 2.1875 | 1.2500 | 0.6146 | 0.7891 |

prod domain 是最值得单独审查的证据：

- Probe Think reward：step 1000 的 8.0625 -> final 的 4.0000。
- Probe Coverage：0.8542 -> 0.3333。
- Probe D_A：1.0000 -> 0.8438。
- 外部 recommendation prod：0.1734 -> 0.1530，是四个推荐域中最大的下降。

## 完整 DSR Pilot200 的边界

- 完整 DSR 的 200-step training-stream Coverage overall 约 0.8806，四个阶段
  没有表现出单调 collapse。
- 但该 forensic 的最终判断是 `INSUFFICIENT EVIDENCE`，因为缺少完整匹配的
  candidate/group baseline 记录，也没有完整 epoch 或外部评测。
- Pilot 中存在一个直接 loophole-shaped case：Raw N=4、Grounded N=2、
  Coverage=0.5、S_cot=1.0，仍获得正 auxiliary advantage。
- 因此不能从 Simple 的失败直接推出“恢复完整 DSR 的全部公式一定有效”。

## 请重点回答

1. Simple step 1000 的 1.3440 应如何定性：真实的早期收益、评测噪声，还是
   二者都可能？还缺什么最小证据才能判断？
2. step 1000 后的回落更符合哪种机制：兴趣数量下边界收缩、grounding 漂移、
   correctness-agnostic diversity、NoThink rescue 副作用、普通过拟合，或其他？
   请按证据强弱排序，而不是只选一个。
3. `S_N` 在 N=2/3/4 上完全同分，是否会自然诱导模型靠近 N=2？现有数据能
   支持到什么程度，哪些部分仍只是机制推断？
4. D_A 和 unique valid target A 后期保持或提高，但外部推荐下降，这能否说明
   “多样性没有转化为正确性”？请指出可能的替代解释。
5. 结合完整 DSR pilot 的 coverage loophole，下一版应当：
   - 直接回到完整 DSR；
   - 做 `Simple + grounding-aware signal`；
   - 只做早停而不改公式；
   - 或采取其他最小改动？
   请给出排序和理由，不要一次改变多个无法归因的因素。
6. checkpoint-1000 是否应成为当前提交/推理候选？在复测前有哪些风险需要
   明示，尤其是单次外评噪声和 Beam invalid 的短时尖峰？
7. 下一轮最小实验应保存哪些 checkpoint、最多训练多少 step、使用哪些固定
   Probe 预警和停止条件？请区分“由本次结果后验提出的启发式阈值”和可以
   正式预注册的判据。
8. 哪些结论当前绝对不能宣称？请列出至少 5 条，防止研究记录过度解读。

## 期望输出格式

请按以下结构回答：

1. 一句话总判断。
2. 证据等级表：结论 / 支持证据 / 反证或替代解释 / 置信度。
3. 对三个 checkpoint 和两个基线的选择建议。
4. 对 Simple 公式各组件的机制判断。
5. 下一轮“最小可归因实验”方案，只允许一个核心算法变量变化。
6. 预注册指标、checkpoint、PASS/WARN/STOP 条件。
7. 明确列出不能从当前数据推出的结论。

请保持批判性。如果你认为现有结论或建议不成立，请直接指出，并说明需要哪项
最小新增证据才能改变判断。
