# R2D-REC 补充探索与诊断

[项目首页](../../README.md) · [方法主线](METHODS.md) · [复现指南](REPRODUCTION.md)

以下整理仓库中未统一纳入最终方案的独立实验与诊断。已有源码不等于已证明收益；小样本 probe、CPU 验证、训练完成和外部评测分别标明。探索不自动进入默认训练链，本页不列中间成绩。

## 奖励与信用分配探索

| 编号 | 方案 | 机制与价值 | 已有证据与边界 | 研究定位 |
| --- | --- | --- | --- | --- |
| 1 | 兴趣覆盖复合奖励（Composite Interest） | 将推荐结果奖励与 CoT 兴趣覆盖/质量结合；用字符 bigram F1 和一对一匹配，减少重复兴趣反复领分 | 有离线校准、GPU 预检与固定域 12-step smoke PASS；不足以证明最终任务收益，部分旧 probe 存在训练重叠限制 | 推理质量奖励探索 |
| 2 | SID 逐层信用分配（FineGrained v6A） | 为 A/B/C 分别构造组内优势；只有前缀正确才给后续层信用，降低不同层混成一个奖励的歧义 | README 明确为 CPU 实现，未正式 GPU 训练；扩展 probe 的重叠限制需说明 | 细粒度信用分配实现 |
| 3 | 稀疏奖励信号补救（DSR） | Think 辅助信号独立归一化；NoThink 仅在整个组全零奖励时补充错误 A 的 unlikelihood 信号 | 200-step pilot 完成，但法证报告结论为 INSUFFICIENT EVIDENCE，缺少足够配对数据证明改善 | 稀疏奖励研究，收益待验证 |
| 4 | Exact-Clamp 与条件层级奖励 | 防止完整命中的候选因组内比较收到负优势；结合前缀条件信用和全零组的 Gold-A 信号 | 最新 Text-Domain 放置版本只通过 CPU 检查，尚未重新做 GPU 审计或训练；不可借用旧版本结果替它背书 | 奖励一致性约束 |
| 5 | 历史复制惩罚课程（OfficialAntiCopy） | 在官方域前缀生成接口下，对历史内候选采用阶段化折扣，尝试释放候选容量 | CPU 实现，未正式训练；历史内 Gold 真实存在，降低复用率不自动等于提高正确召回 | 候选分布控制 |
| 6 | 双接口 SID 优化（ExactSharpen v4） | 同一 CoT 后分别生成 Free 与 Official SID 组，显式覆盖训练自由边界和评测固定前缀 | Code-only，README 明确无 GPU 预检产物 | 双接口训练实现，效果待验证 |

对应文件入口：

1. Composite Interest：[说明](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_composite_interest_v1/README.md) · [Trainer](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_composite_interest_v1/composite_trainer.py) · [兴趣匹配](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_composite_interest_v1/interest_metric.py) · [固定域 Smoke12 结果](../../baselines/native_source_domain_r32_v3/grpo/results/gr_rec_think_composite_interest_v1_fixed_domain_smoke12_20260823.json)。
2. FineGrained v6：[说明](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_official_finegrained_v6/README.md) · [Trainer](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_official_finegrained_v6/official_finegrained_trainer.py)。
3. DSR：[说明](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/README.md) · [训练目标](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/dsr_objectives.py) · [Pilot200 法证报告](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_dsr_v1/results/pilot200_forensic_20260818/GR_REC_DSR_PILOT200_FORENSIC_REPORT.md)。
4. Exact-Clamp：[最新合同](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_clamp_v1/README.md) · [优势约束](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_clamp_v1/think_exact_clamp.py) · [Trainer](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_clamp_v1/think_exact_clamp_trainer.py)。
5. OfficialAntiCopy Mixed v5：[说明](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_official_anticopy_mixed_v5/README.md) · [Trainer](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_official_anticopy_mixed_v5/official_anticopy_mixed_trainer.py)。
6. ExactSharpen v4：[说明](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_sharpen_v4/README.md) · [Trainer](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_exact_sharpen_v4/exact_sharpen_trainer.py)。

## 边界适配、约束与历史对照

| 编号 | 方案 | 机制与价值 | 已有证据与边界 |
| --- | --- | --- | --- |
| 7 | NoThink Frontier 首错归因 | 只惩罚 SID 层级链中首次失败的位置，结合严格格式和正向里程碑信用 | 已完成完整训练；固定 probe 显示共享 adapter 对 Think 路产生干扰，未跑外部 benchmark，不支持“兼顾双路”的结论 |
| 8 | Bridge-Inside 边界适配 | 把自然语言 Bridge 移到 think 结束前，仅监督 Bridge 和闭合边界，使后续 SID 接口更接近评测 | 有独立 runner、损失 mask 和验证流程；本轮未核到可支撑外部收益的独立结果 |
| 9 | Bridge-to-Bare SID 蒸馏 | 冻结带 Bridge 的 teacher，在去 Bridge 的 student 上按 A/B/C SID token family 蒸馏 | 已训练并记录对照；没有解决目标接口差距。适合作为负结果和机制分析 |
| 10 | 重复 CoT 降权 | 同一多正例组重复的 CoT 按 0.5/N 归一，避免重复文本占据过多监督权重 | 已完成两轮训练；epoch 1 的 teacher-forced 指标与外部生成表现背离，不能宣称整体改进 |
| 11 | Action Select 历史 Trie 与长度约束 | 约束完整 SID 四元组，避免不同历史 SID 的分量拼接；删除已选完整路径，并辅助控制提前停止/重复生成 | 有实现、配置和测试入口；这是训练辅助目标，不能像约束解码一样保证推理时绝对合法 |
| 12 | GradNorm-lite 任务梯度平衡 | 用任务梯度范数与学习速度调整 loss 权重，控制多任务贡献 | 有正式实验，但已有对照未优于基础配置；适合负结果或工程扩展说明 |
| 13 | 局部 LoRA 冲突投影 | 复用正常 backward 的任务梯度，在部分 LoRA-B 参数做 PCGrad 风格投影，保留未跟踪残差 | 单卡/双卡测试通过；原正式运行因双阈值门控很少激活有效投影而停止，不是完成验证的增益方案 |
| 14 | Positive A0 | 保留正信号 Think 子集，将仅 A 命中的奖励降为零，尝试减少粗粒度前缀捷径 | 有独立配置与实现，未核到正式运行；可能进一步增加奖励稀疏性 |
| 15 | 固定 CoT 的生成器/解码器交叉诊断 | 固定推理文本，分别交换模型与答案边界，区分推理变化、解码排序和 Bridge 接口影响 | 已有诊断脚本和记录，是分析方法，不直接构成一个训练收益组件 |

对应文件入口：

7. Frontier：[说明](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_nothink_only_frontier_v1/README.md) · [信用分配](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_nothink_only_frontier_v1/frontier_credit.py) · [完整训练法证](../../baselines/native_source_domain_r32_v3/grpo/results/gr_rec_nothink_frontier_v1_formal_e1_forensic_20260822.md)。
8. Bridge-Inside SFT：[说明](../../baselines/native_source_domain_r32_v3/boundary_adapt/bridge_inside_sft/README.md) · [训练入口](../../baselines/native_source_domain_r32_v3/boundary_adapt/bridge_inside_sft/train_transition.py)。
9. Bridge-to-Bare KD：[说明](../../baselines/native_source_domain_r32_v3/boundary_adapt/kd/README.md) · [蒸馏损失](../../baselines/native_source_domain_r32_v3/boundary_adapt/kd/bridge_to_bare_kd_loss.py) · [训练入口](../../baselines/native_source_domain_r32_v3/boundary_adapt/kd/train_bridge_to_bare_kd.py)。
10. CoT 重复降权：[实验说明](../../baselines/native_source_domain_r32_v3/docs/experiment_alpha_cot_repeat05n.md) · [配置](../../baselines/native_source_domain_r32_v3/config/train_alpha_cot_repeat05n_4gpu_gc04_2epoch.yaml) · [缓存构建入口](../../baselines/native_source_domain_r32_v3/scripts/build_alpha_cot_repeat_cache.py)。
11. Action 约束：[实验说明](../../实验记录/实验C_Action-Select历史约束.md) · [辅助目标](../../src/llamafactory/train/sft/user_action_auxiliary.py) · [SID Trie](../../src/llamafactory/data/action_select.py)。
12. GradNorm-lite：[实验说明](../../实验记录/实验A_GradNorm-lite.md) · [多任务梯度控制器](../../src/llamafactory/train/sft/multitask_gradient_controller.py)。
13. 局部 LoRA 冲突投影：[实验说明](../../实验记录/实验B_GradNorm-Ortho-LoRA.md) · [投影与梯度控制实现](../../src/llamafactory/train/sft/multitask_gradient_controller.py)。
14. Positive A0：[说明](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_positive_a0_v3/README.md) · [奖励实现](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_positive_a0_v3/a0_reward.py)。
15. 固定 CoT 交叉诊断：[生成器/解码器交叉](../../baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/controlled_generator_decoder_crossover.py) · [官方提示与 Bare 边界对照](../../baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/recommendation_official_prompt_bare_crossover.py) · [Self-CoT 对照](../../baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/recommendation_self_cot_crossover.py)。
