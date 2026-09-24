# R2D-REC：方法与代码映射

[文档导航](README.md) · [复现指南](REPRODUCTION.md) · [仓库目录索引](REPOSITORY_MAP.md)

本文按“识物 → 察行 → 推意 → 择物”介绍方法。下列代码入口保留历史名称；对应关系用于定位实现，不意味着所有旧实验与最终答辩配置完全相同。

<a id="semantic-alignment-sft"></a>
## 01 识物：Semantic Alignment SFT

**目标：使 SID 对应稳定的物料语义，为用户理解与推荐提供基础。**

同一物料可能有多条 caption，其中既有共享语义，也有颜色、年份等偶然属性。答辩方案以 SID 为单位组织描述，提取稳定语义，并结合 canonical / reverse 表达形成双向语义对齐，再与用户、推荐任务共同进行 SFT。

| 内容 | 实现或合同 |
| --- | --- |
| 已对齐物料数据的接入与格式转换 | [build_material_aligned_beta.py](../../baselines/native_source_domain_r32_v3/scripts/build_material_aligned_beta.py) |
| 数据注册 | [register_material_aligned_beta.py](../../baselines/native_source_domain_r32_v3/scripts/register_material_aligned_beta.py) |
| 来源、行数、哈希与路由构成 | [物料对齐数据 manifest](../../baselines/native_source_domain_r32_v3/dataset_beta_material_aligned_v1/manifest.json) |
| 多任务 LoRA SFT 配置 | [train_beta_material_aligned_sid8_4gpu_gc04_2epoch.yaml](../../baselines/native_source_domain_r32_v3/config/train_beta_material_aligned_sid8_4gpu_gc04_2epoch.yaml) |
| 启动入口 | [launch_beta_material_aligned_sid8_4gpu_gc04_2epoch.sh](../../baselines/native_source_domain_r32_v3/scripts/launch_beta_material_aligned_sid8_4gpu_gc04_2epoch.sh) |
| 数据版本导航 | [ONEREASON_DATASET_VERSIONS.md](../../ONEREASON_DATASET_VERSIONS.md) |

**实现边界。** 当前构建脚本接收已有的对齐物料数据，并保留其 system、prompt、response 与顺序；它不是完整的上游 LLM-judge caption 生成器。不要把答辩中的处理示意直接解释为该脚本会重新生成全部语义描述。实际训练的数据数量、来源与版本以 manifest 为准。

后续 [Rec FDR V4.3 full-SFT](../../reproduction/rec_fdr_v43_strictdet/README.md) 使用独立数据重建与训练合同，不能替换本节历史 SFT 后仍视为同一个 parent。

<a id="mch-grpo"></a>
## 02 察行：Marginal-Credit Hybrid GRPO

**目标：在长历史中定位相关行为及其关系，使奖励能够区分具体行为的贡献。**

MCH-GRPO 结合两个信号：全局分支评价整条输出，局部分支评价移除一个预测行为单元前后的奖励差。Action 以 SID 为单元，Chain 以事件为单元，将边际信用映射到相应 token span。

```text
用户历史 + 查询主题
  -> 同一 prompt 的 K 个候选
  -> Global：整条输出奖励 -> 组内相对 advantage
  -> Local：移除行为单元前后的奖励差 -> 局部 token credit
  -> 加权组合目标 -> 更新共享 adapter
```

| 内容 | 实现或合同 |
| --- | --- |
| 数据构建与划分 | [build_gr_user_v1.py](../../baselines/native_source_domain_r32_v3/grpo/user/scripts/build_gr_user_v1.py) |
| Action / Chain 奖励 | [user_action_reward.py](../../baselines/native_source_domain_r32_v3/grpo/user/scripts/user_action_reward.py)、[user_chain_reward.py](../../baselines/native_source_domain_r32_v3/grpo/user/scripts/user_chain_reward.py) |
| 行为单元的边际信用 | [user_marginal_credit.py](../../baselines/native_source_domain_r32_v3/grpo/user/scripts/user_marginal_credit.py) |
| 局部目标 | [user_mc_objective.py](../../baselines/native_source_domain_r32_v3/grpo/user/scripts/user_mc_objective.py) |
| 全局与局部混合目标 | [user_mc_hybrid_objective.py](../../baselines/native_source_domain_r32_v3/grpo/user/scripts/user_mc_hybrid_objective.py) |
| 四卡 candidate-parallel runner | [run_mc_user_formal_hybrid_k4_ddp_v1.py](../../baselines/native_source_domain_r32_v3/grpo/user/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py) |
| 已选运行的配置 | [mc_user_hybrid_strongparent_lr3e7_200.json](../../baselines/native_source_domain_r32_v3/grpo/user/configs/mc_user_hybrid_strongparent_lr3e7_200.json) |

一般形式为 `L = λ_global × L_global + λ_local × L_local`。本分支的 Hybrid 默认值及已选 strong-parent 配置为 `λ_global=1.0`、`λ_local=0.3`。答辩第 14 页示意式为 `L_local + 0.2 L_global`，与该配置不同；执行时应读取配置，不能根据示意图修改现有目标。

当前 Hybrid 的 global 分支为 group-normalized sequence objective，具体实现不能因展示名称“Vanilla GRPO”而被推断为包含额外 reference KL 或 PPO ratio。历史 `GR_USER_v1` 的局部处罚路线与 `MC_USER Hybrid` 的边际信用路线也应区分。

<a id="orr-grpo"></a>
## 03 推意：Outcome-Reinforced Reasoning GRPO

**目标：让 CoT 的优化信号来自其能够支持的推荐结果。**

对同一用户输入采样多条 CoT，在各自上下文后生成 Beam32 候选，以候选对正确 SID 集合的覆盖情况给推理分配奖励。分层反馈区分 A、AB、完整 SID 等匹配程度；多正样本分组提供比单个目标更丰富的结果信号。

```text
多正样本推荐组
  -> G4 CoT rollout
  -> 每条 CoT 后 Beam32 候选
  -> 候选结果奖励
  -> CoT 组内相对优势
  -> 推理动作上的优化
```

| 内容 | 实现或合同 |
| --- | --- |
| 多正样本数据构建 | [build_rec_mp_grpo_v2.py](../../baselines/native_source_domain_r32_v3/grpo/scripts/build_rec_mp_grpo_v2.py) |
| 数据来源与域特定 prompt | [DATASET_PROVENANCE.md](../../reproduction/final_chain_20260901/docs/DATASET_PROVENANCE.md) |
| Think reward、路由数据与 sampler | [grpo_trl_trainer.py](../../baselines/native_source_domain_r32_v3/grpo/scripts/grpo_trl_trainer.py) |
| SID 解析、分层 credit 与 Beam reward | [grpo_sid.py](../../baselines/native_source_domain_r32_v3/grpo/scripts/grpo_sid.py) |
| Beam 域处理 | [grpo_beam_domain.py](../../baselines/native_source_domain_r32_v3/grpo/scripts/grpo_beam_domain.py) |
| 历史推荐 GRPO runner | [run_grpo_trl_train.py](../../baselines/native_source_domain_r32_v3/grpo/scripts/run_grpo_trl_train.py) |

**映射边界。** 上述 `GR_REC_v1` 的 Think 分支提供与 ORR 对应的“CoT → Beam outcome → 推理奖励”机制，但整个历史 runner 同时包含 NoThink 路由，不能直接称为纯 ORR-only 训练。`think_reward()` 还实现了前缀去重和几何衰减，具体奖励规则以代码为准。

答辩中的决策侧“冻结”表示该阶段不直接对 Beam SID 动作施加对应的决策 loss；它不自动意味着存在独立且参数冻结的 decoder。使用共享 adapter 时仍需通过对照验证推理更新对决策行为的影响。

<a id="joint-grpo"></a>
## 04 择物：Joint Reasoning & Decision GRPO

**目标：同时优化推理与最终 SID 动作，让推理质量与推荐决策相互配合。**

本模块对应历史工程名 `GRPO-TK / GR_REC_ThinkSample8_FullSID_v3` 的双目标实现：

```text
一个业务 group
  -> G4 CoT
  -> 每条 CoT 后独立采样 G8 FullSID continuation
  -> 每个 SID G8 独立计算 advantage
  -> 每组 8 个 SID reward 求和，形成一个 CoT reward
  -> 四条 CoT 再计算全局 G4 advantage
  -> L_total = L_cot + L_sid
```

| 内容 | 实现或合同 |
| --- | --- |
| 拓扑与采样合同 | [Sample8 FullSID README](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/README.md) |
| 训练入口 | [run_sample8_fullsid_train.py](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/run_sample8_fullsid_train.py) |
| 启动脚本 | [launch_sample8_fullsid_train.sh](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/launch_sample8_fullsid_train.sh) |
| 合同检查 | [test_sample8_fullsid_contract.py](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/test_sample8_fullsid_contract.py) |
| 复现步骤 | [reproduce_GRPO_TK.md](../reproduce_GRPO_TK.md) |

SID 的采样上下文是 `prompt + sampled CoT through </think>`。模型自行生成后续内容，不预填目标域或自然语言 Bridge；解析器寻找第一个完整的 `domain + A + B + C`。CoT loss 只覆盖 CoT 动作，SID loss 只覆盖这个完整 SID 的四个 token。四组 G8 各自归一化，不能合并成一个 G32。

这里的 Joint 是一次训练中联合两个动作目标；“先 ORR、再 Joint”则是训练阶段之间的渐进式组织。两者不是同一概念。历史 checkpoint 的真实继承关系见[复现指南](REPRODUCTION.md)。

<a id="bridge"></a>
## 探索：Bridge 与生成接口

答辩最后讨论了 CoT 与答案之间的自然语言 Bridge。训练时出现 Bridge、评测时直接给定域前缀，可能使候选空间发生变化。相关实验用于分析模板、上下文及模型行为，不作为已经证明的普适正则化机制。

| 研究入口 | 内容 |
| --- | --- |
| [boundary_adapt/diagnostics](../../baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics) | Bridge 位置、固定 CoT、生成器/解码器交叉与历史复用诊断 |
| [Bridge memory probe 摘要](../../baselines/native_source_domain_r32_v3/boundary_adapt/results/recommendation_bridge_memory_quick_probe/summary.md) | Bridge 与候选行为的既有观测 |
| [Bridge inside SFT](../../baselines/native_source_domain_r32_v3/boundary_adapt/bridge_inside_sft/README.md) | 边界适配的独立实验 |
| [Bridge-to-bare KD](../../baselines/native_source_domain_r32_v3/boundary_adapt/kd/README.md) | Bridge 到 bare 接口的独立蒸馏探索 |

这些入口属于探索分支。运行任何一项之前，均需单独确认其 parent、数据与输出目录。
