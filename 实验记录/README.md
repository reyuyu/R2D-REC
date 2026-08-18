# 实验记录

本目录记录 OneReason 多任务 SFT 的后续优化方案、实现状态、测试方法和实验结果。

## 当前状态

- [实验一：Lagged GradNorm-lite](./实验A_GradNorm-lite.md)：已实现并通过单卡、双卡及真实 Qwen3/LoRA 梯度路径测试；rank16 双卡正式实验已启动。
- [实验 A0：低学习率、低 LoRA Dropout 的 GradNorm 对照](./实验A0_GradNorm低学习率低Dropout.md)：实验 A 的联合超参数变体，作为历史对照保留。
- [实验二：GradNorm-lite + 局部 LoRA 梯度冲突投影](./实验B_GradNorm-Ortho-LoRA.md)：已实现并通过实验 A 回归、单卡算法和双卡全链路 smoke test；正式训练因门控下长期未触发有效投影，已于 2026-08-03 主动终止。
- [实验三：Action Select 历史约束与长度平衡](./实验C_Action-Select历史约束.md)：第一版已实现，使用完整 SID 动态 Trie、Continue/Stop 平衡、warmup 与 Action CE 比例上限；不拆分顶层任务，不增加 forward/backward。
- [实验三 C-fast：Action Select 向量化优化](./实验C-fast_Action-Select向量化优化.md)：已完成数学/梯度等价测试、单/双卡 smoke、CUDA profiler 和真实 8B 20-step 短跑；保留 legacy 默认路径，使用独立配置与输出目录。
- [实验三 C1：Trie 优先消融](./实验C1_Trie优先消融.md)：正式训练与 checkpoint 评测已完成；1000 到 2500 checkpoint 总分由 1.1824 降至 1.1346，判定为负向消融，不作为后续训练基线。
- [实验三 C2：全词表 Top-K 非法 SID 惩罚](./实验C2_全词表TopK非法SID惩罚.md)：已完成 5200 步训练与 checkpoint 评测；关闭 C/C1 的 allowed-mass 与 Continue/Stop 项，复用动态完整 SID 合法集，在全词表 Top-5 中压低非法竞争 token。
- [实验三 C3：Top-K 非法 SID 与 SID 加权](./实验C3_TopK非法SID与SID加权.md)：在 C2 上叠加全局归一化 SID 加权 CE；rank16 双卡训练中，已记录 1000 checkpoint 结果。
- [实验 E：四任务 GradNorm（无 World）](./实验E_四任务GradNorm无World.md)：已通过单元测试、双卡 smoke 与配置校验，并在 GPU 0/1 正式训练；将 user 拆分为 user_action/user_chain，删除 world，每步固定 {2,2,2,2} 调度，四任务 GradNorm 权重归一化，保留 zero-weight 的 Action 四项诊断监控。

- [实验 Alpha：训练监控与泄漏安全验证](../baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_监控优化.md)：基于清洗后的 `alpha-jiankong`，建立 group-safe 的 train98/dev2 切分；保留 Native SID8 训练目标，新增只读四任务 loss、推荐 teacher-forcing 指标、每 100 step 固定开发集 probe 与 epoch-end full-dev。验证使用 inference-only sidecar，不改变训练梯度或优化器状态。
- [实验 Alpha-mini：组合任务池 R32 两 epoch 训练](../baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_Mini_R32_2E.md)：记录 `/data/lf_data_versions/task_pools` 下懂用户/懂推荐/懂物料三个 alpha_mini task pool 的构造、49,490 行组合合同和 aggregate `1.2861` 及 11 项原始分项。
- [实验 GR_REC_v1：Recommendation Multi-Positive GRPO](../baselines/native_source_domain_r32_v3/docs/experiment_GR_REC_v1.md)：BETA SFT Adapter 上的首个正式 recommendation-only GRPO；2,316 steps 已完成。Step 1500 外部评测 `1.3510` 为当前最佳候选，Step 1000/2000 均与 `1.3313` 基线处于正常波动范围。

## GRPO 系列（当前重点）

### 比较基线

GRPO policy 从 BETA-baseline Adapter 初始化。历史 Epoch 2 首次评测为 `1.3246`；同一模型多次评测中较高的一次为 `1.3313`。考虑外部评测约有 `±0.01` 正常波动，GRPO 系列统一采用较高的 `1.3313` 作保守比较基线，不覆盖原始记录，也不把 `0.01` 内差异解释成确定收益。

### 实验链路

| 实验 | 动机与单一问题 | 冻结项 | 当前结论 |
| --- | --- | --- | --- |
| BETA-baseline | 提供已经完成 SFT 的统一初始 policy；用较高复测值避免夸大 GRPO 增益 | 数据、LoRA 和 SFT 训练结果均不变 | GRPO 比较基线 `1.3313` |
| GR_REC_v1 | 只用 recommendation outcome 做 group-relative 优化，检验生成结果奖励能否提高懂推荐，同时保持懂物料、懂用户和懂世界 | reward、Think G=4、NoThink G=8、Beam32、sampler、route weight、GRPO/PPO 数学和数据顺序 | Step 1500 `1.3510` 为阶段性正向信号；需复测，不能按最终 step 自动选模 |
| 后续 CoT 多样性实验（待立项） | 单独处理后期 CoT 变短、兴趣点收缩，不与稀疏 reward 改动混合 | 首先保持 NoThink、sampler 与训练数据不变 | 尚未配置或启动 |
| 后续稀疏 reward 实验（待立项） | 单独提高 Think/NoThink 难样本的组内区分性，减少 zero-std rollout | 首先保持 CoT 生成与 Beam32 语义不变 | 尚未配置或启动 |

### GR_REC_v1 结果摘要

| 模型 / Step | 总分 | 懂推荐合计 | 相对基线 | 判读 |
| --- | ---: | ---: | ---: | --- |
| BETA-baseline | `1.3313` | `0.6594` | - | 多次评测中的较高基线 |
| Step 1000 | `1.3270` | `0.6563` | `-0.0043` | 正常波动范围内 |
| **Step 1500** | **`1.3510`** | **`0.6800`** | **`+0.0197`** | 当前最佳，优先重复评测 |
| Step 2000 | `1.3326` | `0.6607` | `+0.0013` | 正常波动范围内 |

Step 1500 的提升主要集中于懂推荐，符合实验直接优化目标；Step 2000 又回落至基线水平。结合固定 Probe 中后期 CoT 兴趣覆盖收缩和 zero-std 上升，当前选模原则是“外部评测 + 生成质量 + 有效信号密度”联合判断，而不是默认使用最晚 checkpoint。完整原始 11 项和证据见 [GR_REC_v1 详细记录](../baselines/native_source_domain_r32_v3/docs/experiment_GR_REC_v1.md)。

## 实验关系

实验 A 处理三个主任务的梯度尺度不平衡，`world` 权重固定为 1。实验 B 复用实验 A 的 Hook、任务向量、DDP 同步、动态权重和 checkpoint，只在持续负冲突时对选中的高层 LoRA-B 梯度做局部 PCGrad 投影。

实验 A0 完整复用实验 A 的训练语义，同时将学习率和 LoRA dropout 分别调整为 `1e-4` 和 `0.01`。由于两个超参数同时改变，A0 用于比较联合配置效果，不用于单独归因某一个参数。

两组实验保持相同的 rank16、数据、balanced_40、seed、max_steps 和 batch 语义，并使用不同的 `output_dir`，可直接并行比较。实验 B 不增加任务专属 adapter，不修改 LoRA forward 或推理结构。

实验三建立在实验 A 配置上，只修改 `user/action_nocot` 的 raw loss。Action 辅助项在当前 user GradNorm 权重之前加入，所以继续作为 user 梯度的一部分参与实验 A 的尺度平衡，不改变顶层任务调度。

C-fast 不改变实验三的目标，只把重复的 full-vocabulary denominator 和逐 SID Trie 归约改为 microbatch 级去重、分块和分组批处理。是否将其设为未来默认实现，取决于等价性、显存、双卡 smoke 和真实 8B 短跑结果。


## 实验D：SID 加权 SFT CE

- 状态：已完成 5200 步训练与 checkpoint 评测。
- 配置：[onereason_lora_2gpu_balanced40_r16_gradnorm_sid_weight8.yaml](../configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_sid_weight8.yaml)
- 记录：[实验D_SID加权SFT.md](实验D_SID加权SFT.md)
- 内容：对全部监督的 `<s_a_*>`、`<s_b_*>`、`<s_c_*>` token 使用归一化 CE 权重 8；不改变 GradNorm、Ortho、packing 或 Action 辅助损失的接入顺序。
- [rec_A 与 recB：四任务调度对照](./实验recA_recB_四任务调度.md)：记录 `rec_A` 的当前训练进度，以及 `recB` 的 `1:3:3:1` 调度、两次 step 22 显存异常和配置对照结果。
- [REC 系列：自适应 Packing 与四任务训练](./实验recC_recD_recE_REC_F_自适应Packing.md)：汇总 recC 的 8K BFD、recD 的 cost-aware smoke、recE 的自适应 pack 长度验证，以及 REC_F 的 8K 正式配置。
- [实验 REC_G2：Recommendation No-think Multi-Positive Prefix-Trie Loss](./实验REC_G2_Recommendation-No-think-Multi-Positive-Trie.md)：基于 REC_G1，只替换 recommendation/nocot 的 s_a/s_b/s_c single-gold CE；CPU 回归与 50-step 全量 V3 双卡 smoke 已通过，正式训练已启动。


## 原生参考基线阶段

- [实验 Baseline：NSD-R32-V3-2E-GC04-4GPU](./实验Baseline_NSD-R32-V3.md)：隔离复刻 material-domain 原生 SFT 路线；四卡 8K neat packing、LoRA r32、SID 权重 8、两 epoch、0.4GC。该路线保留物料、用户 Action、用户 Chain、推荐四类数据，但不使用 macro trainer、GradNorm 或推荐辅助损失；四项任务 loss 仅作训练观测。
- [实验 BATA-baseline 纯净版](./实验BATA-baseline纯净版.md)：复现包数据合同下仅替换 active BETA 懂用户的四卡原生 baseline；2 epoch 已完成，历史评测 `1.3246`，GRPO 对照用较高复测值 `1.3313`。
- [实验 BETA-fenpei：Set-PU + PackRatio](../baselines/native_source_domain_r32_v3/docs/experiment_BETA-fenpei.md)：当前四卡正式 Native baseline。使用 `BETA_material_aligned_v1`，保持三路物料 loss 合同；推荐最终 SID 使用 Set-PU scalar replacement，以 `20/45/20/15` pack ratio 调度 material/recommendation/user_action/user_chain；2 epoch 共 1052 optimizer steps，候选 teacher-forcing 指标每 50 step 记录一次。
