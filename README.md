# OneReason 多任务 SFT 工程

本仓库用于微调 `OpenOneRec/OneReason-8B-pretrain-competition`，基于 LLaMA-Factory 扩展多任务 SFT、GradNorm、梯度冲突监控、Action Select 辅助目标和可复现的数据版本管理。

仓库只保存代码、配置、测试、实验记录和数据版本元信息，不保存模型权重、JSONL 数据、日志、预测结果、检查点或任何凭据。

## 项目导航

- [多任务设计说明](./ONEREASON_MULTITASK.md)
- [数据版本管理说明](./ONEREASON_DATASET_VERSIONS.md)
- [实验记录与评分](./实验记录/README.md)
- [竞赛评测说明](./README_ONEREASON_COMPETITION.md)
- [数据集注册表](./data/dataset_info.json)
- [数据版本清单](./data/onereason_dataset_versions.json)

## 新 Baseline 系列

当前主线进入独立的 **Native Source-Domain R32 V3** baseline 系列。它在隔离的 LLaMA-Factory `01398eb` 环境中复刻 source/domain-aware 原生 SFT，和仓库原有 macro trainer、GradNorm、Action 辅助目标、REC packing 系列严格分开。目的不是继续叠加旧多任务算法，而是在固定训练配方与物料合同下，建立可验证的推荐损失消融基线。

### 系列结构

1. **NSD-R32-V3**：四张 A800、8K neat packing、LoRA r32/alpha64/dropout0.05、全局 batch 64、两 epoch、cosine LR `2e-4`、0.4GC（14/36 decoder block）、FA2、Liger 与 BF16。
2. **BETA-MATERIAL-ALIGNED-SID8**：锁定 `BETA_material_aligned_v1`，将懂物料严格对齐压缩包路线；训练数据不含 world。
3. **BETA-SETloss（Set-PU）**：在锁定物料合同上，仅替换 recommendation 最终 SID 的 one-hot CE；以 observed-positive set 为正例集合，并将同层未观测 SID 作为 alpha=0.05 的弱负项。

该系列的完整代码、配置、测试和运行说明见 [baseline 目录](./baselines/native_source_domain_r32_v3/README.md)。数据 JSONL、模型、日志和 checkpoint 不提交仓库。

### 当前正式实验：BETA-fenpei

当前 run 为 `BETA-fenpei`，母版为 `BETA-MATERIAL-ALIGNED-SID8-R32-2E-GC04-4GPU`。

- 数据集：`onereason_beta_material_aligned`，目录为 `/data/lf_data_versions/alltrain/BETA_material_aligned_v1`；正式启动前必须通过 manifest、投影摘要、三路数量、四域数量/权重和实际 loss route 的 preflight。
- 物料合同：`material_sample=100000` 使用普通 token 1、SID/domain token 8 与四域权重；`sid_bucket_canonical_no_think=11298` 使用所有 response token 4、无域权重；`sid_bucket_reverse=29586` 使用普通 1、SID/domain 8、无域权重。
- BETA-SETloss：仅替换 recommendation **final SID a/b/c** 的 one-hot CE。已观测正例为 `P`，同层未观测 SID 为 `U`，其他层 SID 与普通 token 为 `O`；使用真实标量 Set-PU：`log(sum_P exp(z) + 0.05*sum_U exp(z) + sum_O exp(z)) - logsumexp(z[P])`。它直接使用 PyTorch autograd，不是旧的 custom-backward surrogate；仍是 replacement，SID/domain weight 仍为 8，原 SID8 分母不变。
- PackRatio：顶层任务按 `material/recommendation/user_action/user_chain = 20/45/20/15` 目标比例调度；运行时以 `e_share_*` 记录实际 pack 消费占比。
- 正式 YAML：[BETA-fenpei 4 GPU 2 epoch](./baselines/native_source_domain_r32_v3/config/train_rec_pu_beta_material_aligned_r32_b005_2epoch_packratio_20452015_candidate_metrics.yaml)。启动脚本会显式设置 `GLOBAL_ITEM_WEIGHT=8`、`MATERIAL_DOMAIN_MANIFEST` 与 `NATIVE_GC_FRACTION=0.4`。
- 训练规模：33,616 packed samples，526 optimizer steps/epoch，2 epoch 共 1,052 steps；每个 epoch 保存一次 checkpoint。
- 监控：`loss`、`grad_norm`、learning rate、`material / recommendation / user_action / user_chain` 四项 task loss，以及 REC-PU 的 segments、a/b/c positions、singleton/multi-positive 数量和平均正例数。每 50 step 额外记录 teacher-forcing 候选 hit/coverage/chain 指标，复用已有 logits，不增加模型 forward。该路线不使用 GradNorm，因此不记录 GradNorm 权重或任务梯度冲突。
- 验证：P/U/O、prefix metadata、replacement/denominator、SID8 单次加权回归通过；Set-PU 在真实 BETA batch 上与同一 scalar reference 的 LoRA gradient cosine/norm ratio 均为 `1.0000`；40-step 四卡 smoke 中 recommendation loss 未复现旧 surrogate 的 step20 后持续反弹。正式记录见 [实验 BETA-fenpei](./baselines/native_source_domain_r32_v3/docs/experiment_BETA-fenpei.md)。

### Alpha：监控与泄漏安全验证

`ALPHA-JIANKONG-MONITOR` 保持 Native SID8 目标和训练配方不变，仅对清洗后的 `alpha-jiankong` 创建 group-safe 的 train98/dev2 切分，并增加训练侧四任务 loss、推荐 teacher-forcing 监控与验证 sidecar。固定开发集 probe 每 100 step 运行一次，完整 dev 仅在 epoch 末运行；二者均在 `inference_mode` 下执行、恢复 RNG/训练态，且不参与反向或优化器更新。

- [Alpha 实验记录：指标、验证集与开销](baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_监控优化.md)
- [Alpha SID8 cache 修复与 Epoch2 CoT/No-think 观测](baselines/native_source_domain_r32_v3/docs/experiment_ALPHA_SID8_cache_fix.md)
- [正式 4 GPU 配置](baselines/native_source_domain_r32_v3/config/train_alpha_jiankong_monitor_validation_4gpu_gc04_2epoch.yaml)

### 与旧 REC 系列的关系

旧 REC 系列仍是 macro training + 四任务 GradNorm 的独立路线，覆盖 BFD、coverage/deficit 调度与 cost-aware packing。它的结果用于历史比较，不与 Native baseline 的 loss、batch 语义或 checkpoint step 直接横比。

## REC 系列实验

REC 系列使用 `material / user_action / user_chain / recommendation` 四任务，保留四任务 GradNorm 和 SID 加权 CE，并逐步验证 8K BFD、coverage/deficit 调度、cost-aware 分区和按子任务自适应 pack 长度。

- [REC 阶段实验记录](实验记录/实验recC_recD_recE_REC_F_自适应Packing.md)
- [rec_A / recB 调度对照](实验记录/实验recA_recB_四任务调度.md)
- [REC_F 正式配置](configs/onereason/onereason_lora_2gpu_recF_r16_gradnorm_sid_weight8_v2_dual_8k_gc075.yaml)
- [数据版本管理](ONEREASON_DATASET_VERSIONS.md)

REC_F 的 8K BFD pack epoch 共 45,744 个 pack，每个 macro-step 使用 8 个 global pack，因此一个 pack epoch 为 5,718 个 macro-step。训练输出、checkpoint、日志和原始数据不提交到仓库。

## 训练任务

| 顶层任务 | 子任务 | 内容 |
| --- | --- | --- |
| `material` | `cot`、`nocot` | 广告、商品、直播、视频的 SID 与描述理解 |
| `user` | `action_nocot`、`chain_cot`、`chain_nocot` | 用户行为选择与多跳逻辑链 |
| `recommendation` | `cot` | 用户画像与内容推荐理解 |
| `world` | `cot`、`nocot` | 通用世界知识 SFT |

默认多任务训练每个 macro-step 使用 8 个全局 microbatch；双卡 DDP 时每个 rank 处理其中 4 个。task-wise packing 只在同一子任务内进行，segment 之间使用独立的位置和注意力边界。

实验 E 提供可回退的四任务布局：`material`、`user_action`、`user_chain`、`recommendation`，不训练 `world`，配置见 `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_expE_user_split_no_world.yaml`。

## 实验索引

| 实验 | 主要改动 | 记录 |
| --- | --- | --- |
| A | Lagged GradNorm-lite 与梯度监控 | [实验 A](./实验记录/实验A_GradNorm-lite.md) |
| A0 | 学习率、LoRA dropout 与数据版本消融 | [实验 A0](./实验记录/实验A0_GradNorm低学习率低Dropout.md) |
| A1 | `think`/`no_think` 提示补充 | [实验 A1](./实验记录/实验A1_think_prompt-GradNorm.md) |
| B | 局部 LoRA 梯度冲突投影 | [实验 B](./实验记录/实验B_GradNorm-Ortho-LoRA.md) |
| C | Action Select 历史 Trie、完整 SID 去重、继续/终止平衡 | [实验 C](./实验记录/实验C_Action-Select历史约束.md) |
| C-fast | Action Select 辅助损失向量化 | [实验 C-fast](./实验记录/实验C-fast_Action-Select向量化优化.md) |
| C1 | Trie 优先消融，已判定为负向优化 | [实验 C1](./实验记录/实验C1_Trie优先消融.md) |
| C2 | 全词表 Top-5 非法 SID 惩罚 | [实验 C2](./实验记录/实验C2_全词表TopK非法SID惩罚.md) |
| C3 | C2 与 SID 加权 CE 融合 | [实验 C3](./实验记录/实验C3_TopK非法SID与SID加权.md) |
| D | 监督 SID token 归一化加权 CE | [实验 D](./实验记录/实验D_SID加权SFT.md) |
| E | 四任务 GradNorm，删除 world | [实验 E](./实验记录/实验E_四任务GradNorm无World.md) |
| NSD-R32-V3 | 新 Native Source-Domain R32 baseline 系列 | [baseline 目录](./baselines/native_source_domain_r32_v3/README.md) |
| REC-PU | BETA material-aligned 上的 recommendation positive-unlabeled replacement | [实验 REC-PU](./baselines/native_source_domain_r32_v3/docs/实验REC-PU.md) |
| BETA-fenpei | Set-PU + `20/45/20/15` PackRatio + 低频候选质量指标 | [实验 BETA-fenpei](./baselines/native_source_domain_r32_v3/docs/experiment_BETA-fenpei.md) |

所有 checkpoint 分数、训练状态、已知限制和最终结论以实验记录为准。

## 数据版本管理

新的训练统一使用全量训练数据，不再从训练数据中切出验证集。历史 `*_train98` 和 `dev2` 文件只用于复现实验，不参与新的全量版本。

版本清单使用父版本继承：

```text
raw_all
  -> v1_thought_prompt_all
       -> v2_recommendation_cot_complete_all
            -> v3_material_clean
```

- `raw_all`：服务器 `/data/lf_data` 下的 8 个原始全量子集。
- `v1_thought_prompt_all`：按 CoT/non-CoT 追加 `/think` 或 `/no_think`，不改变样本数量。
- `v2_recommendation_cot_complete_all`：只替换懂推荐，保留含 `【兴趣归纳】`、`【行为模式】`、`【预测总结】` 的样本。
- `v3_material_clean`：示例版本，只覆盖懂物料，其余子集从父版本继承。

实际 JSONL 只保存在服务器，不上传 GitHub。代码、注册别名、样本数和 SHA-256 摘要保存在 [数据版本清单](./data/onereason_dataset_versions.json) 中。

### 初始化与审计

```bash
cd /app/LLaMA-Factory
python scripts/manage_onereason_datasets.py init-raw-all
python scripts/manage_onereason_datasets.py create-thought-prompts \
  --version v1_thought_prompt_all --parent raw_all
python scripts/manage_onereason_datasets.py create-recommendation-cot-complete \
  --version v2_recommendation_cot_complete_all --parent v1_thought_prompt_all
python scripts/manage_onereason_datasets.py audit --verify-hashes
```

### 只替换一个子数据集

先在 `/data/clean/` 生成并检查清洗结果，再注册 patch。下面的命令只会复制两个懂物料文件：

```bash
python scripts/manage_onereason_datasets.py register-version \
  --version v3_material_clean \
  --parent v2_recommendation_cot_complete_all \
  --description "懂物料清洗" \
  --patch onereason_material_cot=/data/clean/onereason_material_cot.jsonl \
  --patch onereason_material_nocot=/data/clean/onereason_material_nocot.jsonl
```

### 在训练配置中选择版本

训练 YAML 保持 8 个逻辑数据集名称不变，不使用 `_train98` 后缀：

```yaml
dataset: onereason_material_cot,onereason_material_nocot,onereason_user_action_nocot,onereason_user_chain_cot,onereason_user_chain_nocot,onereason_recommendation_cot,onereason_world_cot,onereason_world_nocot
multitask_train_dataset_suffix: ""
multitask_dataset_version: v2_recommendation_cot_complete_all
multitask_dataset_version_overrides: {}
multitask_dataset_version_manifest: data/onereason_dataset_versions.json
```

只想替换懂物料时，将 `multitask_dataset_version` 改为 `v3_material_clean` 即可，其他 6 个子集自动从父版本解析。

## 常用配置与启动

| 用途 | 配置 |
| --- | --- |
| rank16 基线 | `configs/onereason/onereason_lora_2gpu_balanced40_r16.yaml` |
| 实验 A | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm.yaml` |
| 全量数据 V2 | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_alltrain_v2.yaml` |
| 实验 C2 | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_c2_topk_illegal.yaml` |
| 实验 E | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_expE_user_split_no_world.yaml` |

双卡启动示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NCCL_SOCKET_IFNAME=lo \
  llamafactory-cli train configs/onereason/<实验配置>.yaml
```

多任务 macro 模式下 `gradient_accumulation_steps` 保持为 `1`。从检查点恢复时，继续使用原配置和对应输出目录中的完整检查点。

## 监控指标

macro/GradNorm 路线的日志保留各任务原始损失、GradNorm 加权总损失、任务权重、梯度范数、梯度余弦冲突和 DDP 一致性信息。Action Select 重点监控动态合法 SID 质量、Top-K 非法质量、Top-1 非法命中率和 gold Top-5 命中率。

Native baseline 路线使用 segment-level source/domain-weighted SFT loss；`task_loss_*` 只能在同一任务的时间序列内比较，不能因 SID 权重、序列长度和数据路线差异而与其他任务作绝对横比。REC-PU 指标证明 replacement 是否触发，不等价于生成质量；历史 SID 复制率和多正例召回需要在 checkpoint 生成式评测中判断。

SID 加权实验额外记录 SID token 数量、监督 token 比例、SID token CE、普通文本 CE 和 SID 加权质量占比。

## 测试

```bash
cd /app/LLaMA-Factory
PYTHONPATH=src python3 tests/test_multitask_macro.py
PYTHONPATH=src python3 tests/test_multitask_gradient_controller.py
PYTHONPATH=src python3 tests/test_sid_token_weighting.py
PYTHONPATH=src python3 tests/test_onereason_dataset_versions.py
```

四任务布局另有双进程 CPU/Gloo smoke test；完整 5200 步训练不会作为测试的一部分自动启动。

## 工程边界

- 不上传原始数据、模型权重、checkpoint、日志和密钥。
- 不改变既有 LoRA 结构、模型 forward、optimizer、scheduler 或任务采样语义。
- Action 辅助损失和 SID 加权 CE 都保持可关闭，并继续参与原有任务 raw loss 与 GradNorm 链路。
- 修改数据版本后先执行 `audit --verify-hashes`，再启动训练。

## 许可证

代码沿用 LLaMA-Factory 的 Apache-2.0 许可证；模型和竞赛数据须遵守各自许可证及赛事规则。
- REC_G2：[Recommendation No-think Multi-Positive Prefix-Trie](实验记录/实验REC_G2_Recommendation-No-think-Multi-Positive-Trie.md)
