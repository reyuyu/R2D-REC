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

## 原生参考基线阶段

当前进入独立的 **Native Source-Domain R32 V3** 基线阶段：基于同学提供的 material-domain 路线，在隔离的 LLaMA-Factory `01398eb` 环境中复刻原生 SFT。该路线不使用当前 macro trainer、GradNorm、Action/推荐辅助损失或 coverage packing，目的是真实比较原生训练配方，而不是叠加多任务算法。

- 正式训练：四张 A800、8K neat packing、LoRA r32、全局 batch 64、SID 权重 8、两 epoch、0.4GC（14/36 层）。
- 数据：`native_source_domain_r32_v3`，219,370 条，保留物料、用户 Action、用户 Chain、推荐 V3 多正例元数据，不包含 world。
- 监控：原始 `loss`、`grad_norm`、学习率，以及只观测不反传的 `material / recommendation / user_action / user_chain` 四项任务 loss。
- 复现脚本、配置和数据版本接口见 [baseline 目录](./baselines/native_source_domain_r32_v3/README.md)；正式实验记录见 [Native Source-Domain R32 V3](./实验记录/实验Baseline_NSD-R32-V3.md)。

训练产物、数据 JSONL、模型权重、checkpoint、日志和密钥均不提交仓库。

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

训练日志保留各任务原始损失、GradNorm 加权总损失、任务权重、梯度范数、梯度余弦冲突和 DDP 一致性信息。Action Select 重点监控动态合法 SID 质量、Top-K 非法质量、Top-1 非法命中率和 gold Top-5 命中率。

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
