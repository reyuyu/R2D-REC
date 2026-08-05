# OneReason 多任务 SFT

用于微调 `OpenOneRec/OneReason-8B-pretrain-competition` 的多任务 SFT 工作区，基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 扩展。

仓库保留训练代码、配置、测试、实验说明和评分记录；不包含模型权重、原始数据、预测、日志、checkpoint 或任何凭据。

> 代码基线：LLaMA-Factory commit `01398eb`。

## 项目导航

- [多任务设计说明](./ONEREASON_MULTITASK.md)
- [数据版本管理](./ONEREASON_DATASET_VERSIONS.md)
- [实验记录与评分](./实验记录/README.md)
- [数据集注册表](./data/dataset_info.json)

## 训练设计

| 顶层任务 | 子任务 | 训练内容 |
| --- | --- | --- |
| `material` | CoT、non-CoT | 广告、商品、直播、视频的 SID/描述理解 |
| `user` | action non-CoT、chain CoT/non-CoT | 用户行为选择与多跳逻辑链 |
| `recommendation` | CoT | 推荐用户画像与内容理解 |
| `world` | CoT、non-CoT | 通用世界知识 SFT |

训练使用自定义多任务 macro-step：全局每个优化器更新对应 8 个逻辑 microbatch，双卡 DDP 时每个 rank 处理其中 4 个。任务调度使用 `balanced_40`，每个 macro-step 只执行一次 optimizer/scheduler step。

task-wise packing 仅发生在同一子任务内；每个 segment 重置 `position_ids`，并通过 FlashAttention-2 的变长序列边界实现 block-diagonal attention 隔离。配置中的 `packing: false` 只表示关闭上游原生 packing，本项目的自定义 packing 仍然启用。

## 实验

| 实验 | 内容 | 说明 |
| --- | --- | --- |
| [A](./实验记录/实验A_GradNorm-lite.md) | Lagged GradNorm-lite | 三个主任务动态权重、梯度 norm/cosine 监控，world 固定权重 |
| [A0](./实验记录/实验A0_GradNorm低学习率低Dropout.md) | GradNorm 对照 | 学习率、LoRA dropout、数据版本与选择性 gradient checkpointing 消融 |
| [A1](./实验记录/实验A1_think_prompt-GradNorm.md) | think/no_think 提示 | 思考提示数据版本对照 |
| [B](./实验记录/实验B_GradNorm-Ortho-LoRA.md) | 局部 Ortho-LoRA | 基于同步任务梯度的 LoRA-B PCGrad 风格投影 |
| [C](./实验记录/实验C_Action-Select历史约束.md) | Action Select 辅助损失 | 历史 SID Trie、完整 SID 去重、Continue/Stop 平衡 |
| [C-fast](./实验记录/实验C-fast_Action-Select向量化优化.md) | 向量化 Action auxiliary | C 的等价加速实现 |
| [C1](./实验记录/实验C1_Trie优先消融.md) | Trie 优先 | 负向消融：1000 到 2500 checkpoint 总分由 1.1824 降至 1.1346，不作为后续基线 |
| [C2](./实验记录/实验C2_全词表TopK非法SID惩罚.md) | 全词表 Top-K 非法 SID | 动态完整 SID 合法集上的全词表非法竞争惩罚，当前双卡训练中 |

实验 B 保留为实现和消融基线；正式实验的状态、评测结果与限制以[实验记录](./实验记录/README.md)为准。

## 常用配置

| 用途 | 配置 |
| --- | --- |
| rank16 基线 | `configs/onereason/onereason_lora_2gpu_balanced40_r16.yaml` |
| 实验 A GradNorm | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm.yaml` |
| A0-v2 重跑 | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_a0_v2_think_prompt_recommendation_cot_complete_gc75_lr2e4_dropout005.yaml` |
| 实验 C-fast | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_v1_vectorized.yaml` |
| 实验 C1 | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_c1_trie_priority.yaml` |
| 实验 C2 | `configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_c2_topk_illegal.yaml` |

典型双卡启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NCCL_SOCKET_IFNAME=lo \
  llamafactory-cli train configs/onereason/<experiment>.yaml
```

在 `multitask_macro_training: true` 模式中，`gradient_accumulation_steps` 必须保持为 `1`。恢复训练时使用相同 YAML 与对应输出目录中的完整 checkpoint。

## 监控指标

后续 macro-step 监控会同时记录各任务的原始 microbatch loss：`loss_raw_material`、`loss_raw_user`、`loss_raw_recommendation`、`loss_raw_world`。

并记录 `loss_gradnorm_total`：各 global microbatch 的 `task_weight × raw_loss` 之和除以 global microbatch 数。它与当前 GradNorm 加权、DDP 平均后传给优化器的 macro loss 标量一致。

C2 的 Action Select 另记录全词表动态合法 SID 质量、Top-K 非法质量、Top-1 非法率和 gold Top-5 率。
## 验证与边界

多任务 macro-step、packing attention、GradNorm、Ortho 与 Action Select 辅助损失均有单卡测试；梯度控制器与 Action auxiliary 另有双卡 smoke test。修改训练语义前应先阅读对应实验文档并运行相关测试。

本项目保留上游 LLaMA-Factory 的 Apache-2.0 许可证。模型与比赛数据的使用须遵守各自许可证与赛事规则。
