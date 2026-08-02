# OneReason 多任务 SFT（比赛工作区）

这是一个基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的私有比赛工作区，用于微调 `OpenOneRec/OneReason-8B-pretrain-competition`。仓库的目标不是重新发布上游框架，而是沉淀当前比赛中可复现的多任务 SFT 训练、监控、评测和断点恢复改动，方便人或 AI 快速理解现状并继续迭代。

> 代码基线：LLaMA-Factory commit `01398eb`。模型权重、原始数据、预测文件、训练日志、checkpoint 与凭据均不会提交到本仓库。

## 实验记录

后续多任务优化实验方案统一记录在：

- [实验记录目录](./实验记录/README.md)
- [实验一：GradNorm-lite 多任务梯度平衡](./实验记录/实验一_GradNorm-lite.md)
- [实验二：GradNorm-lite + Ortho-LoRA 梯度解耦](./实验记录/实验二_GradNorm-Ortho-LoRA.md)

## 比赛问题与数据组织

训练数据被归为四个顶层任务，并保留 CoT / 非 CoT 等子任务边界。

| 顶层任务 | 训练内容 |
| --- | --- |
| material（懂物料） | 广告、商品、直播、视频的 SID/描述理解 |
| user（懂用户） | 用户行为选择与多跳逻辑链 |
| recommendation（懂推荐） | 推荐相关 SFT |
| world（懂世界） | 通用世界知识 SFT |

详细训练实现、packing、macro-step、DDP 设计以及当前实验状态保持原 README 后续章节。

## 下一阶段方向

当前实现预留任务梯度方法扩展接口。后续实验基于已有 task 边界实现：

1. GradNorm-lite：调整不同任务梯度贡献；
2. Ortho-LoRA：处理共享 LoRA 参数中的梯度方向冲突。
