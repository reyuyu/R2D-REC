# 实验记录

本目录用于记录 OneReason 多任务 SFT 后续优化实验方案、设计原因和实验结论。

当前规划：

- [实验一：GradNorm-lite 多任务梯度平衡方案](./实验一_GradNorm-lite.md)
- [实验二：GradNorm-lite + Ortho-LoRA 梯度冲突解耦方案](./实验二_GradNorm-Ortho-LoRA.md)

## 实验背景

当前多任务训练包含：

- material（懂物料）
- user（懂用户）
- recommendation（懂推荐）
- world（懂世界）

现有 balanced_40 调度已经保证任务级采样边界，但不同任务仍可能存在：

1. 梯度尺度不平衡：某些任务产生更大的更新幅度；
2. 梯度方向冲突：不同任务对共享 LoRA 参数提出相反更新方向。

因此后续实验采用两阶段思路：

1. GradNorm-lite：先平衡不同任务的梯度贡献；
2. Ortho-LoRA：在平衡后的任务梯度基础上，进一步消除持续性的负方向冲突。
