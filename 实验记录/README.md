# 实验记录

本目录记录 OneReason 多任务 SFT 的后续优化方案、实现状态、测试方法和实验结果。

## 当前状态

- [实验一：Lagged GradNorm-lite](./实验A_GradNorm-lite.md)：已实现并通过单卡、双卡及真实 Qwen3/LoRA 梯度路径测试；rank16 双卡正式实验已启动。
- [实验 A0：低学习率、低 LoRA Dropout 的 GradNorm 对照](./实验A0_GradNorm低学习率低Dropout.md)：实验 A 的联合超参数变体，仅将学习率改为 `1e-4`、LoRA dropout 改为 `0.01`；rank16 双卡正式实验正在运行。
- [实验二：GradNorm-lite + 局部 LoRA 梯度冲突投影](./实验B_GradNorm-Ortho-LoRA.md)：已实现并通过实验 A 回归、单卡算法和双卡全链路 smoke test；正式训练因门控下长期未触发有效投影，已于 2026-08-03 主动终止。
- [实验三：Action Select 历史约束与长度平衡](./实验C_Action-Select历史约束.md)：第一版已实现，使用完整 SID 动态 Trie、Continue/Stop 平衡、warmup 与 Action CE 比例上限；不拆分顶层任务，不增加 forward/backward。
- [实验三 C-fast：Action Select 向量化优化](./实验C-fast_Action-Select向量化优化.md)：已完成数学/梯度等价测试、单/双卡 smoke、CUDA profiler 和真实 8B 20-step 短跑；保留 legacy 默认路径，使用独立配置与输出目录。
- [实验三 C1：Trie 优先消融](./实验C1_Trie优先消融.md)：将动态历史合法性与完整 SID 去重分配为主要辅助预算，Continue/Stop 使用较小独立预算，避免长度目标长期触发联合 cap；已于 2026-08-04 在 GPU 2/3 启动 rank16 双卡正式训练。

## 实验关系

实验 A 处理三个主任务的梯度尺度不平衡，`world` 权重固定为 1。实验 B 复用实验 A 的 Hook、任务向量、DDP 同步、动态权重和 checkpoint，只在持续负冲突时对选中的高层 LoRA-B 梯度做局部 PCGrad 投影。

实验 A0 完整复用实验 A 的训练语义，同时将学习率和 LoRA dropout 分别调整为 `1e-4` 和 `0.01`。由于两个超参数同时改变，A0 用于比较联合配置效果，不用于单独归因某一个参数。

两组实验保持相同的 rank16、数据、balanced_40、seed、max_steps 和 batch 语义，并使用不同的 `output_dir`，可直接并行比较。实验 B 不增加任务专属 adapter，不修改 LoRA forward 或推理结构。

实验三建立在实验 A 配置上，只修改 `user/action_nocot` 的 raw loss。Action 辅助项在当前 user GradNorm 权重之前加入，所以继续作为 user 梯度的一部分参与实验 A 的尺度平衡，不改变顶层任务调度。

C-fast 不改变实验三的目标，只把重复的 full-vocabulary denominator 和逐 SID Trie 归约改为 microbatch 级去重、分块和分组批处理。是否将其设为未来默认实现，取决于等价性、显存、双卡 smoke 和真实 8B 短跑结果。
