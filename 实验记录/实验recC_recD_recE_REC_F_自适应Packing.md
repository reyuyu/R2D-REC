# REC 系列：自适应 Packing 与四任务训练

更新时间：2026-08-07

本页集中记录 `rec_A`、`recB` 以及后续 `recC`、`recD`、`recE`、`REC_F` 的有效工程改动。实验输出目录、checkpoint、训练日志和原始大数据文件不进入 Git；仓库只保存可复现的代码、配置、测试和结论。

## 共同训练语义

- 四个顶层任务：`material`、`user_action`、`user_chain`、`recommendation`；移除 `world`。
- 双卡 DDP，每个 macro-step 产生 8 个 global microbatch，每卡 4 个，gradient accumulation 为 1。
- 四任务 GradNorm 保留原数学定义、warmup、更新间隔和实际 global task microbatch 计数。
- SID 加权 CE 使用归一化分母，`sid_token_weight=8.0`；Action 辅助损失在 REC 正式配置中关闭。
- 默认正式训练使用 8192 cutoff/pack、LoRA rank16、学习率 `2e-4`、dropout `0.05`、bf16 和 0.75 gradient checkpointing。

## 方案演进

### rec_A：固定 4:1:1:2

`rec_A` 使用四任务固定调度 `material:user_action:user_chain:recommendation = 4:1:1:2`，用于对照原有任务配比。训练配置为 [`onereason_lora_2gpu_rec_A_r16_gradnorm_sid_weight8_v2_dual_gc075.yaml`](../configs/onereason/onereason_lora_2gpu_rec_A_r16_gradnorm_sid_weight8_v2_dual_gc075.yaml)。该方案能运行，但固定比例不能根据各队列实际 pack 数自动覆盖一个 epoch。

### recB：固定 1:3:3:1

`recB` 将调度改为 `1:3:3:1`，但在 8192 长序列、0.75 GC 下出现 step 22 的 rank1 backward OOM；将 fused AdamW 换成普通 AdamW 仍复现，因此问题主要来自当步 pack/显存峰值，而不是优化器。该方案保留为失败对照，不建议直接恢复正式训练。

### recC：8K 同子任务 BFD + coverage/deficit

第一阶段将固定长度桶贪心改为 deterministic same-subtask Best-Fit Decreasing（BFD）：同一 subtask 内按长度降序，将样本放入剩余空间最小且可容纳的 pack；禁止跨 subtask、拆样本和改变 segment attention 隔离。调度器从实际 pack 数计算 deficit，每个 macro-step 强制四任务各至少 1 个 pack，再分配剩余 4 个 slot；所有 queue 完成当前 epoch 前设置全局 barrier，禁止某个 queue 提前重复。

相关配置与测试：

- [`onereason_lora_2gpu_recC_bfd_coverage_smoke.yaml`](../configs/onereason/onereason_lora_2gpu_recC_bfd_coverage_smoke.yaml)
- [`tests/test_recb_bfd_coverage.py`](../tests/test_recb_bfd_coverage.py)

CPU smoke 验证了 8 global pack、每 rank 4 pack、四任务非零计数、BFD 不超 8192、无丢样/重复以及 barrier/cursor 恢复。

### recD：cost-aware 双卡分区 smoke

第二阶段在 recC 计划上增加 token-cost 感知的 4+4 rank 分区，只在调度层选择更均衡的分配，不改 GradNorm、loss 或模型。100-step CPU/双进程 smoke 通过，观测 macro 时间约 13.07 秒、两 rank cost gap 约 1.02%，没有 OOM。它是可选调度扩展，不是 REC_F 的默认路径。

相关文件：

- [`onereason_lora_2gpu_recD_bfd_coverage_costaware_smoke.yaml`](../configs/onereason/onereason_lora_2gpu_recD_bfd_coverage_costaware_smoke.yaml)
- [`onereason_lora_2gpu_recD_bfd_coverage_costaware_resume_smoke.yaml`](../configs/onereason/onereason_lora_2gpu_recD_bfd_coverage_costaware_resume_smoke.yaml)
- [`tests/test_recd_costaware.py`](../tests/test_recd_costaware.py)
- [`scripts/benchmark_recd_costaware.py`](../scripts/benchmark_recd_costaware.py)

### recE：按 subtask 自适应 pack 长度

第三阶段允许不同 subtask 使用各自的最大 pack 长度，仍保持每个 pack 单一 subtask、单样本不拆、position reset 和 cu-seqlens 语义不变。验证结果：

| 变体 | 结果 | 结论 |
| --- | --- | --- |
| user 12K + GC0.75 | 首个 macro OOM | 不作为正式配置 |
| user 12K + full GC | 20-step smoke 通过，约 17.8 秒/macro | 可运行但速度慢，显存余量有限 |
| user 10K + GC0.75 | 20-step smoke 通过，约 15.06 秒/macro | 峰值接近 80,677/81,920 MiB，不适合作为稳态默认 |
| 全部 8K + GC0.75 | 通过 | REC_F 正式方案 |

相关配置和测试：

- [`onereason_lora_2gpu_recE_final_adaptive_packing.yaml`](../configs/onereason/onereason_lora_2gpu_recE_final_adaptive_packing.yaml)
- [`onereason_lora_2gpu_recE_adaptive_pack10k_gc075_smoke.yaml`](../configs/onereason/onereason_lora_2gpu_recE_adaptive_pack10k_gc075_smoke.yaml)
- [`onereason_lora_2gpu_recE_adaptive_pack12k_fullgc_smoke.yaml`](../configs/onereason/onereason_lora_2gpu_recE_adaptive_pack12k_fullgc_smoke.yaml)
- [`tests/test_stage3_adaptive_pack.py`](../tests/test_stage3_adaptive_pack.py)
- [`scripts/benchmark_adaptive_pack.py`](../scripts/benchmark_adaptive_pack.py)

### REC_F：最终 8K + SID weight 8

REC_F 采用最稳妥的全 8K 配置：same-subtask BFD、coverage/deficit scheduler、四任务 GradNorm、SID token weight 8 和 0.75 GC。8K BFD 得到的 pack 数为：material 15,428、user_action 9,476、user_chain 12,957、recommendation 7,883，共 45,744 个 pack；8 个 global pack/macro，因此 `45,744 / 8 = 5,718` 个 macro-step 正好覆盖一个 pack epoch。

正式配置：[`onereason_lora_2gpu_recF_r16_gradnorm_sid_weight8_v2_dual_8k_gc075.yaml`](../configs/onereason/onereason_lora_2gpu_recF_r16_gradnorm_sid_weight8_v2_dual_8k_gc075.yaml)。当前训练独立使用 `/data/outputs/onereason_lora_2gpu_recF_r16_gradnorm_sid_weight8_v2_dual_8k_gc075`，日志为 `/data/logs/recF_r16_gradnorm_sid_weight8_v2_dual_8k_gc075.log`；仓库不上传这些运行产物。初始短跑未观察到 NaN、Traceback 或 OOM，monitor 继续记录四项 raw loss、GradNorm 总损失/权重、冲突余弦、SID mass share、tokens/macro 和 macro 时间。

## 数据和自适应行为

REC 配置只引用 material、user_action、user_chain 和 recommendation 数据，推荐任务使用 v2 CoT/No-think 双路数据。BFD 和 coverage 计划在启动时依据 tokenized 数据重建；增加或清洗某个子数据集后，需要生成新的数据版本、清理对应缓存并重启训练，系统会重新计算 pack 数、coverage 比例和一个 epoch 所需 step，不能在已有 checkpoint 中静默改变数据顺序。

## 已验证测试

```text
test_recb_bfd_coverage.py       4 passed
test_recd_costaware.py          4 passed
test_stage3_adaptive_pack.py    5 passed
test_multitask_macro.py        14 passed
test_multitask_gradient_controller.py 15 passed
test_sid_token_weighting.py    13 passed
CPU DDP smoke_expE_4task_ddp.py passed
```

REC_F 的正式 5718-step 训练仍在运行；其最终评测结果待训练完成后补录。
