# OneReason 多任务 SFT（比赛工作区）

这是一个基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的私有比赛工作区，用于微调 `OpenOneRec/OneReason-8B-pretrain-competition`。仓库的目标不是重新发布上游框架，而是沉淀当前比赛中可复现的多任务 SFT 训练、监控、评测和断点恢复改动，方便人或 AI 快速理解现状并继续迭代。

> 代码基线：LLaMA-Factory commit `01398eb`。模型权重、原始数据、预测文件、训练日志、checkpoint 与凭据均不会提交到本仓库。

## 比赛问题与数据组织

训练数据被归为四个**顶层任务**，并保留 CoT / 非 CoT 等子任务边界：

| 顶层任务 | 训练内容 | 当前子任务示例 |
| --- | --- | --- |
| `material`（懂物料） | 广告、商品、直播、视频的 SID/描述理解 | material CoT、material non-CoT |
| `user`（懂用户） | 用户行为选择与多跳逻辑链 | action non-CoT、chain CoT、chain non-CoT |
| `recommendation`（懂推荐） | 推荐相关的用户画像/视频 SFT | recommendation CoT |
| `world`（懂世界） | 通用世界知识 SFT | world CoT、world non-CoT |

数据在训练前按 **98% train / 2% dev** 切分；训练注册表位于 `data/dataset_info.json`。本仓库只保存注册信息，不保存实际 JSONL 文件。

## 当前服务器进展（最后整理：2026-08-02）

- 基座模型已部署在服务器：`/data/models/onereason-8b-pretrain-competition`。
- 两组 2-GPU LoRA 实验已启动并行对比：
  - `onereason_lora_2gpu_balanced40_r16_lr2e4_len8k`（LoRA rank 16）
  - `onereason_lora_2gpu_balanced40_r64_lr2e4_len8k`（LoRA rank 64）
- 两组均以 `max_steps: 5200` 运行；最近一次监控快照（2026-08-02 16:43，UTC+8）为 rank 16：`4940 / 5200`、rank 64：`4930 / 5200`，进程与四张 GPU 均正常工作。请以对应输出目录的 `monitor/metrics.jsonl` 与训练日志为准，不要把本 README 中的数值视为实时状态。
- 训练输出和 checkpoint 位于服务器 `/data/outputs/<experiment>/`；每个保存点为 LoRA adapter，核心文件是 `adapter_model.safetensors` 与 `adapter_config.json`。
- 基础监控记录每个任务的 loss、packing token 数、监督 token 数、segment 数、学习率与耗时。配套本地查看器位于工作区外的 `training_loss_monitor/`，不纳入此仓库。

## 已实现的核心改动

### 1. 多任务宏步训练

开关为 `multitask_macro_training: true`，关闭时完全使用上游 LLaMA-Factory 原始路径。

开启后：

```text
四个独立任务数据流
  -> 任务内子任务采样
  -> 不跨顶层任务的 task-wise packing
  -> MultiTaskMacroStepLoader
  -> 一个 macro-step 的若干 task microbatch
  -> 一次 optimizer.step() / scheduler.step()
```

- `global_step`、`max_steps`、日志/保存步数均是 **macro-step（优化器更新）** 单位。
- `gradient_accumulation_steps` 必须为 `1`：宏步内部 microbatch 由专用 Trainer 显式管理，避免和 Hugging Face 原生累计叠加。
- `num_train_epochs` 在该模式下不控制终止；必须显式提供正数 `max_steps`。
- checkpoint 仅在完整 macro-step 后保存，恢复时从下一个完整宏步继续，并保存任务采样、packing 和 super-cycle 状态。

实现入口：

- `src/llamafactory/data/multitask.py`：任务数据流、采样器、packing、macro loader 与状态序列化。
- `src/llamafactory/train/sft/trainer.py`：最小侵入的宏步 Trainer 分支及任务级 `compute_task_microbatches(...)` 接口。
- `src/llamafactory/train/sft/workflow.py`：开启/关闭两条训练路径的接线。
- `src/llamafactory/hparams/data_args.py`：多任务 YAML 参数和启动校验。

### 2. balanced_40 任务调度

当前实验启用 `multitask_supercycle_mode: balanced_40`。一个 40 macro-step super-cycle 中，每一步总计 8 个 packed microbatch：

```text
35 × [material=3, user=3, recommendation=2, world=0]
 3 × [material=2, user=4, recommendation=2, world=0]
 1 × [material=2, user=3, recommendation=3, world=0]
 1 × [material=2, user=3, recommendation=2, world=1]
```

这样让极小的 `world` 数据集以完整 task microbatch 进入训练，而不是在每个宏步过度重复；同时保留每个任务的显式边界，供下一阶段取得独立任务梯度。

### 3. 2-GPU DDP 与全局 8 microbatch 语义

启用 `multitask_global_microbatch_ddp: true` 后，两个 rank 使用相同的 task 顺序与 super-cycle；每个 rank 只处理本 rank 的不同样本。

当前“每个 macro-step 8 个 microbatch”指**全局逻辑微批**：双卡协同时，每卡处理其中 4 个，从而保持 8 个微批后更新一次的优化语义，而不是把有效累计放大为 16 或 32。

### 4. Task-wise packing 与注意力隔离

- packing 仅在同一子任务内发生，不跨四个顶层任务混合。
- 每个片段重置 `position_ids`，并保留 `segment_offsets`、`cu_seqlens`、`num_segments` 等元数据。
- 实验使用 FlashAttention-2（`flash_attn: fa2`）和 Transformers v5.6+。重置位置会被识别为变长 packed sequence 的边界，底层使用 varlen `cu_seqlens`，实现 block-diagonal attention：一个片段不会注意到另一个片段的 token。
- YAML 中 `packing: false` / `neat_packing: false` 是为了关闭**上游原生** packing；本项目的 task-wise packing 仍由多任务宏步管线接管。

### 5. 监控、评测与数据工具

| 工具 | 作用 |
| --- | --- |
| `scripts/prepare_multitask_dev_split.py` | 生成 98/2 训练/开发集切分 |
| `scripts/audit_cutoff_lengths.py` | 全量统计不同 `cutoff_len` 下的截断情况 |
| `scripts/generate_multitask_predictions.py` | 生成各任务预测 |
| `scripts/evaluate_sid_action_predictions.py` | SID top-k / 三段完全命中与 action 近似指标 |
| `scripts/plot_multitask_metrics.py` | 绘制多任务训练指标 |
| `scripts/watch_multitask_checkpoints.py` | 轮询 checkpoint 并触发后续评测流程 |
| `tests/test_multitask_macro.py` | 宏步、调度与状态恢复的基础测试 |

## 当前训练配置

两组正在比较的配置：

- `configs/onereason/onereason_lora_2gpu_balanced40_r16.yaml`
- `configs/onereason/onereason_lora_2gpu_balanced40_r64.yaml`

共同关键参数：

```yaml
stage: sft
finetuning_type: lora
lora_target: all
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0
cutoff_len: 8192
flash_attn: fa2
gradient_accumulation_steps: 1
multitask_macro_training: true
multitask_supercycle_mode: balanced_40
max_steps: 5200
seed: 42
```

LoRA 对照组差异：rank 16 使用 `alpha: 32`，rank 64 使用 `alpha: 128`；两者均为 `lora_dropout: 0.05`。

## 启动与恢复

在已具备模型、数据和依赖的服务器上，典型启动方式：

```bash
cd /app/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 \\
llamafactory-cli train configs/onereason/onereason_lora_2gpu_balanced40_r16.yaml
```

恢复训练时，使用对应输出目录最后一个完整 checkpoint，并保持原 YAML 的 macro-step / super-cycle / DDP 设置不变。不要把原生 `gradient_accumulation_steps` 改成大于 1。

## 下一阶段：已预留、尚未实现

当前实现**没有**启用 GradNorm、PCGrad、Ortho-LoRA、任务梯度缓存、梯度余弦/冲突裁剪，也没有根据文本规则硬编码 `think_span`、`answer_span`、`sid_spans`。

后续若实现任务梯度方法，应基于宏步 Trainer 已保留的 task 边界，在 `compute_task_microbatches(task_name, microbatches)` 周围采集或归一化梯度；任务专属定位应通过 `sample_metadata` 或明确的新字段接入，避免改变既有 tokenize/loss 语义。

## 给后续 AI / 协作者的工作约束

1. 先读取本 README、`ONEREASON_MULTITASK.md`、两份 YAML 与 `src/llamafactory/data/multitask.py`，再改变训练语义。
2. `multitask_macro_training: false` 必须严格保持上游 LLaMA-Factory 行为；不要把新 loader、packing 或 Trainer 泄漏到关闭路径。
3. 修改调度时始终保证 DDP 各 rank 的任务顺序、microbatch 总数和 optimizer step 边界一致。
4. 不提交模型、adapter、数据集、原始预测、监控日志或任何凭据；`.gitignore` 已覆盖这些内容。
5. 改动梯度算法前，先在单卡和 DDP 下验证 macro-step、恢复训练和关闭开关的回归测试。

## 上游与许可证

本项目保留 LLaMA-Factory 的 Apache-2.0 许可证与上游代码。具体多任务扩展摘要见 [ONEREASON_MULTITASK.md](ONEREASON_MULTITASK.md)。模型与比赛数据的使用须分别遵循其原始许可证与赛事规则。
