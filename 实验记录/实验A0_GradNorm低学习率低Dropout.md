# 实验 A0：低学习率、低 LoRA Dropout 的 GradNorm 对照
## 竞赛评分记录（原 A0）

物料四个域的评分现已确认可用，以下采用完整十一项分项和平台总分。

| checkpoint | 平台总分 | 懂物料（4 项） | 懂用户（2 项） | 懂推荐（4 项） | 懂世界 |
| ---: | ---: | --- | --- | --- | ---: |
| 4000 | 1.1865 | 0.0443, 0.0384, 0.0398, 0.0412 | 0.1518, 0.0957 | 0.0756, 0.1666, 0.1624, 0.1368 | 0.2338 |
| 5200（最终） | 1.2079（由分项求和） | 0.0463, 0.0383, 0.0420, 0.0417 | 0.1525, 0.0962 | 0.0803, 0.1666, 0.1708, 0.1386 | 0.2346 |

用户提供的 A0 最终记录未单列平台总分；按十一项四位小数分项相加得到 1.2079，可能与平台未舍入原始分数相差 0.0001。

## A0-v2：v1 思考提示与推荐 COT 完整性清洗

2026-08-04 启动了独立的 A0-v2 训练，仍使用 learning_rate: 1e-4 与 lora_dropout: 0.01，并新增：

- 全部八个训练子集默认使用 v1_thought_prompt 数据版本；
- onereason_recommendation_cot 单独覆盖为 v2_recommendation_cot_complete，仅保留【兴趣归纳】、【行为模式】、【预测总结】均完整的 40,821 条样本；
- 使用 gradient_checkpointing_layer_ratio: 0.75，保留后 27/36 个 decoder block 的 gradient checkpointing。

A0-v2 使用独立配置与输出目录，是新的数据与训练效率消融，不应与原 A0 的结果混为同一实验。


## 目标

实验 A0 基于实验 A 的 Lagged GradNorm-lite 训练路径，考察更低学习率和更低 LoRA dropout 是否能改善中后期训练稳定性。

实验 A 的阶段性评测显示，排除当前不具备参考意义的“懂物料”分数后，checkpoint-1000 优于 rank16 基础实验，checkpoint-2500 基本持平，checkpoint-3500 出现回落。A0 因此尝试：

- 将学习率从 `2.0e-4` 降至 `1.0e-4`，减小中后期参数更新幅度；
- 将 LoRA dropout 从 `0.05` 降至 `0.01`，降低 LoRA 分支的随机失活强度。

这两个参数在同一次实验中同时变化，所以 A0 是联合超参数变体，不是严格的单因素消融。若 A0 有收益，仍需分别只改学习率、只改 dropout 才能判断收益来源。

## 与实验 A 的差异

| 配置项 | 实验 A | 实验 A0 |
| --- | ---: | ---: |
| `learning_rate` | `2.0e-4` | `1.0e-4` |
| `lora_dropout` | `0.05` | `0.01` |
| `output_dir` | `...r16_gradnorm_lr2e4_len8k` | `...r16_gradnorm_a0_lr1e4_dropout001_len8k` |

除上表三项外，A0 保持实验 A 的以下语义不变：

- Qwen3 8B 基座与 FlashAttention-2；
- LoRA rank 16、alpha 32、`lora_target: all`；
- 相同训练数据和 `balanced_40` 调度；
- task-wise packing、segment attention 隔离和 position IDs 重置；
- 全局 8 microbatch、双卡各 4 microbatch 的 macro-step 语义；
- material、user、recommendation 的 Lagged GradNorm-lite；
- world loss 权重固定为 1.0；
- GradNorm warmup 200 steps、每 10 steps 测量和更新；
- cosine 冲突监控，但不启用 Ortho 投影；
- cosine scheduler、warmup ratio 0.03、seed 42、max steps 5200；
- optimizer、gradient clipping、scheduler 和 checkpoint 顺序。

## 配置与输出

配置文件：

```text
configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_a0_lr1e4_dropout001.yaml
```

核心差异：

```yaml
lora_dropout: 0.01
learning_rate: 1.0e-4
output_dir: /data/outputs/onereason_lora_2gpu_balanced40_r16_gradnorm_a0_lr1e4_dropout001_len8k
```

训练日志：

```text
/data/logs/onereason_r16_gradnorm_a0_lr1e4_dropout001_train.log
```

## 启动命令

```bash
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=2,3 FORCE_TORCHRUN=1 \
llamafactory-cli train \
  configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_a0_lr1e4_dropout001.yaml
```

## 观察指标

A0 沿用实验 A 的精简指标体系：

- `gn_weight/*`：三个主任务当前使用的动态权重；
- `gn_loss_ema/*`：三个主任务未加权 raw loss EMA；
- `gn_raw_norm/*`：还原后的任务平均原始梯度 norm；
- `gn_cos/*`：material-user、material-recommendation、user-recommendation 的梯度 cosine；
- 常规 `loss`、`grad_norm`、`learning_rate` 和 step time。

比较 A 与 A0 时，重点检查：

1. checkpoint-1000、2500、3500、5200 的可比较竞赛分数；
2. 三任务权重是否更平滑，是否频繁触及上下界；
3. 中后期 loss、grad norm 是否较实验 A 更稳定；
4. user、recommendation 和 world 是否改善，而不是只看包含异常物料项的平台总分；
5. 较低 dropout 是否带来更快过拟合迹象。

## 竞赛评分记录

“懂物料”四项目前不具备参考意义，保留原始值仅用于追溯。横向比较继续使用：

```text
可比较分数 = 懂用户 2 项之和 + 懂推荐 4 项之和 + 懂世界
```

| checkpoint | 平台总分 | 懂物料（4 项，不作比较） | 懂用户（2 项） | 懂推荐（4 项） | 懂世界 | 可比较分数 | 相对实验 A 同 checkpoint |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1000 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| 2500 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| 3500 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| 5200 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |

## 当前状态

2026-08-03 已在 GPU 2/3 启动双卡进程。两个 DDP rank 存活，模型权重已加载，GPU 各占用约 17.2 GB；但截至本次记录时尚未出现首个 macro-step 或 checkpoint，日志停在模型加载后的初始化阶段。因此当前状态记为“已启动、尚未确认进入训练”，不能记为正常训练中。

在首个 macro-step 出现前，应继续检查初始化是否完成；若长时间无进展，需要先定位 CPU 初始化或 DDP 边界，再决定是否重启。

## 实验边界

- A0 不启用实验 B 的 Ortho 投影；
- A0 不启用实验三的 Action Select 辅助损失；
- A0 不改变模型 forward、LoRA 结构、任务采样或数据语义；
- A0 不通过一次结果分别归因学习率和 dropout 的作用。


## A0-v2: ??????? COT ?????

2026-08-04 ?? A0 ???????? A0 ? `learning_rate: 1e-4` ? `lora_dropout: 0.01`?????

- ???????`v1_thought_prompt`?
- ?? COT ???`v2_recommendation_cot_complete`???????????????????????????? 40,821 ????
- ??? gradient checkpointing?`gradient_checkpointing_layer_ratio: 0.75`????? 27/36 ? checkpoint?

????????? A0?? A0 ??????????? dropout ? raw ?????

???`configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_a0_v2_think_prompt_recommendation_cot_complete_gc75.yaml`?
