# 实验 C-fast：Action Select 向量化优化
## 竞赛评分记录

物料四个域的评分现已确认可用，以下均按完整十一项分项与平台总分比较。

| checkpoint | 平台总分 | 懂物料（4 项） | 懂用户（2 项） | 懂推荐（4 项） | 懂世界 |
| ---: | ---: | --- | --- | --- | ---: |
| 1000 | 1.2078 | 0.0427, 0.0380, 0.0381, 0.0424 | 0.1398, 0.0885 | 0.0952, 0.1768, 0.1750, 0.1431 | 0.2283 |
| 3000 | 1.1925 | 0.0457, 0.0375, 0.0452, 0.0436 | 0.1431, 0.0969 | 0.0831, 0.1564, 0.1708, 0.1386 | 0.2316 |

checkpoint-1000 到 3000 的总分下降 0.0153。该表仅记录评测结果；在相同 checkpoint 的自由生成错误率尚未统一复测前，不能据此把变化归因于 Action 辅助项本身。


## 目标与边界

正式实验 C 的 Action Select 辅助目标没有增加模型 forward 或 backward，但 legacy 实现对每个 SID、每个位置分别发起小型归约。在真实训练中，C 的最近五个 macro-step 平均约 17.95 秒，A0 约 12.99 秒，C 额外耗时约 38%；而且 SID 越多，step 通常越慢。

C-fast 只改变相同数学目标的执行组织，不改变动态 unused-history Trie、完整 SID 移除、Continue/Stop 公式、loss 权重、warmup、cap、GradNorm user raw loss、loss divisor、packing、balanced_40 或 DDP 语义。Legacy 实现仍保留且全局默认使用。

## Legacy 瓶颈

设一个 Action segment 有 `S` 个有效 gold SID、`B=S-1` 个继续边界、最终 tail 监督位置数为 `T<=4`：

- 每个 SID 的 domain/a/b/c 四层各计算一次 type denominator 和 allowed numerator，即每个 SID 8 次小型 type/allowed `logsumexp`，并伴随逐位置 Python 调度和小 CUDA tensor 创建。
- 每个继续边界最多执行 next-domain CE、separator CE、no-early-stop 三次 full-vocabulary denominator。
- 每个 tail 位置分别为真实 token CE 和 stop-domain mass 重算 denominator，共 `2T` 次。
- separator CE 与 no-early-stop 会复用同一 logits 位置；tail CE 与 stop-domain 也复用同一位置，但 legacy 会重复归一化。

当前模型实测词表大小为 176,255；domain/a/b/c 语义词表大小分别为 4/8192/8192/8192。重复的单行全词表归约和大量小 kernel 是主要 GPU 开销；Python 逐 segment、SID、位置构造集合与 tensor 则是 CPU 开销。双卡上 Action 答案长度不同还会让较快 rank 等待较慢 rank。

## 向量化设计

`UserActionAuxiliaryController.compute()` 根据以下开关分发：

```yaml
user_action_aux_vectorized_enabled: false
user_action_aux_full_vocab_chunk_size: 64
```

`false` 严格调用 `_compute_legacy()`；`true` 调用 `_compute_vectorized()`。新开关没有动态状态，不进入 checkpoint。

### Microbatch Batch Plan

每个 Action microbatch 构建一次临时 `ActionAuxBatchPlan`。Plan 展平 Trie 位置、可变长 allowed IDs、SID/segment 索引、继续边界事件和 tail 事件；生命周期只覆盖当前 microbatch，不修改 metadata，也不在 controller 中缓存计算图。

### Full-vocabulary denominator 复用

所有 CE、no-early-stop 和 stop-domain 事件先收集 target positions，使用：

```python
unique_positions, inverse = torch.unique(positions, sorted=True, return_inverse=True)
```

因果位置统一为 `target_position - 1`。每个唯一位置的 FP32 `logsumexp` 只计算一次，并由 CE、stop-token mass 和 domain mass 共同复用。全词表 rows 按 `user_action_aux_full_vocab_chunk_size` 分块，避免持有过大的 `[positions, vocab]` FP32 中间张量。

### Trie 分组批处理

Trie 位置按 domain/a/b/c 分成四组。每组批量 gather 对应语义类型 logits；可变长 allowed 集合使用 padded IDs 加布尔 mask 表示，PAD 位置在 `logsumexp` 前填为负无穷，不创建 dense `[positions, vocab]` mask。

归一化仍严格恢复为：有效 domain/a/b/c 在 SID 内平均，再在 segment 内按 SID 平均，最后在 microbatch 内按有效 segment 平均。完整四元组移除、共享前缀保留、gold 不在历史时跳过 Trie，以及 gold 重复时关闭动态移除均未改变。

## 配置与启动

独立配置：

```text
configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_v1_vectorized.yaml
```

它与实验 C 原配置仅在 vectorized 开关、chunk size 和独立 output_dir 上不同。原实验 C YAML 和输出目录不修改。

```bash
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 \
llamafactory-cli train configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_v1_vectorized.yaml
```

正式 5200-step 实验已于 2026-08-03 21:05:43 CST 在 GPU 0/1 从初始模型启动，不从 legacy C checkpoint 恢复：

```text
PID: 4024669
log: /data/logs/onereason_r16_gradnorm_action_aux_v1_vectorized_train.log
output: /data/outputs/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_v1_vectorized_lr2e4_len8k
```

原 legacy C 的 checkpoint-500 保留不变，A0 继续在 GPU 2/3 运行。

主要代码入口：

- `src/llamafactory/train/sft/user_action_auxiliary.py`：legacy/vectorized 分发、batch plan 和等价计算。
- `src/llamafactory/hparams/data_args.py`：两个新参数及 chunk 正数校验。
- `scripts/benchmark_action_auxiliary.py`：CUDA event、显存和 profiler 基准。
- `tests/test_user_action_auxiliary.py`：标量、指标、logits 梯度和 Qwen3+LoRA 参数梯度等价测试。
- `tests/test_user_action_auxiliary_qwen3.py`、`tests/test_user_action_auxiliary_ddp.py`：单卡和双卡 smoke。

## 正确性验证

同一随机 FP32 logits 分别通过 legacy 和 vectorized 路径，覆盖单/多/长 SID、多 segment packing、gold 不在历史、gold 重复、解析失败、Trie-only、Length-only、cap、warmup 和不同 chunk size。

当前结果：

```text
Action Select auxiliary：48/48 通过
multitask macro/packing：9/9 通过
GradNorm/Ortho gradient controller：13/13 通过
随机小 Qwen3 + 真实 PEFT LoRA 参数梯度：通过
```

长答案双 segment FP32 对照中：loss 绝对误差为 0，指标最大绝对误差 `2.38e-7`，logits 梯度最大/平均绝对误差分别为 `1.49e-8`/`1.38e-10`，相对 L2 误差 `6.16e-8`。

所有 Action 指标名称和统计定义保持不变。每个 Action microbatch 仍为一次 forward、复用 `outputs.logits`、一次 backward。

最终回归命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 tests/test_user_action_auxiliary.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 tests/test_multitask_macro.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 tests/test_multitask_gradient_controller.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:tests python3 tests/test_user_action_auxiliary_qwen3.py
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src:tests \
torchrun --standalone --nproc_per_node=2 tests/test_user_action_auxiliary_ddp.py
```

## Benchmark 方法与结果

独立脚本：

```bash
PYTHONPATH=src python3 scripts/benchmark_action_auxiliary.py \
  --device cuda:0 --dtype bf16 --warmup 5 --repeats 30 --profile
```

脚本使用相同 logits、labels、metadata、dtype、warmup 和 cap，对 short/medium/long/multi-segment 四档分别报告 auxiliary forward、forward+backward 的中位数和 P90、峰值显存、唯一 full-vocab 位置数、Trie 位置数，以及 profiler 中关键算子调用次数。设备为 A800 80GB，模型词表大小 176,255，dtype 为 BF16。

紧凑序列、`chunk=32`、30 次计时的 auxiliary forward+backward 中位数：

| 档位 | SID/segment | Legacy | Vectorized | 加速 |
|---|---:|---:|---:|---:|
| short | 2/1 | 13.04 ms | 11.11 ms | 1.17x |
| medium | 8/1 | 38.91 ms | 11.50 ms | 3.38x |
| long | 24/1 | 118.97 ms | 13.45 ms | 8.85x |
| multi-segment | 16/2 | 105.17 ms | 16.80 ms | 6.26x |

真实 `sequence_length=8192` shape、10 次计时的结果更能反映反向 scatter 到完整 logits 的成本：

| 档位 | Legacy median/P90 | Vectorized median/P90 | 加速 | Legacy/Vectorized 峰值增量 |
|---|---:|---:|---:|---:|
| medium，8 SID | 795.08/795.25 ms | 118.89/119.14 ms | 6.69x | 5527.0/5511.7 MiB |
| long，24 SID | 2259.89/2260.21 ms | 120.44/120.69 ms | 18.76x | 5561.0/5519.0 MiB |

紧凑 long profiler 中，`aten::logsumexp` 从 492 次降至 22 次，`aten::index_select` 从 219 次降至 21 次，50 次 `cross_entropy_loss/_log_softmax` 被复用 denominator 的显式 CE 取代。对应 CUDA `logsumexp` 总时间从约 10.10 ms 降至 0.71 ms。

chunk 8/16/32/64 均运行了 20 次对照。chunk 64 在 long 和 multi-segment 档最快；long 中位数约 13.33 ms，峰值约 137.5 MiB，相对 chunk 32 只增加约 6.2 MiB，因此正式新配置选择 64。

单卡随机 Qwen3+LoRA vectorized smoke 与双卡 DDP `forward -> aux -> backward -> statistics all_reduce -> optimizer.step` smoke 均已通过。

真实 8B、FA2、双卡、rank16 短跑使用每个子数据集 500 条真实样本，保持模型、seed、balanced_40、packing、GradNorm、batch 和 loss 配置不变，只把 `max_steps` 设为 20、`logging_steps` 设为 1。结果：

| 路径 | 采样点 | mean | median | P90 | min-max |
|---|---:|---:|---:|---:|---:|
| Legacy 正式 C monitor | 50 | 18.16 s | 17.87 s | 21.10 s | 14.07-25.13 s |
| Vectorized 8B smoke | 20 | 13.01 s | 13.06 s | 13.46 s | 12.46-13.52 s |

Vectorized 平均每个 macro-step 减少约 5.16 秒，均值耗时下降 28.4%，P90 下降 36.2%，完整训练 step 加速约 1.40x。两组不是逐 batch 配对实验：legacy 来自全量数据池，vectorized 来自 500 条/子数据集的短跑；不过两组平均 `seen_sid_removed` 接近，分别约 10.15 和 9.93。Legacy 中该值与 step 时间的 Pearson 相关约 `+0.57`，vectorized 的 20 步小样本中不再呈正相关。

1 秒 `nvidia-smi` 采样得到 vectorized GPU 0/1 峰值显存约 49,467/51,547 MiB，平均利用率约 94.3%/95.6%。正式 legacy C 停止前观测到约 52,869/52,729 MiB；因 legacy 未使用同一采样器，这里只作为观测值，不声称是严格峰值对照。短跑无 OOM、NaN 或 Traceback，最终 adapter 与 checkpoint-20 正常保存。

原正式 legacy C 在获得用户授权后于 checkpoint-500 完整落盘后停止，原 YAML、输出目录和 checkpoint 均未覆盖。A0 的 GPU 2/3 训练在整个测试期间保持运行。

## 尚未处理

- CPU 侧动态 Trie 仍使用可靠的集合逻辑，尚未改为计数 Trie。
- 未实现 cost-aware rank balancing，避免改变数据顺序、dropout 轨迹和浮点归约顺序。
- 未使用 `torch.compile` 或 Triton。
- C-fast 不解决 teacher forcing 本身的限制，也不增加 Illegal Top-K 或约束解码。

是否把 vectorized 改为未来默认路径，必须等真实 8B 短跑证明收益、显存可接受且双卡结果稳定后再决定。
