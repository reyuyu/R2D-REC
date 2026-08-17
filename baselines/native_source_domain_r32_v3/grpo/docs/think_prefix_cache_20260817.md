# Think G=4 Shared-Prompt KV Reuse 实验（2026-08-17，64 groups x 256+256 CoTs）

## 方法
- Production 确认：_generate_single_turn 将同一 prompt 复制 4 份 batch 生成（forward [4, prompt_len]，
  prompt prefill 计算 4 次）。
- 实验：prefix=full_ids[:-1] batch=1 prefill -> cache.batch_repeat_interleave(4) ->
  原生 model.generate(batch=4 full prompt + past_key_values=cache, do_sample=True,
  T=0.9/top_p=0.95, max_new=2048, per-sample </think> stop)。
- Stage1：32 prompts（short/median/long）next-token 分布等价性；Stage2：64 groups x G=4
  采样（baseline/cache 同 prompts、同 seed schedule），4 卡并行分片 + checkpoint。

## Stage1 分布等价性（正式 bf16+FA2，temp 0.9/top_p 0.95 处理后）
- raw logits：max abs diff mean 0.849、mean abs diff mean 0.142（bf16 浮点噪声量级）
- 处理后分布：JS=0.0、KL=0.0、mass L1=0.0；top20/top50 overlap=1.0、top-1 agree=100%、
  top-p support overlap=1.0
- 内部一致性：baseline/cache 4 行 logits 互差 max=0.0（cache repeat 后确定一致）

## Forward shape proof
- cache 首次 forward = [4, 1]（无 [4, prompt_len] 重复 prefill）-> 优化技术上生效

## Stage2 采样统计（64 groups / 256+256 CoTs）
| 指标 | baseline | cache |
|---|---|---|
| closure rate | 0.9844-1.0 | 0.9844-1.0（无系统偏移，各 1 组 no-close 互换） |
| length mean / p50 / p90 / p95 / max | 694-722 / ~700 / ~820 / ~860 / 945-2048 | 674-720 / ~700 / ~810 / ~850 / 871-1051 |
| group 内 unique ratio | 1.0 | 1.0（无多样性下降） |
| duplicate pairs | ~0.01 | ~0.01 |
| group 内 length std | 83-104 | 96-109 |

## 性能（CUDA synchronized，per group）
| shard | baseline sec | cache sec | speedup |
|---|---|---|---|
| 0 | 63.12 | 63.37 | 0.996 |
| 1 | 63.22 | 58.50 | 1.081 |
| 2 | 61.70 | 60.89 | 1.013 |
| 3 | 60.64 | 59.70 | 1.016 |
| 合并 | 62.2 | 60.6 | **1.027** |

- prefill 仅 ~0.2s、repeat ~0.002s；cache gen 与 baseline gen 几乎相同（decode 主导）
- peak allocated：26.8GB -> 26.5GB（几乎无差）
- 分档（shard0）：short(549-959) 1.053、medium(1033-1383) 0.982、long(1625-2229) 0.962
  ——长 prompt 无收益甚至更慢（batch=4 FA2 prefill 效率高于 1 行 prefill+增量）

## REDUNDANT_STOP_STRINGS = true
production 同时存在 token-id StoppingCriteria（_ThinkStop）与
generation_config.stop_strings=["</think>"]（transformers 构造 StopStringCriteria，
每 decode step 检查）——两套并存。留待单独微基准，本轮未改动。

## 结论
THINK PREFIX CACHE REJECT: insufficient speedup（合并 speedup 1.027 < 1.05；
理论收益上限 ~3-4%（decode 主导），长 prompt 档甚至 0.962 无收益）
