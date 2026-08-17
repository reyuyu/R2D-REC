# Beam32 Prefix-KV Group-Level Equivalence Audit（2026-08-17，32 groups x 128 CoTs）

## 方法
- 32 真实 Think groups（gids[:32], seed 20260816），每组 G=4 sampled CoTs（T=0.9/top_p=0.95,
  per-sample </think> stop），同一批 128 CoTs 同时过 baseline（production generate_batch cbs=1）
  与 prefix-KV（prefill ctx[:-1] once + cache.batch_repeat_interleave(32) + native generate）。
- 统一 canonicalization（两边 slice + trailing pad strip）；CUDA-synchronized timing；
  轻量记录（candidate ids 即时比较不存储）+ checkpoint 续跑。

## 结果（32 groups / 128 CoTs）
| 指标 | 值 | 门槛 | 判定 |
|---|---|---|---|
| cand token-id parity | 0.0% | - | - |
| SID-set parity | 0.0% | - | - |
| raw reward parity | 75.78%（31/128 mismatch） | - | - |
| reward mean bias | +0.1356（MAE 0.437, max 7.75） | 无明显 bias | 有偏 |
| exact/AB/A/invalid agg | 44=44 / 28→27 / 96→88 / 14=14 | - | AB/A 有差异 |
| argmax agreement | 56.25% | >=97% | FAIL |
| full ranking agreement | 56.25% | - | - |
| top2 agreement | 68.75% | - | - |
| adv sign agreement | 67.97% | >=99% | FAIL |
| adv cosine mean/p50/p10/min | 0.706 / 1.0 / -0.577 / -0.577 (n=15) | >=0.99 | FAIL |
| adv mean abs diff / max | 0.393 / 2.731 | - | - |
| zero-std: base0->mixed / mixed->0 | 5 / 3（8/32 组转换） | 极少 | FAIL |
| sign flips（+->- / -->+） | 4 / 5（9 个） | 不明显 | FAIL |
| pearson surrogate | 0.557 | - | - |
| learning-signal distortion | 50.29 | - | - |
| determinism: base / cache(ids) / cache(reward) | 8/8 / 1/8 / 8/8 | Prefix-KV 自身确定 | FAIL |

## 性能（synchronized）
- baseline Beam：mean 8.58s/cot（p50 8.25 / p90 12.06 / p95 13.02 / max 13.81）
- prefix-KV：prefill 0.28s + repeat 0.011s + beam 2.20s = total 2.49s/cot（p50 2.32 / max 17.47）
- speedup 3.44x；peak 47.1GB -> 30.9GB（reserved 78.1 -> 74.0GB）
- CoT gen：58.97s/group（4 CoTs batch）
- （本批 gids[:32] 无 >3000 长 prompt，long bucket n=0）

## ETA（参考）
Think rollout = gen(59s) + 4xBeam；baseline 387x(59+34.3)=36.1k+2.3k+1.5k≈11.1h；
prefix-cache 387x(59+10.0)=26.7k+2.3k+1.5k≈8.5h；节省约 2.6h（不采纳时仅参考）。

## 结论
BEAM PREFIX CACHE REJECT
核心原因：group-level 等价性不成立——advantage sign agreement 仅 67.97%、
argmax 一致仅 56.25%、mean advantage cosine 0.706（p10 为负）、zero-std↔mixed
转换 8/32 组、learning-direction sign flips 9 个；且 Prefix-KV 自身不 deterministic
（同一 CoT 两次运行 candidate ids/SIDs 仅 1/8 一致，reward 恰好 8/8 一致）。
生产保持 BEAM_CONTEXT_BATCH=1 原生路径。
