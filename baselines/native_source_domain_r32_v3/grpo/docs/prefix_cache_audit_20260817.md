# Beam32 Prefix-KV Group-Level Equivalence Audit（2026-08-17，32 groups x 128 CoTs）

## 方法
- 32 真实 Think groups（gids[:32], seed 20260816），每组 G=4 sampled CoTs（T=0.9/top_p=0.95,
  per-sample </think> stop），同一批 128 CoTs 同时过 baseline（production generate_batch cbs=1）
  与 prefix-KV（prefill ctx[:-1] once + cache.batch_repeat_interleave(32) + native generate）。
- 统一 canonicalization（两边 slice + trailing pad strip）；CUDA-synchronized timing；
  完整逐 CoT candidate/SID/reward 记录 + checkpoint 续跑。详细版本见
  `docs/prefix_cache_audit_detailed_20260817.md`。

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
| adv cosine mean/p50/p10/min | 0.706 / 1.0 / -0.568 / -0.577 (n=15) | >=0.99 | FAIL |
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
- 同步 long 补测（context 3699/3773/3664/3581）：baseline 17.20s/cot，
  Prefix-KV 3.84s/cot，speedup 4.48x，reward parity 3/4。

## ETA（参考）
按 4GPU global block 的最慢 rank 估算（避免把四张卡时间相加）：
baseline Think block mean 106.44s，Prefix-KV Think block mean 77.09s。
代入 `387*Think + 774*NoThink + 2322*policy_update` 后：
baseline 12.49h，Prefix-KV 9.34h，理论节省 3.15h。
该 ETA 仅为性能参考，不能抵消 group-level 等价性失败。

## 结论
BEAM PREFIX CACHE REJECT
核心原因：group-level 等价性不成立——advantage sign agreement 仅 67.97%、
argmax 一致仅 56.25%、mean advantage cosine 0.706（p10 为负）、zero-std↔mixed
转换 8/32 组、learning-direction sign flips 9 个；且 Prefix-KV 自身不 deterministic
（同一 CoT 两次运行 candidate ids/SIDs 仅 1/8 一致，reward 恰好 8/8 一致）。
生产保持 BEAM_CONTEXT_BATCH=1 原生路径。
