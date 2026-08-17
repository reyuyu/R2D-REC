# Think Length-Bucketing 验证（2026-08-17）

## 方案
Think global batch 按 4-group block 做 length bucketing：预计算全部 1549 组 Think
prompt 长度（shared render_prompt + tokenizer），按长度排序后每 4 个相邻组组成一个
block，再只随机 shuffle block 顺序（不从短到长训练）。NoThink/effective weight/
G/reward/loss 全部不变。

## 静态分析（全部 1549 组，不加载 8B）
4 卡长度不平衡（每 block 的 range/std）：
| 指标 | random | bucket | 改善 |
|---|---|---|---|
| mean range | 1056.2 | 5.9 | -99.4% |
| p50 / p90 / p95 / max range | 1024/1626/1823/2538 | 3/9/15/180 | -99%+ |
| mean max/mean | 1.49 | 1.004 | - |
| mean std | 408.3 | 2.4 | -99.4% |
- dropped：random 与 bucket 均 1 group；覆盖组数不变（1548/1548）。

## GPU rollout-only benchmark（单卡模拟 4 rank：G4 生成 + Beam32 cbs=1，同一批 16 组
两种分组）
| block | plens | global wall | straggler | peak |
|---|---|---|---|---|
| random1 | 690/1640/959/1876 | 110.7s | r3 | 44.7GB |
| random2 | 1077/1873/1033/1210 | 118.4s | r0 | 45.3GB |
| random3 | 1383/1736/1037/1625 | 110.8s | r1 | 43.6GB |
| random4 | 916/2229/549/690 | 115.4s | r1 | 48.1GB |
| bucket1 | 549/690/690/916 | 104.0s | r3 | 35.9GB |
| bucket2 | 959/1033/1037/1077 | 109.7s | r3 | 37.9GB |
| bucket3 | 1210/1383/1625/1640 | 105.3s | r2 | 42.0GB |
| bucket4 | 1736/1873/1876/2229 | 128.7s | r0 | 48.9GB |

mean random wall = 113.8s；mean bucket wall = 111.9s；speedup = 1.017（+1.7%）。

## 结论
- 静态不平衡下降 99%+（bucket 有效消除 prompt 长度差异）。
- 但 GPU wall 提升仅 1.7% < 5%：straggler 主要由 CoT 采样长度差异（生成时间）与
  Beam 输入长度（prompt+CoT）驱动，prompt 长度排序无法控制 CoT 长度随机性。
- 按规则（<5% 不采用）：**不采用 length-bucketing，保持当前排序**。
- 附带收益：bucket 前 3 个 block 的 peak 内存明显更低（35.9-42.0 vs 43.6-48.1GB），
  但不足以改变决策。
