# Beam32 Context Micro-Batch Re-benchmark（2026-08-17，最终正确路径）

背景：旧 cbs=2/4 OOM 结论基于 ~6300-token context；final smoke 真实 prompt+CoT 已缩短，
需用真实 closed Think CoT 重测。

## generate_batch padding 修正（本轮）
- 输入 left pad 原用 zeros（token 0 = 普通字符 '!'）-> 改为真实 pad token 151643
  （<|endoftext|>）；生成 pad_token_id 同步统一（原为 eos 151645）。
- 输出按各自 context 长度切片 + 尾部 pad 清理（cbs=1 下更干净，结果不变）。

## 决定性诊断（diag_beam_len.py）
- 等长 batch（2x 同一 context）：两个均 invalid=0/32 -> batch>1 本身无问题。
- 不等长 batch（短+长 / 长+短）：短 context（被 left-pad）beam 输出全 pad
  （invalid=32/32），长 context 正常 -> transformers 5.3 + Qwen3 + FA2 的
  beam search 对 left-padded 不等长输入的缺陷（显式 position_ids 无效，排除位置因素）。

## 结果（真实 context：typical = final smoke Think chunk0 prompt+CoT 1558-2607 tokens；
long = 最长 4 个 think prompt+CoT 3612-3664 tokens；beam 语义不变 32/32/128）

| context set | cbs | wall | sec/cot | peak alloc | reward parity | status |
|---|---|---|---|---|---|---|
| typical | 1 | 40.1s | 10.0s | 43.8GB | OK (0/0/0/0, 1 AB) | PASS |
| typical | 2 | 145.2s | 36.3s | 71.2GB | FAIL (short ctx 32/32 invalid) | BROKEN |
| typical | 4 | - | - | OOM | - | OOM |
| long | 1 | 68.5s | 17.1s | 54.9GB | OK (2 exact/1 AB/2 A) | PASS |
| long | 2 | - | - | OOM | - | OOM |
| long | 4 | - | - | skip | - | SKIP |

## 结论与推荐
- cbs=1 是唯一正确选项（typical 10.0s/cot、long 17.1s/cot；每 CoT 严格 32 candidates、
  归属不混、exact/AB/A/reward/invalid 与基准一致）。
- cbs>=2 不可用：正确性已坏（短 context 全 pad -> invalid 32/32 -> reward 失真），
  且无加速收益（typical cbs=2 实测 0.28x 反而慢 3.6x，peak 71.2GB 逼近 80GB）。
- 推荐 BEAM_CONTEXT_BATCH=1（与 smoke 生产路径一致）。
