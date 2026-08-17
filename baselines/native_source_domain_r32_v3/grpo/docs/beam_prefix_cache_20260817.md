# Beam32 Shared-Prefix KV Reuse 实验（2026-08-17）

## 方案
- prefill prefix（full_context[:-1]）1 次（batch=1）→ cache.batch_repeat_interleave(32)
  （in-place）→ 原生 model.generate(full_context, past_key_values=cache32,
  num_beams=32, num_return_sequences=32, max_new_tokens=128)。不重写 Beam Search。
- Forward hook 证明：beam 首次 forward = [32, 1]（cache_position=len(prefix)），
  全部后续 forward 均为 [32, 1] —— 无 32 x full_context 重复 prefill，优化生效。
- prefill 的 forward shape 未记录（脚本 clear 时机 bug），但 cache.get_seq_length()
  assert（== len(prefix)）通过，且 beam_first=[32,1] 已充分证明。

## 性能（stage1：4 typical + 4 long closed CoTs）
| 组 | baseline sec/cot | prefix-cache sec/cot | speedup | baseline peak | cache peak |
|---|---|---|---|---|---|
| typical | 9.3s | 2.5s | 3.78x | 44.0GB | 29.5GB |
| long | 16.6s | 3.8s | 4.40x | 54.6GB | 34.3GB |

（typical 单条 6.7-12.3s -> 2.1-2.9s；long 单条 16.4-17.0s -> 3.6-3.9s）

## 正确性（硬门槛）
- reward parity：**7/8（87.5%）** —— FAIL
  - [long] ctx3640：reward 12.609 -> 8.438（exact 2 -> 1）
- candidate token-id parity：0/8（全部不同）
- beam SID set parity：0/8（全部不同）
- invalid：8/8 一致（0 -> 0）
- stage2（32 CoTs）未运行（stage1 未达 100% 硬门槛，按规则停止）

## 根因
prefix-cache 使 beam 首次 forward 形状为 [32, 1]（baseline 为 [32, len]），
FA2 kernel 分块不同 -> 浮点差异 -> beam 候选早期分叉（与 cbs 跨形状浮点噪声
同源，已在前轮 v2 确认）-> 多数 CoT 的 exact/AB/A/reward 恰好一致，但边界处
exact 计数翻转（ctx3640）。

## 结论
**BEAM PREFIX CACHE REJECT**：性能提升显著（3.8-4.4x，峰值内存 -37%），但
reward parity 未达 100% 硬门槛（exact 2->1 的系统性变化），按规则不采用。
生产保持 BEAM_CONTEXT_BATCH=1 原生 generate 路径。
