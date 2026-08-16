# REC-MP-GRPO-v1 正式 epoch 前核查报告（2026-08-17）

## 问题 1：Route 1:1 全 epoch 检查
- G=4 静态模拟：1548 policy steps = 774T + 772N + 2 boundary MIX，1:1 成立。
- 结构性冲突（动态 G=T4/N8 + TRL 固定 16 样本 batch）：NoThink 组跨 prompt 混合（view(-1,8)）、
  N 组权重 = 2x T 组权重。rollout 1:1 / 正确 8 样本组归一化 / 组等权三者只能满足其二。
  状态：待用户决策（不变）。

## 问题 2：Think </think> 停止机制（本轮深挖）
- </think> = 单 token 151668（tokenizer.encode 确认 [151668]）。
- token-id StoppingCriteria 正常：单序列 probe 8/8、repro 4/4、bench_fixed2 16/16 全部 closure，
  位置 538-836 tokens；SFT 格式为裸 <think>...</think>（无 <|im_start|> 前缀），模型正确复现。
- 【新 Bug，已修复】TRL 路径 _ThinkStop 用标量 bool(.any()) -> batch 16 序列中任一 closure
  即整批停止（验证：4 prompt x 4 序列总长全部 = S=1922，完美吻合；smoke 的 think 补全
  mean 142 tokens 即此 bug 产物，补全被集体截断、残缺）。
  修复：返回 per-sample BoolTensor (input_ids[:,-1] == tid)，per-sample 测试验证各序列
  独立 closure（pos 632-1745）。备份：grpo_trl_trainer.py.bak_anyfix。
- 【新事实】采样下 closure 率约 69%（11/16）：约 30% think 补全无 </think>，跑到
  max_completion_length=2048。设计决策待定：无 closure 补全如何计分
  （现状：全文当答案 -> beam32 解析；可选：低分/无效；或提高 cap）。
- NoThink 无需停止：天然短（mean 95 字符 ≈ 22 tokens，SID 首现 ~32，+18-19 收尾），
  32 序列批量 10.3s。

## 问题 3：Beam invalid 15.9% 根因
- 根因：TRL batch_decode(skip_special_tokens=True) 剥离 </think> 与 SID added tokens，
  beam32_fn 拿到损坏补全（smoke 的 closure 1/24 同为此 artifact）。
- 修复：beam32_fn 从 completion_ids 重新 decode（skip_special_tokens=False）。
- 验证：invalid 0/512 = 0.0%；beam 9.7s/cot（155.3s/16 cots）。

## 附带：bench_fixed 的 0/16 之谜
- bench_fixed（00:57-01:19）：0/16 closure、34 tok/s、2048 满长，与所有其他运行矛盾。
- verbatim 复跑 bench_fixed2（01:38）：16/16 closure、947.1s（vs bench 955.9s，时间几乎相同）。
- 结论：0/16 为不可复现的一次性环境异常（时间一致但内容 0 closure），代码逻辑正确。

## 1 epoch 时间重估（修复后）
- T rollout：174.3s/rollout（per-sample 实测 16 序列/rank，含 ~30% 无 closure 跑满 2048）。
- N rollout：~45s/rollout（4 GPU 同步估计）。
- Beam32：155s/rollout（16 cots x 9.7s，4 rank 并行）。
- 合计 ≈ 97x(174.3+155) + 97x45 ≈ 36,300s ≈ 10.1h + policy 更新开销 ≈ 10.5-11h。
- 与修复前估计（~10.5h）持平；beam 仍是最大单头（~4.2h）。

## 待用户决策
1. NoThink 组结构冲突（三者取二）。
2. 无 closure 补全（~30%）计分策略。
3. 是否重跑 4GPU smoke 验证修复后的正式路径。
