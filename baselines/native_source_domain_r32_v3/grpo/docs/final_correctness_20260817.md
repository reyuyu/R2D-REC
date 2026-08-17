# REC-MP-GRPO-v1 Final Correctness Report（2026-08-17，resmoke 前）

本轮无训练 / 无 backward / 无 optimizer.step。

## 1. max_prompt_length=512 隐式截断（已修复）
- 全量 3098 条 render_prompt 统计（audit_prompt_lengths.py）：
  p50=1142, p90=1866, p95=2003, p99=2443, max=3036, min=46, mean=1163.6
- **原 TRL 默认 max_prompt_length=512 确认存在，且 100% 样本被截断**（连 p50 都超 512）。
- model config max_position_embeddings = 131072；tokenizer.model_max_length = 131072；
  SFT baseline cutoff_len = 8192（全部 config 一致）。
- 全部 prompt <= 8192（n_over_8192=0）-> 显式设置 max_prompt_length=8192。
- max_prompt_length(8192) + max_completion_length(2048) = 10240 << 131072 -> 不 BLOCKED。
- 真实 parity：4 进程 audit 中 TRL _generate_single_turn 实际 tokenizer/truncation 后
  prompt_ids 与 SFT llamafactory ids 完全一致（4/4 True，len 1115）。

## 2. NoThink reward raw SID parse（已修复）
- make_nothink_reward_func(tokenizer)：从 raw completion_ids decode(skip_special_tokens=False)
  -> final_sid -> q_reward（不再依赖 TRL 解码的 completions 文本）。
- 实测记录：本 tokenizer 下 <|living_begin|> 与 <s_a_N> 等 SID tokens 均非 special tokens，
  skip_special_tokens=True/False 两种 decode 都能解析出 SID（q=8）；<|im_start|>/<|im_end|>
  会被 True 剥离。差异记录：本场景无解析差异；raw decode 为唯一事实源（消除对 TRL
  解码管线/未来 tokenizer 变化的依赖），且 NoThink 真实 valid SID 保证不被打 -1。
- 验证：reward_func 输出 == raw completion_ids 人工解析（rf=[8.0] manual=8.0）；
  GPU rollout 中 NoThink q 值分布合理（0.0/-0.25 混合，非全 -1）。

## 3. route metadata 改 route_id tensor（已修复）
- generation output 移除标量字符串 "route"，改为 batch-aligned route_id tensor
  （think=0, no_think=1，shape=[local_batch]，dtype=long）。
- shuffle_sequence_dict / split_tensor_dict / buffer reuse 安全处理（tensor 按第一维同步
  permute/split；标量字符串会被按字符索引破坏——已确认原实现的问题）。
- _compute_loss：route_id 全同 assert -> route_multiplier（抽成纯函数，可测）。
- 测试走完整路径：_generate_and_score_completions -> shuffle_sequence_dict ->
  split_tensor_dict -> route_multiplier（CPU 单测）+ GPU rollout 完整 _prepare_inputs
  （生成->buffer->split->route_id 断言）全通过。

## 4. 动态 temperature/top_p 同步（已修复）
- 每次 route 切换同步三处：self.args.temperature / self.temperature /
  self.generation_config.temperature（top_p 同理）。generation 用 generation_config，
  policy logprob scoring 用 self.temperature，TRL 内部读 self.args——三者必须一致。
- runtime assert（_generate_and_score_completions 开头）：Think 时三处 == 0.9/0.95，
  NoThink 时 == 1.0/1.0。GPU 完整路径实测通过（assert 未触发）。

## 5. 4 进程 sampler/DataLoader audit（真实 distributed sharding）
- torchrun --nproc_per_node=4 + dummy 小模型（未加载 8B）+ 真实
  RouteAwareRepeatSampler / DataLoader / Accelerator.prepare（trainer.get_train_dataloader）。
- 8 个全局 batch 实测（gather 后）：batch0-3 think（4 unique group_id x 4 repeats）、
  batch4-7 no_think（2 unique x 8 repeats）；view(-1,G) 内 group_id 全同（ok_gid/ok_unique/
  ok_repeat 全 True）。
- num_iterations=2：同一 rollout 全局 batch 连续复用两次（reuse_pairs=[True x4]）。
- generation_batch_size=16（4 进程下正确推导）。
- 真实展开示例：
  Think:  [gA,gA,gA,gA, gB,gB,gB,gB, gC,gC,gC,gC, gD,gD,gD,gD]
  NoThink: [gE x8, gF x8]

## 6. 全 epoch 静态审计（1549 组真实数据）
- Think unique groups covered = 1548（387 chunks x 4）
- NoThink unique groups covered = 1548（774 chunks x 2）
- dropped：1 组（gid c399e01d...，Think/NoThink 各丢 1 次，同一 base group，1/1549=0.065%）
- Think rollouts = 387；NoThink rollouts = 774
- optimizer steps = 2322（(387+774) x num_iterations=2）
- 与预期 1548/1548/387/774/2322 完全一致。

## 7. loss multiplier 在完整 sampler exposure 下
- route effective weight：Think 387 x 16 x 1.0 = 6192 == NoThink 774 x 16 x 0.5 = 6192
  -> ≈ 1:1 保持。
- 单 group：4 x 1.0 == 8 x 0.5 == 4。

## 8. smoke runner 更新（未启动训练）
- run_grpo_trl_smoke.py：docstring/注释更新为新 route 序列
  （T,T,N,N,N,N,T,T,N,N,N,N,T,T,N,N,N,N；max_steps=12 覆盖前 6 rollout = 3T+3N）；
  make_nothink_reward_func(tokenizer=tokenizer)；GRPOConfig 显式 max_prompt_length=8192；
  启动时打印预期 rollout route sequence。

## 测试结果汇总
- test_correctness_v2.py：45 项 ALL PASS（含 raw SID 对比、route_id 管线、multiplier）
- test_grpo_reward.py / test_grpo_trainer.py / test_grpo_trl.py：ALL PASS
- audit_4proc_sampler.py（4 进程真实分布式）：全 True
- audit_epoch_static.py：与预期完全一致
- audit_prompt_lengths.py：全量分布（<=8192 无截断）
- rollout_only_check.py（GPU 完整路径）：THINK 44.8s / NOTHINK 41.9s，routing 正确、
  NoThink beam call 0、route_id 断言通过、temp/top_p asserts 通过

结论：GRPO FINAL CORRECTNESS READY FOR RESMOKE
