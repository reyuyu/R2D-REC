# REC-MP-GRPO-v1 Correctness Round Report（2026-08-17）

本轮只做 correctness 修复与验证：无训练、无 backward、无 optimizer.step。

## 1. route-specific reward（已修复并验证）
- 原问题：两个 reward_func 被 TRL 对所有样本调用并相加（cross-route 串扰），
  NoThink 样本也触发 Beam32。
- 修复：两个 reward_func 读取 TRL reward kwargs 的 dataset 列 `route`；
  Think 样本：think_reward 数值、nothink_reward 返回 None（TRL 转 NaN，nansum 排除）；
  NoThink 样本：反之，且 think_reward 直接短路返回 None，Beam32 完全不调用
  （先筛 think_idx，空则整批返回 None）。
- 验证：THINK batch 的 nothink_reward 列全 NaN、think_reward 列全数值；
  NoThink batch 的 think_reward 列全 NaN；**NoThink 期间 Beam32 call count 严格为 0**
  （GPU rollout-only 实测：beam calls 在 NoThink 前后保持 1，n_cot 保持 8）。
- route 列缺失时 raise（防止静默退回错误行为）。

## 2. 真正的 dynamic G sampler（RouteAwareRepeatSampler）
- 原问题：sampler 初始化 G=4 后 _prepare_inputs 临时改 num_generations=8 ——
  TRL RepeatSampler 的 mini_repeat_count 创建后不变，NoThink 实际是 4 unique x 4 repeats，
  view(-1,8) 把 2 个不同 prompt 的重复混进同一组（确认存在）。
- 修复：RouteAwareRepeatSampler（TRL Sampler 子类）按 dataset 行 route 动态切 chunk：
  - global generation batch 恒为 16
  - Think: 4 unique x 4 repeats = A A A A B B B B C C C C D D D D
  - NoThink: 2 unique x 8 repeats = A A A A A A A A B B B B B B B B
  - chunk unique = generation_batch_size // G（4 / 2）；同 route 段内步进切 chunk，
    尾段不足则丢弃（与 TRL drop_last 一致）
  - repeat_count = num_iterations x steps_per_generation，同一 chunk 连续重放
    -> TRL 前 2 步复用同一 rollout（num_iterations=2 复用正确）
- 硬断言（_generate_and_score_completions 内，真 rollout 执行）：
  每个 rewards.view(-1, G) 组内所有 recommendation_group_id 必须完全相同，
  否则 assert 失败（gather_object 与 rewards gather 同序）。
- 实测展开（单卡 gen_batch=8）：Think chunk gids 2 个唯一值 x 4 重复、
  NoThink chunk gids 1 个唯一值 x 8 重复；12 chunks 顺序
  T,T,T,T,N,N,N,N,N,N,N,N（x2 重放 = 24 个 rollout batch）。
- 4GPU 正式配置（gen_batch=16）下 chunk 序列：
  T(2 chunks), N(4 chunks) 交替 -> T,T,N,N,N,N,T,T,N,N,N,N,T,T,N,N,N,N。

## 3. group / route 等权（loss multiplier）
- Think loss multiplier = 1.0；NoThink = 0.5（ROUTE_LOSS_W）。
- 数学验证：
  - 单个 Think group 总权重 = 4 samples x 1.0 = 4
  - 单个 NoThink group 总权重 = 8 samples x 0.5 = 4  -> 相等
  - epoch：1549 组 x 4 = 6196（Think）== 1549 组 x 4 = 6196（NoThink）
  - 实现：_compute_loss 在 per-sample token-mean 后乘 multiplier 再 mean
    （route 经 generation output 传入，buffered 复用期不变）。
- 无 gold_count weighting（保持）。

## 4. hierarchical reward prefix 去重（think_credits）
- 原问题：per-gold 互斥计数，Exact 命中的 gold 的 AB/A 前缀仍会被同前缀的其他
  gold 重复领取（示例：golds (A1,B1,C1),(A1,B1,C2)，beam exact (A1,B1,C1)
  -> 原实现 credits [8, 2]，错误）。
- 修复（全局 prefix 排除，reward 数值不变）：
  - ABHits = matched_AB_prefixes - exact_covered_AB_prefixes
  - AHits  = matched_A_prefixes - exact_or_AB_covered_A_prefixes
- 单测：用户示例 -> [8]；AB-matched 非 gold -> 仍 [8]；独立 AB -> [8,2]；
  AB 覆盖 A -> 仅 [2]；纯 A -> [0.5]；无命中 -> []；几何衰减数值不变
  ([8,2] -> 9.0)。

## 5. prompt token parity（SFT / TRL / Beam32）
- 审计发现：TRL maybe_apply_chat_template 对非 conversational（str）prompt
  原样返回 -> rollout 输入是裸 prompt 文本（无 chat 模板），与 SFT 不一致。
- 修复：shared renderer `render_prompt`（grpo_model.py，基于 llamafactory
  qwen3_nothink format_user 渲染，含 assistant 头）；TRL _generate_single_turn
  与 Beam32 encode_prompt 均复用。
- 验证（audit_prompt_parity.py，3 个随机真实 prompt）：
  A（SFT llamafactory encode_multiturn ids）== B（TRL render_prompt ids）
  == C（Beam32 encode_prompt ids），长度 1232/1541/1413 全等 -> ALL EQUAL。

## 6. no-closure 策略：不变（本轮不改 reward 数值/策略）。

## 测试结果（CPU + 单卡 GPU rollout-only）
- test_grpo_reward.py: ALL TESTS PASSED
- test_grpo_trainer.py: ALL TRAINER TESTS PASSED
- test_grpo_trl.py: ALL TRL TRAINER STRUCTURE TESTS PASSED
- test_correctness_v2.py: ALL CORRECTNESS V2 TESTS PASSED（29 项）
- rollout_only_check.py（GPU0，无训练）：THINK 42.6s / NOTHINK 42.0s；
  routing NaN 模式正确；NoThink Beam call count == 0；group_id 断言通过。
- prompt_parity_audit.json：ALL EQUAL。

结论：GRPO CORRECTNESS READY FOR RESMOKE
