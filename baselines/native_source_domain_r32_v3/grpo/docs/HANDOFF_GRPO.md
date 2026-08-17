# REC-MP-GRPO-v1 Handoff & 脉络说明（2026-08-17，merged to main @ 925b060）

## 0. 项目定位
在 onereason-multitask-sft（SFT 完成，BATA-BASELINE-R32-2E 为最终 adapter）之上的
强化学习阶段：只做"推荐"单一任务的 Multi-Positive Outcome GRPO，Think（显式思考，
hierarchical credits）与 NoThink（直接回答，六档 q()）双路线。所有代码在
`baselines/native_source_domain_r32_v3/grpo/`（开发机 `/data/GRPO/` 为工作副本）。

## 1. 冻结约束（不可改）
- Reward：NoThink q() 六档（-1/-0.25/0/0.5/2/8，互斥）；Think 分层学分（Exact=8/AB=2/A=0.5，
  全局 prefix 去重，几何衰减 R=sum(c_j*0.5^(j-1))）；gold_count 仅元数据。
- 超参：G=4(Think)/8(NoThink)、lr=1e-6、beta=0、epsilon=0.2、loss_type="grpo"、
  scale_rewards="group"（总体 std）、disable_dropout=True、importance_sampling_level="token"、
  top_entropy_quantile=1.0、mask_truncated_completions=False、max_prompt_length=8192、
  max_completion_length=2048、use_vllm=False。
- 禁止：Liger、vLLM、continuous batching、structured SID decoding、site-packages 修改、
  SFT 工程修改、gold_count weighting、no-closure reward 策略变更（截至 2026-08-17）。
- Think 采样：T=0.9/top_p=0.95；NoThink：T=1.0/top_p=1.0（三处同步：args/self/generation_config）。
- BEAM_CONTEXT_BATCH=1（唯一正确；cbs>=2 在不等长 left-pad 下被 transformers 5.3 beam
  产生错误，见 beam_cbs_v2）。

## 2. 环境
- 开发机 root@103.102.203.0:1058（SSH key ~/.ssh/id_ed25519_onereason_multitask_sft），
  venv /data/venvs/llamafactory-01398eb-liger081（Python 3.11，TRL 0.24.0、transformers 5.3.0、
  torch 2.5.1+cu124、peft 0.18.1、FA2），4x A800-80GB（GPU 与用户 SFT 共享，容器 cgroup 内存
  有上限——大内存实验需 checkpoint/轻量存储）。
- 模型：BASE /data/models/onereason-8b-pretrain-competition +
  ADAPTER /data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333
  （bfloat16 + FA2，grpo_model.load_model）。
- 数据：/data/GRPO/data/rec_mp_grpo_v2/train.jsonl（3098 条 = 1549 组 x Think/NoThink；
  video 1100/ad 854/prod 764/living 380；gold_count mean 7.11/p50 3；manifest 见
  results/data_manifest.json）。数据本体 27MB 不入库。
- GitHub：reyuyu/onereason-multitask-sft，main @ 925b060（GRPO 已合并）；
  agent/grpo-rec-mp-v1-sync @ b281033（工作分支，保留）。

## 3. 代码结构（scripts/）
核心生产：
- grpo_model.py：load_model（BASE+ADAPTER+FA2）、render_prompt（SFT 一致 qwen3_nothink 渲染，
  三路径 parity 唯一事实源）、encode_prompt、generate_batch（真 left padding、统一 max_len 切片、
  尾部 pad 清理、return_ids）。
- grpo_sid.py：SID_RE/parse_sid/final_sid/all_sids、q_reward（NoThink 六档）、
  think_credits/think_reward（hierarchical + 全局 prefix 去重：ABHits=matched_AB-exact_covered_AB、
  AHits=matched_A-exact_or_AB_covered_A）、raw_w（仅记录）。
- grpo_trl_trainer.py：RecGRPOTrainer(GRPOTrainer)——RouteAwareRepeatSampler（真 dynamic G：
  Think 4u x4 / NoThink 2u x8，global batch 恒 16，repeat_count=num_iterations*steps 复用）、
  route-specific reward routing（route 列缺失即 raise；NoThink 严格不触发 Beam32）、
  route_id tensor（shuffle/split 安全）、route_multiplier（Think 1.0/NoThink 0.5）、
  population-std advantage、group_id 硬断言（每 view(-1,G) 同 group）、per-sample </think>
  StoppingCriteria（token 151668）、temp/top_p 三处同步 + runtime assert、
  build_route_dataset（chunk=8 交替 T/N）。
- run_grpo_trl_smoke.py / launch_grpo_trl_smoke.sh：4GPU FINAL RESMOKE 入口
  （n_groups=8 完整 T,T,N,N,N,N 周期，12 steps，contract assert，closed/no-close 分类统计，
  gen wall/prompt lens，ETA 计算）。
- trl_import_fix.py：TRL 0.24 可选依赖探测修正（零安装）。

测试（CPU）：
- test_correctness_v2.py（45 项：routing/Beam count=0/sampler 展开/group_id/multiplier/
  prefix 去重/raw SID/route_id 管线/renderer parity）
- test_grpo_reward.py、test_grpo_trainer.py、test_grpo_trl.py（回归）

审计/benchmark/诊断（GPU 或 CPU，均为一次性验证，结论见 docs/）：
- audit_prompt_lengths.py（全量长度分布 + n_over_512/8192）
- audit_prompt_parity.py（SFT/TRL/Beam 三路径 token ids parity）
- audit_4proc_sampler.py（torchrun 4 进程真分布式 sampler/dataloader 验证）
- audit_epoch_static.py（1549 组全 epoch：1548/1548 组、387/774 rollouts、2322 steps）
- audit_think_bucketing.py / bench_think_bucketing.py（length-bucketing 静态+GPU）
- bench_beam_cbs_final.py / bench_beam_cbs_v2.py（cbs 1/2/4 基准，v2 修复 left pad）
- bench_beam_prefix_cache.py / bench_prefix_cache_audit.py / bench_prefix_cache_long_sync.py
  （Beam prefix-KV 与 group 级审计）
- bench_think_prefix_cache.py（Think G4 shared-prompt KV，4 卡分片）
- diag_*.py（beam cbs/浮点/等长诊断）、probe_think_closure.py、per_sample_test.py、
  batch_think_test.py、rollout_only_check.py（GPU 完整路径验证）

## 4. 演进脉络（commit 顺序，main 视角）
1. GRPO 数据与早期 smoke：rec_mp_grpo_v2 构建、M 消融（结论 M_NO=8：mixed 96.9%）、
   run_smoke/run_grpo_smoke（TRL 接入前验证）、run_grpo_trl_smoke（TRL 0.24 集成）。
2. 正式前 3 问题核查（fix_report_20260817.md）：route 1:1 全 epoch 模拟；Think 停止机制深挖
   ——修复 `.any()` 标量 StoppingCriteria 导致 batch 集体早停（smoke think 补全被截断在
   ~142 tokens 的根因）→ per-sample BoolTensor；Beam invalid 15.9% 根因（TRL
   skip_special_tokens=True 剥离 tokens）→ completion_ids 重解码 → 0%。附带确认
   bench 0/16 closure 为一次性环境异常。
3. correctness 轮（correctness_report_20260817.md）：route reward 串扰修复（route 路由 +
   NoThink Beam count=0）；RouteAwareRepeatSampler（真 dynamic G，替代"init G=4 后临时改 8"
   的错误 hack——原实现把 2 个不同 prompt 混进 G=8 组）；loss multiplier Think 1.0/NoThink 0.5
   （单组 4==4，epoch 6192==6192）；hierarchical prefix 去重（Exact 覆盖的 AB/A 不重复领取）；
   shared renderer（TRL 原把 str prompt 原样透传 → 裸文本无模板，修复后三路径 parity ALL EQUAL）。
4. final correctness 轮（final_correctness_20260817.md）：max_prompt_length 8192（原 512
   截断 92.3% 样本；全量 p50=1142/p95=2003/max=3036 ≤8192；8192+2048<131072 不阻塞）；
   NoThink raw completion_ids 解析 SID（防御性，本 tokenizer 下 SID tokens 非 special）；
   route_id tensor（标量字符串会被 shuffle 按字符索引破坏）；temp/top_p 三处同步+assert；
   4 进程真分布式 audit；全 epoch 静态审计。
5. FINAL RESMOKE（REC-MP-GRPO-V1-FINAL-SMOKE-rank*.json）：4x A800，12 steps / 308s，
   route 序列 T,T,N,N,N,N（2T+4N rollouts），iteration1 ratio=1/clip=0，iteration2
   ratio 1.001-1.005/KL finite，routing NaN 模式正确、NoThink beam call=0、closure 8/8、
   invalid 0/256、LoRA delta 6e-5/base 0、peak 38.2GB；ETA = 387x128.5 + 774x3.0 +
   2322x0.63 ≈ 14.9h。**GRPO FINAL RESMOKE PASSED**。
6. Beam cbs 重新基准（beam_cbs_final/v2）：旧 cbs=2/4 OOM 结论基于 ~6300-token context 失效
   ——先发现 generate_batch 实为 RIGHT padding（短 context 全 pad 输出被误归因
   transformers defect，已更正），修正为真 left padding + 统一 max_len 切片；cbs=2 修复后
   功能正确（invalid=0、reward parity）但仍慢 1.3x 且长 context OOM → 正式冻结
   BEAM_CONTEXT_BATCH=1。
7. Think length-bucketing（think_bucketing_20260817.md）：静态不平衡 -99%（range 1056→5.9），
   但 GPU wall 仅 +1.7%（straggler 由 CoT 采样长度主导，prompt 排序无法控制）→ 不采用。
8. Beam prefix-KV（beam_prefix_cache_20260817.md）：3.8-4.4x 提速、forward proof [32,1]，
   但 raw reward parity 7/8（exact 2->1）→ 按 100% 门槛 REJECT。
9. Prefix-KV group 级审计（prefix_cache_audit_20260817.md + detailed）：128 CoTs 完整证据
   ——adv sign agreement 67.97%、argmax 56.25%、cosine 0.706、zero-std↔mixed 转换 8/32、
   sign flips 9、Prefix-KV 自身不 deterministic（ids 1/8）→ REJECT。
10. Think G4 shared-prompt KV（think_prefix_cache_20260817.md）：分布等价完美（JS=0、
    top-p support=1.0、4 行 logits 一致 0.0），但 speedup 仅 1.027（decode 主导，
    batch=4 FA2 prefill 效率抵消；short 1.05/medium 0.98/long 0.96）→ REJECT insufficient
    speedup。
11. 合并 main（925b060，--no-ff，无冲突）。

## 5. 当前状态与待办
- 状态：correctness 全部 PASS、FINAL RESMOKE PASSED、生产路径冻结
  （BEAM_CONTEXT_BATCH=1、原生 generate、无任何 prefix-cache/bucketing）。
- 待决策（用户）：
  a) NoThink 组结构冲突：TRL 固定 16 样本 batch + 动态 G 下，rollout 1:1 / 正确 8 样本
     组归一化 / 组等权，三者只能取二（当前：rollout 1:1 + loss multiplier 0.5 等权，
     组归一化跨 prompt 混合）。
  b) no-closure 策略（~30% Think 补全无 </think>，当前全文当答案）：
     是否 reward=0 + skip Beam（下一轮可做 closed vs no-close 已收集数据：
     final resmoke closure 8/8 本轮样本全 closed；pre-fix 观察 ~69%）。
  c) REDUNDANT_STOP_STRINGS=true（token-id StoppingCriteria + stop_strings 并存）：
     留待单独微基准，删除与否未定。
  d) 若未来重启 prefix-KV 类优化：先做 downstream Beam reward / GRPO 统计 sanity check
     （本轮判定规则要求）。
- 未启动正式 1 epoch（用户明确：任何时刻需先确认）。

## 6. 快速上手
- 结构测试：venv python test_correctness_v2.py（CPU ~40s）
- 4GPU FINAL RESMOKE：bash launch_grpo_trl_smoke.sh（~6 分钟，308s）
- 审计复跑：audit_*.py / bench_*.py（各自单卡或 4 卡分片）
- 改代码注意：grpo_trl_trainer.py 是唯一生产 trainer；新增实验放独立脚本（勿改 production）。

## 7. 已知坑
- FA2 + 跨 batch 形状 = beam/采样候选浮点分叉（同形状确定；Beam 是 reward evaluator，
  任何形状变化都需 group 级审计——已两次 REJECT）。
- 容器 cgroup 内存上限：大实验必须轻量记录 + checkpoint（曾 OOM 一次）。
- GPU 与用户 SFT 共享：跑 GPU 实验前查 nvidia-smi；SFT 训练在容器 namespace
  （宿主 ps 可见，nvidia-smi pid 不可见）。
- TRL 0.24 的 generation_kwargs 不转发到 generate；stop_strings 走 transformers
  5.3 的 StopStringCriteria（与自定义 StoppingCriteria 并存）。
