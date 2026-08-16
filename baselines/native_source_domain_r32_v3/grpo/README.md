# REC-MP-GRPO-v1（Recommendation-only Multi-Positive Outcome GRPO）

SFT（BATA-BASELINE-R32-2E，见 `../README.md`）之后的强化学习阶段工程。
数据、脚本与报告均围绕 onereason-multitask-sft 推荐任务（video / prod / ad / living 四域）
展开；本目录为 GRPO 阶段的工作区镜像（开发机路径 `/data/GRPO/`）。

## 目标与约束（冻结语义）
- 只做"推荐"单一任务的正向多目标 GRPO：Think（显式思考）与 NoThink（直接回答）双路线。
- Reward 冻结：NoThink 六档 q()（-1/-0.25/0/0.5/2/8，互斥）；Think 分层学分
  （Exact=8 / AB=2 / A=0.5，几何衰减 R = sum(c_j * 0.5^(j-1))）；gold_count 仅作元数据。
- 超参冻结：G ∈ {4(Think), 8(NoThink)}、lr=1e-6、beta=0（无 KL）、epsilon=0.2、
  loss_type="grpo"、scale_rewards="group"（总体标准差，correction=0）、disable_dropout=True。
- 不使用 Liger / vLLM；不修改 site-packages；不修改 SFT/LoRA 基线工程。

## 目录结构
```
grpo/
├── README.md                 # 本文件
├── docs/
│   └── fix_report_20260817.md # 正式 epoch 前 3 问题核查报告（停止机制/beam invalid/route 1:1）
├── scripts/                  # 全部 GRPO 脚本（加载/奖励/训练器/消融/基准/测试）
└── results/                  # 关键实验结果 JSON（smoke/消融/probe/beam）
```

## 关键脚本
| 文件 | 作用 |
|---|---|
| `scripts/grpo_model.py` | 模型加载（BASE 预训练 + BATA-BASELINE LoRA adapter，FA2，bfloat16）+ 批量生成 |
| `scripts/grpo_sid.py` | SID 正则解析、`final_sid`、NoThink 六档 q()、Think 分层学分（纯函数，可测） |
| `scripts/grpo_trl_trainer.py` | `RecGRPOTrainer(GRPOTrainer)`：route-aware 动态 G/温度、`</think>` token-id 停止（per-sample）、population std、smoke 统计、`build_route_dataset` |
| `scripts/run_grpo_trl_smoke.py` | 4GPU TRL smoke 入口（beam32 从 completion_ids 重解码） |
| `scripts/launch_grpo_trl_smoke.sh` | torchrun 4 卡启动（NCCL lo，expandable_segments） |
| `scripts/trl_import_fix.py` | 修正 TRL 0.24 可选依赖探测（tuple-truthy），零安装 |
| `scripts/run_nothink_m_ablation.py` | M 消融（结论 M_NO=8 混合 96.9%） |
| `scripts/run_beam_batch_bench.py` | Beam32 批量基准（结论 cbs=1 唯一可行，~9.7s/cot） |
| `scripts/test_grpo_trl.py` 等 | CPU 结构测试（全部 PASS） |

## 数据（rec_mp_grpo_v2，见 `results/data_manifest.json`）
- 3,098 条 / 1,549 组（Think 1,549 + NoThink 1,549），四域 video 1100 / ad 854 / prod 764 / living 380。
- v2 重写：按目标域指令（video/prod/ad/living）3 模板变体、组内稳定分配；历史 SID 行与
  /think|/no_think 标记不变。
- 数据本体 27MB 不入库（仓库惯例），开发机路径 `/data/GRPO/data/rec_mp_grpo_v2/train.jsonl`。

## 已验证关键事实（2026-08-17，详见 docs/fix_report_20260817.md）
1. `</think>` 为单 token 151668；token-id StoppingCriteria 正常（单序列 probe 8/8、
   verbatim 复跑 16/16 在 538-836 tokens 处 closure）。
2. **已修复**：TRL 路径 `_ThinkStop` 的标量 `.any()` 会使 batch 16 序列在首个 closure 处
   集体停止（smoke 的 think 补全 mean 142 tokens 即此 bug 产物）→ 改为 per-sample
   BoolTensor（`input_ids[:,-1] == tid`），各序列独立停止（per-sample 实测 pos 632-1745）。
3. **已修复**：beam invalid 15.9% 根因是 TRL `batch_decode(skip_special_tokens=True)`
   剥离 `</think>` 与 SID added tokens → beam32_fn 改为从 completion_ids 重解码
   （skip_special_tokens=False），验证 invalid 0/512 = 0.0%。
4. NoThink 天然短（mean 95 字符 ≈ 22 tokens），无需停止。
5. 采样下 Think closure 率约 69%（11/16）：约 30% 补全无 `</think>`，会跑满
   max_completion_length=2048（计分策略待决策）。

## 结果摘要（results/）
- 4GPU TRL smoke（12 步，`REC-MP-GRPO-V1-TRL-SMOKE12-rank*.json`）：route 序列
  T,T,N,N,T,T 四卡同步；LoRA delta 4.6e-05、base delta 0；峰值 37.1GB；
  policy epoch2 ratio 0.9979-1.0039、clip 0-0.039、KL 有限；advantage std≈1.0（总体）。
- M 消融（`nothink_m_ablation_v1.json`）：M=8 混合 96.9%/zero-std 3.1%（推荐），
  M=4 78.1%/21.9%，M=16 100%/0%；成本 M=8 ≈ 2.2s/组。
- Beam 基准（`beam_batch_bench.json`）：cbs=1 唯一可行（257.9s/16 cots）；
  修复后实测 9.7s/cot（155.3s/16 cots）。
- closure 探测（`probe_think_closure.json`）：8/8 在 538-773 tokens 处 closure。

## 1 epoch 时间估算（修复后，4 GPU）
- T 生成 ≈ 174.3s/rollout（16 序列/rank，含 ~30% 无 closure 跑满 2048）
- N 生成 ≈ 45s/rollout；Beam32 ≈ 155s/rollout（16 cots × 9.7s，4 rank 并行）
- 合计 ≈ 97×(174.3+155) + 97×45 ≈ 36,300s ≈ **10.1h**（+ policy 更新 ≈ 10.5-11h）
- 最大单头为 Beam32（≈ 4.2h）。

## Correctness Round（2026-08-17，详见 docs/correctness_report_20260817.md）
- route-specific reward：reward_func 按 dataset `route` 列路由（Think 只算 think_reward，
  NoThink 只算 nothink_reward，另一路返回 None）；NoThink 严格不触发 Beam32（call count 0）。
- RouteAwareRepeatSampler：真正的 dynamic G —— Think 4 unique x 4 repeats、
  NoThink 2 unique x 8 repeats（global batch 恒 16）；rollout 内硬断言
  rewards.view(-1,G) 每组 recommendation_group_id 全同。
- loss multiplier：Think 1.0 / NoThink 0.5（单组总权重 4 == 4；epoch 6196 == 6196）。
- think_credits 全局 prefix 去重：Exact 已覆盖的 AB/A 前缀不再重复领取。
- shared renderer（grpo_model.render_prompt）：SFT/TRL/Beam32 prompt token parity ALL EQUAL。
- 测试：4 套件全 PASS + 单卡 GPU rollout-only 验证通过（无训练）。

## Final Correctness Round（2026-08-17，详见 docs/final_correctness_20260817.md）
- max_prompt_length：全量 3098 条统计 p50=1142 / p95=2003 / max=3036，全部 <= SFT cutoff
  8192 -> 显式 max_prompt_length=8192（原 TRL 默认 512 会截断 100% 样本）；10240 < 131072。
- NoThink reward 改从 raw completion_ids decode(skip_special_tokens=False) 解析 SID。
- route 元数据改 batch-aligned route_id tensor（think=0/no_think=1），shuffle/split/buffer 安全。
- temperature/top_p 三处同步（args/self/generation_config）+ runtime assert。
- 4 进程真实分布式 audit：Think 4u x4 / NoThink 2u x8，view(-1,G) group_id 全同，
  num_iterations=2 复用正确；TRL prompt ids 与 SFT parity 4/4 True。
- 全 epoch 静态审计：1548/1548 组、387/774 rollouts、2322 steps、effective weight 6192==6192。
- smoke runner 已更新新 route 序列（未启动训练）。

## 待决策
1. NoThink 组结构冲突：TRL 固定 16 样本 batch + 动态 G（T4/N8）下，rollout 1:1 /
   正确 8 样本组归一化 / 组等权三者只能满足其二。
2. 无 closure 补全（~30%）计分策略：现状（全文当答案→beam32 解析）/ 低分无效 /
   提高 max_completion_length。
3. 是否重跑 4GPU smoke 验证修复后的正式路径。

## 运行方式（4GPU smoke）
```bash
cd /data/GRPO/scripts
bash launch_grpo_trl_smoke.sh   # torchrun --nproc_per_node=4 --master_port=29519
```
