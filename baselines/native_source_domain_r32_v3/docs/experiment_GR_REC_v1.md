# 实验 GR_REC_v1：Recommendation Multi-Positive GRPO

状态：**正式训练中**。本文记录截至 2026-08-18 Step 1680 的运行快照；Probe 趋势和结论均为中期结果，不代表最终评测。

## 实验标识

- 实验名：`GR_REC_v1`
- 正式 Run：`REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818`
- 启动时间：`2026-08-18 02:49:22`（Asia/Shanghai）
- 代码版本：`a044c9d386173976c5112b2b7a4576e7515a14ae`
- 基座：`/data/models/onereason-8b-pretrain-competition`
- 初始 Adapter：`BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333`
- 数据：`rec_mp_grpo_v2`，原始 1,549 个 recommendation group
- 输出：`/data/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/`
- 监控：`/data/GRPO/runs/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/`

## 目标与假说

本实验在 BATA SFT Adapter 上进行 recommendation-only GRPO，保留 Think 的物料检索与 Beam32 优势，同时用 group-relative outcome reward 强化用户兴趣推断和最终 SID 命中。Think 与 NoThink 使用各自的生产采样形状，但通过 route loss multiplier 保持单 group 总权重一致。

实验不改动 reward 定义、数据顺序、动态 G、Beam32 语义、采样温度、GRPO/PPO 数学或 old policy log-prob。正式训练前的性能改动均要求 token/reward/loss/gradient parity。

## 冻结训练合同

| 项目 | 配置 |
| --- | --- |
| GPU | 4 x NVIDIA A800-SXM4-80GB |
| LoRA | r32 / alpha64 / dropout 0.05，base 冻结 |
| Optimizer steps | 2,316 |
| Learning rate | `1e-6`，constant |
| Seed | `20260816` |
| Think | G=4，temperature=0.9，top_p=0.95，route weight=1.0 |
| NoThink | G=8，temperature=1.0，top_p=1.0，route weight=0.5 |
| GRPO | beta=0，epsilon=0.2，loss_type=grpo，group scaling |
| Iterations | 每个 rollout 复用 2 个 optimizer steps |
| Length | max prompt 8192，max completion 2048 |
| Think stopping | `</think>` 单 token、per-sample StoppingCriteria、生成后防御性截断 |
| Beam32 | context batch=1，num_beams=32，returns=32，max_new_tokens=128 |
| Checkpoint | 每 500 step；最多保留 4 个 |

Sampler 审计结果：四个 Probe group 排除后剩 1,545 个训练候选 group；为满足完整 batch，固定丢弃 `c399e01d...`，实际 Think/NoThink 均覆盖 1,544 个 group。预计 Think rollout 386 次、NoThink rollout 772 次，路线循环为 `T,T,N,N,N,N`，共 2,316 optimizer steps。

## Reward 合同

- NoThink：SID 六档互斥 reward `-1 / -0.25 / 0 / 0.5 / 2 / 8`。
- Think：每条 CoT 进入 Beam32；Exact=8、AB=2、A=0.5，按命中等级与几何衰减聚合。
- Think 只计算 Think reward；NoThink 只计算 NoThink reward。
- reward、advantage、old_per_token_logps 和 GRPO loss 数学保持冻结。

## 四域固定 Probe

固定 Probe 使用 seed `20260818`，在 Step 0、每 200 step 和最终 step 执行。四个 group 永久从训练集排除，评估前后恢复 Python、CPU 和 CUDA RNG。Think 保持 4 groups x G=4；NoThink 保持两批 2 groups x G=8。

| 顺序 | 域 | Group ID |
| ---: | --- | --- |
| 1 | video | `fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e` |
| 2 | living | `6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f` |
| 3 | prod | `281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700` |
| 4 | ad | `2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8` |

早期错误 Run `REC-MP-GRPO-FULL-E1-PROBE4-20260818` 的 Probe 为 `prod/living/living/ad`，缺少 video，已在 Step 36 停止并保留审计记录；不得与本实验曲线混合。

## 中期运行快照

截至 2026-08-18 本次记录时：

- 训练进程正常，最新 Step `1680 / 2316`。
- 已保存 `checkpoint-500`、`checkpoint-1000`、`checkpoint-1500`。
- 已产生 Step 0 至 Step 1600 的 9 轮 Probe，共 36 行。
- 未观察到训练进程退出、OOM、NaN 或 Inf。

四域平均 Probe：

| Step | Think reward | NoThink reward | Think closure | Beam invalid |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3.7305 | 0.0000 | 100% | 0 |
| 200 | 2.5742 | 0.0234 | 100% | 0 |
| 400 | 4.0879 | 0.2266 | 100% | 25 |
| 600 | 3.7617 | 0.2031 | 100% | 1 |
| 800 | 3.6719 | 0.4219 | 100% | 29 |
| 1000 | 4.2012 | 0.1094 | 100% | 0 |
| 1200 | 4.1992 | 0.0625 | 100% | 0 |
| 1400 | 3.7754 | 0.2188 | 100% | 0 |
| 1600 | 2.9492 | 0.4688 | 100% | 0 |

阶段性观察：NoThink 相对 Step 0 整体抬升，Step 1600 为当前最高均值；Think 波动较大，Step 1000/1200 高于基线，但 Step 1600 回落，尚不能判定稳定改善。closure 始终为 100%。Step 400 的 living 和 ad、Step 800 的 ad 出现 Beam invalid，之后 Step 1000-1600 恢复为 0；该异常必须在最终报告中结合原始 Beam SID 复核。

## Monitoring Phase 1

正式运行启用 `GRPO_MONITOR=1`、`DETAILED=0`、`GENERATION_PROFILE=0`、`BEAM_RANK_BALANCE=1`、`TRACE_EVERY=20`。前端每 3 秒拉取 metrics、rollouts、rank、trace、Probe 和 checkpoint API；监控只读取已有训练结果，不进入 loss 或 optimizer。

监控入口：

```text
http://127.0.0.1:8877/?run=REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818
```

下载 Adapter 时必须保持标准 PEFT 文件名并放在同一目录：

```text
adapter_config.json
adapter_model.safetensors
```

当前 checkpoint 权重已核验为真正 LoRA：配置 `peft_type=LORA`、r=32；权重 504 个 tensor 均为 LoRA A/B，非 LoRA tensor 为 0。带 Run 前缀的浏览器下载文件名会使部分平台误判为 full，上传前必须恢复上述标准名称。

## 验证记录

- Monitor、fixed Probe、formal runner、NoThink trace、prompt cache 与 correctness v2 CPU 套件全部通过。
- 固定 Probe 域覆盖测试强制顺序为 `video/living/prod/ad`；错误数量或重复域会 fail-fast。
- 4GPU 真实 smoke 验证 reward/loss/ratio/clip/KL 有限、LoRA 更新且 base 保持冻结。
- 相关提交：`cc3da7d`（固定 Probe）、`a044c9d`（四域分层 Probe）。

## 最终验收

训练完成后必须补充：最终 Step、四个 checkpoint 与最终 Adapter 状态、四域 Step 0/最终 Probe 对照、外部 11 项评测、Think/NoThink reward 与有效信号密度、KL/clip/ratio、Beam invalid 复核，以及 LoRA/base 参数变化。只有固定 Probe 与外部评测共同支持提升，才能将 `GR_REC_v1` 判定为正向实验。
