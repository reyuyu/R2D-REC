# GRPO 正式训练前工程验收（2026-08-18）

## A. Monitor OFF/ON parity

环境为 4 x A800、8B + BATA adapter、seed 20260816、8-group、max_steps=6，
Beam rank balance 开启，generation profile 关闭。为抵消运行顺序影响，共做两组：

- OFF -> ON：`ACCEPT-20260818-OFF` / `ACCEPT-20260818-ON`
- ON -> OFF：`ACCEPT2-20260818-ON` / `ACCEPT2-20260818-OFF`

两组均从同一原始 adapter 独立加载，route 和 recommendation group IDs 全部一致，
route 为 `T,T,N`。反向配对在第一次参数更新前的 rollout 结果如下：

| 项目 | OFF vs ON |
|---|---:|
| completion token IDs / SHA256 | 16/16 exact |
| closure position | 16/16 exact |
| reward | 16/16 exact |
| advantage | exact |
| Exact / AB / A / invalid | 6 / 0 / 2 / 0，exact |
| step-1 loss | `-7.450580596923828e-09`，exact |
| step-1 grad norm | 0.31149328 / 0.31148356，abs diff `9.72e-6` |

第一次 DDP optimizer update 后，独立运行发生随机轨迹分叉。第二个 Think rollout
token exact 为 0/16，第三个 NoThink 为 1/16；因此后续 reward、advantage、loss、
ratio、clip、KL 和最终 LoRA hash 不再 bitwise exact。该现象不是 Monitor 特有：
OFF1 与 OFF2 自身也在第一次更新后分叉。

最终 LoRA 逐参数比较：

| 对比 | max abs | mean abs | parameter cosine |
|---|---:|---:|---:|
| OFF2 vs ON2 | `1.048e-5` | `1.813e-6` | `0.9999999528` |
| OFF1 vs OFF2（自然重复性基线） | `1.068e-5` | `1.889e-6` | `0.9999999493` |

Monitor 差异不大于 OFF/OFF 自然波动。两边 504/504 LoRA tensor 因轨迹分叉均非
bitwise；两边 base SHA256 各自在训练前后完全一致。没有为追求 exact parity 修改
FlashAttention、DDP、sampling 或训练数学。

## B. Monitor overhead

以 rank0 从 `trainer.train()` 前后计时为总 smoke wall：

| 配对 | OFF | ON | overhead |
|---|---:|---:|---:|
| OFF1 / ON1 | 245.301 s | 247.321 s | +2.019 s / +0.82% |
| OFF2 / ON2 | 245.444 s | 246.082 s | +0.638 s / +0.26% |
| 两组均值 | 245.373 s | 246.701 s | +1.329 s / **+0.54%** |

反向配对阶段计时（轨迹分叉后只能作为 wall-time 样本，不是同-token benchmark）：

- Think generation：138.88 / 138.88 s（0.00%）
- Think rollout：219.96 / 220.67 s（+0.71 s / +0.32%）
- policy forward/backward：4.52 / 4.52 s（0.00%，日志精度 0.01 s）
- NoThink rollout：3.04 / 3.03 s（-0.01 s）
- Beam 每 rank 两轮累计：OFF 77.49-78.26 s；ON 78.01-79.07 s
- Beam 最慢 rank：78.26 / 79.07 s（+0.81 s / +1.04%）
- result-gather 最慢 rank：0.886 / 1.058 s（两轮累计）

两组峰值显存均未因 Monitor 增长：max allocated 均值由 45,636.5 MB 降至
45,258 MB，max reserved 均值由 47,444 MB 降至 46,993 MB，属于运行波动。
TRACE_EVERY=1 时 Monitor 文件分别为 54,717 / 55,494 bytes，约 55 KB/6 steps；
其中 trace 约 45 KB。

Monitor 没有新增 collective、CUDA synchronize 或 GPU forward。NoThink 复用既有
group metadata gather，在 trace step 扩大 completion payload；Think 复用既有 Beam
result gather，开启时附带已经解析的 beam SIDs。两次正反配对没有观察到稳定 straggler。

结论：总 wall overhead 0.54%，低于 1%“很好”门槛。

## C. Full runner

新增 `scripts/run_grpo_trl_train.py`，复用 `RecGRPOTrainer`、route dataset、Beam32、
rank-balanced Beam scheduler、monitor writer 和 `run_grpo_trl_smoke.py` 的统一
`make_grpo_config()`。它不拥有或重写 GRPO 算法。

支持：`--max-steps`、`--lr`、`--seed`、`--n-groups`、`--output-dir`、
`--run-id`、`--resume-from-checkpoint`、`--save-steps`、`--save-total-limit`。
`--n-groups` 接受正整数或 `all`；省略 `--max-steps` 时使用 sampler 推导的完整步数。
120-step pilot 与 full epoch 命令见根目录 README，本次未运行二者。

## D. Full dataset audit

真实 `/data/GRPO/data/rec_mp_grpo_v2/train.jsonl` 审计：

| 项目 | 数量 |
|---|---:|
| raw groups / records | 1549 / 3098 |
| trained groups | 1548 |
| dropped groups | 1 |
| Think / NoThink unique groups | 1548 / 1548 |
| Think / NoThink rollouts | 387 / 774 |
| optimizer steps (`num_iterations=2`) | 2322 |

被 drop 的 group ID 为
`c399e01d9eaa30e416df6c29ee118db4b9a6361bf558433439ccec27500e0d46`。
shuffle 后它位于最后一个单 group segment；该尾段既不足 Think 的 4 unique chunk，
也不足 NoThink 的 2 unique chunk，所以匹配 sampler 的 drop-last 语义被同时丢弃。

## E. Checkpoint / resume

正式配置使用 `save_strategy="steps"`，默认 `save_steps=100`、
`save_total_limit=2`。因为一次 rollout 复用两个 policy iterations，安全 checkpoint
边界为偶数 global step：奇数 `save_steps` 和奇数 `checkpoint-N` resume 均抛出
`ValueError`。未改写 Transformers/TRL checkpoint 机制。

12-step resmoke 实际生成 `checkpoint-12`：adapter 349 MB、optimizer 699 MB、4 个
rank RNG state、scheduler、trainer state、tokenizer 均存在，总大小约 1,015 MB。

## F. Monitor defaults

正式 runner 使用 `setdefault`，因此环境变量仍可覆盖：

- `GRPO_MONITOR=1`
- `GRPO_DETAILED_MONITOR=0`
- `GRPO_GENERATION_PROFILE=0`
- `GRPO_BEAM_RANK_BALANCE=1`
- `GRPO_TRACE_EVERY=20`

`--run-id` 必填；非 resume 时若 output 或 monitor 目录已有内容则拒绝启动。manifest
记录 commit、runner args、raw/sampled group、sampler audit、预期 rollout/steps 和
checkpoint 配置。Dashboard 指向 `/data/GRPO/runs/<RUN_ID>`。

## G. Tests / smoke

最终 CPU/correctness 回归全部 PASS：

- `python test_grpo_reward.py`
- `python test_grpo_trl.py`
- `python test_correctness_v2.py`
- `python test_compute_loss_parity.py`
- `python test_beam_prompt_cache.py`
- `python test_monitor_nothink_group.py`
- `python test_formal_runner.py`
- `python -m monitor.test_monitor`

4GPU formal runner resmoke：`FORMAL-ACCEPT-12-20260818`，12 optimizer steps，route
`T,T,N,N,N,N`，reward `[3.0625, 2.8848, 0.3906, -0.125, -0.2188, 0.2969]`，
所有 loss/ratio/clip/KL finite，Beam invalid=0，无 NaN/Inf/OOM。LoRA norm delta
`6.212e-5`（四 rank 一致），base delta=0；peak allocated 44,411-46,620 MB，
peak reserved 46,702-48,134 MB。

## H. Risk check

- sampling / RNG：未修改；独立 DDP 运行的既有非 bitwise 行为仍存在。
- reward / dynamic G / route weight / Beam / stopping：未修改。
- advantage / old logps / GRPO-PPO loss：未修改；现有 loss parity test exact PASS。
- DDP：未修改 collective 数量或 scheduler 算法；正式 runner 复用现有 trainer。
- checkpoint：仅从 smoke 的 `save_strategy=no` 扩展为正式 steps 保存；偶数边界验证。
- dataset / renderer / 数据顺序：未修改；`shuffle_dataset=False` 仍由共享配置固定。

## I. Final conclusion

- Monitor：**通过工程验收**。总开销 0.54%，初始 rollout token/reward exact，最终
  参数偏差不超过 OFF/OFF 自然非确定性基线；但不宣称跨独立 4GPU 运行 bitwise exact。
- Formal runner：**ready**。全量审计、唯一 RUN_ID、manifest、偶数 checkpoint/resume、
  12-step 实际 checkpoint 均验证通过。
- 下一步：**建议进入 120-step pilot**，保持正式默认 Monitor 配置；本轮未启动 pilot
  或 2322-step full epoch。
