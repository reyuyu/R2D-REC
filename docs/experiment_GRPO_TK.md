# GRPO-TK 实验记录

## 1. 实验身份

- **实验代号**：`GRPO-TK`
- **工程名称**：`GR_REC_ThinkSample8_FullSID_v3`
- **正式 Run ID**：`GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828`
- **实验类型**：Think-only、G4 CoT、每条 CoT 独立 Sample8 FullSID 的双目标 GRPO
- **正式启动代码点**：`bcff386ec81104251e93a483785d96ece042f0b2`
- **当前恢复/监控代码点**：`40cb740963f0dfd7f780b461f064b3dd5ee2c517`
- **父模型**：8B Base + `REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500`
- **父模型 Adapter SHA256**：`a5e92db011662799e07b4e1f16a2779afbbab9d66efcb481a5f2e199c75d3436`
- **训练状态**：正式一轮训练进行中；已完成 checkpoint-500 恢复与 checkpoint-550 cadence 验收

本文只记录当前实际运行版本。旧版 fixed-domain Beam8、自然语言 bridge 和未启用的重采样机制，不属于 GRPO-TK 当前训练公式。

## 2. 实验动机

已有推荐 GRPO 能够在最终 SID 上提供直接奖励，但存在两个信用分配问题：

1. 只强化最终答案时，模型可能学会局部 SID 决策，却未必改善生成答案之前的用户兴趣推理。
2. 用固定目标域或固定 Beam 前缀训练时，模型不需要自行完成 domain 决策，训练动作与真实自由生成存在差距。

GRPO-TK 的核心问题是：**能否让一条 CoT 的价值直接由它后续支持的多个完整 SID 采样结果来决定，同时分别优化 CoT 推理动作和完整 SID 动作？**

实验假设如下：

- 好的 CoT 应在相同 prompt 下提高后续 SID 采样的整体质量，而不只偶然命中一次。
- 每条 CoT 后独立采样 8 个完整答案，可以把“这条推理是否形成了更好的预测分布”转化为组内相对信号。
- CoT 和 SID 使用各自的 group advantage，可以避免把 32 条 SID 错误地混成一个 G32，并保持清晰的两级信用分配。
- 不预填 target domain、不插入自然语言 bridge，使 domain、A、B、C 都由模型自己决定，更接近真实推理路径。

## 3. 核心训练拓扑

一个业务 group 的实际计算链为：

```text
1 business group
  -> stochastic G4 CoT
  -> each CoT independently samples 8 answer continuations
  -> scan each continuation for the first complete FullSID
  -> 4 independent SID G8 advantages
  -> sum each G8's raw rewards into 4 CoT rewards
  -> 1 global CoT G4 advantage
  -> L_total = L_cot + L_sid
```

### 3.1 CoT 采样

- 每个业务 group 全局生成 4 条 CoT。
- `temperature=0.9`
- `top_p=0.95`
- CoT action 截止到第一个 `</think>`。
- CoT loss 只更新实际采样的 CoT action tokens。

### 3.2 FullSID Sample8

每条 CoT 后独立采样 8 条 continuation：

- `do_sample=true`
- `temperature=1.0`
- `top_p=1.0`
- `top_k=0`
- `repetition_penalty=1.0`
- `num_return_sequences=8`
- `max_new_tokens=128`

输入上下文只包含：

```text
prompt + sampled CoT through </think>
```

明确不插入：

- fixed target-domain begin
- “用户下一个点击的是……”之类自然语言 bridge

模型可以先输出自然语言，再输出 SID。解析器在整条 continuation 中扫描连续 raw token：

```text
<|domain_begin|> + A + B + C
```

规则为：

- 找到一个完整 SID：正常打分。
- 找到多个完整 SID：只使用第一个打分，同时记录 `multiple_sids` 供监控筛选。
- 未找到完整 SID：记为 invalid，reward 为 `-1`。
- SID loss 只落在第一个完整 SID 的 4 个 action tokens 上；前置自然语言和后续 SID 均不进入 SID loss。

### 3.3 SID reward 与 advantage

继续使用 NoThink-v1 的六档 `q_reward`：

| 预测关系 | Reward |
|---|---:|
| 无法解析完整 SID | -1.00 |
| domain 错误 | -0.25 |
| domain 正确、A 错误 | 0.00 |
| A 命中 | 0.50 |
| AB 命中 | 2.00 |
| Exact 命中 | 8.00 |

每条 CoT 自己的 8 个 SID 构成一个独立 G8：

```python
A_sid = (R_sid - mean(R_sid)) / (population_std(R_sid) + 1e-4)
```

四条 CoT 对应四个独立 G8，严禁 flatten 为 G32。当前正式版本不做 zero-std reroll；如果某个 G8 reward 完全相同，其 SID advantage 为 0。

### 3.4 CoT reward 与 advantage

一条 CoT 的 reward 是其后 8 条 SID raw reward 的算术和：

```python
R_cot_i = sum(R_sid_i)
```

四条 CoT 再构成一个全局 G4：

```python
A_cot = (R_cot - mean(R_cot)) / (population_std(R_cot) + 1e-4)
```

这使 CoT 获得“后续 8 次完整 SID 采样总体表现”的信用，而非旧版 `think_reward()` 或单次答案信用。

### 3.5 PPO/优化器冻结项

- `num_iterations=2`；第二次 policy iteration 完整复用第一次的 CoT、Sample8、reward、advantage 和 detached old logp。
- old logp 来自 unchanged policy 的 no-grad full-forward rescore，不使用 `generate.scores`。
- PPO `epsilon=0.2`。
- `beta=0`。
- `AdamW`，`lr=1e-6`，`weight_decay=0`，constant scheduler。
- `L_total = L_cot + L_sid`，权重为 1:1。
- Base 完全冻结，只训练 LoRA。

## 4. 数据、步数与 Probe

- 数据集：`/data/GRPO/data/rec_mp_grpo_v2/train.jsonl`
- 原始业务 group：1549
- 固定 Probe4：4 个，全部排除训练
- 实际训练 group：1545
- 每个业务 group：G4 CoT × 每条 CoT Sample8 = 32 条 SID continuation
- fresh rollout：1545
- `num_iterations=2`
- 完整一轮 optimizer steps：3090
- Probe：固定 Probe4，沿用 production-shaped Beam32 evaluator，不参与 optimizer
- 当前 Probe cadence：每 50 step
- 当前 checkpoint cadence：每 50 step，另有最终 step 3090
- checkpoint 根目录：`/root/GRPO-checkpoints/GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828`

## 5. 训练前工程证据

四卡 zero-update preflight 已通过：

- 真实一组 G4 CoT reward：`[-1.25, -1.25, -2.0, 0.0]`
- 每条 sampled continuation 均在 8-11 个自然语言前置 token 后出现一个完整 SID
- 四个独立 SID G8 均观察到真实候选间 reward variance
- CoT action gradient 非零
- SID action gradient 非零
- Base 参数保持冻结
- audit 前后 trainable checksum 完全一致
- optimizer step = 0
- scheduler step = 0
- parameter update = false
- Probe4 Beam32 完成，且 RNG/参数未变化

证据文件：

`baselines/native_source_domain_r32_v3/grpo/results/gr_rec_think_sample8_fullsid_v3_gpu_preflight_scan_20260828.json`

## 6. 外部评测成绩

以下 11 个分项严格按现有评测器原始输出顺序记录：前 4 项为懂物料，中间 2 项为懂用户，随后 4 项为懂推荐，最后 1 项为懂世界。

| 模型/检查点 | 总分 | 懂物料（4项） | 懂用户（2项） | 懂推荐（4项） | 懂世界（1项） |
|---|---:|---|---|---|---|
| Parent baseline，第1次 | 1.3436 | 0.0507, 0.0360, 0.0523, 0.0424 | 0.1587, 0.0976 | 0.1363, 0.1598, 0.2142, 0.1629 | 0.2327 |
| Parent baseline，第2次 | 1.3497 | 0.0505, 0.0366, 0.0520, 0.0422 | 0.1587, 0.0976 | 0.1400, 0.1598, 0.2142, 0.1647 | 0.2335 |
| Parent baseline，第3次 | 1.3510 | 0.0510, 0.0366, 0.0515, 0.0425 | 0.1587, 0.0976 | 0.1381, 0.1598, 0.2156, 0.1665 | 0.2331 |
| **Parent baseline 三次均值** | **1.3481** | **0.0507, 0.0364, 0.0519, 0.0424** | **0.1587, 0.0976** | **0.1381, 0.1598, 0.2147, 0.1647** | **0.2331** |
| **GRPO-TK step 250** | **1.3579** | **0.0517, 0.0361, 0.0523, 0.0421** | **0.1593, 0.0982** | **0.1409, 0.1768, 0.2016, 0.1647** | **0.2342** |

### 6.1 Step 250 相对 baseline

- 相对三次 baseline 均值：`+0.0098`，约 `+0.73%`。
- 相对三次 baseline 最好值 1.3510：`+0.0069`。
- baseline 三次范围：`1.3436 - 1.3510`。
- baseline 总分三次样本标准差约：`0.0040`。

各分项相对 baseline 均值的变化：

| 分项 | Delta |
|---|---:|
| 懂物料-1 | +0.0010 |
| 懂物料-2 | -0.0003 |
| 懂物料-3 | +0.0004 |
| 懂物料-4 | -0.0003 |
| 懂用户-1 | +0.0006 |
| 懂用户-2 | +0.0006 |
| 懂推荐-1 | +0.0028 |
| 懂推荐-2 | +0.0170 |
| 懂推荐-3 | -0.0131 |
| 懂推荐-4 | +0.0000 |
| 懂世界-1 | +0.0011 |

四类合计对比：

| 模型/检查点 | 懂物料 | 懂用户 | 懂推荐 | 懂世界 |
|---|---:|---:|---:|---:|
| Parent baseline 三次均值 | 0.1814 | 0.2563 | 0.6773 | 0.2331 |
| GRPO-TK step 250 | 0.1822 | 0.2575 | 0.6840 | 0.2342 |
| Delta | +0.0008 | +0.0012 | +0.0067 | +0.0011 |

## 7. 初步结论

Step 250 的 1.3579 是一个明确的早期正向信号：它同时高于 baseline 三次均值和三次最佳值，说明 GRPO-TK 至少没有在训练早期立即破坏父模型的总体推荐能力，并且存在获得净增益的可能。

但该增益还不能被表述为稳定结论，原因有三点：

1. GRPO-TK step 250 目前只有一次外部评测，而 baseline 有三次，重复次数不对称。
2. 分项改善不均匀。最大增益集中在懂推荐-2（`+0.0170`），同时懂推荐-3 下降 `-0.0131`；总分上涨包含明显的推荐分项迁移，而不是所有指标同步改善。
3. 当前只覆盖完整一轮 3090 steps 的早期约 8.1%，后续可能继续上涨、回落或出现能力迁移。

因此当前最稳妥的判断是：

> **GRPO-TK 在 step 250 显示出优于 parent baseline 的早期总体收益，但证据仍属于单点信号；需要对 step 250 做重复评测，并结合后续每 50 step checkpoint 的同口径结果，确认增益是否稳定以及懂推荐-3 的下降是否持续。**

## 8. 后续评测计划

建议保持训练公式不变，按以下顺序补证据：

1. 对 checkpoint-250 再做至少 2 次同配置评测，使其与 baseline 的 3 次重复对齐。
2. 优先评测 checkpoint-500、550、600，此后根据趋势减少无效评测点。
3. 同时跟踪总分、懂推荐-2 和懂推荐-3，避免只看总分掩盖分项迁移。
4. 将外部分数与训练内 `Exact/AB/A/domain/wrong-domain/invalid`、SID zero-std、CoT reward variance 联合分析。
5. 从 `training_sample_exports/` 筛选正 reward、高 CoT reward、Exact/AB 命中样本，用于定性检查推理是否真实改善。

## 9. 监控与产物位置

- Monitor run：`/data/GRPO/runs/GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828`
- Manifest：`manifest.json`
- 主训练指标：`metrics.jsonl`
- Rollout 摘要：`rollouts.jsonl`
- Sample8/FullSID 明细：`sample8_fullsid.jsonl`
- 固定 Probe：`probes.jsonl`
- 可筛选样本归档：`training_sample_exports/`
- Checkpoint：`/root/GRPO-checkpoints/GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828`
- 前端：`http://127.0.0.1:8878/?kind=recommendation_grpo&run=GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828`

只要上述 run 目录和 checkpoint 根目录未被人工删除，训练证据、已归档样本和恢复训练所需状态会继续保留。

## 10. 代码入口

- `baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/README.md`
- `baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/run_sample8_fullsid_train.py`
- `baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/sample8_fullsid_trainer.py`
- `baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/launch_sample8_fullsid_train.sh`
- `baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/gpu_preflight.py`
- `baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/test_sample8_fullsid_contract.py`

## 11. 独立最优点搜索：OPTSEARCH450

由于原正式 run 的 checkpoint-250 外部分数达到 `1.3579`，而 checkpoint-500 后续分数出现回落，2026-08-29 从同一原始 parent checkpoint-1500、fresh optimizer、相同 seed 和完全相同训练数学重新启动独立搜索 run：

`GR-REC-THINK-SAMPLE8-FULLSID-V3-OPTSEARCH450-20260829`

该 run **不是**从 GRPO-TK checkpoint-1000 恢复；checkpoint-1000 只作为旧正式 run 的完整留档。搜索 run 使用 `max_steps=450`、`save_steps=50`、`probe_every_steps=50`，保留 `50/100/150/200/250/300/350/400/450` 九个检查点。

### 11.1 完成与安全状态

- 完成：`450/450` optimizer steps。
- Fresh rollouts：`225`；业务 group：`225`；CoT：`900`；Sample8 SID：`7200`。
- 训练耗时：`14116.01 s`，约 `3 h 55 min`。
- 最终 train loss：`-0.00727624`。
- `LoRA delta=0.00174821`，`Base delta=0.0`。
- 无 OOM、NCCL error、Traceback、非有限梯度或 checkpoint 保存错误。
- 九个 checkpoint 均包含 adapter、optimizer、scheduler、trainer state 和四卡 RNG 状态。
- 每个 checkpoint 对应的 Probe 均为 `4/4`，并全部出现在 8878 Monitor API。

### 11.2 训练信号分布

7200 条 sampled SID 的层级分布：

| 层级 | 数量 | 占比 |
|---|---:|---:|
| invalid | 21 | 0.29% |
| wrong-domain | 146 | 2.03% |
| domain | 5709 | 79.29% |
| A | 1111 | 15.43% |
| AB | 104 | 1.44% |
| Exact | 109 | 1.51% |

- SID 非零 reward 比例：`20.71%`；正 reward 比例：`18.39%`。
- SID G8 zero-std 比例：`38.78%`。
- CoT reward：mean `1.7533`，median `0.5`，范围 `[-2.5, 64.0]`。
- CoT reward 正值比例：`56.89%`；CoT G4 zero-std 比例：`14.22%`。
- 检测到多 SID 输出 `88` 条；训练按既定 parser 只评分第一个完整 SID，未修改解析或训练数学。

### 11.3 固定 Probe4 趋势

下表的 Think 值是四个固定 Probe group 的 Beam32 reward mean 均值，仅用于同 run 内候选筛选：

| Step | Think Probe mean | NoThink Probe mean | Exact candidate hits | AB candidate hits | A candidate hits |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.3516 | 0.1875 | 4 | 1 | 14 |
| 50 | 1.3633 | 0.1406 | 1 | 4 | 24 |
| 100 | 1.6934 | 0.2500 | 2 | 3 | 23 |
| 150 | **2.0098** | 0.1406 | **3** | 2 | 16 |
| 200 | 1.2891 | 0.1562 | 1 | 3 | 26 |
| 250 | 0.8340 | 0.4844 | 0 | 4 | 22 |
| 300 | 0.9063 | 0.1719 | 0 | 5 | 20 |
| 350 | 0.8984 | 0.5156 | 0 | 4 | 25 |
| 400 | **1.4238** | 0.3906 | 1 | **6** | 19 |
| 450 | 1.0859 | 0.2656 | 0 | **7** | 22 |

训练后 checkpoint 的内部 Think Probe 排序为：

`150 > 100 > 400 > 50 > 200 > 450 > 300 > 350 > 250`

其中 checkpoint-150 是当前固定 Probe4 的首选外测候选，checkpoint-100 与 checkpoint-400 是次选。checkpoint-400/450 的 AB 命中更强，但 Exact 并未持续同步增加，提示后期训练可能更多改善部分层级而非完整 SID 命中。

这个排序不能直接宣布“最佳外部分数点”：Probe4 只有四个固定样本，且 Think CoT 生成具有随机性。正式选点仍应使用同一外部评测协议，优先评测 `150/100/400`，再覆盖用户指定的 `50/200/350/450`；`250/300` 作为复现和补充点保留。本轮没有自动启动 external benchmark。

结构化汇总：

`baselines/native_source_domain_r32_v3/grpo/results/grpo_tk_optsearch450_probe_trend_20260829.json`

### 11.4 新 run 产物位置

- Monitor run：`/data/GRPO/runs/GR-REC-THINK-SAMPLE8-FULLSID-V3-OPTSEARCH450-20260829`
- Checkpoints：`/root/GRPO-checkpoints/GR-REC-THINK-SAMPLE8-FULLSID-V3-OPTSEARCH450-20260829`
- 日志：`/data/GRPO/logs/GR-REC-THINK-SAMPLE8-FULLSID-V3-OPTSEARCH450-20260829.log`
- 前端：`http://127.0.0.1:8878/?kind=recommendation_grpo&run=GR-REC-THINK-SAMPLE8-FULLSID-V3-OPTSEARCH450-20260829`
