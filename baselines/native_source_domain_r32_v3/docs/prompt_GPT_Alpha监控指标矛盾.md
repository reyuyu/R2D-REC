# 请分析 Alpha 监控优化实验中的“官方评测提升、固定验证下降”矛盾

请结合你可见的 GitHub 仓库 `reyuyu/onereason-multitask-sft`，对下面这个实验做一次严格、证据驱动的诊断。不要只给“可能过拟合”之类的泛泛解释；请检查仓库中的训练配置、数据切分、验证指标实现、DDP 汇总、teacher-forcing 代理指标和告警条件，并引用具体文件、函数或代码行支持判断。

## 1. 实验身份

- 实验：Alpha 监控优化 / SID8 修正版。
- Run：`ALPHA-JIANKONG-SID8FIX-R32-2E-GC04-4GPU-20260813-045032`。
- 训练集：`onereason_alpha_jiankong_train98`。
- 固定验证集：`onereason_alpha_jiankong_dev2`。
- 每 100 optimization step 跑固定 probe；每个 epoch 末跑 full dev。
- 每 epoch 529 step，总计 1,058 step。
- 训练：4 GPU，LoRA r32/alpha64，8K neat packing，2 epoch，普通 SID8 CE；Alpha 指标是 monitor-only，不应改变梯度。
- checkpoint：Epoch 1=`checkpoint-529`，Epoch 2=`checkpoint-1058`。

请重点检查仓库中与以下功能对应的代码；如果路径已有调整，请搜索同名类和函数：

- `baselines/native_source_domain_r32_v3/scripts/train_native_source_domain_r32_v3.py`
- `rec_pu/alpha_validation_monitor.py`
- `rec_pu/alpha_recommendation_monitor.py`
- Alpha Jiankong 的训练 YAML、dev/probe 构造脚本、split manifest 和 leakage audit
- `AlphaValidationRunner.record_probe_gap`

## 2. 外部评测结果

指标名称未提供，请保持原始 4/2/4/1 顺序，不要擅自命名。

Epoch 1：

```text
0.0465, 0.0379, 0.0441, 0.0426
0.1502, 0.0915
0.1204, 0.1394, 0.2016, 0.1521
0.2342
```

Epoch 2：

```text
0.0490, 0.0369, 0.0516, 0.0420
0.1556, 0.0955
0.1241, 0.1394, 0.2002, 0.1683
0.2364
```

现象：

- 11 项中 7 项上升、3 项下降、1 项持平。
- 四个组的非加权平均全部上升。
- 11 项同口径求和从 `1.2605` 升至 `1.2990`，增加 `0.0385`，相对提高 `3.05%`。这只是摘要，不是另行定义的官方 aggregate。
- 回落项分别为第 2 项 `-0.0010`、第 4 项 `-0.0006`、第 9 项 `-0.0014`；第 10 项提高 `+0.0162`，是最大单项增益。

所以从外部评测看，Epoch 2 整体更好。

## 3. Full-dev 却显示推荐泛化全面下降

下面比较相同 full dev 在 Epoch 1 和 Epoch 2 的结果。CE/NLL 越低越好，TF hit 越高越好。

| 指标 | Epoch 1 | Epoch 2 | 变化 |
|---|---:|---:|---:|
| COT body CE | 1.2438 | 1.2161 | -0.0276，改善 |
| COT gold SID CE | 4.6503 | 4.7640 | +0.1136，变差 |
| NoCoT gold SID CE | 4.6345 | 4.8311 | +0.1966，明显变差 |
| gold path NLL | 13.9313 | 14.3754 | +0.4441，明显变差 |
| 四域重加权 gold SID CE | 4.7124 | 4.8638 | +0.1514，明显变差 |
| TF A-hit@8 | 0.3619 | 0.3478 | -0.0141 |
| TF B-hit@8 | 0.4204 | 0.4062 | -0.0141 |
| TF C-hit@8 | 0.4826 | 0.4477 | -0.0349 |
| TF chain 32/8/8 | 0.1329 | 0.1112 | -0.0217，下降 16.3% |
| COT TF chain | 0.1304 | 0.1127 | -0.0177 |
| NoCoT TF chain | 0.1364 | 0.1091 | -0.0273，下降 20.0% |

四域 full-dev TF chain 同时下降：

| 域 | Epoch 1 | Epoch 2 | 相对变化 |
|---|---:|---:|---:|
| video | 0.08133 | 0.06667 | -18.0% |
| prod | 0.20588 | 0.16667 | -19.0% |
| ad | 0.27200 | 0.23200 | -14.7% |
| living | 0.29762 | 0.26190 | -12.0% |

这不像单域随机波动，而像推荐 SID 解码/完整路径的系统性退化；但 COT body CE 仍改善。

## 4. 固定 probe 显示下降从第二轮开始并持续存在

| step / epoch | gold path NLL | 重加权 gold SID CE | TF chain |
|---|---:|---:|---:|
| 500 / 0.946 | 14.2771 | 4.8259 | 0.1193 |
| 600 / 1.134 | 14.5952 | 4.9271 | 0.1250 |
| 700 / 1.324 | 14.7252 | 4.9502 | 0.1136 |
| 800 / 1.513 | 14.7280 | 4.9515 | 0.0966 |
| 900 / 1.702 | 14.7290 | 4.9525 | 0.0909 |
| 1000 / 1.892 | 14.7488 | 4.9569 | 0.0966 |

NLL 和重加权 SID CE 从进入第二轮后持续恶化；TF chain 有一次短暂上冲，随后明显下降。

## 5. Train–validation 出现反向分叉

- NoCoT gold SID：rolling train CE 从 step 500 的 `4.8468` 降至 step 1000 的 `4.0696`，probe CE 却从 `4.8748` 升至 `5.0991`。
- Gold C：rolling train CE 从 `4.5822` 降至 `4.0266`，probe CE 却从 `4.3977` 升至 `4.7767`。
- Gold B：训练 CE 总体下降，probe CE 从 `4.6252` 升至 `4.7456`。
- COT body 的 train/dev CE 继续改善。

这很像局部过拟合集中在 NoCoT、SID B/C 和完整 32/8/8 路径，而不是所有能力共同退化。

## 6. 现有告警为何没有触发

日志中的 `alpha_rec_overfit_warning` 始终为 0。当前 `record_probe_gap` 规则据观察要求最近三个 probe 同时满足：

1. 五项训练 gold CE 严格单调下降；
2. 五项验证 gold CE 的平均值严格单调上升；
3. TF chain 严格单调下降。

任意轻微抖动都会让告警归零。请核对实现并判断这是否导致漏报。尤其要检查：把 COT/NoCoT/A/B/C 简单平均后，是否会让已严重分叉的 NoCoT 和 C 分量被稳定项抵消。

## 7. 需要你回答的问题

请按证据强弱回答：

1. 这是全局过拟合、推荐任务局部过拟合、验证代理与官方指标错位、验证/官方数据分布差异，还是实现 bug？可以是多因素，但请排序并给出置信度。
2. 为什么外部评测整体提高，而相同 full dev 的推荐 TF chain、四域 TF chain 和 SID CE 几乎全面变差？重点讨论 teacher-forcing 指标与真实生成/F1 的相关性、固定 2% dev 的代表性、任务混合带来的能力权衡。
3. 请审计验证实现：模型 `eval()/train()` 恢复、LoRA checkpoint 加载、cache 与 checkpoint 对齐、DDP all-reduce 分母、重复样本 exact-once、domain reweight、COT/NoCoT 路由和 label/logit position 是否可能制造假下降。
4. `checkpoint-529` 和 `checkpoint-1058` 应如何选择？请分别从“外部综合目标”和“推荐稳健性”给结论，不要只给一个模糊答案。
5. 设计最小成本的判别实验，至少包括：
   - 用同一官方评测器评估约 0.8/1.0/1.2/1.5/2.0 epoch 的 checkpoint；
   - 在固定 dev 上增加真实 generation 指标，而不仅是 teacher forcing；
   - 对 full dev 的域/路由指标做 bootstrap 置信区间；
   - 检查 train/dev/official 的域比例、COT/NoCoT 比例、多正 group-size 和 prompt 长度分布；
   - 验证第二轮降低推荐采样强度、推荐 loss 权重或学习率是否能保留综合增益并避免路径退化。
6. 重写过拟合告警设计：使用最近 4–5 个 probe 的稳健斜率或 EMA，而非严格三点单调；给出推荐阈值、伪代码和误报/漏报权衡。要求至少对 NoCoT、Gold B、Gold C、TF chain 分别报警，再提供综合报警。
7. 给出下一轮训练建议：合理早停区间是否为 `1.0–1.3 epoch`？是否应在 epoch 1 后改变推荐采样/权重，还是只做 checkpoint selection？

## 8. 希望的输出格式

请按以下结构回答：

1. 一句话结论。
2. 现象是否真实：代码审计证据。
3. 最可能原因排序及置信度。
4. 为什么外部评测与 Alpha 验证矛盾。
5. checkpoint-529 vs checkpoint-1058 的选择表。
6. 最小判别实验清单，按信息增益/成本排序。
7. 新告警算法与伪代码。
8. 下一轮训练配置建议。

如果仓库证据不足，请明确指出缺失的文件或日志，不要用猜测补齐。
