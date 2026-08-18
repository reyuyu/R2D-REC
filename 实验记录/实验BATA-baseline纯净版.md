# 实验 BATA-baseline 纯净版

## 目标

以 `onereason_material_domain_2ep_repro_20260811.tar.gz` 为母版，在当前训练框架中复刻其训练数据与训练配方。唯一有意差异是懂用户数据使用 active BETA；物料、SID canonical/reverse 与懂推荐保持复现包投影一致，世界知识数据不参与训练。

## 训练合同

- Run：`BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333`。
- 数据集：`bata_baseline_v1`，共 222,001 条。
- 数据组成：物料 100,000；SID canonical 11,298；SID reverse 29,586；懂用户 BETA 32,848；懂推荐 48,269；不含 world。
- LoRA：r32 / alpha64 / dropout 0.05，target `all`。
- 训练：4 x A800 80 GB，global batch 64（每卡 micro batch 1，累积 16），2 epoch，8K neat packing，FA2，Liger，bf16 / pure_bf16。
- 优化器与日程：AdamW，LR `2e-4`，cosine，warmup 0.03，weight decay 0.01，seed 20260806。
- 显存策略：fractional gradient checkpointing `GC=0.4`；该项用于适配当前框架的显存实现，不改变 forward objective。
- 损失：普通 one-hot SID8 CE。物料按四域样本 multiplier；SID/domain marker 权重 8；canonical 路由权重 4。`REC-PU=false`，`multitask_pack_ratio=false`。
- 推荐多正 metadata 被保留，但只用于 detached 的 monitor-only 统计；不替换 CE、不改变梯度。测试证明开/关该监控的 loss、logits gradient 与 CE contribution bitwise 相等。

正式入口：

```bash
bash /data/baselines/native_source_domain_r32_v3/scripts/launch_bata_baseline_4gpu_gc04_2epoch.sh
```

启动前 preflight 已通过：222,001 条数据的哈希与路由、BETA 替换合同、registry、锁定 YAML，以及实际 loss route 均被检查。验证摘要：`rec_pu=OFF`、`pack_ratio=OFF`、SID8=8、canonical=4、GC fraction=0.4。

## 评测结果

评测指标名称和顺序沿用外部评测器原输出；本记录不对未提供名称的各子指标擅自重命名。

### Epoch 1

```text
aggregate: 1.3090

0.0515, 0.0355, 0.0532, 0.0421
0.1539, 0.0951
0.1213, 0.1700, 0.1890, 0.1602
0.2372
```

说明：本次 `1.3090` 是最新提供的 epoch 1 评测值，覆盖此前实验记录中的 `1.3073`。

### Epoch 2

```text
aggregate: 1.3246

0.0517, 0.0358, 0.0503, 0.0420
0.1573, 0.0972
0.1223, 0.1598, 0.2058, 0.1656
0.2368
```

### GRPO 对照用复测值

同一 BETA-baseline 在多次外部评测中存在约 `±0.01` 的正常波动。GRPO 系列采用其中较高的一次 `1.3313` 作为保守比较基线，避免使用偏低测次夸大后续提升；历史 Epoch 2 首次记录 `1.3246` 继续保留，不被覆盖。

```text
aggregate: 1.3313

0.0519, 0.0363, 0.0503, 0.0422
0.1573, 0.0972
0.1223, 0.1598, 0.2072, 0.1701
0.2368
```

该复测相对 `1.3246` 高 `0.0067`，仍在正常波动范围内，不能解释为模型发生了变化。GRPO checkpoint 的单次 aggregate 与 `1.3313` 相差不超过 `0.01` 时，默认判为与基线同一水平。

### Epoch 1 → Epoch 2

- aggregate：`1.3090 → 1.3246`，增加 `0.0156`。
- 第一组四项：`+0.0002, +0.0003, -0.0029, -0.0001`。
- 第二组两项：`+0.0034, +0.0021`。
- 第三组四项：`+0.0010, -0.0102, +0.0168, +0.0054`。
- 最后一项：`-0.0004`。

整体 aggregate 在第二个 epoch 提升；增益主要来自第二组两项和第三组的第 1/3/4 项。与此同时，第三组第 2 项下降 `0.0102`，第一组第 3 项下降 `0.0029`，因此不是所有子指标均同步提升。

## Epoch 1 训练特征

- 每 epoch 约 553 optimization steps；全程计划 1,106 steps。
- epoch 1 到达耗时约 5:48:09，含 checkpoint 保存；常态约 38 秒/optimization step。
- warmup 结束后的 epoch 1 末段，总 loss 在约 29.26 到 31.32 范围；step 555 loss 30.41，grad norm 0.606，LR `1.0469e-4`，均为 finite。
- epoch 1 末段（step 540-550）的任务 raw loss：物料约 3.92-4.40，推荐约 2.89-3.20，user action 约 1.62-1.77，user chain 约 0.87-0.90。
- 四任务 token exposure 在 epoch 末段保持混合：物料约 7.8%-9.1%，推荐约 26.0%-30.8%，user action 约 24.7%-25.9%，user chain 约 35.8%-41.6%。
- 运行日志未出现 NaN、OOM、Traceback 或 DDP mismatch；四卡在第二轮中稳定占用约 74.7-74.9 GiB / 80 GiB。

## 相关实现与验证

- 数据构造：`baselines/native_source_domain_r32_v3/scripts/build_bata_baseline_dataset.py`
- 数据注册与训练前验证：`register_bata_baseline_dataset.py`、`validate_bata_baseline.py`
- 正式 YAML 与启动器：`config/train_bata_baseline_4gpu_gc04_2epoch.yaml`、`launch_bata_baseline_4gpu_gc04_2epoch.sh`
- baseline CE 与 monitor-only 推荐统计：`scripts/train_native_source_domain_r32_v3.py`、`rec_pu/sid8_rec_pu_integration.py`
- 监控不干预梯度的回归：`tests/test_bata_monitor_only.py`

## 结论

纯净版 BATA-baseline 已完成 2 epoch。最新评测从 epoch 1 的 aggregate `1.3090` 提升到 epoch 2 的 `1.3246`；在该固定训练合同下，第二轮带来净增益，但部分子指标存在回落，后续对照应继续同时报告 aggregate 与分项而非只看总分。
