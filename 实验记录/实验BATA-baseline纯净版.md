# 实验 BATA-baseline 纯净版

## 目标

以 `onereason_material_domain_2ep_repro_20260811.tar.gz` 为母版，在当前训练框架中复刻其训练数据与训练配方。唯一有意差异是懂用户数据使用 active BETA；物料、SID canonical/reverse 与懂推荐保持复现包投影一致，世界知识数据不参与训练。

## 训练合同

- Run：`BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333`
- 数据集：`bata_baseline_v1`，共 222,001 条。
- 数据组成：物料 100,000；SID canonical 11,298；SID reverse 29,586；懂用户 BETA 32,848；懂推荐 48,269；不含 world。
- LoRA：r32 / alpha64 / dropout 0.05，target `all`。
- 训练：4 x A800 80 GB，global batch 64（每卡 micro batch 1，累积 16），2 epoch，8K neat packing，FA2，Liger，bf16 / pure_bf16。
- 优化器与日程：AdamW，LR `2e-4`，cosine，warmup 0.03，weight decay 0.01，seed 20260806。
- 显存策略：fractional gradient checkpointing `GC=0.4`；该项是为了适配当前框架的显存实现，不改变 forward objective。
- 损失：普通 one-hot SID8 CE。物料按四域样本 multiplier；SID/domain marker 权重 8；canonical 路由权重 4。`REC-PU=false`，`multitask_pack_ratio=false`。
- 推荐多正 metadata 被保留，但只用于 detached 的 monitor-only 统计；不替换 CE、不改变梯度。测试证明开/关该监控的 loss、logits gradient 与 CE contribution bitwise 相等。

正式入口：

```bash
bash /data/baselines/native_source_domain_r32_v3/scripts/launch_bata_baseline_4gpu_gc04_2epoch.sh
```

启动前 preflight 已通过：222,001 条数据的哈希与路由、BETA 替换合同、registry、锁定 YAML，以及实际 loss route 均被检查。验证摘要：`rec_pu=OFF`、`pack_ratio=OFF`、SID8=8、canonical=4、GC fraction=0.4。

## Epoch 1

检查点：`checkpoint-553`。由于 dataloader/epoch 的离散边界，训练 state 记录为 global step 553、epoch 1.0；运行日志在 step 555 显示 epoch 1.0036。

### 评测结果

```text
aggregate: 1.3073

0.0508, 0.0359, 0.0526, 0.0415
0.1539, 0.0949
0.1204, 0.1666, 0.1960, 0.1575
0.2372
```

评测指标名称和顺序沿用外部评测器原输出；本记录不对未提供名称的各子指标擅自重命名。

### 训练特征

- 每 epoch 约 553 optimization steps；全程计划 1,106 steps。
- epoch 1 到达耗时约 5:48:09，含 checkpoint 保存；常态约 38 秒/optimization step。
- warmup 结束后的 epoch 1 末段，总 loss 在约 29.26 到 31.32 范围；step 555 loss 30.41，grad norm 0.606，LR `1.0469e-4`，均为 finite。
- epoch 1 末段（step 540-550）的任务 raw loss：物料约 3.92-4.40，推荐约 2.89-3.20，user action 约 1.62-1.77，user chain 约 0.87-0.90。
- 四任务 token exposure 在 epoch 末段保持混合：物料约 7.8%-9.1%，推荐约 26.0%-30.8%，user action 约 24.7%-25.9%，user chain 约 35.8%-41.6%。
- 运行日志未出现 NaN、OOM、Traceback 或 DDP mismatch；四卡在正在进行的第二轮中均稳定占用约 74.7-74.9 GiB / 80 GiB。
- 当前正在继续 epoch 2；本记录中的分数仅对应 epoch 1 checkpoint，不代表 2 epoch 最终结果。

## 相关实现与验证

- 数据构造：`baselines/native_source_domain_r32_v3/scripts/build_bata_baseline_dataset.py`
- 数据注册与训练前验证：`register_bata_baseline_dataset.py`、`validate_bata_baseline.py`
- 正式 YAML 与启动器：`config/train_bata_baseline_4gpu_gc04_2epoch.yaml`、`launch_bata_baseline_4gpu_gc04_2epoch.sh`
- baseline CE 与 monitor-only 推荐统计：`scripts/train_native_source_domain_r32_v3.py`、`rec_pu/sid8_rec_pu_integration.py`
- 监控不干预梯度的回归：`tests/test_bata_monitor_only.py`

## 结论

纯净版 BATA-baseline 已在严格的数据和训练合同下稳定完成第一个 epoch，并取得 aggregate `1.3073`。第 2 epoch 仍在运行，待其 checkpoint 评测后补充最终对照结论。
