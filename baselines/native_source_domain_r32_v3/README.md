# Native Source-Domain R32 V3

这是 OneReason 当前 Native SFT baseline 的可复现代码快照。它与仓库内早期 macro-step GradNorm 实验隔离：训练使用原生 8K neat packing 和 source-aware SID8 loss，而不是旧的多任务 gradient controller。

## BETA-SETloss 与 BETA-fenpei

`BETA-SETloss` 在 `BETA_material_aligned_v1` 上将 recommendation 最终 SID `a/b/c` 的 one-hot CE 替换为 Set-PU 标量目标；原生分母、SID/domain weight 和其他任务 CE 路径不变。

当前正式 run 为 `BETA-fenpei`：在 BETA-SETloss 上启用 PackRatio `material/recommendation/user_action/user_chain = 20/45/20/15`，并保留低频 teacher-forcing 候选指标（每 50 optimizer step 一次）。详见 [BETA-fenpei 实验记录](docs/experiment_BETA-fenpei.md)。

目录说明：

- `config/`：正式训练 YAML，`train_rec_pu_beta_material_aligned_r32_b005_2epoch_packratio_20452015_candidate_metrics.yaml` 为 BETA-fenpei。
- `rec_pu/`：P/U/O mask、prefix positive 定位和 Set-PU / SID8 integration。
- `scripts/`：训练入口、正式启动与物料三路预检。
- `tests/`：Set-PU 数学、metadata、replacement 与梯度等价测试。
- `docs/`：实验记录与训练约束。

数据、tokenized cache、checkpoint、日志均不随代码提交。正式启动前应在服务器上设置 BETA manifest 与 `GLOBAL_ITEM_WEIGHT=8`，并运行物料预检。
