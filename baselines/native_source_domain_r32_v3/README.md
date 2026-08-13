# Native Source-Domain R32 V3

这是 OneReason 当前 Native SFT baseline 的可复现代码快照。它与仓库内早期 macro-step GradNorm 实验隔离：训练使用原生 8K neat packing 和 source-aware SID8 loss，而不是旧的多任务 gradient controller。

## BETA-SETloss 与 BETA-fenpei

`BETA-SETloss` 在 `BETA_material_aligned_v1` 上将 recommendation 最终 SID `a/b/c` 的 one-hot CE 替换为 Set-PU 标量目标；原生分母、SID/domain weight 和其他任务 CE 路径不变。

当前正式 run 为 `BETA-fenpei`：在 BETA-SETloss 上启用 PackRatio `material/recommendation/user_action/user_chain = 20/45/20/15`，并保留低频 teacher-forcing 候选指标（每 50 optimizer step 一次）。详见 [BETA-fenpei 实验记录](docs/experiment_BETA-fenpei.md)。

## Alpha：监控与泄漏安全验证

`ALPHA-JIANKONG-MONITOR` 保持普通 native SID8 CE，不启用 REC-PU 或 PackRatio。它使用 `alpha-jiankong` 的确定性 group-safe train98/dev2 切分，并增加只读的四任务训练 loss、推荐 teacher-forcing 指标、固定开发集 probe 和 epoch-end full-dev sidecar。验证不会创建额外训练 forward/backward，不改变 optimizer、scheduler、RNG 或 global step。详见 [Alpha 实验记录](docs/experiment_ALPHA_监控优化.md)；`SID8FIX` 正式运行的 epoch 1/2 结果、验证分叉证据和告警漏报分析见 [结果分析](docs/experiment_ALPHA_监控优化_结果分析.md)，供外部复核的完整问题清单见 [GPT 诊断提示词](docs/prompt_GPT_Alpha监控指标矛盾.md)。

Alpha 正式训练额外采用 persisted SID8 cache contract：训练前直接扫描实际 Arrow cache，拒绝仍含 fallback weight 2/3 的缓存；canonical supervised response 必须为 4，非 canonical SID/domain token 必须为 8。缓存修复和 Epoch2 CoT/No-think 观测见 [Alpha SID8 cache 修复记录](docs/experiment_ALPHA_SID8_cache_fix.md)。

目录说明：

- `config/`：正式训练 YAML，`train_rec_pu_beta_material_aligned_r32_b005_2epoch_packratio_20452015_candidate_metrics.yaml` 为 BETA-fenpei。
- `rec_pu/`：P/U/O mask、prefix positive 定位和 Set-PU / SID8 integration。
- `scripts/`：训练入口、正式启动与物料三路预检。
- `tests/`：Set-PU 数学、metadata、replacement 与梯度等价测试。
- `docs/`：实验记录与训练约束。

数据、tokenized cache、checkpoint、日志均不随代码提交。正式启动前应在服务器上设置 BETA manifest 与 `GLOBAL_ITEM_WEIGHT=8`，并运行物料预检。
