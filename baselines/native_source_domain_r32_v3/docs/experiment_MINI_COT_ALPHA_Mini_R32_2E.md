# mini-cot：Alpha-mini CoT repeat-normalized weighting（已放弃）

> **状态：已放弃该 trick。** 本实验只完成了 CPU manifest/cache 验收，未启动正式训练；后续不再沿此路线推进（不进入正式 Alpha 实验、不参与得分对比）。保留本记录仅用于追溯实现与验收过程。

## 定义

`mini-cot = Alpha-mini + recommendation CoT <think>...</think> span weight = 0.5 / N_cot(group)`。

本实验复用 Alpha-CoT 已验证的通用实现，唯一算法变量是 `recommendation_cot` 的完整 supervised `<think>` 到 `</think>` span。span 内普通文本、domain token 和历史 SID 都统一降为 `0.5/N`；`</think>` 后的答案前缀、最终 domain/a/b/c SID、模板 tail、No-think、material、user 和 canonical 路由完全保持 Alpha-mini baseline。

## 真实父实验

- 数据：`onereason_alpha_mini_v1`，49,490 行；recommendation 11,192 行，其中 CoT 6,235、No-think 4,957。
- baseline cache：`/data/lf_data_versions/alltrain/alpha_mini_v1/tokenized_alpha_mini_v1_train_8k_sid8w8`。
- mini manifest：`/data/lf_data_versions/alltrain/alpha_mini_v1/mini_cot_repeat_count_manifest_v1.json`。
- 新 cache：`/data/lf_data_versions/alltrain/alpha_mini_v1/tokenized_alpha_mini_v1_train_8k_sid8w8_cot05n`。
- 配置：`config/train_mini_cot_v1_4gpu_gc04_2epoch.yaml`。

## 边界与安全

`recommendation_nocot`、final Gold SID=8、post-think answer、material、user、canonical 和 Alpha-mini 的 packing/validation/output 设置保持不变。`rec_pu_enabled=false`、`multitask_pack_ratio_enabled=false`，不继承 Alpha train98 的 manifest，也不 resume checkpoint。

本轮只允许 CPU manifest/cache 审计和静态 preflight；不启动 GPU smoke 或正式训练，不干扰其他四卡任务。

## 验收记录

Manifest 的 `N_cot` 只统计 mini 实际训练 JSONL 中的 `recommendation_cot` 行数，禁止使用 `recommendation_group_size` 或上游 all-gold 数量。需完成 baseline/new cache parity：除 `loss_weights` 外所有 packed columns 一致，变化位置 100% 位于 recommendation CoT `<think>...</think>` span。

### 本次 CPU 验收结果

- Mini source rows：49,490；recommendation：11,192；CoT：6,235；No-think：4,957。
- CoT groups：1,270；`N_cot` mean=4.9094、p50=2、p90=12、p95=14、p99=16、min=1、max=18。
- `0.5/N_cot` mean=0.22823、p50=0.25、p10=0.5、min=0.02778、max=0.5。
- Baseline/new packs：4,448 / 4,448。
- 除 `loss_weights` 外的 `input_ids`、`labels`、sample/task/domain metadata、packing 参数和 `rec_pu_targets_json` 全部一致。
- changed loss-weight positions：4,580,317，100% 位于 6,235 个 recommendation CoT think span；final Gold SID 保持 8，No-think 未变化。
- CoT body weighted mass：5,857,201 -> 462,148.5，保留约 7.89%。
- CoT post-think answer（严格从 `</think>` 后开始）：273,288 -> 273,288；CoT final SID：199,520 -> 199,520；No-think：232,357 -> 232,357。
- recommendation total weighted mass：6,362,846 -> 967,793.5。
- `MINI_COT_PREFLIGHT_PASS` 和 `MINI_COT_REPEAT_CACHE_AUDIT_PASS`。

本轮没有启动 GPU smoke、torchrun 或正式训练；当前四卡实验未被停止或干扰。
