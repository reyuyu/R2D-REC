# 实验 Alpha-mini：组合任务池 R32 两 epoch 训练

## 实验定位

Alpha-mini 是从 `/data/lf_data_versions/task_pools` 下三个未注册任务池组合出的训练数据集，用于缩小任务池规模、提高懂推荐多正样本覆盖并保留懂用户/懂物料的结构化监督。它不是 Alpha 监控验证实验的同一数据集；本记录单独保存 Alpha-mini 的数据构造和官方评测结果。

- 训练数据集：`onereason_alpha_mini_v1`
- 组合数据路径：`/data/lf_data_versions/alltrain/alpha_mini_v1`
- 训练数据总量：`49,490` 行
- 组合顺序：懂用户 → 懂推荐 → 懂物料
- 组合规则：逐行拼接，保留原始字段和顺序；不改写、不去重、不 shuffle、不重采样
- 统一 schema：`system`、`instruction`、`input`、`output`、`history`、`data_source`、`source_segment`、`aux_metadata_json`

## 数据构造

| 任务池 | 来源 | 选取规则 | 行数 |
|---|---|---|---:|
| 懂用户 alpha_mini | `/data/lf_data_versions/task_pools/懂用户/alpha_mini` | seed `20260813`；按最终答案完整 SID 数或 `logic_chain.events` 数分桶，再按原始 prompt+output token 四分位分层抽样 | 2,000 |
| 懂推荐 alpha_mini | `/data/lf_data_versions/task_pools/懂推荐/alpha_mini` | 删除单正 group；video 保留多正 group-size 最大的前 550 个 group；prod/ad/living 保留全部多正 group；每个入选 group 保留全部原始行 | 11,192 |
| 懂物料 alpha_mini | `/data/lf_data_versions/task_pools/懂物料/alpha_mini` | canonical 全量；reverse 做 SID 覆盖优先采样；material sample 按域和 COT/NoThink 平衡采样 | 36,298 |
| 合计 |  |  | **49,490** |

### 懂用户

来源目录为 `/data/lf_data_versions/task_pools/懂用户/chian异常清洗`，排除 `removed_samples.jsonl` 和 `understand_user_clean.jsonl`。构成为：

| 子集 | 来源行数 | 选取行数 | 分桶 |
|---|---:|---:|---|
| `user_action` | 16,576 | 1,200 | 完整 SID 数：1–5/6–10/11–20/21–30/31–40/41+ = 180/240/360/240/120/60 |
| `user_chain_cot` | 7,898 | 200 | `logic_chain.events`：2/3/4/5+ = 30/110/50/10 |
| `user_chain_nocot` | 7,625 | 600 | `logic_chain.events`：2/3/4/5+ = 90/330/150/30 |

懂用户 alpha_mini 原始 token 合计 `9,360,263`，平均 `4,680.1315`，范围 `1,238–8,187`。其中 action `5,348,888`，COT chain `1,085,764`，NoCoT chain `2,925,611`。

### 懂推荐

上游为 `/data/lf_data_versions/task_pools/懂推荐/不完整过滤/recommendation_all_after_incomplete_cot_filter.jsonl`，输入 `48,269` 行。最终保留 `1,549` 个多正 group、`11,020` 个唯一正 SID，group size 范围 `2–20`；其中 video/prod/ad/living 分别为 `550/382/427/190` 个 group，保留行分别为 `8,574/1,059/1,087/472`。

保留行包含 recommendation COT `6,235`、NoThink `4,957`。选中 group 内有 `172` 条重复目标行；这是保留 group 全部原始行造成的合法重复，没有做行级去重。推荐多正 metadata 完整保留：`recommendation_group_id`、`recommendation_group_size`、`recommendation_current_gold_sid`、`recommendation_all_gold_sids`。

推荐原始 token 合计 `19,086,812`，prompt `14,336,610`，output `4,750,202`，平均每行 `1,705.3978`，最大 `3,915`，低于 8K cutoff。

### 懂物料

来源为 `/data/lf_data_versions/task_pools/懂物料/beta版/material_beta.jsonl`，固定随机种子 `20260813`：

| 路由 | 行数 | 说明 |
|---|---:|---|
| `sid_bucket_canonical_no_think` | 11,298 | 11,298 个 SID 全量保留，全部 NoThink |
| `sid_bucket_reverse` | 15,000 | 11,298 个 SID 全覆盖；其中 3,702 个 SID 再补一条不同 caption；COT/NoThink 各 7,500 |
| `material_sample` | 10,000 | SID→文本 5,000、文本→SID 5,000；每个方向 video/prod/ad/living = 1,505/1,459/1,138/898，COT/NoThink 各 2,500 |
| 合计 | **36,298** |  |

懂物料原始 token 合计 `5,574,932`，平均 `153.5879`，最大 `297`。canonical/reverse/material sample 的 token 总量分别为 `1,180,639/2,716,920/1,677,373`。

## 训练合同

- 基础模型：`/data/models/onereason-8b-pretrain-competition`
- LoRA：`r32 / alpha64 / dropout 0.05 / target=all`
- 4×A800 80 GB，per-device batch 1，gradient accumulation 16，全局 batch 64
- `cutoff_len=8192`，packing + neat packing，FA2，Liger，bf16/pure_bf16
- 学习率 `2e-4`，cosine，warmup `0.03`，weight decay `0.01`，seed `20260806`
- 训练 2 epoch；使用 Native SID8 CE；`GLOBAL_ITEM_WEIGHT=8`
- `REC-PU=false`、`multitask_pack_ratio=false`；Alpha monitor/TF/validation 为只读监控，不改变训练梯度
- Alpha-mini 正式配置：`train_alpha_mini_v1_4gpu_gc04_2epoch.yaml`

组合数据 manifest 的核心计数为：懂用户 `2,000`、懂推荐 `11,192`、懂物料 `36,298`、总计 `49,490`。原始组合数据的训练 token 统计由构造脚本使用 `AutoTokenizer(add_special_tokens=false)` 计算，不包含 chat-template 固定开销。

## 评测结果

指标名称未提供，以下严格保留外部评测器的原始顺序。首行 `1.2861` 按当前实验记录作为 aggregate 记录；其余 11 项按原始 4/2/4/1 分组记录。

```text
aggregate: 1.2861

0.0621, 0.0357, 0.0516, 0.0426
0.1395, 0.0708
0.1335, 0.1496, 0.1960, 0.1746
0.2301
```

本次提交只记录结果，不对未提供名称的 11 项指标擅自解释或重命名。训练数据、tokenized cache、checkpoint、日志和模型权重不纳入仓库；可复现所需的来源、规则、计数和哈希保留在服务器 manifest 与本记录中。

## 机器可读来源记录

- 懂用户：`/data/lf_data_versions/task_pools/懂用户/alpha_mini/manifest.json`
- 懂推荐：`/data/lf_data_versions/task_pools/懂推荐/alpha_mini/manifest.json`
- 懂物料：`/data/lf_data_versions/task_pools/懂物料/alpha_mini/manifest.json`
- 组合脚本：`build_alpha_mini_v1.py`
- 懂用户构造脚本：`build_user_alpha_mini.py`
- 懂物料构造脚本：`build_material_alpha_mini.py`
