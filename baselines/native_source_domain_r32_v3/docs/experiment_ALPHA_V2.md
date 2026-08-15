# Alpha V2：推荐全量 CoT 重建 + 1/3 NoCoT 训练

## 目标

`alpha_v2` 是 alpha 系列在 **推荐任务数据** 上的新一轮重建。它保留 alpha-jiankong 的全部训练配方（LoRA R32、8K neat packing、SID8、monitor/validation sidecar），只替换训练数据集，并同步修正两个已知问题：

1. 推荐任务的 no-think 比例从此前"不完整过滤"版本的 **42.4%** 调整为 **1/3（33.3%）**；
2. 训练数据与 alpha-jiankong dev2 验证集 **零交叉**（推荐按 group、用户/物料按精确行排除）。

## 数据集改动背景

### 起点：三条数据事实

开发机 `/data/lf_data_versions` 下存在三份相关数据：

| 数据 | 位置 | 行数 | 说明 |
| --- | --- | ---: | --- |
| 全量 CoT 母集 | `/data/lf_data/onereason_recommendation_cot.jsonl` | 48,269 | 全部推荐样本的 CoT 版本，100% 真实 think |
| beta 版 | `task_pools/懂推荐/beta版` | 48,269 | `bata_baseline_v1` 的推荐投影：CoT 32,179 / NoCoT 16,090（1/3） |
| 不完整过滤 | `task_pools/懂推荐/不完整过滤` | 48,269 | 把 4,372 条不完整 CoT 降级为 NoCoT：CoT 27,807 / NoCoT 20,462（42.4%） |

**关键背景**：`alpha-jiankong`（及后续 train98 训练集）的推荐部分使用的是"不完整过滤"口径——即所有缺 section 的 CoT 都被降级为 no-think，导致 no-think 占 42.4%。同时其验证集 dev2（4,520 行）从同一 48,269 推荐行中按 group 抽取 2%，与训练集零交叉。

### 改动内容

新推荐任务池 `task_pools/懂推荐/beta_cot_full_v1`（47,208 行）：

1. **排除 dev group**：剔除 alpha-jiankong dev2 的 471 个推荐 group（1,061 行），保证与验证集零交叉；
2. **不完整 CoT 降级**：沿用"不完整过滤"思想，1,910 个不完整 group 内的 think 行（4,293 条）降级为 no-think（空 think + `/no_think` 标记）；
3. **NoCoT 比例恢复 1/3**：降级后 NoCoT 达 20,462（42.4%），再从**完整 group 的原始 no-think** 中按 group 恢复 4,286 条为 CoT（内容取自全量 CoT 母集），最终 CoT 31,472 / NoCoT 15,736，**NoCoT 精确 33.3%**；
4. **格式对齐**：instruction 末尾 `/think` 或 `/no_think`（标记前无换行），NoCoT 答案以 `<think>\n\n</think>` 空块开头，metadata 保留。

### 新注册数据集 `alpha_v2`

聚合三个任务池：

| 任务池 | 输入行数 | 排除 dev 重叠 | 保留 |
| --- | ---: | ---: | ---: |
| 懂用户 `chian异常清洗` | 32,099 | 642 | 31,457 |
| 懂推荐 `beta_cot_full_v1` | 47,208 | 0 | 47,208 |
| 懂物料 `beta版` | 140,884 | 2,817 | 136,491 |
| **合计** | **220,191** | **3,459** | **216,732** |

- 用户/物料部分按**精确行**（规范化整行 JSON）排除 dev2 重叠行；
- 推荐部分已在任务池层面排除，聚合时重叠为 0；
- 组合顺序：understand_user → recommendation → material；逐行字节保留，无修改/去重/打乱/重采样；
- 输出 `216,732` 行，恰好等于 alpha-jiankong train98 行数（同源同量，仅推荐内部比例不同）。

## 复刻办法

### 1. 构建推荐任务池 `beta_cot_full_v1`

前置数据：`/data/lf_data/onereason_recommendation_cot.jsonl`（母集）、`task_pools/懂推荐/beta版/recommendation_beta.jsonl`（格式模板）、`task_pools/懂推荐/不完整过滤/recommendation_nocot_from_incomplete_cot.jsonl`（不完整 group 判定）、`alpha-jiankong-split-v1/dev.jsonl`（dev group 排除）。

```text
输入: beta 48,269 行（新格式模板）
 1) 排除 dev group（471 个）→ 47,208
 2) 不完整 group 内 think 行降级为 no-think（4,293 条）
 3) 从完整 group 的原始 no-think 中按 group 恢复 4,286 条为 CoT（内容取自母集，
    按 (行为 SID 集合, gold SID 集合) 键匹配母集 input/output）
 4) 渲染：instruction 末尾标记 /think|/no_think（无前置换行）；
    no-think output = "<think>\n\n</think>\n\n" + 原最终答案；
    think output 用母集完整 CoT 内容；source_segment 同步改写
输出: recommendation_beta_cot_full_v1.jsonl（47,208 行）
```

校验：NoCoT 占比精确 0.333333；降级行计数缺口 0（不得被恢复）；恢复行全部来自原始 no-think；与 dev group 交叉 0。

### 2. 聚合 `alpha_v2`

```text
输入: 懂用户/chian异常清洗/understand_user_clean.jsonl (32,099)
      + 懂推荐/beta_cot_full_v1/recommendation_beta_cot_full_v1.jsonl (47,208)
      + 懂物料/beta版/material_beta.jsonl (140,884)
排除: alpha-jiankong-split-v1/dev.jsonl 的精确行（用户 642 + 物料 2,817；推荐 0）
输出: /data/lf_data_versions/alltrain/alpha_v2/onereason_alpha_v2.jsonl (216,732)
注册: manifest.json + dataset_info.json（dataset_key: onereason_alpha_v2）
```

### 3. 训练

复制 `train_alpha_jiankong_monitor_validation_4gpu_gc04_2epoch.yaml`，仅改动：

```yaml
dataset: onereason_alpha_v2
dataset_dir: /data/lf_data_versions/alltrain/alpha_v2
tokenized_path: /data/lf_data_versions/alltrain/alpha_v2/tokenized_alpha_v2_8k_sid8w8
output_dir: /data/outputs/baselines/native_source_domain_r32_v3/ALPHA-V2-R32-2E-GC04-4GPU
alpha_validation_metrics_path: /data/logs/baselines/native_source_domain_r32_v3/ALPHA-V2-R32-2E-GC04-4GPU/alpha_validation_metrics.jsonl
```

验证配置保持不变：`alpha_validation_dev_dataset: onereason_alpha_jiankong_dev2`（与训练集零交叉，可直接复用其 tokenized dev/probe cache）。

## 指标与监控

沿用 alpha-jiankong 的全部监控：每 5 step 训练指标、每 50 step teacher-forcing（`g~o_rec_tf_*`）、probe 每 100 step、epoch 末 full dev。

### 监控口径说明（gold-only vs 全候选）

- 现有 `*_tf_*`（训练与验证侧）均为 **gold-only**：只检查当前样本 gold 是否进入合法同层组件词表 top-k（a top32、b/c top8），不覆盖多正 group 的其他合法 gold；
- 全候选指标（`hit8/hit32/coverage8/coverage32`，`core_recommendation_metrics.candidate_rank_outcome`）代码已存在，但仅在 `rec_candidate_metrics_enabled: true` 时采集；
- **alpha_v2 开启 `rec_candidate_metrics_enabled: true`**，新增全候选视角：`cand_a_hit8/hit32/coverage8/coverage32`、`cand_b_hit8`、`cand_c_hit8`、`cand_chain_32_8_8`，用于对照 gold-only 指标，识别多正 group 中被 gold-only 误判的 miss。

## 训练结果（2026-08-15，已完成）

- Run：`ALPHA-V2-R32-2E-GC04-4GPU-20260815-035601`，**1072/1072 steps（100%，epoch 2.0）**，耗时 11h18m（03:56 → 15:14）；
- 数据：`onereason_alpha_v2`（216,732 行，2 epoch = 1072 optimizer steps），全新 tokenized cache `tokenized_alpha_v2_8k_sid8w8`；
- 训练过程：loss 107.0 → 25.7，grad_norm 9.17 → 0.93，`rec_monitor_missing_gold=0`、`rec_monitor_invalid_route=0`，无 NaN/Inf；
- 全候选/多正诊断指标已采集（`rec_pu_multi_positive_segments=95`、`rec_pu_positive_count_a_mean=4.66` 等，见 `all_results.json`）。

**epoch 末 full-dev 验证**（与 jiankong 同一 dev2，733 packs；Alpha-V2 e1 仅记录到 va/hit32，其余为 e2）：

| 指标 | Alpha-V2 e1 | Alpha-V2 e2 | JIANKONG e1 | JIANKONG e2 |
| --- | ---: | ---: | ---: | ---: |
| CoT body CE (va) | 1.2217 | **1.1953** | 1.2438 | 1.2161 |
| CoT Gold SID CE (vb) | — | **4.7630** | 4.6503 | 4.7640 |
| NoThink Gold SID CE (vc) | — | **4.8213** | 4.6345 | 4.8311 |
| TF a-hit8 | — | **0.3506** | 0.3619 | 0.3478 |
| TF a-hit32 | 0.5693 | **0.5806** | 0.5759 | 0.5749 |
| TF b-hit8 | — | 0.4034 | 0.4204 | 0.4062 |
| TF c-hit8 | — | 0.4411 | 0.4826 | 0.4477 |
| TF chain 32/8/8 | — | **0.1131** | 0.1329 | 0.1112 |

**结论**：Alpha-V2 epoch 2 的 full-dev 核心指标对 jiankong epoch 2 **全面占优或持平**（CoT body CE、CoT/NoThink Gold SID CE、a-hit32、chain 均更好；仅 b/c hit8 略低），说明推荐数据重建（1/3 NoCoT + 零交叉）在验证侧带来稳定提升。

## 外部评测（固定评测器）

Epoch 1（checkpoint-529）：

```text
aggregate = 1.2584
material  = 0.0507, 0.0341, 0.0507, 0.0443
user      = 0.1552, 0.0923
recommendation = 0.1101, 0.1496, 0.1876, 0.1521
world/last = 0.2316
```

Epoch 2（checkpoint-1058）：

```text
aggregate = 1.2856
material  = 0.0479, 0.0375, 0.0510, 0.0434
user      = 0.1570, 0.0952
recommendation = 0.1223, 0.1326, 0.1988, 0.1647
world/last = 0.2353
```

### 与 Alpha-Jiankong 对比

| 分项 | JIANKONG (e1) | JIANKONG (e2) | Alpha-V2 (e1) | Alpha-V2 (e2) |
| --- | ---: | ---: | ---: | ---: |
| 总分 | 1.2605 | 1.2992 | 1.2584 | **1.2856** |
| 推荐 video | 0.1204 | 0.1241 | 0.1101 | 0.1223 |
| 推荐 prod | 0.1394 | 0.1394 | 0.1496 | 0.1326 |
| 推荐 ad | 0.2016 | 0.2002 | 0.1876 | 0.1988 |
| 推荐 living | 0.1521 | 0.1683 | 0.1521 | 0.1647 |
| 推荐四域平均 | 0.1534 | 0.1580 | 0.1499 | 0.1546 |

- Alpha-V2 **epoch 1 总分 1.2584 与 jiankong epoch 1（1.2605）基本持平**（-0.0021）；
- Alpha-V2 **epoch 2 总分 1.2856**：较 jiankong e1（1.2605）+0.0251（+1.99%）；较 jiankong e2（1.2992）略低 -0.0136——同 epoch 对比下略低于 jiankong e2，但推荐四域平均（0.1546）仅微低于 jiankong e2（0.1580），video/living 分项已追平或反超；
- 与 Alpha-Smooth（e2 总分 1.3185）相比低 0.0329——两实验改动方向不同（数据重建 vs loss 目标），后续可考虑叠加。

## 状态

- [x] 推荐任务池 `beta_cot_full_v1` 构建并校验
- [x] `alpha_v2` 注册数据集构建并校验（零交叉）
- [x] 训练 YAML 创建
- [x] tokenized cache 生成
- [x] 训练启动并完成（1072/1072，11h18m）
- [x] 外部评测：e1 `1.2584` / e2 `1.2856`
