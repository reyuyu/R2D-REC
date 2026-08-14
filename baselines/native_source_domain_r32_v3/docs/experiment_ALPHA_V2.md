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

## 状态

- [x] 推荐任务池 `beta_cot_full_v1` 构建并校验
- [x] `alpha_v2` 注册数据集构建并校验（零交叉）
- [x] 训练 YAML 创建
- [ ] tokenized cache 生成
- [ ] 训练启动与结果记录
