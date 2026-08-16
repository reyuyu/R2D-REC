# 实验 Mini-Fix：推荐短思考教学恢复 + NoCoT 高纯度（配比假说验证）

状态：**已完成** 2 epoch 训练（2026-08-16 09:27 → ~11:15，140 步）；epoch 2 外部评测已记录：**总分 1.3093，推荐分项 0.6644（历史最高）**。

母版为 **alpha_mini_v1（Mini 大基线，49,490 行）**：OneReason-8B、LoRA r32/a64/dropout 0.05、8K neat packing、GA16、GC0.4、LR 2e-4 cosine、warmup 0.03、seed 20260806、GLOBAL_ITEM_WEIGHT=8。训练配方、用户/物料/推荐行数与占比、validation 路线（alpha_mini_v1_validation_filtered_v1 dev/probe）与 alpha_mini **完全一致**。

**唯一数据差异**：推荐行内部 cot/nocot 归属调整（见下）。

## 动机与假说

复盘发现推荐分与两个数据维度强相关：

1. **NoCoT 纯度**：NoCoT 中混入"短 think 行"（母集 think < 800 字符）越多，推荐分越低（BETA 9.5% → 0.6594；jk 21.4% → 0.6320；alpha_v2 27.1% → 0.6184）；
2. **cot 中的短思考教学**：BETA 是唯一把短 think 行保留在 cot（9.3%）的数据（思考监督全覆盖），jk/alpha_v2 把短 think 行降级出 cot → **短序列样本的思考监督丢失**。

**假说**：把 NoCoT 中的短 think 行归还 cot（恢复短思考教学）+ NoCoT 提升至 100% 纯度（占比 44.3% 不变）→ 推荐分应高于 alpha_mini（0.6537）。

## 数据构建（mini_fix，49,490 行）

- 基于 alpha_mini_v1，仅调整推荐行（11,192 行）：
  - **NoCoT 中的 1,181 条短 think 行 → 归还 cot**：恢复母集短 think（`<think>{内容}</think>\n{final}`），instruction 标记 `/no_think` → `/think`；
  - **cot 中的 1,181 条完整 think 行（随机，seed 20260806）→ 降级 NoCoT**：去掉 think（`<think>\n\n</think>\n\n{final}`），标记 `/think` → `/no_think`；
- 结果：cot 6,235 / NoCoT 4,957（**NoCoT 占比 44.3% 保持**）；**NoCoT 纯度 100%（short 0 行）**；**cot 含 1,194 条短思考教学**（原 13 条）；
- 校验：final（gold）100% 不变、非推荐行逐字节不变、行数守恒、标记/segment 一致；
- 文件：`/data/lf_data_versions/alltrain/mini_fix/onereason_mini_fix.jsonl`（sha256 `6dc76604...`）；注册 `onereason_mini_fix`（现场 tokenize，不复用旧 cache）。

## 训练

- Run：`MINI-FIX-R32-2E-GC04-4GPU-20260816-092741`，140/140 步完成；
- 训练健康：`rec_monitor_missing_gold=0`、`rec_monitor_invalid_route=0`，无 NaN/Inf；
- 配置：`config/train_mini_fix_4gpu_gc04_2epoch.yaml`；启动：`scripts/launch_mini_fix_4gpu_gc04_2epoch.sh`。

## 外部评测（epoch 2，固定评测器）

```text
aggregate = 1.3093
material  = 0.0631, 0.0363, 0.0517, 0.0426
user      = 0.1479, 0.0720
recommendation = 0.1297, 0.1598, 0.2030, 0.1719
world/last = 0.2312
```

### 与关键实验对比

| 实验 | 总分 | 推荐分项 | 推荐 4 项 |
| --- | ---: | ---: | --- |
| **Mini-Fix** | **1.3093** | **0.6644** 🏆 | 0.1297, 0.1598, 0.2030, 0.1719 |
| BETA（全量参考线） | 1.3313 | 0.6594 | 0.1223, 0.1598, 0.2072, 0.1701 |
| Alpha-Smooth | 1.3185 | 0.6577 | 0.1363, 0.1496, 0.2044, 0.1674 |
| alpha_mini（母版） | 1.2861 | 0.6537 | 0.1335, 0.1496, 0.1960, 0.1746 |
| mini_smooth | 1.2681 | 0.6353 | 0.1157, 0.1598, 0.1834, 0.1764 |

**Mini-Fix 相对 alpha_mini：总分 +0.0232，推荐 +0.0107（video +0.1297 vs 0.1335 略降、prod +0.0102、ad +0.0070、living 持平）**；推荐分项 **0.6644 为全系列最高**（超过 BETA 0.6594）。

## 结论：假说验证成功

- **NoCoT 高纯度（100%）+ cot 保留短思考教学 → 推荐分历史新高**；
- 反向确认了掉分机制：jk/alpha_v2 把短 think 行降级出 cot、污染 NoCoT，正是推荐分低于 BETA 的原因（gold 本身正确，影响来自**教学形式**：短序列样本的思考监督丢失 + NoCoT 纯度下降）；
- Mini-Fix 总分 1.3093 仍低于 BETA（1.3313），差异全部来自**用户分项**（0.2199 vs 0.2545，-0.0346）——mini 用户占比仅 4%（BETA 14.8%），印证"任务梯度权重占比 → 任务分项"规律；
- 推荐配比最优方向：**NoCoT 高占比（44%）+ 高纯度（100%）+ cot 全思考教学（长+短）**。

## 相关文件

- 数据：`/data/lf_data_versions/alltrain/mini_fix/`（jsonl + manifest + dataset_info）
- 配置：`config/train_mini_fix_4gpu_gc04_2epoch.yaml`
- 启动：`scripts/launch_mini_fix_4gpu_gc04_2epoch.sh`
- 输出：`/data/outputs/baselines/native_source_domain_r32_v3/MINI-FIX-R32-2E-GC04-4GPU-20260816-092741/`
