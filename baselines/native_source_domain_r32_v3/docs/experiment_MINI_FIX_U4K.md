# 实验 Mini-Fix-U4K：Mini-Fix 推荐契约 + 用户扩到 4,000（物料完整保留）

状态：**已完成**。2 epoch 训练于 2026-08-17 05:47 启动（RUN_ID `MINI-FIX-U4K-R32-2E-GC04-4GPU-20260817-054726`），07:48 完成（176 步，51,490 行）；epoch 2 外部评测：**总分 1.3002，推荐分项 0.6497**。

母版为 **Mini-Fix**（推荐契约：NoCoT 纯度 100% + cot 短思考教学，推荐分 0.6644 历史最高）。**数据差异**：understand_user 扩到 4,000 行（Mini-V2 3,000 + chian 异常清洗池补充 1,000），**物料完整保留**（36,298，不删 reverse）。训练配方与 Mini-Fix 一致（LoRA r32/a64、2 epoch、LR 2e-4 cosine、8K neat packing、GC0.4），validation 复用 alpha_mini_v1_validation_filtered_v1。

## 动机

用户分横向对比显示：全量（用户 14.8% 占比）0.25 vs mini（4-8%）0.21-0.23，线性外推"用户行数占比 → 用户分"。U3K（用户 3,000 + 删 reverse 1,000）因预算守恒删了物料、且只 +50% 行，收益低于噪声。U4K 两点修正：① 用户提到 4,000（8% 占比）；② **物料不删**（预算放开，总行数 51,490）——验证"绝对行数 + 不删物料"下用户分是否沿线性上升，同时排除 U3K 删物料的干扰。

## 数据构建（mini_fix_u4k，51,490 行）

| 部分 | 行数 | 说明 |
| --- | ---: | --- |
| 懂用户 | 4,000 | Mini-V2 3,000（action 1,800 / cot 300 / nocot 900）+ chian 池补充 1,000（action 600 / cot 100 / nocot 300，确定性、与 Mini-V2 零重叠） |
| 懂推荐 | 11,192 | Mini-Fix 原样（契约 sha `80d721c4...`） |
| 懂物料 | **36,298** | **完整保留**（material 10,000 + canonical 11,298 + reverse 15,000，SID 覆盖 11,298） |
| **合计** | **51,490** | sha `058e7ca4...` |

用户构成：action 2,400 / chain_cot 400 / chain_nocot 1,200（60/10/30）。

## 配置与启动

- 配置：`config/train_mini_fix_u4k_4gpu_gc04_2epoch.yaml`（2 epoch、monitor/validation 与 Mini-Fix 一致）
- 启动：`scripts/launch_mini_fix_u4k_4gpu_gc04_2epoch.sh`
- 构建：`scripts/build_mini_fix_u4k.py`（校验通过：行数守恒、推荐契约、用户零重复、reverse SID 覆盖）

## 外部评测（epoch 2，固定评测器）

```text
aggregate = 1.3002
material  = 0.0625, 0.0358, 0.0522, 0.0426
user      = 0.1463, 0.0780
recommendation = 0.1279, 0.1564, 0.2016, 0.1638
world/last = 0.2331
```

### 用户扩展系列对比（用户分 vs 用户行数）

| 实验 | 用户行数 | 用户占比 | 用户分 | 推荐分 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mini_fix | 2,000 | 4.0% | 0.2199 | **0.6644** | **1.3093** |
| mini_fix_u3k | 3,000 | 6.1% | 0.2141 | 0.6556 | 1.2846 |
| **mini_fix_u4k** | **4,000** | 8.1% | **0.2243** | 0.6497 | 1.3002 |

**结论**：
- **用户分随行数提升的趋势确认**：2K→0.2199、3K→0.2141（U3K 受删物料/噪声影响略降）、4K→**0.2243**（+0.0044 vs mini_fix）——用户扩到 4,000 首次带来用户分净提升；
- 但**推荐分随用户扩展持续下降**（0.6644 → 0.6556 → 0.6497，-0.0147 vs mini_fix）——用户行 input 超长（~6,450 token）挤占 pack 空间，稀释推荐学习；
- 净效果：总分 1.3002 **仍低于 mini_fix（1.3093，-0.0091）**——用户 +0.0044 被推荐 -0.0147 抵消。**"扩用户提总分"在 mini 尺度不成立**（推荐是更高分数杠杆），除非用户大幅扩到 14%（~7,000 行）或裁剪用户 input 释放 pack。

## 相关文件

- 数据：`/data/lf_data_versions/alltrain/mini_fix_u4k/`（jsonl + manifest + dataset_info）
- 配置/启动/构建：`config/train_mini_fix_u4k_4gpu_gc04_2epoch.yaml`、`scripts/launch_mini_fix_u4k_4gpu_gc04_2epoch.sh`、`scripts/build_mini_fix_u4k.py`
