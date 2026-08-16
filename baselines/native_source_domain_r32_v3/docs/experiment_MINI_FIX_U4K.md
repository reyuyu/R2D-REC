# 实验 Mini-Fix-U4K：Mini-Fix 推荐契约 + 用户扩到 4,000（物料完整保留）

状态：**已配置，待训练**（2026-08-17；GPU 卡 0 僵尸显存未释放，等待启动条件）。

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

## 预期

- 用户分：U3K 6.1% → 0.2141；U4K 8.1% → 线性外推 ~0.22（相比 Mini-Fix 0.2199 小幅提升）
- 推荐分：数据未变，预期保持 ~0.66（受 pack 组合变化影响可能小幅波动）
- 物料：完整保留 → 物料分应稳定（约定不可比）
- 世界：预期稳定

## 相关文件

- 数据：`/data/lf_data_versions/alltrain/mini_fix_u4k/`（jsonl + manifest + dataset_info）
- 配置/启动/构建：`config/train_mini_fix_u4k_4gpu_gc04_2epoch.yaml`、`scripts/launch_mini_fix_u4k_4gpu_gc04_2epoch.sh`、`scripts/build_mini_fix_u4k.py`
