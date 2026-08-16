# 实验 Mini-Fix-U3K：保留推荐优势，强化懂用户

状态：**训练中**。Run：`MINI-FIX-U3K-R32-2E-GC04-4GPU-20260816-174640`（2026-08-16 启动）。本实验针对 Mini 系列中“懂物料分高、懂用户分低”的结构性问题：以 Mini-Fix 的推荐路线和总训练预算为锚点，只扩大懂用户监督。

## 动机与假说

Mini-Fix 的 epoch 2 外部评测为总分 `1.3093`、推荐分项 `0.6644`（历史最高），但懂用户仅 `0.1479, 0.0720`，合计 `0.2199`。Mini-V2 已证明 3,000 条用户池可用，但同时改变了推荐 NoCoT 比例，无法单独归因。

**假说**：保留 Mini-Fix 的推荐样本逐字节不变，并将懂用户从 2,000 条扩至已验证的 Mini-V2 3,000 条；为保持 49,490 条总样本和相同步数，从物料反向样本中只移除 1,000 条同 SID 的补充冗余行。这样可以提高用户梯度占比而不稀释 Mini-Fix 的推荐教学合同。

## 数据构建（mini_fix_u3k，49,490 行）

构建脚本：`scripts/build_mini_fix_u3k.py`。远程产物目录：`/data/lf_data_versions/alltrain/mini_fix_u3k/`。

| 部分 | 行数 | 说明 |
| --- | ---: | --- |
| 懂用户 | 3,000 | Mini-V2 已验证池：Action 1,800；Chain CoT 300；Chain NoCoT 900 |
| 懂推荐 | 11,192 | 从 Mini-Fix 原样保留，推荐 contract SHA `80d721c4fb78876b14ba2fb1ffab00a33fa935a65c052191c719610c7e0e082a` |
| 懂物料 | 35,298 | material_sample 10,000 + canonical 11,298 + reverse 14,000；反向补充行移除 1,000，主 SID 覆盖保持 |
| **合计** | **49,490** | 与 Mini-Fix 总行数、2 epoch 训练预算一致 |

来源摘要：Mini-Fix `6dc7660417f093b923c93d4f19734ed9240d5fba4c9beaaa01b6a164ccd94f7f`；Mini-V2 用户池 `9c692e01cb3ea2b44e29c8d21ee85bcd086a7406e99d3c3febd30c59e72fa063`；U3K 合并文件 `297a943bca4f9605607bacf589a59cce8de404d4d77df206848eeac6a958de3f`。

构建后 token 统计：总计 `37,834,171`（均值 `764.4811`）；懂用户 `14,053,326`（均值 `4,684.442`）；懂推荐 `18,403,039`（均值 `1,644.303`）；懂物料 `5,377,806`（均值 `152.3544`）。

## 训练合同

配置：`config/train_mini_fix_u3k_4gpu_gc04_2epoch.yaml`；启动脚本：`scripts/launch_mini_fix_u3k_4gpu_gc04_2epoch.sh`。

模型与优化器完全沿用 Mini-Fix：OneReason-8B、LoRA r32/a64/dropout 0.05、4 GPU、8K neat packing、per-device batch 1、GA16、LR `2e-4`、cosine、warmup `0.03`、GC `0.4`、`GLOBAL_ITEM_WEIGHT=8`、seed `20260806`、2 epoch。沿用固定 Mini-disjoint dev/probe 验证集，现场 tokenize，不复用旧 Mini cache。

启动前已完成构建脚本与 manifest 检查；当前正式训练已启动。正式入口会把训练配置、数据 manifest、日志和输出写入：

`/data/outputs/baselines/native_source_domain_r32_v3/MINI-FIX-U3K-R32-2E-GC04-4GPU-<timestamp>/`

## 评测与验收门槛

以下为本实验的预注册目标，不是已取得的分数：

| 指标 | Mini-Fix 已知值 | U3K 目标 |
| --- | ---: | ---: |
| 总分 | `1.3093` | `>= 1.3093` |
| 推荐分项 | `0.6644` | `>= 0.6600` |
| 懂用户合计 | `0.2199` | `>= 0.2350` |
| 用户 Action | `0.1479` | `>= 0.1550` |
| 用户 Chain | `0.0720` | `>= 0.0800` |

验收必须同时报告总分、物料 4 项、用户 2 项、推荐 4 项和 world；重点检查推荐分项不得明显跌破 Mini-Fix，用户两项应有可解释提升。训练日志仍需确认 `rec_monitor_missing_gold=0`、`rec_monitor_invalid_route=0` 且无 NaN/Inf。

## 复现命令

```bash
/data/venvs/llamafactory-01398eb-liger081/bin/python \
  /data/baselines/native_source_domain_r32_v3/scripts/build_mini_fix_u3k.py

# 仅在明确批准正式训练后执行
bash /data/baselines/native_source_domain_r32_v3/scripts/launch_mini_fix_u3k_4gpu_gc04_2epoch.sh
```

## 相关文件

- 数据：`/data/lf_data_versions/alltrain/mini_fix_u3k/`（JSONL、manifest、dataset_info、README）
- 配置：`config/train_mini_fix_u3k_4gpu_gc04_2epoch.yaml`
- 启动：`scripts/launch_mini_fix_u3k_4gpu_gc04_2epoch.sh`
- 输出：`/data/outputs/baselines/native_source_domain_r32_v3/MINI-FIX-U3K-R32-2E-GC04-4GPU-<timestamp>/`

## 外部评测结果（epoch 2，2026-08-16 完成）

```text
aggregate = 1.2846
material  = 0.0620, 0.0359, 0.0529, 0.0418
user      = 0.1410, 0.0731
recommendation = 0.1400, 0.1564, 0.1918, 0.1674
world/last = 0.2223
```

### 与预注册门槛对照（全部未达）

| 指标 | Mini-Fix 已知值 | U3K 目标 | U3K 实际 | 判定 |
| --- | ---: | ---: | ---: | ---: |
| 总分 | 1.3093 | >= 1.3093 | **1.2846** | ❌ -0.0247 |
| 推荐分项 | 0.6644 | >= 0.6600 | **0.6556** | ❌ -0.0088 |
| 懂用户合计 | 0.2199 | >= 0.2350 | **0.2141** | ❌ -0.0058 |
| 用户 Action | 0.1479 | >= 0.1550 | **0.1410** | ❌ |
| 用户 Chain | 0.0720 | >= 0.0800 | **0.0731** | ❌ |

### 结论

**用户扩到 3,000（+1,000 行）未带来任何预期提升**：用户分项反而略降（0.2199 → 0.2141），推荐分项也略降（-0.0088，video -0.0069 / prod -0.0034），世界分项 -0.0089。可能原因：新增 1,000 行用户样本（来自 Mini-V2 池）与评测分布的边际增益有限，且预算平衡从物料 reverse 移除的 1,000 行带来扰动。**"加用户行数 → 用户分涨"在 mini 尺度（+50% 增幅）上不成立**；用户占比对用户分项的杠杆作用可能需要更大增幅（如翻倍）或从推荐侧挤预算才能显现。Mini-Fix（推荐契约）仍是 mini 系列最优配方；U3K 作为负结果记录。
