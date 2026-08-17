# 实验 Mini-Fix-Whole-Smooth：Mini-Fix 数据 + 全程标签平滑（排除性验证）

状态：**已完成**。2 epoch 训练于 2026-08-17 09:41 启动（RUN_ID `MINI-FIX-WHOLE-SMOOTH-R32-2E-GC04-4GPU-20260817-094117`），~11:15 完成（138 步）；`as_0_active` 全程为 '1'（平滑 step 1 起生效）；epoch 2 外部评测：**总分 1.2884，推荐分项 0.6484**。

母版为 **Mini-Fix**（49,490 行：NoCoT 纯度 100% + cot 短思考教学，总分 1.3093、推荐分项 0.6644 历史最高）。**唯一差异**：开启 AlphaSmooth（`epsilon: 0.05`、`alpha_smooth_start_epoch: 0.0`，两个 epoch 全程平滑）。训练配方、数据、validation 与 Mini-Fix 完全一致。

## 动机（排除性验证）

smooth 的证据链表明其收益来自"软化劣质 gold"：
- jk（nocot 78.6% 纯度、含 21.4% 劣质降级行）+ smooth → **+0.0193**（唯一正收益）
- BETA（纯度 100%，干净）+ smooth → **-0.0135**（负）
- mini_whole_smooth（alpha_mini 纯度 76%）→ ≈持平（-0.0002）
- mini_smooth（第二 epoch 才开）→ **-0.0180**（负）

Mini-Fix 的 nocot 纯度 **100%**（最干净）——按机制预测 smooth 无劣质 gold 可软化、只剩副作用，应为负收益。本实验补齐"纯度 100% + mini 尺度 + smooth"的最后证据点：若负 → smooth 在干净数据上彻底封死（全量 BETA-FIX 也不加 smooth）；若意外持平/正 → smooth×高纯度组合有意外价值。

## 配置

- 数据：`onereason_mini_fix`（49,490 行，同 Mini-Fix）
- smooth：`alpha_smooth_enabled: true`、`epsilon: 0.05`、`start_epoch: 0.0`（门控验证：epoch 0.0/0.5/1.0/2.0 全部激活）
- 训练：2 epoch、LoRA r32/a64、LR 2e-4 cosine、8K neat packing、GC0.4（同 Mini-Fix）
- 配置：`config/train_mini_fix_whole_smooth_4gpu_gc04_2epoch.yaml`
- 启动：`scripts/launch_mini_fix_whole_smooth_4gpu_gc04_2epoch.sh`

## 外部评测（epoch 2，固定评测器）

```text
aggregate = 1.2884
material  = 0.0615, 0.0357, 0.0529, 0.0430
user      = 0.1468, 0.0700
recommendation = 0.1213, 0.1564, 0.1988, 0.1719
world/last = 0.2301
```

### 判定（对照预期）

| 场景 | 判定标准 | 实际 | 结果 |
| --- | ---: | ---: | --- |
| 负收益（预期） | < 1.3093 | **1.2884** | ✅ 负收益确认（-0.0209，推荐 -0.0160） |
| 持平 | ≈ 1.3093 | — | — |
| 正收益（意外） | > 1.3093 | — | — |

**结论（排除性验证成功）**：Mini-Fix（nocot 纯度 100% 最干净数据）+ 全程平滑 = **-0.0209**，推荐分项 0.6644 → 0.6484（-0.0160）。至此 smooth 在干净数据上的负收益证据链完整：BETA+SMOOTH（全量，-0.0135）、mini_whole_smooth（纯度 76%，≈持平）、mini_fix_whole_smooth（纯度 100%，**-0.0209 最负**）——**数据越干净，smooth 副作用越大**；smooth 仅在劣质 gold 数据（jk 21.4% 降级行，+0.0193）上有修复价值。**结论：全量 BETA-FIX 不应加 smooth。**

## 相关文件

- 配置/启动：`config/train_mini_fix_whole_smooth_4gpu_gc04_2epoch.yaml`、`scripts/launch_mini_fix_whole_smooth_4gpu_gc04_2epoch.sh`
- 数据：`/data/lf_data_versions/alltrain/mini_fix/`（同 Mini-Fix）
