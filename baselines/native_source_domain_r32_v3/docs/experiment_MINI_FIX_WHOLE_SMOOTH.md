# 实验 Mini-Fix-Whole-Smooth：Mini-Fix 数据 + 全程标签平滑（排除性验证）

状态：**已配置，待训练**（2026-08-17；GPU 卡 0 僵尸显存未释放，等待启动条件）。

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

## 预期与判定

| 场景 | 总分 | 判定 |
| --- | ---: | --- |
| 负收益（预期） | < 1.3093（约 1.29-1.30） | smooth 在干净数据 mini 尺度封死 |
| 持平 | ≈ 1.3093 | smooth 中性，无价值 |
| 正收益（意外） | > 1.3093 | 值得继续研究 |

## 相关文件

- 配置/启动：`config/train_mini_fix_whole_smooth_4gpu_gc04_2epoch.yaml`、`scripts/launch_mini_fix_whole_smooth_4gpu_gc04_2epoch.sh`
- 数据：`/data/lf_data_versions/alltrain/mini_fix/`（同 Mini-Fix）
