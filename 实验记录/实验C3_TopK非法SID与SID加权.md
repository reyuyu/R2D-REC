# 实验 C3：Top-K 非法 SID 与 SID 加权

## 状态

rank16 双卡正式训练中。C3 在 C2 的动态完整 SID 合法集与全词表 Top-K 非法竞争惩罚基础上，叠加实验 D 的归一化 `sid_token_weight=8` 基础 SFT CE。

## 受控差异

- 保留 C2 的 `user_action_topk_illegal_*` 辅助项与四项 Action 诊断；
- 开启全局监督 SID token 权重 8、普通文本权重 1 的归一化 CE；
- 保持 lr `2e-4`、LoRA dropout `0.05`、0.75 gradient checkpointing、balanced_40、GradNorm、模型和 batch 语义不变。

C3 的目的不是增加 forward/backward，而是检验基础 SID CE 加权是否能让 C2 的非法竞争惩罚得到更强的 gold SID 锚定。

## 配置

`configs/onereason/onereason_lora_2gpu_balanced40_r16_gradnorm_action_aux_c3_topk_illegal_sid_weight8.yaml`

## Checkpoint 评测结果

以下按物料四域、用户两项、推荐四域、world 的固定顺序记录。

| Checkpoint | 总分 | 物料 4 域 | 用户 2 项 | 推荐 4 域 | World |
| ---: | ---: | --- | --- | --- | ---: |
| 1000 | 1.2036 | 0.0424, 0.0374, 0.0378, 0.0431 | 0.1375, 0.0926 | 0.1083, 0.1496, 0.1806, 0.1476 | 0.2268 |
| 5200 | 1.1941 | 0.0474, 0.0380, 0.0420, 0.0420 | 0.1534, 0.0964 | 0.0756, 0.1394, 0.1848, 0.1449 | 0.2301 |

目前已记录 1000 和 5200 checkpoint 结果；5200 总分为 1.1941，低于 1000 checkpoint 的 1.2036，后续比较应结合各域分数与训练目标分别判断。
