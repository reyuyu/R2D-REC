# BETA-SETloss 实验（Set-PU）

## 定位

- 对外实验名：`BETA-SETloss`
- 母版：`BETA-MATERIAL-ALIGNED-SID8-R32-2E-GC04-4GPU`
- 数据版本：`BETA_material_aligned_v1`
- 数据集：`onereason_beta_material_aligned`
- 训练：4 GPU、LoRA r32/alpha64/dropout 0.05、8K neat packing、GA16、global batch 64、GC0.4、学习率 `2e-4`、cosine、warmup ratio `0.03`、2 epoch。

本实验只替换推荐最终 SID 的 `a/b/c` component loss。物料、用户任务、推荐中的自然语言、`<think>` 内 SID 均保持 native SID8 加权 CE。

## Set-PU 标量目标

对最终 SID component 的 logits，按当前 prefix 划分：

- `P`：已观测正 SID token 集；
- `U`：同层级、未观测的 SID token；
- `O`：错层 SID 及普通词表 token。

固定 `alpha=0.05`：

```text
L = logsumexp(z_P, z_U + log(0.05), z_O) - logsumexp(z_P)
```

这是 forward/backward 一致的 PyTorch scalar objective。它替换 final `a/b/c` 的 one-hot CE，不与原 CE 叠加；SID 权重 8、segment domain weight 和 native SID8 的有效监督 token 分母保持不变。

## 已验证项

- `alpha=1` 时，singleton 与 one-hot CE 严格等价；multi-positive 与标准 positive-set NLL 等价；
- P/U/O 互斥且覆盖完整词表，a/b/c prefix filtering 正确；
- final SID 仅 replacement 一次，SID8 仅乘一次，分母不变；
- 真实 BETA packed batch 上，与同一 autograd reference 比较的 LoRA gradient cosine 和 norm ratio 均为 `1.0000`。

## Checkpoint 评测

评测顺序为：总分、物料四域（video/prod/ad/living）、用户两项（action/chain）、推荐四域（video/prod/ad/living）、世界。

| checkpoint | 总分 | 物料 video | 物料 prod | 物料 ad | 物料 living | 用户 action | 用户 chain | 推荐 video | 推荐 prod | 推荐 ad | 推荐 living | 世界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 1 | 1.2471 | 0.0521 | 0.0371 | 0.0500 | 0.0422 | 0.1484 | 0.0944 | 0.1120 | 0.1292 | 0.2058 | 0.1458 | 0.2301 |
| epoch 2 | 1.2321 | 0.0542 | 0.0375 | 0.0511 | 0.0421 | 0.1530 | 0.0957 | 0.0952 | 0.1088 | 0.2072 | 0.1557 | 0.2316 |
| epoch 2 - epoch 1 | -0.0150 | +0.0021 | +0.0004 | +0.0011 | -0.0001 | +0.0046 | +0.0013 | -0.0168 | -0.0204 | +0.0014 | +0.0099 | +0.0015 |

## 当前结论

- epoch 2 总分较 epoch 1 下降 `0.0150`（约 `-1.20%`），因此当前最佳 checkpoint 是 epoch 1。
- 物料四域合计由 `0.1814` 升至 `0.1849`，用户两项由 `0.2428` 升至 `0.2487`；它们不是总分下降的直接来源。
- 推荐四域合计由 `0.5928` 降至 `0.5669`，其中 video `-0.0168`、prod `-0.0204` 是主要下降项；ad 与 living 小幅改善，说明不是四个推荐域同步退化。
- 此结果表明第二个 epoch 的推荐泛化出现方向不一致的漂移，不能仅凭训练 loss 认定 Set-PU 或物料三路 loss 存在实现错误。需要在相同固定验证样本上分开观察推荐 CoT/No-think、各目标域和 final-SID candidate 指标，才能区分过拟合、任务干扰与数据覆盖差异。

总分是评测器的组合指标，不等于表中各项简单相加；比较时应以同一 checkpoint、同一评测版本下的总分和任务分项为准。
