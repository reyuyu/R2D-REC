# BETA-SETloss 实验（REC-PU Set-PU）

## 定位

- 正式运行名：`REC-PU-BETA-MATERIAL-ALIGNED-R32-B005-2E`
- 对外实验名：`BETA-SETloss`
- 母版：`BETA-MATERIAL-ALIGNED-SID8-R32-2E-GC04-4GPU`
- 数据版本：`BETA_material_aligned_v1`
- 数据集注册名：`onereason_beta_material_aligned`

本实验只替换懂推荐最终 SID 的 `a/b/c` 三个 component loss。物料、懂用户、推荐中的自然语言和 `<think>` 内 SID，继续使用原生 SID8 加权 CE。

## Set-PU 标量目标

对一个推荐最终 SID component 的词表 logits，定义：

- `P`：当前 prefix 下全部已观测 positive SID token；
- `U`：同一层级、未出现在 `P` 中的 SID token；
- `O`：其他层级 SID 与普通词表 token。

固定 `alpha = 0.05`：

```text
D = sum(exp(z_p), p in P)
  + alpha * sum(exp(z_u), u in U)
  + sum(exp(z_o), o in O)

L_set_pu = log(D) - logsumexp(z[P])
```

实现中仅给 `U` 的 denominator logits 加上 `log(alpha)`，随后直接由 PyTorch autograd 求导。它不是辅助 loss，也不是旧版的 stop-gradient/custom-backward surrogate。

当 `alpha=1` 且 `|P|=1` 时，严格等价普通 one-hot CE；当 `alpha=1` 且 `|P|>1` 时，等价标准 positive-set NLL。

## 原生 SID8 接入

每个 segment 保持原有归一化：

```text
N_i = sum(token_weight_t * token_loss_t)
L_i = domain_weight_i * N_i / valid_supervised_token_count_i
L_batch = mean(L_i)
```

推荐 final `a/b/c` 仅做 replacement：其 token loss 从 one-hot CE 改为 Set-PU，SID weight 仍只乘一次 `8`，分母仍是有效监督 token 数。不存在 `CE + Set-PU` 双重计数。

## 锁定的物料三路

- `material_sample`：100,000 条，普通 response token 权重 1，SID/物料域标记权重 8，乘四域 sample weight；
- `sid_bucket_canonical_no_think`：11,298 条，全部有效 response token 权重 4，不乘域权重；
- `sid_bucket_reverse`：29,586 条，普通 token 权重 1、SID/域标记权重 8，不乘域权重。

四域权重由正式 manifest 固定：video `1.1791795483`、prod `0.7738397931`、ad `1.1334527237`、living `0.8980530211`。

## 训练配方

- 4 GPU，global batch 64，GA16；
- LoRA r32 / alpha64 / dropout0.05；
- 8K neat packing、FA2、Liger、bf16/pure_bf16；
- LR `2e-4`、cosine、warmup `0.03`、2 epoch；
- gradient checkpointing fraction `0.4`；
- `GLOBAL_ITEM_WEIGHT=8`，`rec_pu_enabled=true`，`rec_pu_unlabeled_sid_grad_scale=0.05`。

## 已验证项

- P/U/O 掩码互斥且覆盖完整词表；
- alpha=1 singleton 与 one-hot CE 的 loss/gradient 等价；
- alpha=1 multi-positive 与 set-NLL 等价；
- alpha=0.05 与手工 scalar reference 的 loss/gradient 等价；
- final SID replacement 一次、SID8 一次、分母不变；
- 真实 BETA batch 中 Set-PU autograd 与同一 scalar reference 的 LoRA gradient cosine/norm ratio 均为 `1.0000`；
- 40-step 四卡稳定性验证中，推荐 loss 在 step20 到 step40 没有复现旧 surrogate 的持续反弹。

## 边界

本目录不包含原始训练数据、tokenized cache、checkpoint 或训练日志。启动前必须使用 `/data/lf_data_versions/alltrain/BETA_material_aligned_v1/manifest.json` 完成物料三路预检。
