# BETA-fenpei：Set-PU + PackRatio 正式两轮训练

## 目标

`BETA-fenpei` 是 Native Source-Domain R32 V3 基线上的正式消融。它保持 BETA material-aligned 的三路物料 loss 合同不变，只在推荐任务最终 SID 的 `a/b/c` 位置使用 Set-PU 标量 replacement，并用 PackRatio 将顶层任务 pack 目标比例固定为 `20/45/20/15`。

## 固定训练配方

- 数据：`BETA_material_aligned_v1`，注册名 `onereason_beta_material_aligned`。
- 模型训练：LoRA r32、alpha 64、dropout 0.05、学习率 `2e-4`、cosine、warmup ratio `0.03`。
- 4 GPU、per-device batch 1、GA16、全局 batch 64、8K neat packing、FA2、Liger、bf16/pure bf16、GC0.4。
- 训练：2 epoch、每 epoch 526 optimizer steps、总计 1052 steps。
- 输出：`/data/outputs/BETA-fenpei`；保存策略为每个 epoch 一次，即预计 checkpoint 位于约 step 526 与 1052。

## Set-PU

对 recommendation 最终 SID 的每个 component，按当前 prefix 构造：

- `P`：观察到的正 SID token 集；
- `U`：同层级但未观察到的 SID token；
- `O`：其他词表 token，包括错层 SID 和普通 token。

使用真实标量目标：

```text
L = logsumexp(z_P, z_U + log(0.05), z_O) - logsumexp(z_P)
```

它是 replacement 而不是 auxiliary：最终 `a/b/c` 不再叠加原 one-hot CE；SID 权重 8、domain 权重和 native SID8 分母仍沿用基线。推荐文本、`<think>` 内 SID、物料与两类用户任务继续走原生 CE/SID8 loss 路由。

此前数学回归与真实 batch 参数梯度检查确认 scalar objective 的 forward/backward 一致，LoRA gradient cosine 与同一 autograd reference 为 1.0000。

## PackRatio

顶层任务目标占比为：

| task | target |
| --- | ---: |
| material | 20% |
| recommendation | 45% |
| user_action | 20% |
| user_chain | 15% |

运行中记录 `e_share_*`，用于观察实际 pack 消费比例。每 epoch 的 33,616 个 neat-packed sequence 对应 526 个全局 optimizer steps；尾部仅有 DDP/GA 对齐所需的极少量填充。

## 指标

每 5 step 记录 total loss、四个 `task_loss_*`、grad norm、LR、PackRatio 实际 share、Set-PU loss/positive mass/gold probability/positive entropy 与 recommendation segment 统计。

每 50 step 额外计算 teacher-forcing 候选指标，使用最终 SID 的已选 logits，不额外 forward：

- `f_rec_tf_a_hit8`、`f_rec_tf_a_hit32`；
- `g_rec_tf_a_cov8`、`g_rec_tf_a_cov32`；
- `h_rec_tf_b_hit8`、`h_rec_tf_c_hit8`；
- `i_rec_tf_chain_32_8_8`。

候选指标的 4 GPU smoke 已验证：无 OOM/NaN/DDP 错误，稳定吞吐相对核心指标版本额外开销约 0.57%。

## 启动

```bash
cd /data/baselines/native_source_domain_r32_v3
bash scripts/launch_beta_fenpei.sh
```

启动脚本显式锁定 BETA manifest、`GLOBAL_ITEM_WEIGHT=8`、GC0.4，并关闭重型 REC-PU 诊断。数据、token cache、日志、checkpoint 和模型权重不提交到 Git。
