# 实验一：GradNorm-lite 多任务梯度平衡

## 目标

验证多任务训练中，仅通过动态调整任务梯度尺度，是否能够改善任务间的不平衡，并为后续 Ortho-LoRA 提供稳定梯度基础。

## 核心思想

GradNorm 不直接修改模型结构，而是动态调整任务 loss 权重：

```
L = w_material * L_material
  + w_user * L_user
  + w_recommendation * L_recommendation
```

其中权重根据：

- 当前任务梯度范数；
- 当前任务学习速度；
- 历史 EMA 平滑统计；

动态更新。

## 任务范围

第一阶段只处理三个主要任务：

- material
- user
- recommendation

world 保持固定权重：

```
w_world = 1.0
```

原因：world 在 balanced_40 中出现频率较低，不适合参与标准 GradNorm。

## 实验流程

### Warmup 阶段

前 200 个 macro-step：

- 不修改任务权重；
- 记录 loss EMA；
- 记录梯度范数 EMA；
- 记录任务间梯度 cosine。

### GradNorm 阶段

200 step 后开启：

- 每 10 个 macro-step 更新一次任务权重；
- 当前步统计结果用于下一 macro-step；
- 使用 EMA 避免单 batch 噪声导致权重震荡。

## 推荐配置

```yaml
gradnorm_enabled: true
gradnorm_update_interval: 10
gradnorm_alpha: 0.5
gradnorm_weight_update_rate: 0.10
gradnorm_weight_min: 0.5
gradnorm_weight_max: 2.0
gradnorm_loss_ema_beta: 0.9
gradnorm_grad_ema_beta: 0.9
```

## 参考梯度参数

为了控制计算成本：

- 不计算全部模型梯度；
- 选择最后四层 LoRA-B；
- 模块包括 q_proj、v_proj、o_proj、down_proj。

## 观察指标

重点记录：

- task weight
- task grad norm
- task loss EMA
- gradient cosine

## 成功标准

如果实验有效，应观察到：

1. 任务权重稳定变化，而不是剧烈震荡；
2. 不同任务梯度范数差异降低；
3. 总体评测提升，尤其减少任务之间互相牺牲。
