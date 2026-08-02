# 实验二：GradNorm-lite + Ortho-LoRA 梯度解耦

## 目标

在实验一基础上，进一步处理多任务共享 LoRA 参数中的梯度方向冲突。

核心假设：

- GradNorm 解决梯度大小不公平；
- Ortho-LoRA 解决梯度方向冲突。

二者分别处理不同问题。

## 总体流程

```
任务 loss
    ↓
GradNorm 动态权重
    ↓
任务梯度归一化
    ↓
梯度冲突检测
    ↓
Ortho 投影
    ↓
optimizer.step()
```

## 启动策略

### 0-200 macro-step

与实验一一致：

- 仅统计梯度信息；
- 不修改梯度。

### 200-600 macro-step

启用 GradNorm-lite：

- 调整任务权重；
- 不执行梯度投影。

### 600 step 后

启用 Ortho-LoRA：

- 检测任务梯度 cosine；
- 对持续负相关方向进行投影。

## Ortho 范围

为了控制计算成本：

只处理：

- 最后四层 Transformer；
- LoRA-B 参数；
- q_proj
- v_proj
- o_proj
- down_proj

其他参数保持普通梯度更新。

## 冲突判断

使用梯度余弦相似度：

```
cos(g_i, g_j)
```

当：

```
cos < -0.05
```

并经过 EMA 平滑确认后，认为存在持续冲突。

## 投影策略

对于冲突任务梯度：

- 删除互相抵消的方向分量；
- 保留任务自身有效方向。

三个任务两两检测：

- material-user
- material-recommendation
- user-recommendation

## 推荐配置

```yaml
ortho_lora_enabled: true
ortho_start_step: 600
ortho_interval: 2
ortho_conflict_cosine_threshold: -0.05
ortho_cosine_ema_beta: 0.9
ortho_final_norm_ratio_min: 0.7
ortho_final_norm_ratio_max: 1.3
```

## DDP 实现要求

- 每个 rank 独立收集任务梯度；
- 使用 all_reduce 同步完整任务梯度向量；
- 在全局梯度上计算 cosine 和投影；
- 保证所有 rank 使用一致结果。

## 预期收益

如果有效：

1. 减少任务之间互相干扰；
2. 提升弱势任务表现；
3. 保持强势任务能力；
4. 总体评测优于单独 GradNorm。

## 计算开销预估

采用：

- Hook 捕获梯度；
- 最后四层 LoRA-B；
- 两个 macro-step 执行一次 Ortho；

预计额外训练成本控制在约 3%-10%。
