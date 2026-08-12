# 实验 Alpha：训练监控与泄漏安全验证

## 目的

本实验以 `bata_baseline_v1` 的 Native SFT + SID8 配方为父基线，只增加可关闭、只读的训练监控和 group-safe 验证集机制。它不修改模型前向、训练损失、反向传播、优化器、学习率调度、packing 或采样比例。

正式训练数据为 `onereason_alpha_jiankong_train98`，验证数据为 `onereason_alpha_jiankong_dev2`；两者来自 `alpha-jiankong` 的确定性 98/2 切分。

## 数据切分

- 源数据 / train98 / dev2：221,252 / 216,732 / 4,520 条。
- 切分 seed：`20260812`。
- 推荐按 `recommendation_group_id`、有序 SID 历史和目标域的连通组切分；同组多正例不会跨 train/dev。
- 用户按归一化的 input/history family 切分，CoT 与 NoThink 配对样本保持同侧。
- 物料按完整 prompt family 切分。
- 还会在三任务间执行 canonical-prompt 并集检查，防止相同 prompt 跨任务泄漏。
- 原样复制记录，不改写 `aux_metadata_json`；推荐多正例元数据保持不变。

验证了 recommendation group、SID-history + domain、用户 family、canonical prompt 与完整 JSON 行在 train/dev 间均无交集。源数据中已有的 244 条完全重复行原样保留，但不跨 split。

## 监控与验证指标

训练仍只做一次 forward/backward。训练侧日志在不影响梯度的前提下记录四任务 loss：

- `task_loss_material`
- `task_loss_recommendation`
- `task_loss_user_action`
- `task_loss_user_chain`

推荐训练侧同时可低频记录 teacher-forcing 的 final SID 指标，均为 `detach` 后统计：当前 gold 的 a/b/c CE、gold 概率、正例集合覆盖、候选质量和路由/元数据异常计数。

验证采用 sidecar：在 optimizer step 完成后切换 `model.eval()` 并在 `torch.inference_mode()` 下执行；结束后恢复模型模式、Python/CPU/CUDA RNG。它不调用 backward、optimizer step、scheduler step，不改变 `global_step`。

- 固定开发集 probe：176 条、85 个完整推荐组、42 个 8K pack；每 100 optimizer steps 运行一次。
- 完整 dev：4,520 条、733 个 8K pack；每个 epoch 末运行一次。
- 每张卡采用 exact rank-stride shard，probe 为 11/11/10/10 pack，full dev 为 184/183/183/183 pack，不 padding、不重复统计。
- 主日志采用 `va`–`vo` 与 `ga`–`ge` 的精简标量；逐域、gold 概率、source-domain reweighting 和重复元数据明细写入独立 JSONL。

## 正式配方

`train_alpha_jiankong_monitor_validation_4gpu_gc04_2epoch.yaml` 继承 Native baseline：4 GPU、LoRA r32/alpha64/dropout 0.05、8K neat packing、FA2、Liger、bf16/pure_bf16、GA16、全局 batch 64、LR 2e-4、cosine、warmup 3%、GC 0.4、SID/domain weight 8。

关闭 REC-PU 和 PackRatio；因此目标函数仍为普通 native SID8 CE。数据缓存共 33,810 个训练 pack，对应 529 optimizer steps/epoch、2 epoch 共 1,058 steps、32 warmup steps。

## 验证结果与开销

CPU 回归覆盖：监控开关的 loss/gradient parity、训练/验证主指标 parity、exact-shard、域重加权、RNG/训练态恢复及 split 泄漏检查，均通过。

四卡 12-step smoke 通过：probe 在 step 5/10、随后一次 full dev；无 OOM、NaN/Inf、DDP mismatch、packed-boundary 或 metadata 错误。

- probe 平均 9.82 秒；
- full dev 172.60 秒；
- 训练稳定时约 37.69 秒/optimizer step；
- 单个 epoch 的验证额外开销约 1.1%（仅 probe 约 0.26%）；
- callback allocator peak 69.29 GiB/rank，full-dev 期间最高 driver memory 约 76.5 GiB。

## 边界

这个实验不是与旧 BATA baseline 的严格单变量评分消融：它使用了清洗后的 alpha 数据和 98/2 group-safe split。监控开关已证明不改变 loss 或梯度；但训练曲线的绝对 loss 不能跨不同数据版本直接比较。模型、训练输出、tokenized cache、训练日志和数据文件均不纳入仓库。
