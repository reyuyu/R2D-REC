# alpha-监控优化

## 目的

以纯净 `bata_baseline` 为训练目标，使用 `alpha-jiankong` 数据只增加训练期的 detached recommendation monitor。该实验不引入验证集、生成评测、额外 forward 或新 loss。

## 锁定配方

- ordinary one-hot SID8 CE；REC-PU / Set-PU / multi-positive / PackRatio 均关闭。
- LoRA r32 / alpha64 / dropout0.05，8K neat packing，FA2、Liger、GA16、GC0.4、4 GPU。
- 数据：`onereason_alpha_jiankong`，路径 `/data/lf_data_versions/alltrain/alpha-jiankong`。

## 数据与定位

推荐 route 来自真实 `source_segment`：`recommendation_cot` 或 `recommendation_nocot`。每个 pack 的 final Gold SID 复用成熟 `locate_packed_rec_pu_targets(..., final_occurrence=True)` 定位；因此 CoT 中的同 SID 引用不会误记为最终答案。每个 component 使用 `logits[label_position - 1]`。

## 新增指标

主指标：`a_rec_cot_body_ce`、`b_rec_cot_gold_sid_ce`、`c_rec_nocot_gold_sid_ce`、`d/e/f_rec_gold_{a,b,c}_ce`；每 50 optimizer step 的 Teacher-Forced current-gold 合法 component Top-K：`g` 至 `o`。此外记录 COT/NoCoT segment 和 Gold position 覆盖计数及异常计数。

CE 指标直接使用 native SID8 已计算的 `base_per_token_ce.detach()`；TF 仅对最终 a/b/c 位置在对应合法 component 词表中取 Top-K。各 rank 只累积 sum/count，日志时以一个紧凑 tensor all-reduce。

## 不变性

`alpha_monitor_enabled: false` 是默认值且不构建 monitor target。开启后所有监控输入均 detached，不增加 model forward、CE、backward、优化器或梯度通信。只有 logging 的一项监控 all-reduce。

## 验证

CPU 回归验证：loss/gradient parity、COT/NoCoT route、final-occurrence、CE 分区、packed mixed target 和 current-gold Top-K。GPU 空闲后才运行 20-30 step 四卡 monitor off/on 性能 smoke；不会与现有正式训练争抢 GPU。

真实四卡 smoke（20 optimizer steps，GC0.4、GA16、8K neat packing）结果：monitor-off rank0 为 40.929s/step，monitor-on 为 40.955s/step，核心监控增加约 0.06%；callback peak 均为 69.288 GiB。无 OOM、NaN、Inf、DDP mismatch 或 packed-boundary error；有效 recommendation route 的 `missing_gold=0`、`invalid_route=0`。

## Phase 1.5 最终验收（TF 实路径）

使用独立的 smoke-only YAML（仅把 `alpha_train_tf_interval` 设为 5，`max_steps=20`，不保存 checkpoint）在真实 `alpha-jiankong` preprocessing、packing、collator 和四卡 Trainer 路径上触发了 4 次 Teacher-Forced Top-K：step 5/10/15/20。`g` 至 `o` 在四个触发点全部写入日志，所有命中率均在 `[0, 1]`，`missing_gold=0`、`invalid_route=0`。

正式 YAML 未被该 smoke 覆盖，仍锁定 `alpha_train_tf_interval: 50`。监控路径只读取 `outputs.logits.detach()` 和既有 `base_per_token_ce.detach()`，在 `no_grad` 下计算；不新增 forward、完整 CE、backward、autograd.grad，也不触碰 loss 分子/分母、优化器梯度或训练配方。

本次 rank0 的 20 个 step：全量 mean/median 为 37.828/37.776 秒；移除前 3 个启动 step 后为 37.656/37.744 秒。TF 触发 step 的 mean/median 为 37.345/37.444 秒，非 TF 稳态 step 为 37.751/37.744 秒。中位数差为 -0.300 秒，属于该短测的自然波动，不能归因于负开销；因此可观测的增量为 0 秒，按正式每 50 step 一次的摊销开销为 0%（测量分辨率内）。

alpha packed cache 有 34,523 个 pack，平均 active/non-padding token 为 7,428.6，p50=8,168，p90/p95/p99=8,191；平均利用率 90.68%，p50=99.71%。每 pack segment 数 mean=6.41、p50=2、p90=9、p95=40、p99=62；`sum(L_i^2)` proxy mean=27.29M、p50=31.85M、p90=44.04M、p95=50.92M、p99=61.84M。实际 smoke 时 GPU 利用率 98--100%，nvidia-smi 显存约 74.6--74.7 GiB，callback CUDA allocated peak 69.288 GiB；功耗约 398--413 W、时钟 1350--1380 MHz，无明显降频或未吃满现象。
