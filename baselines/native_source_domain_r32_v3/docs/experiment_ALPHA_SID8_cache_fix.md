# Alpha：SID8 静态缓存修复与 Epoch2 观测

## 失效运行封存

首个 `ALPHA-JIANKONG-MONITOR` run 的静态 train cache 在构建时未将
`GLOBAL_ITEM_WEIGHT=8` 传入预处理模块。其实际 item 权重为 material /
recommendation 的 2、user action / chain 的 3，且 weight=8 数量为 0。

该 run 已在 Epoch1 完成后封存为 `INVALID SID2/3 CACHE RUN`：保留
checkpoint、full-dev JSONL、配置、trainer state 与日志，但不得作为正式结果
或续训来源。

## 修复内容

- 新增 `scripts/alpha_sid8_cache_contract.py`：直接扫描持久化 Arrow cache，
  不再仅信任启动时环境变量。
- `preflight_alpha_jiankong_formal.py` 在分配 GPU 前执行 fail-closed cache
  contract；出现 2/3、非 item text 异常、canonical 非 4 或 item 非 8 均拒绝启动。
- Alpha YAML 指向重新从正式 tokenizer / qwen3_nothink / 8K neat-packing
  pipeline 构建的 `*_sid8w8` caches。

## 审计结论

新 train cache：33,810 pack；dev：733 pack；固定 probe：42 pack。训练缓存
权重精确为：

| weight | token 数 |
|---:|---:|
| 0 | 235,784,325 |
| 1 | 37,758,578 |
| 2 | 0 |
| 3 | 0 |
| 4 | 765,393 |
| 8 | 2,663,224 |

四任务 item / domain token 全部为 8：material 313,148、recommendation
994,312、user_action 899,460、user_chain 456,304。11,072 个 canonical
segment 的全部 supervised token 均为 4。

旧新 cache 比较中，`input_ids`、labels、sample/task IDs、domain weights、
packing、rec metadata 与 attention/position fields 完全相同；仅发生
2→8（1,307,460）和 3→8（1,355,764），其它 loss-weight 变化为 0。

## Epoch2 的 CoT / No-think 现象

修复后的正式 run 使用 Native SID8 CE，关闭 REC-PU 与 PackRatio。进入 Epoch2
后，训练端的 No-think final Gold-SID CE 常呈下降，而 CoT final Gold-SID CE
在较大范围内抖动。该现象不能直接解释为 No-think 泛化改善：固定开发集 probe
中，Epoch1 末至 Epoch2 中段，CoT 与 No-think Gold-SID CE 都有回升迹象。

可能解释是：No-think 输出格式固定、最终 SID 前上下文短，SID8 梯度更集中；CoT
还同时拟合普通推理文本 CE=1 与最终 SID CE=8，文本长度、措辞、领域和多正例
组规模的差异会通过共享 LoRA 参数影响 final-SID 预测。最终选择 Epoch1 还是
Epoch2 checkpoint 应以 epoch-end full-dev 的推荐 CE 和完整 SID chain 指标为准，
不能仅依据训练滚动 loss 或 42-pack probe 的单点。

## 训练前提

正式启动必须使用 `GLOBAL_ITEM_WEIGHT=8`，并通过 persisted cache contract。
训练数据、Arrow cache、checkpoint、日志和 validation JSONL 不提交到仓库。
