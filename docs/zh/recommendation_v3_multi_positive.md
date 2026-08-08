# Recommendation V3 Multi-Positive 数据版本

## 目的

V3 基于 `v2_recommendation_dual` 生成一个独立的数据版本，只增加推荐样本的组级多正例元数据，不修改任何训练文本。它为后续 recommendation 多正例 loss、合法集合或分支级统计提供数据入口；本版本本身不实现新的 loss、模型 forward、optimizer、GradNorm、Ortho、packing 或 scheduler。

## 分组键与 gold

每行先去掉输入末尾的 `/think` 或 `/no_think`，然后使用下列严格键分组：

```text
JSON([instruction, canonical_input_without_mode, history]) + "\0" + target_domain
```

`target_domain` 从答案 `</think>` 之后（No-think 则为完整 output）的完整 SID 域标记取得，域为商品、视频、直播或广告。COT 正文中出现的历史 SID 不会被当成 gold。

同组内收集并按 V2 双路文件的稳定顺序去重完整 SID 字符串。每个去重 gold 仍保留为一行，V3 不合并行、不扩增样本、不改变 CoT / No-think 路由。若同组 COT 正文不一致，生成器记录数量和示例，不静默选择或重写正文；当前全量数据不一致组数为 0。

## 增加的元数据

V3 每行在原有字段之外增加：

```json
{
  "recommendation_group_id": "SHA256(canonical_prompt + \"\0\" + target_domain)",
  "recommendation_group_size": 2,
  "recommendation_all_gold_sids": ["<|prod_begin|>..."],
  "recommendation_current_gold_sid": "<|prod_begin|>..."
}
```

元数据保存完整 SID 文本，不保存 tokenizer id。`recommendation_all_gold_sids` 只包含同一完整 Prompt 与目标域组内的 gold。

## 输出与注册

原始 V2 文件保持不变：

```text
/data/lf_data_versions/alltrain/v2_recommendation_dual/
```

V3 输出：

```text
/data/lf_data_versions/alltrain/v3_recommendation_multi_positive/
  onereason_recommendation_cot_v3_multi_positive.jsonl
  onereason_recommendation_nocot_v3_multi_positive.jsonl
```

注册别名为：

```text
onereason_recommendation_cot_v3_multi_positive
onereason_recommendation_nocot_v3_multi_positive
```

版本名是 `v3_recommendation_multi_positive`，父版本是 `v2_recommendation_dual`。在多任务配置中仍使用逻辑名 `onereason_recommendation_cot` / `onereason_recommendation_nocot`，并设置：

```yaml
multitask_dataset_version: v3_recommendation_multi_positive
multitask_dataset_version_manifest: data/onereason_dataset_versions.json
```

生成脚本：`scripts/create_recommendation_v3_multi_positive.py`；完整性审计：`scripts/audit_recommendation_v3_multi_positive.py`。两个脚本都拒绝覆盖已有输出。

## 全量统计

| 指标 | 数值 |
| --- | ---: |
| V2 父版本原始样本数 | 41661 |
| V2 COT 双路行数 | 20664 |
| V2 No-think 双路行数 | 20666 |
| V2 双路行数 / V3 输入行数 | 41330 |
| 唯一 Prompt+目标域组数 | 18093 |
| 去重后的 gold 总数 | 41330 |
| 重复 gold 删除数量 | 0 |
| 单 gold 组数 | 11302 |
| 多 gold 组数 | 6791 |
| CoT 不一致组数 | 0 |
| CoT 输出行数 | 20664 |
| No-think 输出行数 | 20666 |
| CoT / No-think | 20664 : 20666 |

组大小统计：均值 `2.2843`，P50 `1`，P90 `5`，P95 `7`，最大 `20`。各域 CoT / No-think：商品 `1630 / 1631`，视频 `15697 / 15698`，直播 `1538 / 1537`，广告 `1799 / 1800`。

完整性等式成立：`20664 + 20666 = 41330`。输出文件 SHA-256：

```text
cot:   a9fcc61db0d0c30e97034f1b920910602d5c1b27ba940b83e283b2d27d99b738
nocot: 84043f25199ad428317a0f527f6a17dd39802562c47393aa77ca3a2fa2012dd8
```

审计同时确认：每个 group id 可由键重算；组大小、all gold 列表和 current gold 一致；同组 current gold 不重复；去掉新增元数据后，V3 与 V2 的原有字段多重集合完全相等。

## 管线接入

1. `AlpacaDatasetConverter` 将四个 `recommendation_*` 字段收拢为内部 `_recommendation_metadata`。
2. supervised processor 在 tokenization 后输出 `recommendation_metadata`，不把它拼接到输入或 labels；普通 packed SFT 路径也保留该字段。
3. `TokenizedSubDataset` 将它合并到现有 `sample_metadata["recommendation_multi_positive"]`。Action Select metadata 使用 `update` 合并，不会相互覆盖。
4. `TaskPackCollator` 原样携带每个 segment 的 `sample_metadata`，所以多个 segment 的组状态互不污染。
5. Trainer 的 `_MODEL_INPUT_KEYS` 不包含这些 metadata key，故它们不会进入 `model.forward`，也不会改变 loss、GradNorm、Ortho 或 DDP collective。

## 测试与限制

CPU 测试命令：

```bash
PYTHONPATH=src python3 tests/test_recommendation_v3_multi_positive.py
python3 scripts/audit_recommendation_v3_multi_positive.py
```

当前两项均通过。测试覆盖文本字段恒等、converter/processor 透传、TokenizedSubDataset 合并、pack segment 元数据和 model input 过滤。本次没有 GPU smoke、没有 forward/backward、没有启动训练，也没有触碰正在运行的 REC_F 与 REC_G1。

V3 只提供数据和 provenance，不会自动改变模型行为；后续若实现多正例目标，必须另建配置和实验，继续保持 V2/V3 文件不可变。
