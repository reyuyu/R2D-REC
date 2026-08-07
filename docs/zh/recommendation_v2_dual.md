# 懂推荐 V2 CoT / No-think 双路数据

## 目的

本版本只改造懂推荐训练数据，不改变模型、Trainer、loss、GradNorm、Ortho、packing 或 micro-step。原始 V2 数据保留不覆盖，双路数据作为独立版本使用。

## 分组与去重

分组键是：完整 Prompt（包含用户历史序列）加目标域。目标域由答案中的完整 SID 域标记确定：商品、视频、直播、广告。

同组内收集答案 `</think>` 之后的完整 SID，按完整四元组去重。CoT 正文中的历史 SID 不会被当作 gold。每个去重后的 gold 只生成一条样本。

## 双路分配

- 偶数个 gold：按稳定顺序交替分到 CoT 和 No-think。
- 奇数个 gold 或单 gold 组：使用 `SHA-256(完整 Prompt + 目标域)` 的稳定排序决定多出的一条，并按目标域分别平衡两路数量。
- CoT 保留共同的原始分析和答案前缀，输入以 `/think` 结尾。
- No-think 只保留一个完整 SID，输入以 `/no_think` 结尾，不包含 `<think>`、`</think>`、解释文本或答案前缀。

## 输出与注册

输出目录：

```text
/data/lf_data_versions/alltrain/v2_recommendation_dual/
```

文件和数据集别名：

```text
onereason_recommendation_cot_v2_dual
onereason_recommendation_nocot_v2_dual
```

注册信息位于 `data/dataset_info.json`，版本图位于 `data/onereason_dataset_versions.json`，版本名为 `v2_recommendation_dual`。生成脚本为 `scripts/create_recommendation_v2_dual.py`，重复运行不会覆盖已存在的输出目录。

直接引用两个双路数据集时，可在数据配置中写：

```yaml
dataset: onereason_recommendation_cot_v2_dual,onereason_recommendation_nocot_v2_dual
```

如果通过多任务版本解析使用，逻辑懂推荐数据集会解析到 CoT 双路文件；No-think 别名仍可直接引用。

## 生成统计

| 指标 | 数值 |
| --- | ---: |
| 原始样本数 | 41661 |
| 唯一 Prompt+目标域组数 | 18093 |
| 去重后的 gold 总数 | 41330 |
| 重复 gold 删除数量 | 331 |
| 单 gold 组数 | 11302 |
| 多 gold 组数 | 6791 |
| CoT 不一致组数 | 0 |
| CoT 样本数 | 20664 |
| No-think 样本数 | 20666 |
| CoT / No-think | 20664 : 20666 |

各目标域数量（CoT / No-think）：商品 `1630 / 1631`，视频 `15697 / 15698`，直播 `1538 / 1537`，广告 `1799 / 1800`。

完整性检查通过：`20664 + 20666 = 41330`；没有解析失败、未知域、跳过组或跳过 gold。版本管理器的 4 个数据版本哈希审计也已通过。

## 测试

```bash
PYTHONPATH=src python3 tests/test_recommendation_v2_dual.py
python3 scripts/manage_onereason_datasets.py audit --verify-hashes
```

测试覆盖多 gold 去重、单 gold 分配、CoT/No-think 格式、总量不膨胀和完整性清单。
