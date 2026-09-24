# R2D-REC 项目文档

[项目首页](../README.md)按照“识物 → 察行 → 推意 → 择物”介绍最终方案，仅展示最终选中模型的成绩。

## 阅读路线

| 层次 | 内容 | 入口 |
| --- | --- | --- |
| 方案 | 方法机制、源码对应、最终结果 | [R2D-REC](r2d-rec/README.md) |
| 探索 | 独立 tricks、诊断与验证状态 | [探索清单](r2d-rec/EXPLORATIONS.md) |
| 复现 | 实际训练顺序、parent、环境与数据合同 | [复现指南](r2d-rec/REPRODUCTION.md) |
| 工程 | 多任务宏步训练、早期数据版本管理 | [训练框架扩展](project/MULTITASK.md)、[数据版本](project/DATASET_VERSIONS.md) |
| 历史 | 早期工程快照与独立对照实验 | [早期 SFT 工程](project/EARLY_SFT_WORKSPACE.md)、[实验记录](../实验记录/README.md)、[基线记录](../baselines/native_source_domain_r32_v3/docs) |
| 上游 | LLaMA-Factory 说明和引用信息 | [框架说明](upstream/LLAMA_FACTORY_ZH.md)、[引用](upstream/LLAMA_FACTORY.cff) |

## 目录约定

```text
README.md                  最终方案与最终成绩
docs/r2d-rec/              方法、结果和复现导航
docs/project/              项目训练框架与历史数据管理
docs/upstream/             基础框架说明与引用
reproduction/              版本化复现入口、环境锁和数据合同
baselines/                 方法实现及独立实验
src/llamafactory/          底层训练框架
tests/                    回归检查
```

训练实现沿用历史路径，以兼容配置、导入关系和已记录的复现版本。顶层 `docs/` 中其余实验文档及 `docs/en/`、`docs/zh/` 保留为技术参考；它们不代表最终链条新增阶段。完整源码入口见[目录索引](r2d-rec/REPOSITORY_MAP.md)。

历史实验中的局部诊断和原始结果用于追溯，不自动构成最终方案的收益结论。
