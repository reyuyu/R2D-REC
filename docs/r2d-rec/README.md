# R2D-REC 文档导航

[返回项目首页](../../README.md)

本目录按照最终答辩《最终PPT_R2Drec.pdf》的项目逻辑整理。介绍顺序为：赛题问题、整体方案、四个方法模块、最终结果、复现与探索。这里不列组件中间成绩或消融分数。

## 研究问题

OneReason 的任务覆盖懂物料、懂用户、懂推荐和懂世界。R2D-REC 将推荐过程划分为两个相互关联的能力：

- **兴趣推理能力**：基于可靠的物料语义，从长历史中找到相关行为证据，并形成有助于推荐的 CoT。
- **推荐决策能力**：将兴趣推理转化为准确、具有覆盖性的 SID 候选。

对应两个研究问题：如何得到更可靠的推理；如何让模型真正利用推理完成推荐。

## 按答辩顺序阅读

| 阅读顺序 | 答辩页码 | 内容 | 对应文档 |
| --- | --- | --- | --- |
| 1 | 5–10 | 赛题理解与 Reasoning-to-Decision 总体方案 | [项目首页](../../README.md) |
| 2 | 12 | 识物：物料语义与 SID 对齐 | [Semantic Alignment SFT](METHODS.md#semantic-alignment-sft) |
| 3 | 13–15 | 察行：行为证据定位与边际信用 | [MCH-GRPO](METHODS.md#mch-grpo) |
| 4 | 16–18 | 推意：推荐结果驱动的 CoT 优化 | [ORR-GRPO](METHODS.md#orr-grpo) |
| 5 | 19–21 | 择物：推理与 SID 动作的联合优化 | [Joint-GRPO](METHODS.md#joint-grpo) |
| 6 | 23 | 最终模型结果 | [最终结果](FINAL_RESULT.md) |
| 7 | 25–27 | Bridge 接口差异与模型行为分析 | [探索入口](METHODS.md#bridge) |

## 按开发任务阅读

- 找代码：看[方法与代码映射](METHODS.md)，每个模块均提供实际存在的实现入口。
- 准备运行：看[复现指南](REPRODUCTION.md)，先区分历史 adapter 链与后续 full-SFT 路线。
- 查目录：看[仓库目录索引](REPOSITORY_MAP.md)，区分方案主线、实验分支、框架代码和工具。
- 核对模型：看[最终结果说明](FINAL_RESULT.md)，使用明确的运行身份和原始评测记录。

## 文档与版本边界

方案命名以最终答辩为准；训练参数、数据构成和 parent 继承以对应版本的源码、配置及 manifest 为准。方法展示顺序不自动改变历史训练顺序，也不将历史双路 runner 改称纯 Think-only 实现。

参考 PDF 共 28 页，SHA256：

```text
f968e062c95ec53858dfae28ae09d3604059fd3bb270e222b17e6bdd1858a825
```

方法映射以代码点 `da1331f` 为核对基准。文档整理不改变训练实现和 checkpoint 合同；旧首页可从 Git 历史查阅，历史技术记录继续保留。
