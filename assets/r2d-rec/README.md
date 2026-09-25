# README 图片来源

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `banner.svg` | 项目标识、渐进式学习主题、竞赛荣誉 | 为项目首页制作的矢量横幅；获奖名称与排名由项目作者提供 |
| `progressive-framework.png` | 从语义与证据基础到 Reasoning / Decision 的整体方案 | 最终答辩《最终PPT_R2Drec.pdf》第 10 页 |
| `sft-rec-bootstrapping.png` | Multi-positive Set Loss、FDR 与 Top-16/32 HCR 的方法示意 | 项目作者补充的“识物：Semantic Alignment & Rec Bootstrapping”答辩页截图；不在下述 PDF 版本中 |
| `joint-grpo.png` | CoT 与 SID 双目标训练的信用分配 | 同一答辩材料第 20 页 |

`progressive-framework.png` 与 `joint-grpo.png` 由 PDF 页面直接导出，分辨率为 1920 × 1080。保留原图中的赛事标识与机制示例；仅选用方案图，未包含组件评测表或个人介绍页面。图中的奖励示例数值不代表外部评测成绩。

`sft-rec-bootstrapping.png` 提取补充截图中的 SFT 目标说明及三栏方法区，分辨率为 1487 × 893。裁去页眉和右侧分数对照，并移除三个阶段分数标签、还原其背后的面板底色与边线；方法文字、SID 路径、候选排名示例和箭头保留。原截图未随仓库发布，以保持首页仅展示最终模型得分。

补充截图 SHA256：`6b18e4f83709713776ad60c78ed63cb5cbd85ecec763d00d6bfed2baa7315489`。

源 PDF 共 28 页，SHA256：

```text
f968e062c95ec53858dfae28ae09d3604059fd3bb270e222b17e6bdd1858a825
```

图片表达方案逻辑。参数、路由和 parent 继承以[方法映射](../../docs/r2d-rec/METHODS.md)和[复现合同](../../docs/r2d-rec/REPRODUCTION.md)为准。
