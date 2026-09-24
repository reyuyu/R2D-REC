# R2D-REC 最终结果

[项目首页](../../README.md) · [文档导航](README.md)

本页只展示最终选中模型的一次外部评测结果，不展示各组件成绩或消融分数。

## 竞赛荣誉

- 决赛获得 **技术创新专项奖**。
- 最终排名 **第 14 名**。

获奖名称与最终名次由项目作者提供；下面的成绩表及来源记录对应最终选中模型。

## 最终模型成绩

| 模型 | 懂物料 | 懂用户 | 懂推荐 | 懂世界 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **R2D-REC 最终模型** | **0.1818** | **0.2572** | **0.6892** | **0.2357** | **1.3639** |

## 来源与模型身份

- 展示来源：《最终PPT_R2Drec.pdf》第 23 页的最终模型行。
- 原始结果：[mc_user_hybrid_strongparent_final_v1.json](../../baselines/native_source_domain_r32_v3/grpo/user/results/mc_user_hybrid_strongparent_final_v1.json) 中 `external_evaluations` 的 `label=repeat_a`。
- 对应运行：`MC-USER-HYBRID-K4-STRONGPARENT-LR3E7-200-20260831-171920`。
- 选中 checkpoint：`prompt-step-0100`。
- 实际模型继承：[历史四阶段复现入口](../../reproduction/final_chain_20260901/README.md)。

答辩用 R2D-REC 的方法主线介绍完整方案；已有结果文件将这次最终模型评测记录在 strong-parent MC_USER 精修运行中。此处保留其真实运行身份，不将该记录改写为一次新的 Joint-only 实验。

## 评测口径

外部结果包含 11 个子指标：懂物料 4 项、懂用户 2 项、懂推荐 4 项、懂世界 1 项。上表为分组汇总与评测器总分，均取自同一条已记录观测。

该结果是最终答辩选用的测次，不代表重复评测均值，也不构成跨数据、硬件或运行环境的成绩保证。后续复现应单独记录所用 checkpoint、数据合同、评测设置及结果。
