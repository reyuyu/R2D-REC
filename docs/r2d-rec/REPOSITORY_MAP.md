# R2D-REC 仓库目录索引

[项目首页](../../README.md) · [文档导航](README.md)

项目文档按方案、工程与上游框架分层；训练文件保留历史路径，以兼容配置、导入关系和复现合同。统一入口见[项目文档总览](../README.md)。

## 展示与阅读层

```text
README.md                         R2D-REC 项目首页与最终结果
docs/README.md                    项目文档总览
docs/project/                     多任务框架、数据版本与早期工程
docs/upstream/                    基础框架说明与引用
docs/r2d-rec/
  README.md                       答辩逻辑与阅读导航
  METHODS.md                      四个方法及实际代码入口
  FINAL_RESULT.md                 最终模型的一行结果与来源
  REPRODUCTION.md                 复现路线、parent 与版本边界
  REPOSITORY_MAP.md               本目录索引
```

## 方法实现层

| 路径 | 角色 | 对应方案 |
| --- | --- | --- |
| [baselines/native_source_domain_r32_v3/scripts](../../baselines/native_source_domain_r32_v3/scripts) | 多任务与物料对齐数据处理、SFT 启动和校验 | 识物 |
| [baselines/native_source_domain_r32_v3/config](../../baselines/native_source_domain_r32_v3/config) | 历史 SFT 实验配置 | 识物 |
| [baselines/native_source_domain_r32_v3/grpo/user](../../baselines/native_source_domain_r32_v3/grpo/user) | Action / Chain 奖励、边际信用、Hybrid 训练器 | 察行 |
| [baselines/native_source_domain_r32_v3/grpo/scripts](../../baselines/native_source_domain_r32_v3/grpo/scripts) | 推荐数据分组、Think / NoThink runner、Beam reward 与 sampler | 推意相关机制及双路历史实现 |
| [grpo/ablations/gr_rec_think_sample8_fullsid_v3](../../baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3) | CoT G4 + 每条 CoT 后独立 FullSID G8，双目标训练 | 择物 |
| [baselines/native_source_domain_r32_v3/boundary_adapt](../../baselines/native_source_domain_r32_v3/boundary_adapt) | Bridge 与推理/答案边界对照实验 | 探索 |

`ablations/` 是历史工程目录名，不表示其中的 Joint 双目标实现只能作为消融使用。阅读入口以方法映射为准。

## 复现、数据与工具层

| 路径 | 作用 |
| --- | --- |
| [reproduction/final_chain_20260901](../../reproduction/final_chain_20260901) | 历史 adapter 链的环境锁、数据来源、checkpoint 验收和只读复现监控 |
| [reproduction/rec_fdr_v43_strictdet](../../reproduction/rec_fdr_v43_strictdet) | 后续 full-SFT 的代码快照、数据重建、环境及 SHA 合同 |
| [baselines/native_source_domain_r32_v3/grpo_fullbase_conservative_v1](../../baselines/native_source_domain_r32_v3/grpo_fullbase_conservative_v1) | full-model parent 接入推荐 GRPO 的独立路线 |
| [grpo/scripts/monitor](../../baselines/native_source_domain_r32_v3/grpo/scripts/monitor) | 推荐与用户实验的监控服务和前端 |
| [数据版本管理](../project/DATASET_VERSIONS.md) | 既有数据版本索引 |
| [多任务框架扩展](../project/MULTITASK.md) | 多任务设计与训练框架接入说明 |

数据目录中的 manifest 和 registry 描述数据合同，不等于已经附带训练样本。实际数据与权重仍需在运行环境中独立准备。

## 历史实验与基础框架

| 路径 | 阅读方式 |
| --- | --- |
| [baselines/native_source_domain_r32_v3/docs](../../baselines/native_source_domain_r32_v3/docs) | Alpha、Mini、BETA 等历史试验；按实验日期和配置理解 |
| [grpo/ablations](../../baselines/native_source_domain_r32_v3/grpo/ablations) | 独立推荐目标、采样与奖励方案；除明确选中实现外，不自动纳入主线 |
| [truerec_grpo](../../baselines/native_source_domain_r32_v3/truerec_grpo) | 另一条奖励与训练合同探索 |
| [实验记录](../../实验记录) | 早期阶段记录 |
| [src/llamafactory](../../src/llamafactory) | 训练框架实现 |
| [examples](../../examples)、[configs](../../configs)、[requirements](../../requirements) | 框架示例、配置及依赖 |
| [基础框架说明](../upstream/LLAMA_FACTORY_ZH.md) | 沿用的 LLaMA-Factory 文档 |

旧首页的完整实验叙述保留在 Git 历史中。历史技术记录和原始结果摘要用于追溯，项目展示入口只列方案和最终结果。
