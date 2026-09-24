# R2D-REC 复现入口与模型来源

[文档导航](README.md) · [方法映射](METHODS.md) · [目录索引](REPOSITORY_MAP.md)

## 先选择复现目标

仓库包含历史 adapter 链和后续 full-SFT 路线。它们使用不同的 base / parent 合同，不能仅凭相似的方法名称混用。

| 目标 | 入口 | 说明 |
| --- | --- | --- |
| 定位最终选中模型的历史训练来源 | [final_chain_20260901](../../reproduction/final_chain_20260901/README.md) | BETA SFT → 双路推荐 GRPO → GRPO-TK → MC_USER Hybrid |
| 独立复现全参数 SFT | [rec_fdr_v43_strictdet](../../reproduction/rec_fdr_v43_strictdet/README.md) | 4-GPU FSDP，冻结原始数据、重建脚本、环境与参考权重合同 |
| 将全参数 SFT 输出接入推荐 GRPO | [grpo_fullbase_conservative_v1](../../baselines/native_source_domain_r32_v3/grpo_fullbase_conservative_v1/README.md) | 独立 full-model parent 路线，先检查 parent 合同 |
| 单独理解或运行 Joint 双目标实现 | [GRPO-TK 复现指南](../reproduce_GRPO_TK.md) | G4 CoT，每条 CoT 后独立 G8 FullSID |

本次 GitHub 整理不启动训练，也不修改上述训练配置。

## 答辩主线与实际训练顺序

答辩的方案组织为：

```text
识物 SFT → 察行 MCH-GRPO → 推意 ORR-GRPO → 择物 Joint-GRPO
```

仓库现有、与最终结果关联的历史冻结链为：

```text
BETA SFT checkpoint-1106
  → GR_REC_v1 checkpoint-1500
  → GRPO-TK checkpoint-250
  → MC_USER Hybrid K4 prompt-step-0100
```

第一个序列用于解释能力如何组织，第二个序列用于追溯实际模型。两者顺序不同；不应通过改名、重新排列脚本或改写结果文件使其看起来相同。

## 历史 adapter 链

入口：[reproduction/final_chain_20260901/run.sh](../../reproduction/final_chain_20260901/run.sh)。

先查看脚本支持的模式：

```bash
bash reproduction/final_chain_20260901/run.sh --help
```

该入口是复现包控制器。执行 `--preflight` 或正式训练前，需要按其 README 准备冻结数据、基础模型、环境锁、`code/` 源码快照与 `dependencies/llamafactory/` 依赖快照；普通 Git clone 并不包含全部运行资产。

在已准备好的独立复现包目录内：

```bash
bash run.sh --preflight
bash run.sh --no-monitor
```

第二条会启动训练。控制器还提供 `--monitor-only` 和 `--from-stage`；已有阶段只有在 PASS 与 checkpoint 合同核验通过后才能跳过。具体可恢复性以该 runner 的合同为准，不把不同日期实现的 resume 能力相互套用。

数据来源与分组规则见 [DATASET_PROVENANCE.md](../../reproduction/final_chain_20260901/docs/DATASET_PROVENANCE.md)。数据根目录可以通过 `REPRO_DATA_ROOT` 指定，文件身份与划分必须符合该复现包的冻结合同。

## 后续全参数 SFT 路线

[Rec FDR V4.3](../../reproduction/rec_fdr_v43_strictdet/README.md) 是代码与合同镜像，输出完整模型。数据重建入口为 [rebuild_rec_fdr_v43_from_raw_800k.py](../../reproduction/rec_fdr_v43_strictdet/scripts/rebuild_rec_fdr_v43_from_raw_800k.py)，训练入口为 [launch_rec_fdr_v43_reproduction.sh](../../reproduction/rec_fdr_v43_strictdet/scripts/launch_rec_fdr_v43_reproduction.sh)。

使用它的 README 中的 `--prepare-only` 可先准备和校验数据、基础模型、配置与缓存。该步骤仍可能消耗较多 CPU、内存和磁盘，应在有足够资源的运行环境中执行。

全参数 SFT 输出作为新的 base 时，不能直接套用另一 base 上训练得到的历史 adapter。后续 GRPO 遵循 [GRPO_COMPATIBILITY.md](../../reproduction/rec_fdr_v43_strictdet/docs/GRPO_COMPATIBILITY.md) 的 parent 检查与初始化规则。

## 其他分支的 pipeline

整理时核对的主分支代码点为 `da1331f`，其中没有根级 `reproduce_pipeline/`。后续全链条入口保存在另一个已存在的分支快照中：

[reproduce_pipeline，固定到 878723e 的 README](https://github.com/reyuyu/R2D-REC/blob/878723e/reproduce_pipeline/README.md)。

它的阶段开关、预算、数据 registry 与 parent resolver 属于该版本的运行合同。本次不将整条开发分支并入主分支，也不把该入口描述为本分支已经具备的一键 R2D-REC 答辩顺序复现。

## 确定性与结果验收

数据字节、顺序、seed、sampler、采样配置、优化器、基础模型、GPU 拓扑及软件环境都应固定。历史 adapter 链只固定合同，并不承诺跨硬件字节一致；full-SFT 的严格确定性要求见其环境文档。

验收依次核对：训练完成状态、step 合同、模型或 adapter 文件完整性、parent 来源、权重 SHA / 张量比较，最后单独进行外部评测。训练 loss、reward 或一次 smoke 通过不能替代最终模型的外部成绩。
