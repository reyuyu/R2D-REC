---
license: other
task_categories:
- text-generation
language:
- zh
tags:
- recommendation
- sft
- deterministic-training
---

# Rec FDR V4.3 S1 严格确定性复现代码

这里保存 `rec_fdr_v43_s1_topk_strictdet_fullft_32k_4gpu_20260903` 的训练代码快照、
冻结配置、数据生成脚本、环境锁与参考运行元数据。GitHub 版本是 **code-only mirror**：
不包含训练数据、tokenized cache、基础模型、训练权重或运行日志。

原始完整复现包来自：

- 数据与代码：[yuanchu0126/rec_fdr_v43_s1_topk_strictdet_reproduction](https://modelscope.cn/datasets/yuanchu0126/rec_fdr_v43_s1_topk_strictdet_reproduction)
- 参考权重：[yuanchu0126/rec_fdr_v43_s1_topk_strictdet_fullft_32k_4gpu_20260903](https://modelscope.cn/models/yuanchu0126/rec_fdr_v43_s1_topk_strictdet_fullft_32k_4gpu_20260903)
- 公共基础模型：[OpenOneRec/OneReason-8B-pretrain-competition](https://modelscope.cn/models/OpenOneRec/OneReason-8B-pretrain-competition)

第三方 LLaMA-Factory 快照保留其原始 `LICENSE`，精确 commit 记录在
`vendor/LLaMA-Factory/COMMIT`。

## 参考合同

- OneReason 8B：`8,389,956,608` 个参数元素
- 训练方式：4 卡 FSDP、BF16 全参数 SFT
- 训练长度：1 epoch / 475 optimizer steps
- 单卡 batch size：1
- gradient accumulation：4
- global batch size：16
- cutoff length：32768
- learning rate：`2e-5`
- scheduler：cosine
- seed / data seed：`19260817`
- 参考 train loss：`1.1368073997999493`
- 参考 validation loss：`1.4387`
- 参考权重 SHA256：`8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59`

参考权重曾记录 `1.34+` 外部成绩。这不是对当前复刻运行的提前结论。当前运行只有在
完成 475 steps、文件可读且最终权重 SHA256 与参考值一致后，才能确认权重级复现；
外部测评仍应单独记录，不能由训练 loss 推导。

## 数据准备

GitHub 不提供数据。先将 ModelScope 完整数据包中的 `data/` 下载到独立目录，例如：

```text
/path/to/rec_fdr_v43_data/
  base/
  recommendation/
```

`PARQUET_MANIFEST.json` 锁定实际训练分片，`RAW_800K_MANIFEST.json` 锁定上游原始
Parquet；它们仅含文件合同和哈希，不含样本正文。数据生成链位于
`scripts/data_generation/`，从原始约 80 万条数据重建的入口是：

```bash
python3 scripts/rebuild_rec_fdr_v43_from_raw_800k.py \
  --raw-root /path/to/SecondRoundDataWithCaptionNoInternalPaths \
  --base-model /path/to/OneReason-8B-pretrain-competition \
  --work-root /path/to/data-rebuild
```

## 启动

使用外置数据目录启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/launch_rec_fdr_v43_reproduction.sh \
  --base-model /path/to/OneReason-8B-pretrain-competition \
  --data-root /path/to/rec_fdr_v43_data \
  --work-root /path/to/new-run
```

只验证数据、基础模型、配置和 tokenized cache，不启动训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/launch_rec_fdr_v43_reproduction.sh \
  --base-model /path/to/OneReason-8B-pretrain-competition \
  --data-root /path/to/rec_fdr_v43_data \
  --work-root /path/to/cache-check \
  --prepare-only
```

启动器依次执行数据 SHA/行数校验、基础模型逐文件 SHA 校验、CPU 单元测试、绝对
路径配置生成、配置校验、32K packed pool 重建和 4 卡训练。默认不保存中间权重，
结束时保存最终 BF16 全量模型。

## 严格确定性设置

除固定 seed 外，启动器还固定：

```text
FLASH_ATTENTION_DETERMINISTIC=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NVIDIA_TF32_OVERRIDE=0
```

字节级复现还要求 GPU 型号/数量、驱动、CUDA、PyTorch、FlashAttention、FSDP/NCCL
拓扑、基础模型和输入数据一致。边界见 `ENVIRONMENT.md`。

## 后续 GRPO

该训练输出是全量模型，不是 PEFT adapter。接入现有 GR_REC / MC_USER 前必须遵循
`docs/GRPO_COMPATIBILITY.md`：将它作为新的只读 base model，第一段 GRPO 创建新的
LoRA；禁止将历史上基于另一 base 的 adapter 直接套到该权重上。

供 ChatGPT 网页版审查新方案的完整提示词见 `docs/CHATGPT_REVIEW_PROMPT.md`。

## 目录

```text
scripts/                     启动、校验、配置生成与数据重建入口
scripts/data_generation/     数据构造脚本链（不含数据）
training/                    实际训练器、HCR loss、测试与 FSDP 配置
vendor/LLaMA-Factory/        精确框架源码快照及许可证
reference_run/               小型参考结果与运行合同（不含权重和完整日志）
PARQUET_MANIFEST.json        训练数据文件合同（无样本正文）
RAW_800K_MANIFEST.json       上游数据文件合同（无样本正文）
BASE_MODEL_MANIFEST.json     基础模型文件合同（无模型权重）
PACKAGE_MANIFEST.json        原完整 ModelScope 包的来源清单
```
