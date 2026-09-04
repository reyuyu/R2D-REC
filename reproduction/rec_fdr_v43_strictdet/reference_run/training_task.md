# Rec FDR v43 S1 strict deterministic full SFT

- 任务 ID：`rec_fdr_v43_s1_topk_strictdet_fullft_32k_4gpu_20260903`
- 创建时间：2026-09-03T17:11:54+08:00

## 目的与假设

Use the deterministic configuration validated by the 10-step A/B benchmark to run one complete SFT from the shared pretrained initialization.

验证假设：The strict FA2, cuBLAS, CUDA connection, NCCL Ring/Simple, and TF32 settings will preserve the intended 475-step SFT trajectory while making an independently repeated run byte reproducible on the same stack.

## 相对基线的改动

- Set FLASH_ATTENTION_DETERMINISTIC=1, CUBLAS_WORKSPACE_CONFIG=:4096:8, CUDA_DEVICE_MAX_CONNECTIONS=1, NCCL_ALGO=Ring, NCCL_PROTO=Simple, and NVIDIA_TF32_OVERRIDE=0.
- Disable intermediate checkpoint writes while retaining evaluation milestones at 20%, 40%, 80%, and 100%.
- Save only the final BF16 full-model weights in the run root and record their SHA-256.

基线任务：rec_fdr_v43_s1_topk_from0_fullft_32k_4gpu

## 数据与模型

- 数据摘要：Frozen rec_fdr_v43_hcr_frozen_v42a training and external deterministic validation split, using the established 32K tokenized curriculum pools.
- 数据来源：rec_full_k1_train, rec_full_val, rec_fdr_v43_hcr_frozen_v42a
- 基础模型：/data/LLm-8B/code/OneReason-8B-pretrain-competition
- 训练方法：Full-parameter BF16 SFT with FA2, Liger, gradient checkpointing, and four-GPU FSDP full shard
- 配置文件：`training_config.yaml`

## 预期结果

Complete all 475 optimizer steps without recovery, write one loadable final model.safetensors, and preserve full trainer logs and evaluation milestones for later stability assessment.

## 备注与风险

Completed all 475 optimizer steps without recovery. Final train loss was 1.1368073997999493 and runtime was 22108.9432 seconds. No intermediate checkpoint directory was produced. The final 16,779,959,728-byte safetensors file passed its recorded SHA-256 check and exposed all 399 tensors. The six inference files were uploaded to the verified private ModelScope repository recorded in modelscope_upload.
