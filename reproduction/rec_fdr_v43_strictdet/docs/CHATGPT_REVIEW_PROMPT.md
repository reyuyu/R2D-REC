# ChatGPT 网页版审查提示词

请审查 GitHub 仓库中：

`reproduction/rec_fdr_v43_strictdet/`

不要只做文字总结，请逐文件核对训练器、HCR objective、FSDP 配置、启动器、数据合同、
严格确定性设置，以及 `docs/GRPO_COMPATIBILITY.md`。背景如下：

1. 基础模型是 OneReason 8B pretrain，参数元素总数应为 8,389,956,608。
2. 新方案是 4 卡 FSDP、BF16 全参数 SFT，1 epoch / 475 optimizer steps，cutoff 32768，
   per-device batch 1，gradient accumulation 4，global batch 16，cosine LR 2e-5，
   seed/data_seed 19260817。
3. 目标参考运行的 train loss 为 1.1368073997999493，validation loss 为 1.4387，最终
   `model.safetensors` SHA256 为
   `8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59`。
4. 参考权重曾记录 1.34+ 外部成绩；当前独立复刻仍在运行，尚不能宣称达到该分数。
5. GitHub 是 code-only mirror，不包含数据、模型、cache 或日志；数据通过外置
   `--data-root` 提供并由 PARQUET_MANIFEST.json 校验。
6. 后续要兼容现有 GR_REC 和 MC_USER Hybrid K4。新 SFT 输出是全量模型，而现有链路
   多数默认“公共 base + PEFT adapter”。拟采用：全量 SFT 输出作为新的 immutable
   base，第一段 GRPO 新建 fresh LoRA，后续 checkpoint 仍 adapter-only。

请重点回答并给出代码级证据：

1. 当前训练代码是否真的执行 full-parameter SFT，而不是 LoRA 或部分参数训练？
2. 475 steps、batch/accumulation、scheduler、packing、数据顺序和严格确定性合同是否
   自洽？是否存在导致复刻不一致的隐藏随机源？
3. HCR loss 的符号、归一化、known-positive exclusion、FDR 边界和 curriculum 是否有
   数学或实现错误？
4. code-only mirror 在外置数据模式下是否闭合？校验器是否仍错误依赖仓库内 data/？
5. 将 full SFT 模型作为后续 GRPO base 是否正确？哪些 runner/config/probe/checkpoint
   合同必须改成 `parent_mode=full_model`？
6. 如何确保 fresh LoRA 只更新 LoRA、全量 base hash 不变，并防止旧 adapter 错配新
   base？
7. tokenizer、embedding、chat template、`qwen3_nothink` 与现有 GRPO prompt/rendering
   是否存在不兼容？
8. 从“参考权重 1.34+”到“当前复刻可进入 GRPO”之间，最小但充分的门禁是什么？
9. 是否需要 Recommendation retention / KL；若需要，应相对哪个 reference policy、
   在什么 token 范围计算，如何避免破坏 MC_USER 的目标？
10. 请按 P0/P1/P2 输出问题，并给出最小改动方案、CPU tests、5-step GPU smoke 和正式
    GRPO 启动前 checklist。不要建议先直接跑完整 GRPO。

最后请明确给出以下结论之一：

```text
READY_TO_IMPLEMENT_FULL_MODEL_GRPO_COMPATIBILITY
BLOCKED_BY_TRAINING_REPRODUCTION
BLOCKED_BY_CODE_OR_DATA_CONTRACT
```
