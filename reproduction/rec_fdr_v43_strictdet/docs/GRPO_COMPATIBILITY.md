# 全量 SFT 模型接入后续 GRPO 的兼容合同

## 结论

Rec FDR V4.3 S1 的输出是 BF16 全参数模型。它可以成为后续 GRPO 的 parent，但接入
方式与历史“公共 OneReason base + SFT LoRA adapter”不同：新权重本身就是 base。

```text
OneReason-8B pretrain
  -> Rec FDR V4.3 full-parameter SFT (new immutable GRPO base)
  -> fresh GR_REC LoRA
  -> later GR_REC LoRA checkpoint
  -> MC_USER Hybrid K4 LoRA refinement
```

## 必须满足的加载合同

1. `base_model` 指向完成训练的 Rec FDR V4.3 全量模型目录。
2. 目录必须包含可读的 `model.safetensors`、`config.json`、tokenizer 与 generation config。
3. 首个 GRPO 阶段在这个 base 上创建 fresh LoRA，不要求 SFT `adapter_model.safetensors`。
4. 后续阶段加载同一个 full-model SHA，再加载由它派生的 LoRA adapter。
5. 每个 manifest 同时记录 `base_model_sha256`、`adapter_sha256` 和 parent lineage。
6. GRPO checkpoint 继续保存 adapter-only，不复制 16.8 GB 全量 base。
7. Probe、外评、monitor 和恢复逻辑使用同一对 base + adapter。
8. 不得加载基于历史 BATA/SFT base 训练的旧 adapter；shape 相同不代表语义兼容。

## 现有代码的兼容缺口

当前部分 GR_REC / MC_USER runner 会强制校验 `adapter_config.json` 与
`adapter_model.safetensors`，并使用 `PeftModel.from_pretrained(base, adapter)`。这适合
adapter parent，但不接受 full-model-only parent。正式改造应显式支持两种模式：

```text
parent_mode=adapter     base_model + parent_adapter
parent_mode=full_model full_model_path + fresh LoRA initialization
```

不要通过空 adapter、伪造 adapter config、把完整权重改名或先 merge 再反解 LoRA 来
绕过合同。

## 上线前门禁

- 当前复刻运行完成 475/475，无 traceback/OOM/non-finite/FSDP/NCCL fatal。
- 最终 safetensors 可读取，399 个 tensor 的合同与参考运行一致。
- 最终 SHA256 与参考 SHA 比较并写入 manifest。
- tokenizer/config 与 OneReason 8B 架构兼容。
- CPU test 覆盖两种 parent mode 的路径和 manifest 序列化。
- 1–5 step GPU smoke 验证 fresh LoRA、base hash 不变、梯度仅进入 LoRA。
- 旧 adapter-parent 流程回归通过。
- 先做固定 probe，再决定是否进入完整 GRPO；训练 loss 不能替代效果门禁。

## 当前成绩措辞

参考权重已有 `1.34+` 记录，但当前复刻尚未完成和外评。除非最终权重 SHA 与参考值
完全一致，否则只能称为“同配置复刻候选”，不能直接继承该成绩。
