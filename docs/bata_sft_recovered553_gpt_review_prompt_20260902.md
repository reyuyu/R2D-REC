# 给网页版 GPT 的 BATA SFT micro-replay 审查提示词

请审查 GitHub 仓库 `reyuyu/onereason-multitask-sft` 的 `main` 分支中以下文件：

1. `docs/bata_sft_recovered553_micro_replay_20260902.md`
2. `docs/bata_sft_recovered553_micro_replay_20260902.json`

背景：目标是恢复 2026-08-12 的 BATA SFT：
`BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333`。

本轮没有从头训练，而是从真实历史 `checkpoint-553` 各复制一个独立起点，
在 recovered 06:33 source、相同 4 张 A800、相同 packed Arrow cache、相同
optimizer/scheduler/RNG 下分别执行 Replay A 和 Replay B，只运行 553 到 560。
`max_steps` 仍保持 1106，使用无训练数学影响的 callback 在 step560 停止。

请重点检查：

1. 报告把结果归为 `CASE C`（相同输入下发生 runtime/kernel
   nondeterminism）是否由证据充分支持。
2. A/B 所有 rank 的 ordered batch fingerprint 和 RNG hash 一致，但 step554
   已出现 loss/grad norm 差异，这是否足以排除 sampler、恢复状态和
   dataloader skip 是 A/B 分叉原因。
3. Replay A 的 step560 loss 非常接近 historical，但 A/B 参数和 optimizer
   fingerprint 已分叉；报告拒绝把这种接近视为“稳定恢复”是否正确。
4. step555 historical loss 与 replay loss 的 logging window 不同，因此报告只把
   step560 当作可直接比较的 loss，这个处理是否正确。
5. checkpoint 恢复必须绕过 Transformers 5.3.0 对 PyTorch 2.5.1 的安全门，
   但仍保留 `weights_only=True` 并只 allowlist NumPy RNG 类型。这个恢复控制路径
   是否可能解释分叉，还是只能视为尚未完全恢复的环境合同。
6. 当前证据能否进一步定位到 DDP/NCCL、FlashAttention、Liger 或 scatter/
   reduction kernel；不能定位时请明确说“证据不足”，不要猜具体 kernel。
7. 下一轮最小化测试应如何设计，才能区分 DDP reduction、FlashAttention、
   Liger 和其他 CUDA nondeterminism，同时避免直接扩成 1106-step 长训练。

请先列出审查发现，按严重程度排序；随后回答：

- CASE 判定是否成立；
- 哪些事实已被证明；
- 哪些事实仍未被证明；
- 下一轮最小测试矩阵；
- 在什么验收条件下才值得重新跑完整 SFT。

不要把流程 PASS、单次 loss 接近或外部分数偶然接近当成历史效果复现成功。
不要建议直接启动 GR_REC、GRPO-TK 或 MC_USER。
