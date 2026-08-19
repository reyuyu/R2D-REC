# GR_USER_v1 template alignment and data audit

## Evidence

- Parent: beta-baseline Epoch 2 at `/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106` (`trainer_state.json`: global_step 1106, epoch 2.0).
- Evaluation-template evidence: user-provided `GR-V1-2000_filtered.log`, external aggregate 1.3326. This log is template evidence only and is not the selected parent.
- Evaluation examples: 5 Action and 5 Chain NoCoT examples.
- Action format-only demonstration present: 5/5.
- Chain format/logic demonstration present: 5/5.

## Alignment decision

Both routes use an evaluation-aligned raw user-content adapter. Action receives the stable cross-user JSON-array demonstration. Chain receives the stable cross-user logic-chain demonstration and evaluation wording. Neither demonstration uses the current row's history, topic, or Gold. Raw prompts contain no chat special tokens; the parent tokenizer renderer adds exactly one user wrapper, one assistant head, and the empty NoCoT think block.

## Dataset

- train: 3000 (Action 1500, Chain 1500)
- Chain native/converted: 1200/300
- pilot: 600
- probe: 20 (native only, disjoint from train)
- max rendered prompt tokens: 8161

Full source hashes, rejection reasons, bucket counts, token distributions, IDs, and output hashes are recorded in `manifest.json`. Per-example source/adapted/rendered tails are recorded in `template_alignment_audit.json`.
