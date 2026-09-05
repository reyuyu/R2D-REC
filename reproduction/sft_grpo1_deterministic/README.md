# Deterministic SFT to GRPO-1 Chain

This package runs the two already validated stages without changing their training math:

1. Rec FDR V4.3 strict-deterministic full-parameter SFT, 4-GPU FSDP, one epoch and 475 optimizer steps.
2. GRPO-1 recommendation bilateral training from the verified SFT model, fresh rank-32 LoRA, 500 continuous optimizer steps.

No dataset or model bytes are stored in Git.

## One-command launch

Run from a clean checkout. The output root must not exist and must be outside the checkout.

```bash
bash reproduction/sft_grpo1_deterministic/run.sh \
  --run-root /root/sft_grpo1_chain_RUN_ID \
  --sft-data-root /path/to/rec_fdr_v43_package/data \
  --grpo-data /path/to/rec_mp_grpo_v2/train.jsonl
```

The default base model is `/data/models/onereason-8b-pretrain-competition`; override it with `--base-model` when needed.

## Fail-closed contracts

- exactly four visible training GPUs, with no existing compute process before launch;
- clean Git worktree and a fresh, locked run root;
- frozen SFT data/package and OneReason-8B base verification;
- SFT output must finish at step 475 and match the already reproduced model/config SHA256;
- GRPO-1 parent, dataset, seeds, sampler, loss, optimizer, LoRA, generation and deterministic environment remain frozen;
- GRPO-1 must run continuously from step 0 to 500 and finish with a complete adapter-only resume checkpoint;
- both the SFT full-model SHA and the GRPO-1 step-500 adapter SHA must match the already completed reference runs byte for byte;
- no automatic retry, resume, overwrite, evaluation, merge or upload.

## Retained model states

Only two model states are retained by contract:

- `checkpoints/01_sft_final` points to the complete SFT model;
- `checkpoints/02_grpo1_step500` points to the complete GRPO-1 adapter checkpoint.

The SFT stage uses `save_strategy: no`. GRPO-1 uses `save_steps: 500` and `save_total_limit: 1`, so checkpoints 100/200/300/400 are never written. Logs, monitor metrics and compact evidence remain available and are not counted as model checkpoints.
