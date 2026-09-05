# GRPO-3 User from GRPO-1 checkpoint 300

This launcher keeps the verified 200-step, four-GPU candidate-parallel User
GRPO training semantics and changes only the selected continued-adapter parent.
It loads the Rec FDR V4.3 full SFT model plus the cumulative GRPO-1
checkpoint-300 adapter. It does not merge, stack, or initialize another LoRA.

The parent contract is checked against the GRPO-1 checkpoint lineage and exact
adapter SHA before execution. Checkpoints remain adapter-only and retain the
optimizer, constant scheduler, trainer state, four-rank RNG state, and lineage
needed for continuation.
