# GRPO-2 Think-only continued adapter validation

This package restores the historical single-adapter stage transition:

```text
immutable Rec FDR V4.3 full SFT
  + immutable GRPO-1 checkpoint-500 adapter
  -> load that adapter as trainable
  -> fresh GRPO-2 optimizer, scheduler, trainer state, and RNG
  -> continue updating the same adapter on Think-only GRPO-2
```

The effective model is always `W_SFT_FULL + DeltaW_adapter`. GRPO-2 does not
merge GRPO-1 into dense weights, create a fresh LoRA, or stack a second LoRA.
Consequently, every GRPO-2 checkpoint contains the complete cumulative
GRPO-1 plus GRPO-2 adapter effect and must be loaded directly on the immutable
full SFT model. Loading the GRPO-1 adapter beside it would double count GRPO-1.

The transition from GRPO-1 checkpoint-500 to GRPO-2 step 0 is adapter weight
initialization, not trainer resume. The source optimizer, scheduler, trainer
state, and rank RNG files are ignored. Resume is supported only from a
checkpoint created inside this GRPO-2 stage.

## Validation gates

The launcher validates the exact full-SFT and GRPO-1 adapter hashes and the
GRPO-1 lineage before loading any model. Every rank then checks all inherited
LoRA tensor names, shapes, dtypes, and values byte-for-byte before optimizer
construction. It requires zero trainable base parameters, 87,293,952 trainable
LoRA parameters across 504 tensors, and optimizer parameter IDs exactly equal
to those inherited trainable tensors.

Two independent five-step runs must produce byte-identical rollout, reward,
gradient, optimizer, RNG, probe, and final adapter evidence. Only then does a
fresh 20-step pilot run from the original GRPO-1 adapter and save complete,
adapter-only, resumable checkpoints at steps 10 and 20. Fixed Think and NoThink
retention runs at step 0 and step 20 with explicit numerators and denominators.

The older dense-merge experiment and its failed artifacts remain audit
evidence. Its precision issue is classified as `OPEN_FINAL_EXPORT_ISSUE`; it
does not block this training path. This package has no formal 300-step or
GRPO-3 launcher.
