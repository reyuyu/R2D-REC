# GRPO-2 Think-only continued single-adapter training

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

The frozen GRPO-1 adapter stores all 504 tensors as FP32 while the full SFT
base runs in BF16. The loader keeps PEFT training autocast enabled so LoRA
parameters are promoted to FP32 before checkpoint assignment; disabling that
behavior would irreversibly round the inherited weights during Step0 loading.

Two independent five-step runs must produce byte-identical rollout, reward,
gradient, optimizer, RNG, probe, and final adapter evidence. Only then does a
fresh 20-step pilot run from the original GRPO-1 adapter and save complete,
adapter-only, resumable checkpoints at steps 10 and 20. Fixed Think and NoThink
retention runs at step 0 and step 20 with explicit numerators and denominators.

The older dense-merge experiment and its failed artifacts remain audit
evidence. Its precision issue is classified as `OPEN_FINAL_EXPORT_ISSUE`; it
does not block this training path.

## Formal 300-step run

`scripts/run_formal_300.sh` starts directly from the immutable Full SFT plus
the original GRPO-1 checkpoint-500 adapter, creates fresh optimizer,
scheduler, trainer, and RNG state at GRPO-2 step 0, and updates that same
adapter for 300 optimizer steps. It does not repeat Smoke or Pilot.

Formal checkpoints are saved exactly at steps 100, 150, 200, 250, and 300.
Each is a combined continued adapter that already contains both GRPO-1 and
GRPO-2 effects. Inference loads `Full SFT + one GRPO-2 checkpoint`; stacking
the GRPO-1 adapter again is incorrect. Every checkpoint includes complete
optimizer, scheduler, trainer, rank RNG, lineage, and manifest state for a
true GRPO-2 resume.

No retention generation runs inside the continuous 0-to-300 trajectory. Once
training exits, the launcher sequentially evaluates the original GRPO-1
adapter and all five GRPO-2 adapters on the same fixed Think/NoThink probe and
writes comparison JSON and Markdown without selecting a best checkpoint. The
launcher does not merge weights, start GRPO-3, run external evaluation, or
upload a model.
