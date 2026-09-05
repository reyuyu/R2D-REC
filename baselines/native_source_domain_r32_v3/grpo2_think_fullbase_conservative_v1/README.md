# GRPO-2 Think-only full-parent conservative validation

This package validates and runs the historical recommendation Think-only second stage.
Its training math is imported from
`gr_rec_think_sample8_fullsid_positive_a0_v3`: one global G4 CoT group, four
independent G8 FullSID groups, population normalization inside those groups,
and `L_cot + L_sid`. The only historical reward change remains A-only
`0.5 -> 0`.

The current parent is deliberately noncanonical. It is an isolated full-model
export of the immutable Rec FDR V4.3 SFT model plus GRPO-1 checkpoint-300.
Every manifest and checkpoint records:

- `test_parent_only=true`
- `canonical_grpo1_parent=false`

The runner rejects this parent unless `--allow-test-parent` is supplied. A
future formal runner must keep that default rejection and replace the parent
with an explicitly approved canonical GRPO-1 checkpoint.

## Validation sequence

1. Export the standalone test parent with `scripts/export_test_parent.py`.
2. Run CPU contracts.
3. Run independent five-step smoke A and B from the same parent and seeds.
4. Require byte-exact rollout, reward, advantage, loss, gradient, optimizer,
   RNG, adapter, and fixed Probe4/Beam32 evidence.
5. Only after `BYTE_EXACT`, run a fresh 20-step pilot at `2e-7`.
6. Verify adapter-only checkpoints 10 and 20, complete resume state, immutable
   parent fingerprints, and Think/NoThink retention probes.
7. Stop at `READY_FOR_GRPO2_PARENT_FINALIZATION`.

## Formal checkpoint-500 parent run

`config/formal_300.json` freezes the validated pilot contract while selecting
GRPO-1 checkpoint-500 as a user-provisional parent. The selection is explicitly
not an externally confirmed best checkpoint. `scripts/run_formal_300.sh` exports
that adapter into a standalone parent, runs merge/reload and fresh-LoRA Step0
gates, then starts a fresh Think-only LoRA at `2e-7` for 300 optimizer steps.

The exact adapter-only checkpoint schedule is 100, 150, 200, 250, and 300. No
inline retention generation is run, so post-training probes cannot perturb the
training RNG trajectory. The formal launcher never starts GRPO-3, evaluation,
publication merging, or model upload.
