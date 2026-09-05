# GRPO-2 Think-only full-parent conservative validation

This package validates the historical recommendation Think-only second stage
without starting formal training. Its training math is imported from
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

This package does not start formal GRPO-2, GRPO-3, evaluation, merging for
publication, or model upload.
