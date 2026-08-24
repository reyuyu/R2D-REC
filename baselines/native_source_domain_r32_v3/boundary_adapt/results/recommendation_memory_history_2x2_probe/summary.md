# Recommendation Root-Cause Phase 1.5.1

## Scope and audit status

This is a bounded 16-group, 2x2 mechanism probe. It is not a natural-distribution sample and is not an official-score simulation. No training, Self-CoT generation, or external evaluation was run.

- GPU cases: 64 (16 groups x 2 models x 2 decode conditions)
- Beams audited: 2048
- Cells: SH, SN, UH, UN; 4 groups per cell
- Domain composition: each cell contains one video, prod, ad, and living group
- Decode conditions: BARE and exact original bridge
- Models share the same base and LoRA shape
- Previous Phase 1.5 Quick result hashes were checked and remained unchanged

Implementation commits:

- GPU path used for records: `63c6c2563db69a9bdb6dbb062e698329d6dff3f2`
- Final audited implementation: `d3a7ba966d20fa112825699ef0f8f5d65f280c7e`
- The latter only changes CPU finalization so invalid beams are audited and retained; it does not change the GPU inference path.

## Model inventory

| Model | Adapter path | Adapter SHA256 |
|---|---|---|
| MiniFix | `/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/MINI-FIX-R32-2E-GC04-4GPU-20260816-092741/checkpoint-138` | `8d61c7d61f1413f3e449bb1aebeca5ee060bd31080959b02cbaf56a8788fc72d` |
| Gamma | `/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/mini_gamma/checkpoint-136` | `772921211cbcd238728cf538fe904042697c60e6034dd0a363081c306e3f48aa` |

`BASE_IDENTICAL=YES`; LoRA config parity passed (`r=32`, `alpha=64`, dropout `0.05`, same seven projection targets).

## Main results

| Model | Cell | Bare MRR | Bridge MRR | Gain | Bare Hit@32 | Bridge Hit@32 | History fraction delta | 1-Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MiniFix | SH | 0.42500 | 0.51667 | +0.09167 | 0.75 | 0.75 | -0.32031 | 0.65896 |
| MiniFix | SN | 0.00000 | 0.00000 | +0.00000 | 0.00 | 0.00 | -0.18750 | 0.70010 |
| MiniFix | UH | 0.08333 | 0.12500 | +0.04167 | 0.25 | 0.25 | -0.44531 | 0.75075 |
| MiniFix | UN | 0.00000 | 0.00000 | +0.00000 | 0.00 | 0.00 | -0.35156 | 0.79923 |
| Gamma | SH | 0.55000 | 0.52273 | -0.02727 | 0.75 | 0.75 | -0.01562 | 0.46361 |
| Gamma | SN | 0.00000 | 0.00000 | +0.00000 | 0.00 | 0.00 | +0.03906 | 0.58384 |
| Gamma | UH | 0.08333 | 0.12500 | +0.04167 | 0.25 | 0.25 | +0.00000 | 0.56837 |
| Gamma | UN | 0.00000 | 0.00000 | +0.00000 | 0.00 | 0.00 | -0.05469 | 0.69400 |

Cell notation:

- SH: MiniFix train-seen and Gold SID appears in history
- SN: MiniFix train-seen and Gold SID does not appear in history
- UH: MiniFix train-unseen and Gold SID appears in history
- UN: MiniFix train-unseen and Gold SID does not appear in history

## Contrasts

| Contrast | MiniFix | Gamma | Interpretation |
|---|---:|---:|---|
| Train-memory: SN gain - UN gain | 0.00000 | 0.00000 | No detectable group-memory advantage when Gold is absent from history |
| History shortcut: UH gain - UN gain | +0.04167 | +0.04167 | Same weak positive direction in both models |
| MiniFix - Gamma specialization, memory | 0.00000 | - | No MiniFix-specific effect |
| MiniFix - Gamma specialization, history | 0.00000 | - | No MiniFix-specific effect |

The bootstrap 95% interval for train-memory contrast is `[0, 0]` in both models. The history-shortcut contrast is `0.04167`, with descriptive interval `[0, 0.125]` in both models. With only four groups per artificial cell, these intervals are descriptive and are not formal significance evidence.

The UH MRR increase does not increase Hit@32: both models remain at `0.25` before and after the bridge. It is a ranking change within the one already-hit group, not acquisition of additional Gold candidates.

## Mechanism conclusion

1. **Train-group memory is weak or unsupported.** SN and UN both have zero Gold MRR and zero Gold Hit@32 for both models. Knowing that a group appeared in MiniFix training did not produce a measurable advantage once its Gold SID was absent from history.
2. **History repeat provides the clearest floor.** UH is the only non-SH cell with non-zero Gold performance. Its identical MiniFix/Gamma result points to a shared history-conditioned capability rather than MiniFix-specific memorization.
3. **Old Bridge behaves more like a generic context/ranking conditioner.** It substantially changes beam sets (`UN` 1-Jaccard `0.79923` for MiniFix and `0.69400` for Gamma), but produces no UN Gold gain. It often reduces, rather than increases, the fraction of history candidates.
4. **Novel recommendation ability remains weak in this local proxy.** Both models score zero in UN for Bare and Bridge despite large response redistribution.

Automated classification:

- `GROUP_MEMORY_SUPPORT=WEAK_OR_UNSUPPORTED`
- `HISTORY_REPEAT_FLOOR_SUPPORT=STRONG`
- `NOVEL_RECOMMENDATION_ABILITY=WEAK_IN_LOCAL_PROXY`
- `OLD_BRIDGE_TRAINED_SPECIALIZATION_SUPPORT=WEAK_OR_UNSUPPORTED`
- `PRIMARY_STORY=STORY_3_GENERIC_CONTEXT_ONLY`

## Invalid beam audit

`INVALID_BEAMS=1`, so the requested strict `INVALID_BEAMS=0` gate did **not** pass. This beam was not silently filtered or replaced:

- Model/group: MiniFix / `0a05e319c879c5da64d1da03a87e59777fdb42e9288e7cc9f7e3a1a38d82d7ee`
- Cell/condition/domain: UN / BARE / video
- Beam index: 0 of 32
- Raw token IDs: `[198, 75882, 20002]`
- Decoded text: `\n该用户`
- Reason: the strict ABC3 parser rejected the retained three-token beam

The invalid rate is `1/2048 = 0.0488%`. It is retained in `invalid_beam_audit.json`; no outcome-based resampling was performed and the 64-case cap was not exceeded.

## Limitations

- Each cell has only four groups.
- The probe is deliberately balanced, not representative of the held-out distribution.
- K is confounded with cell: SH and SN are all `K>=2`; UH contains one `K>=2` and three `K=1`; UN is all `K=1`.
- “Unseen” means unseen by MiniFix training, not globally novel.
- One strict beam parse failure means the zero-invalid contract failed.
- This does not estimate official recommendation score or causal effects at population scale.

## Provenance and stop condition

- Seen CoT, bridge, history, domain, Gold set, and answer were checked byte-for-byte against actual MiniFix training rows: PASS.
- Prompt token/history SID audit across 8 representative groups: PASS.
- Training started: NO
- Optimizer steps: 0
- Self-CoT generation started: NO
- External evaluation started: NO
- Next experiment started: NO

Final action: stop and review; no follow-on experiment was launched.
