# Step900 Bridge-to-Bare SID-family KD V1

## Motivation

Step900 is the peak Boundary CE checkpoint, but Historical Fixed-CoT still has a
0.393229 Bridge-to-Bare BeamRaw gap. Controlled crossover localized the remaining
problem to decoder ranking rather than clear self-CoT regression. This experiment
therefore distills only the A/B/C SID-family distributions from the exact original
BATA bridge context into the same bridge-free context.

## Contract

- Teacher and student both initialize as BASE + Step900.
- Teacher is frozen; student continues the same LoRA with a new AdamW optimizer.
- Gold CE supervises exactly A/B/C. KD is family-local KL at the three Gold-forced
  prefix positions; `lambda=0.30`, `T=1`, `lr=1e-7`.
- The 15943-group, 42799-path expanded dataset retains `1/K` row weights and uses
  `mean(weighted_path_loss) * 42799/15943`.
- Four GPUs, one row per rank, no accumulation, 300 optimizer steps.

## Preflight

Twelve CPU regressions passed, including exact lambda-zero forward/gradient parity
and K=1/2/4 group-gradient parity. Exact bridge extraction found 15943 groups and
four byte-distinct domain bridges; all duplicate source rows agreed byte-for-byte.

On 256 balanced training groups, Step900 Bridge Gold NLL was 4.155791 versus Bare
4.396268. Family KL was A=0.261728, B=0.274186, C=0.483119. The four-GPU zero-update
check had exact student/teacher initialization logits, no teacher/base gradients,
finite nonzero student LoRA gradients, and no parameter change.

## Result

| Decoder | Historical Bare | Bridge | Gap | HistoryNotGold | Bare Gold NLL | G900 Raw |
|---|---:|---:|---:|---:|---:|---:|
| Step900 | 3.533854 | 3.927083 | 0.393229 | 0.341797 | 4.245280 | 3.255208 |
| KD50 | 3.316406 | 3.955729 | 0.639323 | 0.342448 | 4.241023 | 3.220052 |
| KD100 | 3.423177 | 3.878906 | 0.455729 | 0.341797 | 4.239161 | 3.286458 |
| KD200 | 3.470052 | 3.959635 | 0.489583 | 0.345703 | 4.234593 | 3.269531 |
| KD300 | 3.462240 | 4.026042 | 0.563802 | 0.332031 | 4.242828 | 3.255208 |

All Beam evaluations had zero invalid outputs. KD200 is the highest-Bare KD
checkpoint, while KD100 has the strongest G900 result. No KD checkpoint improves
Step900 BareRaw or reduces its Bridge gap, so Step900 remains the deployment
recommendation. V1 provides weak evidence that the family target transfers to
self-CoT contexts, but it does not solve the residual interface gap.

The complete training metrics, preflight evidence, checkpoint paths, and evaluation
objects are in `../results/boundary_adapt_step900_bridge_to_bare_kd_v1_20260824.json`.
