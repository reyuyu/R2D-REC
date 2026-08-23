# MC_USER Hybrid K4 External Evaluation

## Experiment

- Run: `MC-USER-HYBRID-K4-S1-512-20260822-223353`
- Parent: `GR_REC_v1 checkpoint-1500`
- Parent recorded external aggregate: `1.3510`
- Objective: `Sequence GRPO + 0.3 * marginal local loss`
- Schedule: 512 prompts, strict Action/Chain alternation, K=4 candidate parallel, four GPUs
- Learning rate: `1e-6`

The external score order is Material 4, User 2, Recommendation 4, World 1. Scores below were provided by the user. Group sums use the displayed rounded components, so they can differ from the evaluator aggregate by `0.0001`.

## Raw scores

| Model | Aggregate | Material (4) | User (2) | Recommendation (4) | World |
| --- | ---: | --- | --- | --- | ---: |
| Parent GR_REC Step 1500 | `1.3510` | `0.0510, 0.0366, 0.0515, 0.0425` | `0.1587, 0.0976` | `0.1381, 0.1598, 0.2156, 0.1665` | `0.2331` |
| MC Hybrid Step 256 | `1.3472` | `0.0515, 0.0376, 0.0527, 0.0424` | `0.1601, 0.0996` | `0.1381, 0.1632, 0.2030, 0.1647` | `0.2342` |
| MC Hybrid Step 512 | `1.3402` | `0.0522, 0.0368, 0.0518, 0.0422` | `0.1622, 0.1001` | `0.1325, 0.1564, 0.2016, 0.1683` | `0.2361` |

## Group sums and deltas

| Model | Material | User | Recommendation | World | Aggregate vs Parent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent GR_REC Step 1500 | `0.1816` | `0.2563` | `0.6800` | `0.2331` | - |
| MC Hybrid Step 256 | `0.1842` (`+0.0026`) | `0.2597` (`+0.0034`) | `0.6690` (`-0.0110`) | `0.2342` (`+0.0011`) | `-0.0038` |
| MC Hybrid Step 512 | `0.1830` (`+0.0014`) | `0.2623` (`+0.0060`) | `0.6588` (`-0.0212`) | `0.2361` (`+0.0030`) | `-0.0108` |

From Step 256 to Step 512, User gains another `+0.0026`, while Recommendation loses another `-0.0102` and aggregate loses `-0.0070`. The largest Recommendation regression relative to the Parent is metric R3: `-0.0126` at Step 256 and `-0.0140` at Step 512. At Step 512, R1 and R2 also turn negative; only R4 improves (`+0.0018`).

## Interpretation

### What the evidence supports

Step 256 aggregate (`-0.0038`) is within the repository's approximate `+/-0.01` single-evaluation variation band. Step 512 aggregate (`-0.0108`) is only slightly outside it, so aggregate alone is not decisive. The task-level direction is more informative: User rises at both checkpoints while Recommendation falls, and the Recommendation deficit grows with training. This is consistent with cross-task interference from optimizing only User behavior on top of a Recommendation-specialized Parent.

Training itself was healthy: 512/512 optimizer updates completed, valid candidate rate was `99.80%`, no-credit prompt rate was `0`, the base hash stayed unchanged, and all `504/504` trainable tensors were LoRA. The external tradeoff is therefore not explained by a stopped or numerically broken run.

### Is the learning rate too low?

The evidence does not support that diagnosis. With `1e-6`, both User metrics move monotonically upward and Recommendation moves monotonically downward across 256 and 512 steps. The optimizer updated on every prompt. The model is changing; it is changing in the direction selected by the User-only objective. Raising the learning rate alone does not preserve Recommendation and could make this interference happen faster.

### How to preserve Recommendation

There is no guarantee from a User-only objective. A preservation guarantee requires Recommendation to enter either optimization or checkpoint gating:

1. Keep `GR_REC_v1 checkpoint-1500` as the immutable reference and select checkpoints using paired User plus Recommendation probes, not User on-policy reward.
2. Add a small frozen Recommendation replay/retention loss or reference-policy KL on Recommendation prompts. This directly constrains the capability that must be preserved.
3. If Recommendation gradients cannot be included, use a trust-region style update budget and stop at the first checkpoint where the Recommendation guard crosses its allowed drop. Lower LR can slow drift but cannot change its direction.
4. Treat Step 256 as the better of the two evaluated Hybrid checkpoints. It buys `+0.0034` User for `-0.0110` Recommendation; Step 512 buys only another `+0.0026` User while losing another `-0.0102` Recommendation.

The Parent score `1.3510` is itself a single recorded high evaluation and was already marked for repetition. Therefore exact deltas should be confirmed by paired repeated evaluation before claiming a stable regression. The repeated direction across two MC checkpoints is still sufficient to reject the claim that later training is automatically better.
