# GR_REC_NoThinkOnly_Frontier_v1

## Status

- Parent code point: GitHub main `a46375e33770b15baedd0b9375f19495da1e4b36`
- Parent experiment: `GR_REC_NoThinkOnly_Hier_v1`
- Branch: `ablation/gr-rec-nothink-only-frontier-v1`
- Isolated developer-machine worktree: `/data/GRPO-frontier-v1`
- This phase: CPU implementation, regression tests, and static audit only
- GPU used: no
- Training started: no

## Research change

This experiment still trains only the NoThink route. Relative to the parent it
changes exactly two behaviors:

1. A strict NoThink format/validity gate runs before SID hierarchy credit.
2. The first failed D/A/B/C milestone receives an absolute Frontier penalty.

It does not change G8 stochastic on-policy sampling, temperature, reward ladder
coefficients, PPO ratio/clipping/old-logp behavior, credited-token SUM reduction,
learning rate `1e-6`, NoThink route multiplier `0.5`, Gold-A bridge lambda
`0.02`, the dataset, or any Think algorithm.

## Strict NoThink format

A completion is valid in either of these forms:

- optional `<think>` containing whitespace only, then the exact domain-specific
  SFT declaration and one complete final SID;
- optional empty `<think>`, whitespace, and one complete SID with no prose
  (`direct_sid_fallback`).

The accepted declarations are:

- video: `该用户最近喜欢的视频有:`
- prod: `该用户最近点击了商品:`
- ad: `该用户最近感兴趣的广告有:`
- living: `该用户最近首次打赏了主播:`

Nonempty think content, unexpected prose, invalid SID, or malformed templates
are violations. A violation forces scalar reward `-1`, gates D/A/B/C, and
receives only a whole-sequence advantage with total mass `-0.09375`. For `L`
generated non-padding tokens, every token receives `-0.09375 / L`.

## Frontier credit

For format-valid candidates, positive milestone credits retain the parent rule:

`increment(stage) * (1 - full_G8_mean(stage)) / 8`

Only the first failed milestone receives an absolute negative credit:

| First failure | Token credit |
| --- | ---: |
| Domain | -0.03125 |
| A | -0.0625 |
| B | -0.1875 |
| C | -0.75 |

All descendants are gated. Exact candidates have no Frontier penalty. Uniform
exact `[8]*8` remains a legitimate hierarchy zero-signal group.

## Bridge interaction

The existing Gold-A dead-zero bridge is unchanged. In a true `[0]*8` group the
RL path supplies the A Frontier negative while the independent teacher bridge
remains active at lambda `0.02`.

## Monitor schema

The Frontier rank stream distinguishes:

- `format_sequence_penalty`: whole-sequence, fixed total;
- `positive_success`: sparse milestone-token positive credit;
- `frontier_negative`: sparse first-error-token negative credit;
- `gated`: no credit;
- teacher bridge fields remain in the independent bridge stream.

It records format valid/violation counts and reasons, per-stage Frontier counts
and activity, per-stage positive activity, singleton B/C success, bridge state,
and the revised group taxonomy:

- `ALL_WRONG_DOMAIN_FRONTIER`
- `UNIFORM_A_FAILURE_FRONTIER`
- `UNIFORM_B_FAILURE_FRONTIER`
- `UNIFORM_C_FAILURE_FRONTIER`
- `UNIFORM_EXACT_ZERO_SIGNAL`

## CPU acceptance

The regression suite covers the 13 specified archetypes and format cases,
length invariance at 10/100/500 tokens, strict prefix gating, `[0]*8` bridge
activation, route guards, frozen runner parameters, parent/base/adapter
contracts, and manifest semantics. No model is loaded and no training entry
point is invoked during these tests.
