# GR_REC_NoThinkOnly_Frontier_v1

## Status

- Parent code point: GitHub main `a46375e33770b15baedd0b9375f19495da1e4b36`
- Parent experiment: `GR_REC_NoThinkOnly_Hier_v1`
- Branch: `ablation/gr-rec-nothink-only-frontier-v1`
- Isolated developer-machine worktree: `/data/GRPO-frontier-v1`
- Completed phases: CPU implementation, historical forensic, paired zero-update
  GPU audit, and bounded Smoke24
- GPU used: yes, only for the authorized zero-update audit and Smoke24
- Full Frontier training started: no
- External benchmark started: no

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

## Historical forensic (2026-08-22)

Phase A replayed the exact Frontier implementation over immutable NoThink traces
from the completed joint Clamp/Bridge run and the available recent trace sample
from the stopped Hier-only run. It analyzed 652 real G8 groups and 5,216 real
candidates without loading a model or modifying either run directory.

The available trace population contains 95 format violations (1.8213%): 37
`nonempty_think`, 51 `unexpected_prose_before_sid`, 6 `malformed_template`, and
1 `invalid_sid`. Candidate-level Frontier exposure is Domain 153/5,216
(2.9333%), A 3,405/5,216 (65.2799%), B 1,217/5,216 (23.3321%), and C 191/5,216
(3.6618%). Group-level exposure is Domain 66/652 (10.1227%), A 612/652
(93.8650%), B 399/652 (61.1963%), and C 110/652 (16.8712%).

C Frontier is not the dominant population-wide event, but two real groups have
7/8 C-frontier candidates. Both are fixed in the paired-audit selection so the
future zero-update audit can decompose C-only gradients explicitly. The result
also fixes format-violation, wrong-domain-heavy, A-heavy, B-heavy, mixed,
exact-containing, and dead-zero/bridge cases, for ten real audit groups total.

Structured result:
`results/gr_rec_nothink_frontier_v1_historical_forensic_20260822.json`

Coverage limitation: the joint run retained 620 complete NoThink G8 traces;
the stopped Hier run retained only its configured recent 32-G8 trace sample.
These statistics describe all available immutable candidate traces, not all
1,297 optimizer steps of the stopped Hier run.

## Paired zero-update gradient audit (2026-08-22)

Phase B used the fixed ten real G8 groups and two policy states: Fresh original
BATA and Hier v1 `checkpoint-1250`. The checkpoint was used only as a late-policy
gradient stress diagnostic; Frontier training initialization remained Fresh BATA.
For every `policy_state x G8`, generation occurred exactly once, and the saved
completion IDs, masks and old log probabilities were reused for OLD_HIER and
FRONTIER_V1 backward passes.

All 20 policy/group comparisons completed. Fresh BATA OLD_HIER gradient norms
ranged `0.0..2.2520`, Frontier norms `0.1172..4.2753`; finite nonzero ratios had
median `3.0686`, and cosine median `0.5056`. Hier-1250 OLD_HIER norms ranged
`0.0..2.6847`, Frontier norms `0.1881..6.2655`; finite nonzero ratios had median
`2.3338`, and cosine median `0.9496`.

The late policy produced an 8/8 C-frontier stress group. OLD_HIER was zero while
Frontier norm was `4.9874`; C-only norm was `4.8476`, or `97.20%` of the full
Frontier norm. This confirms C can dominate a uniform late-policy C failure, but
the gradient remained finite, so it is a soft risk rather than a hard failure.
The observed real format violation used only the fixed whole-sequence total
`-0.09375`; format-only norm was `0.03504` (`11.02%` of full), and D/A/B/C were
fully gated. Its completion came from the saved paired rollout and was not
regenerated for the decomposition.

LoRA and base checksums were unchanged for both policies, base gradients were
absent, optimizer steps were exactly zero, and there were no alignment,
format/hierarchy overlap, non-finite, OOM or runtime failures.

Structured result:
`results/gr_rec_nothink_frontier_v1_paired_gradient_audit_20260822.json`

## Bounded Smoke24 (2026-08-22)

Run `GR-REC-NOTHINK-ONLY-FRONTIER-V1-SMOKE24-20260822` started from Fresh
original BATA on four A800 GPUs and stopped automatically at 24/24 optimizer
steps (12 fresh stochastic G8 rollouts, 24 real groups, 192 candidates). It used
the frozen contract: temperature/top-p `1/1`, LR `1e-6`, beta `0`, epsilon `.2`,
GRPO loss, two policy iterations, route multiplier `.5`, and Gold-A bridge
lambda `.02`.

Loss ranged `0.01942..0.08235` (mean `0.03390`); gradient norm
`0.2471..1.6692` (mean `0.4843`); approximate KL `0..0.001444`; clip fraction
`0..0.02941`. No sampled candidate violated strict format. Frontier candidate
counts were Domain `65`, A `96`, B `28`, C `2`. One real group activated the
Gold-A bridge; weighted bridge loss peaked at `0.10146`.

The two rollouts containing a C-frontier candidate had maximum gradient norms
with mean `1.0962`, versus `0.3717` across ten rollouts without C-frontier. The
largest gradient (`1.6692`) occurred with one C-frontier candidate. This is a
small-sample association (`n=2`), not evidence of instability or causality.

The checkpoint-24 adapter differs from Fresh BATA (`LoRA L2 delta=0.08176`, max
absolute delta `2.52e-5`), while base delta is zero by the frozen-base,
LoRA-only optimizer and adapter-only checkpoint evidence. There were no NaN,
Inf, OOM, non-finite metrics or runtime exceptions in the accepted run.

An initial step-0 launch failed before rollout generation because the command
omitted the machine's standard loopback NCCL/GLOO environment. That untouched
engineering failure is archived separately and excluded from Smoke metrics; the
retry used the established communication environment without changing any
training or algorithm parameter.

Structured result:
`results/gr_rec_nothink_frontier_v1_smoke24_20260822.json`
