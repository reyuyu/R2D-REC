# GR_REC_NoThinkOnly_Frontier_v1

Independent NoThink-only ablation derived from `GR_REC_NoThinkOnly_Hier_v1`.

It changes only two training semantics:

1. strict NoThink format/validity enforcement, with invalid completions forced to
   scalar reward `-1` and a length-normalized whole-sequence advantage totaling
   `-0.09375`;
2. absolute, prefix-gated first-error Frontier penalties for Domain/A/B/C while
   retaining the full-G8 relative positive milestone credits.

G8 stochastic sampling, the reward ladder, PPO loss, credited-token SUM
reduction, NoThink route multiplier `0.5`, learning rate `1e-6`, and the
Gold-A dead-zero bridge at lambda `0.02` remain frozen.
