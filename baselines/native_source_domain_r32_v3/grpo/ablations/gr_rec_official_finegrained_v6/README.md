# GR_REC_OfficialFineGrained_v6A

Independent all-domain V6-A sibling using the original V3 Think data and the
V3 checkpoint-1500 parent. Rollout topology remains `G4 CoT + 4 independent
Official G8`, with fixed target-domain prefixes and stochastic ABC3 Sample8.

CoT reward is the untouched sum of eight production q_reward values. SID
training uses three independently normalized G8 populations: cumulative A, B,
and C hits. A is always active, B is gated by hit_A, and C is gated by hit_B.
The resulting `[8,3]` advantages and masks are frozen with old logp and reused
for both policy iterations. History-copy status is diagnostic only.

Production Official Beam32 scoring remains unchanged. The primary Probe8 has
two groups per domain; the original held-out Probe4 remains excluded from the
1545-group training set, while the additional diagnostic group per domain is
explicitly marked as training-overlap. A separate History-Copy Probe4 retains
the dedicated copy-anatomy visualization.

This implementation stage is CPU-only and does not launch preflight or formal
training.
