# Changelog

## v3

- Created an isolated sibling from reference commit
  `32c22f3de8e90e226bf3671f64ca9fbfbed90e8d`.
- Replaced deterministic fixed-domain Beam8 ABC3 with stochastic Sample8
  complete SID generation.
- Removed the fixed domain prefix from training SID context.
- Changed SID action scope from three ABC tokens to four domain+A+B+C tokens.
- Replaced `think_reward(Beam8)` with the plain sum of eight sampled SID
  `q_reward` values for the CoT-level G4 reward.
- Preserved independent per-CoT G8 normalization, two-pass cache reuse,
  full-forward detached old logps, Probe4 Beam32, parent guards and all PPO
  configuration.
