# Implementation contract

The production trainer is isolated in
`gr_rec_think_sample8_fullsid_v3`. Shared GRPO math is unchanged.

The only algorithmic delta from `GR_REC_ThinkDualBeam8_v2` is:

1. SID candidates are stochastic Sample8 complete SIDs without a fixed domain.
2. SID action width is four raw tokens.
3. A CoT reward is `sum(sid_rewards)`.

Hard gates are encoded in CPU tests and the four-GPU zero-update preflight:
G4 alignment, exact 4xG8 topology, strict FullSID4 parsing, no G32
normalization, no bridge/prefix, reward-sum parity, detached old-logp rescore,
iteration-2 cache reuse, exact Probe4, parent SHA, frozen Base, finite LoRA
gradients and zero parameter updates.
