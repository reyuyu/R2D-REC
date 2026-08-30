# V3-Official GRPO

This ablation keeps V3's Think G4 training contract and replaces only its
answer rollout with a fixed-domain stochastic Official Sample8 ABC3 branch.

## Frozen contract

- Parent: Beta baseline `checkpoint-1106`, adapter SHA256
  `4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3`.
- Data: V3 source `rec_mp_grpo_v2/train.jsonl`, fixed Probe4 excluded,
  1,545 Think-only business groups across all four domains.
- Rollout: global G4 CoT, then four independent fixed-domain Sample8 ABC3
  groups. There is no G32 normalization.
- Reward: unmodified hierarchical `q_reward` (`-1/-0.25/0/0.5/2/8`).
  Each G8 is normalized independently and its scalar advantage is broadcast
  over the three ABC action tokens. CoT reward is the arithmetic sum of that
  CoT's eight raw SID rewards and is normalized across G4.
- Loss: `L_cot + L_sid`, with two policy iterations reusing the accepted
  rollout, rewards, advantages, masks, and unchanged-policy old log-probs.
- Probe: the original fixed Probe4 with production Official Beam32 scoring.
- Optimizer: fresh AdamW, learning rate `1e-6`, zero weight decay, constant
  scheduler. Resume from a GRPO checkpoint is rejected.

The launcher saves every 50 optimizer steps and preserves at least 64
checkpoints. It is intentionally not run as part of CPU acceptance.
