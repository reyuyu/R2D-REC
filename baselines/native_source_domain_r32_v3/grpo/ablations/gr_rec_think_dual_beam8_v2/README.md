# GR_REC_ThinkDualBeam8_v2

Formal-ready, isolated recommendation GRPO experiment. Formal training is not
authorized by this implementation commit.

Each business group produces one global stochastic G4 of CoTs (one per GPU).
Each CoT closes at its first `</think>` and receives exactly one deterministic
fixed-target-domain Beam8 ABC3 expansion.

- CoT branch: `think_reward` over each Beam8, population-normalized G4, with
  loss restricted to sampled CoT tokens.
- SID branch: `q_reward` for each selected ABC3 SID and four independent G8
  normalizations. This is Beam-selected PPO-style / group-relative optimization,
  not strict on-policy GRPO. Only the three ABC tokens are actions.
- Total loss: `L_cot + L_sid`, weights frozen at 1.0.
- `num_iterations=2` reuses exact CoTs, beams, rewards, advantages and detached
  no-grad full-forward old log-probabilities. There is no zero-std reroll.

Frozen topology: parent `checkpoint-1500` SHA256
`a5e92db011662799e07b4e1f16a2779afbbab9d66efcb481a5f2e199c75d3436`;
1549 raw groups; exact old Probe4 held out; 1545 train groups and rollouts; 3090
policy steps. Training uses Beam8 ABC3 while Probe4 retains production Beam32.
Fresh AdamW uses lr `1e-6`, weight decay 0, constant scheduler, beta 0, epsilon
0.2. Base is frozen and LoRA is trainable. Final step 3090 is explicitly saved.

Entrypoints: `run_dual_beam8_train.py`, `launch_dual_beam8_train.sh`,
`gpu_preflight.py`, and `test_dual_beam8_contract.py`. Do not use the formal
launcher until the GPU preflight is reviewed and training is separately authorized.

## 4-GPU zero-update preflight (2026-08-28)

Final result: PASS. The run proved one-rank-per-CoT G4 alignment, one Beam8 call
per rank, exact 4x8 ABC3 topology, four independent G8 calculations (never G32),
exact iteration-2 cache reuse, detached full-forward old log-probabilities,
isolated CoT/SID action masks, finite combined LoRA gradients, frozen base,
unchanged LoRA checksum, exact Probe4 Beam32 execution, and Probe RNG restore.

The two inspected real groups both produced all-zero `think_reward` and all-zero
SID G8 rewards. Their production advantages and gradients were correctly zero;
there was no reroll. The final boundary-only backward therefore used the same
immutable real actions and old log-probabilities with explicitly diagnostic
zero-mean advantages. This did not alter production reward, advantage, trainer,
or parameters. It establishes the mask/gradient path, while the observed Beam8
signal sparsity remains a formal-training risk that should be reviewed.

No optimizer or scheduler step occurred, no parameter changed, no checkpoint
was saved, and neither training nor formal training was started.
