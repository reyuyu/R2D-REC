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
