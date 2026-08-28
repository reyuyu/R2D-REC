# GR_REC_ThinkSample8_FullSID_v3

Independent ablation derived from the validated
`GR_REC_ThinkDualBeam8_v2` code point
`32c22f3de8e90e226bf3671f64ca9fbfbed90e8d`.

## Frozen topology

- One business group produces one global stochastic G4 of closed CoTs.
- Each CoT context is exactly `prompt + sampled CoT through </think>`.
- No fixed target-domain prefix and no natural-language bridge are inserted.
- Each CoT independently samples eight answer continuations with
  `do_sample=true, temperature=1, top_p=1, top_k=0,
  repetition_penalty=1, max_new_tokens=128`.
- Each continuation is scanned for consecutive raw-ID
  `domain_begin + A + B + C` spans. The first complete SID is scored; later
  complete SIDs are retained and flagged for monitoring. A continuation with
  no complete SID receives `q_reward=-1`.
- The four SID groups are normalized as four independent G8 groups, never G32.
- Each CoT reward is the arithmetic sum of its eight sampled SID rewards.
- The four CoT sums are population-normalized as one global G4.
- CoT loss updates sampled CoT action tokens only. SID loss updates only the
  four tokens of the first complete SID; prose and later SIDs are masked out.
- `L_total = L_cot + L_sid`, `num_iterations=2`; the second policy pass
  reuses CoT, Sample8, rewards, advantages and detached full-forward old logps.

Parent, Probe4, optimizer, PPO epsilon, beta, learning rate, frozen Base and
LoRA-only training remain identical to v2. Probe4 continues to use the existing
production-shaped Beam32 evaluator.

Entrypoints:

- `run_sample8_fullsid_train.py`
- `launch_sample8_fullsid_train.sh`
- `gpu_preflight.py`
- `test_sample8_fullsid_contract.py`
