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

## GPU validation and formal launch (2026-08-28)

The corrected continuation-scanning implementation is commit
`bcff386ec81104251e93a483785d96ece042f0b2`.

Four-GPU zero-update preflight passed all gates. The immutable real rollout
produced CoT rewards `[-1.25, -1.25, -2.0, 0.0]`; every sampled continuation
contained one complete SID after an 8-11 token natural-language prefix. The
four independent SID G8 reward vectors had real between-candidate signal, the
CoT and SID action gradients were non-zero, Base parameters stayed frozen,
and the trainable checksum was identical before and after the audit.

Evidence:

- `results/gr_rec_think_sample8_fullsid_v3_gpu_preflight_scan_20260828.json`
- `PREFLIGHT_PASS=true`
- optimizer/scheduler steps: 0/0
- parameter update: false
- Probe4 Beam32: completed with RNG and parameters unchanged

Formal run:

- run id: `GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828`
- launch commit: `bcff386ec81104251e93a483785d96ece042f0b2`
- topology: 1545 fresh rollouts, 3090 optimizer steps
- checkpoints: every 250 steps plus final step 3090
- monitor stream: `sample8_fullsid.jsonl`
- non-zero and positive training samples are incrementally archived under
  `training_sample_exports/` in the run directory.
