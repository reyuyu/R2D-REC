# CODEX HANDOFF — GR_REC_ThinkDualBeam8_v2

Status: **DESIGN / REFERENCE SCAFFOLD ONLY. DO NOT START FORMAL TRAINING YET.**

This document is the implementation contract for Codex. The Python files currently
present in this directory are a reference scaffold produced while defining the
experiment. Codex should audit/refactor them as needed and must not treat them as
formal-training-approved until the required CPU tests and 4-GPU zero-update preflight
below pass.

## 0. Why this experiment replaces the previous suffix-only attempt

Previous experiment: `gr_rec_think_suffix_sid_v1`.

That version generated a full `CoT -> </think> -> answer suffix` rollout and applied
loss only after the first `</think>`. It did not directly optimize which CoT was useful.
The new experiment must separate two learning questions:

1. Which sampled CoT leads to better recommendation candidates?
2. Given one sampled CoT, which strict ABC3 continuation is better?

The new experiment therefore uses one shared Beam8 continuation set per CoT and two
independent advantage groups / loss masks.

Do **not** modify the old experiment in place. Implement this as the isolated ablation
`gr_rec_think_dual_beam8_v2`.

## 1. Immutable parent and baseline

Parent adapter:

`/root/data_checkpoints_backup_20260824/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500`

Expected adapter SHA256:

`a5e92db011662799e07b4e1f16a2779afbbab9d66efcb481a5f2e199c75d3436`

Recorded external score: `1.3510` (single recorded evaluation, not a confidence
interval).

Requirements:

- fail closed if parent files/SHA do not match;
- fresh optimizer/scheduler state;
- base model frozen; only LoRA trainable;
- do not resume optimizer/RNG from the parent checkpoint.

## 2. Keep the existing Probe4 holdout

The existing fixed recommendation Probe4 must be retained.

- exactly four held-out business groups;
- domain order/coverage remains `video / living / prod / ad` as required by the
  existing probe implementation;
- Probe groups must be excluded from training;
- do not silently reselect a new Probe set because the training route changed;
- if the prior run already records explicit Probe IDs, reuse/assert those exact IDs;
- otherwise use the existing deterministic Probe4 selector with the same seed and
  assert that the resulting IDs match the prior experiment before training.

Probe evaluation contract is **not changed** by this experiment:

- keep the existing production-shaped Probe4 evaluator;
- Think probe continues to use the existing Beam32 path;
- do not replace Probe Beam32 with training Beam8;
- run baseline probe at step 0 and interval probes at the existing schedule (default
  200 logical steps unless explicitly overridden).

Expected topology when the raw set contains 1549 business groups and Probe4 is held
out:

- train business groups: `1545`;
- one Think row per training business group;
- fresh G4 CoT rollouts: `1545`;
- with `num_iterations=2`: `3090` logical policy steps.

Codex must recompute/assert these values from the runtime dataset rather than trusting
this document blindly.

## 3. One business group -> G4 stochastic CoTs

For each training business group, generate exactly four stochastic CoTs globally.
On the intended 4-GPU shape this should be one CoT per rank.

Frozen CoT sampling:

- route: Think only;
- `G_cot = 4`;
- `temperature = 0.9`;
- `top_p = 0.95`;
- `do_sample = true`;
- use the existing Think generation/stopping behavior through the first `</think>`;
- maximum completion length remains the existing 2048-token safety bound;
- rank RNG must remain decorrelated (`seed + rank`, matching the baseline runner).

The global G4 must contain the same `recommendation_group_id` on all four ranks.
Fail closed if a G4 mixes business groups.

## 4. Each CoT -> one deterministic strict Beam8 ABC3 set

For each sampled CoT, create exactly one Beam8 continuation set. **The same eight
candidates are reused by both objectives below. Do not run two Beam8 searches.**

Continuation context:

`official prompt + sampled CoT through </think> + fixed target-domain begin token`

Then generate exactly the three SID tokens `A B C`.

Training Beam8 contract:

- `num_beams = 8`;
- `num_return_sequences = 8`;
- `do_sample = false`;
- `min_new_tokens = 3`;
- `max_new_tokens = 3`;
- fixed target-domain begin token comes from `target_domain`, never from Gold;
- strict raw-token ABC3 parser;
- domain begin token is context, not an action;
- each of the eight candidates must contain exactly three generated action tokens.

On 4 GPUs, the preferred implementation is rank-local:

- rank0 owns CoT0 + its Beam8;
- rank1 owns CoT1 + its Beam8;
- rank2 owns CoT2 + its Beam8;
- rank3 owns CoT3 + its Beam8.

This avoids the old Beam32 task scheduler for training. The held-out Probe4 still uses
the existing Beam32 evaluator.

### Missing `</think>`

Do not invent a new format reward in this ablation. Preserve the existing Think
behavior unless a separate audit explicitly changes it. Record `closed` / closure rate
and fail/stop the run if closure materially collapses. Any change to missing-closure
reward semantics is a separate experiment.

## 5. Objective A — CoT quality, global G4 advantage

For each CoT `i`, parse the eight Beam8 SIDs and compute the existing hierarchical
Think aggregate reward using the current `think_reward(beam_sids, gold_set)` logic.
Do not redesign its prefix de-duplication or geometric credit decay.

This yields four scalar rewards:

`R_cot = [R0, R1, R2, R3]`.

Gather all four rewards globally and compute **population** group normalization only
across the four CoTs:

`A_cot_i = (R_i - mean(R_cot)) / (population_std(R_cot) + 1e-4)`.

If the population std is zero, the four CoT advantages are zero. Do not add reroll in
v2 unless separately approved.

CoT PPO contract:

- old log-prob source: unchanged rollout policy, no-grad full-forward rescore;
- never use `generate.scores` as PPO old log-prob;
- current forward keeps autograd;
- PPO clipping epsilon remains `0.2`;
- `beta = 0`;
- token-level importance ratio;
- CoT loss applies only to the sampled CoT action tokens through the first complete
  `</think>` closure;
- Beam8 ABC tokens must not receive the CoT objective.

Important interpretation: this objective trains the probability of sampled CoTs
according to how useful their Beam8 recommendation set is.

## 6. Objective B — SID quality, four independent G8 advantages

For every CoT `i`, score each of its eight strict ABC3 candidates independently with
the existing NoThink-v1 hierarchical `q_reward`:

- invalid: `-1`;
- wrong domain: `-0.25`;
- correct domain only: `0`;
- A: `0.5`;
- AB: `2`;
- Exact: `8`.

For CoT `i`, this gives:

`R_sid_i = [r_i0, ..., r_i7]`.

Compute the SID advantage **inside that CoT only**:

`A_sid_ij = (r_ij - mean(R_sid_i)) / (population_std(R_sid_i) + 1e-4)`.

There are four separate G8 normalizations. Never normalize all 32 SID candidates as
one group and never mix candidates from different CoTs.

If one CoT's eight SID rewards have zero population std, only that CoT's SID branch
has zero advantage. No reroll is used in v2.

SID PPO-style contract:

- Beam8 candidates are deterministic Beam-selected candidates, not stochastic
  on-policy samples. Therefore this branch must be described as **Beam-selected
  PPO-style / group-relative optimization**, not strict on-policy GRPO;
- immediately after Beam8 generation, full-forward rescore the unchanged policy under
  `no_grad` to obtain old log-probs for the three ABC action tokens;
- current forward keeps autograd;
- PPO clipping epsilon remains `0.2`;
- loss mask is exactly the generated three ABC tokens;
- prompt, user history, sampled CoT, `</think>`, and fixed domain begin token are
  attention context only for this branch.

## 7. Shared rollout reuse across `num_iterations=2`

Keep `num_iterations=2` so the second policy iteration can produce a non-trivial PPO
ratio after the first update.

For one fresh business-group rollout, cache and reuse across both policy iterations:

- the same four sampled CoTs;
- the same four Beam8 candidate sets;
- the same CoT rewards and G4 advantages;
- the same four SID reward vectors and their four G8 advantages;
- the same old CoT log-probs;
- the same old SID ABC3 log-probs.

**Do not regenerate CoTs or rerun Beam8 on policy iteration 2.**

## 8. Combined loss

First implementation default:

`L_total = lambda_cot * L_cot + lambda_sid * L_sid`

with:

- `lambda_cot = 1.0`;
- `lambda_sid = 1.0`.

Expose the two lambdas in the manifest/config but do not tune them before the first
preflight/smoke. Keep branch losses separately normalized over their own action tokens
and candidates.

Fresh optimizer:

- AdamW;
- learning rate `1e-6`;
- weight decay `0`;
- constant scheduler / no LR decay;
- existing max-grad-norm unless an audit changes it.

Do not add a special optimizer skip rule for zero-advantage branches in this version.
Use standard optimizer semantics and monitor zero-advantage frequency. Any change to
optimizer-step semantics is a separate audited decision.

## 9. Required monitor additions

The current reference scaffold does not yet provide enough dual-objective monitoring.
Codex must add explicit fields so a run can be audited without reconstructing it from
raw text.

Per business group / rollout, record at minimum:

- business group id / target domain;
- four CoT raw texts/token IDs and closure flags;
- per-CoT Beam8 parsed SIDs;
- per-CoT Beam8 six-level SID rewards;
- four CoT aggregate rewards;
- global G4 CoT advantages;
- four independent G8 SID advantage vectors;
- CoT reward population std / zero-std flag;
- per-CoT SID reward population std / zero-std flags;
- `cot_loss`, `sid_loss`, `total_loss`;
- CoT action token count;
- SID action token count (must be 8*3 locally for one CoT); 
- old/current ratio mean, clip fraction, approximate KL separately for CoT and SID
  branches;
- Beam8 Exact/AB/A/invalid counts for each CoT;
- Beam8 wall time;
- Probe4 results remain in the existing probe stream.

For debugging/preflight, also expose branch-specific gradient norms.

## 10. Required CPU contract tests

Before any GPU run, tests must cover at least:

1. Probe4 IDs are excluded from training data.
2. Think-only training dataset contains one row per business group.
3. G4 sampler repeats one business group exactly four times globally.
4. Beam8 requires exactly eight candidates and each candidate is strict ABC3.
5. CoT reward uses existing `think_reward` over the eight Beam SIDs.
6. CoT population advantages match a hand-computed G4 example.
7. Each SID G8 population advantage matches hand computation.
8. Four SID groups are normalized independently; a test must fail if flattened to
   G32.
9. Zero population std returns zero advantages without reroll.
10. Total loss weighting defaults to 1:1.
11. `num_iterations=2` reuses cached rollout/Beam data.
12. Parent path and SHA are immutable/fail-closed.
13. Runtime import provenance rejects stale `/data/GRPO/scripts` modules.
14. Launcher keeps 4 GPUs and loopback NCCL control.
15. Probe evaluator remains Beam32 while training uses Beam8.

## 11. Required 4-GPU zero-update preflight

Formal training is blocked until a real GPU preflight passes all conditions below.
The preflight must not call `optimizer.step`, `scheduler.step`, save a checkpoint, or
start the formal training loop.

Required assertions:

### Topology

- world size = 4;
- one global business group per fresh rollout;
- exactly four CoTs globally;
- one local CoT per rank;
- exactly one Beam8 set per local CoT;
- exactly 8 strict ABC3 candidates per Beam8.

### Reward / advantage parity

- gather the four CoT aggregate rewards and compare online G4 advantages to a pure
  reference implementation;
- for every rank, compare its eight SID rewards and G8 advantages to a pure reference
  implementation;
- explicitly prove no G32 normalization.

### Old-policy semantics

- CoT old log-probs are full-forward no-grad rescored from the unchanged rollout
  policy;
- SID old log-probs are full-forward no-grad rescored for the three ABC tokens after
  the unchanged Beam8 rollout;
- no PPO denominator comes from generation scores;
- old-policy tensors are detached;
- no parameter version changes between rollout and old-logp rescore.

### Gradient boundary

Run branch-isolated backward checks:

- `L_cot` alone: gradient exists on CoT action log-probs; no SID-action loss term is
  present;
- `L_sid` alone: gradient exists on ABC3 action log-probs; no CoT-action log-prob loss
  term is present;
- fixed domain prefix has no action loss;
- padding has zero action gradient;
- combined loss produces finite LoRA gradients;
- base model has zero trainable params / zero grads;
- parameter checksums remain unchanged because this is zero-update preflight.

Note: SID loss is allowed to update shared LoRA parameters in formal training and can
therefore affect future CoT generation indirectly. "Loss only on SID actions" refers
to the action-logprob objective boundary, not parameter-level isolation.

### Probe integrity

- exactly four Probe groups;
- zero overlap with training groups;
- step-0 Probe4/Beam32 evaluation can run without changing model parameters;
- probe RNG is restored after evaluation.

## 12. Formal-run blockers / current scaffold audit

The files currently in this directory are useful references but are **not formal
approved**. Codex must specifically audit/fix the following before claiming readiness:

1. Add the dual-branch monitor described above.
2. Add CPU tests for all contracts above.
3. Add the 4-GPU zero-update preflight and archive its JSON result.
4. Verify the existing TRL buffering/shuffling path preserves the cached SID rollout
   correctly for both `num_iterations=2` policy iterations.
5. Verify CoT and SID old-logp rescoring uses the intended unchanged policy mode and
   no generation-score denominator.
6. Verify the old Probe4 set is exactly retained, not merely re-created with a possibly
   different selection.
7. Add an explicit final adapter/checkpoint save at the true final logical step; do not
   rely only on `save_steps=250` when the final step can be 3090.
8. Record training Beam8 as deterministic and the SID branch as Beam-selected
   PPO-style in the manifest/results.
9. Preserve runtime import-provenance hard guards and parent SHA guards.
10. Do not launch formal training as part of implementation or preflight.

## 13. Reference scaffold paths

Current reference files:

- `dual_beam8_trainer.py` — G4 sampler, local Beam8 generation, dual losses.
- `run_dual_beam8_train.py` — parent guard, Probe4 retention, config/bindings.
- `launch_dual_beam8_train.sh` — intended 4-GPU launcher shape.

Codex may refactor these files rather than patching around them, provided all contracts
in this document remain true and the old shared GRPO code is not silently modified in
a way that changes unrelated experiments.

## 14. Success criterion for implementation phase

Implementation phase is complete only when:

- CPU contract tests pass;
- 4-GPU zero-update preflight passes;
- preflight result JSON and test summary are committed to GitHub;
- parent/probe/runtime provenance are all verified;
- `GPU USED` may be YES for preflight only;
- `OPTIMIZER STEP = NO` in preflight;
- `TRAINING STARTED = NO`;
- `FORMAL TRAINING STARTED = NO`.

Only after a separate review should formal training be authorized.
