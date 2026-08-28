# GR_REC_ThinkSuffixSID_Resample_v1

Status: CPU implementation and monitor contract. No GPU validation or training has started.

## Objective

Start a new optimizer from the immutable `GR_REC_v1 checkpoint-1500` adapter
(recorded external aggregate `1.3510`) and train only the answer suffix of
Think recommendation samples. The sampled CoT remains visible to the policy as
context but receives no PPO/GRPO loss.

## Immutable parent

- Base: `/data/models/onereason-8b-pretrain-competition`
- Adapter: `/root/data_checkpoints_backup_20260824/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500`
- Adapter SHA256: `a5e92db011662799e07b4e1f16a2779afbbab9d66efcb481a5f2e199c75d3436`
- Recorded external aggregate: `1.3510` (single evaluation; not a repeated stable estimate)
- Optimizer/scheduler: fresh. The old checkpoint optimizer state is not resumed.

## Rollout and reward

- Route: Think only.
- One global rollout contains one recommendation group and `G=8` independently
  sampled full completions.
- Sampling: temperature `0.9`, top-p `0.95`, max completion `2048`.
- Completion: `CoT -> </think> -> answer suffix` in one autoregressive rollout.
- Beam32: disabled.
- Runtime imports are fail-closed: the formal runner, model, trainer, and
  monitor writer must all resolve from the launch worktree before CUDA init.
- Reward: final complete SID after the first `</think>`, using the original
  NoThink-v1 levels `-1 / -0.25 / 0 / 0.5 / 2 / 8`.
- Advantage: `(R - group_mean) / (population_std + 1e-4)`.

## Loss boundary

The attention mask covers the complete sampled CoT and answer. A separate loss
mask is zero through the first complete `</think>` token sequence and one only
for the following non-padding answer tokens. Therefore SID prediction can
condition on the sampled CoT without updating the policy on CoT tokens.

Missing `</think>` yields no trainable suffix. Its reward is `-1`; the group can
still be rerolled by the zero-std rule.

## Zero-std reroll

The trigger is reward population standard deviation equal to zero, evaluated
before backward. It is not a post-backward gradient-norm check.

1. Generate eight candidates.
2. If their rewards have non-zero population std, accept the round.
3. Otherwise discard all eight and sample a fresh G8 round.
4. Allow at most three additional rounds: `8 + 3*8 = 32` candidates.
5. Use only the first non-zero-std round for GRPO. Never select the best of all
   generated candidates.
6. If all four rounds are zero-std, return the fourth round, produce zero
   advantages, and record `zero_std_rescue_exhausted=true`.

All ranks gather the same G8 reward vector and make one synchronized decision.
Rejected rounds perform no backward and no optimizer update.

## Multi-SID diagnostic

Every complete SID occurrence after `</think>` is retained in monitor traces.
When there is more than one occurrence, the candidate records:

- `multi_sid_output=true`
- `sid_count`
- `all_parsed_sids`
- `parser_status=multiple_suffix_sids`

This is monitor-only. Reward continues to use the final complete suffix SID and
no format penalty is introduced in v1.

## Entry points

- Runner: `ablations/gr_rec_think_suffix_sid_v1/run_think_suffix_sid_train.py`
- Trainer: `ablations/gr_rec_think_suffix_sid_v1/think_suffix_sid_trainer.py`
- Pure contracts: `ablations/gr_rec_think_suffix_sid_v1/suffix_objective.py`
- Launcher: `ablations/gr_rec_think_suffix_sid_v1/launch_think_suffix_sid_train.sh`

GPU validation must first prove G8 cross-rank grouping, synchronized rerolls,
suffix-only non-zero gradients, unchanged CoT-token gradients, frozen base
parameters, and finite LoRA updates. Formal training is not authorized by this
implementation commit alone.
