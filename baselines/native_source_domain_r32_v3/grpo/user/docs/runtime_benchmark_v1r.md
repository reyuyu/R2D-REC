# GR_USER_v1 Phase 5A-R Runtime Benchmark

This benchmark separates generation, cached policy/backward, and a tiny
end-to-end check. It is not a formal Pilot or a full epoch. G=4,
temperature=0.9, top-p=0.95, max-new-tokens=512, sqrt lambda=0.5, reward,
sampling, epsilon, beta, and optimizer mathematics remain frozen.

## Generation

| Config | wall sec | sec/prompt | prompts/hour | generated tok/sec | padding ratio | peak VRAM MiB | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 182.15 | 4.554 | 790.6 | 144.7 | 16.32% | 38145 | 0.00% |
| P1 | 177.94 | 4.448 | 809.3 | 147.7 | 2.33% | 32156 | 2.37% |

One short 8-prompt warmup and one timed 40-prompt pass are used per config.

## Cached policy/backward

| Config | sec/step | sec/prompt | policy tok/sec | padding ratio | grad norm | clip fraction | peak VRAM MiB | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 25.074 | 3.134 | 4553.0 | 21.83% | 0.3543 | 0.0114 | 38046 | 0.00% |
| P1 | 26.595 | 3.324 | 4130.9 | 10.85% | 0.3549 | 0.0100 | 38476 | -5.72% |

Both configs read identical cached completion IDs, rewards, sequence/token
advantages, and penalty masks. Two warmup steps are excluded and five small
global-batch steps are timed on real four-GPU DDP.

## Correctness

```json
{
  "P0": {
    "passed": true,
    "reward_exact": true,
    "sequence_advantage_exact": true,
    "token_advantage_exact": true,
    "penalty_mask_exact": true,
    "loss_abs_error": 0.0,
    "gradient": {
      "max_abs_error": 0.0,
      "relative_l2_error": 0.0,
      "cosine_similarity": 1.0
    },
    "one_step_update": {
      "max_abs_error": 0.0,
      "relative_l2_error": 0.0,
      "cosine_similarity": 1.0
    }
  },
  "P1": {
    "reward_exact": true,
    "sequence_advantage_exact": true,
    "token_advantage_exact": true,
    "penalty_mask_exact": true,
    "loss_abs_error": 0.0,
    "loss_tolerance": 2e-05,
    "gradient": {
      "max_abs_error": 8.798111230134964e-05,
      "relative_l2_error": 0.009153653153786263,
      "cosine_similarity": 0.9999581121843116
    },
    "one_step_update": {
      "max_abs_error": 1.9995495676994324e-06,
      "relative_l2_error": 0.09242958413707435,
      "cosine_similarity": 0.9957283628436171
    },
    "acceptance": {
      "gradient_relative_l2_limit": 0.02,
      "gradient_cosine_limit": 0.9999,
      "one_step_update_cosine_limit": 0.995
    },
    "passed": true
  }
}
```

## Tiny end-to-end

```json
[
  {
    "name": "P0",
    "length_bucketing": false,
    "forward_batch_size": 1,
    "prompts": 8,
    "wall_seconds": 57.01123467087746,
    "seconds_per_prompt": 7.126404333859682,
    "generation_seconds": 31.25031330343336,
    "reward_mask_seconds": 0.11857098434120417,
    "policy_forward_seconds": 14.28510270640254,
    "backward_seconds": 27.96526129823178,
    "optimizer_seconds": 0.121187892742455,
    "generated_tokens": 4114,
    "policy_tokens": 122326,
    "loss": 0.0034474246203899384,
    "grad_norm": 0.5233163217373509,
    "clip_fraction": 0.0,
    "peak_vram_mib": 26682.8251953125,
    "per_gpu_peak_vram_mib": [
      21538.59814453125,
      23474.69384765625,
      25308.48681640625,
      26682.8251953125
    ],
    "speedup_percent_vs_p0": 0.0
  },
  {
    "name": "P1",
    "length_bucketing": true,
    "forward_batch_size": 1,
    "prompts": 8,
    "wall_seconds": 56.92858051881194,
    "seconds_per_prompt": 7.116072564851493,
    "generation_seconds": 31.18776602577418,
    "reward_mask_seconds": 0.30696805752813816,
    "policy_forward_seconds": 14.058477302081883,
    "backward_seconds": 28.03618593327701,
    "optimizer_seconds": 0.11757443193346262,
    "generated_tokens": 4114,
    "policy_tokens": 122326,
    "loss": 0.0034474246203899384,
    "grad_norm": 0.5230418583025109,
    "clip_fraction": 0.0,
    "peak_vram_mib": 26683.81298828125,
    "per_gpu_peak_vram_mib": [
      21538.90185546875,
      23476.22119140625,
      25300.59716796875,
      26683.81298828125
    ],
    "speedup_percent_vs_p0": 0.1451892025275514
  }
]
```

## Extrapolation

| Config | generation sec/prompt | cached policy sec/prompt | 150 | 300 | 600 | 3000 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 4.554 | 3.134 | 1153.2 | 2306.4 | 4612.8 | 23063.8 |
| P1 | 4.448 | 3.324 | 1165.9 | 2331.8 | 4663.7 | 23318.4 |

Recommended runtime config: **P0**.

P1 versus P0: generation speedup `2.37%`, cached policy speedup `-5.72%`,
and combined speedup `-1.09%`. Generation padding fell by `85.73%` relative;
cached policy padding fell by `50.32%`.

Frozen data remained unchanged. No 150-prompt Pilot, formal Pilot, full epoch,
or P2 performance benchmark was run. GR_REC_v1 was not modified.
