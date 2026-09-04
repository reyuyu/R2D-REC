# GR_REC Full-Base Conservative V1

This isolated recipe adapts the reproduced Rec FDR V4.3 full-parameter SFT with a fresh GR_REC LoRA. It does not modify the historical GR_REC implementation.

## Frozen Parent

| Field | Value |
|---|---|
| Parent mode | `full_model` |
| Model role | `rec_fdr_v43_full_sft_grpo_base` |
| `model.safetensors` SHA256 | `8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59` |
| `config.json` SHA256 | `78451878177a5443d87440940c54177da6e833e8ab4c97dab0e263a4c97162e2` |
| Mutation policy | Immutable; no merge and no output beneath the parent directory |

The model and dataset paths are always supplied at runtime. Neither private paths nor data/model bytes are part of this repository.

## Training Contract

The old recommendation data, prompt renderer, SID parser, reward functions, route-aware sampler, Beam32 logic, generation parameters, population-std advantage, GRPO loss, and beta remain unchanged. The intentional changes are:

- load the reproduced full SFT directly instead of public base plus SFT adapter;
- initialize a fresh rank-32 GR_REC LoRA with the historical architecture;
- lower the learning rate from `1e-6` to `5e-7`;
- add fail-closed parent lineage, immutable-base checks, deterministic evidence, and fixed retention probes.

The base has zero trainable parameters. The optimizer parameter IDs must exactly equal the 504 trainable LoRA tensor IDs.

## Gated Execution

`run_smokes_and_pilot.sh` performs the only authorized sequence:

1. preflight and five-step `SMOKE-A`;
2. independent preflight and five-step `SMOKE-B`;
3. exact comparison of rollout order, prompt/completion fingerprints, rewards, advantages, policy losses, LR trajectory, and final LoRA;
4. only after comparison PASS, a 20-step pilot at `5e-7`.

Any mismatch exits before the pilot. This launcher never starts a full epoch or MC_USER.

Runtime example:

```bash
FULLBASE_MODEL=/runtime/full-sft \
GRPO_DATA=/runtime/rec-grpo/train.jsonl \
RUN_ROOT=/runtime/new-output-root \
MONITOR_ROOT=/runtime/new-monitor-root \
bash scripts/run_smokes_and_pilot.sh
```

## Retention Definitions

The four historical fixed groups remain excluded from training. At Step0 the pure full SFT and fresh zero-update LoRA must have identical completion hashes, rewards, parsed SIDs, and Beam32 counts.

- Think `Exact/AB+/A+ beam rate`: matching Beam32 outputs divided by all Beam32 outputs.
- Think `Success@K`: fixed groups with at least one A+ candidate among the four sampled candidates.
- Think `Success@32`: sampled Think candidates with at least one A+ result among their 32 beams.
- Think `Invalid rate`: invalid Beam32 outputs divided by all Beam32 outputs.
- Think `Closure rate`: sampled candidates containing `</think>`.
- NoThink `Exact/AB+/A+ rate`: candidate rewards equal to `8`, at least `2`, and at least `0.5` respectively.
- NoThink `Success@K`: fixed groups with at least one positive candidate among eight samples.

Probe RNG is restored after every evaluation and probe records never enter reward, advantage, or optimizer state.

## Checkpoints

Checkpoints are adapter-only and include adapter config/weights, optimizer, scheduler, trainer state, all four rank RNG states, and `lineage.json`. Reload fails unless the full-SFT parent SHA and lineage match exactly. Odd smoke checkpoint 5 is evidence-only and explicitly non-resumable because `num_iterations=2`; pilot checkpoints 10 and 20 are resumable.
