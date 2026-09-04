# Conservative Full-Base GRPO Pilot Report

## Scope

- Recipe: `grpo_fullbase_conservative_v1`
- Code commit used for execution: `e29d51d4738211453528c4d7f79bd97e7b614c7a`
- Parent mode: `full_model`
- Parent role: Rec FDR V4.3 full-parameter SFT
- Parent model SHA256: `8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59`
- Parent config SHA256: `78451878177a5443d87440940c54177da6e833e8ab4c97dab0e263a4c97162e2`
- GRPO dataset SHA256: `791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc`
- Pilot learning rate: `5e-7`
- Pilot optimizer steps: `20`
- World size: `4`

No model weights, training data, credentials, or private server paths are included in this report.

## Contract Results

| Check | Result |
| --- | --- |
| Full SFT parent SHA verified | PASS |
| Full-model parent loader | PASS |
| Legacy adapter parent regression | PASS |
| Base trainable parameters | 0 |
| LoRA trainable parameters | 87,293,952 |
| LoRA trainable tensors | 504 |
| Optimizer parameters are exactly LoRA parameters | PASS (504 tensors) |
| Old-adapter/new-base mismatch fails closed | PASS |
| New CPU contract tests | PASS (11/11) |
| Legacy directly related regressions | PASS (6 suites) |
| Smoke A | PASS (5/5) |
| Smoke B | PASS (5/5) |
| Smoke A/B repeatability | PASS, no first divergence |
| Step0 pure-base vs zero-update LoRA parity | PASS |
| Pilot | PASS (20/20) |
| Base file SHA unchanged | PASS |
| Sampled in-memory base tensors unchanged | PASS |
| LoRA changed | PASS |
| Checkpoints | PASS (`checkpoint-10`, `checkpoint-20`, adapter-only) |

Smoke A and B produced the same final adapter SHA256:
`65c344435e4c9f683345448c6c9ab635538a6498cd2000c49c415ea8d51ea833`.

The pilot final adapter SHA256 is:
`c2ed3556f0d653dff3d538894900f3b33129bc33cf888fcdb30964851f2b4051`.

## Retention Probe

The fixed four-group probe was excluded from training and ran with RNG save/restore.

| Route metric | Step 0 | Step 20 |
| --- | ---: | ---: |
| Think mean reward | 3.159180 | 2.828125 |
| Think Success@K | 0.750000 | 0.750000 |
| Think Success@32 | 0.625000 | 0.687500 |
| Think exact beam rate | 0.011719 | 0.009766 |
| Think closure rate | 1.000000 | 1.000000 |
| Think invalid rate | 0.000000 | 0.000000 |
| NoThink mean reward | 0.062500 | 0.132812 |
| NoThink Success@K | 0.500000 | 0.250000 |
| NoThink exact rate | 0.000000 | 0.031250 |
| NoThink positive candidate rate | 0.250000 | 0.218750 |

The retention result is mixed rather than a clean improvement or a uniform regression. The probe is intentionally small, so this report does not authorize a full epoch. Review the Step 0/20 trade-off before choosing a lower learning rate or a later KL-retention experiment.

## Outcome

`READY_FOR_CONSERVATIVE_GR_REC_REVIEW`

The run stopped after the single 20-step pilot. No full GRPO epoch and no MC_USER training were started.
