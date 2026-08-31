# MC_USER Hybrid K4 Strong-Parent Final Record

## 1. Final lineage

This record freezes the current selected model lineage. Each stage starts from the selected checkpoint of the preceding stage; these are not four independent experiments.

| Stage | Experiment / checkpoint | Optimization scope | Recorded aggregate |
| --- | --- | --- | ---: |
| SFT | BETA-baseline Epoch 2, `checkpoint-1106` | Multi-task SFT | about `1.33` (`1.3246` first record; `1.3313` higher repeat) |
| Recommendation GRPO V1 | `GR_REC_v1 checkpoint-1500` | Think + NoThink Recommendation | `1.3510` |
| Recommendation Think GRPO | `GR_REC_ThinkSample8_FullSID_v3 checkpoint-250` | Think G4 + per-CoT Sample8 FullSID | `1.3579` |
| User refinement | Strong-parent MC_USER Hybrid K4 `prompt-step-0100` | Action + Chain, sequence GRPO + marginal local credit | project headline about `1.356`; observed repeats `1.3639 / 1.3595` |

The final headline uses the project's user-specified conservative `about 1.356` presentation. It is not a third raw evaluation. The exact currently preserved external observations for Step 100 are `1.3639` and `1.3595`, with mean `1.3617` and range `0.0044`.

## 2. Motivation

### Stage 1: establish a multi-task SFT base

BETA-baseline provides the common OneReason-8B LoRA policy with Material, User, and Recommendation behavior already learned. Its purpose in this chain is broad capability initialization, not task-specific outcome optimization. The external evaluator varies by roughly `+/-0.01`, so the historical `1.3246` and higher repeat `1.3313` are both retained; the README displays this stage as about `1.33`.

### Stage 2: improve Recommendation through both production routes

`GR_REC_v1` optimizes Recommendation only while preserving the SFT base. Think uses G4 sampled reasoning followed by Beam32 outcome reward. NoThink uses G8 direct-SID tiered reward, with a `0.5` route multiplier to keep total per-group route mass comparable to Think. Step 1500 reached `1.3510`; later checkpoints were non-monotonic, so Step 1500, not the final training step, became the next parent.

### Stage 3: refine the Think route and its final SID

`GR_REC_ThinkSample8_FullSID_v3` starts from `GR_REC_v1 checkpoint-1500` and trains only Think. For each of four sampled CoTs, it independently samples eight full-SID continuations. SID advantages are normalized inside each per-CoT G8 group, while the four CoTs receive a second group-relative advantage from their aggregate SID outcomes. The objective is `cot_loss + sid_loss`. Step 250 reached `1.3579` and was selected before later degradation.

### Stage 4: conservative User refinement on the strongest Recommendation parent

The earlier MC_USER Hybrid run used `GR_REC_v1 checkpoint-1500`, learning rate `1e-6`, and 512 prompts. User metrics rose, but Recommendation fell increasingly at Steps 256 and 512. The strong-parent refinement therefore changed the parent to GRPO-TK Step 250, reduced the learning rate to `3e-7`, limited the budget to 200 prompts, and saved dense checkpoints.

The Hybrid objective is:

```text
L_total = 1.0 * L_sequence_GRPO + 0.3 * L_marginal_local
```

Sequence GRPO assigns group-normalized reward advantage to every real completion token. Marginal local loss gives evaluator-aligned leave-one-out credit to predicted Action SIDs or Chain events. It does not add a History-copy reward, PPO ratio, KL term, or Recommendation retention loss.

## 3. Reproducible paths and contracts

| Artifact | Development-machine path |
| --- | --- |
| OneReason-8B base | `/data/models/onereason-8b-pretrain-competition` |
| BETA SFT checkpoint | `/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106` |
| GR_REC_v1 Step 1500 parent used by GRPO-TK | `/root/data_checkpoints_backup_20260824/GRPO/outputs/formal/REC-MP-GRPO-FULL-E1-PROBE4-DOMAIN-20260818/checkpoint-1500` |
| GRPO-TK Step 250 parent used by MC_USER | `/root/GRPO-checkpoints/GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-250` |
| MC_USER run | `/data/GRPO_USER/runs/mc_user_v1_hybrid_formal/MC-USER-HYBRID-K4-STRONGPARENT-LR3E7-200-20260831-171920` |
| Selected final adapter | `/root/GRPO-checkpoints/MC_USER_HYBRID/MC-USER-HYBRID-K4-STRONGPARENT-LR3E7-200-20260831-171920/checkpoints/prompt-step-0100` |

Strong-parent training contract:

| Field | Value |
| --- | --- |
| Git commit | `d961d527d0b088c5907557e4ce12d2013bd8a58e` |
| Data | `/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl` |
| Frozen data SHA256 | `5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801` |
| Selection | 100 Action + 100 Chain, strict alternating |
| Parallelism | 4-GPU candidate parallel, K=4 |
| Sampling | temperature `0.9`, top-p `0.95`, max new tokens `512` |
| Optimizer | AdamW, LR `3e-7`, weight decay `0`, gradient accumulation `1` |
| Objective | sequence weight `1.0`, local weight `0.3` |
| Checkpoints | `25 / 50 / 75 / 100 / 150 / 200` |

## 4. Training integrity

The full 200-prompt run completed before checkpoint selection:

| Check | Result |
| --- | ---: |
| Status | `PASS` |
| Prompt / optimizer steps | `200 / 200` |
| Metrics / rollout rows | `200 / 800` |
| Valid candidate rate | `99.625%` |
| No-credit prompt rate | `0%` |
| Projection-required rate | `2.375%` |
| Overlap candidate rate | `0%` |
| Base hash unchanged | `true` |
| Trainable / LoRA tensors | `504 / 504` |
| Peak VRAM per process | `22740.44 MiB` |
| Wall time | `3972.66 s` |

All six checkpoint directories exist. Step 100 was selected by external evaluation rather than training loss, on-policy reward, the fixed 3+3 User proxy, or the fact that it is later than Step 75.

## 5. Step 100 external evaluation

The evaluator output order is Material 4, User 2, Recommendation 4, and World 1.

| Evaluation | Aggregate | Material | User | Recommendation | World |
| --- | ---: | --- | --- | --- | ---: |
| Repeat A | `1.3639` | `0.0521, 0.0361, 0.0516, 0.0420` | `0.1586, 0.0986` | `0.1400, 0.1700, 0.2100, 0.1692` | `0.2357` |
| Repeat B | `1.3595` | `0.0514, 0.0359, 0.0522, 0.0421` | `0.1586, 0.0986` | `0.1391, 0.1700, 0.2072, 0.1692` | `0.2353` |

Displayed group sums:

| Model | Material | User | Recommendation | World | Aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO-TK parent Step 250 | `0.1822` | `0.2575` | `0.6840` | `0.2342` | `1.3579` |
| MC_USER Step 100, Repeat A | `0.1818` | `0.2572` | `0.6892` | `0.2357` | `1.3639` |
| MC_USER Step 100, Repeat B | `0.1816` | `0.2572` | `0.6855` | `0.2353` | `1.3595` |
| MC_USER Step 100, mean | `0.1817` | `0.2572` | `0.6874` | `0.2355` | `1.3617` |

The rounded Repeat B components sum to `1.3596`; the table retains the evaluator aggregate `1.3595`. Relative to the single recorded GRPO-TK parent score, the two Step 100 aggregate deltas are `+0.0060` and `+0.0016`. Both remain inside the repository's approximate `+/-0.01` single-run variation band, so this is the selected practical checkpoint, not proof of a statistically stable gain.

The final increase does not come from a clear external User-score increase: displayed User total is `0.2572` versus parent `0.2575`. The observed aggregate difference is mainly Recommendation and World. Therefore the correct claim is that an Action/Chain-trained conservative refinement retained the strong parent's overall score and produced two competitive repeats, not that the external User metrics themselves definitely improved.

## 6. Final decision

- Selected adapter: strong-parent MC_USER Hybrid K4 `prompt-step-0100`.
- Public chain shorthand: `SFT ~1.33 -> dual-route Recommendation GRPO 1.3510 -> Think GRPO 1.3579 -> User Hybrid ~1.356`.
- Exact final evidence: `1.3639 / 1.3595`, mean `1.3617`.
- Checkpoint selection is non-monotonic at every GRPO stage; later is not assumed better.
- The exact scores are experimental records, not a guarantee under a noisy external evaluator.

The machine-readable companion is [mc_user_hybrid_strongparent_final_v1.json](../results/mc_user_hybrid_strongparent_final_v1.json).
