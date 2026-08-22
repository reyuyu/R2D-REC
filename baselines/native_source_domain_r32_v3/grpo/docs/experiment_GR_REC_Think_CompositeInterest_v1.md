# GR_REC_Think_CompositeInterest_v1

## Status

CPU-only provenance, parser, cohort, lexical calibration, synthetic metric audit, and tests completed on 2026-08-22.

DATA_PROVENANCE_READY = YES under the eligible-cohort policy.
LEXICAL_METRIC_READY = YES.
Formal runner construction and all GPU work remain intentionally out of scope.

No GPU, generation, optimizer step, smoke run, or training was executed.

## Motivation and scope

This Think-only ablation tests whether a balanced downstream Beam32 and structured-interest reward can resist the information collapse observed in earlier Think training. ExactClamp, NoThink, bridge, teacher, explicit diversity reward, and length reward are excluded.

Gold CoT is reward-side reference data only. It is never appended to generation prompts, Beam prompts, model inputs, or teacher-forcing targets.

## Gold provenance and eligible cohort

The GRPO source is /data/GRPO/data/rec_mp_grpo_v2/train.jsonl. Gold references are read from /data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl where source_segment equals recommendation_cot.

The join is exact equality on aux_metadata_json.recommendation_group_id. No fuzzy matching is used. Repeated rows are resolved only when their CoT prefix is byte-identical.

Counts:
- total Think groups: 1549
- exact safe Gold joins: 1458
- missing Gold: 91
- ambiguous or unresolved joins: 0
- parser-valid non-empty Gold: 1446
- excluded parser-invalid Gold: 12
- eligible Think groups: 1446

Eligibility is per group: route Think, exact safe Gold join, shared parser success, and at least one parsed interest. Missing or invalid groups are excluded; references are never generated or substituted.

## Parser calibration

Before the safe extension, the shared parser accepted 1360 of 1458 joined Gold CoTs. The 98 failures were:
- 86 missing_interest_heading results caused by observed safe title variants
- 12 empty_interest_section results caused by inline prose without line-delimited bullet boundaries

Supported title variants were extended narrowly:
- canonical bracketed Interest Summary with an optional terminal colon
- Markdown emphasis inside the brackets
- plain Markdown heading Interest Summary

All 86 title variants retain the existing numbered line bullets and were recovered. The remaining 12 inline-prose references have no unambiguous interest boundary and remain excluded. No whole-CoT or whole-paragraph fallback was added. Canonical parsing semantics are unchanged and regression-tested.

## Interest parser and evidence semantics

Gold and candidate CoTs use the same extract_interest_units function. Each unit includes order, raw text, normalized text, evidence SIDs, and evidence SIDs grounded in the current user prompt.

Normalization removes bullet numbers, SID serialization, Markdown emphasis, template markers, and redundant whitespace without deleting semantic words.

If a Gold interest has no grounded evidence, evidence is N/A for that pair and S_pair equals S_text. This makes every parser-valid Gold identity pair exactly 1. If Gold has evidence, S_pair remains 0.7 times S_text plus 0.3 times grounded evidence-set F1. Candidate-missing evidence is therefore penalized when the reference contains evidence.

## Lexical calibration protocol

Historical candidates came from GR-REC-CLAMP-BRIDGE-V1-G8BASE-FORMAL1500-20260821. Only exact-safe Gold groups with parser-valid Gold and parser-valid candidates were used.

For every candidate interest, the audit compared all same-group Gold interests and deterministically sampled five real Gold interests from other groups in the same target domain plus five from other domains. Seed was 20260822.

Metrics:
- M1: character bigram multiset F1
- M2: character unigram multiset F1
- M3: pooled namespaced character unigram and bigram multiset F1
- M4: current 8B tokenizer, special-token-free subtoken multiset F1
- M5: exact character LCS / ROUGE-L style F1

All metrics use the same normalized text and the same conditional evidence rule.

### Distribution summary

Values below are P50 / P75 / P90 / P95 for maximum pair score per candidate interest.

| Metric | Same group | Same-domain wrong group | Cross-domain wrong group | Same-vs-same-domain AUC |
|---|---|---|---|---:|
| M1 | .1806 / .2509 / .3235 / .3650 | .1183 / .1520 / .1895 / .2190 | .1132 / .1440 / .1800 / .2035 | .7202 |
| M2 | .3813 / .4640 / .5294 / .5656 | .3250 / .3722 / .4188 / .4485 | .3191 / .3656 / .4107 / .4412 | .6518 |
| M3 | .2813 / .3551 / .4250 / .4591 | .2193 / .2576 / .2994 / .3259 | .2136 / .2504 / .2889 / .3168 | .6925 |
| M4 | .3093 / .3838 / .4529 / .4942 | .2513 / .2927 / .3364 / .3682 | .2446 / .2857 / .3272 / .3580 | .6751 |
| M5 | .2430 / .3091 / .3764 / .4217 | .1923 / .2250 / .2642 / .2936 | .1887 / .2213 / .2564 / .2815 | .6747 |

M1 has the strongest rank separation.

### Threshold sweep

Each row lists threshold .20/.25/.30/.35/.40/.45/.50/.55/.60.

Historical candidate nonzero match rates:
- M1: .7424/.5236/.3125/.1647/.0642/.0236/.0034/.0008/.0000
- M2: .9907/.9713/.9181/.8269/.7035/.5481/.3640/.1740/.0625
- M3: .9493/.8649/.7247/.5380/.3226/.1503/.0532/.0152/.0034
- M4: .9696/.9037/.7973/.6453/.4417/.2551/.1208/.0372/.0118
- M5: .9130/.7838/.5701/.3302/.1731/.0718/.0245/.0059/.0008

Same-domain wrong-group any false-match rates over five sampled references:
- M1: .0788/.0212/.0056/.0012/.0000/.0000/.0000/.0000/.0000
- M2: .9801/.8929/.6541/.3602/.1485/.0483/.0168/.0068/.0012
- M3: .6606/.2867/.0996/.0283/.0081/.0028/.0003/.0000/.0000
- M4: .8291/.5125/.2179/.0747/.0246/.0087/.0022/.0003/.0000
- M5: .4371/.1463/.0411/.0112/.0022/.0000/.0000/.0000/.0000

Cross-domain any false-match rates:
- M1: .0567/.0128/.0028/.0006/.0000/.0000/.0000/.0000/.0000
- M2: .9785/.8776/.6152/.3257/.1242/.0427/.0137/.0037/.0009
- M3: .6205/.2534/.0766/.0202/.0050/.0009/.0003/.0000/.0000
- M4: .8138/.4630/.1787/.0576/.0190/.0059/.0019/.0003/.0000
- M5: .4094/.1227/.0293/.0075/.0009/.0000/.0000/.0000/.0000

The complete JSON also includes mean matched counts, pair-level false-match rates, matched-count histograms, P01-P99 distributions, and review examples.

## Selected metric

The selected S_text remains M1 character bigram multiset F1. MATCH_THRESHOLD is 0.30.

At 0.30:
- historical candidate nonzero match rate: 31.25%
- mean matched count: 0.3606
- same-domain sampled any false-match rate: 0.56%
- cross-domain sampled any false-match rate: 0.28%

Threshold 0.25 gives more signal but raises same-domain any false matches to 2.12%; 0.35 is more precise but reduces nonzero candidates to 16.47%. The conservative 0.30 point was selected from real positive and negative distributions, not from match rate alone.

The match-quality formula remains frozen at Q = clip((mean similarity - 0.60) / 0.40, 0, 1). Lowering the matching threshold does not lower the Q quality floor.

## Matching and reward

Matching is maximum-cardinality thresholded one-to-one matching with total similarity as deterministic tie-break. A candidate and Gold unit are valid only when S_pair is at least 0.30. One Gold unit cannot be consumed more than once.

K is valid match count. Coverage is K / N_gold and monitor-only precision is K / N_pred.

Coverage tier T:
- K = 0: 0
- 0 < coverage <= .25: .20
- .25 < coverage <= .50: .45
- .50 < coverage < 1: .70
- coverage = 1: 1.00

U_cot = .8 * T + .2 * Q.

U_beam = clip(log(1 + max(R_beam, 0)) / log(17), 0, 1), with invalid and non-finite inputs mapped to zero.

R_total = .60 * U_beam + .40 * U_cot.

Standard G4 advantage remains (R_total - mean) / (population_std + 1e-4), with exactly equal rewards producing exactly zero advantages.

## Historical selected-metric replay

Among 1188 historical candidates in eligible groups, 1184 were parser-valid (99.6633%).

At M1 threshold .30:
- nonzero match rate: .3125
- mean matched count: .3606
- matched counts: 0=814, 1=319, 2=45, 3=6
- U_cot mean: .0909
- U_cot P50/P75/P90/P95/P99/max: 0/.16/.36/.36/.80/.80

## Synthetic audit

On 128 parser-valid, domain/interest-count-stratified real Gold groups, mean U_cot was:
- identity: 1.0000
- reorder: 1.0000
- drop one: .5837
- drop half: .4463
- one interest: .5541
- duplicate one: .5543
- add one to three real Chinese Gold interests from other groups: 1.0000
- remove SID: .9029
- replace with wrong SID: .9029
- light lexical edit: .9542

Identity and reorder are highest; drop-half is below drop-one; duplicate does not improve coverage; real extra interests do not improve K; missing or wrong evidence lowers the score.

CPU runtime was about 2.24 ms per candidate and 8.98 ms per G4.

## Cohort topology

The original four fixed probes, deterministically recomputed with seed 20260818 in video/living/prod/ad order, are all eligible.

Static eligible-only topology:
- eligible groups: 1446
- fixed probes: 4
- post-probe groups: 1442
- sampler drop: 2
- training groups: 1440
- fresh G4 rollouts: 360
- optimizer steps with num_iterations 2: 720

This is static audit only. No formal runner was created.

## Data adapter and leakage guard

The read-only adapter now requires an explicit eligible_group_ids set. Non-eligible records are excluded. Every emitted record is asserted to have exact Gold, parser-valid non-empty interests, and a byte-identical original prompt. Eligible groups missing from source records or Gold fail closed.

Tests verify prompt parity and confirm Gold text is absent from model input prompts.

## Real G4 composite activation audit

Artifact: gr_rec_think_composite_interest_v1_g4_activation_audit_20260822.json.

The immutable historical source is the 310 Think G4 traces from
GR-REC-CLAMP-BRIDGE-V1-G8BASE-FORMAL1500-20260821. Every trace row remained an
intact four-candidate group. Thirteen G4s were excluded by current Gold
eligibility and four more because at least one candidate was parser-invalid.
The audited cohort is 293 complete G4s / 1172 candidates.

### Group-level activation

- BEAM_ZERO_STD_RATE: 135/293 = 46.08%
- U_cot all-zero: 102/293 = 34.81%
- COT_VARIANCE_ACTIVE_G4_RATE: 181/293 = 61.77%
- COMPOSITE_ZERO_STD_RATE: 54/293 = 18.43%
- rescued Beam-zero groups: 81
- RESCUED_ZERO_STD_RATE over all G4: 27.65%
- RESCUED_ZERO_STD_RATE over Beam-zero G4: 60.00%

Thus the frozen coverage reward supplies real within-G4 variance and rescues 60%
of historical Beam zero-std groups. It does not remove collapse completely:
54/293 groups remain Composite zero-std.

For the 158 Beam-variance-active groups:
- TOP_WINNER_CHANGED_RATE: 43/158 = 27.22%
- ADVANTAGE_SIGN_CHANGED_RATE: 97/632 candidates = 15.35%
- Beam/composite advantage cosine mean/P25/P50/P75:
  .7363/.7051/.9680/1.0000
- cosine min/max: -1.0000/1.0000

The winner comparison uses stable lowest-candidate-id argmax for both vectors.

### CoT structure activation

Unique U_cot values per G4:
- one value: 112/293 = 38.23%
- at least two values: 181/293 = 61.77%
- at least three values: 23/293 = 7.85%
- four distinct values: 0

Candidate matched-count distribution:
- K=0: 806/1172 = 68.77%
- K=1: 316/1172 = 26.96%
- K=2: 44/1172 = 3.75%
- K=3: 6/1172 = .51%
- K>=4: 0

Groups with at least one K>=1/K>=2/K>=3 candidate:
191/37/6, or 65.19%/12.63%/2.05%. Twelve groups contain a full-coverage
candidate; these have Gold interest counts below four.

Q>0 occurs in 0/1172 candidates and 0/293 groups. Matched-candidate mean
similarity has mean/P25/P50/P75/max .3603/.3236/.3497/.3857/.5694, below the
frozen .60 quality floor. Therefore:

QUALITY_BRANCH_HISTORICALLY_DORMANT = YES

This is not a failure of the coverage branch and no threshold or quality-floor
change was made.

### Structure correlation

Pearson/Spearman:
- U_cot vs Raw N: .1786/.1960
- U_cot vs Grounded N: .2061/.2295
- U_cot vs grounding coverage: .1150/.0846
- matched count vs Raw N: .2289/.2057
- matched count vs Grounded N: .2486/.2354
- U_cot vs completion length: .1462/.1562

The relationship is modest, but U_cot and matched count are closer to Grounded N
than Raw N or completion length. This is evidence against pure length reward,
not proof of semantic quality.

### One-interest shortcut

Across all candidates, Raw N=1 has mean U_cot/matched/composite
.0472/.1698/.2632. Raw N=3 gives .1109/.4326/.2471 and Raw N=4 gives
.1130/.4842/.2638; the unconditional Composite means are confounded by Beam.

Within Beam buckets, Raw N=3/4 minus Raw N=1 mean Composite reward is:
- Beam 0: +.0226
- Beam .5-near: +.0169
- Beam 2-near: +.0299
- Beam 8+: +.0392

Each comparison has both cohorts present. Historical evidence therefore does not
show a one-interest reward shortcut; multi-interest candidates receive higher
Composite reward at comparable Beam levels.

### Real collapse examples

The JSON contains up to five compact group summaries for every requested class.
Representative cases:
- Beam equal, matched counts differ:
  7b91df... has Beam [.5,.5,.5,.5], K [0,0,0,1], U_cot [0,0,0,.16].
- Beam equal, Raw N differs:
  b190ca... has Beam [2.25]*4, Raw N [1,1,4,4], K [0,0,2,1].
- High Beam with low U_cot:
  0d3b5e... has Beam [.5,8,2,8] and U_cot [0,0,0,0].
- Low Beam with high U_cot:
  cda568... has Beam [.5,.5,0,.5] and U_cot [.8,0,0,0].
- Composite winner differs:
  f1bd21... changes the stable winner with Beam [0,2,2,0] and
  U_cot [.36,0,.16,.16].

### Eligible-domain audit

Original / eligible / projected-training counts:
- video: 550 / 546 / 544, eligible 99.27%
- prod: 382 / 355 / 354, eligible 92.93%
- ad: 427 / 381 / 379, eligible 89.23%
- living: 190 / 164 / 163, eligible 86.32%

Overall eligibility is 93.35%. Missing-Gold exclusions are
video/prod/ad/living = 0/24/43/24; parser-invalid exclusions are 4/3/3/2.

DOMAIN_COHORT_SKEW_RISK = YES under the declared rule that a domain's eligible
rate deviates from the overall rate by more than five percentage points. Video
is overrepresented and living underrepresented. The projected training-domain
counts use seed-20260818 post-probe shuffle and G4 tail truncation only for this
audit; no runner was created.

The four fixed probe IDs are unchanged and all remain eligible.


## Artifacts

- gr_rec_think_composite_interest_v1_gold_cot_provenance_20260822.json
- gr_rec_think_composite_interest_v1_similarity_calibration_20260822.json
- gr_rec_think_composite_interest_v1_offline_metric_audit_20260822.json
- gr_rec_think_composite_interest_v1_g4_activation_audit_20260822.json

## Remaining gate

LEXICAL_METRIC_READY is YES for M1 at threshold .30. This does not authorize GPU work. The next phase still requires explicit code review and a separately authorized runner implementation.

## Production training-chain integration (2026-08-22)

Base main: `7b8005aebe8610c0824624be344cececebceaaba`.

The stable Think generation, exact `</think>` stop, G4 sampling and Beam32 path
remain inherited from `RecGRPOTrainer`. The thin experiment subclass reads
reward-only `gold_cot` after generation, computes the frozen Composite reward,
gathers metadata in the same DDP order as rewards, normalizes each contiguous
global G4 with population std (`correction=0`) plus `1e-4`, and replaces only
the final sequence advantage. It does not override `_compute_loss` and does not
import or call ExactClamp.

Gold leakage is fail-closed across raw/rendered prompts and decoded input IDs.
Candidate parser failure yields `U_cot=0`; invalid Gold, non-finite values,
range violations, DDP misalignment and group-contiguity drift are hard errors.
Beam32 never receives Gold CoT.

`composite_interest.jsonl` records every candidate's Beam raw/utility, Gold,
predicted and matched counts, coverage, precision, match similarity, tier,
quality, U_cot, Composite reward, Raw N, Grounded N, grounding coverage and
final advantage. Group rows preserve complete vectors, diversity fields and
separate top-set tie-break versus strict Beam reversal labels. Fixed Think
probes reuse the production generation and Beam functions and never optimize.

Actual CPU dry-run source chain: 1549 original Think, 1458 exact Gold joins,
1446 parser-valid eligible, four fixed probes excluded, 1442 post-probe, two
deterministically tail-dropped, 1440 training groups, 360 fresh G4 rollouts and
720 optimizer steps at two iterations.

The runner asserts the frozen Think-only contract and schedules checkpoints at
100/200/250/300/350/400/450/500/600/720. CPU tests pass 43/43. Dry-run artifact:
`gr_rec_think_composite_interest_v1_runner_dry_run_20260822.json`.

GPU used: NO. Model loaded: NO. Generation: NO. Optimizer step: NO. Training: NO.

## GPU validation gate (2026-08-22)

Launch provenance: `72856f3c927527f26338700a03fc6641de034788`.

The authorized four-A800 zero-update preflight loaded the original 8B base and
fresh original BATA adapter, then stopped at the first TRL prompt-length NCCL
gather with `ncclSystemError` / socket connection abort. This occurred before
candidate generation completed, reward calculation, backward or any optimizer
operation. Optimizer steps remained zero and all GPU worker processes exited.

Per the hard gate, Smoke12 was not started and no retry was attempted. The
structured preflight and not-started Smoke result files preserve this status.
Formal 720-step training remains unstarted.

## Single-node NCCL and 12-probe schedule revision (2026-08-22)

Base main for this revision: `72fa8ab9f16b742a8e8a49dd39ca9b0af2a257f3`.
The failed zero-step GPU preflight above remains historical evidence; this
revision used CPU only and did not retry preflight, Smoke or formal training.

Preflight, Smoke and formal execution now enter through the same minimal
`configure_single_node_nccl()` bootstrap. For the explicitly single-node
four-rank task it validates `WORLD_SIZE=4` and `LOCAL_RANK=0..3`, sets
`NCCL_SOCKET_IFNAME=lo` before process-group initialization, binds the current
CUDA device, and initializes NCCL with that device only when a process group
does not already exist. The launch manifest records the interface, world size,
current local rank/device and the complete local-rank/device mapping.

The fixed Think probe cohort is frozen at seed 20260818 as three deterministic
rounds. Every round maps ranks 0/1/2/3 to video/prod/ad/living respectively:

- round 0: video `662e2114...55cbc`; prod `994c3186...752d5`; ad
  `6aa980e4...cbdc2`; living `9d9852a4...8559`
- round 1: video `2458dad5...e01a8`; prod `39aae420...e7478`; ad
  `30ad9496...b1773`; living `f514f18d...a8afa`
- round 2: video `57e5e216...b5136`; prod `501e2511...ce62c`; ad
  `03d3105f...f8407`; living `31208c4e...7e93a`

All 12 IDs are unique, exact-Gold provenance safe, parser-valid members of the
1446-group eligible cohort, with exactly three probes per domain. Their
intersection with the shuffled training cohort is empty.

The revised full-epoch topology is:

- eligible groups: 1446
- fixed probes: 12
- post-probe groups: 1434
- deterministic G4 tail drop: 2
- training groups: 1432
- fresh G4 rollouts: 358
- optimizer steps at `num_iterations=2`: 716

Formal `max_steps` is therefore 716. Checkpoints are callback-only milestones
at 200/400/600/716; Trainer auto-save remains 10000 and
`save_total_limit=4`. Fixed probes run only at 0/200/400/600/716, with all 12
groups required before a step is considered complete. The formal cost is 60
group-level probe evaluations. Partial JSONL state does not skip a milestone.

The Composite reward, `.30` match threshold, `.60` quality floor, parser, raw
completion decode, DDP alignment, G4 population advantage, generation and
Beam32 contracts are unchanged.

Targeted CPU tests pass 59/59. The refreshed dry-run artifact reports the new
topology, domain-balanced frozen probe mapping, `TRAIN_PROBE_OVERLAP=0`, explicit
checkpoint/probe milestones and `single_node_nccl_socket_ifname="lo"` without
loading a model or initializing CUDA/NCCL.

GPU used: NO. Training started: NO. Formal training started: NO.

## Four-GPU gated validation attempt (2026-08-23)

Exact launch commit: `3f0120d7227bfc0c65f11eeee707ba595209d7f3`.
At launch, `HEAD` and `origin/main` matched exactly and the worktree was clean.
All four A800 GPUs were idle. The single-node bootstrap used
`NCCL_SOCKET_IFNAME=lo`, `WORLD_SIZE=4`, and the fixed rank-to-device mapping
rank 0/1/2/3 to cuda 0/1/2/3. The original 8B base and fresh original BATA
adapter were loaded independently on all ranks.

The one authorized zero-update preflight attempt completed candidate generation,
Beam32 scoring, raw completion decoding, Composite reward construction, and G4
advantage construction. It then failed on every rank while writing the captured
Composite monitor event: the runtime `MonitorWriter` raised `AttributeError` for
`write_composite`. This occurred before loss and backward completed, so the
existing hard-gate conditions could not be fully evaluated. Optimizer and
scheduler steps remained zero; no parameter update was attempted.

Therefore `PREFLIGHT_PASS=NO`. Per the gate, no retry was attempted, Smoke12 was
not started, its parameter audit was not created, and 716-step formal training
was not started. All worker processes exited and all four GPUs returned to about
5 MiB idle usage with no compute processes. The structured preflight result and
the blocked Smoke12 result preserve this outcome.

## Import provenance repair and gated retry (2026-08-23)

Base main: `0ff6709dd688b05efdcf035aca3f8715c7853988`. Before the
repair, a fresh CPU process loaded `grpo_trl_trainer.py` from this worktree but
loaded `monitor.writer` from `/data/GRPO/scripts/monitor/writer.py`; that stale
writer did not provide `write_composite`. The direct cause was an absolute
`sys.path.insert(0, "/data/GRPO/scripts")` in the active training chain.

Commit `61cfae3bac3ffc896bb3fad2db8cb061fba0eca5` replaces those stale
absolute-priority injections with the scripts directory derived from each
module's own file path. A shared fail-closed guard now verifies the trainer and
monitor module paths plus `MonitorWriter.write_composite` before NCCL
initialization or model loading. Preflight output and formal/Smoke manifests
record the resulting module provenance. Targeted CPU tests passed 48/48.

The four GPUs were idle, so the one authorized post-fix retry launched from that
exact commit. Runtime import provenance passed and the previous monitor failure
was eliminated: the captured `composite_interest.jsonl` was written after
generation, Beam32, raw decode, Composite reward and G4 advantage. The preflight
then failed on all ranks when its direct `_compute_loss` call reached an absent
`PreflightTrainer.current_gradient_accumulation_steps` attribute. Loss did not
complete and backward did not start. Optimizer/scheduler steps remained zero and
no parameters were updated.

Per the gate, this new failure was not modified or retried, Smoke12 was not
started, and formal training remained unstarted. All workers exited and the four
GPUs returned to approximately 5 MiB idle usage with no compute processes.

## Preflight loss-context repair and final gated attempt (2026-08-23)

Base main was `959cf6247c9f90e2d75caed750abcd50e3f6a0ee`. Commit
`b9e6962ce83d636c7fb637eb876598795bd144cf` repairs only the manual
preflight harness: it verifies the shared frozen
`gradient_accumulation_steps=1`, initializes
`current_gradient_accumulation_steps=1`, and enters the public
`compute_loss` context before backward. It does not call `training_step` and
does not change the formal trainer, reward, advantage, generation, Beam32,
parser, monitor schema, Smoke12 math, or the 716-step formal contract. Targeted
CPU tests passed 51/51 and Python compilation plus `git diff --check` passed.

The one authorized four-GPU attempt launched from that exact commit with four
idle A800 GPUs and fresh original BATA. Runtime import provenance, NCCL,
generation, Beam32, Composite monitor writing, loss, and backward completed.
The recorded finite loss was `1.30385160446167e-08`; aggregate LoRA gradient
norm was `0.9871858095670214`, with 2016 LoRA gradient tensors and zero base
gradient tensors. The trainable checksum was identical before and after, and
optimizer steps remained zero.

The complete hard gate nevertheless returned `PREFLIGHT_PASS=NO` because
`raw_decode_runtime=false` and `online_advantage_parity=false`. Reward parity,
DDP G4 alignment, Gold-leakage absence, finite loss/gradients, LoRA-gradient
presence, base-gradient absence, checksum invariance, and NCCL/OOM/NaN/Inf
guards all passed. Per the fixed gate, no further retry or modification was
attempted, Smoke12 was not started, and formal training was not started. All
four GPUs returned to approximately 5 MiB idle usage with no compute process.
