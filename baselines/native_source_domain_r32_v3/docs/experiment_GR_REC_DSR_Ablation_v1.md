# GR_REC_DSR_Ablation_v1: Diversity & Signal Rescue GRPO

Status: **IMPLEMENTED / SMOKE ONLY**

- Parent: original BATA baseline adapter
- Frozen baseline: `GR_REC_v1`
- Treatment: `DSR-GRPO`
- Long training: not authorized

This is a multi-mechanism ablation targeting three observed failures:

1. Think CoT single-interest mode collapse.
2. Think zero-signal and exploration collapse.
3. NoThink all-zero repeated wrong-A zero-gradient groups.

The ablation does not modify production source files, sampling, Beam32,
primary rewards, KL/beta, group sizes, temperature, top-p, or model loading.
It adds no model forward and starts from BATA rather than any GRPO checkpoint.

## Frozen primary reward

- Think: Beam32 hierarchical multi-positive reward with Exact=8, AB=2,
  A=0.5, prefix dedup, geometric decay, context batch 1, and rank balancing.
- NoThink: `-1 / -0.25 / 0 / 0.5 / 2 / 8`.

## DSR treatment

Think parses `【兴趣归纳】` deterministically and only counts bullets containing
complete SIDs present in the original prompt. Evidence diversity uses verified
prompt SIDs, so fabricated SIDs cannot lower overlap. Its score and Beam prefix
or dead-zero exploration statistics produce a separately group-normalized
advantage and a clipped PPO surrogate weighted by `0.10`.

NoThink rescue runs only for exact all-zero `G=8` groups. It locates the sampled
`s_A` token from raw completion IDs after auditing the real tokenizer, then
applies repetition-frequency-weighted unlikelihood at that position only.

## A. Isolation

The implementation lives under `scripts/ablations/gr_rec_dsr_v1` and has its
own smoke/train runners, tests, README, monitor files, output tags, and run IDs.
It subclasses `RecGRPOTrainer` and wraps the already-computed primary rewards
and Beam32 records without changing their returned values. The only copied
baseline method is `_compute_loss`, because TRL exposes no post-logprob loss
hook; its zero-coefficient equivalence is covered by an exact parity test.

The four frozen production files are byte-for-byte identical to `origin/main`
after Git normalization. Their Git blob IDs are:

- `grpo_sid.py`: `27d18354d920039fe9e00d803ebeb0fb9ab062fb`
- `grpo_trl_trainer.py`: `a3b89a27b4e88a57a301f60245505baf3e8f8294`
- `run_grpo_trl_smoke.py`: `d52a8d7a5cad06b23f4670d158e3d7e3923ed972`
- `run_grpo_trl_train.py`: `a8a375df09fa95169a4d3f9242c00e8cfc0f5dc7`

Deleting the ablation directory and this document fully removes DSR; the
baseline execution path is unaffected. The train runner rejects checkpoint
resume and only accepts `GR-REC-DSR-V1-*` tags. Both runners use the original
BATA adapter configured by the baseline loader.

## B. Algorithm

For grounded evidence sets `E_i`, `N_g` grounded bullets, and mean pairwise
Jaccard `J_mean`:

```
C(N_g) = 0 (N_g<2), 1 (2<=N_g<=4), 0.5 (N_g>4)
D_cot = 1-J_mean if N_g>=2 else 0
S_cot = C(N_g) * (0.5 + 0.5*D_cot)
S_A = mean_gold_A sqrt(n_a/32)
S_AB = mean_gold_AB sqrt(n_ab/32)
S_prefix = 0.33*S_A + 0.67*S_AB
H_norm = H(A)/log(32), or 0 when <=1 valid same-domain Beam
S_explore = min(U_A/8, 1) * H_norm
S_dead = S_cot * S_explore
```

The per-group Think branch is `S_cot` when primary std is nonzero;
`0.60*S_cot + 0.40*S_dead` for exact `[0x4]`; `0.40*S_cot +
0.60*S_prefix` for zero-std nonzero/no-Exact/any-prefix; otherwise `S_cot`.
`A_aux=(S_aux-mean)/(population_std+1e-4)`, with exact zero when std is zero.
`L_Think_aux` is the same clipped PPO token surrogate and completion-mask
reduction as primary GRPO, with `A_aux` replacing primary advantage.

For an exact NoThink `[0x8]`, `c=max_a freq(a)/8`, `w_i=freq(a_i)/8`,
`lambda_A=0.10` for fewer than three unique gold A values and `0.20`
otherwise. At the located sampled A token only:

```
UL_i = -log1p(-clamp(exp(logp_i), max=1-1e-6))
L_NoThink_rescue = lambda_A * c * mean_i(w_i * UL_i)
L_DSR = L_primary + 0.10*L_Think_aux + L_NoThink_rescue
```

Think and NoThink auxiliary terms are mutually exclusive by route. The same
policy forward supplies all log probabilities; DSR adds no training forward,
generation, Beam call, synchronization, sampling change, or KL/beta change.

## C. Parser

The parser is deterministic CPU regex/state slicing. It accepts the specified
heading and numbering variants, stops at `</think>` or the next major heading,
and retains fabricated SIDs only in the audit field. Only complete SIDs also
present in the prompt can ground a bullet or affect diversity.

One real smoke CoT produced four interests: `男士服饰鞋履`, `短剧/影视娱乐`,
`男性兴趣与个人护理`, and `生活与居家消费`. The compact parser result was:

```json
{
  "section_found": true,
  "parser_success": true,
  "bullet_count": 4,
  "grounded_count": 3,
  "numbering": [1, 2, 3, 4],
  "grounded_evidence": [
    ["<|ad_begin|><s_a_1375><s_b_5419><s_c_3026>", "<|ad_begin|><s_a_7879><s_b_3932><s_c_939>"],
    ["<|prod_begin|><s_a_6716><s_b_3453><s_c_2212>", "<|prod_begin|><s_a_6983><s_b_3497><s_c_288>", "<|prod_begin|><s_a_370><s_b_4782><s_c_288>"],
    ["<|prod_begin|><s_a_7873><s_b_5394><s_c_211>", "<|prod_begin|><s_a_1480><s_b_2289><s_c_7420>", "<|prod_begin|><s_a_7618><s_b_6764><s_c_5926>"]
  ],
  "D_cot": 1.0,
  "S_cot": 1.0,
  "S_prefix": 0.088388,
  "S_explore": 0.175
}
```

The ungrounded first bullet had no SID. A fabricated fourth SID in the third
bullet was recorded but excluded from grounded evidence and diversity.

## D. Think tests

PASS: primary-variance uses only `S_cot`; exact `[0x4]` selects dead-zero;
zero-std positive/no-Exact/any-prefix selects prefix rescue; saturated high
does not rescue; equal auxiliary scores yield exactly zero advantage; prefix
support is concave under dispersed hits; high wrong-A Beam diversity with
`S_cot=0` yields `S_dead=0`. Parser cases also PASS for 4/2/1/>4 interests,
missing section, malformed numbering, all heading/number styles, no SID,
fake SID, repeated/disjoint evidence, `</think>` truncation, and next section.

## E. NoThink tests and real probe

PASS: non-all-zero gate-off; 8/8, 4+4, and 8-unique frequency/concentration;
sparse `lambda_A=0.10`; dense `lambda_A=0.20`; stable unlikelihood; and active
rescue under zero primary advantage. Autograd proves the rescue gradient is
nonzero only at selected A positions and exactly zero at all other completion
positions.

The real tokenizer audit covered `<s_a_0>` through `<s_a_8191>`: all 8192 are
single tokens. A 4xA800, zero-update targeted BATA probe then sampled a real
all-zero group:

```
rewards       = [0,0,0,0,0,0,0,0]
predicted A   = [3835,5832,6358,1967,3835,1277,741,3835]
weights       = [0.375,0.125,0.125,0.125,0.375,0.125,0.125,0.375]
concentration = 0.375; gold A = [4902]; lambda_A = 0.10
A positions   = [13,13,13,13,13,13,13,13]
active        = true; rescue loss = 0.002921235989117407
```

In the same probe a second group contained two `-0.25` rewards and correctly
remained inactive, despite repeated A values. The probe performed no optimizer
step or model update; its diagnostic forward is not part of the training path.

## F. Baseline parity

PASS, exact: with Think and NoThink coefficients set to zero, primary rewards,
advantages, scalar loss, and every tested gradient element equal the frozen
baseline. Primary reward capture wrappers also return the original values
unchanged. The clean isolated branch rerun, including the formal-run contract
tests, is **23/23 PASS**.

## G. 4xA800 smoke

Run `GR-REC-DSR-V1-SMOKE-20260818-01` completed 12 optimizer steps from BATA
on four NVIDIA A800-SXM4-80GB GPUs with routes `T,T,N,N,N,N`. Think rollout
primary means were `3.0625` and `2.44727`; parser success was `0.9375/0.875`,
grounded counts `3.3125/2.875`, `S_cot` means `0.93681/0.875`, and dead-zero
branches were exercised `1/2` groups.

Representative updated-policy losses: rollout 1 primary `-0.00129031`, raw
Think aux `0.000858426`, total `-0.00120446`; rollout 2 primary `0`, raw aux
`-0.000296980`, total `-0.0000296980`. Ratios were `1.00142/1.00341`, clip
fractions `1.72%/2.56%`, and approximate KL `0.001657/0.001948`.

Main-smoke NoThink batches had no exact all-zero group, so rescue stayed zero;
the targeted probe above supplies live active and inactive gate coverage.
LoRA norm delta was `0.000122520`; frozen base delta was exactly `0`. Beam
invalid count was `0`, closure was `100%`, dynamic group sizes stayed Think=4
and NoThink=8, and no NaN/Inf, OOM, assertion, or collective failure occurred.
Peak allocated/reserved memory was `46115/47718 MiB`; GPUs returned to `5 MiB`.

## H. Performance

Total wall time was `269.696 s` versus the comparable frozen-baseline 12-step
smoke's `307.720 s` (`-12.36%`). This is generation variance, not a speedup
claim. Mean policy-forward time was `0.6208 s` versus `0.6317 s` baseline;
Think-only was `0.9525 s` versus `0.9750 s`. The objective therefore showed
no measurable forward overhead, while generation and Beam still dominate.

## I. Risks

- Parser reward hacking remains possible by copying real prompt SIDs into
  semantically empty bullets; strict grounding blocks fake SIDs but not copying.
- Beam exploration can rotate among wrong A values; `S_cot` gating limits but
  cannot prove semantic quality.
- Wrong-A unlikelihood may shift mass to a different wrong A rather than a gold
  A because this ablation intentionally has no teacher forcing.
- Group-normalized auxiliary advantages can be O(1); fixed `0.10`, separate
  normalization, and zero-std gating limit but do not eliminate interference.
- The sparse/dense `lambda_A` split is heuristic and needs outcome-level pilot
  comparison for fairness across gold-set cardinalities.
- Twelve steps cover mechanics and stability, not convergence or long-horizon
  distribution drift.

## J. Smoke-era decision

- Think DSR: **keep, more validation**. Live branches, gradients, and short-run
  stability pass, but two Think rollouts are insufficient for efficacy.
- NoThink rescue: **keep, more validation**. A real all-zero group activated
  exactly at A positions and a near-miss gated off, but behavior after repeated
  updates is untested.
- Whole `GR_REC_DSR_Ablation_v1`: **ready for a bounded pilot**, not a formal or
  full-epoch run. No long training was started.

## K. Pilot-preflight Git and Probe contract

The final acceptance worktree is based on `origin/main` parent
`9d1791ca1ca572d23b15013fdedab4612e70d83c` on branch
`ablation/gr-rec-dsr-v1`. The isolated implementation commit is
`fd12b392f239b263755512bba67f2b7703b4c267`; it contains only this document
and `grpo/ablations/gr_rec_dsr_v1/`.

The formal DSR runner now fail-closes all fairness-sensitive arguments. It
forces train seed `20260816`, probe seed `20260818`, probe interval 200, and
the exact ordered probe groups:

1. video: `fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e`
2. living: `6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f`
3. prod: `281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700`
4. ad: `2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8`

All four are permanently excluded from training. Think remains `G=4`,
NoThink remains `G=8`, and generation/Beam32 stay in the frozen baseline.
The baseline evaluator snapshots and restores Python, CPU Torch, CUDA, and
route state around probes, so probing does not advance later training RNG.

## L. Deterministic train schedule

The DSR formal runner's actual sampler produced 1,549 raw groups, excluded all
four probes, retained 1,545 candidates, dropped one tail group, and trains
1,544 groups. The dropped group is
`c399e01d9eaa30e416df6c29ee118db4b9a6361bf558433439ccec27500e0d46`.
Coverage is 386 Think rollouts, 772 NoThink rollouts, and 2,316 optimizer
steps. Generation routes begin `T,T,N,N,N,N` and repeat.

- candidate order SHA-256: `799a2b7fc5bb01693065e7ec69c590bf3c4598c94ed1baa4b9fbb09790da8127`
- trained group IDs SHA-256: `8d62e95b231d1034344451dcb2f704a04879caae54af2f93225b51f3d7f90dcd`
- generation schedule SHA-256: `ab973b28debb650cf07750e5fbcd7767fe9ebc7a292dcca6fcd757f2bdc7cd21`
- optimizer schedule SHA-256: `ac86490e5660e365fe4adedb6ac9321cffb95db1636b2f4bafad99841ba508b7`

The candidate-order and optimizer-schedule hashes are exactly equal to a
fresh `GR_REC_v1` construction under the same settings.

## M. S_prefix scale audit

The offline audit included all 292 candidates carrying Beam32 data. Means by
`domain|Gold_A bucket` are shown as `S_A / S_AB / S_prefix`:

| Stratum | n | Means |
| --- | ---: | --- |
| ad\|1 | 8 | 0.871005 / 0.292164 / 0.483182 |
| ad\|2 | 56 | 0.040568 / 0.008123 / 0.018830 |
| ad\|3-4 | 4 | 0.214676 / 0.071129 / 0.118500 |
| living\|2 | 56 | 0.022240 / 0 / 0.007339 |
| living\|3-4 | 4 | 0.102062 / 0 / 0.033680 |
| prod\|1 | 8 | 0.428065 / 0.022097 / 0.156067 |
| prod\|2 | 68 | 0.142476 / 0.065294 / 0.090764 |
| video\|5+ | 88 | 0.078805 / 0.026687 / 0.043886 |

Sparse ad/prod one-gold strata are materially larger than dense video, but
domain, gold cardinality, and model correctness are confounded in this trace;
this is a pilot risk rather than evidence for changing the formula here.

## N. Parser reward-hacking audit

Across 50 Think CoTs (8 DSR smoke, 32 frozen probes, 10 formal-train traces),
parser success was 94%. Raw interests were mean/median/p90 `3.44/4/4` and
grounded interests `3.24/3/4`. Evidence SIDs per bullet were `3.26/3/6`;
unique grounded SIDs per completion were `10.84/10.5/17`. Cross-bullet SID
reuse was mean `0.00368`, median/p90 zero, maximum `0.125`; no completion
repeated a normalized title. Fake evidence was 3.03% of observed evidence and
did not ground rewards.

Two of 50 completions were flagged for possible mechanical copying. The fixed
video probe candidate 0 reused
`<|prod_begin|><s_a_3244><s_b_91><s_c_5404>` across bullets; train group
`1e69a31d62cece5fd7a4f6155c0dc1f441d5f7b7609287c44f893e8a2c0fa63d`
candidate 2 reused `<|prod_begin|><s_a_2395><s_b_5451><s_c_5234>`.
Grounding blocks fabricated SIDs, but it does not prevent semantically empty
mechanical reuse of real prompt SIDs.

## O. Real 8B+BATA gradient budget

The final audit loaded the real 8B base and original BATA adapter on four
A800s, enabled only LoRA parameters, created no optimizer, and performed zero
optimizer steps. It generated 16 fresh Think groups: three sampler-selected
stratified train groups plus the fixed probe for each domain. Nine were
NORMAL-SIGNAL, four were DEAD-ZERO, and three were zero-std nonzero cases.

Across all nine NORMAL-SIGNAL groups, aggregate LoRA
`||0.10*g_aux||/||g_primary||` was mean `0.054916`, median `0.092547`, p90
`0.104027`, and max `0.104027`. Primary norms ranged `0.816646-0.922524`;
weighted auxiliary norms ranged `0-0.095967`. Four normal groups had exactly
zero auxiliary gradient because their within-group `S_aux` was constant. For
the five nonzero cases, cosine mean/median/min was
`-0.193800/-0.459247/-0.999920`. Thus the auxiliary does not overpower the
primary by norm at lambda 0.10, although individual groups can strongly oppose
or align with it and must be monitored in a pilot.

Representative normal groups:

| Group/domain | `||g_P||` | `||0.10*g_T||` | ratio | cosine |
| --- | ---: | ---: | ---: | ---: |
| `912228...` video | 0.910340 | 0.090940 | 0.099897 | -0.999920 |
| `0d3b5e...` living | 0.922524 | 0.095967 | 0.104027 | 0.096985 |
| `6d525c...` prod | 0.894388 | 0.082773 | 0.092547 | -0.459247 |

All four real DEAD-ZERO groups had `||g_P||=0` and nonzero weighted auxiliary
norms in `0.082012-0.091948`, with `S_aux` std in `0.038785-0.306002`.
This is the intended rescue behavior; no primary/aux ratio is interpreted for
these groups. The LoRA-A and LoRA-B layers both received rescue gradient.

Primary, auxiliary, and total were independently backpropagated after
`zero_grad(set_to_none=True)` from the same forward graph. In bf16 LoRA
gradients, `g_total` versus `g_P+g_T` relative norm error was mean 0.94% and
maximum 1.32%, consistent with bf16 gradient accumulation/rounding and with no
detach, double scaling, or extra accumulation detected.

## P. NoThink budget and route weighting

The exact raw completion IDs from a fresh zero-update BATA probe reproduced a
real all-zero `G=8` group. Predicted A values were
`[3835,5832,6358,1967,3835,1277,741,3835]`; concentration and maximum repeated
wrong-A frequency were both `0.375`, `K_A=1`, `lambda_A=0.10`, and the combined
coefficient was `0.0375`. Rescue loss was `0.0028283` with
`||g_P||=0`, `||g_N||=0.091107`, and eight active positions. The nonzero token
gradient set exactly equaled the eight sampled s_A positions; all other
completion positions were zero. LoRA-A/B norms were `0.035148/0.084054`.

The inactive real group with rewards
`[0,-0.25,0,0,0,-0.25,0,0]` produced coefficient zero and no rescue. A separate
logit-level autograd check gave `dL_UL/dlogit_wrong=+0.158277`; one gradient
descent step reduced wrong-A probability from `0.158277` to `0.154990`.

The actual code-level losses at gradient accumulation 1 are:

```
L_total_Think   = 1.0 * L_primary_unweighted + 0.10 * L_Think_aux
L_total_NoThink = 0.5 * L_primary_unweighted + L_NoThink_rescue
```

The NoThink rescue is not multiplied by route weight 0.5. This matches the
implemented ablation specification: route weighting belongs to the frozen
primary objective, while the rescue already carries `lambda_A*c`. It is not a
silent double scaling or bypass; changing it would define a different
coefficient and was not authorized here. For the all-zero case,
`g_total` versus `g_N` relative error was 0.44% in bf16.

## Q. Pilot-preflight decision

Baseline parity is exact at zero treatment coefficient, all 23 DSR CPU tests
pass, the baseline fixed-probe and formal-run CPU suites pass, schedule parity
is exact, and the real gradient audit found no auxiliary norm overpowering on
normal-signal groups. Parser copying, sparse/dense prefix scale, negative
gradient cosine, and wrong-A rotation remain bounded-pilot monitoring risks.

Final decision: **READY FOR 400-STEP PILOT**. Status remains
**IMPLEMENTED / SMOKE ONLY** until such a pilot is separately authorized and
completed. No pilot or full epoch was started during this acceptance.
