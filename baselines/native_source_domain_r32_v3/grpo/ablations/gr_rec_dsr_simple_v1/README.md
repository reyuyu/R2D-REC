# GR_REC_DSR_Simple_Ablation_v1

Status: **FULL RUN IN PROGRESS**

DSR-Simple is an isolated sibling of `gr_rec_dsr_v1`. Its parent checkpoint is
the original BATA adapter, its frozen baseline is GR_REC_v1, and DSR v1 is only
a reference ablation. It never resumes either GR_REC_v1 or DSR v1.

## Think objective

For raw numbered interest count `N`:

```text
S_N = 1 if 2 <= N <= 4 else 0
D_A = min(unique_valid_target_domain_A / 8, 1)
```

The G=4 branch is exclusive:

- primary standard deviation greater than zero: `S_aux = 0` (`primary_only`)
- zero standard deviation but not all-zero: `S_aux = S_N`
  (`zero_std_count_rescue`)
- primary `[0,0,0,0]`: `S_aux = S_N + D_A`
  (`dead_zero_count_diversity_rescue`)

`A_aux` uses population standard deviation and `1e-4`; tied groups map to exact
zero. Think loss is `L_primary + 0.10 * L_simple_aux`. The surrogate, clipping,
importance-sampling semantics, completion mask, and whole-completion credit are
inherited from the tested DSR v1 trainer.

## Intentionally removed from training

Grounding, Grounded N, Coverage, evidence Jaccard/diversity, D_cot, S_cot,
S_A, S_AB, S_prefix, entropy exploration, S_dead, correct-prefix support, and
fake-SID gates never affect Simple scores, advantages, or loss. They remain
under explicit `diagnostic_only` JSON fields.

## NoThink

NoThink imports the DSR v1 reward capture, exact G=8 all-zero plan, and
repetition-aware token-local unlikelihood loss directly. Gate, lambda,
concentration, frequency weight, route weighting, and sampled-A token semantics
therefore use the same implementation.

## Frozen contract

- train seed: `20260816`
- fixed-probe seed: `20260818`
- fixed probe IDs: exact video/living/prod/ad four-group tuple from DSR v1
- schedule: raw 1549, exclude 4, candidate 1545, trained 1544, Think 386,
  NoThink 772, optimizer steps 2316
- optimizer schedule SHA-256:
  `ac86490e5660e365fe4adedb6ac9321cffb95db1636b2f4bafad99841ba508b7`

`run_simple_smoke.py` accepts at most 12 optimizer steps. The formal runner is
authorized for one continuous 2316-step run with a step-200 safety gate.

Full runs emit `simple_forensic.jsonl` from the already-gathered rank-0 Python
records. The step-200 fixed-probe callback writes `gate200_report.json`, forces
`checkpoint-200`, and continues the same Trainer unless the preregistered gate
decision is `STOP`. These paths are monitor-only and add no model, generation,
decode, CUDA synchronization, or distributed collective work.

## Known risks

1. N-only optimization may teach the model to mechanically write two to four
   bullets without improving understanding.
2. Target-domain A diversity is correctness-agnostic and may reward wrong
   exploration during dead-zero groups.
3. Whole-CoT credit assignment remains broad; interest-section token masking is
   deliberately not introduced in this ablation.
4. The unchanged NoThink rescue may rotate among several wrong A values; Gold-A
   hit and wrong-domain diagnostics must be interpreted with concentration.

## Accepted development-machine checks

CPU: 49/49 tests passed. The accepted four-A800 smoke completed 12/12 optimizer steps with the exact T,T,N,N,N,N rollout order; LoRA changed and base parameters did not. The zero-step gradient audit found 9 primary-signal groups with exact-zero Simple auxiliary gradient and 4 active rescue groups with nonzero auxiliary gradient.

The accepted smoke took 285 seconds. Its measured full-schedule estimate was 13.6 hours. The shared dashboard selects a dedicated DSR-Simple view while retaining the old DSR v1 view.

Single-node four-GPU launches on this development machine require NCCL_SOCKET_IFNAME=lo, GLOO_SOCKET_IFNAME=lo, and NCCL_IB_DISABLE=1. No pilot was started.
