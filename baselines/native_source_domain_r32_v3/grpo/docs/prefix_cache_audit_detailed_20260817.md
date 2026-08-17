# Beam32 Prefix-KV Group-Level Equivalence Audit

Date: 2026-08-17. Experiment: `REC-MP-GRPO-v1`. Target branch:
`agent/grpo-rec-mp-v1-sync`.

## Decision

```text
BEAM PREFIX CACHE REJECT
```

Prefix-KV is materially faster, but it changes the GRPO group-level learning
signal. Production remains on native full-prefix Beam32 with
`BEAM_CONTEXT_BATCH=1`. No training, backward pass, or optimizer step was run.

## Method

The main audit used 32 real Think groups, G=4 sampled CoTs per group, and the
same 128 CoTs for baseline and Prefix-KV. Seed was `20260816`; sampling used
temperature `0.9`, top-p `0.95`, max completion `2048`, and token-id stopping
at `</think>`.

Both paths held Beam semantics constant: `num_beams=32`,
`num_return_sequences=32`, `max_new_tokens=128`, `do_sample=False`, and the
same reward parser. CUDA synchronization surrounded every timed GPU operation.
Candidate outputs used identical slicing and trailing-pad canonicalization.

Baseline used production `generate_batch` with cbs=1. Prefix-KV prefilled
`ctx[:-1]` once, repeated the cache 32 ways, then called native HF Beam32.

## Main results

| Metric | Result |
|---|---:|
| Candidate token-id parity | 0/128 |
| SID-set parity | 0/128 |
| Raw reward parity | 97/128 (75.78%) |
| Reward mismatch | 31/128 |
| Reward bias / MAE / max diff | +0.1356 / 0.4367 / 7.75 |
| Exact / AB / A aggregate | 44=44 / 28->27 / 96->88 |
| Invalid aggregate | 14=14 |
| Argmax agreement | 18/32 (56.25%) |
| Full ranking agreement | 18/32 (56.25%) |
| Tie-aware top-2 agreement | 22/32 (68.75%) |
| Advantage sign agreement | 87/128 (67.97%) |
| Advantage mean abs / max diff | 0.39286 / 2.73085 |
| Pearson surrogate | 0.55664 |
| Learning-signal distortion | 50.2867 |

Advantage cosine for 15 non-zero-std comparable groups was mean `0.705921`,
p50 `1.000000`, p10 `-0.567769`, and minimum `-0.577350`.

Zero-std state counts were: baseline zero -> cache mixed 5, baseline mixed ->
cache zero 3, both zero 9, both mixed 15. Advantage sign flips were 4
positive-to-negative and 5 negative-to-positive.

## Determinism

The first 8 CoTs were rerun through both paths:

| Check | Baseline | Prefix-KV |
|---|---:|---:|
| Candidate token IDs | 8/8 | 1/8 |
| SID candidates | 8/8 | 1/8 |
| Reward | 8/8 | 8/8 |

Prefix-KV reward was stable in this small rerun, but candidate and SID outputs
were not stable. This fails the candidate-level determinism requirement.

## Performance

All 128 main-audit contexts were below 3000 tokens. Typical-context timing:

| Path | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Baseline Beam32 | 8.58s | 8.25s | 12.06s | 13.02s | 13.81s |
| Prefix-KV total | 2.49s | 2.32s | 2.96s | 3.09s | 17.47s |

Prefix-KV components were prefill 0.279s, cache repeat 0.0112s, Beam 2.20s,
and post-process 0.0012s. Mean speedup was 3.44x. Peak allocated memory was
47,089 -> 30,874 MiB; peak reserved memory was 78,106 -> 74,024 MiB.

A synchronized long-context supplement used four real contexts of lengths
3699, 3773, 3664, and 3581. Results were baseline `17.20s/cot`, Prefix-KV
`3.84s/cot`, speedup `4.48x`, and peak allocated `56,030 -> 34,982 MiB`.
Long reward parity was only 3/4; one sample changed exact count 2 -> 1 and
reward `12.21875 -> 8.4375`.

## Root-cause evidence

Forward instrumentation showed Prefix-KV changes the first Beam forward from
the full-prefix baseline path to an incremental shape:

```text
Prefix prefill:     batch=1, seq=context_len-1
Beam first forward: batch=32, seq=1
```

This changes the FlashAttention2/BF16 numerical path. Early logit differences
can change Beam ordering, which propagates into candidate sets, reward,
ranking, and advantage. The observed mismatches are consistent with this
mechanism.

## ETA reference

For a 4GPU estimate, the 32 groups were partitioned into eight 4-group global
blocks and each block used the maximum rank wall time. Think block means were
baseline `106.44s` and Prefix-KV `77.09s`.

Using final-smoke baselines of 3.0s per NoThink rollout and 0.63s per policy
step:

```text
ETA = 387 * Think + 774 * NoThink + 2322 * policy_update
baseline:  44,975.5s = 12.49h
Prefix-KV:  33,619.6s =  9.34h
saving:                 3.15h
```

Reward parsing overhead was only about 0.00047-0.00049s/cot. ETA is a
performance reference only and does not justify adoption after equivalence
failure.

## Synced evidence

The branch contains the corrected audit helper, long timing helper, report,
and detailed evidence:

- `scripts/bench_prefix_cache_audit.py`
- `scripts/bench_prefix_cache_long_sync.py`
- `results/prefix_cache_audit.json`
- `results/prefix_cache_audit_details.json`
- `results/prefix_cache_audit_records.jsonl`
- `results/prefix_cache_audit_cots.json`
- `results/prefix_cache_long_sync.json`

The JSON/JSONL evidence retains per-CoT candidate IDs, parsed SIDs, rewards,
timings, memory peaks, and per-group advantage records. Do not use Prefix-KV
for formal GRPO training unless a future implementation preserves the Beam
forward semantics and passes this same group-level audit.
