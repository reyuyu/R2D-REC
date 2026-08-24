# Recommendation Bridge Memory Quick Probe

## Scope

This Phase 1.5-Quick diagnostic is a 16-group descriptive screen. It compares Mini-Fix and Mini-Gamma under a shared BATA-source Frozen CoT, with and without the exact BATA bridge. `TRAIN_SEEN` and `TRAIN_UNSEEN` refer only to membership in the two Mini training datasets; they do not establish official train-test overlap.

- Implementation commit: `e26bfed332a5fae5e89bc45acaac6e861db0665b`
- Groups: 16 (8 seen, 8 unseen; 4 per domain)
- New GPU cases: 96
- Decode: deterministic Beam32, exactly three ABC tokens
- Invalid beams: 0 / 3072
- Training, Self-CoT, FREE generation and external evaluation: not run

## Primary Results

| Model | Membership | Bare MRR | Bridge MRR | Bridge gain MRR | 1 - Jaccard | Top-1 flip |
|---|---|---:|---:|---:|---:|---:|
| Mini-Fix | TRAIN_SEEN | 0.02652 | 0.15030 | +0.12378 | 0.79832 | 0.87500 |
| Mini-Fix | TRAIN_UNSEEN | 0.00000 | 0.00417 | +0.00417 | 0.83559 | 0.87500 |
| Mini-Gamma | TRAIN_SEEN | 0.01786 | 0.15924 | +0.14138 | 0.59410 | 0.87500 |
| Mini-Gamma | TRAIN_UNSEEN | 0.00000 | 0.00781 | +0.00781 | 0.59214 | 0.75000 |

Paired 5,000-resample descriptive bootstrap intervals:

| Model | Membership | Bridge gain MRR CI95 | 1 - Jaccard CI95 |
|---|---|---:|---:|
| Mini-Fix | TRAIN_SEEN | [-0.00564, 0.35078] | [0.70272, 0.88451] |
| Mini-Fix | TRAIN_UNSEEN | [0.00000, 0.01250] | [0.71454, 0.93245] |
| Mini-Gamma | TRAIN_SEEN | [0.00456, 0.39138] | [0.48896, 0.70202] |
| Mini-Gamma | TRAIN_UNSEEN | [0.00000, 0.02344] | [0.46032, 0.73613] |

## Native Prefix Check

The original BATA generic prompt was mechanically recoverable, and all selected seen prefixes matched a Mini-Fix Think training prefix. Mini-Fix native-prefix bridge gain MRR was `+0.14600` for seen groups and `0.00000` for unseen groups. Relative to the official-prefix bridge MRR, the native-prefix boost was only `+0.00612` seen and `-0.00417` unseen. This does not provide a strong train-prefix memorization-key signal.

## Interpretation

The gold-ranking signal is strongly seen-dependent, but this pattern is not Mini-Fix-specific: Gamma has the same and slightly larger seen/unseen MRR contrast. Consequently, the screen does not isolate the old bridge training contract as a Mini-Fix-only retrieval key.

The full response distribution gives a different signal. Mini-Fix retains a large unseen bridge response (`1 - Jaccard = 0.83559`), essentially as large as its seen response and materially larger than Gamma's unseen response (`0.59214`). The bridge therefore causes a systematic, model-specific redistribution on groups absent from both Mini training datasets. In this small screen, that redistribution does not translate into meaningful unseen gold-ranking improvement.

Decision labels:

- `MEMORY_KEY_HYPOTHESIS_SUPPORT=WEAK_OR_UNSUPPORTED`
- `GENERALIZABLE_BRIDGE_PRIOR_SUPPORT=MODERATE`
- `GENERIC_BRIDGE_CONDITIONING_SUPPORT=WEAK_OR_UNSUPPORTED`
- `HISTORY_RETRIEVAL_SHORTCUT_SUPPORT=NO_OR_PARTIAL`
- `ROOT_CLASS=SCREEN_SUPPORTS_GENERALIZABLE_PRIOR`

The history conclusion is especially weak because only one seen and one unseen group have GoldSIDInHistory. No formal significance claim is made. The next step is intentionally stopped pending review; no FREE continuation experiment was started.
