# GR_USER_v1 token advantage offline simulation

Source run: `GR-USER-G4-AUDIT-4GPU-20260819-181843`

No generation, GPU, model loading, training, or Trainer integration was used.

## Formula

`A_i = (R_i - group_mean) / (group_population_std + 1e-4)`. Zero-std groups use `A_i=0`.
Unmasked tokens retain `A_i`; masked tokens use `min(A_i, -lambda_eff)`.
Fixed uses `lambda_eff=lambda`; sqrt uses `lambda_eff=lambda/sqrt(span_tokens)`.
Overlaps use the strongest effective lambda and are never summed.
Positive/negative mass is `sum(abs(token_advantage))` over tokens of that sign.
Penalty negative mass is the incremental negative mass beyond the original task advantage.

## Strategy comparison

| Strategy | Lambda | Action neg/pos | Chain neg/pos | Action mean shift | Chain mean shift | duplicate_event share | Zero-std rescued |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.10 | 0.7534 | 1.0099 | 0.002030 | 0.011203 | 12.77% | 0 |
| fixed | 0.25 | 0.7541 | 1.0144 | 0.002450 | 0.013185 | 12.36% | 0 |
| fixed | 0.50 | 0.7554 | 1.0228 | 0.003153 | 0.016836 | 11.59% | 0 |
| fixed | 1.00 | 0.7584 | 1.0430 | 0.004815 | 0.025495 | 10.15% | 0 |
| sqrt | 0.10 | 0.7532 | 1.0078 | 0.001907 | 0.010302 | 5.35% | 0 |
| sqrt | 0.25 | 0.7536 | 1.0090 | 0.002141 | 0.010840 | 5.23% | 0 |
| sqrt | 0.50 | 0.7543 | 1.0111 | 0.002531 | 0.011787 | 5.07% | 0 |
| sqrt | 1.00 | 0.7556 | 1.0155 | 0.003317 | 0.013715 | 4.94% | 0 |

## Route metrics

| Strategy | Lambda | Route | Masked candidates | Masked tokens | Positive masked flips | Avg abs task A | Avg masked abs token A | Negative mass | Positive mass |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.10 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 0.534489 | 6790.886 | 9013.866 |
| fixed | 0.10 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 0.719397 | 21045.815 | 20839.879 |
| fixed | 0.25 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 0.614977 | 6797.486 | 9013.866 |
| fixed | 0.25 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 0.781955 | 21139.965 | 20839.879 |
| fixed | 0.50 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 0.751226 | 6808.659 | 9013.866 |
| fixed | 0.50 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 0.897712 | 21314.180 | 20839.879 |
| fixed | 1.00 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 1.082244 | 6835.802 | 9013.866 |
| fixed | 1.00 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 1.177345 | 21735.027 | 20839.879 |
| sqrt | 0.10 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 0.510812 | 6788.945 | 9013.866 |
| sqrt | 0.10 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 0.690689 | 21002.610 | 20839.879 |
| sqrt | 0.25 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 0.555785 | 6792.632 | 9013.866 |
| sqrt | 0.25 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 0.707521 | 21027.942 | 20839.879 |
| sqrt | 0.50 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 0.630739 | 6798.779 | 9013.866 |
| sqrt | 0.50 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 0.736950 | 21072.232 | 20839.879 |
| sqrt | 1.00 | action | 9.50% | 0.45% | 100.00% | 0.853428 | 0.782751 | 6811.244 | 9013.866 |
| sqrt | 1.00 | chain | 46.50% | 3.08% | 100.00% | 0.857509 | 0.797066 | 21162.707 | 20839.879 |

## Completion-level impact

Relative shift is `abs(mean_token_A - sequence_A) / abs(sequence_A)`. For zero sequence advantage, any nonzero shift is treated as infinite.

| Strategy | Lambda | Action >10% / >25% / >50% | Chain >10% / >25% / >50% | Action positive-to-negative | Chain positive-to-negative |
|---|---:|---:|---:|---:|---:|
| fixed | 0.10 | 1.50% / 0.50% / 0.00% | 4.00% / 1.00% / 0.50% | 0.00% | 0.50% |
| fixed | 0.25 | 1.50% / 1.00% / 0.50% | 7.50% / 2.00% / 1.00% | 0.00% | 0.50% |
| fixed | 0.50 | 1.50% / 1.00% / 1.00% | 9.00% / 4.00% / 2.00% | 0.50% | 0.50% |
| fixed | 1.00 | 1.50% / 1.00% / 1.00% | 17.50% / 4.50% / 4.00% | 1.00% | 1.00% |
| sqrt | 0.10 | 1.50% / 0.50% / 0.00% | 3.00% / 1.00% / 0.50% | 0.00% | 0.50% |
| sqrt | 0.25 | 1.50% / 1.00% / 0.50% | 4.00% / 1.00% / 0.50% | 0.00% | 0.50% |
| sqrt | 0.50 | 1.50% / 1.00% / 1.00% | 6.00% / 1.50% / 0.50% | 0.00% | 0.50% |
| sqrt | 1.00 | 1.50% / 1.00% / 1.00% | 8.00% / 2.50% / 1.00% | 1.00% | 0.50% |

## Violation kinds at recommended setting

| Route | Kind | Candidates | Violations | Masked tokens | Avg span | Penalty negative mass | Share |
|---|---|---:|---:|---:|---:|---:|---:|
| action | `duplicate_sid` | 8 | 10 | 40 | 4.00 | 6.000 | 48.81% |
| action | `hallucinated_sid` | 13 | 16 | 42 | 2.62 | 6.293 | 51.19% |
| chain | `action_mismatch` | 28 | 31 | 404 | 13.03 | 19.521 | 22.72% |
| chain | `chronology_violation` | 6 | 6 | 60 | 10.00 | 6.325 | 7.36% |
| chain | `date_mismatch` | 61 | 71 | 710 | 10.00 | 53.725 | 62.52% |
| chain | `duplicate_event` | 4 | 4 | 319 | 79.75 | 4.359 | 5.07% |
| chain | `excess_event` | 0 | 0 | 0 | 0.00 | 0.000 | 0.00% |
| chain | `hallucinated_sid` | 3 | 3 | 12 | 4.00 | 2.000 | 2.33% |

## High-reward violation examples

| Strategy | Lambda | Example | Sequence A | Span tokens / token A | Completion mean token A |
|---|---:|---|---:|---|---:|
| fixed | 0.10 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-0.1000 | 0.591577 |
| fixed | 0.10 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-0.1000 | 0.480225 |
| fixed | 0.10 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-0.1000; date_mismatch:10@-0.1000 | 0.651536 |
| fixed | 0.25 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-0.2500 | 0.589077 |
| fixed | 0.25 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-0.2500 | 0.476654 |
| fixed | 0.25 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-0.2500; date_mismatch:10@-0.2500 | 0.635829 |
| fixed | 0.50 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-0.5000 | 0.584910 |
| fixed | 0.50 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-0.5000 | 0.470701 |
| fixed | 0.50 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-0.5000; date_mismatch:10@-0.5000 | 0.609651 |
| fixed | 1.00 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-1.0000 | 0.576577 |
| fixed | 1.00 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-1.0000 | 0.458797 |
| fixed | 1.00 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-1.0000; date_mismatch:10@-1.0000 | 0.557295 |
| sqrt | 0.10 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-0.0707 | 0.592065 |
| sqrt | 0.10 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-0.0500 | 0.481416 |
| sqrt | 0.10 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-0.0316; date_mismatch:10@-0.0316 | 0.658696 |
| sqrt | 0.25 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-0.1768 | 0.590297 |
| sqrt | 0.25 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-0.1250 | 0.479630 |
| sqrt | 0.25 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-0.0791; date_mismatch:10@-0.0791 | 0.653729 |
| sqrt | 0.50 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-0.3536 | 0.587351 |
| sqrt | 0.50 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-0.2500 | 0.476654 |
| sqrt | 0.50 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-0.1581; date_mismatch:10@-0.1581 | 0.645451 |
| sqrt | 1.00 | `high_f1_hallucination` | 0.603299 | hallucinated_sid:2@-0.7071 | 0.581459 |
| sqrt | 1.00 | `high_f1_duplicate` | 0.494377 | duplicate_sid:4@-0.5000 | 0.470701 |
| sqrt | 1.00 | `high_chain_reward_mismatch` | 0.739435 | date_mismatch:10@-0.3162; date_mismatch:10@-0.3162 | 0.628895 |

## Zero-std groups

- Action: 2 total, 0 with constraint signal
- Chain: 0 total, 0 with constraint signal

## Span audit

The real rollout contains no 1-token Action hallucination. Observed Action hallucination spans are 2-3 tokens.
Action duplicate spans are 4 tokens. Chain duplicate_event spans average 79.75 tokens (max 86).

## Recommendation

Use **sqrt** with initial lambda **0.50**.
Sqrt normalization keeps short SID corrections meaningful while preventing roughly 80-token duplicate events from receiving linear per-token amplification. At lambda 0.50 it keeps duplicate_event at about 5.07% of Chain penalty negative mass, with 1.5% of Chain completions shifting by more than 25% and 0.5% shifting by more than 50%.

## Integrity

- Input SHA unchanged: True
- Existing group population statistics consistent: True
- Candidates/groups: 400/100
