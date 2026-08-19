# GR_USER_v1 token penalty mask contract

Contract version: `gr_user_penalty_mask_v1`

Phase 3A is a deterministic CPU compiler. Reward scorers continue to own
completion parsing, task reward, constraint detection, and character spans. The
compiler only transforms existing violations into boolean masks over the frozen
parent tokenizer's completion tokens. It does not define a penalty magnitude,
modify advantages, or integrate with a trainer or loss.

## API

```python
compile_penalty_mask(completion, violations, tokenizer, route)
```

The result contains:

- `penalty_mask`: union `[T]` boolean mask;
- `per_kind_masks`: one `[T]` boolean mask for each whitelisted route kind;
- `records`: included and ignored violation decisions with character and token
  spans;
- `token_count`, `input_ids`, `masked_token_count`, and
  `masked_token_fraction`.

The compiler re-tokenizes the exact completion with
`return_offsets_mapping=True`. It does not trust stored token indices from a
different completion. All token intervals are half-open `[start, end)`.

## Whitelists

Action mask kinds:

```text
hallucinated_sid
duplicate_sid
```

Action `wrong_selection_sid` and every format/schema violation are retained in
records as `not_whitelisted` and never enter a mask. Wrong selection remains a
Set-F1 concern.

Chain mask kinds:

```text
hallucinated_sid
date_mismatch
action_mismatch
duplicate_event
chronology_violation
excess_event
```

Low Action/Logic alignment and format/schema violations do not enter masks.

## Locality

Action hallucination is component-aware. The compiler validates that the
violation span and SID metadata identify the same canonical SID, then maps only
`penalized_components`:

```text
domain invalid -> domain + A + B + C
A invalid      -> A + B + C
B invalid      -> B + C
C invalid      -> C
```

Action duplicate masks the second and later complete SID occurrence. A token
may appear in both hallucination and duplicate per-kind masks; the union counts
it once.

Chain spans are compiled exactly as attributed by Phase 2:

- hallucination: SID only;
- date mismatch: date value only;
- action mismatch: incorrect semicolon-separated action part only;
- duplicate: second and later complete duplicate event;
- chronology: date causing the reversal;
- excess: sixth and later complete event.

Any whitelisted violation with an invalid or empty span causes a compiler error
instead of silently broadening the mask.

## Audit fixture

Controlled hallucinations are selected from 89,497 real SIDs in pilot history.
Every selected SID is absent from the current sample history and encodes as the
normal four parent-tokenizer components. Synthetic out-of-vocabulary SID
fixtures are prohibited.

The Action hallucination audit selects a real candidate sharing the sample's
domain/A/B prefix. Therefore its first invalid component is C and its expected
mask is exactly one token.

## Non-goals

This phase does not assign lambda values, create token advantages, alter GRPO
loss, call generation, use a GPU, launch training, or modify frozen data.
