# GR_USER_v1 reward contract

Contract version: `gr_user_reward_v1`

This document separates facts supported by available evaluation evidence from
local approximations. The implementation is a deterministic CPU layer. It does
not generate text and is not integrated with a trainer or token-level loss.

## Evidence boundary

Available evidence identifies the evaluation task as
`challenge_evolution_topic_gen` and records the evaluator class name
`EvolutionTopicGenEvaluator`. Neither the server nor the repository contains
that evaluator's source. Therefore its exact action tokenizer, tie-breaking,
and whether Action and Logic use shared matches are **unknown**.

The following behavior is benchmark-grounded by the supplied task contract:

- Action is evaluated as set precision/recall/F1 over valid SIDs.
- Chain uses ordered one-to-one matching rather than an unordered assignment.
- Chain has separate Action and Logic alignment values.
- Logic similarity combines token F1 and ROUGE-L-F1 with equal weight.
- Dates are constraints and do not directly lower the Chain alignment reward.

The following behavior is an explicit approximation:

- Text normalization is Unicode NFKC plus lowercase.
- Action/logic tokens are complete SID strings, lowercase ASCII words, and
  individual Han characters; punctuation and whitespace are ignored.
- Ordered dynamic programming maximizes summed Action token-set F1.
- Logic is scored on the same action-selected pairs.
- DP ties prefer a positive match, then skipping a predicted event.

These choices are deliberately isolated in `user_common.py` and
`user_chain_reward.py` so an official evaluator implementation can replace them
without changing constraints or token attribution.

## Action reward

The accepted completion is exactly one JSON array after an optional final
`</think>` marker. Leading/trailing whitespace is accepted. Any prose before or
after JSON, malformed JSON, or non-array root makes the main reward zero. A
reliably parsed array may contain non-string, non-SID, or malformed-SID
elements; those elements are diagnosed and excluded while valid SID elements
still enter Set-F1.

For a valid array, duplicate SIDs are removed for metric calculation:

```text
P = unique valid predicted SIDs
G = unique gold SIDs
precision = |P intersect G| / |P|
recall = |P intersect G| / |G|
reward = harmonic_mean(precision, recall)
```

Diagnostics retain every occurrence. The second and each later occurrence of a
SID receives a `duplicate_sid` violation. A predicted SID in history but not
Gold is `wrong_selection_sid`. A valid SID absent from history is
`hallucinated_sid`.

Hallucinated SIDs also receive prefix attribution. Components are checked in
the order domain, `s_a`, `s_b`, `s_c` against prefixes represented in history.
The first absent prefix is reported as `first_invalid_component`; that component
and every suffix component are listed in `penalized_components`.

## Chain reward

The accepted completion is exactly one JSON object with
`logic_chain.events`. Each event must contain string fields `date`, `action`,
and `logic`; date format is `YYYY-MM-DD`.

The Action similarity matrix is approximate token-set F1. A monotonic
one-to-one dynamic program chooses non-crossing positive-similarity pairs that
maximize total Action similarity. For each selected pair:

```text
logic_similarity = 0.5 * logic_token_set_f1 + 0.5 * logic_rouge_l_f1
chain_reward = 0.5 * action_alignment_f1 + 0.5 * logic_alignment_f1
```

Soft matched similarity mass is converted to precision, recall, and F1 using
the predicted and Gold event counts. Zero-similarity pairs remain unmatched.

## Chain constraints

Grounding follows a strict hierarchy for each semicolon-separated action:

1. If the action has a SID, every SID must occur in history. Failure emits only
   `hallucinated_sid` at this level.
2. A known SID must occur on the predicted date. Failure emits
   `date_mismatch`, with no additional action mismatch for the same failure.
3. Date and SID must resolve to an exact history action string. Failure emits
   `action_mismatch`.
4. A text-only action must exactly match history and then match the predicted
   date.

Event grounding is `grounded`, `partially_grounded`, or `ungrounded`. Other
constraints are:

- `duplicate_event`: second and later normalized `(date, action)` occurrence;
- `chronology_violation`: a date lower than the preceding valid date;
- `excess_event`: each event after index 4;
- schema and strict-format violations.

Dates and constraints are diagnostic only in this CPU layer. No undocumented
penalty is subtracted from the alignment reward.

## Span attribution

Every `ConstraintViolation` contains route, kind, message, character start/end,
severity, metadata, and optional token start/end/IDs. Character intervals are
half-open and local to the completion.

Token spans use the frozen parent checkpoint's fast tokenizer and its exact
`offset_mapping`. The audit rejects a tokenizer without offsets. It verifies
100 real SIDs for encode/decode roundtrip and non-empty domain/`s_a`/`s_b`/`s_c`
component spans.

The controlled locality audit covers both hallucination and duplicate
perturbations for at least 50 real Action and 50 real Chain samples. Duplicate
attribution selects only the second occurrence (the second complete event for
Chain), while hallucination attribution selects only the injected SID.

Malformed output still produces recoverable SID diagnostics where possible,
but recoverable content never contributes to the strict main reward.

## Non-goals

This phase does not define GRPO reward weights, convert violations into a loss,
launch inference, occupy a GPU, start training, modify GR_REC, or rebuild the
frozen datasets. Those integration decisions remain intentionally open.
