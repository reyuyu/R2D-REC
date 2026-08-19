"""Compile attributed User GRPO violations into token-local boolean masks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from user_span_attribution import TokenSpanMapper


ACTION_MASK_KINDS = frozenset({"hallucinated_sid", "duplicate_sid"})
CHAIN_MASK_KINDS = frozenset(
    {
        "hallucinated_sid",
        "date_mismatch",
        "action_mismatch",
        "duplicate_event",
        "chronology_violation",
        "excess_event",
    }
)
ROUTE_MASK_KINDS = {"action": ACTION_MASK_KINDS, "chain": CHAIN_MASK_KINDS}
SID_COMPONENT_RE = re.compile(
    r"(?P<domain><\|(?:video|prod|ad|living)_begin\|>)"
    r"(?P<a><s_a_\d+>)(?P<b><s_b_\d+>)(?P<c><s_c_\d+>)"
)


class PenaltyMaskCompileError(ValueError):
    pass


@dataclass(frozen=True)
class _ViolationView:
    kind: str
    route: str
    char_start: int
    char_end: int
    metadata: dict[str, Any]


def _field(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _view(item: Any) -> _ViolationView:
    return _ViolationView(
        kind=str(_field(item, "kind", "")),
        route=str(_field(item, "route", "")),
        char_start=int(_field(item, "char_start", -1)),
        char_end=int(_field(item, "char_end", -1)),
        metadata=dict(_field(item, "metadata", {}) or {}),
    )


def _action_hallucination_spans(completion: str, item: _ViolationView) -> list[tuple[int, int, str]]:
    if item.char_start < 0 or item.char_end > len(completion) or item.char_end <= item.char_start:
        raise PenaltyMaskCompileError("hallucinated SID has an invalid character span")
    raw_sid = completion[item.char_start : item.char_end]
    sid = item.metadata.get("sid")
    if sid != raw_sid:
        raise PenaltyMaskCompileError("hallucinated SID metadata does not match its completion span")
    match = SID_COMPONENT_RE.fullmatch(raw_sid)
    if match is None:
        raise PenaltyMaskCompileError("hallucinated SID span is not one canonical SID")
    components = item.metadata.get("penalized_components")
    if not isinstance(components, list) or not components:
        raise PenaltyMaskCompileError("Action hallucination lacks penalized_components")
    invalid = [name for name in components if name not in ("domain", "a", "b", "c")]
    if invalid:
        raise PenaltyMaskCompileError(f"unknown SID components: {invalid}")
    spans = []
    for name in components:
        start, end = match.span(name)
        spans.append((item.char_start + start, item.char_start + end, name))
    return spans


def compile_penalty_mask(
    completion: str,
    violations: Iterable[Any],
    tokenizer: Any,
    route: str,
) -> dict[str, Any]:
    """Return route-whitelisted per-kind and union masks over completion tokens."""
    if route not in ROUTE_MASK_KINDS:
        raise PenaltyMaskCompileError(f"unknown route: {route}")
    mapper = TokenSpanMapper(tokenizer, completion)
    token_count = len(mapper.input_ids)
    whitelist = ROUTE_MASK_KINDS[route]
    per_kind_masks = {kind: [False] * token_count for kind in sorted(whitelist)}
    records: list[dict[str, Any]] = []

    for index, raw_item in enumerate(violations):
        item = _view(raw_item)
        if item.route != route:
            raise PenaltyMaskCompileError(
                f"violation route mismatch at index {index}: expected {route}, got {item.route}"
            )
        if item.kind not in whitelist:
            records.append(
                {
                    "violation_index": index,
                    "kind": item.kind,
                    "route": route,
                    "included": False,
                    "reason": "not_whitelisted",
                    "char_span": [item.char_start, item.char_end],
                    "token_spans": [],
                    "masked_token_indices": [],
                }
            )
            continue

        if route == "action" and item.kind == "hallucinated_sid":
            char_spans = _action_hallucination_spans(completion, item)
        else:
            if item.char_start < 0 or item.char_end > len(completion) or item.char_end <= item.char_start:
                raise PenaltyMaskCompileError(f"{item.kind} has an invalid character span")
            char_spans = [(item.char_start, item.char_end, None)]

        token_spans = []
        token_indices: set[int] = set()
        for char_start, char_end, component in char_spans:
            span = mapper.map(char_start, char_end)
            if span.end <= span.start:
                raise PenaltyMaskCompileError(f"{item.kind} maps to an empty token span")
            indices = list(range(span.start, span.end))
            token_indices.update(indices)
            token_spans.append(
                {
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": span.start,
                    "token_end": span.end,
                    "token_ids": list(span.token_ids),
                    "component": component,
                }
            )
        for token_index in token_indices:
            per_kind_masks[item.kind][token_index] = True
        records.append(
            {
                "violation_index": index,
                "kind": item.kind,
                "route": route,
                "included": True,
                "reason": "whitelisted",
                "char_span": [item.char_start, item.char_end],
                "token_spans": token_spans,
                "masked_token_indices": sorted(token_indices),
            }
        )

    penalty_mask = [
        any(per_kind_masks[kind][token_index] for kind in per_kind_masks)
        for token_index in range(token_count)
    ]
    return {
        "route": route,
        "token_count": token_count,
        "input_ids": mapper.input_ids,
        "penalty_mask": penalty_mask,
        "per_kind_masks": per_kind_masks,
        "records": records,
        "masked_token_count": sum(penalty_mask),
        "masked_token_fraction": sum(penalty_mask) / token_count if token_count else 0.0,
    }
