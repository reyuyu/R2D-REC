"""Scored-rollout data contract for MC_USER_v1."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Mapping, Sequence

from user_marginal_credit import action_marginal_credit, chain_marginal_credit
from user_mc_projection import project_mc_credit_units_to_generated
from user_span_attribution import TokenSpanMapper


K = 2


def _as_id_list(ids: Any) -> list[int]:
    values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if values and isinstance(values[0], list):
        raise ValueError("each completion_ids entry must be one-dimensional")
    return [int(value) for value in values]


def _canonical_span(mapper: TokenSpanMapper, credit: Mapping[str, Any]) -> dict[str, Any]:
    span = mapper.map(int(credit["char_start"]), int(credit["char_end"]))
    return {
        "token_start": span.start,
        "token_end": span.end,
        "token_ids": list(span.token_ids),
    }


def _action_units(
    marginal_result: Mapping[str, Any],
    mapper: TokenSpanMapper,
) -> list[dict[str, Any]]:
    units = []
    for credit in marginal_result["credits"]:
        units.append(
            {
                "unit_type": "sid",
                "sid": credit["sid"],
                "occurrence": int(credit["occurrence"]),
                "delta": float(credit["delta"]),
                "credit_type": credit["credit_type"],
                "char_start": int(credit["char_start"]),
                "char_end": int(credit["char_end"]),
                **_canonical_span(mapper, credit),
            }
        )
    return units


def _chain_units(
    marginal_result: Mapping[str, Any],
    mapper: TokenSpanMapper,
) -> list[dict[str, Any]]:
    units = []
    for credit in marginal_result["credits"]:
        units.append(
            {
                "unit_type": "event",
                "event_index": int(credit["event_index"]),
                "delta": float(credit["delta"]),
                "delta_action_alignment": float(credit["delta_action_alignment"]),
                "delta_logic_alignment": float(credit["delta_logic_alignment"]),
                "credit_type": credit["credit_type"],
                "char_start": int(credit["char_start"]),
                "char_end": int(credit["char_end"]),
                **_canonical_span(mapper, credit),
            }
        )
    return units


def prepare_mc_scored_rollout(
    rows: Sequence[Mapping[str, Any]],
    completion_ids_list: Sequence[Any],
    tokenizer: Any,
    candidates_per_prompt: int = K,
) -> dict[str, Any]:
    """Decode and independently score a fixed candidate count per prompt."""

    if (
        not isinstance(candidates_per_prompt, int)
        or isinstance(candidates_per_prompt, bool)
        or candidates_per_prompt <= 0
    ):
        raise ValueError("candidates_per_prompt must be a positive integer")
    if len(completion_ids_list) != len(rows) * candidates_per_prompt:
        raise ValueError(
            "MC_USER_v1 rollout completion count must equal "
            "rows * candidates_per_prompt"
        )
    if not rows:
        raise ValueError("MC_USER_v1 rollout requires at least one prompt")
    routes = {row.get("route") for row in rows}
    if len(routes) != 1:
        raise ValueError("Action and Chain must use separate route-homogeneous batches")
    route = next(iter(routes))
    if route not in {"action", "chain"}:
        raise ValueError(f"unsupported MC_USER_v1 route: {route!r}")

    expanded_rows = [row for row in rows for _ in range(candidates_per_prompt)]
    normalized_ids = [_as_id_list(ids) for ids in completion_ids_list]
    completions: list[str] = []
    marginal_results: list[dict[str, Any]] = []
    canonical_credit_units_per_candidate: list[list[dict[str, Any]]] = []
    credit_units_per_candidate: list[list[dict[str, Any]]] = []
    projection_per_candidate: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    valid_count = 0
    marginal_scorer = action_marginal_credit if route == "action" else chain_marginal_credit

    for row, completion_ids in zip(expanded_rows, normalized_ids):
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        marginal_result = marginal_scorer(completion, row)
        if marginal_result["valid"]:
            mapper = TokenSpanMapper(tokenizer, completion)
            canonical_units = (
                _action_units(marginal_result, mapper)
                if route == "action"
                else _chain_units(marginal_result, mapper)
            )
            valid_count += 1
        else:
            canonical_units = []
        projection = project_mc_credit_units_to_generated(
            completion,
            canonical_units,
            completion_ids,
            tokenizer,
        )
        units = projection["projected_units"]
        type_counts.update(unit["credit_type"] for unit in units)
        completions.append(completion)
        marginal_results.append(marginal_result)
        canonical_credit_units_per_candidate.append(copy.deepcopy(canonical_units))
        credit_units_per_candidate.append(units)
        projection_per_candidate.append(
            {
                "projection_required": projection["projection_required"],
                "opcodes": projection["opcodes"],
                "canonical_token_count": len(projection["canonical_ids"]),
                "generated_token_count": len(projection["generated_ids"]),
            }
        )

    candidate_count = len(expanded_rows)
    statistics = {
        "candidate_count": candidate_count,
        "valid_candidate_rate": valid_count / candidate_count,
        "positive_unit_count": type_counts["positive"],
        "negative_unit_count": type_counts["negative"],
        "zero_unit_count": type_counts["zero"],
    }
    return {
        "route": route,
        "K": candidates_per_prompt,
        "token_span_space": "generated_completion_ids",
        "expanded_rows": expanded_rows,
        "completions": completions,
        "completion_ids_list": normalized_ids,
        "scores": marginal_results,
        "marginal_results": marginal_results,
        "canonical_credit_units_per_candidate": canonical_credit_units_per_candidate,
        "credit_units_per_candidate": credit_units_per_candidate,
        "projection_per_candidate": projection_per_candidate,
        "statistics": statistics,
        **statistics,
    }


__all__ = ["K", "prepare_mc_scored_rollout"]
