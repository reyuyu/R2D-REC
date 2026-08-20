#!/usr/bin/env python3
"""Audit overlapping non-zero MC credit units with the production tokenizer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from run_mc_user_real_smoke import (
    BASE_MODEL,
    TRAIN_DATA,
    TRAIN_SHA256,
    file_sha256,
    prepare_fixed_data,
    read_jsonl,
)
from user_marginal_credit import action_marginal_credit, chain_marginal_credit
from user_mc_objective import MCObjectiveError, mc_unit_credit_loss
from user_mc_rollout import _action_units, _chain_units
from user_span_attribution import TokenSpanMapper


RESULT_OUTPUT = "/data/GRPO_USER/results/mc_real_token_overlap_audit.json"
DOC_OUTPUT = "/data/GRPO_USER/docs/mc_real_token_overlap_audit.md"


def token_memberships(
    units: Sequence[Mapping[str, Any]], *, generated: bool
) -> dict[int, list[int]]:
    """Return token-to-unit memberships for non-zero credit units only."""

    memberships: dict[int, list[int]] = defaultdict(list)
    for unit_index, unit in enumerate(units):
        if float(unit["delta"]) == 0.0:
            continue
        indices = (
            [int(index) for index in unit["generated_token_indices"]]
            if generated
            else range(int(unit["token_start"]), int(unit["token_end"]))
        )
        for token_index in indices:
            memberships[token_index].append(unit_index)
    return dict(memberships)


def overlap_memberships(
    units: Sequence[Mapping[str, Any]], *, generated: bool
) -> dict[int, list[int]]:
    return {
        token_index: unit_indices
        for token_index, unit_indices in token_memberships(
            units, generated=generated
        ).items()
        if len(unit_indices) > 1
    }


def spans_overlap(unit_a: Mapping[str, Any], unit_b: Mapping[str, Any]) -> bool:
    return max(int(unit_a["char_start"]), int(unit_b["char_start"])) < min(
        int(unit_a["char_end"]), int(unit_b["char_end"])
    )


def offset_intersects(offset: Sequence[int], unit: Mapping[str, Any]) -> bool:
    start, end = int(offset[0]), int(offset[1])
    return end > int(unit["char_start"]) and start < int(unit["char_end"]) and end > start


def classify_overlap_pair(
    unit_a: Mapping[str, Any],
    unit_b: Mapping[str, Any],
    *,
    token_offset: Sequence[int] | None,
    canonical_overlap: bool,
    generated_overlap: bool,
    canonical_equals_generated: bool,
) -> str:
    if spans_overlap(unit_a, unit_b):
        return "CHAR_SPAN_OVERLAP"
    if generated_overlap and not canonical_overlap and not canonical_equals_generated:
        return "PROJECTION_BUG"
    if (
        canonical_overlap
        and token_offset is not None
        and offset_intersects(token_offset, unit_a)
        and offset_intersects(token_offset, unit_b)
    ):
        return "TOKENIZER_BOUNDARY_OVERLAP"
    return "UNKNOWN"


def summarize_candidate_overlaps(
    units: Sequence[Mapping[str, Any]], *, generated: bool
) -> dict[str, Any]:
    overlaps = overlap_memberships(units, generated=generated)
    pairs: set[tuple[int, int]] = set()
    same_sign = 0
    mixed_sign = 0
    for unit_indices in overlaps.values():
        signs = {float(units[index]["delta"]) > 0.0 for index in unit_indices}
        if len(signs) > 1:
            mixed_sign += 1
        else:
            same_sign += 1
        pairs.update(tuple(sorted(pair)) for pair in combinations(unit_indices, 2))
    return {
        "overlap_token_count": len(overlaps),
        "overlap_unit_pair_count": len(pairs),
        "max_units_per_token": max((len(value) for value in overlaps.values()), default=0),
        "same_sign_overlap_token_count": same_sign,
        "mixed_sign_overlap_token_count": mixed_sign,
        "overlaps": overlaps,
    }


def describe_unit(unit_index: int, unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "unit_index": unit_index,
        "sid": unit.get("sid"),
        "event_index": unit.get("event_index"),
        "occurrence": unit.get("occurrence"),
        "delta": float(unit["delta"]),
        "credit_type": unit["credit_type"],
        "char_start": int(unit["char_start"]),
        "char_end": int(unit["char_end"]),
        "canonical_token_start": int(unit["canonical_token_start"]),
        "canonical_token_end": int(unit["canonical_token_end"]),
        "generated_token_indices": [int(index) for index in unit["generated_token_indices"]],
    }


def audit_fixed_route(
    route: str,
    row: Mapping[str, Any],
    rollout: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    candidates = []
    for candidate_index, completion in enumerate(rollout["completions"]):
        mapper = TokenSpanMapper(tokenizer, completion)
        canonical_units = rollout["canonical_credit_units_per_candidate"][candidate_index]
        generated_units = rollout["credit_units_per_candidate"][candidate_index]
        canonical_summary = summarize_candidate_overlaps(canonical_units, generated=False)
        generated_summary = summarize_candidate_overlaps(generated_units, generated=True)
        canonical_ids = list(mapper.input_ids)
        generated_ids = rollout["completion_ids_list"][candidate_index]
        equal = canonical_ids == generated_ids
        overlap_records = []
        for token_index, unit_indices in generated_summary["overlaps"].items():
            canonical_unit_indices = canonical_summary["overlaps"].get(token_index, []) if equal else []
            pair_records = []
            for first, second in combinations(unit_indices, 2):
                canonical_overlap = first in canonical_unit_indices and second in canonical_unit_indices
                offset = mapper.offsets[token_index] if equal and token_index < len(mapper.offsets) else None
                pair_records.append(
                    {
                        "unit_indices": [first, second],
                        "classification": classify_overlap_pair(
                            generated_units[first],
                            generated_units[second],
                            token_offset=offset,
                            canonical_overlap=canonical_overlap,
                            generated_overlap=True,
                            canonical_equals_generated=equal,
                        ),
                    }
                )
            offset = list(mapper.offsets[token_index]) if equal else None
            overlap_records.append(
                {
                    "candidate_index": candidate_index,
                    "token_index": token_index,
                    "token_id": int(generated_ids[token_index]),
                    "decoded_token": tokenizer.decode(
                        [generated_ids[token_index]], skip_special_tokens=False
                    ),
                    "canonical_offset_mapping": offset,
                    "offset_text": completion[offset[0] : offset[1]] if offset else None,
                    "units": [describe_unit(index, generated_units[index]) for index in unit_indices],
                    "pair_classifications": pair_records,
                }
            )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "completion": completion,
                "canonical_equals_generated": equal,
                "projection_required": rollout["projection_per_candidate"][candidate_index][
                    "projection_required"
                ],
                "projection_opcodes": rollout["projection_per_candidate"][candidate_index]["opcodes"],
                "canonical_overlap": {k: v for k, v in canonical_summary.items() if k != "overlaps"},
                "generated_overlap": {k: v for k, v in generated_summary.items() if k != "overlaps"},
                "overlap_records": overlap_records,
            }
        )
    return {"route": route, "sample_id": row["sample_id"], "candidates": candidates}


def reproduce_objective_result(
    rollout: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Call the unchanged objective on CPU zeros to reproduce overlap validation."""

    completion_mask = batch["completion_mask"].cpu()
    logps = torch.zeros_like(completion_mask)
    try:
        _loss, metadata = mc_unit_credit_loss(
            logps,
            rollout["credit_units_per_candidate"],
            completion_mask,
        )
    except MCObjectiveError as exc:
        return {"status": "ERROR", "error": str(exc)}
    return {
        "status": "PASS",
        "active_unit_count": int(metadata["active_unit_count"]),
        "active_token_count": int(metadata["active_token_count"]),
    }


def add_neighborhood(fixed_action: dict[str, Any], tokenizer: Any) -> None:
    for candidate in fixed_action["candidates"]:
        if not candidate["overlap_records"]:
            continue
        record = candidate["overlap_records"][0]
        completion = candidate["completion"]
        mapper = TokenSpanMapper(tokenizer, completion)
        target = int(record["token_index"])
        nearby = []
        for index in range(max(0, target - 4), min(len(mapper.input_ids), target + 5)):
            start, end = mapper.offsets[index]
            nearby.append(
                {
                    "index": index,
                    "id": int(mapper.input_ids[index]),
                    "decoded_token": tokenizer.decode([mapper.input_ids[index]], skip_special_tokens=False),
                    "offset": [int(start), int(end)],
                    "offset_text": completion[start:end],
                }
            )
        offset = record["canonical_offset_mapping"]
        center = offset[0] if offset else 0
        record["nearby_tokens"] = nearby
        record["completion_context"] = completion[max(0, center - 100) : center + 100]


def audit_full_data(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    aggregates = {
        route: {
            "candidate_count": 0,
            "candidate_with_overlap_count": 0,
            "overlap_token_count": 0,
            "overlap_unit_pair_count": 0,
            "max_units_per_token": 0,
        }
        for route in ("action", "chain")
    }
    for row in rows:
        route = row["route"]
        completion = str(row["raw_gold_output"])
        marginal = (
            action_marginal_credit(completion, row)
            if route == "action"
            else chain_marginal_credit(completion, row)
        )
        mapper = TokenSpanMapper(tokenizer, completion)
        units = (
            _action_units(marginal, mapper)
            if route == "action" and marginal["valid"]
            else _chain_units(marginal, mapper)
            if route == "chain" and marginal["valid"]
            else []
        )
        summary = summarize_candidate_overlaps(units, generated=False)
        aggregate = aggregates[route]
        aggregate["candidate_count"] += 1
        aggregate["candidate_with_overlap_count"] += summary["overlap_token_count"] > 0
        aggregate["overlap_token_count"] += summary["overlap_token_count"]
        aggregate["overlap_unit_pair_count"] += summary["overlap_unit_pair_count"]
        aggregate["max_units_per_token"] = max(
            aggregate["max_units_per_token"], summary["max_units_per_token"]
        )
    return aggregates


def final_conclusion(fixed: Sequence[Mapping[str, Any]]) -> str:
    labels = {
        pair["classification"]
        for route in fixed
        for candidate in route["candidates"]
        for record in candidate["overlap_records"]
        for pair in record["pair_classifications"]
    }
    if "CHAR_SPAN_OVERLAP" in labels:
        return "CHAR_SPAN_BUG"
    if "PROJECTION_BUG" in labels:
        return "PROJECTION_BUG"
    if "TOKENIZER_BOUNDARY_OVERLAP" in labels:
        return "TOKENIZER_BOUNDARY_OVERLAP_CONFIRMED"
    return "UNKNOWN"


def render_markdown(result: Mapping[str, Any]) -> str:
    action = result["fixed"]["action"]
    chain = result["fixed"]["chain"]
    lines = [
        "# MC_USER_v1 Real-tokenizer Credit-unit Overlap Audit",
        "",
        f"Conclusion: **{result['conclusion']}**",
        "",
        "## Fixed real-smoke reproduction",
        "",
    ]
    for route in (action, chain):
        lines.extend([f"### {route['route'].title()}", ""])
        for candidate in route["candidates"]:
            lines.append(
                f"- candidate {candidate['candidate_index']}: "
                f"canonical==generated={candidate['canonical_equals_generated']}, "
                f"projection_required={candidate['projection_required']}, "
                f"overlap_tokens={candidate['generated_overlap']['overlap_token_count']}"
            )
            for record in candidate["overlap_records"]:
                lines.append(
                    f"  - token {record['token_index']} id={record['token_id']} "
                    f"decoded={record['decoded_token']!r} offset={record['canonical_offset_mapping']}"
                )
                for unit in record["units"]:
                    label = unit["sid"] if unit["sid"] is not None else f"event {unit['event_index']}"
                    lines.append(
                        f"    - unit {unit['unit_index']} {label}: delta={unit['delta']:.12g}, "
                        f"chars=[{unit['char_start']},{unit['char_end']})"
                    )
                if record.get("completion_context") is not None:
                    lines.extend(["", "```text", record["completion_context"], "```", ""])
                    lines.append("| index | id | decoded | offset | text |")
                    lines.append("|---:|---:|---|---|---|")
                    for token in record["nearby_tokens"]:
                        lines.append(
                            f"| {token['index']} | {token['id']} | `{token['decoded_token']}` | "
                            f"{token['offset']} | `{token['offset_text']}` |"
                        )
        lines.append("")
    lines.extend(["## Full train_3000 raw-gold audit", ""])
    lines.append("| route | candidates with overlap / total | overlap tokens | unit pairs | max units/token |")
    lines.append("|---|---:|---:|---:|---:|")
    for route in ("action", "chain"):
        stats = result["full_data"][route]
        lines.append(
            f"| {route} | {stats['candidate_with_overlap_count']} / {stats['candidate_count']} | "
            f"{stats['overlap_token_count']} | {stats['overlap_unit_pair_count']} | "
            f"{stats['max_units_per_token']} |"
        )
    lines.extend(["", "## Unchanged objective reproduction", ""])
    for route in ("action", "chain"):
        reproduction = result["objective_reproduction"][route]
        detail = reproduction.get("error", "no overlap error")
        lines.append(f"- {route}: {reproduction['status']} - {detail}")
    lines.extend(["", "## Fixed mixed-sign check", ""])
    for route in ("action", "chain"):
        stats = result["fixed_overlap_summary"][route]
        lines.append(
            f"- {route}: same-sign={stats['same_sign_overlap_token_count']}, "
            f"mixed-sign={stats['mixed_sign_overlap_token_count']}"
        )
    return "\n".join(lines) + "\n"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if file_sha256(args.train_data) != TRAIN_SHA256:
        raise RuntimeError("train_3000 SHA256 mismatch")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = read_jsonl(args.train_data)
    fixed_data = prepare_fixed_data(tokenizer, rows)
    selection = fixed_data["selection"]
    action = audit_fixed_route(
        "action", selection["action"], fixed_data["rollouts"]["action"], tokenizer
    )
    chain = audit_fixed_route(
        "chain", selection["chain"], fixed_data["rollouts"]["chain"], tokenizer
    )
    for fixed_route in (action, chain):
        add_neighborhood(fixed_route, tokenizer)
    fixed_summary = {}
    for route_name, route_result in (("action", action), ("chain", chain)):
        fixed_summary[route_name] = {
            key: sum(candidate["generated_overlap"][key] for candidate in route_result["candidates"])
            for key in (
                "overlap_token_count",
                "same_sign_overlap_token_count",
                "mixed_sign_overlap_token_count",
            )
        }
    result = {
        "status": "PASS",
        "cpu_only": True,
        "tokenizer": str(args.tokenizer),
        "train_data": str(args.train_data),
        "train_sha256": TRAIN_SHA256,
        "fixed": {"action": action, "chain": chain},
        "objective_reproduction": {
            route: reproduce_objective_result(
                fixed_data["rollouts"][route], fixed_data["batches"][route]
            )
            for route in ("action", "chain")
        },
        "fixed_overlap_summary": fixed_summary,
        "full_data": audit_full_data(rows, tokenizer),
    }
    result["conclusion"] = final_conclusion([action, chain])
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.doc_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.doc_output.write_text(render_markdown(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=Path(BASE_MODEL))
    parser.add_argument("--train-data", type=Path, default=Path(TRAIN_DATA))
    parser.add_argument("--result-output", type=Path, default=Path(RESULT_OUTPUT))
    parser.add_argument("--doc-output", type=Path, default=Path(DOC_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    audit_result = run_audit(parse_args())
    print(json.dumps({
        "status": audit_result["status"],
        "conclusion": audit_result["conclusion"],
        "fixed_overlap_summary": audit_result["fixed_overlap_summary"],
        "full_data": audit_result["full_data"],
    }, ensure_ascii=False, indent=2))
