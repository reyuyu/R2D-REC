"""Pure CPU metrics for TrueRec Phase 0.7 G8 rollout census."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


COMPONENT_RE = re.compile(r"^<s_([abc])_(\d+)>$")
FRONTIERS = ("INVALID_FORMAT", "A_FAIL", "B_FAIL", "C_FAIL", "EXACT")
GROUP_CLASSES = ("ALL_INVALID", "NO_A_REACHED", "A_ONLY_MAX", "AB_ONLY_MAX", "EXACT_REACHED")
HPR_CLASSES = ("HPR_A_TRIGGER", "HPR_B_TRIGGER", "HPR_C_TRIGGER", "HPR_NONE")
LEGACY_REWARD = {"INVALID_FORMAT": -1.0, "A_FAIL": 0.0, "B_FAIL": 0.5, "C_FAIL": 2.0, "EXACT": 8.0}


class RolloutMetricError(RuntimeError):
    pass


def load_jsonl_gate(path: Path, expected_sha: str, expected_count: int) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise RolloutMetricError(f"DATASET_SHA_FAIL={path}:{actual_sha}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    group_ids = [str(row["recommendation_group_id"]) for row in rows]
    if len(rows) != expected_count or len(set(group_ids)) != expected_count:
        raise RolloutMetricError(
            f"DATASET_COUNT_OR_UNIQUE_FAIL={path}:{len(rows)},{len(set(group_ids))}"
        )
    return rows


def parse_raw_abc(raw_token_ids: list[int], id_to_token) -> tuple[str, str, str] | None:
    """Parse exactly three position-correct component tokens from raw IDs."""
    if len(raw_token_ids) != 3:
        return None
    tokens = tuple(str(id_to_token(token_id)) for token_id in raw_token_ids)
    matches = [COMPONENT_RE.fullmatch(token) for token in tokens]
    if any(match is None for match in matches):
        return None
    if tuple(match.group(1) for match in matches) != ("a", "b", "c"):
        return None
    return tokens


def split_abc(text: str) -> tuple[str, str, str]:
    matches = list(re.finditer(r"<s_([abc])_\d+>", text))
    tokens = tuple(match.group(0) for match in matches)
    if len(tokens) != 3 or "".join(tokens) != text:
        raise RolloutMetricError(f"INVALID_GOLD_ABC={text!r}")
    if tuple(match.group(1) for match in matches) != ("a", "b", "c"):
        raise RolloutMetricError(f"WRONG_GOLD_COMPONENT_ORDER={text!r}")
    return tokens


def assess_candidate(
    raw_token_ids: list[int], id_to_token, all_gold_abc: Iterable[str],
    fixed_domain_token: str, history_sids: Iterable[str],
) -> dict[str, Any]:
    gold = {split_abc(value) for value in all_gold_abc}
    if not gold:
        raise RolloutMetricError("EMPTY_GOLD")
    parsed = parse_raw_abc(raw_token_ids, id_to_token)
    valid = parsed is not None
    a_hit = bool(valid and any(parsed[0] == item[0] for item in gold))
    ab_hit = bool(valid and any(parsed[:2] == item[:2] for item in gold))
    exact = bool(valid and parsed in gold)
    if exact and not (ab_hit and a_hit) or ab_hit and not a_hit:
        raise RolloutMetricError("HIERARCHY_INVARIANT_FAIL")
    if not valid:
        frontier = "INVALID_FORMAT"
    elif not a_hit:
        frontier = "A_FAIL"
    elif not ab_hit:
        frontier = "B_FAIL"
    elif not exact:
        frontier = "C_FAIL"
    else:
        frontier = "EXACT"
    sid = fixed_domain_token + "".join(parsed) if valid else None
    target_history = {value for value in history_sids if value.startswith(fixed_domain_token)}
    history_copy = bool(sid and sid in target_history)
    result = {
        "format_valid": valid, "parsed_abc": "".join(parsed) if valid else None,
        "A_hit": a_hit, "AB_hit": ab_hit, "exact": exact, "frontier": frontier,
        "history_copy": history_copy,
        "correct_history_copy": bool(history_copy and exact),
        "wrong_history_copy": bool(history_copy and not exact),
        "legacy_scalar": LEGACY_REWARD[frontier],
    }
    return result


def group_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise RolloutMetricError("EMPTY_GROUP")
    any_valid = any(item["format_valid"] for item in candidates)
    any_a = any(item["A_hit"] for item in candidates)
    any_ab = any(item["AB_hit"] for item in candidates)
    any_exact = any(item["exact"] for item in candidates)
    if any_exact:
        group_class, hpr = "EXACT_REACHED", "HPR_NONE"
    elif any_ab:
        group_class, hpr = "AB_ONLY_MAX", "HPR_C_TRIGGER"
    elif any_a:
        group_class, hpr = "A_ONLY_MAX", "HPR_B_TRIGGER"
    elif any_valid:
        group_class, hpr = "NO_A_REACHED", "HPR_A_TRIGGER"
    else:
        group_class, hpr = "ALL_INVALID", "HPR_A_TRIGGER"
    rewards = [item["legacy_scalar"] for item in candidates]
    zero_std = len(set(rewards)) == 1
    uniform = None
    if zero_std:
        frontier_set = {item["frontier"] for item in candidates}
        uniform = "all " + next(iter(frontier_set)) if len(frontier_set) == 1 else "other uniform"
    raw_keys = [tuple(item["raw_token_ids"]) for item in candidates]
    valid = [item for item in candidates if item["format_valid"]]
    abc = [item["parsed_abc"] for item in valid]
    components = [split_abc(value) for value in abc]
    return {
        "ANY_VALID": any_valid, "ANY_A_HIT": any_a, "ANY_AB_HIT": any_ab,
        "ANY_EXACT": any_exact, "group_class": group_class, "hpr_trigger": hpr,
        "legacy_zero_std": zero_std, "legacy_uniform_type": uniform,
        "unique_raw_completion_count": len(set(raw_keys)),
        "unique_valid_ABC_count": len(set(abc)),
        "unique_A_count": len({value[0] for value in components}),
        "unique_AB_count": len({value[:2] for value in components}),
        "all_8_same_completion": len(set(raw_keys)) == 1,
        "any_history_copy": any(item["history_copy"] for item in candidates),
        "any_correct_history_copy": any(item["correct_history_copy"] for item in candidates),
        "any_wrong_history_copy": any(item["wrong_history_copy"] for item in candidates),
    }


def _rate(count: int, total: int) -> float:
    return round(count / total, 8) if total else 0.0


def categorical(rows: list[dict[str, Any]], key: str, categories: Iterable[str]) -> dict[str, Any]:
    counts = Counter(row[key] for row in rows)
    return {value: {"count": counts[value], "rate": _rate(counts[value], len(rows))} for value in categories}


def _candidate_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "N": len(rows),
        "format_valid": {"count": sum(row["format_valid"] for row in rows), "rate": _rate(sum(row["format_valid"] for row in rows), len(rows))},
        "A_hit": {"count": sum(row["A_hit"] for row in rows), "rate": _rate(sum(row["A_hit"] for row in rows), len(rows))},
        "AB_hit": {"count": sum(row["AB_hit"] for row in rows), "rate": _rate(sum(row["AB_hit"] for row in rows), len(rows))},
        "exact": {"count": sum(row["exact"] for row in rows), "rate": _rate(sum(row["exact"] for row in rows), len(rows))},
        "frontier": categorical(rows, "frontier", FRONTIERS),
    }


def _group_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "N": len(rows),
        "ANY_VALID": {"count": sum(row["ANY_VALID"] for row in rows), "rate": _rate(sum(row["ANY_VALID"] for row in rows), len(rows))},
        "ANY_A_HIT": {"count": sum(row["ANY_A_HIT"] for row in rows), "rate": _rate(sum(row["ANY_A_HIT"] for row in rows), len(rows))},
        "ANY_AB_HIT": {"count": sum(row["ANY_AB_HIT"] for row in rows), "rate": _rate(sum(row["ANY_AB_HIT"] for row in rows), len(rows))},
        "ANY_EXACT": {"count": sum(row["ANY_EXACT"] for row in rows), "rate": _rate(sum(row["ANY_EXACT"] for row in rows), len(rows))},
        "taxonomy": categorical(rows, "group_class", GROUP_CLASSES),
        "hpr": categorical(rows, "hpr_trigger", HPR_CLASSES),
    }


def breakdown(rows: list[dict[str, Any]], block_fn) -> dict[str, Any]:
    dimensions = {
        "overall": lambda row: "overall", "domain": lambda row: row["domain"],
        "novelty": lambda row: row["novelty"], "K_bucket": lambda row: row["K_bucket"],
        "domain_novelty": lambda row: f"{row['domain']}|{row['novelty']}",
    }
    output = {}
    for name, key_fn in dimensions.items():
        keys = sorted({key_fn(row) for row in rows})
        output[name] = {key: block_fn([row for row in rows if key_fn(row) == key]) for key in keys}
    return output


def census(candidate_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frontier = breakdown(candidate_rows, _candidate_block)
    reach = breakdown(group_rows, _group_block)
    zero = [row for row in group_rows if row["legacy_zero_std"]]
    diversity_values = {
        "unique_raw_completion_count": [row["unique_raw_completion_count"] for row in group_rows],
        "unique_valid_ABC_count": [row["unique_valid_ABC_count"] for row in group_rows],
        "unique_A_count": [row["unique_A_count"] for row in group_rows],
        "unique_AB_count": [row["unique_AB_count"] for row in group_rows],
    }
    diversity = {
        key: {"mean": round(statistics.fmean(values), 8), "median": statistics.median(values)}
        for key, values in diversity_values.items()
    }
    diversity["all_8_same_completion"] = {
        "count": sum(row["all_8_same_completion"] for row in group_rows),
        "rate": _rate(sum(row["all_8_same_completion"] for row in group_rows), len(group_rows)),
    }
    legacy = {
        "groups": len(group_rows), "zero_std_groups": len(zero), "zero_std_rate": _rate(len(zero), len(group_rows)),
        "uniform_types": dict(sorted(Counter(row["legacy_uniform_type"] for row in zero).items())),
    }
    def history_block(cands, groups):
        valid = [row for row in cands if row["format_valid"]]
        return {
            "valid_candidates": len(valid),
            "history_copy": {"count": sum(row["history_copy"] for row in valid), "rate": _rate(sum(row["history_copy"] for row in valid), len(valid))},
            "correct_history_copy": {"count": sum(row["correct_history_copy"] for row in valid), "rate": _rate(sum(row["correct_history_copy"] for row in valid), len(valid))},
            "wrong_history_copy": {"count": sum(row["wrong_history_copy"] for row in valid), "rate": _rate(sum(row["wrong_history_copy"] for row in valid), len(valid))},
            "groups": len(groups),
            "group_any_history_copy": {"count": sum(row["any_history_copy"] for row in groups), "rate": _rate(sum(row["any_history_copy"] for row in groups), len(groups))},
            "group_any_wrong_history_copy": {"count": sum(row["any_wrong_history_copy"] for row in groups), "rate": _rate(sum(row["any_wrong_history_copy"] for row in groups), len(groups))},
        }
    history = {"overall": history_block(candidate_rows, group_rows), "domain": {}, "novelty": {}}
    for field in ("domain", "novelty"):
        for value in sorted({row[field] for row in group_rows}):
            history[field][value] = history_block(
                [row for row in candidate_rows if row[field] == value],
                [row for row in group_rows if row[field] == value],
            )
    return {"candidate_frontier": frontier, "group_reach": reach, "legacy_zero_std": legacy, "diversity": diversity, "history_copy": history}
