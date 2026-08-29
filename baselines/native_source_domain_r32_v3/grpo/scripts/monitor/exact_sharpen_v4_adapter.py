"""Read-only adapter for captured GR_REC_ThinkExactSharpen_v4 events."""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable


EXPERIMENT = "GR_REC_ThinkExactSharpen_v4"


def is_exact_sharpen_v4_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") == EXPERIMENT


def _population_advantages(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    denominator = math.sqrt(variance + 1e-8)
    return [(value - mean) / denominator for value in values]


def _saturation_reward(raw_reward: float, saturated: bool) -> float:
    if not saturated:
        return raw_reward
    if raw_reward == 0.5:
        return 0.0
    if raw_reward == 2.0:
        return 1.5
    if raw_reward == 8.0:
        return 7.5
    return raw_reward


def _sid_key(value: Any, length: int = 4) -> tuple[Any, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    return tuple(value[:length])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _reward_counts(values: list[float]) -> dict[str, int]:
    levels = {-1.0: "invalid", -0.25: "wrong_domain", 0.0: "domain",
              0.5: "a", 2.0: "ab", 8.0: "exact"}
    counts = {name: 0 for name in levels.values()}
    for value in values:
        name = levels.get(float(value))
        if name is not None:
            counts[name] += 1
    return counts


def _reward_level(value: float) -> str:
    return {-1.0: "invalid", -0.25: "wrong_domain", 0.0: "domain",
            0.5: "A", 2.0: "AB", 8.0: "Exact"}.get(float(value), "unknown")


def _candidate_rows(
    cot: dict[str, Any], branch: str,
    decode_token_ids: Callable[[list[int]], str] | None = None,
) -> list[dict[str, Any]]:
    raw = cot[f"{branch}_raw_rewards"]
    shaped = cot[f"{branch}_rewards"]
    penalties = cot[f"{branch}_duplicate_penalties"]
    advantages = cot[f"{branch}_advantages"]
    sids = cot[f"{branch}_sids"]
    token_ids = cot[f"{branch}_candidate_ids"]
    saturated = bool(cot[f"{branch}_saturated"])
    parser_statuses = cot.get("free_parser_statuses", []) if branch == "free" else []
    action_spans = cot.get("free_action_spans", []) if branch == "free" else []
    return [
        {
            "candidate_id": index,
            "sid": sids[index],
            "token_ids": token_ids[index],
            "raw_reward": float(raw[index]),
            "saturation_reward": _saturation_reward(float(raw[index]), saturated),
            "duplicate_penalty": float(penalties[index]),
            "shaped_reward": float(shaped[index]),
            "advantage": float(advantages[index]),
            "a_reward_removed": saturated and float(raw[index]) == 0.5,
            "exact_duplicate_exempt": float(raw[index]) == 8.0,
            "reward_level": _reward_level(float(raw[index])),
            "parser_status": (
                parser_statuses[index] if index < len(parser_statuses)
                else "strict_abc3" if sids[index] is not None else "invalid"
            ),
            "first_sid_span": action_spans[index] if index < len(action_spans) else None,
            "generated_token_count": len(token_ids[index]),
            "masked_action_token_count": 4 if branch == "free" and sids[index] is not None else 3 if branch == "official" else 0,
            "sampling_contract": (
                "free continuation; first complete domain+A+B+C SID"
                if branch == "free" else "fixed target-domain begin; stochastic ABC3"
            ),
            "completion_text": decode_token_ids(token_ids[index]) if decode_token_ids else None,
        }
        for index in range(8)
    ]


def _branch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [float(row["raw_reward"]) for row in rows]
    shaped = [float(row["shaped_reward"]) for row in rows]
    advantages = [float(row["advantage"]) for row in rows]
    unique_sids = {_sid_key(row.get("sid")) for row in rows}
    unique_sids.discard(None)
    return {
        "raw_reward_mean": _mean(raw), "raw_reward_std": _std(raw),
        "shaped_reward_mean": _mean(shaped), "shaped_reward_std": _std(shaped),
        "advantage_std": _std(advantages), "reward_counts": _reward_counts(raw),
        "parsed_count": sum(row.get("sid") is not None for row in rows),
        "unique_sid_count": len(unique_sids),
        "duplicate_penalized_count": sum(float(row["duplicate_penalty"]) != 0.0 for row in rows),
        "a_reward_removed_count": sum(bool(row["a_reward_removed"]) for row in rows),
    }


def _branch_consistency(free: list[dict[str, Any]], official: list[dict[str, Any]]) -> dict[str, Any]:
    free_sids = {_sid_key(row.get("sid")) for row in free}
    official_sids = {_sid_key(row.get("sid")) for row in official}
    free_sids.discard(None)
    official_sids.discard(None)
    free_ab = {_sid_key(row.get("sid"), 3) for row in free}
    official_ab = {_sid_key(row.get("sid"), 3) for row in official}
    free_ab.discard(None)
    official_ab.discard(None)
    free_a = {_sid_key(row.get("sid"), 2) for row in free}
    official_a = {_sid_key(row.get("sid"), 2) for row in official}
    free_a.discard(None)
    official_a.discard(None)
    union = free_sids | official_sids
    return {
        "exact_sid_overlap": len(free_sids & official_sids),
        "exact_sid_jaccard": len(free_sids & official_sids) / len(union) if union else 0.0,
        "ab_prefix_overlap": len(free_ab & official_ab),
        "a_prefix_overlap": len(free_a & official_a),
        "free_unique_sids": len(free_sids),
        "official_unique_sids": len(official_sids),
        "same_best_reward_level": max((row["raw_reward"] for row in free), default=-1.0)
                                  == max((row["raw_reward"] for row in official), default=-1.0),
        "interpretation": "两路上下文与采样空间不同；一致性仅描述 SID 覆盖重合，不要求逐样本相同",
    }


def adapt_events(
    events: Iterable[dict[str, Any]], *, from_step: int | None = None,
    to_step: int | None = None, rollout_id: int | None = None,
    group_id: str | None = None, limit: int = 40,
    source_index: dict[str, dict[str, Any]] | None = None,
    decode_token_ids: Callable[[list[int]], str] | None = None,
) -> list[dict[str, Any]]:
    groups = []
    for event in events:
        if event.get("type") != "exact_sharpen_v4":
            continue
        step = int(event.get("step", 0))
        current_rollout = int(event.get("rollout_id", 0))
        current_group = event.get("recommendation_group_id")
        if from_step is not None and step < from_step:
            continue
        if to_step is not None and step > to_step:
            continue
        if rollout_id is not None and current_rollout != rollout_id:
            continue
        if group_id is not None and current_group != group_id:
            continue
        cot_rewards = [float(cot["cot_reward"]) for cot in event.get("cots", [])]
        cot_advantages = _population_advantages(cot_rewards)
        cots = []
        for index, cot in enumerate(event.get("cots", [])):
            free_rows = _candidate_rows(cot, "free", decode_token_ids)
            official_rows = _candidate_rows(cot, "official", decode_token_ids)
            cots.append({
                "cot_id": index,
                "cot_text": cot.get("cot_text", ""),
                "cot_reward": cot_rewards[index],
                "cot_advantage": cot_advantages[index],
                "cot_coverage": cot.get("cot_coverage", {}),
                "target_domain": cot.get("target_domain"),
                "official_domain_prefix": cot.get("official_domain_prefix"),
                "rollout_wall_sec": cot.get("rollout_wall_sec"),
                "free": free_rows,
                "official": official_rows,
                "free_summary": _branch_summary(free_rows),
                "official_summary": _branch_summary(official_rows),
                "branch_consistency": _branch_consistency(free_rows, official_rows),
            })
        exact_overlap = sum(cot["branch_consistency"]["exact_sid_overlap"] for cot in cots)
        ab_overlap = sum(cot["branch_consistency"]["ab_prefix_overlap"] for cot in cots)
        a_overlap = sum(cot["branch_consistency"]["a_prefix_overlap"] for cot in cots)
        source = (source_index or {}).get(str(current_group), {})
        groups.append({
            "valid": len(cots) == 4,
            "route": "v4",
            "kind": "exact_sharpen_v4",
            "step": step,
            "rollout_id": current_rollout,
            "group_id": current_group,
            "input_prompt": source.get("prompt"),
            "gold_sids": source.get("all_gold_sids", source.get("gold_sids", [])),
            "target_domain": source.get("target_domain", cots[0].get("target_domain") if cots else None),
            "rollout_fingerprint": event.get("rollout_fingerprint"),
            "free_saturated": bool(event.get("free_saturated")),
            "official_saturated": bool(event.get("official_saturated")),
            "free_a_plus_count": int(event.get("free_a_plus_count", 0)),
            "official_a_plus_count": int(event.get("official_a_plus_count", 0)),
            "normalization_topology": event.get("normalization_topology"),
            "branch_consistency": {
                "exact_sid_overlap": exact_overlap,
                "ab_prefix_overlap": ab_overlap,
                "a_prefix_overlap": a_overlap,
                "cot_same_best_reward_level": sum(
                    cot["branch_consistency"]["same_best_reward_level"] for cot in cots
                ),
                "cot_count": len(cots),
                "interpretation": "Free 与 Official 是不同采样合同；该指标衡量覆盖重合而非逐位相等",
            },
            "cots": cots,
            "provenance": {
                "raw_shaped_duplicate_advantage": "captured",
                "cot_advantage": "read_only_population_reconstruction",
            },
        })
    return groups[-limit:]


def captured_payload(groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "read_only": True,
        "supported": True,
        "formula": "exact_sharpen_v4",
        "provenance": {
            "captured": "训练时直接落盘",
            "reconstructed": "仅 CoT G4 advantage 由已落盘 CoT reward 只读复算",
        },
        "groups": groups,
    }
