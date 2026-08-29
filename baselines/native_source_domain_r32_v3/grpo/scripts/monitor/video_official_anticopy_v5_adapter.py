"""Read-only adapter for captured Video Official Anti-Copy V5 monitor events."""
from __future__ import annotations

import math
from typing import Any, Iterable


EXPERIMENT = "GR_REC_VideoOfficialAntiCopy_v5"


def is_video_official_anticopy_v5_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") == EXPERIMENT


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _candidate_rows(cot: dict[str, Any]) -> list[dict[str, Any]]:
    details = cot.get("candidate_details", [])
    rows = []
    for index in range(8):
        detail = details[index] if index < len(details) else {}
        raw = float(detail.get("raw_q_reward", cot.get("raw_q_rewards", [0.0] * 8)[index]))
        final = float(detail.get("final_sid_reward", cot.get("sid_rewards", [0.0] * 8)[index]))
        copied = bool(detail.get("is_history_copy", cot.get("is_history_copy", [False] * 8)[index]))
        rows.append({
            "candidate_id": index,
            "sid": detail.get("candidate_sid", cot.get("candidate_sids", [None] * 8)[index]),
            "completion_text": detail.get("candidate_text", cot.get("candidate_texts", [""] * 8)[index]),
            "token_ids": cot.get("candidate_ids", [[]] * 8)[index],
            "raw_q_reward": raw,
            "reward_level": detail.get("reward_level", "unknown"),
            "is_history_copy": copied,
            "final_sid_reward": final,
            "sid_advantage": float(detail.get("sid_advantage", cot.get("sid_advantages", [0.0] * 8)[index])),
            "cot_contribution": float(cot.get("cot_contributions", [0.0] * 8)[index]),
            "anti_copy_removed": copied and raw > 0.0 and raw < 8.0 and final == 0.0,
            "copied_exact_preserved": copied and raw == 8.0 and final == 8.0,
            "masked_action_token_count": 3,
            "sampling_contract": "fixed <video_begin>; stochastic Sample8 ABC3",
        })
    return rows


def _cot_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [row["raw_q_reward"] for row in rows]
    final = [row["final_sid_reward"] for row in rows]
    advantage = [row["sid_advantage"] for row in rows]
    return {
        "raw_reward_mean": _mean(raw),
        "raw_reward_std": _std(raw),
        "final_reward_mean": _mean(final),
        "final_reward_std": _std(final),
        "advantage_std": _std(advantage),
        "copy_count": sum(row["is_history_copy"] for row in rows),
        "copy_rate": sum(row["is_history_copy"] for row in rows) / 8.0,
        "anti_copy_removed_count": sum(row["anti_copy_removed"] for row in rows),
        "copy_positive_advantage_count": sum(row["is_history_copy"] and row["sid_advantage"] > 0 for row in rows),
        "noncopy_positive_reward_count": sum(not row["is_history_copy"] and row["final_sid_reward"] > 0 for row in rows),
    }


def adapt_events(
    events: Iterable[dict[str, Any]], *, from_step: int | None = None,
    to_step: int | None = None, rollout_id: int | None = None,
    group_id: str | None = None, limit: int = 40,
    source_index: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    optimization: list[dict[str, Any]] = []
    for event in events:
        step = int(event.get("step", 0))
        current_rollout = int(event.get("rollout_id", 0))
        if from_step is not None and step < from_step or to_step is not None and step > to_step:
            continue
        if rollout_id is not None and current_rollout != rollout_id:
            continue
        if event.get("type") == "optimization":
            optimization.append(dict(event))
            continue
        if event.get("type") != "video_official_anticopy_v5":
            continue
        current_group = str(event.get("recommendation_group_id", ""))
        if group_id is not None and current_group != group_id:
            continue
        source = (source_index or {}).get(current_group, {})
        cots = []
        for index, cot in enumerate(event.get("cots", [])):
            candidates = _candidate_rows(cot)
            cots.append({
                "cot_id": index,
                "cot_text": cot.get("cot_text", ""),
                "cot_length": cot.get("cot_length"),
                "cot_reward": float(cot.get("cot_reward", 0.0)),
                "cot_advantage": float(cot.get("cot_advantage", 0.0)),
                "official_domain_prefix": cot.get("official_domain_prefix", "<|video_begin|>"),
                "history_sids": cot.get("history_sids", []),
                "gold_sids": cot.get("gold_sids", []),
                "history_sid_count": cot.get("history_sid_count"),
                "gold_history_exact_overlap": cot.get("gold_history_exact_overlap"),
                "rollout_wall_sec": cot.get("rollout_wall_sec"),
                "copy_A": cot.get("copy_A", 0), "copy_AB": cot.get("copy_AB", 0),
                "copy_Exact": cot.get("copy_Exact", 0), "noncopy_A": cot.get("noncopy_A", 0),
                "noncopy_AB": cot.get("noncopy_AB", 0), "noncopy_Exact": cot.get("noncopy_Exact", 0),
                "copy_positive_advantage_count": cot.get("copy_positive_advantage_count", 0),
                "copy_mean_advantage": cot.get("copy_mean_advantage", 0.0),
                "noncopy_positive_reward_count": cot.get("noncopy_positive_reward_count", 0),
                "summary": _cot_summary(candidates),
                "candidates": candidates,
            })
        groups.append({
            "valid": len(cots) == 4 and all(len(cot["candidates"]) == 8 for cot in cots),
            "route": "think", "kind": "video_official_anticopy_v5",
            "step": step, "rollout_id": current_rollout, "group_id": current_group,
            "input_prompt": source.get("prompt"),
            "target_domain": source.get("target_domain", "video"),
            "gold_sids": source.get("all_gold_sids", cots[0]["gold_sids"] if cots else []),
            "history_sids": cots[0]["history_sids"] if cots else [],
            "rollout_fingerprint": event.get("rollout_fingerprint"),
            "normalization_topology": event.get("normalization_topology"),
            "cots": cots,
            "provenance": {"rewards_advantages_copy": "captured", "input_prompt": "sha_guarded_source_dataset"},
        })
    return groups[-limit:], optimization


def captured_payload(groups: list[dict[str, Any]], optimization: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "read_only": True, "supported": True, "formula": "video_official_anticopy_v5",
        "provenance": {"captured": "训练时直接落盘；前端不复算 reward/advantage"},
        "groups": groups, "optimization": optimization,
    }
