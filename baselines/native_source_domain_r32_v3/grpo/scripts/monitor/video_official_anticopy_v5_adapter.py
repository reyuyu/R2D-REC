"""Read-only adapter for captured Official Anti-Copy V5 monitor events."""
from __future__ import annotations

import math
from typing import Any, Iterable

VIDEO_EXPERIMENT = "GR_REC_VideoOfficialAntiCopy_v5"
MIXED_EXPERIMENT = "GR_REC_OfficialAntiCopy_Mixed_v5"
EXPERIMENT = VIDEO_EXPERIMENT


def is_video_official_anticopy_v5_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") == VIDEO_EXPERIMENT


def is_official_anticopy_mixed_v5_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") == MIXED_EXPERIMENT


def is_official_anticopy_v5_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") in {VIDEO_EXPERIMENT, MIXED_EXPERIMENT}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _at(values: Any, index: int, default: Any) -> Any:
    return values[index] if isinstance(values, list) and index < len(values) else default


def _candidate_rows(cot: dict[str, Any], target_domain: str) -> list[dict[str, Any]]:
    details = cot.get("candidate_details", [])
    rows = []
    for index in range(8):
        detail = details[index] if index < len(details) else {}
        raw = float(detail.get("raw_q_reward", _at(cot.get("raw_q_rewards"), index, 0.0)))
        final = float(detail.get("final_sid_reward", _at(cot.get("sid_rewards"), index, 0.0)))
        copied = bool(detail.get("is_history_copy", _at(cot.get("is_history_copy"), index, False)))
        sid = detail.get("candidate_sid", _at(cot.get("candidate_sids"), index, None))
        rows.append({
            "candidate_id": index,
            "sid": sid,
            "copied_history_sid": sid if copied else None,
            "completion_text": detail.get("candidate_text", _at(cot.get("candidate_texts"), index, "")),
            "token_ids": _at(cot.get("candidate_ids"), index, []),
            "raw_q_reward": raw,
            "reward_level": detail.get("reward_level", "unknown"),
            "is_history_copy": copied,
            "final_sid_reward": final,
            "reward_delta": final - raw,
            "sid_advantage": float(detail.get("sid_advantage", _at(cot.get("sid_advantages"), index, 0.0))),
            "cot_contribution": float(_at(cot.get("cot_contributions"), index, 0.0)),
            "anti_copy_removed": copied and raw > 0.0 and final == 0.0,
            "copy_discounted": copied and final != raw,
            "copied_exact_preserved": target_domain == "video" and copied and raw == 8.0 and final == 8.0,
            "masked_action_token_count": 3,
            "sampling_contract": f"fixed <{target_domain}_begin>; stochastic Sample8 ABC3",
            "target_domain": target_domain,
        })
    return rows


def _cot_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [row["raw_q_reward"] for row in rows]
    final = [row["final_sid_reward"] for row in rows]
    advantage = [row["sid_advantage"] for row in rows]
    return {
        "raw_reward_mean": _mean(raw), "raw_reward_std": _std(raw),
        "final_reward_mean": _mean(final), "final_reward_std": _std(final),
        "advantage_std": _std(advantage),
        "copy_count": sum(row["is_history_copy"] for row in rows),
        "copy_rate": sum(row["is_history_copy"] for row in rows) / 8.0,
        "anti_copy_removed_count": sum(row["anti_copy_removed"] for row in rows),
        "copy_discounted_count": sum(row["copy_discounted"] for row in rows),
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
        step, current_rollout = int(event.get("step", 0)), int(event.get("rollout_id", 0))
        if from_step is not None and step < from_step or to_step is not None and step > to_step:
            continue
        if rollout_id is not None and current_rollout != rollout_id:
            continue
        if event.get("type") == "optimization":
            optimization.append(dict(event))
            continue
        event_type = event.get("type")
        if event_type not in {"video_official_anticopy_v5", "official_anticopy_mixed_v5"}:
            continue
        current_group = str(event.get("recommendation_group_id", ""))
        if group_id is not None and current_group != group_id:
            continue
        source = (source_index or {}).get(current_group, {})
        target_domain = str(event.get("target_domain") or source.get("target_domain") or "video")
        cots = []
        for index, cot in enumerate(event.get("cots", [])):
            candidates = _candidate_rows(cot, target_domain)
            cots.append({
                "cot_id": index, "cot_text": cot.get("cot_text", ""),
                "cot_length": cot.get("cot_length"), "cot_reward": float(cot.get("cot_reward", 0.0)),
                "cot_advantage": float(cot.get("cot_advantage", 0.0)),
                "official_domain_prefix": cot.get("official_domain_prefix", f"<|{target_domain}_begin|>"),
                "history_sids": cot.get("history_sids", []), "gold_sids": cot.get("gold_sids", []),
                "history_sid_count": cot.get("history_sid_count"),
                "gold_history_exact_overlap": cot.get("gold_history_exact_overlap"),
                "rollout_wall_sec": cot.get("rollout_wall_sec"),
                "copy_A": cot.get("copy_A", 0), "copy_AB": cot.get("copy_AB", 0),
                "copy_Exact": cot.get("copy_Exact", 0), "noncopy_A": cot.get("noncopy_A", 0),
                "noncopy_AB": cot.get("noncopy_AB", 0), "noncopy_Exact": cot.get("noncopy_Exact", 0),
                "copy_positive_advantage_count": cot.get("copy_positive_advantage_count", 0),
                "copy_mean_advantage": cot.get("copy_mean_advantage", 0.0),
                "noncopy_positive_reward_count": cot.get("noncopy_positive_reward_count", 0),
                "zero_std_g8": cot.get("zero_std_g8"), "target_domain": target_domain,
                "summary": _cot_summary(candidates), "candidates": candidates,
            })
        groups.append({
            "valid": len(cots) == 4 and all(len(cot["candidates"]) == 8 for cot in cots),
            "route": "think", "kind": event_type, "step": step,
            "rollout_id": current_rollout, "group_id": current_group,
            "input_prompt": source.get("prompt"), "target_domain": target_domain,
            "fresh_rollout_index": event.get("fresh_rollout_index"), "reward_stage": event.get("reward_stage"),
            "domain_summary": event.get("domain_summary"),
            "gold_sids": source.get("all_gold_sids", cots[0]["gold_sids"] if cots else []),
            "history_sids": cots[0]["history_sids"] if cots else [],
            "rollout_fingerprint": event.get("rollout_fingerprint"),
            "normalization_topology": event.get("normalization_topology", "G4 + 4 independent Official G8; never G32"),
            "cots": cots,
            "provenance": {"rewards_advantages_copy": "captured", "input_prompt": "sha_guarded_source_dataset"},
        })
    return groups[-limit:], optimization


def captured_payload(groups: list[dict[str, Any]], optimization: list[dict[str, Any]]) -> dict[str, Any]:
    mixed = any(group.get("kind") == "official_anticopy_mixed_v5" for group in groups)
    return {
        "read_only": True, "supported": True,
        "formula": "official_anticopy_mixed_v5" if mixed else "video_official_anticopy_v5",
        "provenance": {"captured": "训练时直接落盘；前端不复算 reward/advantage"},
        "groups": groups, "optimization": optimization,
    }
