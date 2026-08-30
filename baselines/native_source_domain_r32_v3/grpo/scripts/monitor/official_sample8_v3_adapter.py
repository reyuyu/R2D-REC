"""Read-only adapter for captured V3-Official Sample8 monitor events."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

EXPERIMENT = "GR_REC_ThinkOfficialSample8_v3"
EVENT_TYPE = "official_sample8"
HISTORY_SID_RE = re.compile(
    r"<\|(video|ad|prod|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)


def is_official_sample8_v3_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") == EXPERIMENT


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _at(values: Any, index: int, default: Any) -> Any:
    return values[index] if isinstance(values, list) and index < len(values) else default


def extract_history_sids(prompt: Any, target_domain: str) -> set[tuple[str, int, int, int]]:
    """Parse complete target-domain SIDs from only the user-history region."""
    if not isinstance(prompt, str):
        return set()
    first_newline = prompt.find("\n")
    if first_newline < 0:
        return set()
    region = prompt[first_newline + 1:]
    boundaries = [
        position for marker in ("/think", "</think>", "<|im_start|>assistant")
        if (position := region.find(marker)) >= 0
    ]
    if boundaries:
        region = region[:min(boundaries)]
    return {
        (domain, int(a), int(b), int(c))
        for domain, a, b, c in HISTORY_SID_RE.findall(region)
        if domain == target_domain
    }


def _sid_tuple(value: Any) -> tuple[str, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return str(value[0]), int(value[1]), int(value[2]), int(value[3])
    except (TypeError, ValueError):
        return None


def _candidate_rows(
    cot: dict[str, Any], target_domain: str,
    history_sids: set[tuple[str, int, int, int]],
) -> list[dict[str, Any]]:
    details = cot.get("candidate_details", [])
    rows = []
    for index in range(8):
        detail = details[index] if isinstance(details, list) and index < len(details) else {}
        reward = float(detail.get("q_reward", _at(cot.get("q_rewards"), index, 0.0)))
        advantage = float(detail.get("sid_advantage", _at(cot.get("sid_advantages"), index, 0.0)))
        sid = detail.get("candidate_sid", _at(cot.get("candidate_sids"), index, None))
        sid_key = _sid_tuple(sid)
        copied = sid_key is not None and sid_key in history_sids
        rows.append({
            "candidate_id": index,
            "sid": sid,
            "completion_text": detail.get("candidate_text", _at(cot.get("candidate_texts"), index, "")),
            "token_ids": _at(cot.get("candidate_ids"), index, []),
            "raw_q_reward": reward,
            "reward_level": detail.get("reward_level", "unknown"),
            "is_history_copy": copied,
            "copied_history_sid": list(sid_key) if copied else None,
            "final_sid_reward": reward,
            "reward_delta": 0.0,
            "sid_advantage": advantage,
            "cot_contribution": reward,
            "anti_copy_removed": False,
            "copy_discounted": False,
            "copied_exact_preserved": False,
            "masked_action_token_count": 3,
            "sampling_contract": f"fixed <{target_domain}_begin>; stochastic Sample8 ABC3",
            "target_domain": target_domain,
        })
    return rows


def _cot_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row["raw_q_reward"] for row in rows]
    advantages = [row["sid_advantage"] for row in rows]
    levels = [str(row["reward_level"]).lower() for row in rows]
    return {
        "raw_reward_mean": _mean(rewards),
        "raw_reward_std": _std(rewards),
        "final_reward_mean": _mean(rewards),
        "final_reward_std": _std(rewards),
        "advantage_std": _std(advantages),
        "positive_reward_count": sum(value > 0 for value in rewards),
        "positive_advantage_count": sum(value > 0 for value in advantages),
        "negative_advantage_count": sum(value < 0 for value in advantages),
        "exact_count": sum(level == "exact" for level in levels),
        "ab_count": sum(level == "ab" for level in levels),
        "a_count": sum(level == "a" for level in levels),
        # Copy remains a passive diagnostic; it never changes V3-Official training math.
        "copy_count": sum(row["is_history_copy"] for row in rows),
        "copy_rate": sum(row["is_history_copy"] for row in rows) / 8.0,
        "anti_copy_removed_count": 0,
        "copy_discounted_count": 0,
        "copy_positive_advantage_count": sum(
            row["is_history_copy"] and row["sid_advantage"] > 0 for row in rows
        ),
        "noncopy_positive_reward_count": sum(
            not row["is_history_copy"] and row["raw_q_reward"] > 0 for row in rows
        ),
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
        if (from_step is not None and step < from_step) or (to_step is not None and step > to_step):
            continue
        if rollout_id is not None and current_rollout != rollout_id:
            continue
        if event.get("type") == "optimization":
            optimization.append(dict(event))
            continue
        if event.get("type") != EVENT_TYPE:
            continue
        current_group = str(event.get("recommendation_group_id", ""))
        if group_id is not None and current_group != group_id:
            continue
        source = (source_index or {}).get(current_group, {})
        target_domain = str(event.get("target_domain") or source.get("target_domain") or "video")
        history_sids = extract_history_sids(source.get("prompt"), target_domain)
        cots = []
        for index, cot in enumerate(event.get("cots", [])):
            candidates = _candidate_rows(cot, target_domain, history_sids)
            cots.append({
                "cot_id": index,
                "cot_text": cot.get("cot_text", ""),
                "cot_length": cot.get("cot_length"),
                "cot_reward": float(cot.get("cot_reward", 0.0)),
                "cot_advantage": float(cot.get("cot_advantage", 0.0)),
                "official_domain_prefix": cot.get("official_domain_prefix", f"<|{target_domain}_begin|>"),
                "gold_sids": cot.get("gold_sids", []),
                "history_sids": [list(sid) for sid in sorted(history_sids)],
                "zero_std_g8": cot.get("zero_std_g8"),
                "target_domain": target_domain,
                "summary": _cot_summary(candidates),
                "candidates": candidates,
            })
        groups.append({
            "valid": len(cots) == 4 and all(len(cot["candidates"]) == 8 for cot in cots),
            "route": "think",
            "kind": EVENT_TYPE,
            "step": step,
            "rollout_id": current_rollout,
            "group_id": current_group,
            "input_prompt": source.get("prompt"),
            "target_domain": target_domain,
            "fresh_rollout_index": current_rollout,
            "reward_stage": "v3_original",
            "gold_sids": source.get("all_gold_sids", cots[0]["gold_sids"] if cots else []),
            "history_sids": [list(sid) for sid in sorted(history_sids)],
            "rollout_fingerprint": event.get("rollout_fingerprint"),
            "normalization_topology": event.get(
                "normalization_topology", "G4 CoT + 4 independent Official G8; never G32",
            ),
            "cots": cots,
            "provenance": {
                "rewards_advantages": "captured training monitor payload",
                "input_prompt": "sha_guarded_source_dataset",
            },
        })
    return groups[-limit:], optimization


def captured_payload(groups: list[dict[str, Any]], optimization: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "read_only": True,
        "supported": True,
        "formula": "official_sample8_v3",
        "provenance": {"captured": "训练时直接落盘；前端不复算 reward/advantage"},
        "groups": groups,
        "optimization": optimization,
    }
