"""Read-only adapter for captured GR_REC_ThinkExactSharpen_v4 events."""
from __future__ import annotations

import math
from typing import Any, Iterable


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


def _candidate_rows(cot: dict[str, Any], branch: str) -> list[dict[str, Any]]:
    raw = cot[f"{branch}_raw_rewards"]
    shaped = cot[f"{branch}_rewards"]
    penalties = cot[f"{branch}_duplicate_penalties"]
    advantages = cot[f"{branch}_advantages"]
    sids = cot[f"{branch}_sids"]
    token_ids = cot[f"{branch}_candidate_ids"]
    saturated = bool(cot[f"{branch}_saturated"])
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
        }
        for index in range(8)
    ]


def adapt_events(
    events: Iterable[dict[str, Any]], *, from_step: int | None = None,
    to_step: int | None = None, rollout_id: int | None = None,
    group_id: str | None = None, limit: int = 40,
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
            cots.append({
                "cot_id": index,
                "cot_text": cot.get("cot_text", ""),
                "cot_reward": cot_rewards[index],
                "cot_advantage": cot_advantages[index],
                "cot_coverage": cot.get("cot_coverage", {}),
                "free": _candidate_rows(cot, "free"),
                "official": _candidate_rows(cot, "official"),
            })
        groups.append({
            "valid": len(cots) == 4,
            "route": "v4",
            "kind": "exact_sharpen_v4",
            "step": step,
            "rollout_id": current_rollout,
            "group_id": current_group,
            "rollout_fingerprint": event.get("rollout_fingerprint"),
            "free_saturated": bool(event.get("free_saturated")),
            "official_saturated": bool(event.get("official_saturated")),
            "free_a_plus_count": int(event.get("free_a_plus_count", 0)),
            "official_a_plus_count": int(event.get("official_a_plus_count", 0)),
            "normalization_topology": event.get("normalization_topology"),
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
