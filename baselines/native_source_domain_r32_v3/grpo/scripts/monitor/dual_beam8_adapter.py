"""Read-only monitor adapter for GR_REC_ThinkDualBeam8_v2."""
from __future__ import annotations

from typing import Any, Iterable


EXPERIMENT = "GR_REC_ThinkDualBeam8_v2"


def is_dual_beam8_manifest(manifest: dict[str, Any]) -> bool:
    return manifest.get("experiment") == EXPERIMENT


def _sid_text(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    domain, a, b, c = value
    return f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def adapt_events(
    events: Iterable[dict[str, Any]],
    *,
    source_index: dict[str, dict[str, Any]] | None = None,
    from_step: int | None = None,
    to_step: int | None = None,
    rollout_id: int | None = None,
    group_id: str | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Expose captured two-level credit without recomputing training math."""
    source_index = source_index or {}
    selected = []
    for event in events:
        if event.get("type") != "dual_beam8" or not isinstance(event.get("cots"), list):
            continue
        step = int(event.get("step", 0))
        rid = int(event.get("rollout_id", 0))
        gid = str(event.get("recommendation_group_id") or "")
        if from_step is not None and step < from_step:
            continue
        if to_step is not None and step > to_step:
            continue
        if rollout_id is not None and rid != rollout_id:
            continue
        if group_id is not None and gid != group_id:
            continue
        selected.append(event)

    groups = []
    for event in selected[-limit:]:
        gid = str(event.get("recommendation_group_id") or "")
        cot_advantages = event.get("cot_advantages") or []
        candidates = []
        for cot_index, cot in enumerate(event.get("cots") or []):
            sid_rewards = cot.get("sid_rewards") or []
            sid_advantages = cot.get("sid_advantages") or []
            sid_levels = cot.get("sid_reward_levels") or []
            sid_ids = cot.get("beam_candidate_ids") or []
            sid_values = cot.get("beam_sids") or []
            sid_candidates = []
            for sid_index in range(max(
                len(sid_rewards), len(sid_advantages), len(sid_ids), len(sid_values)
            )):
                parsed_sid = sid_values[sid_index] if sid_index < len(sid_values) else None
                sid_candidates.append({
                    "candidate_id": sid_index,
                    "generated_token_ids": sid_ids[sid_index] if sid_index < len(sid_ids) else [],
                    "parsed_sid": parsed_sid,
                    "parsed_sid_text": _sid_text(parsed_sid),
                    "reward": sid_rewards[sid_index] if sid_index < len(sid_rewards) else None,
                    "advantage": sid_advantages[sid_index] if sid_index < len(sid_advantages) else None,
                    "reward_level": sid_levels[sid_index] if sid_index < len(sid_levels) else None,
                })
            candidates.append({
                "candidate_id": cot_index,
                "cot_text": cot.get("cot_text"),
                "cot_length": cot.get("cot_length"),
                "closed": cot.get("closed"),
                "reward": cot.get("cot_reward"),
                "final_advantage": (
                    cot_advantages[cot_index] if cot_index < len(cot_advantages) else None
                ),
                "beam8_wall_sec": cot.get("beam8_wall_sec"),
                "domain_prefix": cot.get("domain_prefix"),
                "exact": cot.get("exact"),
                "ab": cot.get("ab"),
                "a": cot.get("a"),
                "invalid": cot.get("invalid"),
                "sid_population_std": cot.get("sid_population_std"),
                "sid_zero_std": cot.get("sid_zero_std"),
                "sid_candidates": sid_candidates,
            })
        source = source_index.get(gid, {})
        groups.append({
            "valid": True,
            "kind": "dual_beam8_advantage",
            "route": "think",
            "step": event.get("step"),
            "rollout_id": event.get("rollout_id"),
            "rollout_fingerprint": event.get("rollout_fingerprint"),
            "group_id": gid,
            "recommendation_group_id": gid,
            "target_domain": event.get("target_domain"),
            "prompt": source.get("prompt"),
            "gold_sids": source.get("all_gold_sids", []),
            "cot_rewards": event.get("cot_rewards"),
            "cot_advantages": cot_advantages,
            "cot_population_std": event.get("cot_population_std"),
            "cot_zero_std": event.get("cot_zero_std"),
            "zero_std": event.get("cot_zero_std"),
            "normalization_topology": event.get("normalization_topology"),
            "candidates": candidates,
        })
    return groups


def captured_payload(groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "read_only": True,
        "supported": True,
        "formula": "dual_beam8_v2",
        "provenance": {
            "mode": "captured",
            "label": "实采",
            "detail": "训练时直接捕获；前端不重算 reward 或 advantage",
        },
        "groups": groups,
    }
