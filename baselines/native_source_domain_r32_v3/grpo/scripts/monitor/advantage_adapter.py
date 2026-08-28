"""Read-only reconstruction for the monitor advantage/credit view."""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Callable


RECONSTRUCTED = {"source": "reconstructed", "label": "复算"}
CAPTURED = {"source": "captured", "label": "实采"}
STAGES = ("Domain", "A", "B", "C")
_TOKENIZER = None
_TOKENIZER_LOCK = Lock()


def _formal_source_roots() -> list[Path]:
    """Locate formal code from both source-tree and standalone deployments."""
    roots = []
    configured = os.environ.get("GRPO_FORMAL_SOURCE_ROOT")
    if configured:
        roots.append(Path(configured))
    roots.append(Path(__file__).resolve().parents[2])

    work_root = Path(os.environ.get("GRPO_WORK_ROOT", "/data/GRPO/work"))
    if work_root.is_dir():
        roots.extend(sorted(work_root.glob("*/baselines/native_source_domain_r32_v3/grpo")))

    unique = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


@lru_cache(maxsize=1)
def _formal() -> dict[str, Any]:
    for root in _formal_source_roots():
        ablation = root / "ablations" / "gr_rec_think_exact_clamp_v1"
        frontier_ablation = root / "ablations" / "gr_rec_nothink_only_frontier_v1"
        if not (ablation / "nothink_hierarchical_credit.py").is_file():
            continue
        for module_dir in (root / "scripts", ablation, frontier_ablation):
            if str(module_dir) not in sys.path:
                sys.path.insert(0, str(module_dir))
        from grpo_sid import parse_sid
        from nothink_hierarchical_credit import (
            HIERARCHY_SCALE,
            STAGE_INCREMENTS,
            conditional_hierarchical_credits,
            hierarchy_state,
            locate_domain_commitment_token,
        )
        from think_exact_clamp import think_exact_clamp_advantages
        from format_validator import validate_nothink_completion
        from frontier_credit import (
            FORMAT_ADV_TOTAL,
            FRONTIER_NEGATIVE,
            GATED,
            HIERARCHY_SCALE as FRONTIER_SCALE,
            POSITIVE_SUCCESS,
            STAGE_INCREMENTS as FRONTIER_INCREMENTS,
            plan_frontier_credits,
        )

        return locals()
    raise RuntimeError("正式 ThinkExactClamp 算法模块不可用，无法只读复算")


def formal_available() -> bool:
    try:
        _formal()
    except RuntimeError:
        return False
    return True


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalize_sid(value: Any) -> tuple | None:
    if isinstance(value, str):
        return _formal()["parse_sid"](value)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(value)
    return None


def _target_domain(trace: dict[str, Any]) -> str | None:
    value = trace.get("target_domain")
    if value:
        return str(value)
    gold = [_normalize_sid(item) for item in (trace.get("gold_sids") or [])]
    gold = [item for item in gold if item is not None]
    return str(gold[0][0]) if gold else None


def reconstruct_think_group(trace: dict[str, Any]) -> dict[str, Any]:
    candidates = trace.get("candidates") or []
    rewards = [_finite(candidate.get("reward")) for candidate in candidates]
    if len(rewards) != 4 or any(value is None for value in rewards):
        return _invalid_group(trace, "Think ExactClamp 需要完整的 G4 reward")
    values = [float(value) for value in rewards]
    mean = sum(values) / 4.0
    raw = [(value - mean) / 8.0 for value in values]
    clamped = [value >= 8.0 and advantage < 0.0 for value, advantage in zip(values, raw)]
    import torch

    final = _formal()["think_exact_clamp_advantages"](
        torch.tensor(values, dtype=torch.float64)
    ).tolist()
    rows = []
    for index, candidate in enumerate(candidates):
        rows.append({
            "candidate_id": candidate.get("candidate_id", index),
            "completion": candidate.get("completion", ""),
            "completion_length": candidate.get("completion_length"),
            "reward": values[index],
            "group_mean": mean,
            "raw_advantage": raw[index],
            "final_advantage": final[index],
            "clamped": clamped[index],
            "provenance": {
                "completion": CAPTURED,
                "reward": CAPTURED,
                "group_mean": RECONSTRUCTED,
                "raw_advantage": RECONSTRUCTED,
                "final_advantage": RECONSTRUCTED,
            },
        })
    return {
        **_identity(trace),
        "valid": True,
        "route": "think",
        "kind": "sequence_advantage",
        "rewards": values,
        "mean_reward": mean,
        "raw_advantages": raw,
        "final_advantages": final,
        "exact_count": sum(value >= 8.0 for value in values),
        "negative_raw_count": sum(value < 0.0 for value in raw),
        "clamped_count": sum(clamped),
        "zero_std": len(set(values)) == 1,
        "candidates": rows,
        "provenance": RECONSTRUCTED,
    }


def reconstruct_think_suffix_sid_group(trace: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact captured G8 population-normalized sequence values.

    The sequence value is applied only to the captured suffix mask by training;
    this adapter does not infer token boundaries or recompute SID rewards.
    """
    candidates = trace.get("candidates") or []
    rewards = [_finite(candidate.get("reward")) for candidate in candidates]
    if len(rewards) != 8 or any(value is None for value in rewards):
        return _invalid_group(trace, "Think suffix SID requires exactly 8 captured rewards")
    values = [float(value) for value in rewards]
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    final = [0.0 if std == 0.0 else (value - mean) / (std + 1e-4) for value in values]
    rows = []
    for index, candidate in enumerate(candidates):
        rows.append({
            "candidate_id": candidate.get("candidate_id", index),
            "completion": candidate.get("completion", ""),
            "completion_length": candidate.get("completion_length"),
            "reward": values[index],
            "group_mean": mean,
            "group_population_std": std,
            "raw_advantage": final[index],
            "final_advantage": final[index],
            "clamped": False,
            "loss_scope": candidate.get("loss_scope", "tokens_after_think_close_only"),
            "suffix_token_count": candidate.get("suffix_token_count"),
            "cot_token_count": candidate.get("cot_token_count"),
            "parser_status": candidate.get("parser_status"),
            "multi_sid_output": candidate.get("multi_sid_output", False),
            "sid_count": candidate.get("sid_count", 0),
            "parsed_sid": candidate.get("parsed_sid"),
            "all_parsed_sids": candidate.get("all_parsed_sids") or [],
            "provenance": {
                "completion": CAPTURED,
                "reward": CAPTURED,
                "group_mean": RECONSTRUCTED,
                "final_advantage": RECONSTRUCTED,
                "loss_scope": CAPTURED,
            },
        })
    return {
        **_identity(trace),
        "valid": True,
        "route": "think",
        "kind": "suffix_sequence_advantage",
        "rewards": values,
        "mean_reward": mean,
        "population_std": std,
        "raw_advantages": final,
        "final_advantages": final,
        "zero_std": std == 0.0,
        "loss_scope": "tokens_after_think_close_only",
        "candidates": rows,
        "provenance": RECONSTRUCTED,
    }


def _default_alignment(candidate: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    completion = str(candidate.get("completion") or "")
    final_sid = candidate.get("parsed_sid")
    if not final_sid or len(final_sid) != 4:
        return {"valid": False, "eligible": False, "failure": "invalid_final_sid", "positions": [None] * 4}
    try:
        encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = [int(value) for value in encoded["input_ids"]]
        alignment = _formal()["locate_domain_commitment_token"](token_ids, final_sid, tokenizer)
        positions = list(alignment.hierarchy_token_positions)
        offsets = encoded.get("offset_mapping") or []
        spans = []
        for position in positions:
            if position is None or position >= len(offsets):
                spans.append(None)
                continue
            start, end = offsets[position]
            spans.append({"start": int(start), "end": int(end), "text": completion[int(start):int(end)]})
        return {
            "valid": True,
            "eligible": bool(alignment.eligible),
            "mode": alignment.mode,
            "failure": alignment.failure,
            "positions": positions,
            "spans": spans,
        }
    except Exception as error:
        return {"valid": False, "eligible": False, "failure": str(error), "positions": [None] * 4, "spans": [None] * 4}


def _taxonomy(rewards: list[float]) -> str | None:
    if len(set(rewards)) != 1:
        return None
    value = rewards[0]
    if value == -0.25:
        return "ALL_WRONG_DOMAIN_ZERO_SIGNAL"
    if value == 0.0:
        return "DEAD_ZERO_BRIDGE"
    if value == 0.5:
        return "UNIFORM_A_ONLY"
    if value == 2.0:
        return "UNIFORM_AB"
    if value == 8.0:
        return "UNIFORM_EXACT"
    return "OTHER_ZERO_VARIANCE"


def reconstruct_nothink_group(
    trace: dict[str, Any],
    tokenizer: Any | None = None,
    aligner: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = trace.get("candidates") or []
    rewards = [_finite(candidate.get("reward")) for candidate in candidates]
    target_domain = _target_domain(trace)
    gold_sids = [_normalize_sid(value) for value in (trace.get("gold_sids") or [])]
    gold_sids = [value for value in gold_sids if value is not None]
    if len(candidates) != 8 or any(value is None for value in rewards) or target_domain is None:
        return _invalid_group(trace, "NoThink hierarchical credit 需要完整 G8、reward 和 gold SID")
    values = [float(value) for value in rewards]
    formal = _formal()
    states = [formal["hierarchy_state"](candidate.get("parsed_sid"), gold_sids, target_domain) for candidate in candidates]
    if aligner is not None:
        alignments = [aligner(candidate, index) for index, candidate in enumerate(candidates)]
    else:
        tokenizer = tokenizer or monitor_tokenizer()
        alignments = [_default_alignment(candidate, tokenizer) for candidate in candidates]
    domain_eligible = [bool(item.get("eligible")) for item in alignments]
    credits = formal["conditional_hierarchical_credits"](states, domain_eligible=domain_eligible)
    indicators = [
        [float(state.domain_correct), float(state.a_correct), float(state.ab_correct), float(state.exact)]
        for state in states
    ]
    means = [sum(row[column] for row in indicators) / 8.0 for column in range(4)]
    rows = []
    for index, (candidate, state, alignment, credit) in enumerate(zip(candidates, states, alignments, credits)):
        eligible = [
            bool(state.valid and alignment.get("eligible")),
            bool(state.domain_correct),
            bool(state.a_correct),
            bool(state.ab_correct),
        ]
        stages = []
        for column, name in enumerate(STAGES):
            stages.append({
                "stage": name,
                "eligible": eligible[column],
                "credit": float(credit[column]) if eligible[column] else None,
                "position": (alignment.get("positions") or [None] * 4)[column],
                "span": (alignment.get("spans") or [None] * 4)[column],
                "milestone_mean": means[column],
                "increment": formal["STAGE_INCREMENTS"][column],
                "scale": formal["HIERARCHY_SCALE"],
                "provenance": RECONSTRUCTED,
            })
        rows.append({
            "candidate_id": candidate.get("candidate_id", index),
            "completion": candidate.get("completion", ""),
            "completion_length": candidate.get("completion_length"),
            "reward": values[index],
            "parsed_sid": candidate.get("parsed_sid"),
            "milestones": {
                "domain": state.domain_correct,
                "a": state.a_correct,
                "ab": state.ab_correct,
                "exact": state.exact,
            },
            "commitment": {
                "mode": alignment.get("mode", "unresolved"),
                "valid": bool(alignment.get("valid")),
                "failure": alignment.get("failure"),
                "provenance": RECONSTRUCTED,
            },
            "stages": stages,
            "provenance": {"completion": CAPTURED, "reward": CAPTURED, "credits": RECONSTRUCTED},
        })
    taxonomy = _taxonomy(values)
    bridge_active = taxonomy == "DEAD_ZERO_BRIDGE"
    return {
        **_identity(trace),
        "valid": True,
        "route": "no_think",
        "kind": "token_credit",
        "target_domain": target_domain,
        "gold_sids": [list(value) for value in gold_sids],
        "rewards": values,
        "milestone_means": dict(zip(STAGES, means)),
        "taxonomy": taxonomy,
        "zero_signal": taxonomy is not None,
        "b_singleton": sum(state.ab_correct for state in states) == 1,
        "c_singleton": sum(state.exact for state in states) == 1,
        "bridge": {
            "active": bridge_active,
            "rl_credit": 0.0 if bridge_active else None,
            "lambda": 0.02 if bridge_active else None,
            "gold_a_targets": sorted({int(value[1]) for value in gold_sids}) if bridge_active else [],
            "provenance": RECONSTRUCTED,
        },
        "candidates": rows,
        "provenance": RECONSTRUCTED,
    }


def reconstruct_frontier_group(
    trace: dict[str, Any],
    tokenizer: Any | None = None,
    aligner: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconstruct the frozen Frontier v1 token plan from an immutable G8 trace."""
    candidates = trace.get("candidates") or []
    rewards = [_finite(candidate.get("reward")) for candidate in candidates]
    target_domain = _target_domain(trace)
    gold_sids = [_normalize_sid(value) for value in (trace.get("gold_sids") or [])]
    gold_sids = [value for value in gold_sids if value is not None]
    if len(candidates) != 8 or any(value is None for value in rewards) or target_domain is None:
        return _invalid_group(trace, "NoThink Frontier credit 需要完整 G8、reward 和 gold SID")

    formal = _formal()
    values = [float(value) for value in rewards]
    validations = [
        formal["validate_nothink_completion"](str(candidate.get("completion") or ""))
        for candidate in candidates
    ]
    states = [
        formal["hierarchy_state"](
            validation.parsed_sid if validation.valid else None,
            gold_sids,
            target_domain,
        )
        for validation in validations
    ]
    if aligner is not None:
        alignments = [
            aligner(candidate, index) if validation.valid else {
                "valid": False,
                "eligible": False,
                "mode": "format_violation",
                "failure": validation.reason,
                "positions": [None] * 4,
                "spans": [None] * 4,
            }
            for index, (candidate, validation) in enumerate(zip(candidates, validations))
        ]
    else:
        tokenizer = tokenizer or monitor_tokenizer()
        alignments = [
            _default_alignment(candidate, tokenizer) if validation.valid else {
                "valid": False,
                "eligible": False,
                "mode": "format_violation",
                "failure": validation.reason,
                "positions": [None] * 4,
                "spans": [None] * 4,
            }
            for candidate, validation in zip(candidates, validations)
        ]
    plan = formal["plan_frontier_credits"](states, [item.valid for item in validations])
    reason_counts: dict[str, int] = {}
    rows = []
    for index, (candidate, validation, state, alignment, planned) in enumerate(zip(
        candidates, validations, states, alignments, plan.candidates
    )):
        if not validation.valid:
            reason = validation.reason or "other"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        stages = []
        positions = alignment.get("positions") or [None] * 4
        spans = alignment.get("spans") or [None] * 4
        for column, name in enumerate(STAGES):
            kind = planned.kinds[column]
            eligible = bool(validation.valid and kind != formal["GATED"] and positions[column] is not None)
            stages.append({
                "stage": name,
                "eligible": eligible,
                "credit": float(planned.credits[column]) if eligible else None,
                "credit_kind": kind,
                "position": positions[column],
                "span": spans[column],
                "milestone_mean": float(plan.milestone_means[column]),
                "increment": formal["FRONTIER_INCREMENTS"][column],
                "scale": formal["FRONTIER_SCALE"],
                "provenance": RECONSTRUCTED,
            })
        completion_length = candidate.get("completion_length")
        penalty_total = float(formal["FORMAT_ADV_TOTAL"]) if not validation.valid else None
        penalty_per_token = (
            penalty_total / int(completion_length)
            if penalty_total is not None and completion_length and int(completion_length) > 0
            else None
        )
        rows.append({
            "candidate_id": candidate.get("candidate_id", index),
            "completion": candidate.get("completion", ""),
            "completion_length": completion_length,
            "reward": values[index],
            "parsed_sid": candidate.get("parsed_sid"),
            "format_valid": bool(validation.valid),
            "format_violation_reason": validation.reason,
            "format_penalty_total": penalty_total,
            "format_penalty_per_token": penalty_per_token,
            "milestones": {
                "domain": state.domain_correct,
                "a": state.a_correct,
                "ab": state.ab_correct,
                "exact": state.exact,
            },
            "commitment": {
                "mode": alignment.get("mode", "unresolved"),
                "valid": bool(alignment.get("valid")),
                "failure": alignment.get("failure"),
                "provenance": RECONSTRUCTED,
            },
            "stages": stages,
            "provenance": {"completion": CAPTURED, "reward": CAPTURED, "credits": RECONSTRUCTED},
        })
    bridge_active = all(validation.valid for validation in validations) and values == [0.0] * 8
    return {
        **_identity(trace),
        "valid": True,
        "route": "no_think",
        "kind": "frontier_token_credit",
        "algorithm": "frontier_v1",
        "target_domain": target_domain,
        "gold_sids": [list(value) for value in gold_sids],
        "rewards": values,
        "milestone_means": dict(zip(STAGES, plan.milestone_means)),
        "taxonomy": plan.taxonomy,
        "zero_signal": len(set(values)) == 1,
        "frontier_negative_counts": dict(zip(STAGES, plan.frontier_negative_counts)),
        "frontier_active": dict(zip(STAGES, plan.frontier_active)),
        "positive_stage_active": dict(zip(STAGES, plan.positive_active)),
        "format_violation_count": sum(not item.valid for item in validations),
        "format_violation_reason_counts": reason_counts,
        "b_singleton": sum(state.ab_correct for state in states) == 1,
        "c_singleton": sum(state.exact for state in states) == 1,
        "bridge": {
            "active": bridge_active,
            "rl_credit": "frontier_primary" if bridge_active else None,
            "lambda": 0.02 if bridge_active else None,
            "gold_a_targets": sorted({int(value[1]) for value in gold_sids}) if bridge_active else [],
            "provenance": RECONSTRUCTED,
        },
        "candidates": rows,
        "provenance": RECONSTRUCTED,
    }


def _identity(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": trace.get("step"),
        "rollout_id": trace.get("rollout_id"),
        "group_id": trace.get("group_id"),
    }


def _invalid_group(trace: dict[str, Any], reason: str) -> dict[str, Any]:
    return {**_identity(trace), "route": trace.get("route"), "valid": False, "reason": reason, "candidates": []}


def monitor_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(
                "/data/models/onereason-8b-pretrain-competition",
                trust_remote_code=True,
                local_files_only=True,
            )
    return _TOKENIZER


def reconstruct_groups(
    rows: list[dict[str, Any]], *, formula: str | None = None
) -> list[dict[str, Any]]:
    result = []
    for trace in rows:
        try:
            route = trace.get("route")
            if route == "think" and formula == "think_suffix_sid_v1":
                result.append(reconstruct_think_suffix_sid_group(trace))
            elif route == "think" and formula == "clamp_bridge_v1":
                result.append(reconstruct_think_group(trace))
            elif route == "no_think":
                if formula == "frontier_v1":
                    result.append(reconstruct_frontier_group(trace))
                elif formula == "clamp_bridge_v1":
                    result.append(reconstruct_nothink_group(trace))
                else:
                    result.append(_invalid_group(
                        trace,
                        "该历史实验未声明可验证的 advantage 公式，已停止复算以避免套用新公式",
                    ))
            elif route == "think":
                result.append(_invalid_group(
                    trace,
                    "该历史实验未声明可验证的 advantage 公式，已停止复算以避免套用新公式",
                ))
        except RuntimeError as error:
            result.append(_invalid_group(trace, str(error)))
    return result
