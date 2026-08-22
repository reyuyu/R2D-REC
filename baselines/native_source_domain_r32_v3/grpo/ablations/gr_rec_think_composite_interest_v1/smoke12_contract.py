"""Pure CPU planning, fixture, summarization and gating for Composite Smoke12."""
from __future__ import annotations

from collections import Counter
import math
import statistics
from typing import Any, Iterable


SMOKE_OPTIMIZER_STEPS = 12
SMOKE_NUM_ITERATIONS = 2
SMOKE_FRESH_ROLLOUTS = 6
SMOKE_GROUPS_PER_ROLLOUT = 4
SMOKE_G4_COUNT = 24
SMOKE_CANDIDATE_COUNT = 96


def smoke12_plan() -> dict[str, Any]:
    plan = {
        "optimizer_steps": SMOKE_OPTIMIZER_STEPS,
        "num_iterations": SMOKE_NUM_ITERATIONS,
        "fresh_rollouts": SMOKE_FRESH_ROLLOUTS,
        "groups_per_fresh_rollout": SMOKE_GROUPS_PER_ROLLOUT,
        "g4_count": SMOKE_G4_COUNT,
        "candidate_count": SMOKE_CANDIDATE_COUNT,
        "enable_probes": False,
        "enable_checkpoints": False,
    }
    assert plan["fresh_rollouts"] == plan["optimizer_steps"] // plan["num_iterations"]
    assert plan["g4_count"] == plan["fresh_rollouts"] * plan["groups_per_fresh_rollout"]
    assert plan["candidate_count"] == plan["g4_count"] * 4
    return plan


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _stats(values: Iterable[Any], *, median: bool = True) -> dict[str, float | None]:
    numbers = _finite(values)
    result = {
        "min": min(numbers) if numbers else None,
        "mean": statistics.fmean(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
    }
    if median:
        result["median"] = statistics.median(numbers) if numbers else None
    return result


def _mean(values: Iterable[Any]) -> float | None:
    numbers = _finite(values)
    return statistics.fmean(numbers) if numbers else None


def summarize_composite_smoke(
    manifest: dict[str, Any],
    metrics: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    composite_events: list[dict[str, Any]],
    parameter_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate captured scalars and flags only; never recompute reward mathematics."""
    parameter_audit = parameter_audit or {}
    groups = [group for event in composite_events for group in event.get("groups", [])]
    candidates = [candidate for event in composite_events for candidate in event.get("candidates", [])]
    optimizer_steps = len({int(row["step"]) for row in metrics if int(row.get("step", 0)) > 0})
    rollout_ids = {
        row.get("rollout_id") for row in rollouts
        if row.get("rollout_id") is not None and row.get("route", "think") == "think"
    }
    if not rollout_ids:
        rollout_ids = {event.get("rollout_id") for event in composite_events}
    rollout_ids.discard(None)

    k_distribution = Counter()
    for candidate in candidates:
        count = int(candidate.get("matched_interest_count", 0))
        k_distribution[str(count) if count < 4 else "4+"] += 1

    beam_zero = sum(bool(group.get("beam_all_equal")) for group in groups)
    composite_zero = sum(bool(group.get("composite_all_equal")) for group in groups)
    rescued = sum(
        bool(group.get("beam_all_equal")) and not bool(group.get("composite_all_equal"))
        for group in groups
    )
    cot_active = sum(bool(group.get("cot_active")) for group in groups)
    group_count = len(groups)
    nonzero_advantage = any(
        abs(value) > 0 for value in _finite(
            candidate.get("final_sequence_advantage") for candidate in candidates
        )
    )

    all_numbers = [
        value for row in metrics for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    metrics_optimizer_steps = optimizer_steps
    runtime_optimizer_steps = parameter_audit.get(
        "runtime_optimizer_steps", parameter_audit.get("optimizer_steps")
    )
    summary = {
        "synthetic": bool(manifest.get("synthetic", False)),
        "optimizer_steps": metrics_optimizer_steps,
        "metrics_optimizer_steps": metrics_optimizer_steps,
        "runtime_optimizer_steps": runtime_optimizer_steps,
        "fresh_rollouts": len(rollout_ids),
        "g4_count": group_count,
        "candidate_count": len(candidates),
        "loss": _stats(row.get("loss") for row in metrics),
        "grad_norm": _stats(row.get("grad_norm") for row in metrics),
        "kl": _stats((row.get("approx_kl", row.get("kl")) for row in metrics), median=False),
        "clip_ratio": _stats(
            (row.get("clip_ratio", row.get("clip_fraction")) for row in metrics),
            median=False,
        ),
        "beam_raw_mean": _mean(candidate.get("beam_raw") for candidate in candidates),
        "u_cot_mean": _mean(candidate.get("cot_utility") for candidate in candidates),
        "composite_reward_mean": _mean(candidate.get("composite_reward") for candidate in candidates),
        "k_distribution": {key: k_distribution.get(key, 0) for key in ("0", "1", "2", "3", "4+")},
        "raw_n_mean": _mean(candidate.get("raw_n") for candidate in candidates),
        "grounded_n_mean": _mean(candidate.get("grounded_n") for candidate in candidates),
        "grounding_coverage_mean": _mean(candidate.get("grounding_coverage") for candidate in candidates),
        "completion_length_mean": _mean(candidate.get("completion_length") for candidate in candidates),
        "parser_failure_rate": (
            sum(candidate.get("parser_success") is False for candidate in candidates) / len(candidates)
            if candidates else None
        ),
        "q_active_rate": (
            sum(float(candidate.get("Q") or 0.0) > 0 for candidate in candidates) / len(candidates)
            if candidates else None
        ),
        "beam_zero_std_count": beam_zero,
        "beam_zero_std_rate": beam_zero / group_count if group_count else None,
        "composite_zero_std_count": composite_zero,
        "composite_zero_std_rate": composite_zero / group_count if group_count else None,
        "rescued_zero_count": rescued,
        "rescued_rate_among_beam_zero": rescued / beam_zero if beam_zero else None,
        "u_cot_variance_active_count": cot_active,
        "u_cot_variance_active_rate": cot_active / group_count if group_count else None,
        "tie_break_count": sum(bool(group.get("top_set_tie_break")) for group in groups),
        "tie_break_rate": (
            sum(bool(group.get("top_set_tie_break")) for group in groups) / group_count
            if group_count else None
        ),
        "strict_reversal_count": sum(bool(group.get("strict_beam_reversal")) for group in groups),
        "strict_reversal_rate": (
            sum(bool(group.get("strict_beam_reversal")) for group in groups) / group_count
            if group_count else None
        ),
        "BASE_DELTA": parameter_audit.get("BASE_DELTA"),
        "BASE_CHANGED": parameter_audit.get("BASE_CHANGED"),
        "base_version_changed_count": parameter_audit.get("base_version_changed_count"),
        "base_requires_grad_count": parameter_audit.get("base_requires_grad_count"),
        "LORA_CHANGED": parameter_audit.get("LORA_CHANGED"),
        "lora_total_l2_delta": parameter_audit.get("lora_total_l2_delta"),
        "lora_max_abs_delta": parameter_audit.get("lora_max_abs_delta"),
        "lora_parameter_tensor_count": parameter_audit.get("lora_parameter_tensor_count"),
        "parameter_audit_finite": bool(parameter_audit.get("finite")),
        "nan": bool(manifest.get("nan")) or any(math.isnan(value) for value in all_numbers),
        "inf": bool(manifest.get("inf")) or any(math.isinf(value) for value in all_numbers),
        "oom": bool(manifest.get("oom")),
        "nccl_error": bool(manifest.get("nccl_error")),
        "runtime_error": (
            bool(manifest.get("runtime_error"))
            or bool(parameter_audit.get("runtime_error"))
            or not bool(parameter_audit)
        ),
        "nonzero_advantage_signal": nonzero_advantage,
    }
    summary["loss_finite"] = bool(_finite(row.get("loss") for row in metrics)) and not summary["nan"] and not summary["inf"]
    summary["grad_finite"] = bool(_finite(row.get("grad_norm") for row in metrics)) and not summary["nan"] and not summary["inf"]
    return summary


def evaluate_smoke_conditions(summary: dict[str, Any]) -> dict[str, Any]:
    conditions = {
        "metrics_optimizer_steps_12": summary.get("metrics_optimizer_steps") == 12,
        "runtime_optimizer_steps_12": summary.get("runtime_optimizer_steps") == 12,
        "fresh_rollouts_6": summary.get("fresh_rollouts") == 6,
        "g4_count_24": summary.get("g4_count") == 24,
        "candidate_count_96": summary.get("candidate_count") == 96,
        "nan_absent": not bool(summary.get("nan")),
        "inf_absent": not bool(summary.get("inf")),
        "oom_absent": not bool(summary.get("oom")),
        "nccl_error_absent": not bool(summary.get("nccl_error")),
        "runtime_error_absent": not bool(summary.get("runtime_error")),
        "loss_finite": bool(summary.get("loss_finite")),
        "grad_finite": bool(summary.get("grad_finite")),
        "base_delta_zero": summary.get("BASE_DELTA") == 0,
        "base_unchanged": summary.get("BASE_CHANGED") is False,
        "base_requires_grad_false": summary.get("base_requires_grad_count") == 0,
        "parameter_audit_finite": bool(summary.get("parameter_audit_finite")),
        "lora_changed": summary.get("LORA_CHANGED") is True,
        "nonzero_advantage_signal": bool(summary.get("nonzero_advantage_signal")),
    }
    failures = [name for name, passed in conditions.items() if not passed]
    return {"conditions": conditions, "smoke_pass": not failures, "failure_reasons": failures}


def synthetic_smoke12_fixture() -> dict[str, Any]:
    """Compact structural fixture: 6 captured rollouts, 24 G4s, 96 short candidates."""
    manifest = {
        "synthetic": True,
        "run_id": "GR-REC-THINK-COMPOSITE-INTEREST-V1-SMOKE12-SYNTHETIC",
        "experiment": "GR_REC_Think_CompositeInterest_v1",
        "nan": False,
        "inf": False,
        "oom": False,
        "nccl_error": False,
        "runtime_error": False,
    }
    metrics = [
        {"step": step, "loss": 0.1 + step / 1000, "grad_norm": 0.2 + step / 1000,
         "approx_kl": step / 10000, "clip_ratio": step / 1000}
        for step in range(1, 13)
    ]
    rollouts = [{"rollout_id": rollout, "step": rollout * 2, "route": "think"} for rollout in range(6)]
    events = []
    for rollout in range(6):
        groups, candidates = [], []
        for offset in range(4):
            group_id = f"synthetic-r{rollout}-g{offset}"
            case = offset % 4
            beam_equal = case in (0, 1)
            composite_equal = case == 1
            tie_break = case == 2
            reversal = case == 3
            group_candidates = []
            for candidate_id in range(4):
                matched = (rollout * 16 + offset * 4 + candidate_id) % 5
                if beam_equal:
                    beam_raw, beam_utility = 0.0, 0.0
                elif tie_break:
                    beam_raw = float(4 if candidate_id in (0, 1) else 0)
                    beam_utility = 0.5 if candidate_id in (0, 1) else 0.0
                else:
                    beam_raw = float(4 if candidate_id == 0 else 0)
                    beam_utility = 0.5 if candidate_id == 0 else 0.0
                cot = 0.5 if composite_equal else candidate_id / 3
                beam_contribution = 0.6 * beam_utility
                cot_contribution = 0.4 * cot
                reward = beam_contribution + cot_contribution
                advantage = 0.0 if composite_equal else (-1.0, -0.3, 0.3, 1.0)[candidate_id]
                candidate = {
                    "group_id": group_id,
                    "candidate_id": candidate_id,
                    "completion": "synthetic",
                    "completion_length": 12 + candidate_id,
                    "beam_raw": beam_raw,
                    "beam_utility": beam_utility,
                    "beam_contribution": beam_contribution,
                    "cot_utility": cot,
                    "cot_contribution": cot_contribution,
                    "composite_reward": reward,
                    "final_sequence_advantage": advantage,
                    "matched_interest_count": matched,
                    "raw_n": max(1, matched),
                    "grounded_n": min(matched, 2),
                    "grounding_coverage": min(matched, 2) / max(1, matched),
                    "parser_success": not (rollout == 0 and offset == 0 and candidate_id == 0),
                    "Q": 0.2 if rollout == 5 and candidate_id == 3 else 0.0,
                    "pred_interest_units": [], "match_details": [],
                    "unmatched_pred_indices": [], "unmatched_gold_indices": [],
                    "unmatched_pred_best_alternatives": [], "unmatched_gold_best_alternatives": [],
                }
                group_candidates.append(candidate)
                candidates.append(candidate)
            groups.append({
                "group_id": group_id,
                "gold_interest_units": [],
                "beam_raw_vector": [row["beam_raw"] for row in group_candidates],
                "beam_utility_vector": [row["beam_utility"] for row in group_candidates],
                "cot_utility_vector": [row["cot_utility"] for row in group_candidates],
                "Q_vector": [row["Q"] for row in group_candidates],
                "composite_reward_vector": [row["composite_reward"] for row in group_candidates],
                "final_advantage_vector": [row["final_sequence_advantage"] for row in group_candidates],
                "beam_all_equal": beam_equal, "cot_all_equal": composite_equal,
                "composite_all_equal": composite_equal, "beam_active": not beam_equal,
                "cot_active": not composite_equal, "composite_active": not composite_equal,
                "composite_zero_std": composite_equal,
                "matched_candidate_count": sum(row["matched_interest_count"] > 0 for row in group_candidates),
                "quality_active_candidate_count": sum(row["Q"] > 0 for row in group_candidates),
                "beam_top_set": [0, 1, 2, 3] if beam_equal else [0, 1] if tie_break else [0],
                "beam_stable_winner": 0, "composite_winner": 1 if tie_break else 3,
                "top_set_tie_break": tie_break, "strict_beam_reversal": reversal,
                "pairwise_interest_similarity": 0.0, "unique_interest_set_count": 4,
            })
        events.append({
            "synthetic": True, "step": rollout * 2, "rollout_id": rollout,
            "route": "think", "g": 4, "groups": groups,
            "candidates": candidates,
            "advantages": [row["final_sequence_advantage"] for row in candidates],
        })
    parameter_audit = {
        "optimizer_steps": 12,
        "runtime_optimizer_steps": 12,
        "BASE_DELTA": 0,
        "BASE_CHANGED": False,
        "base_version_changed_count": 0,
        "base_requires_grad_count": 0,
        "LORA_CHANGED": True,
        "lora_total_l2_delta": 0.25,
        "lora_max_abs_delta": 0.01,
        "lora_parameter_tensor_count": 2,
        "finite": True,
        "runtime_error": False,
    }
    return {
        "manifest": manifest,
        "metrics": metrics,
        "rollouts": rollouts,
        "composite_events": events,
        "parameter_audit": parameter_audit,
    }
