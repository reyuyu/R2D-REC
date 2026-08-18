"""Compact rank-0 forensics and the preregistered step-200 gate."""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


BEAM_SIZE = 32
GATE_STEP = 200


def _mean(values: Iterable[float]):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _rate(values: Iterable[bool]):
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else None


def _groups(records: Sequence[dict], size: int):
    if len(records) % size:
        raise ValueError(f"record count {len(records)} is not divisible by G={size}")
    return [records[start:start + size] for start in range(0, len(records), size)]


def think_forensic_rows(records: Sequence[dict], step: int, rollout_id: int):
    rows = []
    for group in _groups(records, 4):
        rewards = [float(item["primary_reward"]) for item in group]
        group_mean = statistics.fmean(rewards)
        group_std = statistics.pstdev(rewards)
        group_zero = group_std == 0.0
        group_all_zero = all(value == 0.0 for value in rewards)
        for candidate_id, item in enumerate(group):
            diagnostic = item.get("diagnostic_only") or {}
            raw_n = int(item["raw_interest_n"])
            grounded_n = int(diagnostic.get("grounded_n", 0))
            rows.append({
                "type": "think_candidate",
                "step": int(step),
                "rollout_id": int(rollout_id),
                "group_id": item["group_id"],
                "candidate_id": candidate_id,
                "domain": item.get("target_domain"),
                "gold_count": int(item.get("gold_count", 0)),
                "unique_gold_A": int(item.get("unique_gold_a", 0)),
                "primary_reward": float(item["primary_reward"]),
                "group_primary_mean": group_mean,
                "group_primary_std": group_std,
                "group_primary_zero_std": group_zero,
                "group_primary_all_zero": group_all_zero,
                "Raw_N": raw_n,
                "S_N": float(item["s_n"]),
                "unique_valid_target_A": int(item["unique_valid_target_a"]),
                "D_A": float(item["d_a"]),
                "simple_S_aux": float(item["simple_s_aux"]),
                "simple_A_aux": float(item["simple_a_aux"]),
                "simple_branch": item["simple_branch"],
                "completion_length": int(item.get("completion_length", 0)),
                "closed": bool(item.get("closed", False)),
                "Beam": {
                    "exact": int(diagnostic.get("exact", 0)),
                    "ab": int(diagnostic.get("ab", 0)),
                    "a": int(diagnostic.get("a", 0)),
                    "invalid_count_for_this_Beam32_task": int(item.get("beam_invalid", 0)),
                },
                "Grounded_N": grounded_n,
                "Coverage": grounded_n / raw_n if raw_n > 0 else None,
                "Raw_Grounded_gap": raw_n - grounded_n,
                "fake_sid_count": int(diagnostic.get("fake_or_ungrounded_sid_count", 0)),
                "parser_success": bool((item.get("parsed") or {}).get("parser_success", False)),
            })
    return rows


def nothink_forensic_rows(
    records: Sequence[dict], plans: Sequence[object], step: int, rollout_id: int
):
    rows = []
    grouped = _groups(records, 8)
    if len(grouped) != len(plans):
        raise ValueError("NoThink records and rescue plans are not aligned")
    for group, plan in zip(grouped, plans):
        rewards = [float(item["primary_reward"]) for item in group]
        rows.append({
            "type": "nothink_group",
            "step": int(step),
            "rollout_id": int(rollout_id),
            "group_id": group[0]["group_id"],
            "domain": group[0].get("target_domain"),
            "gold_count": int(group[0].get("gold_count", 0)),
            "unique_gold_A": int(group[0].get("unique_gold_a", len(group[0].get("gold_as", ())))),
            "candidate_rewards": rewards,
            "predicted_A": list(plan.predicted_as),
            "gold_A": list(group[0].get("gold_as", ())),
            "primary_mean": statistics.fmean(rewards),
            "primary_std": statistics.pstdev(rewards),
            "zero_std": statistics.pstdev(rewards) == 0.0,
            "all_zero": all(value == 0.0 for value in rewards),
            "A_concentration": float(plan.concentration),
            "rescue_active": bool(plan.active),
            "lambda_A": float(plan.lambda_a),
            "combined_coefficient": float(plan.coefficient),
            "frequency_weights": list(plan.frequency_weights),
            "any_Gold_A": any(value >= 0.5 for value in rewards),
            "any_Gold_AB": any(value >= 2.0 for value in rewards),
            "any_Exact": any(value == 8.0 for value in rewards),
            "valid_sid_count": sum(value != -1.0 for value in rewards),
            "wrong_domain_count": sum(value == -0.25 for value in rewards),
            "invalid_count": sum(value == -1.0 for value in rewards),
        })
    return rows


def append_forensic_rows(monitor, rows: Sequence[dict]):
    if not monitor.enabled or monitor.rank != 0 or not rows:
        return False

    def write():
        path = monitor.run_dir / "simple_forensic.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")

    return monitor._guard(write)


def _recent(rows: Sequence[dict], step: int, width: int):
    return [row for row in rows if step - width <= int(row.get("step", -1)) < step]


def _probe_declines(probes: Sequence[dict], step: int):
    by_step = {}
    for row in probes:
        row_step = int(row.get("step", -1))
        if row_step not in (0, step):
            continue
        think = (row.get("think") or {}).get("reward_mean")
        no = (row.get("nothink") or {}).get("reward_mean")
        values = [float(value) for value in (think, no) if value is not None]
        by_step[(row_step, row.get("target_domain"))] = _mean(values)
    declines = {}
    for domain in ("video", "living", "prod", "ad"):
        baseline = by_step.get((0, domain))
        current = by_step.get((step, domain))
        declines[domain] = bool(
            baseline is not None and baseline > 0.0 and current is not None and current < 0.5 * baseline
        )
    return declines


def evaluate_gate_data(
    think_rows: Sequence[dict],
    nothink_rows: Sequence[dict],
    policy_rows: Sequence[dict],
    probes: Sequence[dict],
    step: int = GATE_STEP,
):
    think = [row for row in think_rows if int(row.get("step", -1)) < step]
    no = [row for row in nothink_rows if int(row.get("step", -1)) < step]
    recent_think = _recent(think, step, 50)
    recent_no = _recent(no, step, 50)
    recent_policy = _recent(policy_rows, step, 20)

    cumulative_invalid = sum(int((row.get("Beam") or {}).get("invalid_count_for_this_Beam32_task", 0)) for row in think)
    recent_invalid = sum(int((row.get("Beam") or {}).get("invalid_count_for_this_Beam32_task", 0)) for row in recent_think)
    no_rewards = [float(value) for row in no for value in row.get("candidate_rewards", ())]
    recent_no_rewards = [float(value) for row in recent_no for value in row.get("candidate_rewards", ())]
    raw_values = [int(row["Raw_N"]) for row in recent_think]
    probe_declines = _probe_declines(probes, step)
    metrics = {
        "think_candidate_count": len(think),
        "think_recent_candidate_count": len(recent_think),
        "think_beam_invalid_rate_cumulative": cumulative_invalid / (BEAM_SIZE * len(think)) if think else None,
        "think_beam_invalid_rate_recent50_steps": recent_invalid / (BEAM_SIZE * len(recent_think)) if recent_think else None,
        "think_raw_n_mean_recent50_steps": _mean(raw_values),
        "think_raw_n_le_1_rate_recent50_steps": _rate(value <= 1 for value in raw_values),
        "think_primary_zero_std_rate": _rate(row.get("group_primary_zero_std") for row in think),
        "think_primary_all_zero_rate": _rate(row.get("group_primary_all_zero") for row in think),
        "think_grounding_coverage_mean": _mean(row.get("Coverage") for row in think),
        "nothink_group_count": len(no),
        "nothink_primary_zero_std_rate": _rate(row.get("zero_std") for row in no),
        "nothink_all_zero_rate": _rate(row.get("all_zero") for row in no),
        "nothink_rescue_active_rate": _rate(row.get("rescue_active") for row in no),
        "nothink_a_concentration_mean": _mean(row.get("A_concentration") for row in no),
        "nothink_valid_sid_rate": _rate(value != -1.0 for value in no_rewards),
        "nothink_gold_a_or_better_candidate_rate": _rate(value >= 0.5 for value in no_rewards),
        "nothink_wrong_domain_rate": _rate(value == -0.25 for value in no_rewards),
        "nothink_wrong_domain_rate_recent50_steps": _rate(value == -0.25 for value in recent_no_rewards),
        "policy_approx_kl_mean_recent20_steps": _mean(row.get("approx_kl") for row in recent_policy),
        "policy_clip_fraction_mean_recent20_steps": _mean(row.get("clip_fraction") for row in recent_policy),
        "policy_observed_through_step": max((int(row.get("step", -1)) for row in policy_rows if int(row.get("step", -1)) < step), default=None),
        "probe_primary_decline_over_50pct": probe_declines,
        "probe_decline_domain_count": sum(probe_declines.values()),
    }

    catastrophic = []
    def stop(code, condition):
        if condition:
            catastrophic.append(code)

    stop("think_beam_invalid_cumulative_gt_2pct", (metrics["think_beam_invalid_rate_cumulative"] or 0.0) > 0.02)
    stop("think_beam_invalid_recent50_gt_5pct", (metrics["think_beam_invalid_rate_recent50_steps"] or 0.0) > 0.05)
    stop("think_raw_n_mean_recent50_lt_1_5", metrics["think_raw_n_mean_recent50_steps"] is not None and metrics["think_raw_n_mean_recent50_steps"] < 1.5)
    stop("think_raw_n_le_1_recent50_gt_60pct", (metrics["think_raw_n_le_1_rate_recent50_steps"] or 0.0) > 0.60)
    stop("nothink_valid_sid_rate_lt_95pct", metrics["nothink_valid_sid_rate"] is not None and metrics["nothink_valid_sid_rate"] < 0.95)
    stop("nothink_wrong_domain_recent50_gt_40pct", (metrics["nothink_wrong_domain_rate_recent50_steps"] or 0.0) > 0.40)
    stop("policy_approx_kl_recent20_gt_0_03", (metrics["policy_approx_kl_mean_recent20_steps"] or 0.0) > 0.03)
    stop("policy_clip_recent20_gt_0_20", (metrics["policy_clip_fraction_mean_recent20_steps"] or 0.0) > 0.20)

    serious = []
    if (metrics["think_primary_zero_std_rate"] or 0.0) > 0.70:
        serious.append("think_primary_zero_std_gt_70pct")
    if (metrics["nothink_primary_zero_std_rate"] or 0.0) > 0.55:
        serious.append("nothink_primary_zero_std_gt_55pct")
    if metrics["nothink_gold_a_or_better_candidate_rate"] is not None and metrics["nothink_gold_a_or_better_candidate_rate"] < 0.12:
        serious.append("nothink_gold_a_or_better_lt_12pct")
    if (metrics["nothink_wrong_domain_rate"] or 0.0) > 0.30:
        serious.append("nothink_wrong_domain_gt_30pct")
    if (metrics["think_beam_invalid_rate_cumulative"] or 0.0) > 0.01:
        serious.append("think_beam_invalid_gt_1pct")
    if metrics["probe_decline_domain_count"] >= 3:
        serious.append("probe_3_of_4_primary_decline_gt_50pct")

    warnings = []
    if (metrics["think_primary_zero_std_rate"] or 0.0) > 0.65:
        warnings.append("think_primary_zero_std_gt_65pct")
    invalid = metrics["think_beam_invalid_rate_cumulative"] or 0.0
    if 0.005 < invalid <= 0.02:
        warnings.append("think_beam_invalid_between_0_5_and_2pct")
    if (metrics["nothink_primary_zero_std_rate"] or 0.0) > 0.45:
        warnings.append("nothink_primary_zero_std_gt_45pct")
    gold_rate = metrics["nothink_gold_a_or_better_candidate_rate"]
    if gold_rate is not None and gold_rate < 0.15:
        warnings.append("nothink_gold_a_or_better_lt_15pct")
    warnings.extend(code for code in serious if code not in warnings)

    reasons = list(catastrophic)
    if len(serious) >= 2:
        reasons.append("combined_serious_outcomes_at_least_2")
    if reasons:
        decision = "STOP"
    elif warnings:
        decision = "WARN"
    else:
        decision = "PASS"
    return {
        "experiment": "GR_REC_DSR_Simple_Ablation_v1",
        "gate_step": int(step),
        "decision": decision,
        "decision_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "reasons": reasons,
        "warnings": warnings,
        "serious_outcomes": serious,
        "metrics": metrics,
        "windows": {"recent_think_steps": [step - 50, step - 1], "recent_nothink_steps": [step - 50, step - 1], "recent_policy_steps": [step - 20, step - 1]},
        "training_action": "stop_after_checkpoint_200" if decision == "STOP" else "continue_same_process",
    }


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_gate200_report(run_dir: str | Path, step: int = GATE_STEP):
    run_dir = Path(run_dir)
    forensic = _read_jsonl(run_dir / "simple_forensic.jsonl")
    report = evaluate_gate_data(
        [row for row in forensic if row.get("type") == "think_candidate"],
        [row for row in forensic if row.get("type") == "nothink_group"],
        _read_jsonl(run_dir / "dsr_steps.jsonl"),
        _read_jsonl(run_dir / "probes.jsonl"),
        step=step,
    )
    target = run_dir / "gate200_report.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return report


__all__ = [
    "append_forensic_rows", "evaluate_gate_data", "nothink_forensic_rows",
    "think_forensic_rows", "write_gate200_report",
]
