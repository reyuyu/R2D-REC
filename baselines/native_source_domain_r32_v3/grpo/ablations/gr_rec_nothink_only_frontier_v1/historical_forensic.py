"""CPU-only replay of Frontier v1 over immutable historical NoThink traces."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from grpo_sid import parse_sid
from nothink_hierarchical_credit import hierarchy_state

try:
    from .format_validator import validate_nothink_completion
    from .frontier_credit import (
        FORMAT_ADV_TOTAL,
        FRONTIER_NEGATIVE,
        plan_frontier_credits,
    )
except ImportError:
    from format_validator import validate_nothink_completion
    from frontier_credit import FORMAT_ADV_TOTAL, FRONTIER_NEGATIVE, plan_frontier_credits


DEFAULT_SOURCES = (
    (
        "hier_v1_partial_step1297",
        "/data/GRPO/runs/GR-REC-NOTHINK-ONLY-HIER-G8BASE-E1-20260821",
    ),
    (
        "joint_clamp_bridge",
        "/data/GRPO/runs/GR-REC-CLAMP-BRIDGE-V1-G8BASE-FORMAL1500-20260821",
    ),
)
REWARD_LEVELS = (-1.0, -0.25, 0.0, 0.5, 2.0, 8.0)
STAGE_NAMES = ("domain", "a", "b", "c")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_nothink_traces(run_dir: Path) -> list[dict]:
    path = run_dir / "traces" / "traces.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing trace source: {path}")
    traces = [row for row in read_jsonl(path) if row.get("route") == "no_think"]
    for row in traces:
        if row.get("scope") != "global_group" or len(row.get("candidates", ())) != 8:
            raise ValueError(f"trace {row.get('group_id')} is not one immutable NoThink G8")
    return traces


def load_dataset(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    mapping = {
        row["recommendation_group_id"]: row
        for row in rows
        if row.get("route") == "no_think"
    }
    if not mapping:
        raise ValueError("NoThink dataset mapping is empty")
    return mapping


def _reward_key(value: float) -> str:
    return str(float(value))


def _reward_topology(rewards: list[float]) -> dict:
    counts = Counter(rewards)
    maximum = max(counts.values())
    modes = sorted(value for value, count in counts.items() if count == maximum)
    uniform = rewards[0] if len(counts) == 1 else None
    return {
        "reward_pattern": rewards,
        "reward_level_counts": {_reward_key(level): counts[level] for level in REWARD_LEVELS},
        "uniform_reward": uniform,
        "dominant_reward": modes[0] if len(modes) == 1 else None,
        "dominant_count": maximum,
        "dominant_tie": len(modes) != 1,
        "distinct_reward_levels": len(counts),
        "exact_containing": counts[8.0] > 0,
        "mixed_hierarchy": len(counts) > 1,
        "dead_zero": rewards == [0.0] * 8,
    }


def analyze_trace(trace: dict, source: str) -> dict:
    candidates = trace["candidates"]
    gold_sids = [parse_sid(value) for value in trace["gold_sids"]]
    gold_sids = [value for value in gold_sids if value is not None]
    if not gold_sids:
        raise ValueError(f"group {trace['group_id']} has no valid gold SID")
    target_domain = gold_sids[0][0]
    if any(sid[0] != target_domain for sid in gold_sids):
        raise ValueError(f"group {trace['group_id']} has mixed gold domains")
    validations = [validate_nothink_completion(item.get("completion", "")) for item in candidates]
    states = [
        hierarchy_state(validation.parsed_sid, gold_sids, target_domain)
        for validation in validations
    ]
    plan = plan_frontier_credits(states, [item.valid for item in validations])
    rewards = [float(item["reward"]) for item in candidates]
    effective_rewards = [
        reward if validation.valid else -1.0
        for reward, validation in zip(rewards, validations)
    ]
    success_counts = [
        sum(
            validation.valid and flag
            for validation, flag in zip(
                validations,
                (
                    state.domain_correct if column == 0 else
                    state.a_correct if column == 1 else
                    state.ab_correct if column == 2 else state.exact
                    for state in states
                ),
            )
        )
        for column in range(4)
    ]
    frontier_counts = list(plan.frontier_negative_counts)
    format_signed = sum(not item.valid for item in validations) * FORMAT_ADV_TOTAL
    stage_signed = [
        sum(candidate.credits[column] for candidate in plan.candidates)
        for column in range(4)
    ]
    stage_abs = [
        sum(abs(candidate.credits[column]) for candidate in plan.candidates)
        for column in range(4)
    ]
    format_abs = abs(format_signed)
    token_credit_rows = []
    for item, validation, candidate in zip(candidates, validations, plan.candidates):
        token_credit_rows.append({
            "candidate_id": item.get("candidate_id"),
            "original_reward": float(item["reward"]),
            "effective_reward": (
                float(item["reward"]) if validation.valid else -1.0
            ),
            "format_valid": validation.valid,
            "format_violation_reason": validation.reason,
            "commitment_mode": validation.mode,
            "stage_credits": list(candidate.credits),
            "stage_credit_kinds": list(candidate.kinds),
            "format_credit_total": None if validation.valid else FORMAT_ADV_TOTAL,
        })
    result = {
        "source": source,
        "run_rollout_id": trace.get("rollout_id"),
        "step": trace.get("step"),
        "group_id": trace["group_id"],
        "target_domain": target_domain,
        "gold_sids": trace["gold_sids"],
        **_reward_topology(rewards),
        "effective_reward_pattern": effective_rewards,
        "format_valid_count": sum(item.valid for item in validations),
        "format_violation_count": sum(not item.valid for item in validations),
        "format_violation_reason_counts": dict(Counter(
            item.reason for item in validations if not item.valid
        )),
        "stage_success_counts": dict(zip(STAGE_NAMES, success_counts)),
        "frontier_counts": dict(zip(STAGE_NAMES, frontier_counts)),
        "frontier_active": dict(zip(STAGE_NAMES, plan.frontier_active)),
        "positive_active": dict(zip(STAGE_NAMES, plan.positive_active)),
        "zero_signal_taxonomy": plan.taxonomy,
        "credit_mass": {
            "sum_abs_token_credit": format_abs + sum(stage_abs),
            "sum_signed_token_credit": format_signed + sum(stage_signed),
            "format": {"signed": format_signed, "absolute": format_abs},
            **{
                name: {"signed": stage_signed[index], "absolute": stage_abs[index]}
                for index, name in enumerate(STAGE_NAMES)
            },
        },
        "c_frontier_heavy": frontier_counts[3] >= 6,
        "candidates": token_credit_rows,
    }
    return result


def aggregate_groups(groups: list[dict]) -> dict:
    candidate_count = len(groups) * 8
    format_reasons = Counter()
    frontier_candidates = Counter()
    frontier_groups = Counter()
    success_candidates = Counter()
    reward_levels = Counter()
    dominant = Counter()
    c_histogram = Counter()
    taxonomy = Counter()
    contribution_signed = Counter()
    contribution_abs = Counter()
    format_violations = 0
    patterns = Counter()
    for group in groups:
        format_violations += group["format_violation_count"]
        format_reasons.update(group["format_violation_reason_counts"])
        reward_levels.update(group["reward_pattern"])
        if group["dominant_tie"]:
            dominant["tie"] += 1
        else:
            dominant[_reward_key(group["dominant_reward"])] += 1
        patterns["exact_containing"] += group["exact_containing"]
        patterns["mixed_hierarchy"] += group["mixed_hierarchy"]
        patterns["dead_zero"] += group["dead_zero"]
        if group["uniform_reward"] is not None:
            patterns[f"uniform_{_reward_key(group['uniform_reward'])}"] += 1
        taxonomy[group["zero_signal_taxonomy"]] += 1
        for name in STAGE_NAMES:
            count = group["frontier_counts"][name]
            frontier_candidates[name] += count
            frontier_groups[name] += count > 0
            success_candidates[name] += group["stage_success_counts"][name]
        c_histogram[str(group["frontier_counts"]["c"])] += 1
        for name in ("format",) + STAGE_NAMES:
            contribution_signed[name] += group["credit_mass"][name]["signed"]
            contribution_abs[name] += group["credit_mass"][name]["absolute"]
    group_count = len(groups)
    return {
        "group_count": group_count,
        "candidate_count": candidate_count,
        "format": {
            "valid_candidate_count": candidate_count - format_violations,
            "violation_candidate_count": format_violations,
            "violation_candidate_rate": format_violations / candidate_count if candidate_count else 0.0,
            "reason_counts": dict(format_reasons),
        },
        "reward_level_candidate_counts": {
            _reward_key(level): reward_levels[level] for level in REWARD_LEVELS
        },
        "group_patterns": dict(patterns),
        "dominant_reward_group_counts": dict(dominant),
        "frontier_candidate_exposure": {
            name: {
                "count": frontier_candidates[name],
                "rate": frontier_candidates[name] / candidate_count if candidate_count else 0.0,
            }
            for name in STAGE_NAMES
        },
        "frontier_group_exposure": {
            name: {
                "count": frontier_groups[name],
                "rate": frontier_groups[name] / group_count if group_count else 0.0,
            }
            for name in STAGE_NAMES
        },
        "stage_success_candidate_counts": dict(success_candidates),
        "c_frontier_count_per_g8_histogram": {
            str(index): c_histogram[str(index)] for index in range(9)
        },
        "c_frontier_heavy_group_count": sum(group["c_frontier_heavy"] for group in groups),
        "zero_signal_taxonomy_counts": dict(taxonomy),
        "credit_contribution_totals": {
            name: {
                "signed": contribution_signed[name],
                "absolute": contribution_abs[name],
            }
            for name in ("format",) + STAGE_NAMES
        },
        "sum_abs_token_credit": sum(
            group["credit_mass"]["sum_abs_token_credit"] for group in groups
        ),
        "sum_signed_token_credit": sum(
            group["credit_mass"]["sum_signed_token_credit"] for group in groups
        ),
    }


def _pick_best(groups: Iterable[dict], predicate, score) -> dict | None:
    matches = [group for group in groups if predicate(group)]
    return max(matches, key=score) if matches else None


def select_audit_groups(groups: list[dict], dataset: dict[str, dict], limit: int = 12) -> list[dict]:
    selected: list[tuple[str, dict]] = []
    seen = set()

    def add(label: str, group: dict | None):
        if group is not None and group["group_id"] not in seen and len(selected) < limit:
            seen.add(group["group_id"])
            selected.append((label, group))

    add("format_violation", _pick_best(
        groups, lambda g: g["format_violation_count"] > 0,
        lambda g: (g["format_violation_count"], g["credit_mass"]["sum_abs_token_credit"]),
    ))
    for stage in STAGE_NAMES:
        add(f"{stage}_frontier_heavy", _pick_best(
            groups, lambda g, name=stage: g["frontier_counts"][name] > 0,
            lambda g, name=stage: (
                g["frontier_counts"][name], g["credit_mass"][name]["absolute"]
            ),
        ))
    add("mixed_hierarchy", _pick_best(
        groups, lambda g: g["mixed_hierarchy"],
        lambda g: (g["distinct_reward_levels"], g["credit_mass"]["sum_abs_token_credit"]),
    ))
    add("exact_containing", _pick_best(
        groups, lambda g: g["exact_containing"],
        lambda g: (g["reward_level_counts"]["8.0"], g["distinct_reward_levels"]),
    ))
    add("dead_zero_bridge", _pick_best(
        groups, lambda g: g["dead_zero"], lambda g: g["step"] or -1,
    ))
    for group in sorted(
        groups,
        key=lambda g: (
            g["credit_mass"]["sum_abs_token_credit"],
            g["distinct_reward_levels"],
        ),
        reverse=True,
    ):
        add("high_total_credit_mass", group)
        if len(selected) >= min(10, limit):
            break

    output = []
    for label, group in selected:
        row = dataset.get(group["group_id"])
        if row is None:
            raise KeyError(f"selected group missing from dataset: {group['group_id']}")
        output.append({
            "selection_label": label,
            "group_id": group["group_id"],
            "source": group["source"],
            "observed_rollout_id": group["run_rollout_id"],
            "observed_step": group["step"],
            "target_domain": row["target_domain"],
            "gold_sids": row["all_gold_sids"],
            "prompt": row["prompt"],
            "observed_reward_pattern": group["reward_pattern"],
            "observed_format_violation_count": group["format_violation_count"],
            "observed_frontier_counts": group["frontier_counts"],
            "observed_credit_mass": group["credit_mass"],
        })
    return output


def build_report(source_specs, dataset_path: Path, implementation_commit: str) -> dict:
    dataset = load_dataset(dataset_path)
    groups = []
    source_summaries = {}
    sources = []
    for source, run_dir_text in source_specs:
        run_dir = Path(run_dir_text)
        traces = load_nothink_traces(run_dir)
        analyzed = [analyze_trace(trace, source) for trace in traces]
        groups.extend(analyzed)
        source_summaries[source] = aggregate_groups(analyzed)
        sources.append({
            "name": source,
            "run_dir": str(run_dir),
            "trace_path": str(run_dir / "traces" / "traces.jsonl"),
            "nothink_g8_count": len(analyzed),
        })
    heavy_groups = [
        {
            "source": group["source"],
            "group_id": group["group_id"],
            "step": group["step"],
            "reward_pattern": group["reward_pattern"],
            "c_frontier_count": group["frontier_counts"]["c"],
            "c_credit_mass": group["credit_mass"]["c"],
            "full_credit_mass": {
                "signed": group["credit_mass"]["sum_signed_token_credit"],
                "absolute": group["credit_mass"]["sum_abs_token_credit"],
            },
        }
        for group in groups
        if group["c_frontier_heavy"]
    ]
    group_summaries = [
        {key: value for key, value in group.items() if key != "candidates"}
        for group in groups
    ]
    return {
        "experiment": "GR_REC_NoThinkOnly_Frontier_v1",
        "phase": "CPU historical rollout forensic",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": implementation_commit,
        "gpu_used": False,
        "run_files_modified": False,
        "coverage_note": (
            "The joint run retained complete NoThink global-group traces. "
            "The stopped Hier partial run retained only its configured recent trace sample; "
            "aggregates describe available immutable traces, not every generated training group."
        ),
        "dataset_path": str(dataset_path),
        "sources": sources,
        "summary": aggregate_groups(groups),
        "summary_by_source": source_summaries,
        "c_frontier_heavy_groups": heavy_groups,
        "selected_paired_audit_groups": select_audit_groups(groups, dataset),
        "groups": group_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"),
    )
    parser.add_argument("--implementation-commit", default="12240c1bf45c22ac6222b1bbe792b8332428beea")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = build_report(DEFAULT_SOURCES, args.dataset, args.implementation_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    print(json.dumps({
        "output": str(args.output),
        "groups": summary["group_count"],
        "candidates": summary["candidate_count"],
        "format_violation_rate": summary["format"]["violation_candidate_rate"],
        "frontier_candidate_exposure": summary["frontier_candidate_exposure"],
        "c_frontier_heavy_group_count": summary["c_frontier_heavy_group_count"],
        "selected_audit_groups": len(report["selected_paired_audit_groups"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
