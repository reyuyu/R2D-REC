"""CPU-only audit of TrueRec Formal4096 training dynamics for groups 0..1999."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


RUN_ID = "TRUEREC-V1-BETA-FORMAL4096-4GPU-E1"
GROUP_LIMIT = 2000
POSITIONS = ("A", "B", "C")
HPR_TRIGGERS = ("HPR_A", "HPR_B", "HPR_C", "HPR_NONE")
BUCKETS = (
    (0, 255), (256, 511), (512, 767), (768, 1023),
    (1024, 1279), (1280, 1535), (1536, 1791), (1792, 1999),
)
CUMULATIVE_STEPS = (256, 512, 768, 1024, 1280, 1536, 1792, 2000)
CREDIT_EPS = 1e-12
HPR_TREND_THRESHOLD = 0.02
FINE_SIGNAL_THRESHOLD = 0.005
PROBE_PROGRESS_THRESHOLD = 0.005


class AuditError(RuntimeError):
    pass


def _rate(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 8) if values else 0.0


def read_jsonl_prefix(path: Path, limit: int = GROUP_LIMIT) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # A concurrently appended final line may be incomplete, but only after the audit range.
                if len(selected) >= limit:
                    break
                raise AuditError(f"INVALID_JSON_BEFORE_LIMIT={path}:{line_number}") from exc
            index = int(row["group_index"])
            if index >= limit:
                continue
            if index in selected:
                raise AuditError(f"DUPLICATE_GROUP_INDEX={path}:{index}")
            selected[index] = row
    expected = set(range(limit))
    actual = set(selected)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise AuditError(f"GROUP_RANGE_INCOMPLETE={path}:missing={missing}:extra={extra}")
    return [selected[index] for index in range(limit)]


def _canonical_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_and_join(
    group_rows: list[dict[str, Any]], explain_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(group_rows) != GROUP_LIMIT or len(explain_rows) != GROUP_LIMIT:
        raise AuditError("ROW_COUNT_NOT_2000")
    joined = []
    for expected_index, (group, explain) in enumerate(zip(group_rows, explain_rows)):
        if int(group["group_index"]) != expected_index or int(explain["group_index"]) != expected_index:
            raise AuditError(f"GROUP_ORDER_MISMATCH={expected_index}")
        if group["recommendation_group_id"] != explain["recommendation_group_id"]:
            raise AuditError(f"GROUP_ID_MISMATCH={expected_index}")
        candidates = explain["candidates"]
        if [int(item["candidate_index"]) for item in candidates] != list(range(8)):
            raise AuditError(f"CANDIDATE_ORDER_NOT_G8={expected_index}")
        for candidate in candidates:
            tokens = candidate["action_tokens"]
            if [item["position_name"] for item in tokens] != list(POSITIONS):
                raise AuditError(f"ACTION_POSITION_NOT_ABC={expected_index}")
        if explain["hpr_trigger"] not in HPR_TRIGGERS:
            raise AuditError(f"UNKNOWN_HPR_TRIGGER={explain['hpr_trigger']}")
        joined.append({"group": group, "explain": explain})
    return joined


def summarize_training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        raise AuditError("EMPTY_TRAINING_WINDOW")
    hpr = Counter(row["explain"]["hpr_trigger"] for row in rows)
    any_a = sum(any(c["A_hit"] for c in row["explain"]["candidates"]) for row in rows)
    any_ab = sum(any(c["AB_hit"] for c in row["explain"]["candidates"]) for row in rows)
    any_exact = sum(any(c["exact"] for c in row["explain"]["candidates"]) for row in rows)
    sign_counts = {position: Counter() for position in POSITIONS}
    for row in rows:
        for candidate in row["explain"]["candidates"]:
            for token in candidate["action_tokens"]:
                value = float(token["effective_signed_credit"])
                sign = "positive" if value > CREDIT_EPS else "negative" if value < -CREDIT_EPS else "zero"
                sign_counts[token["position_name"]][sign] += 1
    total_tokens_per_position = n * 8
    output: dict[str, Any] = {
        "groups": n,
        **{f"{trigger}_rate": _rate(hpr[trigger], n) for trigger in HPR_TRIGGERS},
        "GROUP_ANY_A_rate": _rate(any_a, n),
        "GROUP_ANY_AB_rate": _rate(any_ab, n),
        "GROUP_ANY_EXACT_rate": _rate(any_exact, n),
        "format_valid_rate": _mean(float(row["group"]["format_valid_rate"]) for row in rows),
        "wrong_history_copy_rate": _mean(float(row["group"]["wrong_history_copy_rate"]) for row in rows),
    }
    for position in POSITIONS:
        counts = sign_counts[position]
        if sum(counts.values()) != total_tokens_per_position:
            raise AuditError(f"TOKEN_COUNT_MISMATCH={position}")
        output[f"{position}_positive_tokens_per_group"] = _rate(counts["positive"], n)
        output[f"{position}_negative_tokens_per_group"] = _rate(counts["negative"], n)
        output[f"{position}_nonzero_credit_token_rate"] = _rate(
            counts["positive"] + counts["negative"], total_tokens_per_position
        )
    return output


def training_tables(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    buckets = {}
    for start, end in BUCKETS:
        subset = rows[start : end + 1]
        metrics = summarize_training(subset)
        metrics.update({"group_index_start": start, "group_index_end": end})
        buckets[f"{start}-{end}"] = metrics
    cumulative = {}
    for step in CUMULATIVE_STEPS:
        metrics = summarize_training(rows[:step])
        cumulative[str(step)] = {
            key: value for key, value in metrics.items()
            if key == "groups" or key.startswith("HPR_") or key.startswith("GROUP_ANY_")
        }
    return buckets, cumulative


def _read_jsonl_complete(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AuditError(f"INVALID_PROBE_JSON={path}:{line_number}") from exc
    return rows


def summarize_probe_step(directory: Path) -> dict[str, Any]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    groups = _read_jsonl_complete(directory / "groups.jsonl")
    explains = _read_jsonl_complete(directory / "explain.jsonl")
    if len(groups) != 20 or len(explains) != 20:
        raise AuditError(f"PROBE_NOT_20_GROUPS={directory}")
    group_ids = [row["recommendation_group_id"] for row in groups]
    if group_ids != [row["recommendation_group_id"] for row in explains] or len(set(group_ids)) != 20:
        raise AuditError(f"PROBE_GROUP_ALIGNMENT_FAIL={directory}")
    calculated = {
        "A_hit_rate": _mean(float(row["A_hit_rate"]) for row in groups),
        "AB_hit_rate": _mean(float(row["AB_hit_rate"]) for row in groups),
        "exact_rate": _mean(float(row["exact_rate"]) for row in groups),
    }
    for key, value in calculated.items():
        if abs(value - float(summary[key])) > 1e-8:
            raise AuditError(f"PROBE_SUMMARY_MISMATCH={directory}:{key}")
    hpr = Counter(row["hpr_trigger"] for row in groups)
    return {
        "probe_step": int(summary["probe_step"]),
        "groups": 20,
        **calculated,
        "GROUP_ANY_A": sum(float(row["A_hit_rate"]) > 0 for row in groups),
        "GROUP_ANY_AB": sum(float(row["AB_hit_rate"]) > 0 for row in groups),
        "GROUP_ANY_EXACT": sum(float(row["exact_rate"]) > 0 for row in groups),
        **{f"{trigger}_count": hpr[trigger] for trigger in HPR_TRIGGERS},
    }


def probe_table(probe_root: Path, max_step: int = GROUP_LIMIT) -> dict[str, Any]:
    steps = []
    for directory in probe_root.iterdir():
        if directory.is_dir() and directory.name.isdigit() and int(directory.name) <= max_step:
            steps.append(summarize_probe_step(directory))
    steps.sort(key=lambda row: row["probe_step"])
    if not steps or steps[0]["probe_step"] != 0:
        raise AuditError("PROBE_STEP0_MISSING")
    first, latest = steps[0], steps[-1]
    metric_keys = (
        "A_hit_rate", "AB_hit_rate", "exact_rate",
        "GROUP_ANY_A", "GROUP_ANY_AB", "GROUP_ANY_EXACT",
    )
    return {
        "steps": steps,
        "latest_step_le_2000": latest["probe_step"],
        "step0_vs_latest": {
            key: {
                "step0": first[key], "latest": latest[key],
                "delta": round(float(latest[key]) - float(first[key]), 8),
            }
            for key in metric_keys
        },
    }


def _window_weighted_metric(buckets: dict[str, Any], names: tuple[str, ...], key: str) -> float:
    numerator = sum(float(buckets[name][key]) * int(buckets[name]["groups"]) for name in names)
    denominator = sum(int(buckets[name]["groups"]) for name in names)
    return numerator / denominator


def classify_trends(buckets: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    early_names = ("0-255", "256-511")
    late_names = ("1536-1791", "1792-1999")
    early_hpr_a = _window_weighted_metric(buckets, early_names, "HPR_A_rate")
    late_hpr_a = _window_weighted_metric(buckets, late_names, "HPR_A_rate")
    hpr_delta = late_hpr_a - early_hpr_a
    hpr_trend = "DECREASING" if hpr_delta < -HPR_TREND_THRESHOLD else "INCREASING" if hpr_delta > HPR_TREND_THRESHOLD else "FLAT"

    position_deltas = {}
    for position in ("B", "C"):
        key = f"{position}_nonzero_credit_token_rate"
        early = _window_weighted_metric(buckets, early_names, key)
        late = _window_weighted_metric(buckets, late_names, key)
        position_deltas[position] = {"early": round(early, 8), "late": round(late, 8), "delta": round(late - early, 8)}
    early_fine = (position_deltas["B"]["early"] + position_deltas["C"]["early"]) / 2
    late_fine = (position_deltas["B"]["late"] + position_deltas["C"]["late"]) / 2
    fine_delta = late_fine - early_fine
    fine_trend = "IMPROVING" if fine_delta > FINE_SIGNAL_THRESHOLD else "DECREASING" if fine_delta < -FINE_SIGNAL_THRESHOLD else "FLAT"
    bc_clear = "YES" if all(position_deltas[p]["delta"] > FINE_SIGNAL_THRESHOLD for p in ("B", "C")) else "NO"

    probe_deltas = probes["step0_vs_latest"]
    rate_deltas = [
        float(probe_deltas[key]["delta"])
        for key in ("A_hit_rate", "AB_hit_rate", "exact_rate")
    ]
    count_deltas_as_rates = [
        float(probe_deltas[key]["delta"]) / 20.0
        for key in ("GROUP_ANY_A", "GROUP_ANY_AB", "GROUP_ANY_EXACT")
    ]
    probe_composite_delta = sum(rate_deltas + count_deltas_as_rates) / 6
    probe_trend = "IMPROVING" if probe_composite_delta > PROBE_PROGRESS_THRESHOLD else "DECREASING" if probe_composite_delta < -PROBE_PROGRESS_THRESHOLD else "FLAT"
    return {
        "HPR_A_TREND": hpr_trend,
        "FINE_GRAINED_SIGNAL_TREND": fine_trend,
        "PROBE_PROGRESS": probe_trend,
        "B_C_NONZERO_TRAINING_CREDIT_CLEARLY_INCREASED": bc_clear,
        "evidence": {
            "early_window": "0-511", "late_window": "1536-1999",
            "HPR_A_early": round(early_hpr_a, 8), "HPR_A_late": round(late_hpr_a, 8),
            "HPR_A_delta": round(hpr_delta, 8), "HPR_A_threshold": HPR_TREND_THRESHOLD,
            "B_C_nonzero_credit": position_deltas,
            "fine_signal_early": round(early_fine, 8), "fine_signal_late": round(late_fine, 8),
            "fine_signal_delta": round(fine_delta, 8), "fine_signal_threshold": FINE_SIGNAL_THRESHOLD,
            "probe_composite_delta": round(probe_composite_delta, 8),
            "probe_progress_threshold": PROBE_PROGRESS_THRESHOLD,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_review(path: Path, summary: dict[str, Any], probes: dict[str, Any]) -> None:
    trends = summary["mechanical_judgment"]
    evidence = trends["evidence"]
    latest = probes["latest_step_le_2000"]
    changes = probes["step0_vs_latest"]
    lines = [
        "TrueRec-GRPO Step0->2000 Training Dynamics Audit",
        "",
        "Scope",
        "- Read-only CPU audit of group_index 0..1999 from the running Formal4096 monitor files.",
        "- No training process, checkpoint, model, GPU, or optimizer was touched.",
        "- Token signs use the recorded effective_signed_credit; nonzero means abs(credit) > 1e-12.",
        "",
        "Mechanical results",
        f"- HPR_A_TREND={trends['HPR_A_TREND']}: {evidence['HPR_A_early']:.6f} in groups 0-511 vs {evidence['HPR_A_late']:.6f} in groups 1536-1999 (delta {evidence['HPR_A_delta']:+.6f}).",
        f"- FINE_GRAINED_SIGNAL_TREND={trends['FINE_GRAINED_SIGNAL_TREND']}: pooled B/C nonzero-credit rate changed from {evidence['fine_signal_early']:.6f} to {evidence['fine_signal_late']:.6f} (delta {evidence['fine_signal_delta']:+.6f}).",
        f"- B/C non-zero training credit clearly increased: {trends['B_C_NONZERO_TRAINING_CREDIT_CLEARLY_INCREASED']}.",
        f"- B delta={evidence['B_C_nonzero_credit']['B']['delta']:+.6f}; C delta={evidence['B_C_nonzero_credit']['C']['delta']:+.6f}.",
        f"- PROBE_PROGRESS={trends['PROBE_PROGRESS']}: compared Probe step0 with latest available step{latest}.",
        f"- Probe A/AB/Exact deltas: {changes['A_hit_rate']['delta']:+.6f}, {changes['AB_hit_rate']['delta']:+.6f}, {changes['exact_rate']['delta']:+.6f}.",
        f"- Probe GROUP_ANY A/AB/Exact deltas: {changes['GROUP_ANY_A']['delta']:+.0f}, {changes['GROUP_ANY_AB']['delta']:+.0f}, {changes['GROUP_ANY_EXACT']['delta']:+.0f} groups.",
        "",
        "Interpretation boundary",
        "These labels describe only the recorded first 2000 training groups and fixed Probe20 checkpoints. They do not change or prescribe the training strategy.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    group_rows = read_jsonl_prefix(run_dir / "train_groups.jsonl")
    explain_rows = read_jsonl_prefix(run_dir / "train_explain.jsonl")
    rows = validate_and_join(group_rows, explain_rows)
    buckets, cumulative = training_tables(rows)
    probes = probe_table(run_dir / "probe")
    trends = classify_trends(buckets, probes)
    provenance = {
        "run_id": RUN_ID, "run_dir": str(run_dir),
        "group_index_start": 0, "group_index_end": 1999, "groups": GROUP_LIMIT,
        "train_groups_selected_sha256": _canonical_sha(group_rows),
        "train_explain_selected_sha256": _canonical_sha(explain_rows),
        "source_files_modified": False, "gpu_inference_started": False,
        "training_started": False, "optimizer_steps": 0,
    }
    bucket_output = {"provenance": provenance, "buckets": buckets, "cumulative": cumulative}
    summary = {
        "provenance": provenance,
        "overall": summarize_training(rows),
        "mechanical_judgment": trends,
        "latest_probe_step_le_2000": probes["latest_step_le_2000"],
    }
    write_json(output_dir / "bucket_metrics.json", bucket_output)
    write_json(output_dir / "probe_progress.json", probes)
    write_json(output_dir / "summary.json", summary)
    write_review(output_dir / "REVIEW.txt", summary, probes)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.run_dir, args.output_dir)
    trends = summary["mechanical_judgment"]
    print(f"GROUPS={summary['provenance']['groups']}")
    print(f"HPR_A_TREND={trends['HPR_A_TREND']}")
    print(f"FINE_GRAINED_SIGNAL_TREND={trends['FINE_GRAINED_SIGNAL_TREND']}")
    print(f"PROBE_PROGRESS={trends['PROBE_PROGRESS']}")
    print(f"B_C_NONZERO_CREDIT_CLEARLY_INCREASED={trends['B_C_NONZERO_TRAINING_CREDIT_CLEARLY_INCREASED']}")


if __name__ == "__main__":
    main()
