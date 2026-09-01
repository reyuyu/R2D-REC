#!/usr/bin/env python3
"""Read-only four-stage reproduction quality comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_ADAPTER_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin",
}
METRIC_FLOORS = {
    "loss": 0.001,
    "grad_norm": 0.1,
    "reward_mean": 0.1,
    "reward_std": 0.1,
    "zero_std_ratio": 0.05,
    "completion_mean_length": 10.0,
    "clip_fraction": 0.002,
    "approx_kl": 0.0002,
    "sequence_loss": 0.01,
    "local_loss": 0.01,
    "group_reward_mean": 0.05,
    "group_reward_std": 0.03,
}
_SHA_CACHE: dict[str, tuple[int, int, str]] = {}
METRIC_ALIASES = {
    "reward_mean": "reward",
    "zero_std_ratio": "frac_reward_zero_std",
    "completion_mean_length": "completions/mean_length",
    "clip_fraction": "completions/clipped_ratio",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A writer may be in the middle of appending the final line.
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256_file(path: Path) -> str:
    """Hash once per immutable file stat so dashboard polling stays low-I/O."""
    stat = path.stat()
    key = str(path.resolve())
    cached = _SHA_CACHE.get(key)
    signature = (stat.st_size, stat.st_mtime_ns)
    if cached is not None and cached[:2] == signature:
        return cached[2]
    digest = sha256_file(path)
    _SHA_CACHE[key] = (signature[0], signature[1], digest)
    return digest


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def row_step(row: dict[str, Any], step_field: str) -> int:
    for key in (step_field, "step", "current_steps", "prompt_step", "global_step"):
        value = row.get(key)
        if finite_number(value):
            return int(value)
    return -1


def locate_first(root: Path, candidates: Iterable[str]) -> Path | None:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    return None


def load_metric_rows(root: Path, stage: dict[str, Any]) -> tuple[Path | None, list[dict[str, Any]]]:
    path = locate_first(root, stage.get("metrics_candidates", []))
    if path is None:
        return None, []
    return path, load_metric_rows_from_path(path)


def load_metric_rows_from_path(path: Path) -> list[dict[str, Any]]:
    if path.name == "trainer_state.json":
        state = load_json(path)
        rows = state.get("log_history", []) if isinstance(state, dict) else []
        raw_rows = [row for row in rows if isinstance(row, dict)]
    else:
        raw_rows = read_jsonl(path)
    normalized = []
    for row in raw_rows:
        value = dict(row)
        for canonical, source in METRIC_ALIASES.items():
            if canonical not in value and finite_number(value.get(source)):
                value[canonical] = value[source]
        normalized.append(value)
    return normalized


def window_summary(
    rows: list[dict[str, Any]],
    *,
    step_field: str,
    milestone: int,
    window_rows: int,
    metric_names: Iterable[str],
) -> dict[str, dict[str, float | int]]:
    eligible = sorted(
        (row for row in rows if 0 <= row_step(row, step_field) <= milestone),
        key=lambda row: row_step(row, step_field),
    )[-window_rows:]
    result: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        values = [float(row[name]) for row in eligible if finite_number(row.get(name))]
        if values:
            result[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
    return result


def metric_gap(name: str, reproduced: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    value = float(reproduced["mean"])
    expected = float(reference["mean"])
    reference_min = float(reference["min"])
    reference_max = float(reference["max"])
    span = max(reference_max - reference_min, 0.0)
    padding = max(span * 0.25, abs(expected) * 0.10, METRIC_FLOORS.get(name, 0.01))
    lower = reference_min - padding
    upper = reference_max + padding
    absolute = value - expected
    denominator = max(abs(expected), METRIC_FLOORS.get(name, 0.01))
    return {
        "metric": name,
        "reference_mean": expected,
        "reproduced_mean": value,
        "absolute_delta": absolute,
        "relative_delta": absolute / denominator,
        "reference_band": [lower, upper],
        "status": "within_reference_band" if lower <= value <= upper else "review",
    }


def checkpoint_paths(root: Path, stage: dict[str, Any]) -> list[tuple[int, Path]]:
    selected = root / stage["selected_checkpoint_relative"]
    parent = selected.parent
    kind = stage["checkpoint_kind"]
    paths = []
    for step in stage["expected_checkpoints"]:
        name = f"checkpoint-{step}" if kind == "trainer" else f"prompt-step-{step:04d}"
        paths.append((int(step), parent / name))
    return paths


def inspect_checkpoint(path: Path, kind: str, expected_step: int) -> dict[str, Any]:
    if not path.is_dir():
        return {"step": expected_step, "path": str(path), "status": "missing"}
    names = {child.name for child in path.iterdir() if child.is_file()}
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if kind == "trainer":
        required.update({"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"})
        required.update({f"rng_state_{index}.pth" for index in range(4)})
    else:
        required.add("formal_state.json")
    missing = sorted(required - names)
    forbidden = sorted(name for name in names if name in FORBIDDEN_ADAPTER_FILES or name.startswith("model-") and name.endswith(".safetensors"))
    return {
        "step": expected_step,
        "path": str(path),
        "status": "complete" if not missing and not forbidden else "invalid",
        "missing": missing,
        "forbidden": forbidden,
        "adapter_only": not forbidden,
    }


def load_external_evaluation(root: Path, stage_id: str) -> tuple[Path | None, float | None]:
    candidates = [
        root / "evaluations" / f"{stage_id}.json",
        root / "results" / f"{stage_id}_external_eval.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_json(path)
        for key in ("aggregate", "score", "total_score"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if finite_number(value):
                return path, float(value)
    return None, None


def sampled_curve(rows: list[dict[str, Any]], stage: dict[str, Any], limit: int = 600) -> list[dict[str, Any]]:
    metrics = list(stage.get("metric_definitions", {}))
    target_step = int(stage.get("target_step", 2**63 - 1))
    by_step: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        step = row_step(row, stage["step_field"])
        if step < 0 or step > target_step:
            continue
        metric_values = by_step.setdefault(step, {})
        for name in metrics:
            if finite_number(row.get(name)):
                metric_values.setdefault(name, []).append(float(row[name]))
    points = [
        {
            "step": step,
            **{name: sum(values) / len(values) for name, values in metric_values.items() if values},
        }
        for step, metric_values in sorted(by_step.items())
        if metric_values
    ]
    if len(points) <= limit:
        return points
    stride = max(1, math.ceil(len(points) / limit))
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def load_adapter_comparisons(root: Path, stage: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_root = root / "evidence" / "adapter_comparisons"
    results = []
    for step in stage["expected_checkpoints"]:
        path = evidence_root / f"{stage['id']}-{int(step)}.json"
        if not path.is_file():
            results.append({"step": int(step), "status": "pending", "path": str(path)})
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            results.append({"step": int(step), "status": "invalid", "path": str(path), "error": str(error)})
            continue
        result = dict(payload) if isinstance(payload, dict) else {"status": "invalid"}
        result["step"] = int(step)
        result["path"] = str(path)
        results.append(result)
    return results


def compare_stage(root: Path, stage: dict[str, Any], historical_curve: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics_path, rows = load_metric_rows(root, stage)
    current_step = max((row_step(row, stage["step_field"]) for row in rows), default=0)
    pass_marker = root / "state" / f"{stage['id']}.PASS.json"
    stopped_marker = root / "state" / f"{stage['id']}.STOPPED.json"
    checkpoints = [inspect_checkpoint(path, stage["checkpoint_kind"], step) for step, path in checkpoint_paths(root, stage)]
    selected = root / stage["selected_checkpoint_relative"]

    if stopped_marker.is_file():
        runtime_status = "stopped"
    elif pass_marker.is_file():
        runtime_status = "pass"
    elif current_step > 0:
        runtime_status = "running"
    elif any(item["status"] != "missing" for item in checkpoints):
        runtime_status = "incomplete"
    else:
        runtime_status = "not_started"

    milestones = []
    review_count = 0
    for step_text, reference_metrics in stage["milestones"].items():
        step = int(step_text)
        actual = window_summary(
            rows,
            step_field=stage["step_field"],
            milestone=step,
            window_rows=int(stage["window_rows"]),
            metric_names=reference_metrics,
        ) if current_step >= step else {}
        gaps = []
        for metric_name, reference_metric in reference_metrics.items():
            if metric_name in actual:
                gap = metric_gap(metric_name, actual[metric_name], reference_metric)
                gaps.append(gap)
                review_count += int(gap["status"] == "review")
        milestones.append({
            "step": step,
            "available": current_step >= step and bool(gaps),
            "window_rows": stage["window_rows"],
            "gaps": gaps,
        })

    target_checkpoint = next((item for item in checkpoints if item["step"] == stage["target_step"]), None)
    adapter_sha = (
        cached_sha256_file(selected / "adapter_model.safetensors")
        if target_checkpoint and target_checkpoint["status"] == "complete"
        else None
    )
    if adapter_sha is None:
        adapter_status = "pending"
    elif adapter_sha == stage["historical_adapter_sha256"]:
        adapter_status = "bitwise_match"
    else:
        adapter_status = "different_expected_stochastic"

    eval_path, eval_score = load_external_evaluation(root, stage["id"])
    expected_score = float(stage["external_score"]["primary"])
    external = {
        "status": "pending" if eval_score is None else "recorded",
        "path": str(eval_path) if eval_path else None,
        "reference_score": expected_score,
        "reproduced_score": eval_score,
        "delta": None if eval_score is None else eval_score - expected_score,
        "reference": stage["external_score"],
    }

    complete_invalid = [item for item in checkpoints if item["status"] == "invalid"]
    contract_status = "pending"
    if complete_invalid:
        contract_status = "fail"
    elif runtime_status == "pass" and target_checkpoint and target_checkpoint["status"] == "complete":
        contract_status = "pass"
    elif runtime_status == "stopped":
        contract_status = "fail"

    if not any(item["available"] for item in milestones):
        trajectory_status = "pending"
    elif review_count:
        trajectory_status = "review"
    else:
        trajectory_status = "within_reference_band"

    latest = max(rows, key=lambda row: row_step(row, stage["step_field"])) if rows else {}
    latest_metrics = {
        name: float(latest[name])
        for name in stage.get("metric_definitions", {})
        if finite_number(latest.get(name))
    }
    return {
        "id": stage["id"],
        "label": stage["label"],
        "short_label": stage["short_label"],
        "objective": stage["objective"],
        "runtime_status": runtime_status,
        "contract_status": contract_status,
        "trajectory_status": trajectory_status,
        "current_step": current_step,
        "target_step": stage["target_step"],
        "progress": min(current_step / max(int(stage["target_step"]), 1), 1.0),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "latest_metrics": latest_metrics,
        "metric_definitions": stage.get("metric_definitions", {}),
        "curve": sampled_curve(rows, stage),
        "historical_curve": historical_curve or [],
        "milestones": milestones,
        "checkpoints": checkpoints,
        "adapter_comparisons": load_adapter_comparisons(root, stage),
        "selected_checkpoint": str(selected),
        "adapter": {
            "status": adapter_status,
            "reference_sha256": stage["historical_adapter_sha256"],
            "reproduced_sha256": adapter_sha,
            "historical_path": stage["historical_checkpoint_path"],
        },
        "external_evaluation": external,
    }


def build_snapshot(root: Path, reference_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    reference_path = reference_path or Path(__file__).with_name("historical_reference.json")
    reference = load_json(reference_path)
    historical_curves_path = reference_path.with_name(reference.get("historical_curves_file", "historical_curves.json"))
    historical_payload = load_json(historical_curves_path) if historical_curves_path.is_file() else {}
    historical_stages = historical_payload.get("stages", {}) if isinstance(historical_payload, dict) else {}
    stages = [
        compare_stage(root, stage, historical_stages.get(stage["id"], {}).get("curve", []))
        for stage in reference["stages"]
    ]
    contract_failures = sum(stage["contract_status"] == "fail" for stage in stages)
    contract_passes = sum(stage["contract_status"] == "pass" for stage in stages)
    trajectory_reviews = sum(stage["trajectory_status"] == "review" for stage in stages)
    completed = sum(stage["runtime_status"] == "pass" for stage in stages)
    current = next((stage["id"] for stage in stages if stage["runtime_status"] in {"running", "incomplete", "stopped"}), None)
    if current is None and completed < len(stages):
        current = stages[completed]["id"]
    overall = "contract_failure" if contract_failures else "review" if trajectory_reviews else "healthy" if completed else "ready"
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root": str(root),
        "reference_name": reference["reference_name"],
        "overall_status": overall,
        "summary": {
            "stage_count": len(stages),
            "completed_stage_count": completed,
            "contract_pass_count": contract_passes,
            "contract_fail_count": contract_failures,
            "trajectory_review_count": trajectory_reviews,
            "external_evaluation_count": sum(stage["external_evaluation"]["status"] == "recorded" for stage in stages),
            "current_stage": current,
        },
        "comparison_policy": reference["comparison_policy"],
        "datasets": reference["datasets"],
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = build_snapshot(args.root, args.reference)
    rendered = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
