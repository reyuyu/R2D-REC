from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from repro_quality import (  # noqa: E402
    build_snapshot,
    cached_sha256_file,
    inspect_checkpoint,
    metric_gap,
    read_jsonl,
    window_summary,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_jsonl_reader_ignores_partial_append(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":1,"loss":1.0}\n{"step":', encoding="utf-8")
    assert read_jsonl(path) == [{"step": 1, "loss": 1.0}]


def test_window_summary_uses_rows_at_or_before_milestone() -> None:
    rows = [{"step": step, "loss": float(step)} for step in range(1, 7)]
    summary = window_summary(rows, step_field="step", milestone=5, window_rows=3, metric_names=["loss"])
    assert summary["loss"] == {"mean": 4.0, "min": 3.0, "max": 5.0, "count": 3}


def test_metric_gap_separates_reference_band_from_contract() -> None:
    reference = {"mean": 1.0, "min": 0.9, "max": 1.1}
    assert metric_gap("reward_mean", {"mean": 1.05}, reference)["status"] == "within_reference_band"
    assert metric_gap("reward_mean", {"mean": 2.0}, reference)["status"] == "review"


def test_adapter_hash_is_cached_until_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(b"first")
    first = cached_sha256_file(path)
    assert cached_sha256_file(path) == first
    path.write_bytes(b"second payload")
    assert cached_sha256_file(path) != first


def test_trainer_checkpoint_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    required = ["adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"]
    required += [f"rng_state_{index}.pth" for index in range(4)]
    for name in required:
        (checkpoint / name).write_bytes(b"x")
    assert inspect_checkpoint(checkpoint, "trainer", 10)["status"] == "complete"
    (checkpoint / "model.safetensors").write_bytes(b"base")
    result = inspect_checkpoint(checkpoint, "trainer", 10)
    assert result["status"] == "invalid"
    assert result["adapter_only"] is False


def test_snapshot_marks_external_evaluation_pending(tmp_path: Path) -> None:
    reference = {
        "reference_name": "fixture",
        "comparison_policy": {},
        "datasets": {},
        "stages": [{
            "id": "s1", "label": "Stage 1", "short_label": "S1", "objective": "fixture",
            "target_step": 2, "step_field": "step", "window_rows": 2,
            "expected_checkpoints": [2], "checkpoint_kind": "trainer",
            "selected_checkpoint_relative": "outputs/checkpoint-2",
            "historical_adapter_sha256": "0" * 64, "historical_checkpoint_path": "/historical/checkpoint-2",
            "metrics_candidates": ["metrics.jsonl"],
            "external_score": {"primary": 1.0, "recorded": [1.0]},
            "metric_definitions": {"loss": "fixture loss"},
            "milestones": {"2": {"loss": {"mean": 1.0, "min": 0.5, "max": 1.5}}}
        }],
    }
    reference_path = tmp_path / "reference.json"
    write_json(reference_path, reference)
    (tmp_path / "metrics.jsonl").write_text('{"step":1,"loss":1.1}\n{"step":2,"loss":0.9}\n', encoding="utf-8")
    snapshot = build_snapshot(tmp_path, reference_path)
    stage = snapshot["stages"][0]
    assert stage["runtime_status"] == "running"
    assert stage["trajectory_status"] == "within_reference_band"
    assert stage["external_evaluation"]["status"] == "pending"
    assert snapshot["overall_status"] == "ready"


def test_incomplete_checkpoint_is_not_hashed(tmp_path: Path) -> None:
    reference = {
        "reference_name": "fixture", "comparison_policy": {}, "datasets": {},
        "stages": [{
            "id": "s1", "label": "Stage 1", "short_label": "S1", "objective": "fixture",
            "target_step": 2, "step_field": "step", "window_rows": 1,
            "expected_checkpoints": [2], "checkpoint_kind": "trainer",
            "selected_checkpoint_relative": "outputs/checkpoint-2",
            "historical_adapter_sha256": "0" * 64, "historical_checkpoint_path": "/historical/checkpoint-2",
            "metrics_candidates": [], "external_score": {"primary": 1.0, "recorded": [1.0]},
            "metric_definitions": {"loss": "fixture"},
            "milestones": {"2": {"loss": {"mean": 1.0, "min": 0.5, "max": 1.5}}}
        }],
    }
    reference_path = tmp_path / "reference.json"
    write_json(reference_path, reference)
    checkpoint = tmp_path / "outputs" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"still-being-written")
    snapshot = build_snapshot(tmp_path, reference_path)
    assert snapshot["stages"][0]["adapter"]["reproduced_sha256"] is None


def test_reference_has_exact_four_stage_lineage() -> None:
    reference = json.loads((MODULE_ROOT / "historical_reference.json").read_text(encoding="utf-8"))
    assert [stage["id"] for stage in reference["stages"]] == ["01_sft_beta", "02_gr_rec_v1", "03_grpo_tk", "04_mc_user"]
    assert [stage["target_step"] for stage in reference["stages"]] == [1106, 1500, 250, 100]
