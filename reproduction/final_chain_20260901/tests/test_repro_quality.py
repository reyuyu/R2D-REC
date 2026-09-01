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
    load_metric_rows_from_path,
    metric_gap,
    read_jsonl,
    sampled_curve,
    window_summary,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_jsonl_reader_ignores_partial_append(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":1,"loss":1.0}\n{"step":', encoding="utf-8")
    assert read_jsonl(path) == [{"step": 1, "loss": 1.0}]


def test_trainer_metric_aliases_are_normalized(tmp_path: Path) -> None:
    path = tmp_path / "trainer_state.json"
    write_json(path, {"log_history": [{
        "step": 1, "reward": 0.5, "frac_reward_zero_std": 0.25,
        "completions/mean_length": 64.0, "completions/clipped_ratio": 0.125,
    }]})
    assert load_metric_rows_from_path(path)[0] | {"step": 1} == {
        "step": 1,
        "reward": 0.5,
        "frac_reward_zero_std": 0.25,
        "completions/mean_length": 64.0,
        "completions/clipped_ratio": 0.125,
        "reward_mean": 0.5,
        "zero_std_ratio": 0.25,
        "completion_mean_length": 64.0,
        "clip_fraction": 0.125,
    }


def test_window_summary_uses_rows_at_or_before_milestone() -> None:
    rows = [{"step": step, "loss": float(step)} for step in range(1, 7)]
    summary = window_summary(rows, step_field="step", milestone=5, window_rows=3, metric_names=["loss"])
    assert summary["loss"] == {"mean": 4.0, "min": 3.0, "max": 5.0, "count": 3}


def test_llamafactory_current_steps_drives_live_progress(tmp_path: Path) -> None:
    reference = {
        "reference_name": "fixture", "comparison_policy": {}, "datasets": {},
        "stages": [{
            "id": "s1", "label": "Stage 1", "short_label": "S1", "objective": "fixture",
            "target_step": 1106, "step_field": "step", "window_rows": 1,
            "expected_checkpoints": [553, 1106], "checkpoint_kind": "trainer",
            "selected_checkpoint_relative": "outputs/checkpoint-1106",
            "historical_adapter_sha256": "0" * 64, "historical_checkpoint_path": "/historical/checkpoint-1106",
            "metrics_candidates": ["trainer_log.jsonl"],
            "external_score": {"primary": 1.0, "recorded": [1.0]},
            "metric_definitions": {"loss": "fixture"},
            "milestones": {"553": {"loss": {"mean": 1.0, "min": 0.5, "max": 1.5}}},
        }],
    }
    reference_path = tmp_path / "reference.json"
    write_json(reference_path, reference)
    (tmp_path / "trainer_log.jsonl").write_text(
        '{"current_steps":620,"total_steps":1106,"loss":1.1}\n', encoding="utf-8"
    )
    snapshot = build_snapshot(tmp_path, reference_path)
    stage = snapshot["stages"][0]
    assert stage["current_step"] == 620
    assert stage["runtime_status"] == "running"
    assert stage["progress"] == pytest.approx(620 / 1106)
    assert stage["curve"] == [{"step": 620, "loss": 1.1}]


def test_sampled_curve_aggregates_duplicate_rank_rows_by_step() -> None:
    stage = {"step_field": "step", "target_step": 2, "metric_definitions": {"loss": "fixture", "grad_norm": "fixture"}}
    rows = [
        {"step": 1, "loss": 1.0, "grad_norm": 2.0},
        {"step": 1, "loss": 3.0, "grad_norm": 4.0},
        {"step": 2, "loss": 5.0},
        {"step": 3, "loss": 999.0},
    ]
    assert sampled_curve(rows, stage) == [
        {"step": 1, "loss": 2.0, "grad_norm": 3.0},
        {"step": 2, "loss": 5.0},
    ]


def test_snapshot_exposes_historical_curve_and_adapter_comparison(tmp_path: Path) -> None:
    reference = {
        "reference_name": "fixture", "comparison_policy": {}, "datasets": {},
        "historical_curves_file": "historical_curves.json",
        "stages": [{
            "id": "s1", "label": "Stage 1", "short_label": "S1", "objective": "fixture",
            "target_step": 2, "step_field": "step", "window_rows": 1,
            "expected_checkpoints": [2], "checkpoint_kind": "trainer",
            "selected_checkpoint_relative": "outputs/checkpoint-2",
            "historical_adapter_sha256": "0" * 64, "historical_checkpoint_path": "/historical/checkpoint-2",
            "metrics_candidates": [], "external_score": {"primary": 1.0, "recorded": [1.0]},
            "metric_definitions": {"loss": "fixture"},
            "milestones": {"2": {"loss": {"mean": 1.0, "min": 0.5, "max": 1.5}}},
        }],
    }
    reference_path = tmp_path / "historical_reference.json"
    write_json(reference_path, reference)
    write_json(tmp_path / "historical_curves.json", {"schema_version": 1, "stages": {"s1": {"curve": [{"step": 1, "loss": 1.25}]}}})
    write_json(tmp_path / "evidence" / "adapter_comparisons" / "s1-2.json", {
        "status": "numerically_compared", "cosine_similarity": 0.99, "relative_l2": 0.1, "max_abs_delta": 0.01,
    })
    stage = build_snapshot(tmp_path, reference_path)["stages"][0]
    assert stage["historical_curve"] == [{"step": 1, "loss": 1.25}]
    assert stage["adapter_comparisons"][0]["cosine_similarity"] == 0.99
    assert stage["adapter_comparisons"][0]["step"] == 2


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
