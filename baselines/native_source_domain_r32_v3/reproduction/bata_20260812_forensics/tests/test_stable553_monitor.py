from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "stable553_monitor.py"
    spec = importlib.util.spec_from_file_location("stable553_monitor_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scalar_rows_are_deduplicated_and_sorted():
    module = load()
    log = "{'loss': 2.0, 'grad_norm': 1.0, 'learning_rate': 1e-4, 'epoch': 0.1, 'step': 10}\n"
    log += "{'loss': 1.0, 'grad_norm': 0.5, 'learning_rate': 9e-5, 'epoch': 0.2, 'step': 5}\n"
    log += "{'loss': 1.5, 'grad_norm': 0.8, 'learning_rate': 8e-5, 'epoch': 0.2, 'step': 10}\n"
    rows = module.scalar_rows(log)
    assert [row["step"] for row in rows] == [5, 10]
    assert rows[-1]["loss"] == 1.5


def test_scalar_rows_infer_trainer_logging_steps_and_ignore_info_text():
    module = load()
    log = "[INFO] Using FlashAttention for inference.\n"
    log += "{'loss': '2.5', 'grad_norm': '1.25', 'learning_rate': '2e-5', 'epoch': '0.01'}\n"
    log += "{'loss': '2.0', 'grad_norm': '1.0', 'learning_rate': '3e-5', 'epoch': '0.02'}\n"
    rows = module.scalar_rows(log)
    assert [row["step"] for row in rows] == [5, 10]
    assert rows[0]["loss"] == 2.5
    assert module.ERROR_RE.search("[INFO] inference") is None


def test_scalar_rows_can_continue_inferred_steps_after_553():
    module = load()
    log = "{'loss': 2.5, 'grad_norm': 1.25}\n{'loss': 2.0, 'grad_norm': 1.0}\n"
    rows = module.scalar_rows(log, start_step=553)
    assert [row["step"] for row in rows] == [558, 563]


def test_split_training_logs_preserves_independent_a_b_series():
    module = load()
    log = "setup\n***** Running training *****\n{'loss': 2.0, 'step': 5}\n"
    log += "finished A\n***** Running training *****\n{'loss': 1.0, 'step': 5}\n"
    sections = module.split_training_logs(log)
    assert module.scalar_rows(sections["STABLE553-A"])[0]["loss"] == 2.0
    assert module.scalar_rows(sections["STABLE553-B"])[0]["loss"] == 1.0


def test_run_status_requires_four_pass_evidence_and_complete_checkpoint(tmp_path):
    module = load()
    root = tmp_path
    evidence = root / "runs/STABLE553-A/evidence"
    checkpoint = root / "runs/STABLE553-A/output/checkpoint-553"
    evidence.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    for rank in range(4):
        (evidence / f"runtime_rank{rank}.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    for name in (
        "adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
        "trainer_state.json", "training_args.bin", "rng_state_0.pth", "rng_state_1.pth",
        "rng_state_2.pth", "rng_state_3.pth",
    ):
        (checkpoint / name).write_bytes(b"x")
    row = module.run_status(root, "STABLE553-A", "")
    assert row["status"] == "PASS"
    assert row["step"] == 553
    assert row["checkpoint_complete"] is True
    assert row["download_ready"] is True


def test_checkpoint_download_path_is_allowlisted(tmp_path):
    module = load()
    checkpoint = tmp_path / "runs/STABLE553-A/output/checkpoint-553"
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    assert module.checkpoint_download_path(
        tmp_path, "STABLE553-A", "adapter_model.safetensors"
    ) == adapter
    assert module.checkpoint_download_path(tmp_path, "STABLE553-C", adapter.name) is None
    assert module.checkpoint_download_path(tmp_path, "STABLE553-A", "optimizer.pt") is None
    assert module.checkpoint_download_path(tmp_path, "STABLE553-A", "../adapter_model.safetensors") is None


def test_comparison_status_waits_for_public_result(tmp_path):
    module = load()
    assert module.comparison_status(tmp_path) == {"status": "WAITING"}


def test_comparison_status_exposes_dashboard_subset(tmp_path):
    module = load()
    public = tmp_path / "public"
    public.mkdir()
    metric = {"cosine": 1.0, "relative_l2": 0.0}
    result = {
        "verdict": "STABLE553_EXACT",
        "first_epoch_repeatability": "CONFIRMED",
        "pairwise_repeatability": {
            "adapter_file_sha_exact": True,
            "adapter_canonical_sha_exact": True,
            "optimizer_exact": True,
            "scheduler_exact": True,
            "rng_exact": True,
            "raw_lora": metric,
            "effective_ba": metric,
        },
        "historical553_comparison": {
            "raw_lora": {"cosine": 0.88, "relative_l2": 0.48},
            "effective_ba": {"cosine": 0.66, "relative_l2": 0.83},
            "layer_summary": {"cosine": {"min": 0.2, "mean": 0.6, "max": 0.9}},
            "projections": {"q": {"cosine": 0.5, "relative_l2": 1.0}},
        },
        "external_evaluation": {"status": "NOT_EXECUTED"},
        "private_path": "/must/not/be/exposed",
    }
    (public / "stable553_result.json").write_text(json.dumps(result), encoding="utf-8")

    comparison = module.comparison_status(tmp_path)
    assert comparison["status"] == "READY"
    assert comparison["verdict"] == "STABLE553_EXACT"
    assert comparison["pairwise"]["adapter_file_sha_exact"] is True
    assert comparison["historical"]["effective_ba"]["cosine"] == 0.66
    assert "private_path" not in comparison
    assert module.snapshot(tmp_path)["comparison"] == comparison
    assert "Checkpoint-553 权重比较" in module.HTML
    assert "Effective B@A" in module.HTML


def test_manual_scores_are_atomic_and_run_allowlisted(tmp_path):
    module = load()
    run_id = "EPOCH2-S42-20260903-120000"
    (tmp_path / "epoch2_runs" / run_id).mkdir(parents=True)
    record = module.save_manual_score(tmp_path, run_id, 1.2345, "external run")
    assert record == {"score": 1.2345, "notes": "external run"}
    assert module.manual_scores(tmp_path)[run_id] == record
    with pytest.raises(ValueError):
        module.save_manual_score(tmp_path, "../bad", 1.0, "")


def test_epoch2_checkpoint_download_is_allowlisted(tmp_path):
    module = load()
    run_id = "EPOCH2-S42-20260903-120000"
    checkpoint = tmp_path / "epoch2_runs" / run_id / "output/checkpoint-1106"
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    assert module.epoch2_checkpoint_download_path(tmp_path, run_id, adapter.name) == adapter
    assert module.epoch2_checkpoint_download_path(tmp_path, "../bad", adapter.name) is None
    assert module.epoch2_checkpoint_download_path(tmp_path, run_id, "optimizer.pt") is None


def test_epoch2_controls_are_present_without_arbitrary_command_input():
    module = load()
    assert "启动四卡续训" in module.HTML
    assert "随机种子" in module.HTML
    assert "保存成绩" in module.HTML
    assert "X-Stable553-Action" in module.HTML
    assert "command" not in module.HTML.lower()
