from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
