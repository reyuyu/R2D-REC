from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(f"stable_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launcher_enables_all_deterministic_controls_and_historical_horizon():
    launcher = (ROOT / "launch_stable_pilot.sh").read_text(encoding="utf-8")
    contract = json.loads((ROOT / "stable560_contract.json").read_text(encoding="utf-8"))
    assert 'export CUBLAS_WORKSPACE_CONFIG=":4096:8"' in launcher
    assert 'export FLASH_ATTENTION_DETERMINISTIC="1"' in launcher
    assert 'export BATA_STABLE_V0="1"' in launcher
    assert contract["deterministic_controls"]["torch_use_deterministic_algorithms"] is True
    assert contract["max_steps_scheduler_horizon"] == 1106
    assert contract["stop_after_optimizer_step"] == 560


def test_stop_callback_preserves_horizon_and_requests_checkpoint(monkeypatch, tmp_path):
    module = load("stable_pilot_runtime.py")
    callback = module.StableStopAndSaveCallback(560)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    state = SimpleNamespace(global_step=559, max_steps=1106)
    callback.on_step_end(None, state, control)
    assert not control.should_save
    assert not control.should_training_stop
    state.global_step = 560
    callback.on_step_end(None, state, control)
    assert control.should_save
    assert control.should_training_stop
    assert state.max_steps == 1106


def test_production_runtime_has_no_heavy_forensic_instrumentation():
    runtime = (ROOT / "stable_pilot_runtime.py").read_text(encoding="utf-8")
    integration = (ROOT / "stable_trainer_integration.patch").read_text(encoding="utf-8")
    assert "register_comm_hook" not in runtime
    assert "install_replay_instrumentation" not in runtime
    assert "ReplayForensics" not in runtime
    assert "replay_forensics" not in integration


def test_effective_ba_pair_matches_explicit_dense_math():
    module = load("compare_stable_pilots.py")
    left = {
        "x.lora_A.weight": torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        "x.lora_B.weight": torch.tensor([[1.0, 0.0], [2.0, 1.0]]),
    }
    right = {
        "x.lora_A.weight": torch.tensor([[0.5, 2.0], [1.0, -1.0]]),
        "x.lora_B.weight": torch.tensor([[1.0, 1.0], [0.0, 2.0]]),
    }
    actual = module.effective_pair(left, 2.0, right, 2.0)
    dense_left = 2.0 * left["x.lora_B.weight"].double() @ left["x.lora_A.weight"].double()
    dense_right = 2.0 * right["x.lora_B.weight"].double() @ right["x.lora_A.weight"].double()
    delta = dense_right - dense_left
    expected_relative = torch.linalg.vector_norm(delta) / max(
        torch.linalg.vector_norm(dense_left), torch.linalg.vector_norm(dense_right)
    )
    expected_cosine = torch.sum(dense_left * dense_right) / (
        torch.linalg.vector_norm(dense_left) * torch.linalg.vector_norm(dense_right)
    )
    assert actual["relative_l2"] == pytest.approx(expected_relative.item(), abs=1e-12)
    assert actual["cosine"] == pytest.approx(expected_cosine.item(), abs=1e-12)


def test_historical_direction_metrics_match_explicit_dense_math():
    module = load("compare_stable_pilots.py")

    def adapter(a, b):
        return {
            "x.lora_A.weight": torch.tensor(a, dtype=torch.float64),
            "x.lora_B.weight": torch.tensor(b, dtype=torch.float64),
        }

    h553 = adapter([[1.0, 0.0]], [[1.0], [0.0]])
    h1106 = adapter([[2.0, 1.0]], [[1.0], [1.0]])
    stable = adapter([[1.2, 0.1]], [[1.0], [0.2]])
    actual = module.direction_metrics((stable, 1.0), (h553, 1.0), (h1106, 1.0))
    dense = lambda value: value["x.lora_B.weight"] @ value["x.lora_A.weight"]
    current = dense(stable) - dense(h553)
    historical = dense(h1106) - dense(h553)
    dot = torch.sum(current * historical)
    historical_norm_sq = torch.sum(historical * historical)
    progress = dot / historical_norm_sq
    residual = torch.linalg.vector_norm(current - progress * historical) / torch.linalg.vector_norm(historical)
    cosine = dot / (torch.linalg.vector_norm(current) * torch.linalg.vector_norm(historical))
    assert actual["hist_update_cosine"] == pytest.approx(cosine.item(), abs=1e-12)
    assert actual["hist_progress"] == pytest.approx(progress.item(), abs=1e-12)
    assert actual["hist_residual"] == pytest.approx(residual.item(), abs=1e-12)


def test_source_contract_mismatch_fails_closed(tmp_path):
    module = load("prepare_stable_pilot.py")
    source = tmp_path / "source"
    for name in module.PRISTINE_SOURCE_SHA256:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("wrong", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source contract mismatch"):
        module.verify_pristine_source(source)


def test_canonical_state_hash_is_order_independent_and_value_sensitive():
    module = load("compare_stable_pilots.py")
    left = {"b": torch.tensor([2]), "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": torch.tensor([2])}
    changed = {"a": {"x": 1}, "b": torch.tensor([3])}
    assert module.canonical_state_hash(left) == module.canonical_state_hash(right)
    assert module.canonical_state_hash(left) != module.canonical_state_hash(changed)
