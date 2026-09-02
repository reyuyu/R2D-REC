from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "replay_forensics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bata_replay_forensics_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rng_fingerprint_does_not_advance_rng(monkeypatch, tmp_path):
    module = load_module()
    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("BATA_REPLAY_CHECKPOINT", str(tmp_path))
    controller = module.ReplayForensics()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    hashes = controller.rng_hashes()

    assert set(hashes) >= {"python", "numpy", "torch_cpu"}
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_deterministic_setup_is_strict_and_pre_cuda(monkeypatch):
    module = load_module()
    calls = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(module.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        module.torch,
        "use_deterministic_algorithms",
        lambda enabled, warn_only=False: calls.append((enabled, warn_only)),
    )
    monkeypatch.setattr(module.torch, "are_deterministic_algorithms_enabled", lambda: True)

    result = module.configure_deterministic_diagnostic(enabled=True)

    assert calls == [(True, False)]
    assert result == {
        "enabled": True,
        "warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "configured_before_cuda": True,
    }


def test_deterministic_setup_fails_if_cuda_already_initialized(monkeypatch):
    module = load_module()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(module.torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="after CUDA initialization"):
        module.configure_deterministic_diagnostic(enabled=True)


def test_deterministic_setup_requires_cublas_contract(monkeypatch):
    module = load_module()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        module.configure_deterministic_diagnostic(enabled=True)


def test_compute_loss_capture_occurs_after_forward_and_before_caller_backward(monkeypatch):
    module = load_module()
    order = []

    class FakeController:
        def record_batch(self, trainer, inputs):
            order.append("batch_fingerprint")

        def record_loss(self, trainer, result):
            order.append("loss_capture")

        def record_grad_norm(self, result):
            pass

        def restored(self, trainer, checkpoint):
            pass

    class FakeTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            order.append("forward_and_local_loss")
            return torch.tensor(1.0)

        def _load_rng_state(self, checkpoint):
            return None

        def _clip_grad_norm(self, model):
            return torch.tensor(0.0)

    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", "enabled")
    monkeypatch.delenv("BATA_REPLAY_ALLOW_TRUSTED_TORCH_LOAD", raising=False)
    monkeypatch.setattr(module, "get_controller", lambda: FakeController())
    module.install_replay_instrumentation(FakeTrainer)

    trainer = FakeTrainer()
    trainer.compute_loss(None, {})
    order.append("caller_backward")

    assert order == ["batch_fingerprint", "forward_and_local_loss", "loss_capture", "caller_backward"]


def test_batch_fingerprint_covers_all_training_fields(monkeypatch, tmp_path):
    module = load_module()
    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("BATA_REPLAY_CHECKPOINT", str(tmp_path))
    controller = module.ReplayForensics()
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=553))
    inputs = {
        name: torch.tensor([[index]], dtype=torch.long)
        for index, name in enumerate(module._BATCH_FIELDS)
    }
    controller.record_batch(trainer, inputs)
    record = json.loads(controller.events_path.read_text(encoding="utf-8"))

    assert record["target_step"] == 554
    assert set(record["fields"]) == set(module._BATCH_FIELDS)
    assert all(record["fields"][name]["sha256"] for name in module._BATCH_FIELDS)


def test_stop_callback_does_not_change_scheduler_horizon():
    module = load_module()
    callback = module.StopAfterOptimizerStepCallback(554)
    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=553, max_steps=1106)
    callback.on_step_end(None, state, control)
    assert not control.should_training_stop
    assert state.max_steps == 1106
    state.global_step = 554
    callback.on_step_end(None, state, control)
    assert control.should_training_stop
    assert state.max_steps == 1106


def test_heavy_step_parser_is_explicit():
    module = load_module()
    assert module._parse_step_set("554") == {554}
    assert module._parse_step_set("555,560") == {555, 560}
    with pytest.raises(ValueError):
        module._parse_step_set("-1")


def test_checkpoint_compatibility_bypass_fails_closed_on_sha_mismatch(monkeypatch, tmp_path):
    module = load_module()
    checkpoint = tmp_path / "checkpoint-553"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"unexpected")
    expected = tmp_path / "expected.json"
    expected.write_text(
        json.dumps({"optimizer.pt": hashlib.sha256(b"expected").hexdigest()}),
        encoding="utf-8",
    )

    class FakeTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return torch.tensor(0.0)

        def _load_rng_state(self, checkpoint):
            return None

        def _clip_grad_norm(self, model):
            return torch.tensor(0.0)

    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("BATA_REPLAY_ALLOW_TRUSTED_TORCH_LOAD", "1")
    monkeypatch.setenv("BATA_REPLAY_EXPECTED_SHA_JSON", str(expected))
    monkeypatch.setenv("BATA_REPLAY_CHECKPOINT", str(checkpoint))

    with pytest.raises(RuntimeError, match="checkpoint SHA mismatch"):
        module.install_replay_instrumentation(FakeTrainer)


def test_compatibility_bypass_never_disables_weights_only_loading():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "check_torch_load_is_safe = lambda: None" in source
    assert "weights_only=False" not in source
