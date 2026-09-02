from __future__ import annotations

import hashlib
import inspect
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


def test_frozen_batch_contract_avoids_gpu_value_hashes(monkeypatch, tmp_path):
    module = load_module()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "step": 554,
                "world_size": 4,
                "ranks": {
                    "0": {
                        "microbatch_count": 16,
                        "ordered_batch_fingerprint": "frozen-rank0",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("BATA_REPLAY_CHECKPOINT", str(tmp_path))
    monkeypatch.setenv("BATA_REPLAY_BATCH_FINGERPRINT_MODE", "frozen_step554_contract")
    monkeypatch.setenv("BATA_REPLAY_BATCH_CONTRACT_JSON", str(contract_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(
        module,
        "_tensor_fingerprint",
        lambda *args: (_ for _ in ()).throw(AssertionError("value hash must not run")),
    )
    controller = module.ReplayForensics()
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=553))
    inputs = {name: torch.zeros((1, 2), dtype=torch.long) for name in module._BATCH_FIELDS}

    for _ in range(16):
        controller.record_batch(trainer, inputs)
    controller.step_end(
        SimpleNamespace(global_step=554, epoch=1.0),
        torch.nn.Linear(1, 1),
        object(),
        SimpleNamespace(get_last_lr=lambda: [1e-4]),
    )

    rows = [
        json.loads(line)
        for line in controller.events_path.read_text(encoding="utf-8").splitlines()
    ]
    observations = [row for row in rows if row["event"] == "microbatch_contract_observation"]
    optimizer = next(row for row in rows if row["event"] == "optimizer_step")
    assert len(observations) == 16
    assert all("sha256" not in (field or {}) for row in observations for field in row["fields"].values())
    assert optimizer["ordered_batch_fingerprint"] == "frozen-rank0"
    assert optimizer["batch_contract"]["observed_microbatch_count"] == 16


def test_ddp_hook_wraps_official_default_and_records_pre_post(monkeypatch, tmp_path):
    module = load_module()
    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("BATA_REPLAY_CHECKPOINT", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "4")
    parameter = torch.nn.Parameter(torch.ones(2))
    calls = []

    class FakeDDP:
        process_group = "group"

        def named_parameters(self):
            return [("module.layer.lora_A.default.weight", parameter)]

        def register_comm_hook(self, state, hook):
            self.state = state
            self.hook = hook

    class FakeBucket:
        def __init__(self):
            self.value = torch.tensor([2.0, 4.0])

        def buffer(self):
            return self.value

        def parameters(self):
            return [parameter]

        def index(self):
            return 7

        def is_last(self):
            return True

    def official_default(state, bucket):
        calls.append((state, bucket.index()))
        bucket.buffer().div_(2)
        future = torch.futures.Future()
        future.set_result(bucket.buffer())
        return future

    monkeypatch.setattr(module, "DistributedDataParallel", FakeDDP)
    monkeypatch.setattr(module.default_hooks, "allreduce_hook", official_default)
    controller = module.ReplayForensics()
    model = FakeDDP()
    controller.install_ddp_comm_hook(model)
    signature = inspect.signature(model.hook)
    assert signature.parameters["bucket"].annotation is inspect.Signature.empty
    assert signature.return_annotation is inspect.Signature.empty
    result = model.hook(model.state, FakeBucket()).wait()

    assert calls == [("group", 7)]
    assert torch.equal(result, torch.tensor([1.0, 2.0]))
    rows = [
        json.loads(line)
        for line in controller.events_path.read_text(encoding="utf-8").splitlines()
    ]
    pre = next(row for row in rows if row["event"] == "ddp_pre_allreduce")
    post = next(row for row in rows if row["event"] == "ddp_post_allreduce")
    assert pre["layout_sha256"] == post["layout_sha256"]
    assert pre["fingerprint"] != post["fingerprint"]
    assert pre["parameter_count"] == 1


def test_clip_wrapper_records_pre_and_post_without_changing_original(monkeypatch):
    module = load_module()
    calls = []

    class FakeController:
        def record_clip(self, stage, model, returned_grad_norm=None):
            calls.append((stage, model.value, returned_grad_norm))

        def record_grad_norm(self, value):
            calls.append(("norm", value))

    class FakeTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return torch.tensor(0.0)

        def _load_rng_state(self, checkpoint):
            return None

        def _clip_grad_norm(self, model):
            model.value = "clipped"
            return torch.tensor(0.75)

    model = SimpleNamespace(value="unclipped")
    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", "enabled")
    monkeypatch.delenv("BATA_REPLAY_ALLOW_TRUSTED_TORCH_LOAD", raising=False)
    monkeypatch.setattr(module, "get_controller", lambda: FakeController())
    module.install_replay_instrumentation(FakeTrainer)

    result = FakeTrainer()._clip_grad_norm(model)

    assert torch.equal(result, torch.tensor(0.75))
    assert calls[0] == ("PRE_CLIP", "unclipped", None)
    assert calls[1][0:2] == ("POST_CLIP", "clipped")
    assert torch.equal(calls[1][2], torch.tensor(0.75))
    assert torch.equal(calls[2][1], torch.tensor(0.75))


def test_ddp_hook_is_installed_after_prepare_and_before_training(monkeypatch):
    module = load_module()
    calls = []
    wrapped_model = object()

    class FakeController:
        def install_ddp_comm_hook(self, model):
            calls.append(("hook", model))

    class FakeTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return torch.tensor(0.0)

        def _load_rng_state(self, checkpoint):
            return None

        def _clip_grad_norm(self, model):
            return torch.tensor(0.0)

        def _prepare_for_training(self, *args, **kwargs):
            calls.append(("prepare", args, kwargs))
            return wrapped_model, "dataloader"

    monkeypatch.setenv("BATA_REPLAY_EVIDENCE_DIR", "enabled")
    monkeypatch.delenv("BATA_REPLAY_ALLOW_TRUSTED_TORCH_LOAD", raising=False)
    monkeypatch.setattr(module, "get_controller", lambda: FakeController())
    module.install_replay_instrumentation(FakeTrainer)

    result = FakeTrainer()._prepare_for_training(1106, "input-loader", "/checkpoint-553")

    assert result == (wrapped_model, "dataloader")
    assert calls == [
        ("prepare", (1106, "input-loader", "/checkpoint-553"), {}),
        ("hook", wrapped_model),
    ]


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
