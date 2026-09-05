import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from checkpointing import save_lineage, validate_adapter_lineage
from modeling import (
    assert_optimizer_lora_only,
    enforce_lora_only_trainable,
    load_probe_model,
    sample_positions,
)
from parent_contract import ParentSpec, file_sha256, validate_parent_files
from run_conservative_grpo import _assert_formal_startup_audit, _validate_formal_contract


def _base(tmp_path: Path) -> tuple[Path, str, str]:
    base = tmp_path / "base"
    base.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (base / name).write_text("{}\n", encoding="utf-8")
    (base / "config.json").write_text('{"model_type":"qwen3"}\n', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"full-model")
    return base, file_sha256(base / "model.safetensors"), file_sha256(base / "config.json")


def _checkpoint(tmp_path: Path, step: int = 2) -> Path:
    checkpoint = tmp_path / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    return checkpoint


# 1. full_model ParentSpec parse.
def test_full_model_parent_spec_parse(tmp_path):
    base, model_sha, config_sha = _base(tmp_path)
    spec = ParentSpec.from_mapping({
        "parent_mode": "full_model", "base_model_path": str(base),
        "base_model_sha256": model_sha, "config_sha256": config_sha,
    })
    assert spec.parent_mode == "full_model" and spec.parent_adapter_path is None
    with pytest.raises(ValueError, match="64-character SHA256"):
        ParentSpec("full_model", str(base), model_sha + "0")


# 2. adapter ParentSpec regression.
def test_adapter_parent_spec_parse(tmp_path):
    base, model_sha, _ = _base(tmp_path)
    spec = ParentSpec.from_mapping({
        "parent_mode": "adapter", "base_model_path": str(base),
        "base_model_sha256": model_sha, "parent_adapter_path": "/adapter",
        "parent_adapter_sha256": "a" * 64,
    })
    assert spec.parent_mode == "adapter" and spec.parent_adapter_sha256 == "a" * 64


# 3. full_model does not require a parent adapter.
def test_full_model_requires_no_adapter(tmp_path):
    base, model_sha, _ = _base(tmp_path)
    ParentSpec("full_model", str(base), model_sha)


# 4. adapter mode still requires a parent adapter.
def test_adapter_mode_requires_adapter(tmp_path):
    base, model_sha, _ = _base(tmp_path)
    with pytest.raises(ValueError, match="requires parent adapter"):
        ParentSpec("adapter", str(base), model_sha)


# 5. wrong base SHA fails closed.
def test_wrong_base_sha_fails(tmp_path):
    base, _, config_sha = _base(tmp_path)
    spec = ParentSpec("full_model", str(base), "0" * 64, config_sha)
    with pytest.raises(RuntimeError, match="base model SHA mismatch"):
        validate_parent_files(spec)


# 6. adapter lineage from another base fails even when shapes could match.
def test_old_adapter_new_full_base_mismatch_fails(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    save_lineage(
        checkpoint, parent_mode="full_model", parent_base_sha256="0" * 64,
        config_sha256="cfg", dataset_sha256="data", code_commit="commit",
        seed=1, lora_seed=2,
    )
    with pytest.raises(RuntimeError, match="parent base SHA mismatch"):
        validate_adapter_lineage(checkpoint, expected_base_sha256="1" * 64)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Parameter(torch.ones(2, 1))
        self.lora_B = torch.nn.Parameter(torch.zeros(1, 2))


# 7. fresh LoRA freezes base and exposes only LoRA tensors.
def test_fresh_lora_trainable_contract():
    model = _TinyModel()
    audit = enforce_lora_only_trainable(model)
    assert audit["base_trainable_parameter_count"] == 0
    assert audit["lora_trainable_parameter_count"] == 4
    assert all(("lora_" in name) == parameter.requires_grad for name, parameter in model.named_parameters())
    positions = sample_positions(152064 * 4096, 256)
    assert int(positions[0]) == 0 and int(positions[-1]) == 152064 * 4096 - 1


# 8. optimizer IDs exactly equal LoRA trainable IDs.
def test_optimizer_parameter_identity():
    model = _TinyModel()
    enforce_lora_only_trainable(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-6)
    assert assert_optimizer_lora_only(model, optimizer)["optimizer_lora_only"] is True
    optimizer.add_param_group({"params": [model.base.weight]})
    with pytest.raises(RuntimeError, match="do not exactly match"):
        assert_optimizer_lora_only(model, optimizer)


# 9. adapter save/load lineage roundtrip.
def test_lineage_roundtrip(tmp_path):
    checkpoint = _checkpoint(tmp_path, step=10)
    saved = save_lineage(
        checkpoint, parent_mode="full_model", parent_base_sha256="b" * 64,
        config_sha256="cfg", dataset_sha256="data", code_commit="commit",
        seed=11, lora_seed=12,
    )
    loaded = validate_adapter_lineage(
        checkpoint, expected_base_sha256="b" * 64, expected_config_sha256="cfg",
        expected_dataset_sha256="data", expected_seed=11, require_resumable=True,
    )
    assert loaded == saved and loaded["adapter_sha256"] == file_sha256(checkpoint / "adapter_model.safetensors")


# 10. probe loader constructs full base + lineage-compatible adapter.
def test_probe_loader_full_base_plus_adapter(tmp_path, monkeypatch):
    base, model_sha, config_sha = _base(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    save_lineage(
        checkpoint, parent_mode="full_model", parent_base_sha256=model_sha,
        config_sha256="cfg", dataset_sha256="data", code_commit="commit",
        seed=1, lora_seed=2,
    )
    calls = {}

    class FakeBase:
        def eval(self):
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["base"] = path
            return FakeBase()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer"] = path
            return "tokenizer"

    class FakePeft:
        @staticmethod
        def from_pretrained(base_model, adapter_path, **kwargs):
            calls["adapter"] = adapter_path
            return FakeBase()

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModel
    transformers.AutoTokenizer = FakeTokenizer
    peft = types.ModuleType("peft")
    peft.PeftModel = FakePeft
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    spec = ParentSpec("full_model", str(base), model_sha, config_sha)
    model, tokenizer = load_probe_model(
        spec, adapter_checkpoint=str(checkpoint), device="cpu", dtype=torch.float32
    )
    assert tokenizer == "tokenizer" and calls["adapter"] == str(checkpoint)
    assert model is not None


# 11. legacy loader/trainer/probe source remains byte-for-byte unchanged.
def test_legacy_grpo_loader_regression():
    legacy = Path(__file__).resolve().parents[2] / "grpo" / "scripts"
    expected = {
        "grpo_model.py": "fb1e4749926c0300502f71e06ac94779e2463507cbb8dcec7d2884ca22841520",
        "grpo_trl_trainer.py": "4fd22ff7bc9ccddac22b23502075f930f185817a203ae767c443180145925d3e",
        "grpo_probe.py": "218d8793c9ef8177496a4fcfb502af46e73a8d493b5d455a5ea82293c9684cd3",
    }
    for name, expected_sha in expected.items():
        payload = (legacy / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected_sha
        ast.parse(payload.decode("utf-8"))
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_smokes_and_pilot.sh"
    ).read_text(encoding="utf-8")
    assert "NCCL_SOCKET_IFNAME=lo" in launcher
    assert "GLOO_SOCKET_IFNAME=lo" in launcher
    assert "NCCL_IB_DISABLE=1" in launcher


# 12. Formal training remains continuous and retains every 100-step checkpoint.
def test_formal_500_contract():
    package = Path(__file__).resolve().parents[1]
    config = json.loads((package / "config" / "formal_500.json").read_text(encoding="utf-8"))
    audit = _validate_formal_contract(config)
    assert audit == {
        "formal_run": True,
        "checkpoint_mode": "ADAPTER_ONLY_MODEL_WITH_FULL_RESUME_STATE",
        "midrun_retention_probe": "DISABLED",
        "midrun_resume_audit": "DISABLED",
        "save_steps": 100,
        "save_total_limit": 5,
    }
    assert config["retention_probe"]["group_ids"]
    assert config["retention_probe"]["excluded_from_training"] is True


# 13. Formal preflight fails closed if checkpoint retention or probes drift.
def test_formal_500_contract_rejects_midrun_work():
    package = Path(__file__).resolve().parents[1]
    config = json.loads((package / "config" / "formal_500.json").read_text(encoding="utf-8"))
    config["checkpoint"]["save_total_limit"] = 4
    config["retention_probe"]["enabled"] = True
    with pytest.raises(RuntimeError, match="formal GRPO-1 contract mismatch"):
        _validate_formal_contract(config)


def test_original_formal_contract_still_allows_larger_retention_limit():
    package = Path(__file__).resolve().parents[1]
    config = json.loads((package / "config" / "formal_500.json").read_text(encoding="utf-8"))
    config["checkpoint"]["save_total_limit"] = 6
    assert _validate_formal_contract(config)["save_total_limit"] == 6


def test_formal_500_final_only_contract():
    package = Path(__file__).resolve().parents[1]
    config = json.loads(
        (package / "config" / "formal_500_final_only.json").read_text(encoding="utf-8")
    )
    audit = _validate_formal_contract(config)
    assert audit == {
        "formal_run": True,
        "checkpoint_mode": "ADAPTER_ONLY_MODEL_WITH_FULL_RESUME_STATE",
        "midrun_retention_probe": "DISABLED",
        "midrun_resume_audit": "DISABLED",
        "save_steps": 500,
        "save_total_limit": 1,
    }
    assert config["optimization"] == json.loads(
        (package / "config" / "formal_500.json").read_text(encoding="utf-8")
    )["optimization"]


def test_formal_500_final_only_rejects_extra_checkpoints():
    package = Path(__file__).resolve().parents[1]
    config = json.loads(
        (package / "config" / "formal_500_final_only.json").read_text(encoding="utf-8")
    )
    config["checkpoint"]["save_steps"] = 250
    config["checkpoint"]["save_total_limit"] = 2
    with pytest.raises(RuntimeError, match="formal GRPO-1 contract mismatch"):
        _validate_formal_contract(config)


# 14. Formal training cannot enter trainer.train with a non-LoRA optimizer.
def test_formal_startup_audit():
    contract = {"formal_run": True}
    trainable = {
        "base_trainable_parameter_count": 0,
        "lora_trainable_parameter_count": 87_293_952,
        "lora_trainable_tensor_count": 504,
    }
    optimizer = {
        "optimizer_lora_only": True,
        "optimizer_parameter_tensor_count": 504,
    }
    assert _assert_formal_startup_audit(contract, trainable, optimizer)["ready_to_train"]
    optimizer["optimizer_parameter_tensor_count"] = 505
    with pytest.raises(RuntimeError, match="startup audit failed"):
        _assert_formal_startup_audit(contract, trainable, optimizer)
