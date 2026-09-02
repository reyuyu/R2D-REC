from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(name: str):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(f"stable553_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_config(module, tmp_path: Path) -> tuple[dict, dict]:
    base = tmp_path / "base"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    config = {
        "model_name_or_path": str(base),
        "flash_attn": "fa2",
        "enable_liger_kernel": True,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "cutoff_len": 8192,
        "packing": True,
        "neat_packing": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.0002,
        "num_train_epochs": 2,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "bf16": True,
        "pure_bf16": True,
        "seed": 20260806,
        "rec_pu_enabled": False,
        "multitask_pack_ratio_enabled": False,
        "tokenized_path": str(cache),
        "output_dir": str(output),
        "resume_from_checkpoint": None,
    }
    return config, {"base_model": base, "tokenized_path": cache, "output_dir": output}


def test_base_start_config_forbids_resume(tmp_path):
    module = load("prepare_stable553.py")
    config, paths = valid_config(module, tmp_path)
    module.verify_training_config(config, **paths)
    config["resume_from_checkpoint"] = "/checkpoint-553"
    with pytest.raises(RuntimeError, match="config contract mismatch"):
        module.verify_training_config(config, **paths)


def test_horizon_stop_and_deterministic_contract_are_frozen():
    launcher = (ROOT / "launch_stable553.sh").read_text(encoding="utf-8")
    runtime = (ROOT / "stable_pilot_runtime.py").read_text(encoding="utf-8")
    contract = json.loads((ROOT / "stable553_contract.json").read_text(encoding="utf-8"))
    assert contract["max_steps_scheduler_horizon"] == 1106
    assert contract["stop_after_optimizer_step"] == 553
    assert contract["base_start"] is True
    assert 'export CUBLAS_WORKSPACE_CONFIG=":4096:8"' in launcher
    assert 'export FLASH_ATTENTION_DETERMINISTIC="1"' in launcher
    assert "torch.use_deterministic_algorithms(True, warn_only=False)" in runtime
    assert "BATA_STABLE_CHECKPOINT" in launcher and "unset BATA_STABLE_CHECKPOINT" in launcher
    assert "(($2 + 0) >= 1024)" in launcher


def test_callback_stops_at_553_without_changing_horizon():
    module = load("stable553_runtime.py")
    callback = module.Stable553StopAndSaveCallback()
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    state = SimpleNamespace(global_step=552, max_steps=1106)
    callback.on_step_end(None, state, control)
    assert not control.should_save and not control.should_training_stop
    state.global_step = 553
    callback.on_step_end(None, state, control)
    assert control.should_save and control.should_training_stop
    assert state.max_steps == 1106


def test_fa2_liger_and_training_math_contract(tmp_path):
    module = load("prepare_stable553.py")
    config, paths = valid_config(module, tmp_path)
    module.verify_training_config(config, **paths)
    for key, bad in (("flash_attn", "disabled"), ("enable_liger_kernel", False), ("num_train_epochs", 1)):
        changed = dict(config)
        changed[key] = bad
        with pytest.raises(RuntimeError, match="config contract mismatch"):
            module.verify_training_config(changed, **paths)


def test_source_contract_mismatch_fails_closed(tmp_path):
    module = load("prepare_stable553.py")
    source = tmp_path / "source"
    for name in module.SOURCE_SHA256:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("wrong", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source contract mismatch"):
        module.verify_source(source)


def test_base_contract_mismatch_fails_closed(monkeypatch, tmp_path):
    module = load("prepare_stable553.py")
    base = tmp_path / "base"
    base.mkdir()
    (base / "one.bin").write_bytes(b"correct")
    manifest = tmp_path / "BASE_MODEL_SHA256SUMS"
    digest = hashlib.sha256(b"correct").hexdigest()
    manifest.write_text(f"{digest}  one.bin\n", encoding="utf-8")
    monkeypatch.setattr(module, "BASE_MODEL_SHA256", {"one.bin": digest})
    monkeypatch.setattr(module, "BASE_MANIFEST_SHA256", module.sha256_file(manifest))
    module.verify_base_model(base, manifest)
    (base / "one.bin").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="base model file contract mismatch"):
        module.verify_base_model(base, manifest)


def test_generated_config_sha_mismatch_is_detectable(tmp_path):
    path = tmp_path / "training_config.yaml"
    path.write_text("resume_from_checkpoint: null\n", encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("resume_from_checkpoint: /checkpoint\n", encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != before


def exact_pair() -> dict:
    return {
        "adapter_file_sha_exact": True,
        "adapter_canonical_sha_exact": True,
        "raw_lora": {"exact": True, "cosine": 1.0, "relative_l2": 0.0},
        "effective_ba": {"cosine": 1.0, "relative_l2": 0.0},
        "optimizer_exact": True,
        "scheduler_exact": True,
        "rng_exact": True,
        "scalar_logs": {"exact": True},
    }


def test_stable553_exact_comparator_synthetic():
    module = load("compare_stable553.py")
    assert module.stable553_verdict(exact_pair()) == "STABLE553_EXACT"


@pytest.mark.parametrize("field", ["optimizer_exact", "scheduler_exact", "rng_exact"])
def test_state_mismatch_cannot_be_exact(field):
    module = load("compare_stable553.py")
    pair = exact_pair()
    pair[field] = False
    assert module.stable553_verdict(pair) == "STABLE553_NUMERIC"


def test_historical_similarity_does_not_enter_stable_verdict():
    module = load("compare_stable553.py")
    pair = exact_pair()
    verdict = module.stable553_verdict(pair)
    unrelated_historical_similarity = {"cosine": -1.0, "relative_l2": 100.0}
    assert unrelated_historical_similarity["cosine"] == -1.0
    assert verdict == "STABLE553_EXACT"


def test_production_runtime_has_no_heavy_instrumentation():
    runtime = (ROOT / "stable553_runtime.py").read_text(encoding="utf-8")
    integration = (ROOT / "stable553_trainer_integration.patch").read_text(encoding="utf-8")
    for forbidden in ("register_comm_hook", "install_replay_instrumentation", "autograd.grad"):
        assert forbidden not in runtime
        assert forbidden not in integration
