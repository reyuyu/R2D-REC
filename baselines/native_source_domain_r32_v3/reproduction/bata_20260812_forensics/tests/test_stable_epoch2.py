from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(name: str):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(f"epoch2_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_checkpoint(path: Path) -> None:
    module = load("prepare_stable_epoch2.py")
    path.mkdir(parents=True)
    for name in module.REQUIRED_CHECKPOINT_FILES:
        if name == "trainer_state.json":
            (path / name).write_text(json.dumps({"global_step": 553, "max_steps": 1106}), encoding="utf-8")
        else:
            (path / name).write_bytes(name.encode())


def source_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "output_dir": "/old",
                "resume_from_checkpoint": None,
                "seed": 20260806,
                "num_train_epochs": 2,
                "save_strategy": "epoch",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_seed_and_run_id_are_strict():
    module = load("prepare_stable_epoch2.py")
    assert module.validate_seed(20260806) == 20260806
    assert module.validate_run_id("EPOCH2-S20260806-20260903-120000", 20260806)
    for invalid in (-1, 2**31, True, "7"):
        with pytest.raises(ValueError):
            module.validate_seed(invalid)
    with pytest.raises(ValueError):
        module.validate_run_id("../../bad", 7)


def test_build_config_sets_isolated_output_resume_and_both_seeds(tmp_path):
    module = load("prepare_stable_epoch2.py")
    source = tmp_path / "source.yaml"
    source_config(source)
    config = module.build_config(source, tmp_path / "output", tmp_path / "checkpoint-553", 42)
    assert config["output_dir"] == str(tmp_path / "output")
    assert config["resume_from_checkpoint"] == str(tmp_path / "checkpoint-553")
    assert config["seed"] == 42
    assert config["data_seed"] == 42
    assert config["num_train_epochs"] == 2


def test_checkpoint_contract_requires_midpoint_and_all_state(tmp_path):
    module = load("prepare_stable_epoch2.py")
    checkpoint = tmp_path / "checkpoint-553"
    make_checkpoint(checkpoint)
    contract = module.checkpoint_contract(checkpoint)
    assert set(contract) == set(module.REQUIRED_CHECKPOINT_FILES)
    (checkpoint / "rng_state_3.pth").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        module.checkpoint_contract(checkpoint)


def test_epoch2_callback_stops_only_at_1106(monkeypatch, tmp_path):
    monkeypatch.setenv("BATA_STABLE_V0", "0")
    module = load("stable_epoch2_runtime.py")
    callback = module.StableEpoch2Callback()
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    state = SimpleNamespace(global_step=1105, max_steps=1106)
    callback.on_step_end(None, state, control)
    assert not control.should_save
    assert not control.should_training_stop
    state.global_step = 1106
    callback.on_step_end(None, state, control)
    assert control.should_save
    assert control.should_training_stop


def test_launcher_is_four_gpu_fail_closed_and_has_no_shell_seed_input():
    launcher = (ROOT / "launch_stable_epoch2.sh").read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES="0,1,2,3"' in launcher
    assert 'export BATA_STABLE_STOP_STEP="1106"' in launcher
    assert 'export BATA_STABLE_EPOCH2="1"' in launcher
    assert "all four GPUs must be free" in launcher
    assert "$SEED" not in launcher
    assert "$(python " not in launcher
    assert launcher.count("/data/venvs/llamafactory-01398eb-liger081/bin/python") >= 3
