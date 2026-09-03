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
    spec = importlib.util.spec_from_file_location(f"seed_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_source_config(path: Path) -> None:
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


def make_checkpoint(path: Path, step: int = 553) -> None:
    module = load("prepare_stable_seed.py")
    path.mkdir(parents=True)
    for name in module.REQUIRED_CHECKPOINT_FILES:
        if name == "trainer_state.json":
            (path / name).write_text(json.dumps({"global_step": step, "max_steps": 1106}), encoding="utf-8")
        else:
            (path / name).write_bytes(name.encode())


def test_seed_run_id_and_seed_are_strict():
    module = load("prepare_stable_seed.py")
    assert module.validate_seed(42) == 42
    assert module.validate_run_id("SEED-S42-20260903-120000", 42)
    for invalid in (-1, 2**31, True, "42"):
        with pytest.raises(ValueError):
            module.validate_seed(invalid)
    with pytest.raises(ValueError):
        module.validate_run_id("EPOCH2-S42-20260903-120000", 42)


def test_fresh_and_resume_configs_preserve_two_epoch_horizon(tmp_path):
    module = load("prepare_stable_seed.py")
    source = tmp_path / "source.yaml"
    make_source_config(source)
    checkpoint = tmp_path / "output/checkpoint-553"
    fresh, resume = module.build_configs(source, tmp_path / "output", checkpoint, 123)
    assert fresh["resume_from_checkpoint"] is None
    assert resume["resume_from_checkpoint"] == str(checkpoint)
    for config in (fresh, resume):
        assert config["num_train_epochs"] == 2
        assert config["seed"] == 123
        assert config["data_seed"] == 123


def test_resume_checkpoint_requires_optimizer_scheduler_and_four_rng_files(tmp_path):
    module = load("prepare_stable_seed.py")
    checkpoint = tmp_path / "checkpoint-553"
    make_checkpoint(checkpoint)
    assert set(module.checkpoint_contract(checkpoint, 553)) == set(module.REQUIRED_CHECKPOINT_FILES)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        module.checkpoint_contract(checkpoint, 553)


@pytest.mark.parametrize("start,stop", [(0, 553), (0, 1106), (553, 1106)])
def test_seed_callback_accepts_only_supported_boundaries(start, stop):
    module = load("stable_seed_runtime.py")
    callback = module.StableSeedCallback(start, stop)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    state = SimpleNamespace(global_step=stop - 1, max_steps=1106)
    callback.on_step_end(None, state, control)
    assert not control.should_save
    state.global_step = stop
    callback.on_step_end(None, state, control)
    assert control.should_save and control.should_training_stop


def test_seed_callback_factory_rejects_changed_horizon(monkeypatch):
    module = load("stable_seed_runtime.py")
    monkeypatch.setenv("BATA_STABLE_SEED", "1")
    monkeypatch.setenv("BATA_STABLE_V0", "1")
    monkeypatch.setenv("BATA_STABLE_START_STEP", "0")
    monkeypatch.setenv("BATA_STABLE_STOP_STEP", "700")
    with pytest.raises(RuntimeError, match="unsupported"):
        module.stable_seed_callbacks()


def test_launcher_has_fresh_base_and_pinned_resume_modes():
    launcher = (ROOT / "launch_stable_seed.sh").read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES="0,1,2,3"' in launcher
    assert 'export BATA_STABLE_START_MODE="base"' in launcher
    assert 'export BATA_STABLE_START_MODE="checkpoint-553"' in launcher
    assert 'export BATA_STABLE_SEED_RESUME="1"' in launcher
    assert 'checkpoint_553_sha256.json' in launcher
    assert "all four GPUs must be free" in launcher

