import importlib.util
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).resolve().parents[1] / "prepare_kernel_matrix.py"
SPEC = importlib.util.spec_from_file_location("bata_prepare_kernel_matrix_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
KERNEL_CASES = MODULE.KERNEL_CASES
configure_prepared_replay = MODULE.configure_prepared_replay


def prepared_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "F2-A"
    (run_dir / "evidence").mkdir(parents=True)
    (run_dir / "replay_config.yaml").write_text(
        yaml.safe_dump(
            {
                "flash_attn": "fa2",
                "enable_liger_kernel": True,
                "gradient_accumulation_steps": 16,
                "num_train_epochs": 2,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_dir


@pytest.mark.parametrize("case", sorted(KERNEL_CASES))
def test_configures_only_kernel_switches(tmp_path: Path, case: str) -> None:
    run_dir = prepared_run(tmp_path)
    record = configure_prepared_replay(run_dir, case)
    config = yaml.safe_load((run_dir / "replay_config.yaml").read_text(encoding="utf-8"))

    assert config["flash_attn"] == KERNEL_CASES[case]["flash_attn"]
    assert config["enable_liger_kernel"] is KERNEL_CASES[case]["enable_liger_kernel"]
    assert config["gradient_accumulation_steps"] == 16
    assert config["num_train_epochs"] == 2
    assert record["case"] == case


def test_rejects_existing_evidence(tmp_path: Path) -> None:
    run_dir = prepared_run(tmp_path)
    (run_dir / "evidence" / "rank0.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence"):
        configure_prepared_replay(run_dir, "F2")


def test_rejects_nonhistorical_source_contract(tmp_path: Path) -> None:
    run_dir = prepared_run(tmp_path)
    config_path = run_dir / "replay_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["flash_attn"] = "disabled"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="historical F1"):
        configure_prepared_replay(run_dir, "F3")
