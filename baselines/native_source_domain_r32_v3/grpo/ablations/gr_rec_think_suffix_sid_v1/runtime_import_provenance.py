"""Fail-closed import provenance for the Think suffix training chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any


GRPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SCRIPTS_DIR = (GRPO_ROOT / "scripts").resolve()


def evaluate_runtime_import_provenance(
    baseline_runner_module: Any,
    grpo_model_module: Any,
    grpo_trainer_module: Any,
    monitor_writer_module: Any,
    *,
    expected_scripts_dir: Path = EXPECTED_SCRIPTS_DIR,
) -> dict:
    expected_scripts_dir = Path(expected_scripts_dir).resolve()
    expected = {
        "baseline_runner_module_path": (expected_scripts_dir / "run_grpo_trl_train.py").resolve(),
        "grpo_model_module_path": (expected_scripts_dir / "grpo_model.py").resolve(),
        "grpo_trl_trainer_module_path": (expected_scripts_dir / "grpo_trl_trainer.py").resolve(),
        "monitor_writer_module_path": (expected_scripts_dir / "monitor" / "writer.py").resolve(),
    }
    modules = {
        "baseline_runner_module_path": baseline_runner_module,
        "grpo_model_module_path": grpo_model_module,
        "grpo_trl_trainer_module_path": grpo_trainer_module,
        "monitor_writer_module_path": monitor_writer_module,
    }
    actual = {
        name: Path(module.__file__).resolve()
        for name, module in modules.items()
    }
    trainer_identity = (
        baseline_runner_module.RecGRPOTrainer is grpo_trainer_module.RecGRPOTrainer
    )
    passed = actual == expected and trainer_identity
    result = {name: str(path) for name, path in actual.items()}
    result.update({f"expected_{name}": str(path) for name, path in expected.items()})
    result.update({
        "baseline_trainer_identity": trainer_identity,
        "runtime_import_provenance": "PASS" if passed else "FAIL",
    })
    return result


def assert_runtime_import_provenance() -> dict:
    import grpo_model
    import grpo_trl_trainer
    import monitor.writer
    import run_grpo_trl_train

    result = evaluate_runtime_import_provenance(
        run_grpo_trl_train, grpo_model, grpo_trl_trainer, monitor.writer
    )
    if result["runtime_import_provenance"] != "PASS":
        raise RuntimeError(
            "THINK_SUFFIX_RUNTIME_IMPORT_PROVENANCE_MISMATCH: "
            f"expected_scripts={EXPECTED_SCRIPTS_DIR} "
            f"actual_trainer={result['grpo_trl_trainer_module_path']} "
            f"actual_monitor={result['monitor_writer_module_path']}"
        )
    return result
