"""Fail-closed import provenance for the Composite runtime chain."""
from __future__ import annotations

from pathlib import Path
from typing import Any


GRPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SCRIPTS_DIR = (GRPO_ROOT / "scripts").resolve()


def evaluate_runtime_import_provenance(
    grpo_trainer_module: Any,
    monitor_writer_module: Any,
    *,
    expected_scripts_dir: Path = EXPECTED_SCRIPTS_DIR,
) -> dict[str, Any]:
    expected_scripts_dir = Path(expected_scripts_dir).resolve()
    expected_trainer = (expected_scripts_dir / "grpo_trl_trainer.py").resolve()
    expected_monitor = (expected_scripts_dir / "monitor" / "writer.py").resolve()
    actual_trainer = Path(grpo_trainer_module.__file__).resolve()
    actual_monitor = Path(monitor_writer_module.__file__).resolve()
    writer = monitor_writer_module.MonitorWriter
    available = hasattr(writer, "write_composite")
    passed = (
        actual_trainer == expected_trainer
        and actual_monitor == expected_monitor
        and available
    )
    return {
        "grpo_trl_trainer_module_path": str(actual_trainer),
        "monitor_writer_module_path": str(actual_monitor),
        "expected_grpo_trl_trainer_module_path": str(expected_trainer),
        "expected_monitor_writer_module_path": str(expected_monitor),
        "monitor_write_composite_available": available,
        "runtime_import_provenance": "PASS" if passed else "FAIL",
    }


def assert_runtime_import_provenance() -> dict[str, Any]:
    import grpo_trl_trainer
    import monitor.writer

    result = evaluate_runtime_import_provenance(grpo_trl_trainer, monitor.writer)
    if result["runtime_import_provenance"] != "PASS":
        raise RuntimeError(
            "MONITOR_IMPORT_PROVENANCE_MISMATCH: "
            f"expected={result['expected_monitor_writer_module_path']} "
            f"actual={result['monitor_writer_module_path']} "
            f"write_composite={result['monitor_write_composite_available']}"
        )
    return result
