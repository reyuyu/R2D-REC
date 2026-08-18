"""Assertion-test runner for DSR-Simple plus the complete old DSR suite."""
from __future__ import annotations

import importlib
import inspect


DEFAULT_MODULES = (
    "gr_rec_dsr_v1.test_dsr_contract",
    "gr_rec_dsr_v1.test_dsr_parser",
    "gr_rec_dsr_v1.test_dsr_objectives",
    "gr_rec_dsr_v1.test_dsr_monitor",
    "gr_rec_dsr_v1.test_dsr_trainer",
    "gr_rec_dsr_simple_v1.test_simple_contract",
    "gr_rec_dsr_simple_v1.test_simple_objectives",
    "gr_rec_dsr_simple_v1.test_simple_monitor",
    "gr_rec_dsr_simple_v1.test_simple_trainer",
)


def main(module_names=None):
    failures = []
    total = 0
    for module_name in module_names or DEFAULT_MODULES:
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or function.__module__ != module.__name__:
                continue
            total += 1
            try:
                function()
                print(f"[PASS] {module_name}:{name}")
            except Exception as error:
                failures.append((module_name, name, error))
                print(f"[FAIL] {module_name}:{name}: {type(error).__name__}: {error}")
    print(f"{total - len(failures)}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
