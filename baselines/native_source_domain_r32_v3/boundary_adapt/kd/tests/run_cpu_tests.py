"""Minimal dependency-free runner for the repository's plain assert tests."""
from __future__ import annotations

import runpy
from pathlib import Path

FILES = (Path(__file__).with_name("test_bridge_to_bare_kd.py"), Path(__file__).parents[2] / "tests/test_boundary_adapt.py")


if __name__ == "__main__":
    total = 0
    for path in FILES:
        namespace = runpy.run_path(str(path))
        for name, value in namespace.items():
            if name.startswith("test_") and callable(value):
                value(); total += 1; print(f"PASS {path.name}::{name}")
    print(f"TOTAL_PASS={total}")
