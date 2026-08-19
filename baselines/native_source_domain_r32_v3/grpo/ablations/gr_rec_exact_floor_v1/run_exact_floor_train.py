"""Formal runner wrapper: GR_REC_v1 contracts with ExactFloor trainer only."""

from __future__ import annotations

import sys
from pathlib import Path

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

import run_grpo_trl_train as baseline_runner

try:
    from .exact_floor_trainer import ExactFloorRecGRPOTrainer
except ImportError:  # Direct script entry point.
    from exact_floor_trainer import ExactFloorRecGRPOTrainer


class _ManifestWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload["runner"] = "ablations/gr_rec_exact_floor_v1/run_exact_floor_train.py"
        payload["parent"] = "GR_REC_v1 / original BATA adapter"
        payload["advantage"] = {
            "name": "exact_floor_v1",
            "formula": "(reward - min(group_mean, 8.0)) / 8.0",
            "group_std_normalization": False,
        }
        return self._writer.write_manifest(payload)


def main(argv=None):
    original_monitor_factory = baseline_runner.monitor_from_env

    def exact_floor_monitor(*args, **kwargs):
        return _ManifestWriter(original_monitor_factory(*args, **kwargs))

    baseline_runner.RecGRPOTrainer = ExactFloorRecGRPOTrainer
    baseline_runner.monitor_from_env = exact_floor_monitor
    return baseline_runner.main(argv)


if __name__ == "__main__":
    main()
