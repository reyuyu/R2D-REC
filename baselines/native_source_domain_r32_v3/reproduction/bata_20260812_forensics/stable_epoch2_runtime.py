"""Deterministic checkpoint-553 to checkpoint-1106 continuation controls."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from stable_pilot_runtime import (
    EXPECTED_SCHEDULER_HORIZON,
    EXPECTED_WORLD_SIZE,
    _expected_checkpoint,
    install_trusted_checkpoint_compatibility,
    runtime_snapshot,
)


EXPECTED_START_STEP = 553
EXPECTED_STOP_STEP = 1106

# This runs during module import, before Trainer attempts to restore optimizer/RNG files.
install_trusted_checkpoint_compatibility()


class StableEpoch2Callback(TrainerCallback):
    """Validate the resume contract and save the completed two-epoch checkpoint."""

    def __init__(self, stop_step: int = EXPECTED_STOP_STEP) -> None:
        self.stop_step = int(stop_step)
        self.started_at = time.time()

    @property
    def evidence_path(self) -> Path:
        root = Path(os.environ["BATA_STABLE_EVIDENCE_DIR"])
        return root / f"runtime_rank{int(os.environ.get('RANK', '0'))}.json"

    def _write(self, payload: dict[str, Any]) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.evidence_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.evidence_path)

    def on_train_begin(self, args, state, control, **kwargs):
        if int(state.max_steps) != EXPECTED_SCHEDULER_HORIZON:
            raise RuntimeError(
                f"epoch-2 continuation requires scheduler horizon {EXPECTED_SCHEDULER_HORIZON}, "
                f"got {state.max_steps}"
            )
        if int(state.global_step) != EXPECTED_START_STEP:
            raise RuntimeError(
                f"epoch-2 continuation must resume at step {EXPECTED_START_STEP}, "
                f"got {state.global_step}"
            )
        if int(os.environ.get("WORLD_SIZE", "1")) != EXPECTED_WORLD_SIZE:
            raise RuntimeError("epoch-2 continuation requires exactly four ranks")
        checkpoint, expected = _expected_checkpoint()
        snapshot = runtime_snapshot()
        snapshot.update(
            {
                "status": "RUNNING",
                "start_global_step": int(state.global_step),
                "scheduler_horizon": int(state.max_steps),
                "stop_step": self.stop_step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected,
                "configured_seed": int(args.seed),
                "configured_data_seed": int(args.data_seed),
            }
        )
        self._write(snapshot)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        payload = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "status": "PASS" if int(state.global_step) == self.stop_step else "STOPPED",
                "completed_global_step": int(state.global_step),
                "wall_seconds": time.time() - self.started_at,
            }
        )
        self._write(payload)
        return control


def stable_epoch2_callbacks() -> list[TrainerCallback]:
    if os.environ.get("BATA_STABLE_EPOCH2") != "1":
        return []
    if os.environ.get("BATA_STABLE_V0") != "1":
        raise RuntimeError("epoch-2 continuation requires BATA-STABLE-V0")
    stop_step = int(os.environ.get("BATA_STABLE_STOP_STEP", str(EXPECTED_STOP_STEP)))
    if stop_step != EXPECTED_STOP_STEP:
        raise RuntimeError(f"epoch-2 stop step must remain {EXPECTED_STOP_STEP}")
    return [StableEpoch2Callback(stop_step)]
