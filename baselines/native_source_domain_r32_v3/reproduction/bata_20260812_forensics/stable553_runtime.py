"""Base-to-step553 controls for the BATA-STABLE-V0 recipe.

The deterministic backend setup is imported from the already validated
STABLE560 runtime. This module adds only a fresh-base start contract and a
save-and-stop callback; it does not inspect gradients or alter training math.
"""

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
    runtime_snapshot,
)


EXPECTED_START_STEP = 0
EXPECTED_STOP_STEP = 553


class Stable553StopAndSaveCallback(TrainerCallback):
    """Require a fresh base start and save before stopping at step 553."""

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
                f"BATA-STABLE-V0 requires scheduler horizon {EXPECTED_SCHEDULER_HORIZON}, "
                f"got {state.max_steps}"
            )
        if int(state.global_step) != EXPECTED_START_STEP:
            raise RuntimeError(
                f"BATA-STABLE-V0 base run must start at step {EXPECTED_START_STEP}, "
                f"got {state.global_step}"
            )
        if int(os.environ.get("WORLD_SIZE", "1")) != EXPECTED_WORLD_SIZE:
            raise RuntimeError("BATA-STABLE-V0 requires exactly four ranks")
        if os.environ.get("BATA_STABLE_START_MODE") != "base":
            raise RuntimeError("BATA-STABLE-V0 step553 run requires start mode 'base'")

        snapshot = runtime_snapshot()
        snapshot.update(
            {
                "status": "RUNNING",
                "start_mode": "base",
                "start_global_step": int(state.global_step),
                "scheduler_horizon": int(state.max_steps),
                "stop_step": self.stop_step,
                "base_contract_sha256": os.environ["BATA_STABLE_BASE_CONTRACT_SHA256"],
                "config_sha256": os.environ["BATA_STABLE_CONFIG_SHA256"],
                "source_contract_sha256": os.environ["BATA_STABLE_SOURCE_CONTRACT_SHA256"],
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


def stable553_callbacks() -> list[TrainerCallback]:
    if os.environ.get("BATA_STABLE553_V0") != "1":
        return []
    if os.environ.get("BATA_STABLE_V0") != "1":
        raise RuntimeError("base-to-553 requires the validated BATA-STABLE-V0 runtime")
    stop_step = int(os.environ.get("BATA_STABLE_STOP_STEP", str(EXPECTED_STOP_STEP)))
    if stop_step != EXPECTED_STOP_STEP:
        raise RuntimeError(f"BATA-STABLE-V0 stop step must remain {EXPECTED_STOP_STEP}")
    return [Stable553StopAndSaveCallback(stop_step)]
