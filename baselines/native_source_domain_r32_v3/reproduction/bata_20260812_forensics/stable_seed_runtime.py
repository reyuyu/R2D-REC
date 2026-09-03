"""Fresh-base and resumable seed experiment controls for BATA SFT."""

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


MIDPOINT_STEP = 553
FINAL_STEP = 1106

# Resume compatibility must be installed before Trainer restores optimizer/RNG.
if os.environ.get("BATA_STABLE_SEED_RESUME") == "1":
    install_trusted_checkpoint_compatibility()


class StableSeedCallback(TrainerCallback):
    """Validate seed-run boundaries and save exactly at the requested stop."""

    def __init__(self, start_step: int, stop_step: int) -> None:
        self.start_step = int(start_step)
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
                f"seed experiment requires scheduler horizon {EXPECTED_SCHEDULER_HORIZON}, "
                f"got {state.max_steps}"
            )
        if int(state.global_step) != self.start_step:
            raise RuntimeError(
                f"seed experiment expected start step {self.start_step}, got {state.global_step}"
            )
        if int(os.environ.get("WORLD_SIZE", "1")) != EXPECTED_WORLD_SIZE:
            raise RuntimeError("seed experiment requires exactly four ranks")
        start_mode = os.environ.get("BATA_STABLE_START_MODE")
        expected_mode = "checkpoint-553" if self.start_step == MIDPOINT_STEP else "base"
        if start_mode != expected_mode:
            raise RuntimeError(f"seed experiment requires start mode {expected_mode!r}")

        snapshot = runtime_snapshot()
        snapshot.update(
            {
                "status": "RUNNING",
                "experiment_kind": "fresh_seed",
                "start_mode": start_mode,
                "start_global_step": int(state.global_step),
                "scheduler_horizon": int(state.max_steps),
                "stop_step": self.stop_step,
                "configured_seed": int(args.seed),
                "configured_data_seed": int(args.data_seed),
                "config_sha256": os.environ["BATA_STABLE_CONFIG_SHA256"],
            }
        )
        if self.start_step == MIDPOINT_STEP:
            checkpoint, expected = _expected_checkpoint()
            snapshot.update({"checkpoint": str(checkpoint), "checkpoint_sha256": expected})
        else:
            snapshot.update(
                {
                    "base_contract_sha256": os.environ["BATA_STABLE_BASE_CONTRACT_SHA256"],
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


def stable_seed_callbacks() -> list[TrainerCallback]:
    if os.environ.get("BATA_STABLE_SEED") != "1":
        return []
    if os.environ.get("BATA_STABLE_V0") != "1":
        raise RuntimeError("seed experiment requires BATA-STABLE-V0")
    start_step = int(os.environ.get("BATA_STABLE_START_STEP", "-1"))
    stop_step = int(os.environ.get("BATA_STABLE_STOP_STEP", "-1"))
    if (start_step, stop_step) not in ((0, MIDPOINT_STEP), (0, FINAL_STEP), (MIDPOINT_STEP, FINAL_STEP)):
        raise RuntimeError(f"unsupported seed experiment boundary: {start_step}->{stop_step}")
    return [StableSeedCallback(start_step, stop_step)]
