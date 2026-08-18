"""Frozen fairness and runner contract for DSR-Simple."""
from __future__ import annotations

from gr_rec_dsr_v1.dsr_contract import (
    PROBE_EVERY_STEPS,
    PROBE_GROUP_IDS,
    PROBE_SEED,
    TRAIN_SEED,
    enforce_formal_contract as _enforce_dsr_contract,
)


OPTIMIZER_SCHEDULE_SHA256 = "ac86490e5660e365fe4adedb6ac9321cffb95db1636b2f4bafad99841ba508b7"
EXPECTED_SCHEDULE = {
    "raw_groups": 1549,
    "excluded_probe_groups": 4,
    "candidate_groups": 1545,
    "trained_groups": 1544,
    "think_rollouts": 386,
    "nothink_rollouts": 772,
    "optimizer_steps": 2316,
}


def enforce_simple_contract(argv) -> list[str]:
    values = _enforce_dsr_contract(argv)
    if "--run-id" in values:
        position = values.index("--run-id")
        if position + 1 >= len(values) or not values[position + 1].startswith(
            "GR-REC-DSR-SIMPLE-V1-"
        ):
            raise ValueError("DSR-Simple --run-id must start with GR-REC-DSR-SIMPLE-V1-")
    return values


__all__ = [
    "EXPECTED_SCHEDULE",
    "OPTIMIZER_SCHEDULE_SHA256",
    "PROBE_EVERY_STEPS",
    "PROBE_GROUP_IDS",
    "PROBE_SEED",
    "TRAIN_SEED",
    "enforce_simple_contract",
]
