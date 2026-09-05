#!/usr/bin/env python3
"""Build the public GRPO-2 smoke/pilot decision report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    checkpoints = {row["step"]: row for row in pilot["checkpoints"]}
    ready = (
        comparison.get("status") == "BYTE_EXACT"
        and pilot.get("status") == "PASS"
        and set(checkpoints) == {10, 20}
        and all(row.get("adapter_only") and row.get("training_state_complete") for row in checkpoints.values())
        and pilot.get("parent_unchanged") is True
    )
    value = {
        "status": "PASS" if ready else "FAIL",
        "final_state": "READY_FOR_GRPO2_PARENT_FINALIZATION" if ready else "GRPO2_PILOT_BLOCKED",
        "test_parent_only": True,
        "canonical_grpo1_parent": False,
        "historical_source": "Positive-A0 V3 Think G4 + four independent G8 FullSID",
        "historical_learning_rate": 1e-6,
        "pilot_learning_rate": 2e-7,
        "smoke_repeatability": comparison,
        "pilot": pilot,
        "formal_training_started": False,
        "grpo3_started": False,
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
