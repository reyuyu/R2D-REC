#!/usr/bin/env python3
"""Build the final continued-adapter smoke and pilot decision report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison, pilot = load(args.comparison), load(args.pilot)
    checkpoints = {row["step"]: row for row in pilot.get("checkpoints", [])}
    ready = (
        comparison.get("status") == "BYTE_EXACT"
        and pilot.get("status") == "PASS"
        and pilot.get("step0_adapter_parity") == "PASS"
        and pilot.get("adapter_delta_norm", 0) > 0
        and pilot.get("base_changed") is False
        and pilot.get("optimizer_lora_only") is True
        and set(checkpoints) == {10, 20}
        and all(row.get("adapter_only") and row.get("training_state_complete") for row in checkpoints.values())
    )
    value = {
        "status": "PASS" if ready else "FAIL",
        "final_state": "READY_FOR_GRPO2_CONTINUED_ADAPTER_FORMAL_TRAINING" if ready else "BLOCKED_BY_GRPO2_VALIDATION",
        "grpo2_mode": "CONTINUED_SINGLE_ADAPTER",
        "smoke_repeatability": comparison,
        "pilot": pilot,
        "formal_300_started": False,
        "grpo3_started": False,
        "merge_precision_status": "OPEN_FINAL_EXPORT_ISSUE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
