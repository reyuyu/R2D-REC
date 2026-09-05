#!/usr/bin/env python3
"""Demand byte-exact evidence and adapter equality from GRPO-2 smoke A/B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contracts import file_sha256

VOLATILE_KEYS = {"timestamp", "created_at", "elapsed", "wall_time", "rollout_sec"}


def _is_volatile_key(key: str) -> bool:
    return key in VOLATILE_KEYS or key.endswith("_wall_sec")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable(item)
            for key, item in sorted(value.items())
            if not _is_volatile_key(key)
        }
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _difference(left: Any, right: Any, path: str = "$") -> dict | None:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {"path": path, "left_keys": sorted(left), "right_keys": sorted(right)}
        for key in sorted(left):
            found = _difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            found = _difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if left == right else {"path": path, "left": left, "right": right}


def compare(run_a: Path, run_b: Path, monitor_a: Path, monitor_b: Path) -> dict:
    checks: dict[str, bool] = {}
    first = None
    for rank in range(4):
        pairs = {
            f"rank{rank}_trainer_evidence": (
                _load(run_a / f"trainer-evidence-rank{rank}.json"),
                _load(run_b / f"trainer-evidence-rank{rank}.json"),
            ),
            f"rank{rank}_step_evidence": (
                _jsonl(run_a / f"step-evidence-rank{rank}.jsonl"),
                _jsonl(run_b / f"step-evidence-rank{rank}.jsonl"),
            ),
        }
        for name, (left, right) in pairs.items():
            difference = _difference(_stable(left), _stable(right))
            checks[name] = difference is None
            if difference and first is None:
                first = {"check": name, **difference}
    sample_difference = _difference(
        _stable(_jsonl(monitor_a / "sample8_fullsid.jsonl")),
        _stable(_jsonl(monitor_b / "sample8_fullsid.jsonl")),
    )
    checks["sample8_fullsid_events"] = sample_difference is None
    if sample_difference and first is None:
        first = {"check": "sample8_fullsid_events", **sample_difference}
    probe_difference = _difference(
        _stable(_jsonl(monitor_a / "probes.jsonl")),
        _stable(_jsonl(monitor_b / "probes.jsonl")),
    )
    checks["probe4_beam32_events"] = probe_difference is None
    if probe_difference and first is None:
        first = {"check": "probe4_beam32_events", **probe_difference}
    sha_a = file_sha256(run_a / "checkpoint-5" / "adapter_model.safetensors")
    sha_b = file_sha256(run_b / "checkpoint-5" / "adapter_model.safetensors")
    checks["adapter_byte_exact"] = sha_a == sha_b
    if all(checks.values()):
        status = "BYTE_EXACT"
    elif checks.get("sample8_fullsid_events") and checks.get("probe4_beam32_events"):
        status = "ROLLOUT_EXACT_NUMERIC_DRIFT"
    else:
        status = "ROLLOUT_DIVERGED"
    return {
        "status": status,
        "determinism_level": status,
        "pilot_allowed": status == "BYTE_EXACT",
        "checks": checks,
        "adapter_sha256_a": sha_a,
        "adapter_sha256_b": sha_b,
        "first_divergence": first or "NONE",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--monitor-a", type=Path, required=True)
    parser.add_argument("--monitor-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run_a, args.run_b, args.monitor_a, args.monitor_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "BYTE_EXACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
