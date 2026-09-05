#!/usr/bin/env python3
"""Compare two independent five-step MC_USER runs for byte-exact determinism."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {"path": path, "left_keys": sorted(left), "right_keys": sorted(right)}
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else {"path": path, "left": left, "right": right}


def compare(run_a: Path, run_b: Path) -> dict[str, Any]:
    summary_a, summary_b = load_json(run_a / "summary.json"), load_json(run_b / "summary.json")
    evidence_a = load_jsonl(run_a / "determinism_evidence.jsonl")
    evidence_b = load_jsonl(run_b / "determinism_evidence.jsonl")
    adapter_a = run_a / "final_adapter" / "adapter_model.safetensors"
    adapter_b = run_b / "final_adapter" / "adapter_model.safetensors"
    adapter_sha_a, adapter_sha_b = file_sha256(adapter_a), file_sha256(adapter_b)

    first_divergence = None
    for index, (left, right) in enumerate(zip(evidence_a, evidence_b), 1):
        difference = first_difference(left, right)
        if difference:
            first_divergence = {"step": index, **difference}
            break
    if first_divergence is None and len(evidence_a) != len(evidence_b):
        first_divergence = {
            "step": min(len(evidence_a), len(evidence_b)) + 1,
            "path": "$",
            "left_length": len(evidence_a),
            "right_length": len(evidence_b),
        }
    checks = {
        "smoke_a_pass": summary_a.get("status") == "PASS",
        "smoke_b_pass": summary_b.get("status") == "PASS",
        "smoke_a_five_optimizer_steps": summary_a.get("optimizer_step") == 5,
        "smoke_b_five_optimizer_steps": summary_b.get("optimizer_step") == 5,
        "five_evidence_rows_each": len(evidence_a) == len(evidence_b) == 5,
        "registered_dataset_used_by_trainer": bool(
            summary_a.get("registered_dataset_used_by_trainer")
            and summary_b.get("registered_dataset_used_by_trainer")
        ),
        "training_semantics_unchanged": not bool(
            summary_a.get("training_semantics_changed")
            or summary_b.get("training_semantics_changed")
        ),
        "step_evidence_exact": first_divergence is None,
        "final_adapter_byte_exact": adapter_sha_a == adapter_sha_b,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "SMOKE_A": "PASS" if checks["smoke_a_pass"] else "FAIL",
        "SMOKE_B": "PASS" if checks["smoke_b_pass"] else "FAIL",
        "FIRST_DIVERGENCE": "NONE" if first_divergence is None else first_divergence,
        "DETERMINISM_LEVEL": "BYTE_EXACT" if passed else "NOT_BYTE_EXACT",
        "SMOKE_A_FINAL_ADAPTER_SHA256": adapter_sha_a,
        "SMOKE_B_FINAL_ADAPTER_SHA256": adapter_sha_b,
        "REGISTERED_DATASET_USED_BY_TRAINER": "YES" if checks["registered_dataset_used_by_trainer"] else "NO",
        "TRAINING_SEMANTICS_CHANGED": "NO" if checks["training_semantics_unchanged"] else "YES",
        "READY_FOR_GRPO3_FORMAL_TRAINING": "YES" if passed else "NO",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run_a, args.run_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
