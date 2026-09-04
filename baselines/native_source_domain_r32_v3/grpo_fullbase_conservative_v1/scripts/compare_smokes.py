#!/usr/bin/env python3
"""Compare two five-step smoke runs and fail if repeatability contracts diverge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from checkpointing import write_json_atomic


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {"path": path, "left_keys": sorted(left), "right_keys": sorted(right)}
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else {"path": path, "left": left, "right": right}


def _rollout_contract(summary: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "rollout_id", "route", "num_generations", "recommendation_group_ids",
        "reward_mean", "reward_std", "zero_std_ratio", "exact_count", "ab_count",
        "a_count", "invalid_count",
    )
    return [{key: row.get(key) for key in keys} for row in summary["rollouts"]]


def _parity_contract(summary: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "rollout_id", "route", "recommendation_group_ids", "prompt_token_fingerprints",
        "completion_sha256", "rewards", "advantages", "policy_losses", "exact_count",
        "ab_count", "a_count", "invalid_count",
    )
    return [{key: row.get(key) for key in keys} for row in summary["parity_log"]]


def compare(run_a: Path, run_b: Path) -> dict[str, Any]:
    summaries_a = [_load(run_a / f"run-summary-rank{rank}.json") for rank in range(4)]
    summaries_b = [_load(run_b / f"run-summary-rank{rank}.json") for rank in range(4)]
    checks: dict[str, bool] = {}
    first_divergence = None
    for rank, (left, right) in enumerate(zip(summaries_a, summaries_b)):
        comparisons = {
            f"rank{rank}_rollout_contract": (_rollout_contract(left), _rollout_contract(right)),
            f"rank{rank}_parity_contract": (_parity_contract(left), _parity_contract(right)),
            f"rank{rank}_final_lora": (
                left["lora_sample_fingerprint_after"], right["lora_sample_fingerprint_after"]
            ),
        }
        for name, (left_value, right_value) in comparisons.items():
            difference = _first_difference(left_value, right_value)
            checks[name] = difference is None
            if difference and first_divergence is None:
                first_divergence = {"check": name, **difference}
    checks.update({
        "rank0_full_lora_sha256": (
            summaries_a[0]["lora_sha256_after"] == summaries_b[0]["lora_sha256_after"]
        ),
        "base_unchanged_a": all(row["base_tensor_unchanged"] for row in summaries_a),
        "base_unchanged_b": all(row["base_tensor_unchanged"] for row in summaries_b),
        "lora_changed_a": all(row["lora_changed"] for row in summaries_a),
        "lora_changed_b": all(row["lora_changed"] for row in summaries_b),
    })
    repeatability_pass = all(checks.values())
    return {
        "status": "PASS" if repeatability_pass else "FAIL",
        "repeatability_pass": repeatability_pass,
        "checks": checks,
        "first_divergence": first_divergence,
        "smoke_a": run_a.name,
        "smoke_b": run_b.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run_a, args.run_b)
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["repeatability_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
