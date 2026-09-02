#!/usr/bin/env python3
"""Configure prepared one-step replays for the FA2/Liger isolation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


KERNEL_CASES = {
    "F1": {"flash_attn": "fa2", "enable_liger_kernel": True},
    "F2": {"flash_attn": "disabled", "enable_liger_kernel": True},
    "F3": {"flash_attn": "fa2", "enable_liger_kernel": False},
    "F4": {"flash_attn": "disabled", "enable_liger_kernel": False},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_prepared_replay(run_dir: Path, case: str) -> dict[str, object]:
    if case not in KERNEL_CASES:
        raise ValueError(f"Unknown kernel case: {case}")
    config_path = run_dir / "replay_config.yaml"
    evidence_dir = run_dir / "evidence"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if list(evidence_dir.glob("rank*.jsonl")):
        raise RuntimeError(f"Refusing to modify a replay with evidence: {run_dir}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    before = {
        "flash_attn": config.get("flash_attn"),
        "enable_liger_kernel": config.get("enable_liger_kernel"),
    }
    if before != KERNEL_CASES["F1"]:
        raise RuntimeError(f"Prepared config is not the historical F1 kernel contract: {before}")

    config.update(KERNEL_CASES[case])
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    return {
        "case": case,
        "run_dir": str(run_dir),
        **KERNEL_CASES[case],
        "config_sha256": sha256_file(config_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases", required=True)
    args = parser.parse_args()

    records = []
    for case in args.cases:
        for suffix in ("A", "B"):
            records.append(configure_prepared_replay(args.runs_root / f"{case}-{suffix}", case))
    manifest = {
        "diagnostic_only": True,
        "comparison_rule": "compare A/B only within each case",
        "cases": records,
    }
    manifest_path = args.runs_root / "kernel_matrix.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
