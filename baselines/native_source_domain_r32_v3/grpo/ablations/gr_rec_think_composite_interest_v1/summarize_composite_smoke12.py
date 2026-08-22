#!/usr/bin/env python3
"""CPU-only CLI for a completed Composite Smoke12 monitor directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke12_contract import evaluate_smoke_conditions, summarize_composite_smoke


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def summarize_run(run_dir: Path) -> dict:
    summary = summarize_composite_smoke(
        read_json(run_dir / "manifest.json"),
        read_jsonl(run_dir / "metrics.jsonl"),
        read_jsonl(run_dir / "rollouts.jsonl"),
        read_jsonl(run_dir / "composite_interest.jsonl"),
    )
    return {**summary, **evaluate_smoke_conditions(summary)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = summarize_run(args.run_dir)
    output = args.output or args.run_dir / "smoke12_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"smoke_pass": payload["smoke_pass"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
