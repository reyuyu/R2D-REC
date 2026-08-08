#!/usr/bin/env python3
"""Audit recommendation dual data for CoT/final SID roles and history overlap."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

FULL_SID_RE = re.compile(
    r"<\|(?:video|prod|living|ad)_begin\|>\s*"
    r"<s_a_\d+>\s*<s_b_\d+>\s*<s_c_\d+>"
)
SID_COMPONENT_RE = re.compile(r"<s_[abc]_\d+>")


def full_sids(text: str) -> set[str]:
    return set(FULL_SID_RE.findall(text))


def quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return float(ordered[index])


def audit(path: Path, route: str) -> dict:
    result = {
        "route": route,
        "samples": 0,
        "cot_samples": 0,
        "nocot_samples": 0,
        "supervised_sid_token_count": 0,
        "think_sid_token_count": 0,
        "final_sid_token_count": 0,
        "samples_with_think_sid": 0,
        "think_full_sid_count": 0,
        "think_history_overlap_count": 0,
        "think_history_overlap_rate": 0.0,
        "gold_sid_count": 0,
        "gold_in_history_samples": 0,
        "gold_in_history_rate": 0.0,
        "think_sid_count_distribution": {"p50": 0.0, "p90": 0.0, "mean": 0.0},
    }
    think_counts: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output = str(row.get("output", ""))
            prompt = str(row.get("input", "")) + "\n" + json.dumps(row.get("history", []), ensure_ascii=False)
            result["samples"] += 1
            if route == "cot":
                result["cot_samples"] += 1
            else:
                result["nocot_samples"] += 1
            close = output.find("</think>")
            if close >= 0:
                think_text = output[:close]
                final_text = output[close + len("</think>") :]
            else:
                think_text = ""
                final_text = output
            think_components = len(SID_COMPONENT_RE.findall(think_text))
            final_components = len(SID_COMPONENT_RE.findall(final_text))
            result["supervised_sid_token_count"] += think_components + final_components
            result["think_sid_token_count"] += think_components
            result["final_sid_token_count"] += final_components
            think_sids = full_sids(think_text)
            history_sids = full_sids(prompt)
            gold_sids = full_sids(final_text)
            think_counts.append(len(think_sids))
            result["think_full_sid_count"] += len(think_sids)
            result["think_history_overlap_count"] += len(think_sids & history_sids)
            result["gold_sid_count"] += len(gold_sids)
            result["gold_in_history_samples"] += bool(gold_sids & history_sids)
            result["samples_with_think_sid"] += bool(think_sids)
    result["samples_with_think_sid_rate"] = (
        result["samples_with_think_sid"] / result["samples"] if result["samples"] else 0.0
    )
    result["think_history_overlap_rate"] = (
        result["think_history_overlap_count"] / result["think_full_sid_count"]
        if result["think_full_sid_count"]
        else 0.0
    )
    result["gold_in_history_rate"] = (
        result["gold_in_history_samples"] / result["samples"] if result["samples"] else 0.0
    )
    result["think_sid_count_distribution"] = {
        "p50": quantile(think_counts, 0.50),
        "p90": quantile(think_counts, 0.90),
        "mean": mean(think_counts) if think_counts else 0.0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--nocot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "cot": audit(args.cot, "cot"),
        "no_think": audit(args.nocot, "no_think"),
        "definition": {
            "think_sid": "complete SID occurrences before </think> in the supervised output",
            "gold_in_history": "any final answer SID intersecting Prompt history; not automatically an error",
            "think_history_overlap": "complete think SID occurrences intersecting Prompt history",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
