"""Retention summaries with explicit numerators and denominators."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def summarize(path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    by_step: dict[int, list[dict]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    result = {}
    for step, step_rows in sorted(by_step.items()):
        think = [candidate for row in step_rows for candidate in row["think"]["candidates"]]
        nothink = [candidate for row in step_rows for candidate in row["nothink"]["candidates"]]
        beam_den = 32 * len(think)
        think_success = sum(any((candidate.get("a") or 0) > 0 for candidate in row["think"]["candidates"]) for row in step_rows)
        nothink_success = sum(any(float(candidate["reward"]) > 0 for candidate in row["nothink"]["candidates"]) for row in step_rows)
        exact_num = sum((candidate.get("exact") or 0) for candidate in think)
        ab_num = sum((candidate.get("ab") or 0) for candidate in think)
        a_num = sum((candidate.get("a") or 0) for candidate in think)
        invalid_num = sum((candidate.get("invalid") or 0) for candidate in think)
        closure_num = sum(bool(candidate.get("closed")) for candidate in think)
        no_exact = sum(float(candidate["reward"]) == 8.0 for candidate in nothink)
        no_ab = sum(float(candidate["reward"]) >= 2.0 for candidate in nothink)
        no_a = sum(float(candidate["reward"]) >= 0.5 for candidate in nothink)
        no_positive = sum(float(candidate["reward"]) > 0 for candidate in nothink)
        result[str(step)] = {
            "group_count": len(step_rows),
            "think": {
                "mean_reward": statistics.fmean(float(candidate["reward"]) for candidate in think),
                "success_at_k": {"value": think_success / len(step_rows), "numerator": think_success, "denominator": len(step_rows)},
                "success_at_32": {"value": sum((candidate.get("a") or 0) > 0 for candidate in think) / len(think), "numerator": sum((candidate.get("a") or 0) > 0 for candidate in think), "denominator": len(think)},
                "exact": {"value": exact_num / beam_den, "numerator": exact_num, "denominator": beam_den},
                "ab_plus": {"value": ab_num / beam_den, "numerator": ab_num, "denominator": beam_den},
                "a_plus": {"value": a_num / beam_den, "numerator": a_num, "denominator": beam_den},
                "closure": {"value": closure_num / len(think), "numerator": closure_num, "denominator": len(think)},
                "invalid": {"value": invalid_num / beam_den, "numerator": invalid_num, "denominator": beam_den},
            },
            "nothink": {
                "mean_reward": statistics.fmean(float(candidate["reward"]) for candidate in nothink),
                "success_at_k": {"value": nothink_success / len(step_rows), "numerator": nothink_success, "denominator": len(step_rows)},
                "exact": {"value": no_exact / len(nothink), "numerator": no_exact, "denominator": len(nothink)},
                "ab_plus": {"value": no_ab / len(nothink), "numerator": no_ab, "denominator": len(nothink)},
                "a_plus": {"value": no_a / len(nothink), "numerator": no_a, "denominator": len(nothink)},
                "positive_candidate_rate": {"value": no_positive / len(nothink), "numerator": no_positive, "denominator": len(nothink)},
            },
        }
    return result
