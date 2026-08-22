"""CPU-only metric quality and historical-trace audit."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time

from .interest_metric import beam_utility, composite_reward, score_interest_cot
from .provenance import DEFAULT_GRPO, DEFAULT_SOURCE, load_gold, load_think_groups
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import SID_RE, extract_interest_units

DEFAULT_TRACE = Path("/data/GRPO/runs/GR-REC-CLAMP-BRIDGE-V1-G8BASE-FORMAL1500-20260821/traces/traces.jsonl")


def make_cot(items: list[str]) -> str:
    return "<think>\n\u3010\u5174\u8da3\u5f52\u7eb3\u3011\n" + "\n".join(str(i) + ". " + item for i, item in enumerate(items, 1)) + "\n</think>"


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else None


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for position in range(cursor, end):
            result[order[position]] = rank
        cursor = end
    return result


def correlation(left: list[float], right: list[float]) -> dict:
    return {"pearson": pearson(left, right), "spearman": pearson(ranks(left), ranks(right))}


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def audit_variants(groups: dict[str, dict], gold: dict[str, str], sample_size: int) -> tuple[dict, list[str]]:
    strata = defaultdict(list)
    for group_id in sorted(gold):
        record = groups[group_id]
        parsed = extract_interest_units(gold[group_id], record["prompt"])
        if not parsed.parser_success:
            continue
        strata[(record.get("target_domain"), len(parsed.units))].append(group_id)
    rng = random.Random(20260822)
    ordered_strata = sorted(strata)
    selected = []
    while len(selected) < min(sample_size, len(gold)):
        progress = False
        for key in ordered_strata:
            remaining = [item for item in strata[key] if item not in selected]
            if remaining:
                selected.append(rng.choice(remaining))
                progress = True
                if len(selected) >= sample_size:
                    break
        if not progress:
            break
    values = defaultdict(list)
    real_negative_pool = []
    for other_group_id in sorted(gold):
        other_parsed = extract_interest_units(gold[other_group_id], groups[other_group_id]["prompt"])
        if other_parsed.parser_success:
            real_negative_pool.extend((other_group_id, unit.raw_text) for unit in other_parsed.units)
    wrong_sid = "<|video_begin|><s_a_99999><s_b_99999><s_c_99999>"
    for sample_index, group_id in enumerate(selected):
        record, gold_cot = groups[group_id], gold[group_id]
        parsed = extract_interest_units(gold_cot, record["prompt"])
        items = [unit.raw_text for unit in parsed.units]
        if not items:
            continue
        available_negatives = [text for other_group_id, text in real_negative_pool if other_group_id != group_id]
        extra_count = 1 + sample_index % 3
        real_extras = rng.sample(available_negatives, extra_count)
        variants = {
            "identity": items,
            "reorder": list(reversed(items)),
            "drop_one": items[:-1],
            "drop_half": items[:len(items) // 2],
            "one_interest": items[:1],
            "duplicate_one": [items[0]] * len(items),
            "extra_real_negative": items + real_extras,
            "remove_sid": [SID_RE.sub("", item) for item in items],
            "wrong_sid": [SID_RE.sub(wrong_sid, item) for item in items],
            "light_lexical": [item.replace("，", " ").replace("、", " ").replace("用户", "") for item in items],
        }
        for name, variant in variants.items():
            values[name].append(score_interest_cot(make_cot(variant), gold_cot, record["prompt"]).cot_utility)
    return {name: summarize(scores) for name, scores in sorted(values.items())}, selected


def audit_history(groups: dict[str, dict], gold: dict[str, str], trace_path: Path) -> dict:
    rows = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            trace = json.loads(line)
            if trace.get("route") != "think" or trace.get("group_id") not in gold:
                continue
            group_id = trace["group_id"]
            prompt = groups[group_id]["prompt"]
            gold_parsed = extract_interest_units(gold[group_id], prompt)
            if not gold_parsed.parser_success:
                continue
            for candidate in trace.get("candidates", []):
                completion = candidate.get("completion") or ""
                score = score_interest_cot(completion, gold[group_id], prompt)
                parsed = extract_interest_units(completion, prompt)
                if not parsed.parser_success:
                    continue
                raw_n = len(parsed.units)
                grounded_n = sum(bool(unit.grounded_evidence_sids) for unit in parsed.units)
                raw_reward = float(candidate.get("reward") or 0.0)
                rows.append({
                    "group_id": group_id,
                    "step": trace.get("step"),
                    "length": float(candidate.get("completion_length") or len(completion)),
                    "raw_n": float(raw_n),
                    "grounded_n": float(grounded_n),
                    "grounding_coverage": grounded_n / raw_n if raw_n else 0.0,
                    "candidate_parser_success": parsed.parser_success,
                    "gold_parser_success": score.gold_parser_success,
                    "matched_interest_count": score.matched_interest_count,
                    "u_cot": score.cot_utility,
                    "u_beam": beam_utility(raw_reward),
                    "total": composite_reward(raw_reward, score.cot_utility),
                })
    if not rows:
        return {"candidate_count": 0}
    u_cot = [row["u_cot"] for row in rows]
    result = {
        "candidate_count": len(rows),
        "u_cot_vs_completion_length": correlation(u_cot, [row["length"] for row in rows]),
        "candidate_parser_success_rate": sum(row["candidate_parser_success"] for row in rows) / len(rows),
        "gold_parser_success_rate": sum(row["gold_parser_success"] for row in rows) / len(rows),
        "nonzero_match_rate": sum(row["matched_interest_count"] > 0 for row in rows) / len(rows),
        "u_cot_vs_raw_n": correlation(u_cot, [row["raw_n"] for row in rows]),
        "u_cot_vs_grounded_n": correlation(u_cot, [row["grounded_n"] for row in rows]),
        "u_cot_vs_grounding_coverage": correlation(u_cot, [row["grounding_coverage"] for row in rows]),
        "u_beam_vs_u_cot": correlation([row["u_beam"] for row in rows], u_cot),
        "u_beam_vs_total": correlation([row["u_beam"] for row in rows], [row["total"] for row in rows]),
        "u_cot_vs_total": correlation(u_cot, [row["total"] for row in rows]),
    }
    beam_threshold = 0.5
    cot_threshold = 0.5
    quadrants = {}
    for beam_high, cot_high, name in (
        (True, True, "high_beam_high_cot"),
        (True, False, "high_beam_low_cot"),
        (False, True, "low_beam_high_cot"),
        (False, False, "low_beam_low_cot"),
    ):
        matches = [
            row for row in rows
            if (row["u_beam"] >= beam_threshold) == beam_high and (row["u_cot"] >= cot_threshold) == cot_high
        ]
        quadrants[name] = [{
            "group_id": row["group_id"],
            "step": row["step"],
            "u_beam": row["u_beam"],
            "u_cot": row["u_cot"],
            "composite_reward": row["total"],
        } for row in matches[:5]]
    result["quadrant_examples"] = quadrants
    return result


def runtime_audit(groups: dict[str, dict], gold: dict[str, str], selected: list[str]) -> dict:
    candidates = []
    for group_id in selected:
        parsed = extract_interest_units(gold[group_id], groups[group_id]["prompt"])
        candidates.append((make_cot([unit.raw_text for unit in parsed.units]), gold[group_id], groups[group_id]["prompt"]))
    loops = 10
    started = time.perf_counter()
    for _ in range(loops):
        for candidate, reference, prompt in candidates:
            score_interest_cot(candidate, reference, prompt)
    elapsed = time.perf_counter() - started
    calls = loops * len(candidates)
    per_candidate = elapsed * 1000 / calls if calls else None
    return {"calls": calls, "average_ms_per_candidate": per_candidate, "average_ms_per_g4": per_candidate * 4 if per_candidate else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    args = parser.parse_args()
    groups = load_think_groups(DEFAULT_GRPO)
    gold, _, _ = load_gold(DEFAULT_SOURCE, set(groups))
    variants, selected = audit_variants(groups, gold, args.sample_size)
    history = audit_history(groups, gold, args.trace)
    representative = []
    for beam in (0, 0.5, 2, 8, 16):
        representative.append({"beam_raw": beam, "cot_utility": 1.0 if beam == 0 else 0.0, "composite_reward": composite_reward(beam, 1.0 if beam == 0 else 0.0)})
    report = {
        "experiment": "GR_REC_Think_CompositeInterest_v1",
        "sample_size": len(selected),
        "sample_group_ids": selected,
        "variant_cot_utility": variants,
        "historical_trace": history,
        "representative_tradeoffs": representative,
        "runtime": runtime_audit(groups, gold, selected),
        "gpu_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_size": len(selected), "historical_candidates": history.get("candidate_count"), "runtime": report["runtime"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
