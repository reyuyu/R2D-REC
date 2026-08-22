"""Generate a CPU-only Composite Interest monitor demo run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ablations.gr_rec_think_composite_interest_v1.interest_metric import beam_utility


STEPS = (0, 200, 400, 600, 716)
DOMAINS = ("video", "prod", "ad", "living")


def advantages(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return [(value - mean) / (std + 1e-4) for value in values]


def candidate(group_id, index, beam, cot, matched, gold_n=2, parser=True, q=0.0):
    beam_value = beam_utility(beam)
    composite = 0.60 * beam_value + 0.40 * cot
    raw_n = index if index < 3 else 2
    grounded_n = min(raw_n, matched)
    pred = [
        {"index": item + 1, "normalized_text": f"候选兴趣 {item + 1}",
         "grounded_evidence_sids": [f"sid-{item + 1}"] if item < grounded_n else []}
        for item in range(raw_n)
    ]
    details = [
        {"pred_index": item + 1, "gold_index": item + 1, "text_similarity": 0.8,
         "evidence_similarity": 1.0, "combined_similarity": 0.86, "matched": True}
        for item in range(min(matched, raw_n, gold_n))
    ]
    return {
        "group_id": group_id, "candidate_id": index,
        "completion": f"demo candidate {index}", "completion_length": 48 + index,
        "beam_raw": beam, "beam_utility": beam_value,
        "beam_contribution": 0.60 * beam_value,
        "gold_interest_count": gold_n, "pred_interest_count": raw_n,
        "matched_interest_count": matched,
        "interest_coverage": matched / gold_n,
        "interest_precision": matched / raw_n if raw_n else 0.0,
        "mean_match_similarity": 0.86 if matched else 0.0,
        "coverage_tier": matched / gold_n, "match_quality": q, "Q": q,
        "cot_utility": cot, "cot_contribution": 0.40 * cot,
        "composite_reward": composite,
        "raw_n": raw_n, "grounded_n": grounded_n,
        "grounding_coverage": grounded_n / raw_n if raw_n else None,
        "parser_success": parser,
        "parser_failure_reason": None if parser else "demo_parser_failure",
        "pred_interest_units": pred, "match_details": details,
        "unmatched_pred_indices": list(range(matched + 1, raw_n + 1)),
        "unmatched_gold_indices": list(range(matched + 1, gold_n + 1)),
        "unmatched_pred_best_alternatives": [],
        "unmatched_gold_best_alternatives": [], "demo": True,
    }


def group(case, beam, cot, matches, parser_failure=False, q_active=False):
    group_id = f"demo-{case}"
    candidates = [
        candidate(group_id, index, beam[index], cot[index], matches[index],
                  parser=not (parser_failure and index == 0),
                  q=0.25 if q_active and index == 3 else 0.0)
        for index in range(4)
    ]
    rewards = [row["composite_reward"] for row in candidates]
    final = advantages(rewards)
    for row, value in zip(candidates, final):
        row["final_sequence_advantage"] = value
    beam_top = [index for index, value in enumerate(beam) if value == max(beam)]
    winner = max(range(4), key=rewards.__getitem__)
    std = math.sqrt(sum((value - sum(rewards) / 4) ** 2 for value in rewards) / 4)
    metadata = {
        "group_id": group_id,
        "gold_interest_units": [
            {"index": 1, "normalized_text": "汽车生活", "grounded_evidence_sids": ["sid-1"]},
            {"index": 2, "normalized_text": "维修配件", "grounded_evidence_sids": ["sid-2"]},
        ],
        "beam_raw_vector": beam,
        "beam_utility_vector": [row["beam_utility"] for row in candidates],
        "cot_utility_vector": cot,
        "Q_vector": [row["Q"] for row in candidates],
        "composite_reward_vector": rewards,
        "final_advantage_vector": final,
        "beam_all_equal": len(set(beam)) == 1,
        "cot_all_equal": len(set(cot)) == 1,
        "composite_all_equal": len(set(rewards)) == 1,
        "beam_active": len(set(beam)) > 1, "cot_active": len(set(cot)) > 1,
        "composite_active": std > 0, "composite_zero_std": std == 0,
        "composite_reward_mean": sum(rewards) / 4,
        "composite_reward_population_std": std,
        "matched_candidate_count": sum(row["matched_interest_count"] > 0 for row in candidates),
        "quality_active_candidate_count": sum(row["Q"] > 0 for row in candidates),
        "beam_top_set": beam_top, "beam_stable_winner": beam_top[0],
        "composite_winner": winner,
        "top_set_tie_break": winner in beam_top and winner != beam_top[0],
        "strict_beam_reversal": winner not in beam_top,
        "pairwise_interest_similarity": 0.42, "unique_interest_set_count": 4,
        "demo": True,
    }
    return metadata, candidates


def generate(root: Path) -> Path:
    run = root / "GR-REC-THINK-COMPOSITE-DEMO"
    run.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run.name, "experiment": "GR_REC_Think_CompositeInterest_v1",
        "demo": True, "max_steps": 716, "probe_steps": list(STEPS),
        "fixed_probe_group_ids": [f"probe-{domain}-r{round_id}" for round_id in range(1, 4) for domain in DOMAINS],
    }
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    cases = [
        group("rescued", [0, 0, 0, 0], [0.0, 0.3, 0.6, 1.0], [0, 1, 2, 2], parser_failure=True, q_active=True),
        group("zero", [0, 0, 0, 0], [0.5] * 4, [1] * 4),
        group("tie", [8, 8, 0, 0], [0.2, 0.8, 0.1, 0.1], [1, 2, 0, 0]),
        group("reversal", [4, 0, 0, 0], [0.0, 0.1, 1.0, 0.1], [0, 1, 2, 1]),
    ]
    events = [
        {"step": index * 200, "rollout_id": index, "route": "think", "g": 4,
         "formula": {"beam_weight": 0.60, "cot_weight": 0.40},
         "groups": [metadata], "candidates": candidates,
         "advantages": metadata["final_advantage_vector"], "demo": True}
        for index, (metadata, candidates) in enumerate(cases)
    ]
    (run / "composite_interest.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")
    probes = []
    template_group, template_candidates = cases[0]
    for step in STEPS:
        for round_id in range(3):
            for domain in DOMAINS:
                probes.append({
                    "step": step, "reason": "baseline" if step == 0 else "final" if step == 716 else "milestone",
                    "probe_round": round_id, "group_id": f"probe-{domain}-r{round_id + 1}",
                    "target_domain": domain, "probe_route": "think_only", "demo": True,
                    "gold_interest_units": template_group["gold_interest_units"],
                    "think": {"beam_reward_mean": 0.0, "composite_reward_mean": template_group["composite_reward_mean"],
                              "composite_reward_std": template_group["composite_reward_population_std"],
                              "candidates": template_candidates},
                })
    (run / "probes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in probes), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": step, "route": "think", "reward": 0.0}) + "\n" for step in STEPS), encoding="utf-8")
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    print(generate(parser.parse_args().output))
