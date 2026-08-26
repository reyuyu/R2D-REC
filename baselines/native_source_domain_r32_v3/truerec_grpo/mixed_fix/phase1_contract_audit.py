"""Generate the auditable CPU-only Mixed-Fix Think Phase-1 contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from think_credit_v1 import (
    DELTA_A, DELTA_B, DELTA_C, G, HIERARCHY_SCALE, plan_think_credit, split_abc,
)
from think_rescue_v1 import uniform_positive_soft_ce


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/phase1_think_credit"


def abc(a: int, b: int = 1, c: int = 1) -> str:
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


def candidate(value: str, gold: list[str]) -> dict[str, Any]:
    parsed = split_abc(value)
    gold_parts = [split_abc(item) for item in gold]
    return {
        "format_valid": True,
        "parsed_abc": value,
        "A_hit": parsed[0] in {item[0] for item in gold_parts},
        "AB_hit": parsed[:2] in {item[:2] for item in gold_parts},
        "exact": parsed in set(gold_parts),
    }


def make(values: list[str], gold: list[str]) -> list[dict[str, Any]]:
    return [candidate(value, gold) for value in values]


def summarize(name: str, values: list[str], gold: list[str]) -> dict[str, Any]:
    plan = plan_think_credit(make(values, gold), gold)
    return {
        "name": name,
        "predictions": values,
        "token_credits": plan.token_credits,
        "a_statistics": plan.a_statistics,
        "a_rescue": {
            "triggered": plan.a_rescue.triggered,
            "reason": plan.a_rescue.reason,
            "target_a_tokens": plan.a_rescue.target_a_tokens,
            "reduction": plan.a_rescue.reduction,
        },
        "bc_hpr": plan.bc_hpr.trigger,
        "bc_hpr_sites": [
            {
                "level": site.level,
                "prefix_tokens": site.prefix_tokens,
                "target_tokens": site.target_tokens,
                "onpolicy_positions": site.onpolicy_positions,
            }
            for site in plan.bc_hpr.sites
        ],
        "monitoring": plan.monitoring,
    }


def build(source_commit: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold3 = [abc(1, 1, 1), abc(2, 2, 2), abc(3, 3, 3)]
    cases = [
        summarize("all_wrong", [abc(9)] * 8, gold3),
        summarize("single_correct_A", [abc(1, 9, 9)] + [abc(9)] * 7, gold3),
        summarize("five_same_A_HPR_B", [abc(1, 9, 9)] * 5 + [abc(9)] * 3, gold3),
        summarize("five_same_A_HPR_C", [abc(1, 1, 9)] * 5 + [abc(9)] * 3, gold3),
        summarize("five_same_A_HPR_NONE", [abc(1)] * 5 + [abc(9)] * 3, gold3),
        summarize(
            "coverage_satisfied_despite_five_same_A",
            [abc(1, 9, 9)] * 5 + [abc(2, 9, 9), abc(3, 9, 9), abc(9)],
            gold3,
        ),
    ]
    logits = torch.tensor([2.0, 0.0, -1.0, 1.0], dtype=torch.float64)
    soft_ce = uniform_positive_soft_ce(logits, [0, 2]).item()
    log_probs = torch.log_softmax(logits, dim=-1)
    log_mass = (-torch.logsumexp(log_probs[[0, 2]], dim=0)).item()
    single = cases[1]["token_credits"][0][0]
    repeated = cases[2]["token_credits"][0][0]
    invariants = {
        "all_semantic_credits_nonnegative": all(
            value >= 0 for case in cases for row in case["token_credits"] for value in row
        ),
        "single_A_credit_greater_than_five_repeat_credit": single > repeated,
        "satisfied_coverage_never_rescued": not cases[-1]["a_rescue"]["triggered"],
        "hpr_a_absent": all(case["bc_hpr"] != "HPR_A" for case in cases),
        "soft_ce_differs_from_log_mass": abs(soft_ce - log_mass) > 1e-8,
    }
    contract = {
        "source_main_commit": source_commit,
        "G": G,
        "hierarchy_scale": HIERARCHY_SCALE,
        "deltas": {"A": DELTA_A, "B": DELTA_B, "C": DELTA_C},
        "think_semantic_negative_credit": "DISABLED",
        "A_reward": "MODE_FREQUENCY_AWARE_FRONTIER",
        "B_reward": "POSITIVE_FRONTIER",
        "C_reward": "POSITIVE_FRONTIER",
        "A_rescue": {
            "no_gold_A": "ENABLED",
            "mode_collapse": "ENABLED",
            "coverage_target": "min(K_A,3)",
            "collapse_threshold": "5_OF_8",
            "loss": "UNIFORM_POSITIVE_SOFT_CE",
            "reduction": "one_group_level_A_distribution",
        },
        "HPR": {
            "A": "DISABLED_FOR_THINK",
            "B": "REUSED_FROM_FROZEN_hpr_plan_v1",
            "C": "REUSED_FROM_FROZEN_hpr_plan_v1",
        },
        "numeric_examples": {
            "single_correct_A_credit": single,
            "five_repeat_correct_A_credit": repeated,
            "uniform_positive_soft_ce_2_targets": soft_ce,
            "multi_positive_log_mass_same_targets": log_mass,
        },
        "invariants": invariants,
        "model_loaded": False,
        "gpu_started": False,
        "generation_started": False,
        "backward_started": False,
        "optimizer_steps": 0,
        "phase1_contract": "PASS" if all(invariants.values()) else "FAIL",
    }
    return contract, cases


def review(contract: dict[str, Any]) -> str:
    fields = {
        "SOURCE_MAIN_COMMIT": contract["source_main_commit"],
        "G": contract["G"],
        "DELTA_A": contract["deltas"]["A"],
        "DELTA_B": contract["deltas"]["B"],
        "DELTA_C": contract["deltas"]["C"],
        "THINK_SEMANTIC_NEGATIVE_CREDIT": "DISABLED",
        "A_REWARD": "MODE_FREQUENCY_AWARE_FRONTIER",
        "B_REWARD": "POSITIVE_FRONTIER",
        "C_REWARD": "POSITIVE_FRONTIER",
        "A_RESCUE_NO_GOLD_A": "ENABLED",
        "A_RESCUE_MODE_COLLAPSE": "ENABLED",
        "A_COVERAGE_TARGET": "min(K_A,3)",
        "A_COLLAPSE_THRESHOLD": "5_OF_8",
        "A_RESCUE_LOSS": "UNIFORM_POSITIVE_SOFT_CE",
        "HPR_A": "DISABLED_FOR_THINK",
        "HPR_B": "REUSED",
        "HPR_C": "REUSED",
        "MODEL_LOADED": "NO", "GPU_STARTED": "NO", "GENERATION_STARTED": "NO",
        "BACKWARD_STARTED": "NO", "OPTIMIZER_STEPS": 0,
        "PHASE1_CONTRACT": contract["phase1_contract"],
    }
    return "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract, cases = build(args.source_commit)
    write_json(args.output_dir / "contract.json", contract)
    write_json(args.output_dir / "deterministic_cases.json", cases)
    (args.output_dir / "REVIEW.txt").write_text(review(contract), encoding="utf-8")
    print(review(contract), end="")
    if contract["phase1_contract"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
