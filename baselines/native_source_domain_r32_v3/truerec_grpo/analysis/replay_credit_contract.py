"""TrueRec Phase 0.8 CPU-only Frontier/HPR replay over frozen Phase 0.7 raw."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterable


RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
TRUE_REC = RUNTIME / "truerec_grpo"
PILOT = TRUE_REC / "data/pilot4096/pilot4096_records.jsonl"
RAW_ROLLOUT = TRUE_REC / "results/phase0_7/raw/pilot4096_g8_candidates.jsonl"
OUTPUT = TRUE_REC / "results/phase0_8"
RUNTIME_PLAN = OUTPUT / "raw/hpr_plan_v1.jsonl"
PILOT_SHA = "ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879"
RAW_SHA = "39d1a4862c7239b07a6576aae24ea3e0d2aa55bebc1cd9c01d51ad6c46f6001d"
GROUPS = 4096
G = 8
EXPECTED_HPR = {"HPR_A": 2651, "HPR_B": 1012, "HPR_C": 219, "HPR_NONE": 214}
RELATIVE_FILES = (
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/credit/frontier_credit_v1.py"),
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/credit/hpr_plan_v1.py"),
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/analysis/replay_credit_contract.py"),
)

sys.path.insert(0, str(TRUE_REC / "credit"))
from frontier_credit_v1 import GATED, LEVELS, NEGATIVE, POSITIVE, group_credit_totals, plan_frontier_credit  # noqa: E402
from hpr_plan_v1 import plan_hpr, validate_plan  # noqa: E402


class ReplayError(RuntimeError):
    pass


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_raw_gate(path: Path, expected_sha: str, expected_candidates: int, expected_groups: int) -> list[dict[str, Any]]:
    actual = file_sha(path)
    if actual != expected_sha:
        raise ReplayError(f"RAW_SHA_FAIL={actual}")
    rows = read_jsonl(path)
    if len(rows) != expected_candidates:
        raise ReplayError(f"RAW_CANDIDATE_COUNT_FAIL={len(rows)}")
    validate_raw_records(rows, expected_groups)
    return rows


def validate_raw_records(rows: list[dict[str, Any]], expected_groups: int) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(int(row["sample_index"]))
    if len(grouped) != expected_groups:
        raise ReplayError(f"RAW_GROUP_COUNT_FAIL={len(grouped)}")
    bad = [group_id for group_id, indices in grouped.items() if sorted(indices) != list(range(G))]
    if bad:
        raise ReplayError(f"RAW_G8_SAMPLE_INDEX_FAIL={bad[:5]}")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True).strip()


def code_audit() -> dict[str, Any]:
    head, origin, status = git("rev-parse", "HEAD"), git("rev-parse", "origin/main"), git("status", "--short")
    files = {}
    for relative in RELATIVE_FILES:
        source = SOURCE_REPO / relative
        runtime = TRUE_REC / Path(*relative.parts[3:])
        blob = subprocess.check_output(["git", "-C", str(SOURCE_REPO), "show", f"{head}:{relative.as_posix()}"])
        values = {"github": hashlib.sha256(blob).hexdigest(), "source": file_sha(source), "runtime": file_sha(runtime)}
        if len(set(values.values())) != 1:
            raise ReplayError(f"CODE_SHA_PARITY_FAIL={relative}:{values}")
        files[relative.as_posix()] = values
    if head != origin or status:
        raise ReplayError(f"GIT_PROVENANCE_FAIL={head},{origin},{status!r}")
    return {"implement_commit": head, "push_status": "PASS", "git_status_short": "EMPTY", "github_runtime_parity": "PASS", "files": files}


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"N": 0, "mean": None, "abs_mean": None, "median": None, "min": None, "max": None}
    return {"N": len(values), "mean": round(statistics.fmean(values), 10), "abs_mean": round(statistics.fmean(abs(value) for value in values), 10), "median": statistics.median(values), "min": min(values), "max": max(values)}


def _level_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"candidates": len(rows), "format_invalid": sum(row["format_invalid"] for row in rows), "levels": {}}
    for index, level in enumerate(LEVELS):
        kinds = Counter(row["kinds"][index] for row in rows)
        values = [row["credits"][index] for row in rows]
        output["levels"][level] = {"positive_token_count": kinds[POSITIVE], "negative_token_count": kinds[NEGATIVE], "gated_token_count": kinds[GATED], "credit": _stats(values)}
    return output


def _group_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"groups": len(rows), "signed_credit": _stats([row["signed_credit"] for row in rows]), "absolute_credit": _stats([row["absolute_credit"] for row in rows])}


def breakdown(rows: list[dict[str, Any]], block: Callable[[list[dict[str, Any]]], dict[str, Any]]) -> dict[str, Any]:
    dimensions = {"overall": lambda row: "overall", "domain": lambda row: row["domain"], "novelty": lambda row: row["novelty"], "K_bucket": lambda row: row["K_bucket"]}
    result = {}
    for name, key_fn in dimensions.items():
        result[name] = {}
        for key in sorted({key_fn(row) for row in rows}):
            result[name][key] = block([row for row in rows if key_fn(row) == key])
    return result


def target_cardinality(sites: list[dict[str, Any]], level: str) -> dict[str, Any]:
    values = [len(site["target_tokens"]) for site in sites if site["level"] == level]
    return {"sites": len(values), "mean": round(statistics.fmean(values), 8) if values else None, "median": statistics.median(values) if values else None, "max": max(values) if values else None}


def set_intersection_counts(zero_std: set[str], hpr_a: set[str]) -> dict[str, int]:
    return {"ZERO_STD_AND_HPR_A": len(zero_std & hpr_a), "ZERO_STD_NOT_HPR_A": len(zero_std - hpr_a), "HPR_A_NOT_ZERO_STD": len(hpr_a - zero_std)}


def wrong_history_frontier(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["frontier"] for row in rows if row["wrong_history_copy"])
    return {name: counts[name] for name in ("A_FAIL", "B_FAIL", "C_FAIL", "EXACT")}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def replay(test_status: str) -> None:
    if test_status != "PASS" or "torch" in sys.modules:
        raise ReplayError("CPU_OR_TEST_GATE_FAIL")
    audit = code_audit()
    if file_sha(PILOT) != PILOT_SHA:
        raise ReplayError("PILOT_SHA_FAIL")
    pilot = read_jsonl(PILOT)
    if len(pilot) != GROUPS or len({row["recommendation_group_id"] for row in pilot}) != GROUPS:
        raise ReplayError("PILOT_GROUP_GATE_FAIL")
    raw = load_raw_gate(RAW_ROLLOUT, RAW_SHA, GROUPS * G, GROUPS)
    pilot_by_id = {row["recommendation_group_id"]: row for row in pilot}
    raw_by_id = defaultdict(list)
    for row in raw:
        raw_by_id[row["group_id"]].append(row)
    candidate_plans, group_plans, hpr_plans, all_sites = [], [], [], []
    zero_std, hpr_a = set(), set()
    RUNTIME_PLAN.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_PLAN.open("w", encoding="utf-8") as plan_handle:
        for group_id in sorted(pilot_by_id):
            record = pilot_by_id[group_id]
            candidates = sorted(raw_by_id[group_id], key=lambda row: row["sample_index"])
            if len(candidates) != G or any(row["domain"] != record["target_domain"] for row in candidates):
                raise ReplayError(f"RAW_PILOT_JOIN_FAIL={group_id}")
            frontier = plan_frontier_credit(candidates)
            hpr = plan_hpr(candidates, record["all_gold_abc"])
            validate_plan(hpr)
            signed, absolute = group_credit_totals(frontier)
            metadata = {"group_id": group_id, "domain": record["target_domain"], "novelty": record["novelty"], "K_bucket": record["K_bucket"]}
            for raw_row, credit in zip(candidates, frontier.candidates):
                candidate_plans.append({**metadata, "sample_index": raw_row["sample_index"], "frontier": raw_row["frontier"], "wrong_history_copy": raw_row["wrong_history_copy"], "correct_history_copy": raw_row["correct_history_copy"], "credits": credit.credits, "kinds": credit.kinds, "format_invalid": credit.format_invalid, "format_credit_total": credit.format_credit_total})
            group_plans.append({**metadata, "signed_credit": signed, "absolute_credit": absolute, "positive_frontier_present": any(POSITIVE in item.kinds for item in frontier.candidates), "only_negative_frontier": not any(POSITIVE in item.kinds for item in frontier.candidates), "frontier_active": any(any(value != 0 for value in item.credits) for item in frontier.candidates), "exact_reached": any(row["exact"] for row in candidates), "hpr_trigger": hpr.trigger})
            hpr_payload = {**metadata, **hpr.to_dict()}
            plan_handle.write(json.dumps(hpr_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            hpr_plans.append(hpr_payload)
            for site in hpr_payload["sites"]:
                all_sites.append({**metadata, **site})
            legacy_values = [row["legacy_scalar"] for row in candidates]
            if len(set(legacy_values)) == 1:
                zero_std.add(group_id)
            if hpr.trigger == "HPR_A":
                hpr_a.add(group_id)
    triggers = Counter(row["trigger"] for row in hpr_plans)
    if dict(triggers) != EXPECTED_HPR:
        raise ReplayError(f"HPR_TRIGGER_PARITY_FAIL={triggers}")
    missing_positions = sum(not site["onpolicy_positions"] for site in all_sites)
    if missing_positions:
        raise ReplayError(f"HPR_SITE_WITHOUT_ONPOLICY_POSITION={missing_positions}")
    candidate_signed = sum(sum(row["credits"]) for row in candidate_plans)
    group_signed = sum(row["signed_credit"] for row in group_plans)
    candidate_absolute = sum(sum(abs(value) for value in row["credits"]) for row in candidate_plans)
    group_absolute = sum(row["absolute_credit"] for row in group_plans)
    if abs(candidate_signed - group_signed) > 1e-9 or abs(candidate_absolute - group_absolute) > 1e-9:
        raise ReplayError("CREDIT_CONSERVATION_FAIL")
    frontier_audit = {"contract": {"levels": list(LEVELS), "domain_credit": False, "deltas": {"A": 0.5, "B": 1.5, "C": 6.0}, "hierarchy_scale": 8.0, "format_invalid_separate_total": -0.09375}, "candidate_credit": breakdown(candidate_plans, _level_block), "group_credit": breakdown(group_plans, _group_block), "conservation": {"candidate_signed_sum": candidate_signed, "group_signed_sum": group_signed, "candidate_absolute_sum": candidate_absolute, "group_absolute_sum": group_absolute, "status": "PASS"}}
    hpr_audit = {"contract": {"objective": "multi-positive next-token log-sum-probability", "site_reduction": "mean_within_group", "extra_hpr_model_forward": False}, "trigger_counts": {name: triggers[name] for name in EXPECTED_HPR}, "site_counts": {level: sum(site["level"] == level for site in all_sites) for level in LEVELS}, "site_without_onpolicy_position": missing_positions, "target_cardinality": {level: target_cardinality(all_sites, level) for level in LEVELS}, "runtime_plan": {"path": str(RUNTIME_PLAN), "sha256": file_sha(RUNTIME_PLAN), "groups": len(hpr_plans), "pushed": False}}
    signal = {"legacy_zero_signal_groups": len(zero_std), "frontier_active_groups": sum(row["frontier_active"] for row in group_plans), "hpr_trigger_groups": sum(row["hpr_trigger"] != "HPR_NONE" for row in group_plans), "only_negative_frontier_groups": sum(row["only_negative_frontier"] for row in group_plans), "positive_frontier_present_groups": sum(row["positive_frontier_present"] for row in group_plans), "hpr_rescue_needed_groups": sum(row["hpr_trigger"] != "HPR_NONE" for row in group_plans), "exact_reached_groups": sum(row["exact_reached"] for row in group_plans), "interpretation_limit": "offline signal coverage only; no claim that gradients are effective"}
    intersection = set_intersection_counts(zero_std, hpr_a)
    intersection.update({"legacy_zero_std_groups": len(zero_std), "hpr_a_groups": len(hpr_a), "sets_computed_independently": True})
    history = {"wrong_history_by_frontier": wrong_history_frontier(candidate_plans), "wrong_history_candidates": sum(row["wrong_history_copy"] for row in candidate_plans), "correct_history_candidates": sum(row["correct_history_copy"] for row in candidate_plans), "history_penalty_added": False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "frontier_credit_audit.json", frontier_audit)
    write_json(OUTPUT / "hpr_plan_audit.json", hpr_audit)
    write_json(OUTPUT / "signal_coverage_audit.json", signal)
    write_json(OUTPUT / "zero_std_hpr_intersection.json", intersection)
    write_json(OUTPUT / "history_frontier_intersection.json", history)
    review = render_review(audit, frontier_audit, hpr_audit, signal, intersection, history)
    (OUTPUT / "CHATGPT_PHASE0_8_REVIEW.txt").write_text(review, encoding="utf-8")
    print(review, end="")


def render_review(audit, frontier, hpr, signal, intersection, history) -> str:
    levels = frontier["candidate_credit"]["overall"]["overall"]["levels"]
    sites = hpr["site_counts"]; triggers = hpr["trigger_counts"]; wrong = history["wrong_history_by_frontier"]
    values = {
        "IMPLEMENT_COMMIT": audit["implement_commit"], "RESULT_COMMIT": "PENDING_REPORT_COMMIT", "PUSH_STATUS": audit["push_status"], "GIT_STATUS_SHORT": audit["git_status_short"], "GITHUB_RUNTIME_PARITY": audit["github_runtime_parity"],
        "GROUPS": GROUPS, "CANDIDATES": GROUPS * G, "HPR_A": triggers["HPR_A"], "HPR_B": triggers["HPR_B"], "HPR_C": triggers["HPR_C"], "HPR_NONE": triggers["HPR_NONE"], "HPR_A_SITES": sites["A"], "HPR_B_SITES": sites["B"], "HPR_C_SITES": sites["C"], "HPR_SITE_WITHOUT_ONPOLICY_POSITION": hpr["site_without_onpolicy_position"],
        "A_POSITIVE_TOKEN_COUNT": levels["A"]["positive_token_count"], "A_NEGATIVE_TOKEN_COUNT": levels["A"]["negative_token_count"], "B_POSITIVE_TOKEN_COUNT": levels["B"]["positive_token_count"], "B_NEGATIVE_TOKEN_COUNT": levels["B"]["negative_token_count"], "C_POSITIVE_TOKEN_COUNT": levels["C"]["positive_token_count"], "C_NEGATIVE_TOKEN_COUNT": levels["C"]["negative_token_count"],
        "LEGACY_ZERO_STD_GROUPS": signal["legacy_zero_signal_groups"], "FRONTIER_ACTIVE_GROUPS": signal["frontier_active_groups"], "HPR_TRIGGER_GROUPS": signal["hpr_trigger_groups"], **intersection,
        "WRONG_HISTORY_A_FAIL": wrong["A_FAIL"], "WRONG_HISTORY_B_FAIL": wrong["B_FAIL"], "WRONG_HISTORY_C_FAIL": wrong["C_FAIL"], "WRONG_HISTORY_EXACT": wrong["EXACT"], "EXTRA_HPR_MODEL_FORWARD": "NO", "TEST_STATUS": "PASS", "GPU_INFERENCE_STARTED": "NO", "MODEL_FORWARD_STARTED": "NO", "TRAINING_STARTED": "NO", "OPTIMIZER_STEPS": 0, "NEXT_EXPERIMENT_STARTED": "NO",
    }
    return "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-status", required=True, choices=("PASS", "FAIL"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ReplayError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
    replay(args.test_status)


if __name__ == "__main__":
    main()
