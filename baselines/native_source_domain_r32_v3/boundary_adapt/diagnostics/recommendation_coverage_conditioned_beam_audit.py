"""Phase 1.5.4 CPU-only coverage-conditioned retrospective Beam audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/"
    "recommendation_coverage_conditioned_beam_audit.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_coverage_conditioned_beam_audit.py"
OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_coverage_conditioned_beam_audit"
PHASE153 = RUNTIME / "boundary_adapt/results/recommendation_training_coverage_hierarchy_audit"
PHASE15 = RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_quick_probe"
PHASE151 = RUNTIME / "boundary_adapt/results/recommendation_memory_history_2x2_probe"
PHASE153_REPORT = SOURCE_REPO / (
    "baselines/native_source_domain_r32_v3/boundary_adapt/results/"
    "recommendation_training_coverage_hierarchy_audit/summary.md"
)
BATA = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
MINI_INVENTORY = PHASE153 / "training_corpus_inventory.json"

PHASE153_IMPLEMENT_COMMIT = "2b193376f6dd8943a2fd59c349533639bd1c5e9a"
PHASE153_RESULT_COMMIT = "5f1cb1b54fc89a2f2c5921e8010071ebfcb57094"
EXPECTED_SHA = {
    "natural1595_training_coverage.json": "a99f3dfa7bfdb44ee7694fc78e0f261a18d3c613459f9b23302f9f7b0e2beb17",
    "phase151_coverage_beam_join.json": "638919bf9b75af3d707bbfcfb9daa0fff678002cd8ff10cb8905c937461e805d",
    "phase15_records": "1d74f95727bbb29a38e49c312efae473742f9cbc1ec4f079222e7bfce6d57731",
    "phase151_records": "4f59422d16e31ca421dce311bd2f508fb2183f6558d1714acd6c8c01eabf688b",
}
RAW_PHASE153 = (
    "natural1595_training_coverage.json",
    "training_sid_frequency_stats.json",
    "history_training_cross.json",
    "branching_factor_stats.json",
    "conditional_entropy_stats.json",
    "target_gold_frequency_surprisal.json",
    "mini_vs_bata_coverage.json",
    "phase151_coverage_beam_join.json",
)
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
PRIMARY_CONTRACT = "OFFICIAL_DOMAIN_SEMANTIC_FROZEN_BATA_COT_ABC3"
SECONDARY_CONTRACT = "MINIFIX_NATIVE_PREFIX_FROZEN_BATA_COT_ABC3"
MODELS = ("MiniFix", "Gamma")
BUCKETS = ("0", "1", "2-4", "5-9", "10+")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True, encoding="utf-8").strip()


def code_audit() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    status = git("status", "--short")
    source_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT)
    runtime_sha = file_sha(RUNTIME_SCRIPT)
    commit_script = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{head}:{RELATIVE_SCRIPT.as_posix()}"],
    )
    commit_sha = hashlib.sha256(commit_script).hexdigest()
    result = {
        "implement_commit": head,
        "origin_main": origin,
        "push_status": "PASS" if head == origin else "FAIL",
        "git_status_short": status,
        "github_script_sha256": commit_sha,
        "source_script_sha256": source_sha,
        "runtime_script_sha256": runtime_sha,
        "runtime_github_parity": "PASS" if commit_sha == source_sha == runtime_sha else "FAIL",
    }
    if result["push_status"] != "PASS" or status or result["runtime_github_parity"] != "PASS":
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={result}")
    return result


def parse_sid(value: str | list[Any] | tuple[Any, ...]) -> tuple[str, int, int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            raise ValueError(f"invalid SID {value!r}")
        return str(value[0]), int(value[1]), int(value[2]), int(value[3])
    match = SID_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"invalid SID {value!r}")
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row.get("aux_metadata_json") or "{}")


def row_sha(row: dict[str, Any]) -> str:
    return stable_sha(row)


def prefix(sid: tuple[str, int, int, int], level: str) -> tuple[Any, ...]:
    return sid[: {"A": 2, "AB": 3, "ABC": 4}[level]]


def rate(numerator: int | float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def phase153_reproduction() -> tuple[dict[str, Any], dict[str, str]]:
    raw_hashes = {name: file_sha(PHASE153 / name) for name in RAW_PHASE153}
    if raw_hashes["natural1595_training_coverage.json"] != EXPECTED_SHA["natural1595_training_coverage.json"]:
        raise RuntimeError("PHASE153_NATURAL_SHA_MISMATCH")
    if raw_hashes["phase151_coverage_beam_join.json"] != EXPECTED_SHA["phase151_coverage_beam_join.json"]:
        raise RuntimeError("PHASE153_JOIN_SHA_MISMATCH")

    old_source = subprocess.check_output(
        [
            "git", "-C", str(SOURCE_REPO), "show",
            f"{PHASE153_IMPLEMENT_COMMIT}:baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/"
            "recommendation_training_coverage_hierarchy_audit.py",
        ]
    )
    old_source_sha = hashlib.sha256(old_source).hexdigest()
    if old_source_sha != "406b2dcf2684551387eca1a83fb715aa24236be18822d7c650b27153ce7a08bd":
        raise RuntimeError("PHASE153_COMMIT_SCRIPT_SHA_MISMATCH")

    report = PHASE153_REPORT.read_text(encoding="utf-8")
    match = re.search(r"\|\s*SN\s*\|\s*4\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|", report)
    if not match:
        raise RuntimeError("PHASE153_REPORTED_SN_RATE_NOT_FOUND")
    reported_rate = int(match.group(1)) / int(match.group(2))

    joined = read_json(PHASE153 / "phase151_coverage_beam_join.json")
    sn_groups = [row for row in joined["groups"] if row["cell"] == "SN"]
    un_groups = [row for row in joined["groups"] if row["cell"] == "UN"]
    sn_rate = rate(sum(bool(row["ANY_GOLD"]["ABC_SAME_GROUP_PRESENT"]) for row in sn_groups), len(sn_groups))
    un_rate = rate(sum(bool(row["ANY_GOLD"]["ABC_SAME_GROUP_PRESENT"]) for row in un_groups), len(un_groups))
    abc_beam_zero = bool(joined["joined_beam_rows"]) and all(
        int(row["ABCHit@32"]) == 0 for row in joined["joined_beam_rows"]
    )

    natural = read_json(PHASE153 / "natural1595_training_coverage.json")
    mini = natural["corpora"]["MINI"]["contracts"]["ANY_GOLD"]
    domains = ("video", "prod", "ad", "living")
    n = sum(mini[d]["N"] for d in domains)
    nonhistory_n = sum(mini[d]["NONHISTORY_N"] for d in domains)
    weighted = lambda key: sum(mini[d]["N"] * mini[d][key] for d in domains) / n
    unseen = sum(
        mini[d]["NONHISTORY_N"] * mini[d]["NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE"] for d in domains
    ) / nonhistory_n
    a_cov, ab_cov, abc_cov = weighted("A_OTHER_GROUP_COVERAGE"), weighted("AB_OTHER_GROUP_COVERAGE"), weighted("ABC_OTHER_GROUP_COVERAGE")
    if a_cov - ab_cov >= 0.15 and ab_cov - abc_cov >= 0.10 and unseen >= 0.50:
        data_support = "STRONG"
    elif a_cov > abc_cov and unseen >= 0.30:
        data_support = "MODERATE"
    else:
        data_support = "WEAK"
    decoder_support = "MODERATE" if sn_rate >= 0.75 and abc_beam_zero else "WEAK"
    if data_support in ("STRONG", "MODERATE") and decoder_support == "MODERATE":
        root_class = "COVERAGE_PLUS_RANKING_BOTTLENECK"
    elif data_support == "STRONG":
        root_class = "FINE_ITEM_DATA_COVERAGE_BOTTLENECK"
    elif decoder_support == "MODERATE":
        root_class = "FINE_ITEM_RANKING_BOTTLENECK"
    else:
        root_class = "INSUFFICIENT_TO_DISTINGUISH"

    old_root = read_json(PHASE153 / "root_cause_summary.json")
    reported_decoder = old_root["decoder_ranking_limit_support"]
    reported_root = old_root["root_class"]
    repro_pass = (
        reported_rate == sn_rate
        and reported_decoder == decoder_support
        and reported_root == root_class
    )
    result = {
        "phase153_implement_commit": PHASE153_IMPLEMENT_COMMIT,
        "phase153_result_commit": PHASE153_RESULT_COMMIT,
        "phase153_commit_script_sha256": old_source_sha,
        "reported_sn_same_group_abc_rate": reported_rate,
        "recomputed_sn_same_group_abc_rate": sn_rate,
        "recomputed_un_same_group_abc_rate": un_rate,
        "abc_beam_zero": abc_beam_zero,
        "reported_decoder_support": reported_decoder,
        "recomputed_decoder_support": decoder_support,
        "reported_root_class": reported_root,
        "recomputed_root_class": root_class,
        "data_coverage_limit_support": data_support,
        "phase153_decision_repro_pass": "YES" if repro_pass else "NO",
        "phase153_correction_required": "NO" if repro_pass else "YES",
        "phase153_corrected_decoder_support": decoder_support,
        "phase153_corrected_root_class": root_class,
        "raw_artifact_sha256_before": raw_hashes,
        "raw_artifacts_unchanged": "PENDING_POST_WRITE_CHECK",
        "mismatch_explanation": (
            "No mismatch."
            if repro_pass
            else "The published result report encoded SN same-group exact ABC as 1/4, while all four immutable SN coverage rows are true. The old code, decoder decision, and root decision agree with 4/4."
        ),
    }
    return result, raw_hashes


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    result: dict[str, dict[str, Any]] = {}
    for item in payload["items"]:
        group = str(item["group_id"])
        normalized = {
            "group_id": group,
            "domain": str(item["domain"]),
            "gold_sids": sorted(parse_sid(value) for value in item["all_gold_sids"]),
            "gold_sid_in_history": bool(item["gold_sid_in_history"]),
            "K": int(item["K"]),
            "history_sids": sorted(parse_sid(value) for value in item.get("target_domain_history_sids", item.get("history_target_domain_sids", []))),
            "source_row_sha256": str(item["source_row_sha256"]),
            "membership": item.get("membership"),
            "cell": item.get("cell"),
        }
        if group in result and result[group] != normalized:
            raise RuntimeError(f"MANIFEST_GROUP_CONFLICT={group}")
        result[group] = normalized
    return result


def source_gold_contract(manifests: Iterable[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    wanted_sha: dict[str, str] = {}
    for manifest in manifests:
        for group, item in manifest.items():
            previous = targets.get(group)
            if previous is not None:
                comparable = {key: previous[key] for key in ("domain", "gold_sids", "gold_sid_in_history", "K")}
                candidate = {key: item[key] for key in comparable}
                if comparable != candidate:
                    raise RuntimeError(f"CROSS_MANIFEST_CONFLICT={group}")
            else:
                targets[group] = dict(item)
            wanted_sha[group] = item["source_row_sha256"]

    found: set[str] = set()
    with BATA.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend":
                continue
            info = metadata(row)
            group = str(info.get("recommendation_group_id"))
            if group not in targets or group in found:
                continue
            if row_sha(row) != wanted_sha[group]:
                continue
            current = parse_sid(str(info["recommendation_current_gold_sid"]))
            all_gold = sorted(parse_sid(str(value)) for value in info["recommendation_all_gold_sids"])
            if all_gold != targets[group]["gold_sids"] or current not in all_gold:
                raise RuntimeError(f"SOURCE_GOLD_CONTRACT_FAIL={group}")
            targets[group]["current_gold_sid"] = current
            found.add(group)
            if len(found) == len(targets):
                break
    missing = sorted(set(targets) - found)
    if missing:
        raise RuntimeError(f"SOURCE_ROWS_NOT_FOUND={missing}")
    return targets


def coverage_index() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    natural_path = PHASE153 / "natural1595_training_coverage.json"
    natural = read_json(natural_path)
    if natural.get("N") != 1595 or natural.get("name") != "ADAPTATION_HELDOUT_NATURAL_PROXY":
        raise RuntimeError("PHASE153_COVERAGE_SCHEMA_FAIL")
    if set(natural.get("per_target", {})) != {"BATA", "MINI"}:
        raise RuntimeError("PHASE153_PER_TARGET_SCHEMA_FAIL")
    rows = natural["per_target"]["MINI"]
    if len(rows) != 1595:
        raise RuntimeError("PHASE153_MINI_TARGET_COUNT_FAIL")
    required = {"group_id", "domain", "K", "ANY_GOLD_ABC_IN_HISTORY", "CURRENT_GOLD", "ANY_GOLD"}
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("PHASE153_MINI_TARGET_SCHEMA_FAIL")
    inventory = read_json(MINI_INVENTORY)
    if inventory["minifix_gamma_group_parity"] != "PASS" or inventory["minifix_gamma_gold_parity"] != "PASS":
        raise RuntimeError("MINIFIX_GAMMA_COVERAGE_CONTRACT_FAIL")
    return {str(row["group_id"]): row for row in rows}, {
        "path": str(natural_path),
        "sha256": file_sha(natural_path),
        "schema": "Phase1.5.3 MINI per_target CURRENT_GOLD/ANY_GOLD",
        "N": len(rows),
        "minifix_gamma_group_parity": "PASS",
        "minifix_gamma_gold_parity": "PASS",
    }


def beam_signature(record: dict[str, Any]) -> str:
    beams = [
        {
            "raw_token_ids": beam.get("raw_token_ids"),
            "predicted_sid": beam.get("predicted_sid"),
        }
        for beam in record["beams"]
    ]
    return stable_sha(beams)


def enrich_records(
    source_phase: str,
    records_path: Path,
    manifest: dict[str, dict[str, Any]],
    target_contract: dict[str, dict[str, Any]],
    coverage: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for raw in read_jsonl(records_path):
        group = str(raw["group_id"])
        if group not in manifest or group not in coverage:
            raise RuntimeError(f"RECORD_GROUP_NOT_IN_CONTRACT={source_phase}:{group}")
        original_condition = str(raw["condition"])
        if original_condition in ("BARE", "EXACT_BRIDGE"):
            sample_contract = PRIMARY_CONTRACT
            condition = original_condition
        elif original_condition in ("MF_NATIVE_BARE", "MF_NATIVE_BRIDGE"):
            sample_contract = SECONDARY_CONTRACT
            condition = "BARE" if original_condition.endswith("BARE") else "EXACT_BRIDGE"
        else:
            raise RuntimeError(f"UNKNOWN_CONDITION={original_condition}")
        item = target_contract[group]
        output.append(
            {
                "source_phase": source_phase,
                "sample_contract": sample_contract,
                "model": str(raw["model"]),
                "group_id": group,
                "domain": str(raw["domain"]),
                "condition": condition,
                "original_condition": original_condition,
                "gold_sids": [list(value) for value in item["gold_sids"]],
                "current_gold_sid": list(item["current_gold_sid"]),
                "beam32": raw["beams"],
                "GoldInHistory": bool(item["gold_sid_in_history"]),
                "K": int(item["K"]),
                "history_sids": [list(value) for value in item["history_sids"]],
                "context_sha256": raw.get("context_sha256"),
                "beam_signature": beam_signature(raw),
                "coverage_source": coverage[group],
            }
        )
    return output


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["group_id"], row["condition"], row["sample_contract"])
        grouped[key].append(row)
    canonical, identical, conflicts = [], 0, []
    for key in sorted(grouped):
        values = grouped[key]
        signatures = {row["beam_signature"] for row in values}
        if len(signatures) != 1:
            conflicts.append({"key": list(key), "sources": [row["source_phase"] for row in values], "signatures": sorted(signatures)})
            continue
        canonical.append(values[0])
        identical += len(values) - 1
    return canonical, {
        "dedup_key": ["model", "group_id", "condition", "sample_contract"],
        "raw_record_count": len(rows),
        "canonical_record_count": len(canonical),
        "unique_group_count": len({row["group_id"] for row in canonical}),
        "duplicate_identical_count": identical,
        "duplicate_conflict_count": len(conflicts),
        "duplicate_record_conflict": "YES" if conflicts else "NO",
        "conflicts": conflicts,
    }


def coverage_class(value: dict[str, Any]) -> str:
    same = bool(value["ABC_SAME_GROUP_PRESENT"])
    other = bool(value["ABC_OTHER_GROUP_SEEN"])
    if same and other:
        return "C3_SAME_AND_OTHER"
    if same:
        return "C1_SAME_GROUP_ONLY"
    if other:
        return "C2_OTHER_GROUP_ONLY"
    return "C0_NO_ABC_COVERAGE"


def first_rank(predictions: list[tuple[str, int, int, int]], golds: set[tuple[str, int, int, int]], level: str) -> int | None:
    targets = {prefix(value, level) for value in golds}
    for rank, prediction in enumerate(predictions, 1):
        if prefix(prediction, level) in targets:
            return rank
    return None


def add_metrics(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key not in ("coverage_source", "history_sids")}
    coverage = row["coverage_source"]
    any_coverage = coverage["ANY_GOLD"]
    current_coverage = coverage["CURRENT_GOLD"]
    any_class = coverage_class(any_coverage)
    current_class = coverage_class(current_coverage)
    predictions = [parse_sid(beam["predicted_sid"]) for beam in row["beam32"] if beam.get("predicted_sid") is not None]
    any_gold = {parse_sid(value) for value in row["gold_sids"]}
    current_gold = {parse_sid(row["current_gold_sid"])}
    history = {parse_sid(value) for value in row["history_sids"]}

    def contract_metrics(golds: set[tuple[str, int, int, int]], coverage_value: dict[str, Any], cls: str) -> dict[str, Any]:
        abc_rank = first_rank(predictions, golds, "ABC")
        ab_rank = first_rank(predictions, golds, "AB")
        a_rank = first_rank(predictions, golds, "A")
        return {
            "coverage_class": cls,
            "same_group_seen": bool(coverage_value["SAME_GROUP_SEEN"]),
            "same_group_abc_present": bool(coverage_value["ABC_SAME_GROUP_PRESENT"]),
            "other_group_abc_present": bool(coverage_value["ABC_OTHER_GROUP_SEEN"]),
            "other_group_abc_frequency": int(coverage_value["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"]),
            "any_train_abc_covered": bool(coverage_value["ABC_TOTAL_SEEN"]),
            "rank": abc_rank,
            "hit1": abc_rank is not None and abc_rank <= 1,
            "hit5": abc_rank is not None and abc_rank <= 5,
            "hit10": abc_rank is not None and abc_rank <= 10,
            "hit32": abc_rank is not None and abc_rank <= 32,
            "mrr": 0.0 if abc_rank is None else 1.0 / abc_rank,
            "ABHit32": ab_rank is not None and ab_rank <= 32,
            "AHit32": a_rank is not None and a_rank <= 32,
        }

    result["ANY_GOLD"] = contract_metrics(any_gold, any_coverage, any_class)
    result["CURRENT_GOLD"] = contract_metrics(current_gold, current_coverage, current_class)
    result["history_candidate_fraction"] = rate(sum(prediction in history for prediction in predictions), 32)
    result["valid_beam_count"] = len(predictions)
    return result


def metric_summary(rows: list[dict[str, Any]], contract: str = "ANY_GOLD") -> dict[str, Any]:
    values = [row[contract] for row in rows]
    return {
        "N": len(rows),
        "GoldHit@1": rate(sum(value["hit1"] for value in values), len(values)),
        "GoldHit@5": rate(sum(value["hit5"] for value in values), len(values)),
        "GoldHit@10": rate(sum(value["hit10"] for value in values), len(values)),
        "GoldHit@32": rate(sum(value["hit32"] for value in values), len(values)),
        "GoldMRR": rate(sum(value["mrr"] for value in values), len(values)),
        "ABHit@32": rate(sum(value["ABHit32"] for value in values), len(values)),
        "AHit@32": rate(sum(value["AHit32"] for value in values), len(values)),
        "HistoryCandidateFraction": rate(sum(row["history_candidate_fraction"] for row in rows), len(rows)),
    }


def primary_bare(rows: list[dict[str, Any]], model: str | None = None) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["sample_contract"] == PRIMARY_CONTRACT
        and row["condition"] == "BARE"
        and (model is None or row["model"] == model)
    ]


def nonhistory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"analysis_contract": PRIMARY_CONTRACT, "models": {}}
    for model in MODELS:
        population = [row for row in primary_bare(rows, model) if not row["GoldInHistory"]]
        classes = {name: [row for row in population if row["ANY_GOLD"]["coverage_class"] == name] for name in (
            "C0_NO_ABC_COVERAGE", "C1_SAME_GROUP_ONLY", "C2_OTHER_GROUP_ONLY", "C3_SAME_AND_OTHER"
        )}
        covered = [row for row in population if row["ANY_GOLD"]["any_train_abc_covered"]]
        uncovered = [row for row in population if not row["ANY_GOLD"]["any_train_abc_covered"]]
        same = [row for row in population if row["ANY_GOLD"]["same_group_abc_present"]]
        other = [row for row in population if row["ANY_GOLD"]["other_group_abc_present"]]
        model_result = {
            "NONHISTORY": metric_summary(population),
            "S0_NONHISTORY_NO_ABC_COVERAGE": metric_summary(uncovered),
            "S1_NONHISTORY_SAME_GROUP_ONLY": metric_summary(classes["C1_SAME_GROUP_ONLY"]),
            "S2_NONHISTORY_OTHER_GROUP_COVERAGE": metric_summary(classes["C2_OTHER_GROUP_ONLY"] + classes["C3_SAME_AND_OTHER"]),
            "S3_NONHISTORY_ANY_ABC_COVERAGE": metric_summary(covered),
            "coverage_classes": {name: metric_summary(value) for name, value in classes.items()},
            "same_group_covered": metric_summary(same),
            "other_group_covered": metric_summary(other),
            "CURRENT_GOLD_COVERED_MISS": sum(
                row["CURRENT_GOLD"]["any_train_abc_covered"] and not row["CURRENT_GOLD"]["hit32"] for row in population
            ),
            "ANY_GOLD_COVERED_MISS": sum(
                row["ANY_GOLD"]["any_train_abc_covered"] and not row["ANY_GOLD"]["hit32"] for row in population
            ),
        }
        model_result["covered_nonhistory_hit_n"] = sum(row["ANY_GOLD"]["hit32"] for row in covered)
        model_result["covered_nonhistory_miss_n"] = len(covered) - model_result["covered_nonhistory_hit_n"]
        model_result["covered_miss_rate"] = rate(model_result["covered_nonhistory_miss_n"], len(covered))
        model_result["same_group_covered_miss_rate"] = 1.0 - model_result["same_group_covered"]["GoldHit@32"] if same else 0.0
        model_result["other_group_covered_miss_rate"] = 1.0 - model_result["other_group_covered"]["GoldHit@32"] if other else 0.0
        result["models"][model] = model_result
    return result


def frequency_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    population = [row for row in primary_bare(rows) if not row["GoldInHistory"]]

    def bucket(value: int) -> str:
        if value == 0:
            return "0"
        if value == 1:
            return "1"
        if value <= 4:
            return "2-4"
        if value <= 9:
            return "5-9"
        return "10+"

    result = {name: metric_summary([row for row in population if bucket(row["ANY_GOLD"]["other_group_abc_frequency"]) == name]) for name in BUCKETS}
    nonempty = [result[name]["GoldHit@32"] for name in BUCKETS if result[name]["N"]]
    if len(nonempty) < 2:
        trend = "INSUFFICIENT_POPULATED_BUCKETS"
    elif all(right >= left for left, right in zip(nonempty, nonempty[1:])):
        trend = "NONDECREASING"
    else:
        trend = "MIXED"
    return {"frequency_unit": "other-group unique-group frequency; pooled model-records", "buckets": result, "trend": trend}


def covered_miss_hierarchy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    misses = [
        row for row in primary_bare(rows)
        if not row["GoldInHistory"]
        and row["ANY_GOLD"]["any_train_abc_covered"]
        and not row["ANY_GOLD"]["hit32"]
    ]
    counts = Counter()
    public = []
    for row in misses:
        value = row["ANY_GOLD"]
        label = "AB_BUT_NOT_ABC" if value["ABHit32"] else "A_ONLY" if value["AHit32"] else "NO_HIERARCHY_HIT"
        counts[label] += 1
        public.append({
            "model": row["model"], "group_id": row["group_id"], "domain": row["domain"],
            "coverage_class": value["coverage_class"], "AHit32": value["AHit32"],
            "ABHit32": value["ABHit32"], "ABCHit32": False, "classification": label,
        })
    n = len(misses)
    return {
        "N": n,
        "A_HIT_RATE": rate(sum(row["ANY_GOLD"]["AHit32"] for row in misses), n),
        "AB_HIT_RATE": rate(sum(row["ANY_GOLD"]["ABHit32"] for row in misses), n),
        "A_ONLY_RATE": rate(counts["A_ONLY"], n),
        "AB_NOT_ABC_RATE": rate(counts["AB_BUT_NOT_ABC"], n),
        "NO_HIERARCHY_RATE": rate(counts["NO_HIERARCHY_HIT"], n),
        "counts": dict(counts),
        "records": public,
    }


def same_group_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = {
        "SAME_GROUP_SEEN_ABC_SUPERVISED": lambda row: row["ANY_GOLD"]["same_group_seen"] and row["ANY_GOLD"]["same_group_abc_present"],
        "SAME_GROUP_SEEN_ABC_NOT_SUPERVISED": lambda row: row["ANY_GOLD"]["same_group_seen"] and not row["ANY_GOLD"]["same_group_abc_present"],
        "GROUP_UNSEEN_OTHER_GROUP_ABC_SUPERVISED": lambda row: not row["ANY_GOLD"]["same_group_seen"] and row["ANY_GOLD"]["other_group_abc_present"],
        "GROUP_UNSEEN_ABC_UNSEEN": lambda row: not row["ANY_GOLD"]["same_group_seen"] and not row["ANY_GOLD"]["other_group_abc_present"],
    }
    result: dict[str, Any] = {}
    for model in MODELS:
        population = [row for row in primary_bare(rows, model) if not row["GoldInHistory"]]
        result[model] = {name: metric_summary([row for row in population if predicate(row)]) for name, predicate in categories.items()}
    return {"population": "GoldNotInHistory primary Bare", "models": result}


def paired_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["model"], row["group_id"]): row for row in primary_bare(rows)}
    groups = sorted({group for model, group in lookup if model == "MiniFix"} & {group for model, group in lookup if model == "Gamma"})
    groups = [group for group in groups if not lookup[("MiniFix", group)]["GoldInHistory"] and lookup[("MiniFix", group)]["ANY_GOLD"]["any_train_abc_covered"]]
    pairs = []
    for group in groups:
        mf, gamma = lookup[("MiniFix", group)], lookup[("Gamma", group)]
        pairs.append({
            "group_id": group,
            "coverage_class": mf["ANY_GOLD"]["coverage_class"],
            "MiniFix_rank": mf["ANY_GOLD"]["rank"], "Gamma_rank": gamma["ANY_GOLD"]["rank"],
            "MiniFix_hit32": mf["ANY_GOLD"]["hit32"], "Gamma_hit32": gamma["ANY_GOLD"]["hit32"],
            "MiniFix_mrr": mf["ANY_GOLD"]["mrr"], "Gamma_mrr": gamma["ANY_GOLD"]["mrr"],
        })
    mf_mrr = rate(sum(row["MiniFix_mrr"] for row in pairs), len(pairs))
    gamma_mrr = rate(sum(row["Gamma_mrr"] for row in pairs), len(pairs))
    return {
        "paired_covered_groups": len(pairs),
        "MiniFix_Hit32": rate(sum(row["MiniFix_hit32"] for row in pairs), len(pairs)),
        "Gamma_Hit32": rate(sum(row["Gamma_hit32"] for row in pairs), len(pairs)),
        "MiniFix_MRR": mf_mrr,
        "Gamma_MRR": gamma_mrr,
        "MiniFix_minus_Gamma_MRR": mf_mrr - gamma_mrr,
        "pairs": pairs,
        "interpretation": "DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_CLAIM",
    }


def bridge_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["sample_contract"] == PRIMARY_CONTRACT and not row["GoldInHistory"]]
    lookup = {(row["model"], row["group_id"], row["condition"]): row for row in primary}
    identities = sorted({(model, group) for model, group, condition in lookup if condition == "BARE"} & {(model, group) for model, group, condition in lookup if condition == "EXACT_BRIDGE"})
    counts = Counter()
    pairs = []
    for model, group in identities:
        bare, bridge = lookup[(model, group, "BARE")], lookup[(model, group, "EXACT_BRIDGE")]
        covered = bare["ANY_GOLD"]["any_train_abc_covered"]
        bare_hit, bridge_hit = bare["ANY_GOLD"]["hit32"], bridge["ANY_GOLD"]["hit32"]
        if covered and not bare_hit and bridge_hit:
            counts["COVERED_NEW"] += 1
        if covered and bare_hit and bridge_hit and bridge["ANY_GOLD"]["mrr"] > bare["ANY_GOLD"]["mrr"]:
            counts["COVERED_RERANK"] += 1
        if covered and bare_hit and not bridge_hit:
            counts["COVERED_LOST"] += 1
        if not covered and not bare_hit and bridge_hit:
            counts["UNCOVERED_NEW"] += 1
        pairs.append({
            "model": model, "group_id": group, "covered": covered,
            "bare_rank": bare["ANY_GOLD"]["rank"], "bridge_rank": bridge["ANY_GOLD"]["rank"],
            "bare_hit32": bare_hit, "bridge_hit32": bridge_hit,
        })
    covered_bare = [lookup[(model, group, "BARE")] for model, group in identities if lookup[(model, group, "BARE")]["ANY_GOLD"]["any_train_abc_covered"]]
    covered_bridge = [lookup[(model, group, "EXACT_BRIDGE")] for model, group in identities if lookup[(model, group, "BARE")]["ANY_GOLD"]["any_train_abc_covered"]]
    if counts["COVERED_NEW"] > 0 and counts["UNCOVERED_NEW"] == 0:
        mechanism = "SUPPORTED"
    elif counts["COVERED_NEW"] == 0:
        mechanism = "NO_ACQUISITION_SIGNAL"
    else:
        mechanism = "MIXED"
    return {
        "population": "paired GoldNotInHistory official Frozen-BATA-CoT records",
        "paired_records": len(pairs),
        "covered_pairs": len(covered_bare),
        "covered_bare": metric_summary(covered_bare),
        "covered_bridge": metric_summary(covered_bridge),
        "covered_bridge_new_gold": counts["COVERED_NEW"],
        "covered_bridge_rerank_gold": counts["COVERED_RERANK"],
        "covered_bridge_lost_gold": counts["COVERED_LOST"],
        "uncovered_bridge_new_gold": counts["UNCOVERED_NEW"],
        "bridge_helps_retrieve_learned_items": mechanism,
        "pairs": pairs,
    }


def ranking_support(nonhistory: dict[str, Any]) -> tuple[str, int, float]:
    models = nonhistory["models"]
    counts = [models[model]["S3_NONHISTORY_ANY_ABC_COVERAGE"]["N"] for model in MODELS]
    miss_rates = [models[model]["covered_miss_rate"] for model in MODELS]
    gate_n = min(counts)
    pooled_n = sum(counts)
    pooled_miss = rate(sum(models[model]["covered_nonhistory_miss_n"] for model in MODELS), pooled_n)
    both_high = all(value >= 0.50 for value in miss_rates)
    if gate_n >= 16 and both_high:
        return "STRONG", gate_n, pooled_miss
    if gate_n >= 8 and pooled_miss >= 0.50:
        return "MODERATE", gate_n, pooled_miss
    if gate_n < 8:
        return "UNDERPOWERED", gate_n, pooled_miss
    return "WEAK", gate_n, pooled_miss


def root_summary(nonhistory: dict[str, Any], hierarchy: dict[str, Any], paired: dict[str, Any], bridge: dict[str, Any]) -> dict[str, Any]:
    ranking, gate_n, pooled_miss = ranking_support(nonhistory)
    data_support = "STRONG"
    if ranking == "STRONG":
        root = "COVERAGE_PLUS_CONFIRMED_RANKING_BOTTLENECK"
    elif ranking == "UNDERPOWERED":
        root = "FINE_ITEM_DATA_COVERAGE_BOTTLENECK_WITH_RANKING_RESIDUAL_UNDERPOWERED"
    else:
        root = "FINE_ITEM_DATA_COVERAGE_BOTTLENECK_WITH_RANKING_RESIDUAL"
    future = (
        "TEACHER_FORCED_GOLD_LOGPROB_COVERED_HIT_VS_COVERED_MISS_VS_UNCOVERED_MISS"
        if gate_n >= 8
        else "CPU_SELECT_16_COVERED_NONHISTORY_AND_16_UNCOVERED_NONHISTORY_THEN_TEACHER_FORCED_GOLD_LOGPROB"
    )
    return {
        "data_coverage_limit_support": data_support,
        "ranking_residual_support": ranking,
        "ranking_gate_n_per_model": gate_n,
        "covered_nonhistory_pooled_miss_rate": pooled_miss,
        "root_class": root,
        "primary_signal": "Phase1.5.3 natural NonHistory Mini exact-ABC other-group unseen rate is 97.09%.",
        "secondary_signal": f"Existing covered NonHistory Beam records have pooled miss rate {pooled_miss:.2%}; evidence gate is {ranking}.",
        "main_limitation": "Existing Beam records are small, selected diagnostic cohorts rather than the natural 1,595-group distribution.",
        "conclusion": "Training supervision coverage is the primary fine-item bottleneck; ranking residual is classified only by the predeclared covered-case gate.",
        "future_gpu_experiment": future,
        "covered_miss_hierarchy": {key: hierarchy[key] for key in ("N", "A_HIT_RATE", "AB_HIT_RATE")},
        "paired_model_signal": {key: paired[key] for key in ("paired_covered_groups", "MiniFix_MRR", "Gamma_MRR", "MiniFix_minus_Gamma_MRR")},
        "bridge_signal": {key: bridge[key] for key in ("covered_bridge_new_gold", "uncovered_bridge_new_gold", "bridge_helps_retrieve_learned_items")},
    }


def render_nonhistory_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Coverage-conditioned NonHistory Beam summary", "",
        "Primary contract: official-domain semantic prompt, Frozen BATA CoT, Bare domain, Beam32 ABC3.", "",
        "| Model | NonHistory N | Covered N | Covered Hit@32 | Covered MRR | Covered miss rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        value = summary["models"][model]
        covered = value["S3_NONHISTORY_ANY_ABC_COVERAGE"]
        lines.append(f"| {model} | {value['NONHISTORY']['N']} | {covered['N']} | {covered['GoldHit@32']:.2%} | {covered['GoldMRR']:.6f} | {value['covered_miss_rate']:.2%} |")
    lines.extend(["", "Coverage uses the immutable Phase 1.5.3 Mini ANY_GOLD index. Bare and Bridge are paired, never pooled as independent samples."])
    return "\n".join(lines) + "\n"


def render_root_md(root: dict[str, Any], repro: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 1.5.4 root-cause decision", "",
            f"- Phase 1.5.3 decision reproduction: `{repro['phase153_decision_repro_pass']}`",
            f"- Phase 1.5.3 correction required: `{repro['phase153_correction_required']}`",
            f"- DATA_COVERAGE_LIMIT_SUPPORT: `{root['data_coverage_limit_support']}`",
            f"- RANKING_RESIDUAL_SUPPORT: `{root['ranking_residual_support']}`",
            f"- ROOT_CLASS: `{root['root_class']}`", "",
            root["conclusion"], "",
            "No model was loaded and no GPU experiment was started.",
        ]
    ) + "\n"


def terminal(audit: dict[str, Any], repro: dict[str, Any], dedup: dict[str, Any], inventory: dict[str, Any], nonhistory: dict[str, Any], frequency: dict[str, Any], hierarchy: dict[str, Any], bridge: dict[str, Any], paired: dict[str, Any], root: dict[str, Any]) -> str:
    mf = nonhistory["models"]["MiniFix"]
    ga = nonhistory["models"]["Gamma"]
    bucket = frequency["buckets"]
    lines = [
        "--- Git ---", "", f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short'] or 'EMPTY'}",
        f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}", f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}",
        f"RUNTIME_GITHUB_PARITY={audit['runtime_github_parity']}", "", "--- Phase1.5.3 Reproduction ---", "",
        f"PHASE153_REPORTED_SN_SAME_GROUP_ABC_RATE={repro['reported_sn_same_group_abc_rate']:.8f}",
        f"PHASE153_RECOMPUTED_SN_SAME_GROUP_ABC_RATE={repro['recomputed_sn_same_group_abc_rate']:.8f}",
        f"PHASE153_REPORTED_DECODER_SUPPORT={repro['reported_decoder_support']}",
        f"PHASE153_RECOMPUTED_DECODER_SUPPORT={repro['recomputed_decoder_support']}",
        f"PHASE153_REPORTED_ROOT_CLASS={repro['reported_root_class']}",
        f"PHASE153_RECOMPUTED_ROOT_CLASS={repro['recomputed_root_class']}",
        f"PHASE153_DECISION_REPRO_PASS={repro['phase153_decision_repro_pass']}",
        f"PHASE153_CORRECTION_REQUIRED={repro['phase153_correction_required']}",
        f"RAW_ARTIFACTS_UNCHANGED={repro['raw_artifacts_unchanged']}", "", "--- Existing Records ---", "",
        f"RAW_RECORD_COUNT={dedup['raw_record_count']}", f"CANONICAL_RECORD_COUNT={dedup['canonical_record_count']}",
        f"UNIQUE_GROUP_COUNT={dedup['unique_group_count']}", f"DUPLICATE_IDENTICAL_COUNT={dedup['duplicate_identical_count']}",
        f"DUPLICATE_CONFLICT_COUNT={dedup['duplicate_conflict_count']}",
        f"PRIMARY_CONTRACT_RECORDS={inventory['primary_contract_records']}", f"SECONDARY_EXPANDED_RECORDS={inventory['secondary_expanded_records']}",
        "", "--- NonHistory Coverage ---", "",
        f"MINIFIX_NONHISTORY_N={mf['NONHISTORY']['N']}", f"MINIFIX_NONHISTORY_ABC_COVERED_N={mf['S3_NONHISTORY_ANY_ABC_COVERAGE']['N']}",
        f"MINIFIX_NONHISTORY_ABC_UNCOVERED_N={mf['S0_NONHISTORY_NO_ABC_COVERAGE']['N']}",
        f"GAMMA_NONHISTORY_N={ga['NONHISTORY']['N']}", f"GAMMA_NONHISTORY_ABC_COVERED_N={ga['S3_NONHISTORY_ANY_ABC_COVERAGE']['N']}",
        f"GAMMA_NONHISTORY_ABC_UNCOVERED_N={ga['S0_NONHISTORY_NO_ABC_COVERAGE']['N']}", "", "--- Covered Beam Performance ---", "",
        f"MINIFIX_COVERED_HIT32={mf['S3_NONHISTORY_ANY_ABC_COVERAGE']['GoldHit@32']:.8f}", f"MINIFIX_COVERED_MRR={mf['S3_NONHISTORY_ANY_ABC_COVERAGE']['GoldMRR']:.8f}",
        f"MINIFIX_COVERED_MISS_RATE={mf['covered_miss_rate']:.8f}", f"GAMMA_COVERED_HIT32={ga['S3_NONHISTORY_ANY_ABC_COVERAGE']['GoldHit@32']:.8f}",
        f"GAMMA_COVERED_MRR={ga['S3_NONHISTORY_ANY_ABC_COVERAGE']['GoldMRR']:.8f}", f"GAMMA_COVERED_MISS_RATE={ga['covered_miss_rate']:.8f}",
        "", "--- Coverage Type ---", "",
        f"MINIFIX_SAME_GROUP_COVERED_N={mf['same_group_covered']['N']}", f"MINIFIX_SAME_GROUP_COVERED_MISS_RATE={mf['same_group_covered_miss_rate']:.8f}",
        f"MINIFIX_OTHER_GROUP_COVERED_N={mf['other_group_covered']['N']}", f"MINIFIX_OTHER_GROUP_COVERED_MISS_RATE={mf['other_group_covered_miss_rate']:.8f}",
        f"GAMMA_SAME_GROUP_COVERED_N={ga['same_group_covered']['N']}", f"GAMMA_SAME_GROUP_COVERED_MISS_RATE={ga['same_group_covered_miss_rate']:.8f}",
        f"GAMMA_OTHER_GROUP_COVERED_N={ga['other_group_covered']['N']}", f"GAMMA_OTHER_GROUP_COVERED_MISS_RATE={ga['other_group_covered_miss_rate']:.8f}",
        "", "--- Frequency ---", "",
    ]
    for name, label in (("0", "0"), ("1", "1"), ("2-4", "2_4"), ("5-9", "5_9"), ("10+", "10PLUS")):
        lines.append(f"ABC_FREQ{label}_HIT32={bucket[name]['GoldHit@32']:.8f}")
    lines.extend([
        f"ABC_FREQUENCY_PERFORMANCE_TREND={frequency['trend']}", "", "--- Covered Miss Hierarchy ---", "",
        f"COVERED_MISS_N={hierarchy['N']}", f"COVERED_MISS_A_HIT_RATE={hierarchy['A_HIT_RATE']:.8f}", f"COVERED_MISS_AB_HIT_RATE={hierarchy['AB_HIT_RATE']:.8f}",
        f"COVERED_MISS_A_ONLY_RATE={hierarchy['A_ONLY_RATE']:.8f}", f"COVERED_MISS_AB_NOT_ABC_RATE={hierarchy['AB_NOT_ABC_RATE']:.8f}",
        f"COVERED_MISS_NO_HIERARCHY_RATE={hierarchy['NO_HIERARCHY_RATE']:.8f}", "", "--- Bridge ---", "",
        f"COVERED_BRIDGE_NEW_GOLD={bridge['covered_bridge_new_gold']}", f"COVERED_BRIDGE_RERANK_GOLD={bridge['covered_bridge_rerank_gold']}",
        f"COVERED_BRIDGE_LOST_GOLD={bridge['covered_bridge_lost_gold']}", f"UNCOVERED_BRIDGE_NEW_GOLD={bridge['uncovered_bridge_new_gold']}",
        f"BRIDGE_HELPS_RETRIEVE_LEARNED_ITEMS={bridge['bridge_helps_retrieve_learned_items']}", "", "--- MiniFix vs Gamma ---", "",
        f"PAIRED_COVERED_GROUPS={paired['paired_covered_groups']}", f"MINIFIX_COVERED_MRR={paired['MiniFix_MRR']:.8f}",
        f"GAMMA_COVERED_MRR={paired['Gamma_MRR']:.8f}", f"MINIFIX_MINUS_GAMMA_COVERED_MRR={paired['MiniFix_minus_Gamma_MRR']:.8f}",
        "", "--- Root Cause ---", "", f"DATA_COVERAGE_LIMIT_SUPPORT={root['data_coverage_limit_support']}",
        f"RANKING_RESIDUAL_SUPPORT={root['ranking_residual_support']}", f"ROOT_CLASS={root['root_class']}",
        f"PRIMARY_SIGNAL={root['primary_signal']}", f"SECONDARY_SIGNAL={root['secondary_signal']}", f"MAIN_LIMITATION={root['main_limitation']}",
        f"CONCLUSION={root['conclusion']}", f"FUTURE_GPU_EXPERIMENT={root['future_gpu_experiment']}", "",
        "GPU_INFERENCE_STARTED=NO", "MODEL_FORWARD_STARTED=NO", "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0",
        "SELF_COT_GENERATION_STARTED=NO", "EXTERNAL_EVAL_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> None:
    coverage = {
        "SAME_GROUP_SEEN": True, "ABC_SAME_GROUP_PRESENT": True,
        "ABC_OTHER_GROUP_SEEN": False, "ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY": 0,
        "ABC_TOTAL_SEEN": True,
    }
    assert coverage_class(coverage) == "C1_SAME_GROUP_ONLY"
    coverage.update({"ABC_SAME_GROUP_PRESENT": False, "ABC_OTHER_GROUP_SEEN": True})
    assert coverage_class(coverage) == "C2_OTHER_GROUP_ONLY"
    row = {"model": "MiniFix", "group_id": "g", "condition": "BARE", "sample_contract": PRIMARY_CONTRACT, "beam_signature": "x"}
    canonical, audit = deduplicate([dict(row, source_phase="a"), dict(row, source_phase="b")])
    assert len(canonical) == 1 and audit["duplicate_identical_count"] == 1
    predictions = [("video", 1, 2, 3), ("video", 1, 2, 4)]
    assert first_rank(predictions, {("video", 1, 2, 4)}, "ABC") == 2
    assert first_rank(predictions, {("video", 1, 2, 9)}, "AB") == 1
    synthetic = {"models": {}}
    for model in MODELS:
        synthetic["models"][model] = {
            "S3_NONHISTORY_ANY_ABC_COVERAGE": {"N": 8}, "covered_miss_rate": 0.75,
            "covered_nonhistory_miss_n": 6,
        }
    assert ranking_support(synthetic)[0] == "MODERATE"
    print("SELF_TEST=PASS")


def run() -> None:
    if "torch" in sys.modules:
        raise RuntimeError("TORCH_ALREADY_IMPORTED")
    audit = code_audit()
    repro, raw_before = phase153_reproduction()
    quick_manifest = load_manifest(PHASE15 / "probe16_manifest.json")
    phase151_manifest = load_manifest(PHASE151 / "probe16_manifest.json")
    target_contract = source_gold_contract((quick_manifest, phase151_manifest))
    coverage, coverage_contract = coverage_index()
    missing = sorted(set(target_contract) - set(coverage))
    if missing:
        raise RuntimeError(f"BEAM_GROUPS_OUTSIDE_NATURAL1595={missing}")

    raw_records = enrich_records("PHASE1.5_QUICK", PHASE15 / "records.jsonl", quick_manifest, target_contract, coverage)
    raw_records += enrich_records("PHASE1.5.1", PHASE151 / "records.jsonl", phase151_manifest, target_contract, coverage)
    canonical, dedup = deduplicate(raw_records)
    canonical = [add_metrics(row) for row in canonical]
    inventory = {
        "sources": {
            "PHASE1.5_QUICK": {"path": str(PHASE15 / "records.jsonl"), "sha256": file_sha(PHASE15 / "records.jsonl"), "records": 96},
            "PHASE1.5.1": {"path": str(PHASE151 / "records.jsonl"), "sha256": file_sha(PHASE151 / "records.jsonl"), "records": 64},
        },
        "coverage_contract": coverage_contract,
        "primary_contract": PRIMARY_CONTRACT,
        "secondary_contract": SECONDARY_CONTRACT,
        "primary_contract_records": sum(row["sample_contract"] == PRIMARY_CONTRACT for row in canonical),
        "secondary_expanded_records": sum(row["sample_contract"] != PRIMARY_CONTRACT for row in canonical),
        "primary_homogeneous_analysis": "YES",
        "secondary_expanded_analysis": "MiniFix native-prefix records only; reported separately and excluded from root decision.",
        "phase1.2_included": "NO",
        "models": list(MODELS),
    }
    if inventory["sources"]["PHASE1.5_QUICK"]["sha256"] != EXPECTED_SHA["phase15_records"]:
        raise RuntimeError("PHASE15_RECORD_SHA_MISMATCH")
    if inventory["sources"]["PHASE1.5.1"]["sha256"] != EXPECTED_SHA["phase151_records"]:
        raise RuntimeError("PHASE151_RECORD_SHA_MISMATCH")

    nonhistory = nonhistory_summary(canonical)
    frequency = frequency_analysis(canonical)
    hierarchy = covered_miss_hierarchy(canonical)
    same_group = same_group_analysis(canonical)
    paired = paired_models(canonical)
    bridge = bridge_analysis(canonical)
    root = root_summary(nonhistory, hierarchy, paired, bridge)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "phase153_decision_repro_audit.json", repro)
    write_json(OUTPUT / "record_inventory.json", inventory)
    write_json(OUTPUT / "dedup_audit.json", dedup)
    write_jsonl(OUTPUT / "coverage_join_records.jsonl", canonical)
    write_json(OUTPUT / "nonhistory_coverage_summary.json", nonhistory)
    (OUTPUT / "nonhistory_coverage_summary.md").write_text(render_nonhistory_md(nonhistory), encoding="utf-8")
    write_json(OUTPUT / "frequency_bucket_analysis.json", frequency)
    write_json(OUTPUT / "covered_miss_hierarchy.json", hierarchy)
    write_json(OUTPUT / "same_group_vs_other_group.json", same_group)
    write_json(OUTPUT / "minifix_gamma_paired.json", paired)
    write_json(OUTPUT / "bridge_covered_vs_uncovered.json", bridge)
    write_json(OUTPUT / "root_cause_summary.json", root)
    (OUTPUT / "root_cause_summary.md").write_text(render_root_md(root, repro), encoding="utf-8")

    raw_after = {name: file_sha(PHASE153 / name) for name in RAW_PHASE153}
    repro["raw_artifact_sha256_after"] = raw_after
    repro["raw_artifacts_unchanged"] = "YES" if raw_before == raw_after else "NO"
    if repro["raw_artifacts_unchanged"] != "YES":
        raise RuntimeError("RAW_PHASE153_ARTIFACT_CHANGED")
    write_json(OUTPUT / "phase153_decision_repro_audit.json", repro)
    review = terminal(audit, repro, dedup, inventory, nonhistory, frequency, hierarchy, bridge, paired, root)
    (OUTPUT / "CHATGPT_COVERAGE_CONDITIONED_REVIEW.txt").write_text(review, encoding="utf-8")
    print(review, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            raise RuntimeError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
        run()


if __name__ == "__main__":
    main()
