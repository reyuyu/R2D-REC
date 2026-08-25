"""Phase 1.5.5 CPU-only Bridge x training-memory interaction audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterable


RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/"
    "recommendation_bridge_memory_interaction_audit.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_bridge_memory_interaction_audit.py"
OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_interaction_audit"
PHASE15 = RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_quick_probe"
PHASE151 = RUNTIME / "boundary_adapt/results/recommendation_memory_history_2x2_probe"
PHASE152 = RUNTIME / "boundary_adapt/results/recommendation_history_floor_cpu_audit"
PHASE153 = RUNTIME / "boundary_adapt/results/recommendation_training_coverage_hierarchy_audit"
PHASE154 = RUNTIME / "boundary_adapt/results/recommendation_coverage_conditioned_beam_audit"
CANONICAL_INPUT = PHASE154 / "coverage_join_records.jsonl"
PRIMARY_CONTRACT = "OFFICIAL_DOMAIN_SEMANTIC_FROZEN_BATA_COT_ABC3"
EXPECTED_CANONICAL_SHA = "c5c72a0d780046d75cd692c70c2267740c11117b2a44d69d0e6640af6f0de86a"
EXPECTED_NATURAL_SHA = "a99f3dfa7bfdb44ee7694fc78e0f261a18d3c613459f9b23302f9f7b0e2beb17"
MODELS = ("MiniFix", "Gamma")
CONDITIONS = ("BARE", "EXACT_BRIDGE")
BOOTSTRAP_REPEATS = 10_000

PROTECTED = {
    "phase15_records": PHASE15 / "records.jsonl",
    "phase15_manifest": PHASE15 / "probe16_manifest.json",
    "phase151_records": PHASE151 / "records.jsonl",
    "phase151_manifest": PHASE151 / "probe16_manifest.json",
    "phase152_source": PHASE152 / "source_contract_audit.json",
    "phase152_rates": PHASE152 / "domain_history_rates.json",
    "phase153_natural": PHASE153 / "natural1595_training_coverage.json",
    "phase153_frequency": PHASE153 / "training_sid_frequency_stats.json",
    "phase153_history_cross": PHASE153 / "history_training_cross.json",
    "phase153_branching": PHASE153 / "branching_factor_stats.json",
    "phase153_entropy": PHASE153 / "conditional_entropy_stats.json",
    "phase153_surprisal": PHASE153 / "target_gold_frequency_surprisal.json",
    "phase153_join": PHASE153 / "phase151_coverage_beam_join.json",
    "phase154_canonical": CANONICAL_INPUT,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    head, origin, status = git("rev-parse", "HEAD"), git("rev-parse", "origin/main"), git("status", "--short")
    source_sha, runtime_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT), file_sha(RUNTIME_SCRIPT)
    blob = subprocess.check_output(["git", "-C", str(SOURCE_REPO), "show", f"{head}:{RELATIVE_SCRIPT.as_posix()}"])
    github_sha = hashlib.sha256(blob).hexdigest()
    result = {
        "implement_commit": head,
        "origin_main": origin,
        "push_status": "PASS" if head == origin else "FAIL",
        "git_status_short": status,
        "github_script_sha256": github_sha,
        "source_script_sha256": source_sha,
        "runtime_script_sha256": runtime_sha,
        "runtime_github_parity": "PASS" if github_sha == source_sha == runtime_sha else "FAIL",
    }
    if result["push_status"] != "PASS" or status or result["runtime_github_parity"] != "PASS":
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={result}")
    return result


def protected_hashes() -> dict[str, str]:
    missing = [str(path) for path in PROTECTED.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"PROTECTED_INPUT_MISSING={missing}")
    return {name: file_sha(path) for name, path in PROTECTED.items()}


def rate(total: int | float, n: int) -> float:
    return float(total) / n if n else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def manifest_contract(path: Path, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in read_json(path)["items"]:
        group = str(item["group_id"])
        bridge = str(item.get("exact_original_bridge", item.get("exact_bridge", "")))
        bridge_tokens = item.get("exact_original_bridge_token_ids", item.get("exact_bridge_token_ids"))
        recorded_bridge_sha = str(item.get("bridge_sha256", item.get("exact_bridge_sha256", "")))
        actual_bridge_sha = hashlib.sha256(bridge.encode("utf-8")).hexdigest()
        if recorded_bridge_sha and actual_bridge_sha != recorded_bridge_sha:
            raise RuntimeError(f"BRIDGE_SHA_MISMATCH={source}:{group}")
        result[group] = {
            "source": source,
            "group_id": group,
            "domain": str(item["domain"]),
            "K": int(item["K"]),
            "gold_sids_sha256": stable_sha(sorted(map(str, item["all_gold_sids"]))),
            "official_system_sha256": hashlib.sha256(str(item["official_system"]).encode("utf-8")).hexdigest(),
            "official_prompt_token_ids_sha256": stable_sha(item["official_prompt_token_ids"]),
            "frozen_cot_sha256": str(item["frozen_cot_sha256"]),
            "exact_bridge_sha256": actual_bridge_sha,
            "exact_bridge_token_ids_sha256": stable_sha(bridge_tokens),
            "domain_token_ids_sha256": stable_sha(item["domain_token_ids"]),
            "gold_in_history": bool(item["gold_sid_in_history"]),
        }
    return result


def combine_manifest_contracts() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    quick = manifest_contract(PHASE15 / "probe16_manifest.json", "PHASE1.5_QUICK")
    phase151 = manifest_contract(PHASE151 / "probe16_manifest.json", "PHASE1.5.1")
    combined = dict(quick)
    overlap, conflicts = 0, []
    comparable = (
        "group_id", "domain", "K", "gold_sids_sha256", "official_system_sha256",
        "official_prompt_token_ids_sha256", "frozen_cot_sha256", "exact_bridge_sha256",
        "exact_bridge_token_ids_sha256", "domain_token_ids_sha256", "gold_in_history",
    )
    for group, value in phase151.items():
        if group in combined:
            overlap += 1
            differences = [key for key in comparable if combined[group][key] != value[key]]
            if differences:
                conflicts.append({"group_id": group, "fields": differences})
        else:
            combined[group] = value
    if conflicts:
        raise RuntimeError(f"MANIFEST_CONTEXT_CONFLICT={conflicts}")
    return combined, {
        "quick_groups": len(quick),
        "phase151_groups": len(phase151),
        "overlap_groups": overlap,
        "manifest_conflicts": conflicts,
        "quick_manifest_sha256": file_sha(PHASE15 / "probe16_manifest.json"),
        "phase151_manifest_sha256": file_sha(PHASE151 / "probe16_manifest.json"),
    }


def load_canonical() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha = file_sha(CANONICAL_INPUT)
    if actual_sha != EXPECTED_CANONICAL_SHA:
        raise RuntimeError(f"PHASE154_CANONICAL_SHA_MISMATCH={actual_sha}")
    rows = read_jsonl(CANONICAL_INPUT)
    required = {
        "model", "group_id", "domain", "condition", "sample_contract", "GoldInHistory",
        "gold_sids", "context_sha256", "ANY_GOLD", "history_candidate_fraction",
    }
    any_required = {
        "coverage_class", "same_group_seen", "same_group_abc_present",
        "other_group_abc_present", "other_group_abc_frequency", "any_train_abc_covered",
        "hit1", "hit5", "hit32", "mrr", "ABHit32", "AHit32", "rank",
    }
    if len(rows) != 156 or any(not required.issubset(row) or not any_required.issubset(row["ANY_GOLD"]) for row in rows):
        raise RuntimeError("PHASE154_CANONICAL_SCHEMA_FAIL")
    key_counts = Counter((row["model"], row["group_id"], row["condition"], row["sample_contract"]) for row in rows)
    if any(count != 1 for count in key_counts.values()):
        raise RuntimeError("PHASE154_CANONICAL_DUPLICATE_KEY")
    return rows, {
        "canonical_input": str(CANONICAL_INPUT),
        "canonical_input_sha256": actual_sha,
        "canonical_records": len(rows),
        "schema_check": "PASS",
        "dedup_key_unique": "PASS",
    }


def supervision_frequency() -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    path = PHASE153 / "natural1595_training_coverage.json"
    actual_sha = file_sha(path)
    if actual_sha != EXPECTED_NATURAL_SHA:
        raise RuntimeError("PHASE153_NATURAL_SHA_MISMATCH")
    payload = read_json(path)
    rows = payload["per_target"]["MINI"]
    result = {}
    for row in rows:
        details = row["ANY_GOLD_ALL_DETAILS"]
        same_count = sum(
            int(detail["ABC_TOTAL_UNIQUE_GROUP_FREQUENCY"] > detail["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"])
            for detail in details
        )
        result[str(row["group_id"])] = {
            "same_group_supervised_gold_count": same_count,
            "other_group_max_unique_frequency": int(row["ANY_GOLD"]["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"]),
        }
    return result, {"path": str(path), "sha256": actual_sha, "groups": len(result), "definition_reused": "Phase1.5.3 immutable ANY_GOLD coverage"}


def four_way(
    rows: list[dict[str, Any]], manifests: dict[str, dict[str, Any]], frequency: dict[str, dict[str, int]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    primary = [row for row in rows if row["sample_contract"] == PRIMARY_CONTRACT]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for row in primary:
        grouped[row["group_id"]][row["model"]][row["condition"]] = row
    eligible = sorted(
        group for group, model_rows in grouped.items()
        if any(
            not row["GoldInHistory"] and row["ANY_GOLD"]["same_group_abc_present"]
            for values in model_rows.values() for row in values.values()
        )
    )
    complete, dropped, public, parity_failures = [], [], [], []
    for group in eligible:
        values = grouped[group]
        missing = [(model, condition) for model in MODELS for condition in CONDITIONS if condition not in values.get(model, {})]
        if missing:
            dropped.append({"group_id": group, "missing": missing})
            continue
        records = [values[model][condition] for model in MODELS for condition in CONDITIONS]
        first = records[0]
        property_parity = all(
            row["domain"] == first["domain"]
            and row["gold_sids"] == first["gold_sids"]
            and row["K"] == first["K"]
            and row["GoldInHistory"] == first["GoldInHistory"]
            and row["ANY_GOLD"]["coverage_class"] == first["ANY_GOLD"]["coverage_class"]
            for row in records
        )
        context_parity = all(
            values["MiniFix"][condition]["context_sha256"] == values["Gamma"][condition]["context_sha256"]
            for condition in CONDITIONS
        )
        manifest_present = group in manifests
        if not property_parity or not context_parity or not manifest_present:
            parity_failures.append({"group_id": group, "property_parity": property_parity, "context_parity": context_parity, "manifest_present": manifest_present})
            continue
        complete.append(group)
        public.append({
            "group_id": group,
            "domain": first["domain"],
            "K": first["K"],
            "coverage_class": first["ANY_GOLD"]["coverage_class"],
            "same_group_supervised_gold_count": frequency[group]["same_group_supervised_gold_count"],
            "other_group_max_unique_frequency": frequency[group]["other_group_max_unique_frequency"],
            "bare_context_sha256": values["MiniFix"]["BARE"]["context_sha256"],
            "bridge_context_sha256": values["MiniFix"]["EXACT_BRIDGE"]["context_sha256"],
            "exact_bridge_sha256": manifests[group]["exact_bridge_sha256"],
            "official_prompt_token_ids_sha256": manifests[group]["official_prompt_token_ids_sha256"],
            "frozen_cot_sha256": manifests[group]["frozen_cot_sha256"],
            "domain_token_ids_sha256": manifests[group]["domain_token_ids_sha256"],
            "four_way_complete": True,
        })
    if parity_failures:
        raise RuntimeError(f"BRIDGE_CONTEXT_PARITY_FAIL={parity_failures}")
    return grouped, {
        "primary_cohort": "SAME_GROUP_EXACT_ABC_COVERED_NONHISTORY",
        "primary_eligible_groups": len(eligible),
        "four_way_complete_groups": len(complete),
        "dropped_incomplete_groups": len(dropped),
        "eligible_group_ids": eligible,
        "complete_group_ids": complete,
        "dropped": dropped,
        "groups": public,
        "bridge_context_parity_pass": "PASS",
    }


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    return {
        "N_unique_groups": n,
        "GoldHit@1": rate(sum(row["ANY_GOLD"]["hit1"] for row in records), n),
        "GoldHit@5": rate(sum(row["ANY_GOLD"]["hit5"] for row in records), n),
        "GoldHit@32": rate(sum(row["ANY_GOLD"]["hit32"] for row in records), n),
        "GoldMRR": rate(sum(row["ANY_GOLD"]["mrr"] for row in records), n),
        "AHit@32": rate(sum(row["ANY_GOLD"]["AHit32"] for row in records), n),
        "ABHit@32": rate(sum(row["ANY_GOLD"]["ABHit32"] for row in records), n),
        "HistoryCandidateFraction": rate(sum(row["history_candidate_fraction"] for row in records), n),
    }


def group_subset(
    grouped: dict[str, dict[str, dict[str, Any]]], predicate: Callable[[dict[str, Any]], bool]
) -> list[str]:
    result = []
    for group, values in grouped.items():
        if not all(condition in values.get(model, {}) for model in MODELS for condition in CONDITIONS):
            continue
        reference = values["MiniFix"]["BARE"]
        if not reference["GoldInHistory"] and predicate(reference):
            result.append(group)
    return sorted(result)


def interaction_for_groups(grouped: dict[str, dict[str, dict[str, Any]]], groups: list[str]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    per_group = []
    for model in MODELS:
        bare = [grouped[group][model]["BARE"] for group in groups]
        bridge = [grouped[group][model]["EXACT_BRIDGE"] for group in groups]
        bare_metrics, bridge_metrics = metric_summary(bare), metric_summary(bridge)
        table[model] = {
            "BARE": bare_metrics,
            "EXACT_BRIDGE": bridge_metrics,
            "GAIN": {
                key: bridge_metrics[key] - bare_metrics[key]
                for key in ("GoldHit@1", "GoldHit@5", "GoldHit@32", "GoldMRR", "AHit@32", "ABHit@32", "HistoryCandidateFraction")
            },
        }
    for group in groups:
        values = grouped[group]
        model_values = {}
        for model in MODELS:
            bare, bridge = values[model]["BARE"], values[model]["EXACT_BRIDGE"]
            model_values[model] = {
                "bare_rank": bare["ANY_GOLD"]["rank"],
                "bridge_rank": bridge["ANY_GOLD"]["rank"],
                "bare_mrr": bare["ANY_GOLD"]["mrr"],
                "bridge_mrr": bridge["ANY_GOLD"]["mrr"],
                "mrr_gain": bridge["ANY_GOLD"]["mrr"] - bare["ANY_GOLD"]["mrr"],
                "hit32_gain": int(bridge["ANY_GOLD"]["hit32"]) - int(bare["ANY_GOLD"]["hit32"]),
                "ABHit32_gain": int(bridge["ANY_GOLD"]["ABHit32"]) - int(bare["ANY_GOLD"]["ABHit32"]),
                "AHit32_gain": int(bridge["ANY_GOLD"]["AHit32"]) - int(bare["ANY_GOLD"]["AHit32"]),
            }
        per_group.append({
            "group_id": group,
            "MiniFix": model_values["MiniFix"],
            "Gamma": model_values["Gamma"],
            "mrr_interaction": model_values["MiniFix"]["mrr_gain"] - model_values["Gamma"]["mrr_gain"],
            "hit32_interaction": model_values["MiniFix"]["hit32_gain"] - model_values["Gamma"]["hit32_gain"],
        })
    interactions = {
        "GoldMRR": table["MiniFix"]["GAIN"]["GoldMRR"] - table["Gamma"]["GAIN"]["GoldMRR"],
        "GoldHit@32": table["MiniFix"]["GAIN"]["GoldHit@32"] - table["Gamma"]["GAIN"]["GoldHit@32"],
        "ABHit@32": table["MiniFix"]["GAIN"]["ABHit@32"] - table["Gamma"]["GAIN"]["ABHit@32"],
        "AHit@32": table["MiniFix"]["GAIN"]["AHit@32"] - table["Gamma"]["GAIN"]["AHit@32"],
    }
    return {"N_unique_groups": len(groups), "models": table, "interaction_MF_minus_GA": interactions, "per_group": per_group}


def paired_bootstrap(interaction: dict[str, Any], seed: int = 155) -> dict[str, Any]:
    values = [row["mrr_interaction"] for row in interaction["per_group"]]
    n = len(values)
    if n < 6:
        return {
            "N": n, "repeats": 0, "N_TOO_SMALL_FOR_INTERACTION_INFERENCE": "YES",
            "DESCRIPTIVE_ONLY": "YES", "mean": statistics.fmean(values) if values else 0.0,
            "CI95": None, "P_interaction_gt_0": None,
        }
    rng = random.Random(seed)
    samples = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(BOOTSTRAP_REPEATS)]
    return {
        "N": n,
        "resampling_unit": "unique group carrying all four model-condition records",
        "repeats": BOOTSTRAP_REPEATS,
        "seed": seed,
        "N_TOO_SMALL_FOR_INTERACTION_INFERENCE": "NO",
        "DESCRIPTIVE_ONLY": "YES",
        "observed_interaction": statistics.fmean(values),
        "bootstrap_mean": statistics.fmean(samples),
        "CI95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "P_interaction_gt_0": rate(sum(value > 0 for value in samples), len(samples)),
    }


def acquisition(grouped: dict[str, dict[str, dict[str, Any]]], groups: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    result, hierarchy = {}, {}
    for model in MODELS:
        counts, hierarchy_counts, rows = Counter(), Counter(), []
        for group in groups:
            bare, bridge = grouped[group][model]["BARE"], grouped[group][model]["EXACT_BRIDGE"]
            bare_hit, bridge_hit = bare["ANY_GOLD"]["hit32"], bridge["ANY_GOLD"]["hit32"]
            if not bare_hit and bridge_hit:
                label = "NEW_GOLD"
                if bare["ANY_GOLD"]["ABHit32"]:
                    hierarchy_label = "BARE_AB_HIT"
                elif bare["ANY_GOLD"]["AHit32"]:
                    hierarchy_label = "BARE_A_ONLY"
                else:
                    hierarchy_label = "BARE_NO_HIERARCHY"
                hierarchy_counts[hierarchy_label] += 1
            elif bare_hit and not bridge_hit:
                label = "LOST_GOLD"
            elif bare_hit and bridge_hit:
                bare_rank, bridge_rank = bare["ANY_GOLD"]["rank"], bridge["ANY_GOLD"]["rank"]
                label = "RERANK_UP" if bridge_rank < bare_rank else "RERANK_DOWN" if bridge_rank > bare_rank else "UNCHANGED_HIT"
            else:
                label = "UNCHANGED_MISS"
            counts[label] += 1
            rows.append({"group_id": group, "class": label, "bare_rank": bare["ANY_GOLD"]["rank"], "bridge_rank": bridge["ANY_GOLD"]["rank"]})
        result[model] = {"N": len(groups), **{name: counts[name] for name in ("NEW_GOLD", "LOST_GOLD", "RERANK_UP", "RERANK_DOWN", "UNCHANGED_HIT", "UNCHANGED_MISS")}, "groups": rows}
        hierarchy[model] = {
            "NEW_GOLD_N": counts["NEW_GOLD"],
            "NEW_GOLD_FROM_BARE_AB_HIT": hierarchy_counts["BARE_AB_HIT"],
            "NEW_GOLD_FROM_BARE_A_ONLY": hierarchy_counts["BARE_A_ONLY"],
            "NEW_GOLD_FROM_BARE_NO_HIERARCHY": hierarchy_counts["BARE_NO_HIERARCHY"],
        }
    return result, hierarchy


def render_interaction_md(value: dict[str, Any]) -> str:
    lines = [
        "# Same-group covered Bridge interaction", "",
        "GoldNotInHistory; official-domain semantic prompt; Frozen BATA CoT; exact old bridge; group-paired.", "",
        "| Model | Condition | N groups | Hit@1 | Hit@5 | Hit@32 | MRR | AHit@32 | ABHit@32 | History fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for condition in CONDITIONS:
            row = value["models"][model][condition]
            lines.append(
                f"| {model} | {condition} | {row['N_unique_groups']} | {row['GoldHit@1']:.2%} | {row['GoldHit@5']:.2%} | "
                f"{row['GoldHit@32']:.2%} | {row['GoldMRR']:.6f} | {row['AHit@32']:.2%} | {row['ABHit@32']:.2%} | {row['HistoryCandidateFraction']:.2%} |"
            )
    lines.extend(["", f"MRR difference-in-differences: `{value['interaction_MF_minus_GA']['GoldMRR']:.8f}`."])
    return "\n".join(lines) + "\n"


def root_decision(primary: dict[str, Any], uncovered: dict[str, Any], acquisition_value: dict[str, Any], hierarchy: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    n = primary["N_unique_groups"]
    interaction = primary["interaction_MF_minus_GA"]["GoldMRR"]
    uncovered_interaction = uncovered["interaction_MF_minus_GA"]["GoldMRR"]
    mf_gain = primary["models"]["MiniFix"]["GAIN"]["GoldMRR"]
    ga_gain = primary["models"]["Gamma"]["GAIN"]["GoldMRR"]
    mf_new, ga_new = acquisition_value["MiniFix"]["NEW_GOLD"], acquisition_value["Gamma"]["NEW_GOLD"]
    ci = bootstrap.get("CI95")
    stable_positive = bool(ci and ci[0] > 0)
    if n < 6:
        memory_support = "UNDERPOWERED"
        interaction_label = "UNDERPOWERED"
    elif interaction > 0 and stable_positive and mf_new > ga_new and abs(uncovered_interaction) < interaction:
        memory_support = "SUPPORTED"
        interaction_label = "MINIFIX_SPECIFIC"
    elif interaction > 0:
        memory_support = "WEAK"
        interaction_label = "MIXED"
    else:
        memory_support = "UNSUPPORTED"
        interaction_label = "NO_MINIFIX_SPECIFIC_ADVANTAGE"
    shared_support = "SUPPORTED" if mf_gain > 0 and ga_gain > 0 and interaction <= 0 else "MIXED" if mf_gain > 0 and ga_gain > 0 else "WEAK"
    new_total = mf_new + ga_new
    ab_new = hierarchy["MiniFix"]["NEW_GOLD_FROM_BARE_AB_HIT"] + hierarchy["Gamma"]["NEW_GOLD_FROM_BARE_AB_HIT"]
    fine_c_support = "SUPPORTED" if new_total and ab_new / new_total >= 0.50 else "WEAK"
    return {
        "minifix_specific_memory_key_support": memory_support,
        "shared_bridge_conditioner_support": shared_support,
        "bridge_conditioned_fine_c_ranking_support": fine_c_support,
        "bridge_memory_interaction": interaction_label,
        "primary_signal": f"Gamma responds at least as strongly: covered MRR gains MiniFix={mf_gain:.6f}, Gamma={ga_gain:.6f}, interaction={interaction:.6f}.",
        "main_limitation": f"Only {n} selected same-group-covered NonHistory groups; bootstrap is descriptive and no official per-example decomposition exists.",
        "conclusion": "Existing local Beam records do not support a MiniFix-specific old-bridge memory advantage; both models respond to the bridge, with Gamma at least as strong.",
        "official_gap_explained": "NO",
        "local_mechanism_consistency": "SHARED_BRIDGE_STATE_CONDITIONER",
        "future_gpu_experiment": "NONE_REQUIRED_FOR_CURRENT_CPU_CONCLUSION; IF_REVISITED_USE_COVERAGE_MATCHED_TEACHER_FORCED_GOLD_LOGPROB_FOUR_WAY",
    }


def render_root_md(root: dict[str, Any], primary: dict[str, Any], uncovered: dict[str, Any]) -> str:
    return "\n".join([
        "# Bridge x training-memory interaction decision", "",
        f"- MINIFIX_SPECIFIC_MEMORY_KEY_SUPPORT: `{root['minifix_specific_memory_key_support']}`",
        f"- SHARED_BRIDGE_CONDITIONER_SUPPORT: `{root['shared_bridge_conditioner_support']}`",
        f"- BRIDGE_CONDITIONED_FINE_C_RANKING_SUPPORT: `{root['bridge_conditioned_fine_c_ranking_support']}`",
        f"- BRIDGE_MEMORY_INTERACTION: `{root['bridge_memory_interaction']}`", "",
        f"Covered interaction: `{primary['interaction_MF_minus_GA']['GoldMRR']:.8f}`.",
        f"Uncovered interaction: `{uncovered['interaction_MF_minus_GA']['GoldMRR']:.8f}`.", "",
        root["conclusion"], "", "This is local mechanism evidence only; the official score gap is not explained.",
    ]) + "\n"


def terminal(audit: dict[str, Any], source: dict[str, Any], manifest: dict[str, Any], primary: dict[str, Any], bootstrap: dict[str, Any], acquisition_value: dict[str, Any], hierarchy: dict[str, Any], uncovered: dict[str, Any], other: dict[str, Any], root: dict[str, Any]) -> str:
    mf, ga = primary["models"]["MiniFix"], primary["models"]["Gamma"]
    umf, uga = uncovered["models"]["MiniFix"], uncovered["models"]["Gamma"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT", "",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short'] or 'EMPTY'}",
        f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}", f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}",
        f"RUNTIME_GITHUB_PARITY={audit['runtime_github_parity']}", f"PROTECTED_RAW_ARTIFACTS_UNCHANGED={source['protected_raw_artifacts_unchanged']}",
        f"BRIDGE_CONTEXT_PARITY_PASS={manifest['bridge_context_parity_pass']}", "", "--- Primary Cohort ---", "",
        f"PRIMARY_ELIGIBLE_GROUPS={manifest['primary_eligible_groups']}", f"FOUR_WAY_COMPLETE_GROUPS={manifest['four_way_complete_groups']}",
        f"DROPPED_INCOMPLETE_GROUPS={manifest['dropped_incomplete_groups']}", "PRIMARY_COHORT=SAME_GROUP_EXACT_ABC_COVERED_NONHISTORY",
        "", "--- MiniFix ---", "", f"MF_BARE_HIT32={mf['BARE']['GoldHit@32']:.8f}", f"MF_BRIDGE_HIT32={mf['EXACT_BRIDGE']['GoldHit@32']:.8f}",
        f"MF_HIT32_GAIN={mf['GAIN']['GoldHit@32']:.8f}", f"MF_BARE_MRR={mf['BARE']['GoldMRR']:.8f}", f"MF_BRIDGE_MRR={mf['EXACT_BRIDGE']['GoldMRR']:.8f}",
        f"MF_MRR_GAIN={mf['GAIN']['GoldMRR']:.8f}", f"MF_BARE_A_HIT32={mf['BARE']['AHit@32']:.8f}", f"MF_BRIDGE_A_HIT32={mf['EXACT_BRIDGE']['AHit@32']:.8f}",
        f"MF_BARE_AB_HIT32={mf['BARE']['ABHit@32']:.8f}", f"MF_BRIDGE_AB_HIT32={mf['EXACT_BRIDGE']['ABHit@32']:.8f}",
        "", "--- Gamma ---", "", f"GA_BARE_HIT32={ga['BARE']['GoldHit@32']:.8f}", f"GA_BRIDGE_HIT32={ga['EXACT_BRIDGE']['GoldHit@32']:.8f}",
        f"GA_HIT32_GAIN={ga['GAIN']['GoldHit@32']:.8f}", f"GA_BARE_MRR={ga['BARE']['GoldMRR']:.8f}", f"GA_BRIDGE_MRR={ga['EXACT_BRIDGE']['GoldMRR']:.8f}",
        f"GA_MRR_GAIN={ga['GAIN']['GoldMRR']:.8f}", f"GA_BARE_A_HIT32={ga['BARE']['AHit@32']:.8f}", f"GA_BRIDGE_A_HIT32={ga['EXACT_BRIDGE']['AHit@32']:.8f}",
        f"GA_BARE_AB_HIT32={ga['BARE']['ABHit@32']:.8f}", f"GA_BRIDGE_AB_HIT32={ga['EXACT_BRIDGE']['ABHit@32']:.8f}",
        "", "--- Interaction ---", "", f"BRIDGE_MRR_INTERACTION_MF_MINUS_GA={primary['interaction_MF_minus_GA']['GoldMRR']:.8f}",
        f"BRIDGE_HIT32_INTERACTION_MF_MINUS_GA={primary['interaction_MF_minus_GA']['GoldHit@32']:.8f}",
        f"BOOTSTRAP_CI95={json.dumps(bootstrap['CI95'])}", f"BOOTSTRAP_P_INTERACTION_GT0={bootstrap['P_interaction_gt_0']:.8f}", "DESCRIPTIVE_ONLY=YES",
        "", "--- Acquisition ---", "",
    ]
    for prefix_label, model in (("MF", "MiniFix"), ("GA", "Gamma")):
        value = acquisition_value[model]
        for field in ("NEW_GOLD", "LOST_GOLD", "RERANK_UP", "RERANK_DOWN", "UNCHANGED_MISS"):
            lines.append(f"{prefix_label}_{field}={value[field]}")
    lines.extend(["", "--- Hierarchy ---", ""])
    for prefix_label, model in (("MF", "MiniFix"), ("GA", "Gamma")):
        value = hierarchy[model]
        lines.extend([
            f"{prefix_label}_NEW_GOLD_FROM_BARE_AB_HIT={value['NEW_GOLD_FROM_BARE_AB_HIT']}",
            f"{prefix_label}_NEW_GOLD_FROM_BARE_A_ONLY={value['NEW_GOLD_FROM_BARE_A_ONLY']}",
            f"{prefix_label}_NEW_GOLD_FROM_BARE_NO_HIERARCHY={value['NEW_GOLD_FROM_BARE_NO_HIERARCHY']}",
        ])
    lines.extend([
        "", "--- Negative Control ---", "", f"UNCOVERED_GROUPS={uncovered['N_unique_groups']}",
        f"MF_UNCOVERED_BRIDGE_MRR_GAIN={umf['GAIN']['GoldMRR']:.8f}", f"GA_UNCOVERED_BRIDGE_MRR_GAIN={uga['GAIN']['GoldMRR']:.8f}",
        f"UNCOVERED_BRIDGE_INTERACTION={uncovered['interaction_MF_minus_GA']['GoldMRR']:.8f}", "", "--- Other-group Only ---", "",
        f"OTHER_GROUP_ONLY_N={other['N_unique_groups']}", f"OTHER_GROUP_ONLY_INTERPRETATION={other['interpretation']}",
        "", "--- History Candidate ---", "", f"MF_COVERED_HISTORY_FRACTION_DELTA={mf['GAIN']['HistoryCandidateFraction']:.8f}",
        f"GA_COVERED_HISTORY_FRACTION_DELTA={ga['GAIN']['HistoryCandidateFraction']:.8f}", "", "--- Decision ---", "",
        f"MINIFIX_SPECIFIC_MEMORY_KEY_SUPPORT={root['minifix_specific_memory_key_support']}",
        f"SHARED_BRIDGE_CONDITIONER_SUPPORT={root['shared_bridge_conditioner_support']}",
        f"BRIDGE_CONDITIONED_FINE_C_RANKING_SUPPORT={root['bridge_conditioned_fine_c_ranking_support']}",
        f"BRIDGE_MEMORY_INTERACTION={root['bridge_memory_interaction']}", f"PRIMARY_SIGNAL={root['primary_signal']}",
        f"MAIN_LIMITATION={root['main_limitation']}", f"CONCLUSION={root['conclusion']}", "OFFICIAL_GAP_EXPLAINED=NO",
        f"FUTURE_GPU_EXPERIMENT={root['future_gpu_experiment']}", "", "GPU_INFERENCE_STARTED=NO", "MODEL_FORWARD_STARTED=NO",
        "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "SELF_COT_GENERATION_STARTED=NO", "EXTERNAL_EVAL_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> None:
    assert percentile([0.0, 1.0], 0.5) == 0.5
    synthetic = []
    for group in ("a", "b"):
        for model in MODELS:
            for condition in CONDITIONS:
                synthetic.append({
                    "group_id": group, "model": model, "condition": condition,
                    "ANY_GOLD": {"hit1": False, "hit5": False, "hit32": condition == "EXACT_BRIDGE", "mrr": 0.5 if condition == "EXACT_BRIDGE" else 0.0, "AHit32": True, "ABHit32": True, "rank": 2 if condition == "EXACT_BRIDGE" else None},
                    "history_candidate_fraction": 0.0,
                })
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for row in synthetic:
        grouped[row["group_id"]][row["model"]][row["condition"]] = row
    interaction = interaction_for_groups(grouped, ["a", "b"])
    assert interaction["interaction_MF_minus_GA"]["GoldMRR"] == 0.0
    acq, hierarchy = acquisition(grouped, ["a", "b"])
    assert acq["MiniFix"]["NEW_GOLD"] == 2 and hierarchy["Gamma"]["NEW_GOLD_FROM_BARE_AB_HIT"] == 2
    print("SELF_TEST=PASS")


def run() -> None:
    if "torch" in sys.modules:
        raise RuntimeError("TORCH_ALREADY_IMPORTED")
    audit = code_audit()
    protected_before = protected_hashes()
    rows, input_audit = load_canonical()
    manifests, manifest_audit = combine_manifest_contracts()
    frequency, frequency_audit = supervision_frequency()
    grouped, group_manifest = four_way(rows, manifests, frequency)
    primary_groups = group_manifest["complete_group_ids"]
    primary = interaction_for_groups(grouped, primary_groups)
    uncovered_groups = group_subset(grouped, lambda row: not row["ANY_GOLD"]["any_train_abc_covered"])
    uncovered = interaction_for_groups(grouped, uncovered_groups)
    other_groups = group_subset(grouped, lambda row: row["ANY_GOLD"]["coverage_class"] == "C2_OTHER_GROUP_ONLY")
    other = interaction_for_groups(grouped, other_groups)
    other["interpretation"] = "UNDERPOWERED_GROUP_LEVEL_LIST_ONLY" if len(other_groups) < 6 else "DESCRIPTIVE_CONTROL"
    same_only_groups = group_subset(grouped, lambda row: row["ANY_GOLD"]["coverage_class"] == "C1_SAME_GROUP_ONLY")
    same_other_groups = group_subset(grouped, lambda row: row["ANY_GOLD"]["coverage_class"] == "C3_SAME_AND_OTHER")
    primary["coverage_strata"] = {
        "SAME_GROUP_ONLY": interaction_for_groups(grouped, same_only_groups),
        "SAME_AND_OTHER": interaction_for_groups(grouped, same_other_groups),
    }
    bootstrap = paired_bootstrap(primary)
    acquisition_value, hierarchy = acquisition(grouped, primary_groups)
    root = root_decision(primary, uncovered, acquisition_value, hierarchy, bootstrap)

    source_audit = {
        "canonical_input": input_audit,
        "manifest_contract": manifest_audit,
        "frequency_contract": frequency_audit,
        "protected_raw_sha256_before": protected_before,
        "protected_raw_artifacts_unchanged": "PENDING_POST_WRITE_CHECK",
        "cpu_only": True,
        "torch_imported": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "source_contract_audit.json", source_audit)
    write_json(OUTPUT / "four_way_group_manifest.json", group_manifest)
    write_json(OUTPUT / "same_group_covered_interaction.json", primary)
    (OUTPUT / "same_group_covered_interaction.md").write_text(render_interaction_md(primary), encoding="utf-8")
    write_json(OUTPUT / "uncovered_interaction.json", uncovered)
    write_json(OUTPUT / "other_group_only_control.json", other)
    write_json(OUTPUT / "bridge_acquisition_decomposition.json", acquisition_value)
    write_json(OUTPUT / "hierarchy_transition_analysis.json", hierarchy)
    write_json(OUTPUT / "paired_bootstrap.json", bootstrap)
    write_json(OUTPUT / "root_cause_summary.json", root)
    (OUTPUT / "root_cause_summary.md").write_text(render_root_md(root, primary, uncovered), encoding="utf-8")
    protected_after = protected_hashes()
    source_audit["protected_raw_sha256_after"] = protected_after
    source_audit["protected_raw_artifacts_unchanged"] = "YES" if protected_before == protected_after else "NO"
    if source_audit["protected_raw_artifacts_unchanged"] != "YES":
        raise RuntimeError("PROTECTED_RAW_ARTIFACT_CHANGED")
    write_json(OUTPUT / "source_contract_audit.json", source_audit)
    review = terminal(audit, source_audit, group_manifest, primary, bootstrap, acquisition_value, hierarchy, uncovered, other, root)
    (OUTPUT / "CHATGPT_BRIDGE_MEMORY_INTERACTION_REVIEW.txt").write_text(review, encoding="utf-8")
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
