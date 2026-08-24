"""Phase 1.4 natural-proxy audit and conditional bounded No-Think probe."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any

import torch

RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
RELATIVE_SCRIPT = Path("baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/recommendation_root_cause_phase1_4.py")
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_root_cause_phase1_4.py"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]

from boundary_adapt.bridge_inside_sft.build_dataset import load_unique_source  # noqa: E402
from boundary_adapt.bridge_inside_sft.common import (  # noqa: E402
    BASE, DOMAIN, DOMAIN_ORDER, SOURCE, SOURCE_SHA256, file_sha, infer_domain, source_metadata,
)
from boundary_adapt.diagnostics.controlled_generator_decoder_crossover import load_model, strict_beam  # noqa: E402
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import history_from_prompt  # noqa: E402
from boundary_adapt.diagnostics.recommendation_official_prompt_bare_crossover import render_prompt_ids  # noqa: E402
from boundary_adapt.diagnostics.recommendation_root_cause_phase1_1 import build_manifold, sid  # noqa: E402

OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_4"
PHASE10 = RUNTIME / "boundary_adapt/results/recommendation_decoder_diagnostic_v1_40g"
PHASE12 = RUNTIME / "boundary_adapt/results/recommendation_official_prompt_bare_crossover_40g"
PHASE13 = RUNTIME / "boundary_adapt/results/recommendation_self_cot_crossover_20g"
SPLIT_MANIFEST = Path("/root/GRPO_audit_results/bridge_inside_transition_sft_v1_20260824/split_manifest.json")
CANONICAL_ROWS = RUNTIME / "boundary_adapt/results/boundary_adapt_rows.jsonl"
MODELS = ("Beta", "Step900")
MODEL_PATHS = {
    "Beta": Path("/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"),
    "Step900": Path("/root/data_checkpoints_backup_20260824/outputs/boundary_adapt/continuation_from300_to1500/checkpoint-900"),
}
EXPECTED_ADAPTER_SHA = {
    "Beta": "4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3",
    "Step900": "ea395df041ef7bc13d5e5e54aec7a0d376b8f6ae4ae064eb6a7d89dd3c7cdffa",
}
ARTIFACT_ROOTS = (Path("/output/merged"), Path("/wanqing-develop/data/all"), Path("/wanqing-models/data/all"))
SID_RE = __import__("re").compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
CUTS = (1, 5, 10, 32)
METRICS = ("hit32", "mrr", "ab32", "a32", "history_fraction")
BOOTSTRAP_SEED = 20260825


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True).strip()


def code_audit() -> dict[str, Any]:
    commit = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    status = git("status", "--short")
    source_script = SOURCE_REPO / RELATIVE_SCRIPT
    source_sha = file_sha(source_script)
    runtime_sha = file_sha(RUNTIME_SCRIPT)
    result = {
        "implement_commit": commit, "origin_main": origin, "push_status": "PASS" if commit == origin else "FAIL",
        "git_status_short": status, "github_script_sha256": source_sha,
        "runtime_script_sha256": runtime_sha, "runtime_github_parity": "PASS" if source_sha == runtime_sha else "FAIL",
    }
    if result["push_status"] != "PASS" or status or result["runtime_github_parity"] != "PASS":
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={result}")
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def stratum(row: dict[str, Any]) -> str:
    history = "HISTORY" if row["gold_sid_in_history"] else "NONHISTORY"
    k = "K1" if int(row["K"]) == 1 else "K2PLUS"
    return f"{row['domain']}|{history}|{k}"


def natural_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if file_sha(SOURCE) != SOURCE_SHA256 or split["source_sha256"] != SOURCE_SHA256:
        raise RuntimeError("SOURCE_SHA256_MISMATCH")
    train_ids, holdout_ids = set(split["train_group_ids"]), set(split["holdout_group_ids"])
    if len(train_ids) != 14348 or len(holdout_ids) != 1595 or train_ids & holdout_ids:
        raise RuntimeError("SPLIT_ID_CONTRACT_FAIL")
    raw_rows, duplicates = load_unique_source(SOURCE, CANONICAL_ROWS)
    selected = []
    for raw in raw_rows:
        metadata = source_metadata(raw)
        group = str(metadata["recommendation_group_id"])
        if group not in holdout_ids:
            continue
        domain = infer_domain(metadata)
        golds = {str(value) for value in metadata["recommendation_all_gold_sids"]}
        prompt = str(raw.get("instruction", "")) + str(raw.get("input", ""))
        history = [match.group(0) for match in SID_RE.finditer(prompt) if match.group(1) == domain]
        unique_history = list(dict.fromkeys(history))
        selected.append({
            "group_id": group, "domain": domain, "K": len(golds),
            "gold_sid_in_history": bool(golds.intersection(history)),
            "target_domain_history_sid_count": len(history),
            "target_domain_unique_sid_count": len(unique_history),
            "target_domain_duplicate_count": len(history) - len(unique_history),
            "any_target_history_sid_is_gold": bool(golds.intersection(history)),
            "last_target_history_sid_is_gold": bool(history and history[-1] in golds),
        })
    if len(selected) != 1595 or {row["group_id"] for row in selected} != holdout_ids:
        raise RuntimeError("HOLDOUT_RECONSTRUCTION_FAIL")
    return selected, {"source_streaming_passes": 1, "legacy_duplicate_rows_dropped": duplicates,
                      "train_groups": len(train_ids), "holdout_groups": len(holdout_ids),
                      "train_holdout_intersection": 0}


def summarize_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    history_n = sum(row["gold_sid_in_history"] for row in rows)
    k2_n = sum(row["K"] >= 2 for row in rows)
    counts = Counter(stratum(row).split("|", 1)[1] for row in rows)
    history_lengths = [row.get("target_domain_history_sid_count", 0) for row in rows]
    unique_lengths = [row.get("target_domain_unique_sid_count", 0) for row in rows]
    occurrences = sum(history_lengths)
    duplicates = sum(row.get("target_domain_duplicate_count", 0) for row in rows)
    return {
        "N": n, "GoldSIDInHistory": history_n, "GoldSIDNotInHistory": n - history_n,
        "GoldSIDInHistoryRate": history_n / n, "K1": n - k2_n, "K2PLUS": k2_n,
        "K2PLUSRate": k2_n / n,
        "bucket_counts": {key: counts[key] for key in ("HISTORY|K1", "HISTORY|K2PLUS", "NONHISTORY|K1", "NONHISTORY|K2PLUS")},
        "bucket_proportions": {key: counts[key] / n for key in ("HISTORY|K1", "HISTORY|K2PLUS", "NONHISTORY|K1", "NONHISTORY|K2PLUS")},
        "target_domain_history_sid_count": {"mean": statistics.fmean(history_lengths), "median": statistics.median(history_lengths),
            "p25": percentile(history_lengths, .25), "p75": percentile(history_lengths, .75), "p90": percentile(history_lengths, .90)},
        "target_domain_unique_sid_count": {"mean": statistics.fmean(unique_lengths), "median": statistics.median(unique_lengths)},
        "target_domain_duplicate_fraction": 0.0 if not occurrences else duplicates / occurrences,
        "target_domain_duplicate_fraction_definition": "duplicate occurrences / all target-domain history SID occurrences",
        "ANY_TARGET_HISTORY_SID_IS_GOLD_RATE": statistics.fmean(row.get("any_target_history_sid_is_gold", row["gold_sid_in_history"]) for row in rows),
        "LAST_TARGET_HISTORY_SID_IS_GOLD_RATE": statistics.fmean(row.get("last_target_history_sid_is_gold", False) for row in rows),
    }


def diagnostic_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest40 = json.loads((PHASE10 / "manifest_40.json").read_text(encoding="utf-8"))["items"]
    lookup = {row["group_id"]: row for row in manifest40}
    def convert(row: dict[str, Any]) -> dict[str, Any]:
        source = lookup[row["group_id"]]
        history = [sid(value) for value in source["history_target_domain_sids"]]
        golds = {sid(value) for value in source["all_gold_sids"]}
        return {"group_id": row["group_id"], "domain": row["domain"], "K": row["K"],
                "gold_sid_in_history": row["gold_sid_in_history"],
                "target_domain_history_sid_count": len(history),
                "target_domain_unique_sid_count": len(set(history)),
                "target_domain_duplicate_count": len(history) - len(set(history)),
                "any_target_history_sid_is_gold": bool(golds.intersection(history)),
                "last_target_history_sid_is_gold": bool(history and history[-1] in golds)}
    rows40 = [convert(row) for row in manifest40]
    rows20 = [convert(row) for row in json.loads((PHASE13 / "manifest_20.json").read_text(encoding="utf-8"))["items"]]
    return rows40, rows20


def distribution_payload(natural: list[dict[str, Any]], rows40: list[dict[str, Any]], rows20: list[dict[str, Any]], provenance) -> tuple[dict, dict]:
    def complete(rows):
        return {"overall": summarize_distribution(rows),
                "per_domain": {domain: summarize_distribution([row for row in rows if row["domain"] == domain]) for domain in DOMAIN_ORDER}}
    stats = {"name": "ADAPTATION_HELDOUT_NATURAL_PROXY", "not_official_test_distribution": True,
             "provenance": provenance, "overall": complete(natural)["overall"], "per_domain": complete(natural)["per_domain"]}
    populations = {"natural": complete(natural), "probe40": complete(rows40), "probe20": complete(rows20)}
    comparisons, max_difference = {}, 0.0
    for scope in ("overall", *DOMAIN_ORDER):
        values = {name: payload["overall"] if scope == "overall" else payload["per_domain"][scope] for name, payload in populations.items()}
        metrics = ("GoldSIDInHistoryRate", "K2PLUSRate")
        comparison = {}
        for metric in metrics:
            comparison[metric] = {"natural": values["natural"][metric], "probe40": values["probe40"][metric], "probe20": values["probe20"][metric],
                                  "40G_MINUS_NATURAL": values["probe40"][metric] - values["natural"][metric],
                                  "20G_MINUS_NATURAL": values["probe20"][metric] - values["natural"][metric]}
            max_difference = max(max_difference, abs(comparison[metric]["40G_MINUS_NATURAL"]), abs(comparison[metric]["20G_MINUS_NATURAL"]))
        comparison["bucket_proportions"] = {}
        for bucket in values["natural"]["bucket_proportions"]:
            n, p40, p20 = (values[name]["bucket_proportions"][bucket] for name in ("natural", "probe40", "probe20"))
            comparison["bucket_proportions"][bucket] = {"natural": n, "probe40": p40, "probe20": p20, "40G_MINUS_NATURAL": p40 - n, "20G_MINUS_NATURAL": p20 - n}
            max_difference = max(max_difference, abs(p40 - n), abs(p20 - n))
        comparisons[scope] = comparison
    classification = "LARGE" if max_difference > .15 else "MODERATE" if max_difference >= .05 else "SMALL"
    shift = {"populations": populations, "comparisons": comparisons, "max_absolute_core_difference": max_difference,
             "classification_rule": "SMALL iff all core absolute differences <5pp; MODERATE iff maximum is 5-15pp; LARGE iff any exceeds 15pp",
             "diagnostic_distribution_shift": classification}
    return stats, shift


def row_metrics(row: dict[str, Any], item: dict[str, Any]) -> dict[str, float]:
    predictions = [sid(beam["predicted_sid"]) if beam["predicted_sid"] else None for beam in row["beams"]]
    golds = [sid(value) for value in item["all_gold_sids"]]
    def rank(prefix: int):
        return next((index for index, value in enumerate(predictions, 1) if value is not None and any(value[:prefix] == gold[:prefix] for gold in golds)), None)
    exact, ab, a = rank(4), rank(3), rank(2)
    return {"hit1": float(exact is not None and exact <= 1), "hit5": float(exact is not None and exact <= 5),
            "hit10": float(exact is not None and exact <= 10), "hit32": float(exact is not None),
            "mrr": 0.0 if exact is None else 1.0 / exact, "ab32": float(ab is not None), "a32": float(a is not None),
            "history_fraction": sum(beam["copy_class"] == "EXACT_COPY" for beam in row["beams"]) / 32,
            "top1_is_history": float(row["beams"][0]["copy_class"] == "EXACT_COPY"),
            "unique_a": len({value[:2] for value in predictions if value is not None}),
            "unique_ab": len({value[:3] for value in predictions if value is not None}),
            "unique_abc": len({value for value in predictions if value is not None})}


def natural_counts(rows: list[dict[str, Any]]) -> Counter:
    return Counter(stratum(row) for row in rows)


def poststratify(cell_rows: list[dict[str, Any]], items: dict[str, dict[str, Any]], counts: Counter) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in cell_rows:
        item = items[row["group_id"]]
        grouped[stratum(item)].append(row_metrics(row, item))
    unsupported = sorted(key for key, count in counts.items() if count and key not in grouped)
    total = sum(counts.values())
    covered = sum(counts[key] for key in grouped)
    def weighted(keys: list[str]) -> dict[str, float | None]:
        denominator = sum(counts[key] for key in keys if key in grouped)
        if not denominator:
            return {metric: None for metric in METRICS}
        return {metric: sum(counts[key] * statistics.fmean(value[metric] for value in grouped[key]) for key in keys if key in grouped) / denominator for metric in METRICS}
    per_domain = {}
    for domain in DOMAIN_ORDER:
        keys = [key for key in counts if key.startswith(domain + "|")]
        domain_total = sum(counts[key] for key in keys)
        domain_covered = sum(counts[key] for key in keys if key in grouped)
        per_domain[domain] = {"metrics": weighted(keys), "covered_natural_mass": domain_covered / domain_total,
                              "unsupported_strata": [key for key in keys if key not in grouped]}
    source = weighted(list(counts))
    equal_macro = {metric: statistics.fmean(per_domain[domain]["metrics"][metric] for domain in DOMAIN_ORDER if per_domain[domain]["metrics"][metric] is not None) for metric in METRICS}
    return {"source_frequency_weighted": source, "equal_domain_macro": equal_macro, "per_domain_weighted": per_domain,
            "covered_natural_mass": covered / total, "coverage_status": "LOW_COVERAGE" if covered / total < .90 else "ADEQUATE_COVERAGE",
            "unsupported_strata": unsupported, "stratum_sample_counts": {key: len(values) for key, values in grouped.items()},
            "per_group": {row["group_id"]: row_metrics(row, items[row["group_id"]]) for row in cell_rows}}


def weighted_bootstrap(cells: dict[str, list[dict[str, Any]]], items, counts: Counter) -> dict[str, Any]:
    by_cell = {cell: {row["group_id"]: row_metrics(row, items[row["group_id"]]) for row in rows} for cell, rows in cells.items()}
    groups: dict[str, list[str]] = defaultdict(list)
    for group, item in items.items():
        if all(group in by_cell[cell] for cell in cells):
            groups[stratum(item)].append(group)
    supported = [key for key in counts if groups[key]]
    denominator = sum(counts[key] for key in supported)
    def estimate(sampled: dict[str, list[str]]) -> float:
        def cell_value(cell):
            return sum(counts[key] * statistics.fmean(by_cell[cell][group]["mrr"] for group in sampled[key]) for key in supported) / denominator
        return (cell_value("G9_D9") - cell_value("G9_DB")) - (cell_value("GB_D9") - cell_value("GB_DB"))
    observed = estimate(groups)
    rng, samples = random.Random(BOOTSTRAP_SEED), []
    for _ in range(1000):
        sampled = {key: [rng.choice(groups[key]) for _ in groups[key]] for key in supported}
        samples.append(estimate(sampled))
    return {"estimate": observed, "ci95": [percentile(samples, .025), percentile(samples, .975)],
            "replicates": 1000, "seed": BOOTSTRAP_SEED, "method": "paired group resampling within observed strata plus fixed natural post-stratification weights",
            "covered_natural_mass": denominator / sum(counts.values()), "coverage_status": "LOW_COVERAGE" if denominator / sum(counts.values()) < .90 else "ADEQUATE_COVERAGE"}


def existing_reweighting(natural: list[dict[str, Any]], rows40, rows20) -> dict[str, Any]:
    counts = natural_counts(natural)
    items40 = {row["group_id"]: row for row in json.loads((PHASE12 / "prompt_audit.json").read_text(encoding="utf-8"))["items"]}
    official = read_jsonl(PHASE12 / "records.jsonl")
    phase12 = {model: poststratify([row for row in official if row["model"] == model], items40, counts) for model in MODELS}
    items20 = {row["group_id"]: row for row in json.loads((PHASE13 / "manifest_20.json").read_text(encoding="utf-8"))["items"]}
    decoder = read_jsonl(PHASE13 / "decoder_records.jsonl")
    cell_specs = {"GB_DB": ("Beta", "Beta"), "GB_D9": ("Beta", "Step900"), "G9_DB": ("Step900", "Beta"), "G9_D9": ("Step900", "Step900")}
    raw_cells = {key: [row for row in decoder if row["generator"] == generator and row["decoder"] == model] for key, (generator, model) in cell_specs.items()}
    phase13 = {key: poststratify(rows, items20, counts) for key, rows in raw_cells.items()}
    interaction = weighted_bootstrap(raw_cells, items20, counts)
    def order(a, b):
        return "Beta > Step900" if a > b else "Step900 > Beta" if b > a else "Beta = Step900"
    p12_beta, p12_step = (phase12[model]["source_frequency_weighted"]["mrr"] for model in MODELS)
    p13_beta, p13_step = (phase13[key]["source_frequency_weighted"]["mrr"] for key in ("GB_DB", "G9_D9"))
    return {"natural_stratum_counts": dict(counts), "phase12": phase12, "phase13": phase13,
            "phase13_weighted_interaction_mrr": interaction,
            "phase12_weighted_mrr_order": order(p12_beta, p12_step), "phase13_weighted_own_pair_order": order(p13_beta, p13_step)}


def bounded_artifact_check() -> dict[str, Any]:
    candidates = []
    for root in ARTIFACT_ROOTS:
        if not root.is_dir():
            continue
        patterns = ("challenge_recommendation_*/test_generated.json", "challenge_recommendation_*/test_generated.json.debug", "sid2pid*.json", "*sid2pid*.json")
        for pattern in patterns:
            candidates.extend(Path(value) for value in glob.glob(str(root / pattern)))
    artifacts = []
    for path in sorted(set(candidates)):
        schema: dict[str, Any]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            schema = {"type": type(value).__name__, "first_level_keys": list(value)[:50] if isinstance(value, dict) else None,
                      "length": len(value) if isinstance(value, (dict, list)) else None}
        except Exception as error:
            schema = {"parse_error": type(error).__name__}
        artifacts.append({"path": str(path), "size": path.stat().st_size, "sha256": file_sha(path), "schema": schema})
    return {"official_artifact_available": bool(artifacts), "artifacts": artifacts,
            "roots_checked": [{"path": str(root), "exists": root.is_dir()} for root in ARTIFACT_ROOTS],
            "patterns_checked": ["challenge_recommendation_*/test_generated.json", "challenge_recommendation_*/test_generated.json.debug", "sid2pid*.json", "*sid2pid*.json"],
            "exact_log_paths_checked": [], "bounded_lookup": True, "recursive_disk_scan": False,
            "gpu_gate": "FAIL_OFFICIAL_ARTIFACT_FOUND" if artifacts else "PASS_PENDING_CPU_AND_CODE_GATES"}


def render_distribution_md(stats, shift) -> str:
    lines = ["# Adaptation Heldout Natural Proxy", "", "This is not the official test distribution.", "", "| Scope | N | Gold in history | K2+ | History SID mean | Duplicate fraction |", "|---|---:|---:|---:|---:|---:|"]
    for scope, value in (("overall", stats["overall"]), *((domain, stats["per_domain"][domain]) for domain in DOMAIN_ORDER)):
        lines.append(f"| {scope} | {value['N']} | {value['GoldSIDInHistoryRate']:.3f} | {value['K2PLUSRate']:.3f} | {value['target_domain_history_sid_count']['mean']:.2f} | {value['target_domain_duplicate_fraction']:.3f} |")
    lines += ["", f"Distribution shift classification: **{shift['diagnostic_distribution_shift']}**.", "", shift["classification_rule"]]
    return "\n".join(lines) + "\n"


def render_post_md(result) -> str:
    lines = ["# Existing Results: Natural-Proxy Post-Stratification", "", "Estimates are conditional on covered natural strata and remain diagnostic.", "", "| Cell | Weighted MRR | Hit@32 | Coverage | Status |", "|---|---:|---:|---:|---|"]
    for name, value in (("Phase12 Beta", result["phase12"]["Beta"]), ("Phase12 Step900", result["phase12"]["Step900"]), *((f"Phase13 {key}", result["phase13"][key]) for key in ("GB_DB", "GB_D9", "G9_DB", "G9_D9"))):
        metrics = value["source_frequency_weighted"]
        lines.append(f"| {name} | {metrics['mrr']:.6f} | {metrics['hit32']:.4f} | {value['covered_natural_mass']:.3f} | {value['coverage_status']} |")
    interaction = result["phase13_weighted_interaction_mrr"]
    lines += ["", f"Weighted interaction MRR: {interaction['estimate']:.6f}; bootstrap 95% CI {interaction['ci95']}."]
    return "\n".join(lines) + "\n"


def run_cpu() -> None:
    audit = code_audit()
    natural, provenance = natural_rows()
    rows40, rows20 = diagnostic_rows()
    stats, shift = distribution_payload(natural, rows40, rows20, provenance)
    reweighted = existing_reweighting(natural, rows40, rows20)
    artifact = bounded_artifact_check()
    write_json(OUTPUT / "implementation_audit.json", audit)
    write_json(OUTPUT / "natural_distribution_stats.json", stats)
    (OUTPUT / "natural_distribution_stats.md").write_text(render_distribution_md(stats, shift), encoding="utf-8")
    write_json(OUTPUT / "diagnostic_distribution_shift.json", shift)
    write_json(OUTPUT / "poststratified_existing_metrics.json", reweighted)
    (OUTPUT / "poststratified_existing_metrics.md").write_text(render_post_md(reweighted), encoding="utf-8")
    write_json(OUTPUT / "bounded_official_artifact_check.json", artifact)
    print(f"NATURAL_PROXY_GROUPS={stats['overall']['N']}")
    print(f"DIAGNOSTIC_DISTRIBUTION_SHIFT={shift['diagnostic_distribution_shift']}")
    print(f"OFFICIAL_ARTIFACT_AVAILABLE={'YES' if artifact['official_artifact_available'] else 'NO'}")
    print(f"GPU_GATE={'FAIL_OFFICIAL_ARTIFACT_FOUND' if artifact['official_artifact_available'] else 'PASS'}")


def prepare_nothink() -> None:
    audit = code_audit()
    artifact = json.loads((OUTPUT / "bounded_official_artifact_check.json").read_text(encoding="utf-8"))
    if artifact["official_artifact_available"]:
        raise RuntimeError("GPU_GATE_FAIL_OFFICIAL_ARTIFACT_FOUND")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    phase12 = json.loads((PHASE12 / "prompt_audit.json").read_text(encoding="utf-8"))
    items, gates = [], []
    for row in phase12["items"]:
        old_user = row["new_user"]
        if not old_user.endswith("/think"):
            raise RuntimeError("PHASE12_USER_SUFFIX_FAIL")
        new_user = old_user[:-len("/think")] + "/no_think"
        old_sids = [match.group(0) for match in SID_RE.finditer(old_user)]
        new_sids = [match.group(0) for match in SID_RE.finditer(new_user)]
        old_tokens = [tokenizer.encode(value, add_special_tokens=False) for value in old_sids]
        new_tokens = [tokenizer.encode(value, add_special_tokens=False) for value in new_sids]
        gate = {"group_id": row["group_id"], "domain": row["domain"], "group_id_unchanged": True,
                "gold_unchanged": True, "history_sid_sequence_parity": old_sids == new_sids,
                "history_sid_token_parity": old_tokens == new_tokens, "only_suffix_changed": old_user[:-len("/think")] == new_user[:-len("/no_think")]}
        gates.append(gate)
        items.append({**row, "think_user": old_user, "nothink_user": new_user,
                      "nothink_prompt_token_ids": render_prompt_ids(tokenizer, new_user, row["new_system"])})
    prompt_pass = all(all(gate.values()) for gate in gates)
    if not prompt_pass:
        raise RuntimeError("NO_THINK_PROMPT_AUDIT_FAIL")
    payload = {"implement_commit": audit["implement_commit"], "groups": 40, "sample_audit_count": 8,
               "no_think_renderer": "CONTROLLED_PROXY", "exact_official_route": False,
               "renderer_detail": "Phase1.2 qwen3_nothink renderer with only terminal /think -> /no_think",
               "no_think_prompt_audit_pass": True, "history_sid_sequence_parity": True,
               "history_sid_token_parity": True, "sample_gates": [gate for domain in DOMAIN_ORDER for gate in [value for value in gates if value["domain"] == domain][:2]],
               "items": items}
    write_json(OUTPUT / "nothink_prompt_audit.json", payload)
    print("NO_THINK_PROMPT_AUDIT_PASS=YES\nHISTORY_SID_SEQUENCE_PARITY=PASS\nHISTORY_SID_TOKEN_PARITY=PASS")


def setup_rank() -> tuple[int, int, str]:
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("NOTHINK_REQUIRES_4_GPU_RANKS")
    torch.cuda.set_device(rank)
    return rank, world, f"cuda:{rank}"


def decode_nothink(model_label: str) -> None:
    code_audit()
    artifact = json.loads((OUTPUT / "bounded_official_artifact_check.json").read_text(encoding="utf-8"))
    prompt_audit = json.loads((OUTPUT / "nothink_prompt_audit.json").read_text(encoding="utf-8"))
    if artifact["official_artifact_available"] or not prompt_audit["no_think_prompt_audit_pass"]:
        raise RuntimeError("GPU_GATE_FAIL")
    if file_sha(MODEL_PATHS[model_label] / "adapter_model.safetensors") != EXPECTED_ADAPTER_SHA[model_label]:
        raise RuntimeError("ADAPTER_SHA_MISMATCH")
    rank, world, device = setup_rank()
    model, tokenizer = load_model(MODEL_PATHS[model_label], device)
    records = []
    for index, item in enumerate(prompt_audit["items"]):
        if index % world != rank:
            continue
        prompt = list(map(int, item["nothink_prompt_token_ids"]))
        domain_ids = tokenizer.encode(DOMAIN[item["domain"]], add_special_tokens=False)
        if len(domain_ids) != 1:
            raise RuntimeError("DOMAIN_TOKEN_NOT_ATOMIC")
        history = history_from_prompt(tokenizer, prompt)
        gold_set = {sid(value) for value in item["all_gold_sids"]}
        result = strict_beam(model, tokenizer, prompt + domain_ids, item["domain"], gold_set, history)
        records.append({"model": model_label, "group_id": item["group_id"], "domain": item["domain"],
                        "K": item["K"], "gold_sid_in_history": item["gold_sid_in_history"], "mode": "nothink_controlled_proxy", **result})
        print(f"NOTHINK_PROGRESS model={model_label} rank={rank} case={index // world + 1}/10", flush=True)
    write_jsonl(OUTPUT / "parts" / f"nothink_{model_label}_rank{rank}.jsonl", records)


def merge_nothink() -> None:
    rows = [row for model in MODELS for rank in range(4) for row in read_jsonl(OUTPUT / "parts" / f"nothink_{model}_rank{rank}.jsonl")]
    if len(rows) != 80 or len({(row["model"], row["group_id"]) for row in rows}) != 80 or any(len(row["beams"]) != 32 for row in rows):
        raise RuntimeError("NOTHINK_80_BEAM32_CONTRACT_FAIL")
    rows.sort(key=lambda row: (MODELS.index(row["model"]), row["group_id"]))
    write_jsonl(OUTPUT / "nothink_records.jsonl", rows)
    print("NO_THINK_CASES=80")


def plain_summary(rows, items, manifold) -> dict[str, Any]:
    values = [row_metrics(row, items[row["group_id"]]) for row in rows]
    result = {"N": len(rows)}
    for metric in ("hit1", "hit5", "hit10", "hit32", "mrr", "ab32", "a32", "history_fraction", "top1_is_history", "unique_a", "unique_ab", "unique_abc"):
        result[metric] = statistics.fmean(value[metric] for value in values)
        if metric in ("hit1", "hit5", "hit10", "hit32", "ab32", "a32", "top1_is_history"):
            result[f"{metric}_numerator"] = int(sum(value[metric] for value in values))
    classes = Counter()
    for row in rows:
        for beam in row["beams"]:
            value = sid(beam["predicted_sid"]) if beam["predicted_sid"] else None
            if value is None: category = "INVALID"
            elif beam["copy_class"] == "EXACT_COPY": category = "HISTORY_COPY"
            elif value in manifold[value[0]]: category = "SEEN_NOVEL"
            else: category = "UNSEEN_RECOMBINATION"
            classes[category] += 1
    total = len(rows) * 32
    result["manifold"] = {key: classes[key] / total for key in ("HISTORY_COPY", "SEEN_NOVEL", "UNSEEN_RECOMBINATION", "INVALID")}
    result["stratified"] = {}
    for name, predicate in (("GoldSIDInHistory", lambda item: item["gold_sid_in_history"]),
                            ("GoldSIDNotInHistory", lambda item: not item["gold_sid_in_history"]),
                            ("K1", lambda item: item["K"] == 1), ("K2PLUS", lambda item: item["K"] >= 2)):
        selected = [row for row in rows if predicate(items[row["group_id"]])]
        selected_values = [row_metrics(row, items[row["group_id"]]) for row in selected]
        result["stratified"][name] = {"N": len(selected), "hit32_numerator": int(sum(value["hit32"] for value in selected_values)),
            "hit32": statistics.fmean(value["hit32"] for value in selected_values), "mrr": statistics.fmean(value["mrr"] for value in selected_values),
            "history_fraction": statistics.fmean(value["history_fraction"] for value in selected_values)}
    return result


def order(a: float, b: float) -> str:
    return "Beta > Step900" if a > b else "Step900 > Beta" if b > a else "Beta = Step900"


def finalize_nothink() -> dict[str, Any]:
    audit = code_audit()
    records = read_jsonl(OUTPUT / "nothink_records.jsonl")
    prompt_audit = json.loads((OUTPUT / "nothink_prompt_audit.json").read_text(encoding="utf-8"))
    items = {row["group_id"]: row for row in prompt_audit["items"]}
    manifold, manifold_stats = build_manifold()
    natural, _ = natural_rows()
    counts = natural_counts(natural)
    metrics, weighted = {}, {}
    for model in MODELS:
        rows = [row for row in records if row["model"] == model]
        metrics[model] = plain_summary(rows, items, manifold)
        weighted[model] = poststratify(rows, items, counts)
    existing = json.loads((OUTPUT / "poststratified_existing_metrics.json").read_text(encoding="utf-8"))
    phase12 = read_jsonl(PHASE12 / "records.jsonl")
    phase12_items = {row["group_id"]: row for row in json.loads((PHASE12 / "prompt_audit.json").read_text(encoding="utf-8"))["items"]}
    think_mrr = {model: statistics.fmean(row_metrics(row, phase12_items[row["group_id"]])["mrr"] for row in phase12 if row["model"] == model) for model in MODELS}
    phase13 = json.loads((PHASE13 / "crossover_summary.json").read_text(encoding="utf-8"))["cells_all20"]
    nothink_order = order(metrics["Beta"]["mrr"], metrics["Step900"]["mrr"])
    weighted_order = order(weighted["Beta"]["source_frequency_weighted"]["mrr"], weighted["Step900"]["source_frequency_weighted"]["mrr"])
    explains = "YES" if nothink_order == weighted_order == "Beta > Step900" else "PARTIAL" if "Beta > Step900" in (nothink_order, weighted_order) else "NO"
    result = {"implement_commit": audit["implement_commit"], "groups": 40, "cases": 80,
              "no_think_renderer": "CONTROLLED_PROXY", "exact_official_route": False,
              "metrics": metrics, "natural_weighted": weighted, "train_manifold_stats": manifold_stats,
              "think_frozen_order": order(think_mrr["Beta"], think_mrr["Step900"]),
              "think_own_self_order": order(phase13["GB_DB"]["mrr"], phase13["G9_D9"]["mrr"]),
              "nothink_order": nothink_order, "nothink_weighted_order": weighted_order,
              "nothink_explains_external_order": explains, "training_started": False, "optimizer_steps": 0,
              "self_cot_generation_started": False, "external_eval_started": False}
    write_json(OUTPUT / "nothink_summary.json", result)
    md = render_nothink_md(result)
    (OUTPUT / "nothink_summary.md").write_text(md, encoding="utf-8")
    return result


def render_nothink_md(result) -> str:
    lines = ["# Controlled-Proxy No-Think Probe", "", "This is not an exact official soft-switch route.", "",
             "| Model | Hit@32 | MRR | AB@32 | A@32 | History fraction | Weighted MRR | Coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        value, weighted = result["metrics"][model], result["natural_weighted"][model]
        lines.append(f"| {model} | {value['hit32']:.3f} | {value['mrr']:.6f} | {value['ab32']:.3f} | {value['a32']:.3f} | {value['history_fraction']:.4f} | {weighted['source_frequency_weighted']['mrr']:.6f} | {weighted['covered_natural_mass']:.3f} |")
    lines += ["", f"No-Think external-order explanation: **{result['nothink_explains_external_order']}**."]
    return "\n".join(lines) + "\n"


def final_report() -> dict[str, Any]:
    audit = code_audit()
    stats = json.loads((OUTPUT / "natural_distribution_stats.json").read_text(encoding="utf-8"))
    shift = json.loads((OUTPUT / "diagnostic_distribution_shift.json").read_text(encoding="utf-8"))
    existing = json.loads((OUTPUT / "poststratified_existing_metrics.json").read_text(encoding="utf-8"))
    artifact = json.loads((OUTPUT / "bounded_official_artifact_check.json").read_text(encoding="utf-8"))
    nothink_path = OUTPUT / "nothink_summary.json"
    nothink = json.loads(nothink_path.read_text(encoding="utf-8")) if nothink_path.is_file() else None
    p12 = existing["phase12"]
    p13 = existing["phase13"]
    coverage = min(p12[model]["covered_natural_mass"] for model in MODELS)
    coverage = min(coverage, *(p13[key]["covered_natural_mass"] for key in p13))
    original_p12_order = "Beta > Step900"
    original_p13_order = "Beta > Step900"
    weighted_orders_unchanged = existing["phase12_weighted_mrr_order"] == original_p12_order and existing["phase13_weighted_own_pair_order"] == original_p13_order
    sampling = "INCONCLUSIVE" if coverage < .90 else "NO" if weighted_orders_unchanged else "PARTIAL"
    if nothink:
        if nothink["nothink_explains_external_order"] == "YES":
            root, next_variable = "NOTHINK_ROUTE_DEGRADATION", "OFFICIAL_SOFT_SWITCH_ROUTE_CONFIRMATION"
        else:
            root, next_variable = "MODEL_SIDE_LOCAL_METRICS_INSUFFICIENT", "OFFICIAL_EVALUATOR_REWARD_OR_TEST_DISTRIBUTION"
        conclusion = f"Natural reweighting sampling={sampling}; controlled No-Think explanation={nothink['nothink_explains_external_order']}."
    else:
        root, next_variable = "OFFICIAL_ARTIFACT_AVAILABLE", "OFFICIAL_ARTIFACT_AUDIT"
        conclusion = "Bounded official artifact was found; No-Think GPU probe was correctly blocked."
    result = {"code_audit": audit, "natural": stats, "shift": shift, "existing": existing,
              "artifact": artifact, "nothink": nothink, "sampling_bias_explains_local_external_gap": sampling,
              "root_class": root, "conclusion": conclusion, "next_root_variable": next_variable,
              "next_experiment_needed": "WAIT_FOR_CHATGPT_CODE_AUDIT", "training_started": False,
              "optimizer_steps": 0, "self_cot_generation_started": False, "external_eval_started": False}
    text = terminal(result)
    (OUTPUT / "CHATGPT_ROOT_CAUSE_PHASE1_4.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return result


def terminal(result) -> str:
    audit, stats, shift, existing, artifact, nothink = result["code_audit"], result["natural"], result["shift"], result["existing"], result["artifact"], result["nothink"]
    p40, p20 = shift["populations"]["probe40"]["overall"], shift["populations"]["probe20"]["overall"]
    p12, p13 = existing["phase12"], existing["phase13"]
    interaction = existing["phase13_weighted_interaction_mrr"]
    lines = [f"IMPLEMENT_COMMIT={audit['implement_commit']}", f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short'] or 'EMPTY'}", "",
             f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}", f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}", f"RUNTIME_GITHUB_PARITY={audit['runtime_github_parity']}", "", "--- Natural Distribution ---", "",
             f"NATURAL_PROXY_GROUPS={stats['overall']['N']}", f"NATURAL_GOLD_IN_HISTORY_RATE={stats['overall']['GoldSIDInHistoryRate']:.6f}", f"PROBE40_GOLD_IN_HISTORY_RATE={p40['GoldSIDInHistoryRate']:.6f}", f"PROBE20_GOLD_IN_HISTORY_RATE={p20['GoldSIDInHistoryRate']:.6f}",
             f"NATURAL_K2PLUS_RATE={stats['overall']['K2PLUSRate']:.6f}", f"PROBE40_K2PLUS_RATE={p40['K2PLUSRate']:.6f}", f"PROBE20_K2PLUS_RATE={p20['K2PLUSRate']:.6f}", f"DIAGNOSTIC_DISTRIBUTION_SHIFT={shift['diagnostic_distribution_shift']}", "", "--- Existing Reweighting ---", "",
             f"PHASE12_WEIGHTED_BETA_MRR={p12['Beta']['source_frequency_weighted']['mrr']:.6f}", f"PHASE12_WEIGHTED_STEP900_MRR={p12['Step900']['source_frequency_weighted']['mrr']:.6f}", f"PHASE12_WEIGHTED_MRR_ORDER={existing['phase12_weighted_mrr_order']}", f"PHASE12_COVERED_NATURAL_MASS={p12['Beta']['covered_natural_mass']:.6f}",
             f"PHASE13_WEIGHTED_GB_DB_MRR={p13['GB_DB']['source_frequency_weighted']['mrr']:.6f}", f"PHASE13_WEIGHTED_G9_D9_MRR={p13['G9_D9']['source_frequency_weighted']['mrr']:.6f}", f"PHASE13_WEIGHTED_OWN_PAIR_ORDER={existing['phase13_weighted_own_pair_order']}", f"PHASE13_WEIGHTED_INTERACTION_MRR={interaction['estimate']:.6f}", f"PHASE13_WEIGHTED_INTERACTION_CI95={interaction['ci95']}", f"PHASE13_COVERED_NATURAL_MASS={interaction['covered_natural_mass']:.6f}", f"SAMPLING_BIAS_EXPLAINS_LOCAL_EXTERNAL_GAP={result['sampling_bias_explains_local_external_gap']}", "", "--- Bounded Official Artifact Check ---", "",
             f"OFFICIAL_ARTIFACT_AVAILABLE={'YES' if artifact['official_artifact_available'] else 'NO'}", f"OFFICIAL_ARTIFACT_PATHS={[value['path'] for value in artifact['artifacts']]}", "", "--- GPU Gate ---", "", f"GPU_GATE={'FAIL_OFFICIAL_ARTIFACT_FOUND' if artifact['official_artifact_available'] else 'PASS'}", f"GPU_RERUN_STARTED={'YES' if nothink else 'NO'}", ""]
    if nothink:
        beta, step = nothink["metrics"]["Beta"], nothink["metrics"]["Step900"]
        beta_w, step_w = (nothink["natural_weighted"][model] for model in MODELS)
        lines += ["NO_THINK_CASES=80", f"NO_THINK_RENDERER={nothink['no_think_renderer']}", "EXACT_OFFICIAL_ROUTE=NO", f"BETA_NOTHINK_HIT32={beta['hit32']:.6f}", f"STEP900_NOTHINK_HIT32={step['hit32']:.6f}", f"BETA_NOTHINK_MRR={beta['mrr']:.6f}", f"STEP900_NOTHINK_MRR={step['mrr']:.6f}", f"BETA_NOTHINK_WEIGHTED_MRR={beta_w['source_frequency_weighted']['mrr']:.6f}", f"STEP900_NOTHINK_WEIGHTED_MRR={step_w['source_frequency_weighted']['mrr']:.6f}", f"NOTHINK_ORDER={nothink['nothink_order']}", f"NOTHINK_WEIGHTED_ORDER={nothink['nothink_weighted_order']}", f"NOTHINK_EXPLAINS_EXTERNAL_ORDER={nothink['nothink_explains_external_order']}", ""]
    else:
        lines += ["NO_THINK_CASES=0", ""]
    lines += ["TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "SELF_COT_GENERATION_STARTED=NO", "EXTERNAL_EVAL_STARTED=NO", "", f"ROOT_CLASS={result['root_class']}", f"CONCLUSION={result['conclusion']}", f"NEXT_ROOT_VARIABLE={result['next_root_variable']}", f"NEXT_EXPERIMENT_NEEDED={result['next_experiment_needed']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu", "prepare-nothink", "decode-nothink", "merge-nothink", "finalize-nothink", "report"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.action == "cpu": run_cpu()
    elif args.action == "prepare-nothink": prepare_nothink()
    elif args.action == "decode-nothink":
        if not args.model: parser.error("--model is required")
        decode_nothink(args.model)
    elif args.action == "merge-nothink": merge_nothink()
    elif args.action == "finalize-nothink": finalize_nothink()
    else: final_report()


if __name__ == "__main__":
    main()
