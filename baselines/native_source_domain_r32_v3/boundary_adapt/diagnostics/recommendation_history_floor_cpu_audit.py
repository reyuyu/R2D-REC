"""Phase 1.5.2 CPU-only history-floor and novel-recommendation audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable


RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/"
    "recommendation_history_floor_cpu_audit.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_history_floor_cpu_audit.py"
OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_history_floor_cpu_audit"

SOURCE = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
SOURCE_SHA256 = "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca"
SPLIT_MANIFEST = Path(
    "/root/GRPO_audit_results/bridge_inside_transition_sft_v1_20260824/split_manifest.json"
)
CANONICAL_ROWS = RUNTIME / "boundary_adapt/results/boundary_adapt_rows.jsonl"
DOMAIN_ORDER = ("video", "prod", "ad", "living")
DOMAIN_TOKEN = {domain: f"<|{domain}_begin|>" for domain in DOMAIN_ORDER}
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")

PHASE10 = RUNTIME / "boundary_adapt/results/recommendation_decoder_diagnostic_v1_40g"
PHASE12 = RUNTIME / "boundary_adapt/results/recommendation_official_prompt_bare_crossover_40g"
PHASE13 = RUNTIME / "boundary_adapt/results/recommendation_self_cot_crossover_20g"
PHASE14 = RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_4"
PHASE141 = RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_4_1"
PHASE15 = RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_quick_probe"
PHASE151 = RUNTIME / "boundary_adapt/results/recommendation_memory_history_2x2_probe"

PROTECTED_FILES = {
    "phase141_summary": PHASE141 / "summary.json",
    "phase141_review": PHASE141 / "CHATGPT_ROOT_CAUSE_PHASE1_4_1.txt",
    "phase15_records": PHASE15 / "records.jsonl",
    "phase15_summary": PHASE15 / "seen_unseen_summary.json",
    "phase151_records": PHASE151 / "records.jsonl",
    "phase151_summary": PHASE151 / "cell_summary.json",
    "phase151_manifest": PHASE151 / "probe16_manifest.json",
    "phase151_invalid": PHASE151 / "invalid_beam_audit.json",
}

OFFICIAL_FILES = (
    ("README_OVERVIEW", SOURCE_REPO / "README.md", ["Beta", "MiniFix"], "score overview"),
    (
        "BETA_SCORE_RECORD",
        SOURCE_REPO / "baselines/native_source_domain_r32_v3/docs/BATA_BASELINE.md",
        ["Beta"],
        "material,user,recommendation,world",
    ),
    (
        "MINIFIX_SCORE_RECORD",
        SOURCE_REPO / "baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX.md",
        ["MiniFix"],
        "material,user,recommendation,world",
    ),
    (
        "GRPO_EXTERNAL_SCORE_ARTIFACT",
        SOURCE_REPO
        / "baselines/native_source_domain_r32_v3/grpo/results/"
        "gr_rec_clamp_bridge_v1_external_checkpoint_eval_20260821.json",
        ["GRPO_CLAMP"],
        "material,user,recommendation,world",
    ),
)

OFFICIAL_SCORE_CONTRACT = {
    "Beta": {
        "domain_scores": [0.1223, 0.1598, 0.2072, 0.1701],
        "rec_sum": 0.6594,
        "source": "baselines/native_source_domain_r32_v3/docs/BATA_BASELINE.md",
        "checkpoint": "BATA-BASELINE checkpoint-1106",
    },
    "MiniFix": {
        "domain_scores": [0.1297, 0.1598, 0.2030, 0.1719],
        "rec_sum": 0.6644,
        "source": "baselines/native_source_domain_r32_v3/docs/experiment_MINI_FIX.md",
        "checkpoint": "MINI-FIX checkpoint-138",
    },
    "Gamma": {"domain_scores": None, "rec_sum": None, "source": None, "checkpoint": "MISSING"},
    "Sol": {"domain_scores": None, "rec_sum": None, "source": None, "checkpoint": "MISSING"},
    "Step900": {"domain_scores": None, "rec_sum": None, "source": None, "checkpoint": "checkpoint-900"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args], text=True, encoding="utf-8"
    ).strip()


def code_audit() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    status = git("status", "--short")
    source_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT)
    runtime_sha = file_sha(RUNTIME_SCRIPT)
    result = {
        "implement_commit": head,
        "origin_main": origin,
        "push_status": "PASS" if head == origin else "FAIL",
        "git_status_short": status,
        "github_script_sha256": source_sha,
        "runtime_script_sha256": runtime_sha,
        "runtime_github_parity": "PASS" if source_sha == runtime_sha else "FAIL",
    }
    if head != origin or status or source_sha != runtime_sha:
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={result}")
    return result


def protected_hashes() -> dict[str, str]:
    missing = [str(path) for path in PROTECTED_FILES.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"PROTECTED_RESULT_MISSING={missing}")
    return {name: file_sha(path) for name, path in PROTECTED_FILES.items()}


def parse_sid(value: str | list[Any] | tuple[Any, ...]) -> tuple[str, int, int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            raise ValueError(f"invalid SID tuple {value!r}")
        return str(value[0]), int(value[1]), int(value[2]), int(value[3])
    match = SID_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"invalid SID {value!r}")
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row.get("aux_metadata_json") or "{}")


def load_unique_source() -> tuple[list[dict[str, Any]], int]:
    canonical: dict[str, tuple[str, str]] = {}
    with CANONICAL_ROWS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            group = str(row["boundary_group_id"])
            identity = (str(row["prompt"]), str(row["original_response"]))
            if group in canonical and canonical[group] != identity:
                raise RuntimeError(f"CANONICAL_BOUNDARY_GROUP_CONFLICT={group}")
            canonical[group] = identity
    if len(canonical) != 15943:
        raise RuntimeError(f"CANONICAL_BOUNDARY_GROUP_COUNT={len(canonical)}")

    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows = 0
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend" or row.get("source_segment") != "recommendation_cot":
                continue
            source_rows += 1
            metadata = source_metadata(row)
            group = str(metadata["recommendation_group_id"])
            identity = (str(row.get("instruction", "")) + str(row.get("input", "")), str(row["output"]))
            if canonical.get(group) == identity:
                matches[group].append(row)
    if set(matches) != set(canonical):
        raise RuntimeError(f"CANONICAL_SOURCE_MATCH_MISSING={len(set(canonical) - set(matches))}")
    selected = []
    for group in canonical:
        candidates = matches[group]
        signatures = {
            (row.get("system"), row.get("instruction"), row.get("input"), row.get("output"), row.get("aux_metadata_json"))
            for row in candidates
        }
        if len(signatures) != 1:
            raise RuntimeError(f"CANONICAL_SOURCE_MATCH_AMBIGUOUS={group}")
        selected.append(candidates[0])
    return selected, source_rows - len(selected)


def natural_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if file_sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("SOURCE_SHA256_MISMATCH")
    split = read_json(SPLIT_MANIFEST)
    if split.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("SPLIT_SOURCE_SHA256_MISMATCH")
    train_ids = set(map(str, split["train_group_ids"]))
    holdout_ids = set(map(str, split["holdout_group_ids"]))
    if len(train_ids) != 14348 or len(holdout_ids) != 1595 or train_ids & holdout_ids:
        raise RuntimeError("SPLIT_CONTRACT_FAIL")
    source_rows, duplicates = load_unique_source()
    selected = []
    for raw in source_rows:
        metadata = source_metadata(raw)
        group = str(metadata["recommendation_group_id"])
        if group not in holdout_ids:
            continue
        current = str(metadata["recommendation_current_gold_sid"])
        domains = [name for name, token in DOMAIN_TOKEN.items() if current.startswith(token)]
        if len(domains) != 1:
            raise RuntimeError(f"DOMAIN_PARSE_FAIL={group}")
        domain = domains[0]
        golds = {parse_sid(str(value)) for value in metadata["recommendation_all_gold_sids"]}
        if not golds or any(value[0] != domain for value in golds):
            raise RuntimeError(f"GOLD_DOMAIN_FAIL={group}")
        prompt = str(raw.get("instruction", "")) + str(raw.get("input", ""))
        history = [parse_sid(match.group(0)) for match in SID_RE.finditer(prompt) if match.group(1) == domain]
        history_a = {(value[0], value[1]) for value in history}
        history_ab = {(value[0], value[1], value[2]) for value in history}
        selected.append(
            {
                "group_id": group,
                "domain": domain,
                "K": len(golds),
                "gold_a_in_history": any((value[0], value[1]) in history_a for value in golds),
                "gold_ab_in_history": any((value[0], value[1], value[2]) in history_ab for value in golds),
                "gold_abc_in_history": bool(golds.intersection(history)),
            }
        )
    if len(selected) != 1595 or {row["group_id"] for row in selected} != holdout_ids:
        raise RuntimeError("HOLDOUT_RECONSTRUCTION_FAIL")
    provenance = {
        "source": str(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": file_sha(SPLIT_MANIFEST),
        "canonical_rows": str(CANONICAL_ROWS),
        "canonical_rows_sha256": file_sha(CANONICAL_ROWS),
        "unique_groups": 15943,
        "train_groups": len(train_ids),
        "holdout_groups": len(holdout_ids),
        "train_holdout_intersection": 0,
        "legacy_duplicates_dropped": duplicates,
        "source_streaming_passes": 1,
        "full_sid_match_contract": "STRICT_DOMAIN_A_B_C",
    }
    return selected, provenance


def summarize_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "ADAPTATION_HELDOUT_NATURAL_PROXY",
        "not_official_test_distribution": True,
        "N": len(rows),
        "per_domain": {},
    }
    for domain in DOMAIN_ORDER:
        values = [row for row in rows if row["domain"] == domain]
        n = len(values)
        counts = Counter(
            ("HISTORY" if row["gold_abc_in_history"] else "NONHISTORY")
            + "_"
            + ("K1" if row["K"] == 1 else "K2PLUS")
            for row in values
        )
        entry = {
            "N": n,
            "GoldSIDInHistory": sum(row["gold_abc_in_history"] for row in values),
            "GoldSIDNotInHistory": sum(not row["gold_abc_in_history"] for row in values),
            "GoldAInHistory": sum(row["gold_a_in_history"] for row in values),
            "GoldABInHistory": sum(row["gold_ab_in_history"] for row in values),
            "GoldABCInHistory": sum(row["gold_abc_in_history"] for row in values),
            "K1": sum(row["K"] == 1 for row in values),
            "K2PLUS": sum(row["K"] >= 2 for row in values),
            "history_x_k": {
                key: counts[key]
                for key in ("HISTORY_K1", "HISTORY_K2PLUS", "NONHISTORY_K1", "NONHISTORY_K2PLUS")
            },
        }
        for level in ("A", "AB", "ABC"):
            entry[f"Gold{level}InHistoryRate"] = entry[f"Gold{level}InHistory"] / n
        entry["GoldSIDInHistoryRate"] = entry["GoldABCInHistoryRate"]
        entry["GoldSIDNotInHistoryRate"] = 1.0 - entry["GoldABCInHistoryRate"]
        result["per_domain"][domain] = entry
    return result


def render_history_md(stats: dict[str, Any]) -> str:
    lines = [
        "# Natural history overlap",
        "",
        "`ADAPTATION_HELDOUT_NATURAL_PROXY` is a 1595-group adaptation holdout, not the official test distribution.",
        "All overlaps use strict target-domain SID prefixes.",
        "",
        "| Domain | N | A history | AB history | ABC history | H-K1 | H-K2+ | NH-K1 | NH-K2+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAIN_ORDER:
        value = stats["per_domain"][domain]
        bucket = value["history_x_k"]
        lines.append(
            f"| {domain} | {value['N']} | {value['GoldAInHistoryRate']:.4%} | "
            f"{value['GoldABInHistoryRate']:.4%} | {value['GoldABCInHistoryRate']:.4%} | "
            f"{bucket['HISTORY_K1']} | {bucket['HISTORY_K2PLUS']} | "
            f"{bucket['NONHISTORY_K1']} | {bucket['NONHISTORY_K2PLUS']} |"
        )
    return "\n".join(lines) + "\n"


def git_date(path: Path) -> str | None:
    relative = path.relative_to(SOURCE_REPO)
    value = git("log", "-1", "--format=%cI", "--", str(relative))
    return value or None


def official_inventory() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    files = []
    for label, path, models, tasks in OFFICIAL_FILES:
        if not path.is_file():
            continue
        files.append(
            {
                "label": label,
                "path": str(path.relative_to(SOURCE_REPO)),
                "sha256": file_sha(path),
                "models_or_checkpoints": models,
                "evaluation_date": git_date(path),
                "tasks_included": tasks,
                "per_example_artifact_available": False,
                "artifact_kind": "json_score_artifact" if path.suffix == ".json" else "versioned_score_record",
            }
        )
    expected_text = {
        "Beta": "recommendation: 0.1223, 0.1598, 0.2072, 0.1701",
        "MiniFix": "0.1297, 0.1598, 0.2030, 0.1719",
    }
    for model, needle in expected_text.items():
        contract = OFFICIAL_SCORE_CONTRACT[model]
        source = SOURCE_REPO / str(contract["source"])
        if needle not in source.read_text(encoding="utf-8"):
            raise RuntimeError(f"OFFICIAL_SCORE_CONTRACT_NOT_FOUND={model}")
    table = {
        "component_order": list(DOMAIN_ORDER),
        "models": OFFICIAL_SCORE_CONTRACT,
        "official_max_score_confirmed": "NO",
        "task_max_source": None,
        "normalized_scores": None,
        "normalization_reason": "No project file in the bounded search explicitly confirms per-task maxima.",
        "missing_models": [name for name, value in OFFICIAL_SCORE_CONTRACT.items() if value["domain_scores"] is None],
    }
    inventory = {
        "bounded_roots": ["README.md", "baselines/native_source_domain_r32_v3/docs", "known grpo/results score artifact"],
        "recursive_disk_scan": False,
        "files_found": files,
        "count": len(files),
        "target_model_status": {
            model: "AVAILABLE" if value["domain_scores"] is not None else "MISSING"
            for model, value in OFFICIAL_SCORE_CONTRACT.items()
        },
    }
    decomposition = {
        "official_per_example_available": "NO",
        "reason": "No bounded project artifact contains official per-example prediction, gold, history, and evaluator-supported score fields.",
        "official_history_contribution": None,
        "not_official_score_decomposition": True,
    }
    return inventory, table, decomposition


def first_rank(beams: Iterable[dict[str, Any]], golds: set[tuple[str, int, int, int]], depth: int) -> int | None:
    gold_prefixes = {value[: depth + 1] for value in golds}
    ranks = []
    for beam in beams:
        predicted = beam.get("predicted_sid")
        if predicted is None:
            continue
        value = parse_sid(predicted)
        if value[: depth + 1] in gold_prefixes:
            ranks.append(int(beam["beam_index"]) + 1)
    return min(ranks) if ranks else None


def hierarchy_metrics(record: dict[str, Any], golds: set[tuple[str, int, int, int]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for label, depth in (("A", 1), ("AB", 2), ("ABC", 3)):
        rank = first_rank(record["beams"], golds, depth)
        result[f"{label}Rank"] = rank
        result[f"{label}MRR"] = 0.0 if rank is None else 1.0 / rank
        for cutoff in (1, 5, 32):
            result[f"{label}Hit@{cutoff}"] = int(rank is not None and rank <= cutoff)
    result["HistoryCandidateFraction"] = sum(
        beam.get("copy_class") == "EXACT_COPY" for beam in record["beams"]
    ) / 32.0
    return result


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"N": 0}
    keys = [
        key for key in rows[0]
        if key != "group_id" and all(isinstance(row.get(key), (int, float)) for row in rows)
    ]
    return {"N": len(rows), **{key: statistics.fmean(float(row[key]) for row in rows) for key in keys}}


def phase151_hierarchy() -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    manifest = read_json(PHASE151 / "probe16_manifest.json")
    items = {item["group_id"]: item for item in manifest["items"]}
    records = read_jsonl(PHASE151 / "records.jsonl")
    if len(records) != 64:
        raise RuntimeError(f"PHASE151_RECORD_COUNT={len(records)}")
    per_record: dict[tuple[str, str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        item = items[record["group_id"]]
        golds = {parse_sid(value) for value in item["all_gold_sids"]}
        metrics = hierarchy_metrics(record, golds)
        key = (record["model"], record["cell"], record["condition"])
        grouped[key].append({"group_id": record["group_id"], **metrics})
        per_record[(record["model"], record["group_id"], record["condition"])] = {
            "cell": record["cell"], "domain": record["domain"], **metrics
        }
    summary: dict[str, Any] = {"models": {}, "per_group": []}
    for model in ("MiniFix", "Gamma"):
        summary["models"][model] = {}
        for cell in ("SH", "SN", "UH", "UN"):
            summary["models"][model][cell] = {}
            for condition in ("BARE", "EXACT_BRIDGE"):
                values = grouped[(model, cell, condition)]
                summary["models"][model][cell][condition] = average_metrics(values)
                summary["per_group"].extend(
                    {"model": model, "cell": cell, "condition": condition, **value} for value in values
                )
    return summary, per_record


def render_hierarchy_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 1.5.1 hierarchy audit",
        "",
        "Ranks use the persisted Beam32 order. No model inference was run.",
        "",
        "| Model | Cell | Route | A@1/5/32 | AB@1/5/32 | ABC@1/5/32 | A MRR | AB MRR | ABC MRR |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for model in ("MiniFix", "Gamma"):
        for cell in ("SH", "SN", "UH", "UN"):
            for route in ("BARE", "EXACT_BRIDGE"):
                value = summary["models"][model][cell][route]
                hits = lambda prefix: "/".join(f"{value[f'{prefix}Hit@{cut}']:.2f}" for cut in (1, 5, 32))
                lines.append(
                    f"| {model} | {cell} | {route} | {hits('A')} | {hits('AB')} | {hits('ABC')} | "
                    f"{value['AMRR']:.4f} | {value['ABMRR']:.4f} | {value['ABCMRR']:.4f} |"
                )
    return "\n".join(lines) + "\n"


def manifest_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {item["group_id"]: item for item in read_json(path)["items"]}


def add_evidence(
    output: list[dict[str, Any]], source_phase: str, contract: str, records_path: Path,
    items: dict[str, dict[str, Any]], model_field: str, route_field: str,
) -> None:
    rows = read_jsonl(records_path)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        item = items[record["group_id"]]
        model = str(record[model_field])
        if model_field == "decoder":
            model = f"G{record['generator']}_D{record['decoder']}"
        route = str(record[route_field])
        history = "HISTORY" if bool(item["gold_sid_in_history"]) else "NONHISTORY"
        golds = {parse_sid(value) for value in item["all_gold_sids"]}
        metrics = hierarchy_metrics(record, golds)
        grouped[(model, route, record["domain"], history)].append(metrics)
    for (model, route, domain, history), values in sorted(grouped.items()):
        avg = average_metrics(values)
        output.append(
            {
                "SOURCE_PHASE": source_phase,
                "SAMPLE_CONTRACT": contract,
                "MODEL": model,
                "ROUTE": route,
                "DOMAIN": domain,
                "HISTORY_SPLIT": history,
                "N": avg["N"],
                "GoldMRR": avg["ABCMRR"],
                "Hit32": avg["ABCHit@32"],
                "ABHit32": avg["ABHit@32"],
                "AHit32": avg["AHit@32"],
                "HistoryCandidateFraction": avg["HistoryCandidateFraction"],
            }
        )


def existing_evidence_table() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    manifest40 = manifest_lookup(PHASE10 / "manifest_40.json")
    add_evidence(rows, "Phase1.0", "40G_FROZEN_COT_ORIGINAL_STYLE", PHASE10 / "records.jsonl", manifest40, "model", "mode")
    add_evidence(rows, "Phase1.2", "40G_FROZEN_COT_OFFICIAL_STYLE", PHASE12 / "records.jsonl", manifest40, "model", "mode")
    manifest20 = manifest_lookup(PHASE13 / "manifest_20.json")
    add_evidence(rows, "Phase1.3", "20G_SELF_COT_GENERATOR_DECODER", PHASE13 / "decoder_records.jsonl", manifest20, "decoder", "status")
    phase14_items = {item["group_id"]: item for item in read_json(PHASE14 / "nothink_prompt_audit.json")["items"]}
    add_evidence(rows, "Phase1.4", "40G_CONTROLLED_PROXY_NOTHINK", PHASE14 / "nothink_records.jsonl", phase14_items, "model", "mode")
    phase15_items = manifest_lookup(PHASE15 / "probe16_manifest.json")
    add_evidence(rows, "Phase1.5-Quick", "16G_SEEN_UNSEEN_FIXED_COT", PHASE15 / "records.jsonl", phase15_items, "model", "condition")
    phase151_items = manifest_lookup(PHASE151 / "probe16_manifest.json")
    add_evidence(rows, "Phase1.5.1", "16G_TRAIN_MEMORY_X_HISTORY_2X2", PHASE151 / "records.jsonl", phase151_items, "model", "condition")
    return {
        "do_not_pool_across_sample_contracts": True,
        "derived_phase_notes": {
            "Phase1.1": "CPU behavior analysis derives from Phase1.0 records; not duplicated.",
            "Phase1.4.1": "Adjudication derives from Phase1.4 records; not duplicated.",
        },
        "rows": rows,
    }


def paired_copy_source(
    source_phase: str, contract: str, records_path: Path, bare_name: str, bridge_name: str,
    condition_field: str, items: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = read_jsonl(records_path)
    index = {(row["model"], row["group_id"], row[condition_field]): row for row in records}
    output = []
    for model, group, condition in sorted(index):
        if condition != bare_name or (model, group, bridge_name) not in index:
            continue
        bare, bridge = index[(model, group, bare_name)], index[(model, group, bridge_name)]
        bare_fraction = sum(beam.get("copy_class") == "EXACT_COPY" for beam in bare["beams"]) / 32.0
        bridge_fraction = sum(beam.get("copy_class") == "EXACT_COPY" for beam in bridge["beams"]) / 32.0
        output.append(
            {
                "SOURCE_PHASE": source_phase,
                "SAMPLE_CONTRACT": contract,
                "model": model,
                "group_id": group,
                "domain": bare["domain"],
                "history_split": "HISTORY"
                if bool(
                    bare.get("gold_sid_in_history")
                    if "gold_sid_in_history" in bare
                    else (items or {})[group]["gold_sid_in_history"]
                )
                else "NONHISTORY",
                "bare_history_fraction": bare_fraction,
                "bridge_history_fraction": bridge_fraction,
                "delta": bridge_fraction - bare_fraction,
            }
        )
    return output


def bridge_copy_analysis() -> dict[str, Any]:
    pairs = []
    pairs.extend(
        paired_copy_source(
            "Phase1.0", "40G_ORIGINAL_STYLE", PHASE10 / "records.jsonl",
            "bare", "old_bridge", "mode", manifest_lookup(PHASE10 / "manifest_40.json"),
        )
    )
    pairs.extend(paired_copy_source("Phase1.5-Quick", "16G_SEEN_UNSEEN", PHASE15 / "records.jsonl", "BARE", "EXACT_BRIDGE", "condition"))
    pairs.extend(paired_copy_source("Phase1.5.1", "16G_2X2", PHASE151 / "records.jsonl", "BARE", "EXACT_BRIDGE", "condition"))
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in pairs:
        grouped[(row["SOURCE_PHASE"], row["model"], row["domain"], row["history_split"])].append(row["delta"])
    summary = [
        {
            "SOURCE_PHASE": key[0], "model": key[1], "domain": key[2], "history_split": key[3],
            "N": len(values), "mean_delta": statistics.fmean(values),
            "increase_n": sum(value > 0 for value in values),
            "decrease_n": sum(value < 0 for value in values),
            "unchanged_n": sum(value == 0 for value in values),
        }
        for key, values in sorted(grouped.items())
    ]
    deltas = [row["delta"] for row in pairs]
    increases, decreases = sum(value > 0 for value in deltas), sum(value < 0 for value in deltas)
    direction = "DECREASE" if decreases > increases else "INCREASE" if increases > decreases else "MIXED"
    return {
        "paired_experiments": ["Phase1.0", "Phase1.5-Quick", "Phase1.5.1"],
        "pairs": len(pairs),
        "mean_delta": statistics.fmean(deltas),
        "increase_n": increases,
        "decrease_n": decreases,
        "unchanged_n": sum(value == 0 for value in deltas),
        "bridge_copy_direction": direction,
        "bridge_is_not_a_simple_copy_switch": "SUPPORTED" if direction != "INCREASE" else "NOT_SUPPORTED",
        "stratified": summary,
        "per_group": pairs,
    }


def history_retrieval(per_record: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    manifest = read_json(PHASE151 / "probe16_manifest.json")
    history_groups = [item for item in manifest["items"] if item["cell"] in ("SH", "UH")]
    for model in ("MiniFix", "Gamma"):
        new, rerank, lost, unchanged = 0, 0, 0, 0
        audit = []
        for item in history_groups:
            bare = per_record[(model, item["group_id"], "BARE")]
            bridge = per_record[(model, item["group_id"], "EXACT_BRIDGE")]
            bare_rank, bridge_rank = bare["ABCRank"], bridge["ABCRank"]
            if bare_rank is None and bridge_rank is not None:
                new += 1
                change = "NEW_GOLD_ACQUISITION"
            elif bare_rank is not None and bridge_rank is None:
                lost += 1
                change = "LOST_GOLD"
            elif bare_rank is not None and bridge_rank is not None and bare_rank != bridge_rank:
                rerank += 1
                change = "EXISTING_GOLD_RERANK"
            else:
                unchanged += 1
                change = "UNCHANGED"
            audit.append(
                {
                    "group_id": item["group_id"], "domain": item["domain"], "cell": item["cell"],
                    "history_overlap_class": "EXACT_GOLD_IN_HISTORY",
                    "bare_gold_rank": bare_rank, "bridge_gold_rank": bridge_rank, "change": change,
                }
            )
        result[model] = {
            "history_groups": len(history_groups),
            "prefix_only_history_groups": 0,
            "bridge_new_gold_acquisition_count": new,
            "bridge_existing_gold_rerank_count": rerank,
            "bridge_lost_gold_count": lost,
            "bridge_unchanged_count": unchanged,
            "per_group": audit,
        }
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right))
    return None if denominator == 0 else numerator / denominator


def rankdata(values: list[float]) -> list[float]:
    output = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in ordered[start:end]:
            output[position] = rank
        start = end
    return output


def association(reference: str, modified: str, rates: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    left, right = scores["models"][reference]["domain_scores"], scores["models"][modified]["domain_scores"]
    if left is None or right is None:
        return {"classification": "INSUFFICIENT", "N_DOMAINS": 4, "DESCRIPTIVE_ONLY": "YES", "reason": f"{modified} domain scores missing"}
    drops = [a - b for a, b in zip(left, right)]
    history = [rates["per_domain"][domain]["GoldABCInHistoryRate"] for domain in DOMAIN_ORDER]
    p = pearson(history, drops)
    s = pearson(rankdata(history), rankdata(drops))
    classification = "MIXED" if p is None or s is None or p * s <= 0 else "POSITIVE" if p > 0 else "NEGATIVE"
    return {
        "classification": classification, "N_DOMAINS": 4, "DESCRIPTIVE_ONLY": "YES",
        "drops": dict(zip(DOMAIN_ORDER, drops)), "history_rates": dict(zip(DOMAIN_ORDER, history)),
        "pearson_r": p, "spearman_rho": s,
    }


def root_summary(
    rates: dict[str, Any], scores: dict[str, Any], hierarchy: dict[str, Any],
    retrieval: dict[str, Any], copy: dict[str, Any], decomposition: dict[str, Any],
) -> dict[str, Any]:
    associations = {
        "MiniFix_to_Gamma": association("MiniFix", "Gamma", rates, scores),
        "MiniFix_to_Sol": association("MiniFix", "Sol", rates, scores),
        "Beta_to_Step900": association("Beta", "Step900", rates, scores),
    }
    nonhistory = {}
    for model in ("MiniFix", "Gamma"):
        nonhistory[model] = {
            cell: {
                route: {
                    metric: hierarchy["models"][model][cell][route][metric]
                    for metric in ("AHit@32", "ABHit@32", "ABCHit@32", "AMRR", "ABMRR", "ABCMRR")
                }
                for route in ("BARE", "EXACT_BRIDGE")
            }
            for cell in ("SN", "UN")
        }
    video = rates["per_domain"]["video"]["GoldABCInHistoryRate"]
    other = [rates["per_domain"][domain]["GoldABCInHistoryRate"] for domain in DOMAIN_ORDER if domain != "video"]
    beta_video_lowest = scores["models"]["Beta"]["domain_scores"][0] == min(scores["models"]["Beta"]["domain_scores"])
    minifix_video_lowest = scores["models"]["MiniFix"]["domain_scores"][0] == min(scores["models"]["MiniFix"]["domain_scores"])
    video_class = "PARTIAL" if video < min(other) and beta_video_lowest and minifix_video_lowest else "INSUFFICIENT"
    return {
        "official_associations": associations,
        "nonhistory_hierarchy": nonhistory,
        "history_retrieval": retrieval,
        "bridge_copy": {
            key: copy[key]
            for key in ("pairs", "mean_delta", "increase_n", "decrease_n", "unchanged_n", "bridge_copy_direction", "bridge_is_not_a_simple_copy_switch")
        },
        "official_per_example_available": decomposition["official_per_example_available"],
        "official_gap_decomposition": "UNAVAILABLE",
        "history_floor_gap_estimate": "NOT_OFFICIAL_SCORE_DECOMPOSITION",
        "video_supports_novel_difficulty_hypothesis": video_class,
        "mini_data_supports_repeat_not_novel_hypothesis": "PARTIAL",
        "beta_true_novel_ability_identifiable": "NO",
        "history_repeat_floor_support": "SUPPORTED_LOCAL",
        "novel_recommendation_ability": "WEAK_WITHIN_BEAM32_LOCAL_PROXY",
        "hierarchical_recommendation_signal": "A_PRESENT_AB_LIMITED_ABC_ZERO_IN_NONHISTORY",
        "root_class": "LOCAL_HISTORY_FLOOR_SUPPORTED_OFFICIAL_ATTRIBUTION_UNRESOLVED",
        "primary_signal": "Exact-history cells retain a strong local floor; SN/UN have zero ABC Hit@32 while A-level signal remains.",
        "main_limitation": "No official per-example artifact and missing Gamma/Sol/Step900 official domain scores; local probes are small and distribution-specific.",
        "conclusion": "History repeat floor is supported locally, but project artifacts cannot attribute the official score gaps to history cases.",
        "future_gpu_experiment": "Teacher-forced Gold ABC log-prob on SeenNonHistory versus UnseenNonHistory.",
    }


def render_root_md(summary: dict[str, Any], rates: dict[str, Any], hierarchy: dict[str, Any]) -> str:
    lines = [
        "# History-floor root-cause audit",
        "",
        "This audit is CPU-only and reuses persisted records. It is not an official score decomposition.",
        "",
        "## Natural history composition",
        "",
        "| Domain | N | A overlap | AB overlap | exact ABC overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAIN_ORDER:
        value = rates["per_domain"][domain]
        lines.append(
            f"| {domain} | {value['N']} | {value['GoldAInHistoryRate']:.2%} | "
            f"{value['GoldABInHistoryRate']:.2%} | {value['GoldABCInHistoryRate']:.2%} |"
        )
    lines.extend(["", "## NonHistory Beam32 hierarchy", "", "| Model | Cell | Route | A@32 | AB@32 | ABC@32 |", "|---|---|---|---:|---:|---:|"])
    for model in ("MiniFix", "Gamma"):
        for cell in ("SN", "UN"):
            for route in ("BARE", "EXACT_BRIDGE"):
                value = hierarchy["models"][model][cell][route]
                lines.append(f"| {model} | {cell} | {route} | {value['AHit@32']:.2f} | {value['ABHit@32']:.2f} | {value['ABCHit@32']:.2f} |")
    lines.extend(
        [
            "", "## Decision", "",
            f"- Root class: `{summary['root_class']}`",
            f"- History floor: `{summary['history_repeat_floor_support']}`",
            f"- Novel recommendation: `{summary['novel_recommendation_ability']}`",
            f"- Hierarchy: `{summary['hierarchical_recommendation_signal']}`",
            f"- Video hypothesis: `{summary['video_supports_novel_difficulty_hypothesis']}`",
            "- Official attribution remains unresolved because no official per-example artifact was found and three comparison models lack domain score files.",
            "- Bridge is not a simple copy switch: persisted paired experiments show its history-candidate direction separately from Gold ranking.",
            "", "No next experiment was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def terminal(
    audit: dict[str, Any], rates: dict[str, Any], inventory: dict[str, Any], scores: dict[str, Any],
    hierarchy: dict[str, Any], retrieval: dict[str, Any], copy: dict[str, Any], summary: dict[str, Any],
) -> str:
    def score_text(model: str) -> str:
        value = scores["models"][model]
        return "MISSING" if value["domain_scores"] is None else json.dumps(value["domain_scores"])

    mf, ga = hierarchy["models"]["MiniFix"], hierarchy["models"]["Gamma"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short'] or 'EMPTY'}",
        f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}", f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}",
        f"RUNTIME_GITHUB_PARITY={audit['runtime_github_parity']}", "", "--- Natural History Rates ---", "",
    ]
    for domain in DOMAIN_ORDER:
        label, value = domain.upper(), rates["per_domain"][domain]
        lines.extend(
            [
                f"{label}_N={value['N']}", f"{label}_GOLD_A_HISTORY_RATE={value['GoldAInHistoryRate']:.9f}",
                f"{label}_GOLD_AB_HISTORY_RATE={value['GoldABInHistoryRate']:.9f}",
                f"{label}_GOLD_ABC_HISTORY_RATE={value['GoldABCInHistoryRate']:.9f}",
            ]
        )
    lines.extend(
        [
            "", "--- Official Logs ---", "", f"OFFICIAL_LOG_FILES_FOUND={inventory['count']}",
            "OFFICIAL_PER_EXAMPLE_AVAILABLE=NO", f"OFFICIAL_MAX_SCORE_CONFIRMED={scores['official_max_score_confirmed']}",
            f"BETA_REC_DOMAIN_SCORES={score_text('Beta')}", f"MINIFIX_REC_DOMAIN_SCORES={score_text('MiniFix')}",
            f"GAMMA_REC_DOMAIN_SCORES={score_text('Gamma')}", f"SOL_REC_DOMAIN_SCORES={score_text('Sol')}",
            f"STEP900_REC_DOMAIN_SCORES={score_text('Step900')}",
            f"MF_GAMMA_DROP_HISTORY_ASSOCIATION={summary['official_associations']['MiniFix_to_Gamma']['classification']}",
            f"BETA_STEP900_DROP_HISTORY_ASSOCIATION={summary['official_associations']['Beta_to_Step900']['classification']}",
            "", "--- Phase1.5.1 NonHistory Hierarchy ---", "",
        ]
    )
    for prefix, model_values in (("MF", mf), ("GA", ga)):
        for cell in ("SN", "UN"):
            for route in ("BARE", "EXACT_BRIDGE"):
                value = model_values[cell][route]
                route_label = "BRIDGE" if route == "EXACT_BRIDGE" else "BARE"
                lines.extend(
                    [
                        f"{prefix}_{cell}_{route_label}_A_HIT32={value['AHit@32']:.8f}",
                        f"{prefix}_{cell}_{route_label}_AB_HIT32={value['ABHit@32']:.8f}",
                        f"{prefix}_{cell}_{route_label}_ABC_HIT32={value['ABCHit@32']:.8f}",
                    ]
                )
    lines.extend(["", "--- History Retrieval ---", ""])
    for prefix, model in (("MF", "MiniFix"), ("GA", "Gamma")):
        value = retrieval[model]
        lines.extend(
            [
                f"{prefix}_BRIDGE_NEW_GOLD_ACQUISITION_COUNT={value['bridge_new_gold_acquisition_count']}",
                f"{prefix}_BRIDGE_EXISTING_GOLD_RERANK_COUNT={value['bridge_existing_gold_rerank_count']}",
                f"{prefix}_BRIDGE_LOST_GOLD_COUNT={value['bridge_lost_gold_count']}",
            ]
        )
    lines.extend(
        [
            f"BRIDGE_COPY_DIRECTION={copy['bridge_copy_direction']}",
            f"BRIDGE_IS_NOT_A_SIMPLE_COPY_SWITCH={copy['bridge_is_not_a_simple_copy_switch']}",
            "", "--- Interpretation ---", "",
            f"VIDEO_SUPPORTS_NOVEL_DIFFICULTY_HYPOTHESIS={summary['video_supports_novel_difficulty_hypothesis']}",
            f"MINI_DATA_SUPPORTS_REPEAT_NOT_NOVEL_HYPOTHESIS={summary['mini_data_supports_repeat_not_novel_hypothesis']}",
            f"BETA_TRUE_NOVEL_ABILITY_IDENTIFIABLE={summary['beta_true_novel_ability_identifiable']}",
            f"HISTORY_REPEAT_FLOOR_SUPPORT={summary['history_repeat_floor_support']}",
            f"NOVEL_RECOMMENDATION_ABILITY={summary['novel_recommendation_ability']}",
            f"HIERARCHICAL_RECOMMENDATION_SIGNAL={summary['hierarchical_recommendation_signal']}",
            f"ROOT_CLASS={summary['root_class']}", f"PRIMARY_SIGNAL={summary['primary_signal']}",
            f"MAIN_LIMITATION={summary['main_limitation']}", f"CONCLUSION={summary['conclusion']}",
            f"FUTURE_GPU_EXPERIMENT={summary['future_gpu_experiment']}", "",
            "GPU_INFERENCE_STARTED=NO", "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0",
            "SELF_COT_GENERATION_STARTED=NO", "EXTERNAL_EVAL_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    golds = {("video", 1, 2, 3), ("video", 4, 5, 6)}
    record = {
        "beams": [
            {"beam_index": 0, "predicted_sid": None, "copy_class": None},
            {"beam_index": 1, "predicted_sid": ["video", 1, 9, 9], "copy_class": "NOVEL"},
            {"beam_index": 4, "predicted_sid": ["video", 4, 5, 9], "copy_class": "AB_COPY"},
            {"beam_index": 9, "predicted_sid": ["video", 1, 2, 3], "copy_class": "EXACT_COPY"},
        ]
    }
    value = hierarchy_metrics(record, golds)
    assert value["ARank"] == 2 and value["ABRank"] == 5 and value["ABCRank"] == 10
    assert value["AHit@1"] == 0 and value["AHit@5"] == 1 and value["ABCHit@5"] == 0
    assert math.isclose(float(value["ABCMRR"]), 0.1)
    averaged = average_metrics([{"rank": None, "hit": 0}, {"rank": 2, "hit": 1}])
    assert averaged == {"N": 2, "hit": 0.5}
    assert rankdata([3, 1, 1, 2]) == [4.0, 1.5, 1.5, 3.0]
    assert math.isclose(float(pearson([1, 2, 3], [2, 4, 6])), 1.0)
    print("CPU_SELF_TEST_PASS=YES")


def run() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU_ONLY_GATE_REQUIRES_CUDA_VISIBLE_DEVICES_EMPTY")
    if "torch" in sys.modules:
        raise RuntimeError("CPU_ONLY_GATE_TORCH_IMPORTED")
    audit = code_audit()
    before = protected_hashes()
    natural, source_contract = natural_rows()
    rates = summarize_history(natural)
    inventory, scores, decomposition = official_inventory()
    hierarchy, per_record = phase151_hierarchy()
    retrieval = history_retrieval(per_record)
    evidence = existing_evidence_table()
    copy = bridge_copy_analysis()
    summary = root_summary(rates, scores, hierarchy, retrieval, copy, decomposition)
    after = protected_hashes()
    if before != after:
        raise RuntimeError("PROTECTED_RESULT_SHA256_CHANGED")
    source_contract["protected_result_sha256_before"] = before
    source_contract["protected_result_sha256_after"] = after
    source_contract["protected_results_unchanged"] = True
    source_contract["cpu_only"] = True
    source_contract["torch_imported"] = False
    source_contract["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "source_contract_audit.json", source_contract)
    write_json(OUTPUT / "domain_history_rates.json", rates)
    (OUTPUT / "domain_history_rates.md").write_text(render_history_md(rates), encoding="utf-8")
    write_json(OUTPUT / "official_log_inventory.json", inventory)
    write_json(OUTPUT / "official_score_table.json", scores)
    write_json(OUTPUT / "official_history_decomposition_unavailable.json", decomposition)
    write_json(OUTPUT / "existing_local_evidence_table.json", evidence)
    write_json(OUTPUT / "phase151_hierarchy_analysis.json", hierarchy)
    (OUTPUT / "phase151_hierarchy_analysis.md").write_text(render_hierarchy_md(hierarchy), encoding="utf-8")
    write_json(OUTPUT / "bridge_copy_direction_analysis.json", copy)
    write_json(OUTPUT / "history_floor_root_cause_summary.json", summary)
    (OUTPUT / "history_floor_root_cause_summary.md").write_text(
        render_root_md(summary, rates, hierarchy), encoding="utf-8"
    )
    review = terminal(audit, rates, inventory, scores, hierarchy, retrieval, copy, summary)
    (OUTPUT / "CHATGPT_HISTORY_FLOOR_CPU_REVIEW.txt").write_text(review + "\n", encoding="utf-8")
    print(review)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 1:
        run()
    else:
        raise SystemExit("usage: recommendation_history_floor_cpu_audit.py [--self-test]")
