"""Phase 1.5.1: train-memory x history-repeat 2x2 quick probe."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Iterable

import torch

RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/"
    "recommendation_memory_history_2x2_probe.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_memory_history_2x2_probe.py"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]

from boundary_adapt.bridge_inside_sft.build_dataset import load_unique_source  # noqa: E402
from boundary_adapt.bridge_inside_sft.common import (  # noqa: E402
    BASE, DOMAIN, DOMAIN_ORDER, SOURCE, SOURCE_SHA256, extract_response, file_sha,
    infer_domain, source_metadata, stable_hash, token_ids_sha,
)
from boundary_adapt.diagnostics.controlled_generator_decoder_crossover import load_model, strict_beam  # noqa: E402
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import history_from_prompt, parse_gold  # noqa: E402
from boundary_adapt.diagnostics.recommendation_official_prompt_bare_crossover import render_prompt_ids  # noqa: E402
from boundary_adapt.diagnostics import recommendation_bridge_memory_quick_probe as quick  # noqa: E402

OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_memory_history_2x2_probe"
PARTS = OUTPUT / "parts"
PREVIOUS = RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_quick_probe"
SPLIT_MANIFEST = Path("/root/GRPO_audit_results/bridge_inside_transition_sft_v1_20260824/split_manifest.json")
CANONICAL_ROWS = RUNTIME / "boundary_adapt/results/boundary_adapt_rows.jsonl"
MODELS = ("MiniFix", "Gamma")
CELLS = ("SH", "SN", "UH", "UN")
CELL_LABELS = {
    "SH": "TRAIN_SEEN|GOLD_IN_HISTORY", "SN": "TRAIN_SEEN|GOLD_NOT_IN_HISTORY",
    "UH": "TRAIN_UNSEEN|GOLD_IN_HISTORY", "UN": "TRAIN_UNSEEN|GOLD_NOT_IN_HISTORY",
}
SELECTION_SALT = "recommendation_memory_history_2x2_probe|"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260825
import re

SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True).strip()


def code_audit() -> dict[str, Any]:
    commit, origin, status = git("rev-parse", "HEAD"), git("rev-parse", "origin/main"), git("status", "--short")
    github_sha, runtime_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT), file_sha(RUNTIME_SCRIPT)
    value = {
        "implement_commit": commit, "origin_main": origin,
        "push_status": "PASS" if commit == origin else "FAIL", "git_status_short": status,
        "github_script_sha256": github_sha, "runtime_script_sha256": runtime_sha,
        "runtime_github_parity": "PASS" if github_sha == runtime_sha else "FAIL",
    }
    if value["push_status"] != "PASS" or status or value["runtime_github_parity"] != "PASS":
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={value}")
    return value


def row_sha(row: dict[str, Any]) -> str:
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def sid_sequence(text: str) -> list[str]:
    return [match.group(0) for match in SID_RE.finditer(text)]


def cell_name(seen: bool, history: bool) -> str:
    return ("S" if seen else "U") + ("H" if history else "N")


def natural_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = read_json(SPLIT_MANIFEST)
    holdout = set(map(str, split["holdout_group_ids"]))
    if file_sha(SOURCE) != SOURCE_SHA256 or split["source_sha256"] != SOURCE_SHA256 or len(holdout) != 1595:
        raise RuntimeError("NATURAL_SOURCE_CONTRACT_FAIL")
    raw_rows, duplicates = load_unique_source(SOURCE, CANONICAL_ROWS)
    rows = []
    for raw in raw_rows:
        metadata = source_metadata(raw)
        group = str(metadata["recommendation_group_id"])
        if group not in holdout:
            continue
        domain = infer_domain(metadata)
        cot_before_close, bridge, answer_abc = extract_response(str(raw["output"]), domain)
        user = str(raw.get("instruction", "")) + str(raw.get("input", ""))
        history = quick.extract_history(user)
        all_history = sid_sequence(history)
        target_history = [value for value in all_history if value.startswith(DOMAIN[domain])]
        golds = list(map(str, metadata["recommendation_all_gold_sids"]))
        rows.append({
            "group_id": group, "domain": domain, "all_gold_sids": golds,
            "current_gold_sid": str(metadata["recommendation_current_gold_sid"]), "K": len(golds),
            "gold_sid_in_history": bool(set(golds) & set(target_history)),
            "history": history, "history_sid_sequence": all_history,
            "target_domain_history_sids": target_history, "target_domain_history_sid_count": len(target_history),
            "original_system": str(raw["system"]), "original_user": user,
            "frozen_cot": cot_before_close + "</think>", "cot_before_close": cot_before_close,
            "exact_bridge": bridge, "source_answer_sid": DOMAIN[domain] + answer_abc,
            "source_row_sha256": row_sha(raw),
        })
    if len(rows) != 1595 or {row["group_id"] for row in rows} != holdout:
        raise RuntimeError("NATURAL_1595_RECONSTRUCTION_FAIL")
    return rows, {"groups": len(rows), "legacy_duplicates_dropped": duplicates, "source": str(SOURCE)}


def training_membership() -> tuple[dict[str, set[str]], dict[str, Any]]:
    scans = {label: quick.scan_membership(label) for label in MODELS}
    sets = {label: scan["groups"] for label, scan in scans.items()}
    intersection, union = sets["MiniFix"] & sets["Gamma"], sets["MiniFix"] | sets["Gamma"]
    if len(sets["MiniFix"]) != 1549 or len(sets["Gamma"]) != 1549 or len(intersection) != 1549:
        raise RuntimeError("TRAIN_MEMBERSHIP_CONTRACT_FAIL")
    return sets, {
        "minifix_groups": len(sets["MiniFix"]), "gamma_groups": len(sets["Gamma"]),
        "intersection": len(intersection), "minifix_only": len(sets["MiniFix"] - sets["Gamma"]),
        "gamma_only": len(sets["Gamma"] - sets["MiniFix"]),
        "train_seen_definition": "group in both Mini training recommendation sets",
        "train_unseen_definition": "UNSEEN_FROM_MINI_TRAIN: group absent from both Mini training recommendation sets",
        "intersection_ids": sorted(intersection), "union_ids": sorted(union),
    }


def provenance_audit(natural_by_group: dict[str, dict[str, Any]], seen_ids: set[str]) -> dict[str, dict[str, Any]]:
    candidates = {group: natural_by_group[group] for group in seen_ids if group in natural_by_group}
    attempts = defaultdict(list)
    with quick.DATASETS["MiniFix"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend" or row.get("source_segment") != "recommendation_cot":
                continue
            metadata = source_metadata(row)
            group = str(metadata["recommendation_group_id"])
            if group not in candidates:
                continue
            source = candidates[group]
            try:
                domain = infer_domain(metadata)
                cot_before_close, bridge, answer_abc = extract_response(str(row["output"]), domain)
                answer = DOMAIN[domain] + answer_abc
                checks = {
                    "group_id_match": group == source["group_id"],
                    "history_business_sequence_match": sid_sequence(str(row.get("instruction", "")) + str(row.get("input", ""))) == source["history_sid_sequence"],
                    "target_domain_match": domain == source["domain"],
                    "gold_set_match": set(map(str, metadata["recommendation_all_gold_sids"])) == set(source["all_gold_sids"]),
                    "cot_body_byte_match": cot_before_close == source["cot_before_close"],
                    "bridge_byte_match": bridge == source["exact_bridge"],
                    "answer_in_all_gold": answer in set(source["all_gold_sids"]),
                }
            except Exception as error:
                checks = {"parse_error": repr(error)}
            attempts[group].append({"row_sha256": row_sha(row), "checks": checks})
    result = {}
    required = (
        "group_id_match", "history_business_sequence_match", "target_domain_match", "gold_set_match",
        "cot_body_byte_match", "bridge_byte_match", "answer_in_all_gold",
    )
    for group, source in candidates.items():
        passing = [attempt for attempt in attempts[group] if all(attempt["checks"].get(key) for key in required)]
        best = passing[0] if passing else (attempts[group][0] if attempts[group] else {"row_sha256": None, "checks": {}})
        result[group] = {
            "group_id": group, "domain": source["domain"], "pass": bool(passing),
            "minifix_train_cot_match": bool(best["checks"].get("cot_body_byte_match")),
            "minifix_train_bridge_byte_match": bool(best["checks"].get("bridge_byte_match")),
            "minifix_train_history_match": bool(best["checks"].get("history_business_sequence_match")),
            "matched_training_row_sha256": passing[0]["row_sha256"] if passing else None,
            "training_rows_checked": len(attempts[group]), "representative_checks": best["checks"],
        }
    return result


def pool_counts(rows: list[dict[str, Any]], intersection: set[str], union: set[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    pools = {cell: [] for cell in CELLS}
    one_model_only = 0
    for row in rows:
        group = row["group_id"]
        if group in intersection:
            seen = True
        elif group not in union:
            seen = False
        else:
            one_model_only += 1
            continue
        cell = cell_name(seen, row["gold_sid_in_history"])
        pools[cell].append({**row, "cell": cell, "membership": "TRAIN_SEEN" if seen else "TRAIN_UNSEEN"})
    per_domain = {
        domain: {cell: sum(row["domain"] == domain for row in pools[cell]) for cell in CELLS}
        for domain in DOMAIN_ORDER
    }
    return {
        "per_domain": per_domain, "global": {cell: len(pools[cell]) for cell in CELLS},
        "one_model_only_excluded": one_model_only,
    }, pools


def select_probe(pools: dict[str, list[dict[str, Any]]], provenance: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = {
        cell: [row for row in rows if not cell.startswith("S") or provenance.get(row["group_id"], {}).get("pass")]
        for cell, rows in pools.items()
    }
    def ordered(values: list[dict[str, Any]], cell: str) -> list[dict[str, Any]]:
        return sorted(values, key=lambda row: (stable_hash(SELECTION_SALT + cell + "|" + row["group_id"]), row["group_id"]))
    full_grid = all(any(row["domain"] == domain for row in eligible[cell]) for domain in DOMAIN_ORDER for cell in CELLS)
    selected = []
    if full_grid:
        mode = "PER_DOMAIN_FULL_2X2"
        for domain in DOMAIN_ORDER:
            for cell in CELLS:
                selected.append(ordered([row for row in eligible[cell] if row["domain"] == domain], cell)[0])
    else:
        mode = "GLOBAL_CELL_BALANCED_FALLBACK"
        for cell in CELLS:
            chosen = []
            for domain in DOMAIN_ORDER:
                values = ordered([row for row in eligible[cell] if row["domain"] == domain], cell)
                if values:
                    chosen.append(values[0])
            if len(chosen) < 4:
                remaining = ordered([row for row in eligible[cell] if row["group_id"] not in {value["group_id"] for value in chosen}], cell)
                chosen.extend(remaining[:4 - len(chosen)])
            if len(chosen) != 4:
                raise RuntimeError(f"INSUFFICIENT_ELIGIBLE_CELL cell={cell} n={len(chosen)}")
            selected.extend(chosen)
    if len(selected) != 16 or Counter(row["cell"] for row in selected) != Counter({cell: 4 for cell in CELLS}):
        raise RuntimeError("PROBE16_CELL_BALANCE_FAIL")
    selected.sort(key=lambda row: (CELLS.index(row["cell"]), DOMAIN_ORDER.index(row["domain"]), row["group_id"]))
    return selected, {
        "sampling_mode": mode, "salt": SELECTION_SALT,
        "eligible_counts": {cell: len(eligible[cell]) for cell in CELLS},
        "domain_composition": {cell: dict(Counter(row["domain"] for row in selected if row["cell"] == cell)) for cell in CELLS},
    }


def prepare() -> None:
    from transformers import AutoTokenizer

    audit = code_audit()
    if not (PREVIOUS / "records.jsonl").is_file() or not (PREVIOUS / "seen_unseen_summary.json").is_file():
        raise RuntimeError("PHASE15_QUICK_PRESERVATION_FAIL")
    previous_sha = {name: file_sha(PREVIOUS / name) for name in ("records.jsonl", "seen_unseen_summary.json")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory = quick.build_model_inventory()
    inventory.update({"implement_commit": audit["implement_commit"], "previous_phase15_preserved_sha256": previous_sha})
    write_json(OUTPUT / "model_inventory.json", inventory)
    sets, membership = training_membership()
    natural, source_audit = natural_rows()
    natural_by_group = {row["group_id"]: row for row in natural}
    intersection, union = sets["MiniFix"] & sets["Gamma"], sets["MiniFix"] | sets["Gamma"]
    counts, pools = pool_counts(natural, intersection, union)
    provenance = provenance_audit(natural_by_group, intersection)
    selected, selection = select_probe(pools, provenance)
    counts.update({
        "membership": {key: value for key, value in membership.items() if not key.endswith("_ids")},
        "source_audit": source_audit, "selection": selection,
        "probe_is_natural_distribution": False,
    })
    write_json(OUTPUT / "pool_audit.json", counts)

    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("THINK_CLOSE_NOT_ATOMIC")
    items = []
    for row in selected:
        official_user = "\u7528\u6237\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff1a\n" + row["history"] + "\n\n" + quick.QUESTION[row["domain"]]
        prompt_ids = render_prompt_ids(tokenizer, official_user, quick.SYSTEM[row["domain"]])
        cot_ids = list(map(int, tokenizer.encode(row["frozen_cot"], add_special_tokens=False)))
        bridge_ids = list(map(int, tokenizer.encode(row["exact_bridge"], add_special_tokens=False)))
        domain_ids = list(map(int, tokenizer.encode(DOMAIN[row["domain"]], add_special_tokens=False)))
        if cot_ids[-1:] != close_ids or not bridge_ids or len(domain_ids) != 1:
            raise RuntimeError(f"TOKEN_BOUNDARY_FAIL group={row['group_id']}")
        items.append({
            **{key: row[key] for key in (
                "group_id", "domain", "cell", "membership", "gold_sid_in_history", "all_gold_sids", "K",
                "target_domain_history_sid_count", "target_domain_history_sids", "source_row_sha256",
            )},
            "official_system": quick.SYSTEM[row["domain"]], "official_prompt": official_user,
            "official_prompt_token_ids": prompt_ids,
            "frozen_cot_source": "BATA_SOURCE", "frozen_cot_token_ids": cot_ids,
            "frozen_cot_sha256": token_ids_sha(cot_ids), "exact_bridge": row["exact_bridge"],
            "exact_bridge_token_ids": bridge_ids, "exact_bridge_sha256": token_ids_sha(bridge_ids),
            "domain_token_ids": domain_ids,
        })
    seen_audit = [provenance[item["group_id"]] for item in items if item["cell"].startswith("S")]
    if len(seen_audit) != 8 or not all(row["pass"] for row in seen_audit):
        raise RuntimeError("SELECTED_SEEN_PROVENANCE_FAIL")
    write_json(OUTPUT / "seen_bridge_provenance_audit.json", {
        "seen_groups": 8, "seen_bridge_provenance_pass": "PASS", "seen_cot_provenance_pass": "PASS",
        "seen_history_provenance_pass": "PASS", "items": seen_audit,
    })

    prompt_items = []
    for domain in DOMAIN_ORDER:
        for history_flag in (True, False):
            choices = sorted(
                (row for row in natural if row["domain"] == domain and row["gold_sid_in_history"] == history_flag),
                key=lambda row: (stable_hash(SELECTION_SALT + "PROMPT|" + row["group_id"]), row["group_id"]),
            )
            if not choices:
                raise RuntimeError(f"PROMPT_AUDIT_POOL_EMPTY domain={domain} history={history_flag}")
            row = choices[0]
            user = "\u7528\u6237\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff1a\n" + row["history"] + "\n\n" + quick.QUESTION[domain]
            proxy, soft = render_prompt_ids(tokenizer, user, quick.SYSTEM[domain]), quick.soft_switch_ids(tokenizer, user, quick.SYSTEM[domain])
            prompt_items.append({
                "group_id": row["group_id"], "domain": domain, "gold_sid_in_history": history_flag,
                "token_identical": proxy == soft, "history_sid_parity": sid_sequence(user) == row["history_sid_sequence"],
                "proxy_sha256": token_ids_sha(proxy), "soft_switch_sha256": token_ids_sha(soft),
            })
    if len(prompt_items) != 8 or not all(row["token_identical"] and row["history_sid_parity"] for row in prompt_items):
        raise RuntimeError("PROMPT_AUDIT_FAIL")
    write_json(OUTPUT / "prompt_audit.json", {"prompt_token_audit_pass": "PASS", "history_sid_parity": "PASS", "items": prompt_items})
    manifest = {
        "version": "recommendation_memory_history_2x2_probe", "implement_commit": audit["implement_commit"],
        "sampling_mode": selection["sampling_mode"], "groups": 16,
        "cell_counts": dict(Counter(row["cell"] for row in items)),
        "domain_composition": selection["domain_composition"],
        "cell_k_counts": {cell: dict(Counter("K1" if row["K"] == 1 else "K2PLUS" for row in items if row["cell"] == cell)) for cell in CELLS},
        "probe_is_natural_distribution": False, "previous_phase15_sha256": previous_sha, "items": items,
        "training_started": False, "self_cot_generation": False, "external_eval": False,
    }
    write_json(OUTPUT / "probe16_manifest.json", manifest)
    print("CPU_PREFLIGHT_PASS=YES")
    for domain in DOMAIN_ORDER:
        print(f"POOL_{domain.upper()}=" + json.dumps(counts["per_domain"][domain], sort_keys=True))
    print(f"SAMPLING_MODE={manifest['sampling_mode']}")
    print(f"DOMAIN_COMPOSITION={manifest['domain_composition']}")
    print(f"CELL_K_COUNTS={manifest['cell_k_counts']}")


def run(model_label: str) -> None:
    audit, manifest = code_audit(), read_json(OUTPUT / "probe16_manifest.json")
    if manifest["implement_commit"] != audit["implement_commit"]:
        raise RuntimeError("MANIFEST_COMMIT_MISMATCH")
    if read_json(OUTPUT / "seen_bridge_provenance_audit.json")["seen_bridge_provenance_pass"] != "PASS":
        raise RuntimeError("GPU_BLOCKED_BY_PROVENANCE")
    rank, world = int(os.environ.get("LOCAL_RANK", "-1")), int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("PROBE_REQUIRES_4_LOCAL_RANKS")
    torch.cuda.set_device(rank)
    model, tokenizer = load_model(quick.MODEL_PATHS[model_label], f"cuda:{rank}")
    local = []
    for index, item in enumerate(manifest["items"]):
        if index % world != rank:
            continue
        prompt, cot, bridge, domain_ids = (
            list(map(int, item["official_prompt_token_ids"])), list(map(int, item["frozen_cot_token_ids"])),
            list(map(int, item["exact_bridge_token_ids"])), list(map(int, item["domain_token_ids"])),
        )
        history = history_from_prompt(tokenizer, prompt)
        golds = {parse_gold(value) for value in item["all_gold_sids"]}
        for condition, context in (("BARE", prompt + cot + domain_ids), ("EXACT_BRIDGE", prompt + cot + bridge + domain_ids)):
            result = strict_beam(model, tokenizer, context, item["domain"], golds, history)
            local.append({
                "implement_commit": audit["implement_commit"], "model": model_label,
                "group_id": item["group_id"], "domain": item["domain"], "cell": item["cell"],
                "gold_sid_in_history": item["gold_sid_in_history"], "K": item["K"],
                "condition": condition, "context_sha256": token_ids_sha(context), **result,
            })
        print(f"MEMORY_HISTORY_PROGRESS model={model_label} rank={rank} group={index // world + 1}/4", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    write_jsonl(PARTS / f"{model_label}_rank{rank}.jsonl", local)


def paired_metrics(index: dict[tuple[str, str, str], dict[str, Any]], model: str,
                   items: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for group, item in items.items():
        bare, bridge = index[(model, group, "BARE")], index[(model, group, "EXACT_BRIDGE")]
        bm, rm = quick.case_metrics(bare, item), quick.case_metrics(bridge, item)
        result[group] = {
            "BareMRR": bm["GoldMRR"], "BridgeMRR": rm["GoldMRR"], "BridgeGain_MRR": rm["GoldMRR"] - bm["GoldMRR"],
            "BareHit32": bm["GoldSIDHit@32"], "BridgeHit32": rm["GoldSIDHit@32"], "BridgeGain_Hit32": rm["GoldSIDHit@32"] - bm["GoldSIDHit@32"],
            "BareHistoryFraction": bm["HistoryCandidateFraction"], "BridgeHistoryFraction": rm["HistoryCandidateFraction"],
            "DeltaHistoryFraction": rm["HistoryCandidateFraction"] - bm["HistoryCandidateFraction"],
            **quick.pair_response(bare, bridge),
            **{f"Bare_{key}": value for key, value in bm.items()}, **{f"Bridge_{key}": value for key, value in rm.items()},
        }
    return result


def percentile(values: list[float], q: float) -> float:
    ordered, position = sorted(values), (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def bootstrap_cell(groups: dict[str, dict[str, float]], label: str) -> dict[str, Any]:
    ids = sorted(groups)
    keys = ("BridgeGain_MRR", "BridgeGain_Hit32", "DeltaHistoryFraction", "Response1MinusJaccard", "Top1Flip")
    seed = (BOOTSTRAP_SEED + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)) % (2**32)
    rng, samples = random.Random(seed), {key: [] for key in keys}
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = [rng.choice(ids) for _ in ids]
        for key in keys:
            samples[key].append(statistics.fmean(groups[group][key] for group in chosen))
    return {"groups": len(ids), "replicates": BOOTSTRAP_REPLICATES, "seed": seed, "metrics": {
        key: {"estimate": statistics.fmean(groups[group][key] for group in ids), "ci95": [percentile(samples[key], .025), percentile(samples[key], .975)]}
        for key in keys
    }}


def contrast_bootstrap(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]], metric: str, label: str) -> dict[str, Any]:
    lids, rids = sorted(left), sorted(right)
    seed = (BOOTSTRAP_SEED + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)) % (2**32)
    rng, values = random.Random(seed), []
    for _ in range(BOOTSTRAP_REPLICATES):
        lsample, rsample = [rng.choice(lids) for _ in lids], [rng.choice(rids) for _ in rids]
        values.append(statistics.fmean(left[group][metric] for group in lsample) - statistics.fmean(right[group][metric] for group in rsample))
    estimate = statistics.fmean(row[metric] for row in left.values()) - statistics.fmean(row[metric] for row in right.values())
    return {"estimate": estimate, "ci95": [percentile(values, .025), percentile(values, .975)], "direction_only_not_significance": True}


def hypothesis(summary: dict[str, Any], contrasts: dict[str, Any]) -> dict[str, str]:
    mf, ga = summary["MiniFix"], summary["Gamma"]
    memory_values = [contrasts[model]["train_memory"]["BridgeGain_MRR"] for model in MODELS]
    memory_bridge = [mf["SN"]["BridgeMRR"] - mf["UN"]["BridgeMRR"], ga["SN"]["BridgeMRR"] - ga["UN"]["BridgeMRR"]]
    if all(value >= .05 for value in memory_values) or all(value >= .05 for value in memory_bridge):
        group_memory = "STRONG"
    elif any(value >= .05 for value in memory_values + memory_bridge):
        group_memory = "MODERATE"
    else:
        group_memory = "WEAK_OR_UNSUPPORTED"
    history_values = [contrasts[model]["history_shortcut"]["BridgeGain_MRR"] for model in MODELS]
    history_bridge = [mf["UH"]["BridgeMRR"] - mf["UN"]["BridgeMRR"], ga["UH"]["BridgeMRR"] - ga["UN"]["BridgeMRR"]]
    if all(value >= .05 for value in history_values) or all(value >= .05 for value in history_bridge):
        history_support = "STRONG"
    elif any(value >= .05 for value in history_values + history_bridge):
        history_support = "MODERATE"
    else:
        history_support = "WEAK_OR_UNSUPPORTED"
    un_mrr = [mf["UN"][key] for key in ("BareMRR", "BridgeMRR")] + [ga["UN"][key] for key in ("BareMRR", "BridgeMRR")]
    un_response = [mf["UN"]["Response1MinusJaccard"], ga["UN"]["Response1MinusJaccard"]]
    novel = "WEAK_IN_LOCAL_PROXY" if max(un_mrr) <= .02 and min(un_response) >= .40 else "DETECTABLE" if max(un_mrr) > .02 else "INCONCLUSIVE"
    memory_diff = contrasts["specialization"]["memory_diff"]
    history_diff = contrasts["specialization"]["history_diff"]
    specialization = "STRONG" if abs(memory_diff) >= .05 and abs(history_diff) >= .05 else "MODERATE" if max(abs(memory_diff), abs(history_diff)) >= .05 else "WEAK_OR_UNSUPPORTED"
    if group_memory == "STRONG" and history_support == "STRONG" and novel == "WEAK_IN_LOCAL_PROXY":
        story = "STORY_1_MEMORY_PLUS_HISTORY_FLOOR"
    elif mf["UN"]["BridgeGain_MRR"] - ga["UN"]["BridgeGain_MRR"] >= .03 and mf["UN"]["BridgeMRR"] > .03:
        story = "STORY_2_BRIDGE_GENERALIZES_RECOMMENDATION"
    elif novel == "WEAK_IN_LOCAL_PROXY" and statistics.fmean(un_response) >= .40 and specialization == "WEAK_OR_UNSUPPORTED":
        story = "STORY_3_GENERIC_CONTEXT_ONLY"
    else:
        story = "STORY_4_MIXED"
    return {
        "GROUP_MEMORY_SUPPORT": group_memory, "HISTORY_REPEAT_FLOOR_SUPPORT": history_support,
        "NOVEL_RECOMMENDATION_ABILITY": novel, "OLD_BRIDGE_TRAINED_SPECIALIZATION_SUPPORT": specialization,
        "PRIMARY_STORY": story,
    }


def finalize() -> None:
    audit, manifest = code_audit(), read_json(OUTPUT / "probe16_manifest.json")
    if {name: file_sha(PREVIOUS / name) for name in manifest["previous_phase15_sha256"]} != manifest["previous_phase15_sha256"]:
        raise RuntimeError("PREVIOUS_PHASE15_RESULTS_CHANGED")
    items = {row["group_id"]: row for row in manifest["items"]}
    rows = []
    for model in MODELS:
        for rank in range(4):
            path = PARTS / f"{model}_rank{rank}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"MISSING_PART={path}")
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    index = {(row["model"], row["group_id"], row["condition"]): row for row in rows}
    invalid = sum(beam.get("predicted_sid") is None for row in rows for beam in row["beams"])
    if len(rows) != 64 or len(index) != 64 or invalid != 0 or any(len(row["beams"]) != 32 for row in rows):
        raise RuntimeError(f"RECORD_CONTRACT_FAIL rows={len(rows)} keys={len(index)} invalid={invalid}")
    rows.sort(key=lambda row: (MODELS.index(row["model"]), row["group_id"], row["condition"]))
    write_jsonl(OUTPUT / "records.jsonl", rows)
    per_group, summary, boot = {}, {}, {}
    for model in MODELS:
        paired = paired_metrics(index, model, items)
        per_group[model], summary[model], boot[model] = paired, {}, {}
        for cell in CELLS:
            subset = {group: value for group, value in paired.items() if items[group]["cell"] == cell}
            summary[model][cell] = {"N": len(subset), **quick.mean_metrics(list(subset.values()))}
            boot[model][cell] = bootstrap_cell(subset, f"{model}|{cell}")
    write_json(OUTPUT / "per_group_metrics.json", per_group)
    contrasts = {}
    for model in MODELS:
        sn, un = summary[model]["SN"], summary[model]["UN"]
        uh = summary[model]["UH"]
        contrasts[model] = {
            "train_memory": {
                "BridgeGain_MRR": sn["BridgeGain_MRR"] - un["BridgeGain_MRR"],
                "BareMRR": sn["BareMRR"] - un["BareMRR"], "BridgeMRR": sn["BridgeMRR"] - un["BridgeMRR"],
            },
            "history_shortcut": {
                "BridgeGain_MRR": uh["BridgeGain_MRR"] - un["BridgeGain_MRR"],
                "BareMRR": uh["BareMRR"] - un["BareMRR"], "BridgeMRR": uh["BridgeMRR"] - un["BridgeMRR"],
                "BareHistoryFraction": uh["BareHistoryFraction"] - un["BareHistoryFraction"],
                "BridgeHistoryFraction": uh["BridgeHistoryFraction"] - un["BridgeHistoryFraction"],
            },
        }
        boot[model]["train_memory_contrast"] = contrast_bootstrap(
            {g: v for g, v in per_group[model].items() if items[g]["cell"] == "SN"},
            {g: v for g, v in per_group[model].items() if items[g]["cell"] == "UN"}, "BridgeGain_MRR", f"{model}|MEMORY",
        )
        boot[model]["history_shortcut_contrast"] = contrast_bootstrap(
            {g: v for g, v in per_group[model].items() if items[g]["cell"] == "UH"},
            {g: v for g, v in per_group[model].items() if items[g]["cell"] == "UN"}, "BridgeGain_MRR", f"{model}|HISTORY",
        )
    contrasts["specialization"] = {
        "memory_diff": contrasts["MiniFix"]["train_memory"]["BridgeGain_MRR"] - contrasts["Gamma"]["train_memory"]["BridgeGain_MRR"],
        "history_diff": contrasts["MiniFix"]["history_shortcut"]["BridgeGain_MRR"] - contrasts["Gamma"]["history_shortcut"]["BridgeGain_MRR"],
    }
    decisions = hypothesis(summary, contrasts)
    write_json(OUTPUT / "cell_summary.json", {"models": summary, "invalid_beams": invalid, "hypotheses": decisions})
    write_json(OUTPUT / "contrasts.json", contrasts)
    write_json(OUTPUT / "bootstrap.json", {"N_TOO_SMALL_FOR_FORMAL_SIGNIFICANCE": "YES", "models": boot})

    lines = [
        "# Train-Memory x History-Repeat 2x2 Probe", "",
        "This is an artificially balanced causal mechanism probe, not a natural distribution or official-score simulation.", "",
        "| Model | Cell | Bare MRR | Bridge MRR | Gain | 1-Jaccard | Top1 flip |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for cell in CELLS:
            value = summary[model][cell]
            lines.append(f"| {model} | {cell} | {value['BareMRR']:.5f} | {value['BridgeMRR']:.5f} | {value['BridgeGain_MRR']:+.5f} | {value['Response1MinusJaccard']:.5f} | {value['Top1Flip']:.5f} |")
    lines.extend(["", "## Decisions", "", *[f"- {key}: {value}" for key, value in decisions.items()], "", "Each cell has N=4. Bootstrap intervals are descriptive only."])
    (OUTPUT / "cell_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = final_review(audit, manifest, summary, contrasts, decisions)
    (OUTPUT / "CHATGPT_MEMORY_HISTORY_2X2_REVIEW.txt").write_text(review + "\n", encoding="utf-8")
    print(review)


def final_review(audit: dict[str, Any], manifest: dict[str, Any], summary: dict[str, Any], contrasts: dict[str, Any], decisions: dict[str, str]) -> str:
    mf, ga = summary["MiniFix"], summary["Gamma"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT",
        "PUSH_STATUS=PASS", "GIT_STATUS_SHORT=EMPTY", f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}",
        f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}", "RUNTIME_GITHUB_PARITY=PASS", "", "--- Sampling ---",
        f"SAMPLING_MODE={manifest['sampling_mode']}", "GROUPS=16", "SEEN_HISTORY_N=4", "SEEN_NONHISTORY_N=4",
        "UNSEEN_HISTORY_N=4", "UNSEEN_NONHISTORY_N=4", f"DOMAIN_COMPOSITION={manifest['domain_composition']}",
        "PROBE_IS_NATURAL_DISTRIBUTION=NO", "", "--- Provenance ---", "SEEN_BRIDGE_PROVENANCE_PASS=PASS",
        "SEEN_COT_PROVENANCE_PASS=PASS", "SEEN_HISTORY_PROVENANCE_PASS=PASS", "PROMPT_TOKEN_AUDIT_PASS=PASS",
    ]
    for label, values in (("MF", mf), ("GA", ga)):
        lines.extend(["", f"--- {'MiniFix' if label == 'MF' else 'Gamma'} 2x2 ---"])
        for cell in CELLS:
            value = values[cell]
            lines.extend([f"{label}_{cell}_BARE_MRR={value['BareMRR']:.8f}", f"{label}_{cell}_BRIDGE_MRR={value['BridgeMRR']:.8f}", f"{label}_{cell}_GAIN={value['BridgeGain_MRR']:.8f}"])
            if cell == "UN":
                lines.append(f"{label}_UN_1_MINUS_JACCARD={value['Response1MinusJaccard']:.8f}")
    lines.extend([
        "", "--- Contrasts ---",
        f"MF_TRAIN_MEMORY_CONTRAST_MRR={contrasts['MiniFix']['train_memory']['BridgeGain_MRR']:.8f}",
        f"GA_TRAIN_MEMORY_CONTRAST_MRR={contrasts['Gamma']['train_memory']['BridgeGain_MRR']:.8f}",
        f"MF_HISTORY_SHORTCUT_CONTRAST_MRR={contrasts['MiniFix']['history_shortcut']['BridgeGain_MRR']:.8f}",
        f"GA_HISTORY_SHORTCUT_CONTRAST_MRR={contrasts['Gamma']['history_shortcut']['BridgeGain_MRR']:.8f}",
        f"BRIDGE_TRAIN_SPECIALIZATION_MEMORY_DIFF={contrasts['specialization']['memory_diff']:.8f}",
        f"BRIDGE_TRAIN_SPECIALIZATION_HISTORY_DIFF={contrasts['specialization']['history_diff']:.8f}",
        "", "--- Hypotheses ---", *[f"{key}={value}" for key, value in decisions.items()],
        "PRIMARY_SIGNAL=2x2 separation of familiar-group and history-repeat effects",
        "MAIN_LIMITATION=4 groups per artificial cell; UNSEEN is only relative to Mini training; no formal significance",
        "CONCLUSION=See cell_summary.md; bridge response and Gold ranking are reported separately.",
        "NEXT_EXPERIMENT=STOP_AND_REVIEW", "", "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0",
        "SELF_COT_GENERATION_STARTED=NO", "EXTERNAL_EVAL_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.model:
        run(args.model)
    else:
        finalize()


if __name__ == "__main__":
    main()
