"""Phase 1.5: exact-bridge response crossover for existing Mini models.

The prepare phase is CPU-only. GPU inference is deliberately a separate command and
is blocked unless source/runtime code provenance and all data/prompt gates pass.
"""
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
    "recommendation_bridge_response_crossover.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_bridge_response_crossover.py"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]

from boundary_adapt.bridge_inside_sft.build_dataset import load_unique_source  # noqa: E402
from boundary_adapt.bridge_inside_sft.common import (  # noqa: E402
    BASE,
    DOMAIN,
    DOMAIN_ORDER,
    SOURCE,
    SOURCE_SHA256,
    file_sha,
    infer_domain,
    source_metadata,
    stable_hash,
    token_ids_sha,
)
from boundary_adapt.diagnostics.controlled_generator_decoder_crossover import (  # noqa: E402
    load_model,
    strict_beam,
)
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import (  # noqa: E402
    history_from_prompt,
    parse_gold,
)
from boundary_adapt.diagnostics.recommendation_official_prompt_bare_crossover import (  # noqa: E402
    render_prompt_ids,
)

OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_bridge_response_crossover_v1"
PARTS = OUTPUT / "parts"
SPLIT_MANIFEST = Path("/root/GRPO_audit_results/bridge_inside_transition_sft_v1_20260824/split_manifest.json")
CANONICAL_ROWS = RUNTIME / "boundary_adapt/results/boundary_adapt_rows.jsonl"
PHASE141 = RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_4_1"

MODEL_PATHS = {
    "MiniFix": Path(
        "/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/"
        "MINI-FIX-R32-2E-GC04-4GPU-20260816-092741/checkpoint-138"
    ),
    "MiniGamma": Path(
        "/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/"
        "mini_gamma/checkpoint-136"
    ),
    "MiniSol": Path(
        "/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/"
        "mini_sol-20260824-165955/checkpoint-68"
    ),
}
EXPECTED_ADAPTER_SHA = {
    "MiniFix": "8d61c7d61f1413f3e449bb1aebeca5ee060bd31080959b02cbaf56a8788fc72d",
    "MiniGamma": "772921211cbcd238728cf538fe904042697c60e6034dd0a363081c306e3f48aa",
    "MiniSol": "7084b3c8c97ff2adc1585f4094bb051f4b32bd61f0a047b66b572df34b903b31",
}
DATASETS = {
    "MiniFix": Path("/data/lf_data_versions/alltrain/mini_fix/onereason_mini_fix.jsonl"),
    "MiniGamma": Path(
        "/data/lf_data_versions/alltrain/mini_fix_eval_align_answer_only_v1/"
        "onereason_mini_fix_eval_align_answer_only.jsonl"
    ),
    "MiniSol": Path("/data/lf_data_versions/alltrain/mini_sol_v1/onereason_mini_sol.jsonl"),
}
EXPECTED_DATASET_SHA = {
    "MiniFix": "6dc7660417f093b923c93d4f19734ed9240d5fba4c9beaaa01b6a164ccd94f7f",
    "MiniGamma": "216857d8c5d0049a3e9642279acd89051c40f5bcddb1d4afe513e36080b086cf",
    "MiniSol": "a9083e87715f1bb48d0b50894fccb5661716d7ebf420aded622bf37e50b446f4",
}
ROUTES = ("recommendation_cot", "recommendation_nocot")
CONDITIONS = ("THINK_BARE", "THINK_BRIDGE", "NOTHINK_BARE", "NOTHINK_BRIDGE")
MODELS = tuple(MODEL_PATHS)
SELECTION_SALT = "recommendation_bridge_response_natural40_v2|"
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_REPLICATES = 5000

SYSTEM = {
    "video": "\u4f60\u662f\u4e00\u4e2a\u63a8\u8350\u7cfb\u7edf\u52a9\u624b\uff0c\u64c5\u957f\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\u9884\u6d4b\u7528\u6237\u7684\u89c6\u9891\u504f\u597d\u3002",
    "prod": "\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u63a8\u8350\u52a9\u7406\uff0c\u80fd\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff0c\u63a8\u8350\u7528\u6237\u4e0b\u4e00\u4e2a\u611f\u5174\u8da3\u7684\u5546\u54c1\u3002",
    "ad": "\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u63a8\u8350\u52a9\u7406\uff0c\u80fd\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff0c\u63a8\u8350\u7528\u6237\u4e0b\u4e00\u4e2a\u611f\u5174\u8da3\u7684\u5e7f\u544a\u3002",
    "living": "\u4f60\u662f\u4e00\u4e2a\u63a8\u8350\u7cfb\u7edf\u52a9\u624b\uff0c\u64c5\u957f\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\u9884\u6d4b\u7528\u6237\u7684\u4e3b\u64ad\u504f\u597d\u3002",
}
QUESTION = {
    "video": "\u8bf7\u63a8\u65ad\u7528\u6237\u63a5\u4e0b\u6765\u4f1a\u70b9\u51fb\u7684\u89c6\u9891\u3002/{route}",
    "prod": "\u8bf7\u63a8\u65ad\u7528\u6237\u63a5\u4e0b\u6765\u4f1a\u70b9\u51fb\u7684\u5546\u54c1\u3002/{route}",
    "ad": "\u8bf7\u63a8\u8350\u7528\u6237\u4e0b\u4e00\u4e2a\u4f1a\u70b9\u51fb\u7684\u5e7f\u544a\u3002/{route}",
    "living": "\u8bf7\u63a8\u65ad\u7528\u6237\u63a5\u4e0b\u6765\u4f1a\u70b9\u51fb\u7684\u4e3b\u64ad\u3002/{route}",
}
OLD_ENDINGS = (
    "\u8bf7\u6839\u636e\u4ee5\u4e0a\u4fe1\u606f\uff0c\u7ed9\u51fa\u8be5\u7528\u6237\u5728\u76f4\u64ad\u3001\u7535\u5546\u3001\u89c6\u9891\u3001\u5e7f\u544a\u573a\u666f\u4e2d\u7684\u76ee\u6807\u5185\u5bb9\u3002/think",
    "\u8bf7\u8f93\u51fa\u8be5\u7528\u6237\u5728\u4e0d\u540c\u573a\u666f\u4e0b\u5bf9\u5e94\u7684\u76ee\u6807\u5185\u5bb9\u3002/think",
    "\u8bf7\u57fa\u4e8e\u8fd9\u4e9b\u7ebf\u7d22\u603b\u7ed3\u8be5\u7528\u6237\u5728\u5404\u573a\u666f\u4e2d\u7684\u76ee\u6807\u5185\u5bb9\u3002/think",
)
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


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"], text=True).strip()


def code_audit() -> dict[str, Any]:
    commit = source_commit()
    origin = subprocess.check_output(["git", "-C", str(SOURCE_REPO), "rev-parse", "origin/main"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(SOURCE_REPO), "status", "--short"], text=True).strip()
    github_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT)
    runtime_sha = file_sha(RUNTIME_SCRIPT)
    audit = {
        "implement_commit": commit,
        "origin_main": origin,
        "push_status": "PASS" if commit == origin else "FAIL",
        "git_status_short": status,
        "github_script_sha256": github_sha,
        "runtime_script_sha256": runtime_sha,
        "runtime_github_parity": "PASS" if github_sha == runtime_sha else "FAIL",
    }
    if audit["push_status"] != "PASS" or status or audit["runtime_github_parity"] != "PASS":
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={audit}")
    return audit


def row_sha(row: dict[str, Any]) -> str:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sid_tuple(value: Any) -> tuple[str, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        match = SID_RE.fullmatch(value)
        return None if match is None else (match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)))
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return str(value[0]), int(value[1]), int(value[2]), int(value[3])
    return None


def prompt_sid_sequence(row: dict[str, Any]) -> list[str]:
    text = str(row.get("instruction", "")) + str(row.get("input", ""))
    return [match.group(0) for match in SID_RE.finditer(text)]


def extract_history(user: str) -> str:
    lines = user.splitlines()
    if lines and ("\u5386\u53f2\u884c\u4e3a" in lines[0] or "\u884c\u4e3a\u5e8f\u5217" in lines[0]):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    body = "\n".join(lines).rstrip()
    for ending in OLD_ENDINGS:
        if body.endswith(ending):
            body = body[:-len(ending)].rstrip()
            break
    else:
        for suffix in ("/think", "/no_think"):
            if body.endswith(suffix):
                body = body[:-len(suffix)].rstrip()
                break
    if not body or SID_RE.search(body) is None:
        raise RuntimeError("HISTORY_EXTRACTION_FAILED")
    return body


def official_user(history: str, domain: str, route: str) -> str:
    return "\u7528\u6237\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff1a\n" + history + "\n\n" + QUESTION[domain].format(route=route)


def normalized_lora_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "peft_type": config.get("peft_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "target_modules": sorted(config.get("target_modules", [])),
    }


def model_inventory() -> dict[str, Any]:
    models = {}
    normalized = []
    for label, path in MODEL_PATHS.items():
        actual = file_sha(path / "adapter_model.safetensors")
        if actual != EXPECTED_ADAPTER_SHA[label]:
            raise RuntimeError(f"ADAPTER_SHA_MISMATCH model={label} actual={actual}")
        config = read_json(path / "adapter_config.json")
        norm = normalized_lora_config(config)
        normalized.append(norm)
        models[label] = {
            "path": str(path),
            "adapter_model_sha256": actual,
            "adapter_config_sha256": file_sha(path / "adapter_config.json"),
            "normalized_lora_config": norm,
        }
    parity = len({json.dumps(value, sort_keys=True) for value in normalized}) == 1
    base_parity = len({value["base_model_name_or_path"] for value in normalized}) == 1
    if not parity or not base_parity:
        raise RuntimeError("MODEL_BASE_OR_LORA_CONFIG_PARITY_FAIL")
    return {
        "base_model_identical": "YES",
        "lora_config_parity": "PASS",
        "models": models,
        "optional_beta_run": "NO",
        "optional_step900_run": "NO",
    }


def natural_holdout() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = read_json(SPLIT_MANIFEST)
    if file_sha(SOURCE) != SOURCE_SHA256 or split["source_sha256"] != SOURCE_SHA256:
        raise RuntimeError("SOURCE_SHA256_MISMATCH")
    holdout = set(map(str, split["holdout_group_ids"]))
    train = set(map(str, split["train_group_ids"]))
    if len(holdout) != 1595 or len(train) != 14348 or holdout & train:
        raise RuntimeError("NATURAL_SPLIT_CONTRACT_FAIL")
    raw_rows, duplicates = load_unique_source(SOURCE, CANONICAL_ROWS)
    rows = []
    for raw in raw_rows:
        metadata = source_metadata(raw)
        group = str(metadata["recommendation_group_id"])
        if group not in holdout:
            continue
        domain = infer_domain(metadata)
        golds = list(map(str, metadata["recommendation_all_gold_sids"]))
        current = str(metadata["recommendation_current_gold_sid"])
        history_sids = prompt_sid_sequence(raw)
        target_history = [value for value in history_sids if value.startswith(DOMAIN[domain])]
        rows.append({
            "group_id": group,
            "domain": domain,
            "all_gold_sids": golds,
            "current_gold_sid": current,
            "K": len(golds),
            "gold_sid_in_history": bool(set(golds) & set(target_history)),
            "target_domain_history_sids": target_history,
            "original_system": str(raw.get("system", "")),
            "original_user": str(raw.get("instruction", "")) + str(raw.get("input", "")),
            "source_row_sha256": row_sha(raw),
        })
    if len(rows) != 1595 or {row["group_id"] for row in rows} != holdout:
        raise RuntimeError("NATURAL_1595_RECONSTRUCTION_FAIL")
    return rows, {
        "population": "ADAPTATION_HELDOUT_NATURAL_PROXY",
        "not_official_test_distribution": True,
        "groups": len(rows),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "source_streaming_passes": 1,
        "legacy_duplicate_rows_dropped": duplicates,
    }


def scan_dataset(label: str, holdout_ids: set[str]) -> dict[str, Any]:
    path = DATASETS[label]
    actual_sha = file_sha(path)
    if actual_sha != EXPECTED_DATASET_SHA[label]:
        raise RuntimeError(f"DATASET_SHA_MISMATCH model={label} actual={actual_sha}")
    groups: dict[str, dict[str, Any]] = {}
    route_keys: set[tuple[str, str, str]] = set()
    route_signatures: dict[str, dict[tuple[str, str], str]] = {route: {} for route in ROUTES}
    holdout_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    recommendation_rows = 0
    identical_duplicate_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            route = row.get("source_segment")
            if row.get("data_source") != "recommend" or route not in ROUTES:
                continue
            recommendation_rows += 1
            metadata = source_metadata(row)
            group = str(metadata["recommendation_group_id"])
            current = str(metadata["recommendation_current_gold_sid"])
            golds = tuple(sorted(map(str, metadata["recommendation_all_gold_sids"])))
            domain = infer_domain(metadata)
            history_sha = hashlib.sha256(json.dumps(prompt_sid_sequence(row), separators=(",", ":")).encode()).hexdigest()
            contract = groups.setdefault(group, {
                "domain": domain,
                "all_gold_sids": golds,
                "current_gold_sids": set(),
                "history_sha256": set(),
                "routes": set(),
            })
            if contract["domain"] != domain or contract["all_gold_sids"] != golds:
                raise RuntimeError(f"INTRA_DATASET_GROUP_CONFLICT model={label} group={group}")
            contract["current_gold_sids"].add(current)
            contract["history_sha256"].add(history_sha)
            contract["routes"].add(route)
            key = (group, current, route)
            signature = row_sha(row)
            if key in route_keys:
                if route_signatures[route].get((group, current)) != signature:
                    raise RuntimeError(f"CONFLICTING_RECOMMENDATION_KEY model={label} key={key}")
                identical_duplicate_rows += 1
                continue
            route_keys.add(key)
            route_signatures[route][(group, current)] = signature
            if group in holdout_ids:
                holdout_rows[key] = row
    public_contract = {
        group: {
            "domain": value["domain"],
            "all_gold_sids": value["all_gold_sids"],
            "current_gold_sids": tuple(sorted(value["current_gold_sids"])),
            "history_sha256": tuple(sorted(value["history_sha256"])),
            "routes": tuple(sorted(value["routes"])),
        }
        for group, value in groups.items()
    }
    return {
        "path": path,
        "sha256": actual_sha,
        "recommendation_rows": recommendation_rows,
        "identical_duplicate_rows": identical_duplicate_rows,
        "groups": public_contract,
        "group_ids": set(groups),
        "route_keys": route_keys,
        "route_signatures": route_signatures,
        "holdout_rows": holdout_rows,
    }


def dataset_audit(scans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    group_sets = [scan["group_ids"] for scan in scans.values()]
    aligned = set.intersection(*group_sets)
    union = set.union(*group_sets)
    conflicts = []
    for group in sorted(aligned):
        contracts = [scans[label]["groups"][group] for label in MODELS]
        if len({json.dumps(value, sort_keys=True) for value in contracts}) != 1:
            conflicts.append(group)
    key_parity = all(scans[MODELS[0]]["route_keys"] == scans[label]["route_keys"] for label in MODELS[1:])
    gamma_no = scans["MiniGamma"]["route_signatures"]["recommendation_nocot"]
    sol_no = scans["MiniSol"]["route_signatures"]["recommendation_nocot"]
    no_keys = set(gamma_no) | set(sol_no)
    no_mismatch = [key for key in sorted(no_keys) if gamma_no.get(key) != sol_no.get(key)]
    gamma_think = scans["MiniGamma"]["route_signatures"]["recommendation_cot"]
    sol_think = scans["MiniSol"]["route_signatures"]["recommendation_cot"]
    think_diff = sum(gamma_think.get(key) != sol_think.get(key) for key in set(gamma_think) | set(sol_think))
    pass_gate = len(aligned) == len(union) and not conflicts and key_parity and not no_mismatch and think_diff > 0
    payload = {
        "aligned_recommendation_groups": len(aligned),
        "union_recommendation_groups": len(union),
        "group_counts": {label: len(scan["group_ids"]) for label, scan in scans.items()},
        "recommendation_row_counts": {label: scan["recommendation_rows"] for label, scan in scans.items()},
        "identical_duplicate_rows": {label: scan["identical_duplicate_rows"] for label, scan in scans.items()},
        "dataset_sha256": {label: scan["sha256"] for label, scan in scans.items()},
        "group_contract_conflicts": conflicts[:100],
        "route_key_parity": "PASS" if key_parity else "FAIL",
        "gamma_sol_nothink_byte_identical": "PASS" if not no_mismatch else "FAIL",
        "gamma_sol_nothink_mismatch_count": len(no_mismatch),
        "gamma_sol_think_different_row_count": think_diff,
        "sol_diff_from_gamma_think_only": "PASS" if not no_mismatch and think_diff > 0 else "FAIL",
        "data_alignment_pass": "PASS" if pass_gate else "FAIL",
    }
    if not pass_gate:
        raise RuntimeError(f"DATA_ALIGNMENT_FAIL={payload}")
    return payload


def select_natural40(natural: list[dict[str, Any]], train_union: set[str],
                     minifix_think_groups: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in natural if row["group_id"] in minifix_think_groups]
    outside = [row for row in eligible if row["group_id"] not in train_union]
    outside_counts = Counter(row["domain"] for row in outside)
    use_outside = len(outside) >= 40 and all(outside_counts[domain] >= 10 for domain in DOMAIN_ORDER)
    pool = outside if use_outside else eligible
    selected = []
    for domain in DOMAIN_ORDER:
        candidates = [row for row in pool if row["domain"] == domain]
        candidates.sort(key=lambda row: (stable_hash(SELECTION_SALT + row["group_id"]), row["group_id"]))
        if len(candidates) < 10:
            raise RuntimeError(f"INSUFFICIENT_NATURAL_CANDIDATES domain={domain}")
        selected.extend(candidates[:10])
    if len(selected) != 40 or len({row["group_id"] for row in selected}) != 40:
        raise RuntimeError("NATURAL40_CARDINALITY_FAIL")
    return selected, {
        "selection_salt": SELECTION_SALT,
        "selection_method": "MiniFix-Think-eligible natural holdout; per-domain stable-hash random top-10; no history/K balancing",
        "natural_groups_before_minifix_think_eligibility": len(natural),
        "minifix_think_eligible_groups": len(eligible),
        "minifix_think_eligible_per_domain": dict(Counter(row["domain"] for row in eligible)),
        "outside_all_three_train_groups": len(outside),
        "outside_all_three_per_domain": dict(outside_counts),
        "outside_only_selected": use_outside,
        "fallback_reason": None if use_outside else "fewer than 10 outside-train candidates in at least one domain",
    }


def extract_cot_bridge(row: dict[str, Any], domain: str) -> tuple[str, str]:
    output = str(row["output"])
    if output.count("</think>") != 1:
        raise RuntimeError("MINIFIX_CLOSE_CONTRACT_FAIL")
    close_end = output.index("</think>") + len("</think>")
    domain_pos = output.index(DOMAIN[domain], close_end)
    cot, bridge = output[:close_end], output[close_end:domain_pos]
    if not cot.startswith("<think>") or not bridge:
        raise RuntimeError("MINIFIX_COT_BRIDGE_EXTRACTION_FAIL")
    return cot, bridge


def soft_switch_ids(tokenizer, user: str, system: str) -> list[int]:
    return list(map(int, tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )))


def prepare() -> None:
    from transformers import AutoTokenizer

    audit = code_audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory = model_inventory()
    inventory["source_commit"] = audit["implement_commit"]
    write_json(OUTPUT / "model_inventory.json", inventory)

    natural, natural_stats = natural_holdout()
    holdout_ids = {row["group_id"] for row in natural}
    scans = {label: scan_dataset(label, holdout_ids) for label in MODELS}
    alignment = dataset_audit(scans)
    alignment["natural_population"] = natural_stats
    write_json(OUTPUT / "data_alignment_audit.json", alignment)

    train_union = set.union(*(scan["group_ids"] for scan in scans.values()))
    minifix_think_groups = {
        key[0] for key in scans["MiniFix"]["route_keys"] if key[2] == "recommendation_cot"
    }
    selected, selection = select_natural40(natural, train_union, minifix_think_groups)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("THINK_CLOSE_NOT_ATOMIC")
    items = []
    bridge_audit = []
    prompt_checks = []
    for row in selected:
        key = (row["group_id"], row["current_gold_sid"], "recommendation_cot")
        source_row = scans["MiniFix"]["holdout_rows"].get(key)
        source_current = row["current_gold_sid"]
        if source_row is None:
            candidates = sorted(
                (
                    (candidate_key[1], candidate_row)
                    for candidate_key, candidate_row in scans["MiniFix"]["holdout_rows"].items()
                    if candidate_key[0] == row["group_id"] and candidate_key[2] == "recommendation_cot"
                ),
                key=lambda value: value[0],
            )
            extracted = {extract_cot_bridge(candidate, row["domain"]) for _, candidate in candidates}
            if not candidates or len(extracted) != 1:
                raise RuntimeError(f"MINIFIX_GROUP_COT_BRIDGE_NOT_UNIQUE key={key} variants={len(extracted)}")
            source_current, source_row = candidates[0]
        cot, bridge = extract_cot_bridge(source_row, row["domain"])
        cot_ids = list(map(int, tokenizer.encode(cot, add_special_tokens=False)))
        bridge_ids = list(map(int, tokenizer.encode(bridge, add_special_tokens=False)))
        domain_ids = list(map(int, tokenizer.encode(DOMAIN[row["domain"]], add_special_tokens=False)))
        if cot_ids[-1:] != close_ids or len(domain_ids) != 1 or not bridge_ids:
            raise RuntimeError(f"TOKEN_BOUNDARY_FAIL group={row['group_id']}")
        history = extract_history(row["original_user"])
        think_user = official_user(history, row["domain"], "think")
        nothink_user = official_user(history, row["domain"], "no_think")
        think_prompt = render_prompt_ids(tokenizer, think_user, SYSTEM[row["domain"]])
        nothink_prompt = render_prompt_ids(tokenizer, nothink_user, SYSTEM[row["domain"]])
        item = {
            **{key: row[key] for key in (
                "group_id", "domain", "all_gold_sids", "current_gold_sid", "K",
                "gold_sid_in_history", "target_domain_history_sids", "source_row_sha256",
            )},
            "official_style_system": SYSTEM[row["domain"]],
            "official_style_think_prompt": think_user,
            "official_style_nothink_prompt": nothink_user,
            "think_prompt_token_ids": think_prompt,
            "nothink_prompt_token_ids": nothink_prompt,
            "frozen_cot_source": "MiniFix aligned recommendation_cot row",
            "frozen_cot_training_current_gold_sid": source_current,
            "frozen_cot_current_gold_matches_canonical": source_current == row["current_gold_sid"],
            "frozen_cot_body": cot,
            "frozen_cot_token_ids": cot_ids,
            "frozen_cot_sha256": token_ids_sha(cot_ids),
            "exact_original_bridge": bridge,
            "exact_original_bridge_token_ids": bridge_ids,
            "exact_original_bridge_sha256": token_ids_sha(bridge_ids),
            "domain_token_ids": domain_ids,
        }
        items.append(item)
        bridge_audit.append({
            "group_id": row["group_id"], "domain": row["domain"],
            "close_atomic": True, "bridge_nonempty": True,
            "cot_tokens": len(cot_ids), "bridge_tokens": len(bridge_ids),
            "training_current_gold_sid": source_current,
            "current_gold_matches_canonical": source_current == row["current_gold_sid"],
            "source_row_sha256": row_sha(source_row),
        })

    audit_ids = {domain: 0 for domain in DOMAIN_ORDER}
    for item in items:
        domain = item["domain"]
        if audit_ids[domain] >= 2:
            continue
        audit_ids[domain] += 1
        comparisons = {}
        for route in ("think", "nothink"):
            user = item[f"official_style_{route}_prompt"]
            proxy = item[f"{route}_prompt_token_ids"]
            soft = soft_switch_ids(tokenizer, user, item["official_style_system"])
            comparisons[route] = {
                "proxy_length": len(proxy), "soft_switch_length": len(soft),
                "token_identical": proxy == soft,
                "proxy_sha256": token_ids_sha(proxy), "soft_switch_sha256": token_ids_sha(soft),
            }
        prompt_checks.append({"group_id": item["group_id"], "domain": domain, "routes": comparisons})
    prompt_pass = len(prompt_checks) == 8 and all(
        comparison["token_identical"]
        for row in prompt_checks for comparison in row["routes"].values()
    )
    prior_soft = read_json(PHASE141 / "soft_switch_token_diff.json")
    if not prompt_pass or prior_soft.get("soft_switch_proxy_token_identical_count") != 8:
        raise RuntimeError("SOFT_SWITCH_PROMPT_PARITY_FAIL")

    manifest = {
        "version": "recommendation_bridge_response_natural40_v2",
        "source_commit": audit["implement_commit"],
        "population": natural_stats,
        "selection": selection,
        "groups": 40,
        "domain_counts": dict(Counter(item["domain"] for item in items)),
        "items": items,
        "training_started": False,
        "self_cot_generation": False,
        "external_eval": False,
    }
    write_json(OUTPUT / "natural40_v2_manifest.json", manifest)
    overlap = {
        "natural40_groups": 40,
        "per_model": {
            label: {
                "overlap_count": sum(item["group_id"] in scans[label]["group_ids"] for item in items),
                "overlap_group_ids": [item["group_id"] for item in items if item["group_id"] in scans[label]["group_ids"]],
            }
            for label in MODELS
        },
        "outside_all_three_available": selection["outside_all_three_train_groups"],
        "outside_only_selected": selection["outside_only_selected"],
    }
    write_json(OUTPUT / "mini_train_overlap_audit.json", overlap)
    write_json(OUTPUT / "prompt_audit.json", {
        "prompt_render_contract": "SOFT_SWITCH_EQUIVALENT_ON_AUDIT_SAMPLE",
        "sample_groups": 8,
        "route_comparisons": 16,
        "token_identical": "PASS",
        "phase141_evidence": str(PHASE141 / "soft_switch_token_diff.json"),
        "phase141_token_identical": "8/8",
        "current_samples": prompt_checks,
    })
    write_json(OUTPUT / "bridge_extraction_audit.json", {
        "frozen_cot_source": "MiniFix",
        "exact_bridge_source": "same aligned MiniFix recommendation_cot row",
        "groups": 40,
        "pass": "PASS",
        "items": bridge_audit,
    })
    print("CPU_AUDIT_PASS=YES")
    print(f"ALIGNED_RECOMMENDATION_GROUPS={alignment['aligned_recommendation_groups']}")
    print("PROMPT_RENDER_CONTRACT=SOFT_SWITCH_EQUIVALENT_ON_AUDIT_SAMPLE")
    print("SOL_DIFF_FROM_GAMMA_THINK_ONLY=PASS")
    print(f"DOMAIN_COUNTS={manifest['domain_counts']}")


def run(model_label: str) -> None:
    if model_label not in MODELS:
        raise ValueError(model_label)
    audit = code_audit()
    manifest = read_json(OUTPUT / "natural40_v2_manifest.json")
    if manifest["source_commit"] != audit["implement_commit"]:
        raise RuntimeError("MANIFEST_CODE_COMMIT_MISMATCH")
    if read_json(OUTPUT / "data_alignment_audit.json")["data_alignment_pass"] != "PASS":
        raise RuntimeError("GPU_BLOCKED_BY_DATA_ALIGNMENT")
    if read_json(OUTPUT / "prompt_audit.json")["token_identical"] != "PASS":
        raise RuntimeError("GPU_BLOCKED_BY_PROMPT_AUDIT")
    rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("INFERENCE_REQUIRES_4_LOCAL_RANKS")
    torch.cuda.set_device(rank)
    model, tokenizer = load_model(MODEL_PATHS[model_label], f"cuda:{rank}")
    local = []
    for index, item in enumerate(manifest["items"]):
        if index % world != rank:
            continue
        domain = item["domain"]
        domain_ids = list(map(int, item["domain_token_ids"]))
        cot = list(map(int, item["frozen_cot_token_ids"]))
        bridge = list(map(int, item["exact_original_bridge_token_ids"]))
        think = list(map(int, item["think_prompt_token_ids"]))
        nothink = list(map(int, item["nothink_prompt_token_ids"]))
        contexts = {
            "THINK_BARE": think + cot + domain_ids,
            "THINK_BRIDGE": think + cot + bridge + domain_ids,
            "NOTHINK_BARE": nothink + domain_ids,
            "NOTHINK_BRIDGE": nothink + bridge + domain_ids,
        }
        gold_set = {parse_gold(value) for value in item["all_gold_sids"]}
        histories = {
            "THINK": history_from_prompt(tokenizer, think),
            "NOTHINK": history_from_prompt(tokenizer, nothink),
        }
        for condition in CONDITIONS:
            route = "THINK" if condition.startswith("THINK_") else "NOTHINK"
            result = strict_beam(model, tokenizer, contexts[condition], domain, gold_set, histories[route])
            local.append({
                "source_commit": audit["implement_commit"],
                "model": model_label,
                "group_id": item["group_id"],
                "domain": domain,
                "K": item["K"],
                "gold_sid_in_history": item["gold_sid_in_history"],
                "condition": condition,
                "route": route,
                "bridge_present": condition.endswith("_BRIDGE"),
                "context_token_count": len(contexts[condition]),
                "context_token_ids_sha256": token_ids_sha(contexts[condition]),
                **result,
            })
        print(f"BRIDGE_RESPONSE_PROGRESS model={model_label} rank={rank} group={index // world + 1}/10", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    write_jsonl(PARTS / f"{model_label}_rank{rank}.jsonl", local)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def predictions(row: dict[str, Any]) -> list[tuple[str, int, int, int] | None]:
    return [sid_tuple(beam.get("predicted_sid")) for beam in row["beams"]]


def best_rank(values: list[tuple | None], golds: list[tuple], prefix: int) -> int | None:
    return next((index for index, value in enumerate(values, 1) if value is not None and any(value[:prefix] == gold[:prefix] for gold in golds)), None)


def build_frequency_manifold() -> tuple[dict[str, dict[str, Counter]], dict[str, Any]]:
    frequencies = {domain: {"A": Counter(), "AB": Counter()} for domain in DOMAIN_ORDER}
    rows = 0
    with DATASETS["MiniFix"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend" or row.get("source_segment") != "recommendation_cot":
                continue
            metadata = source_metadata(row)
            value = sid_tuple(metadata["recommendation_current_gold_sid"])
            if value is None:
                raise RuntimeError("INVALID_TRAIN_TARGET_SID")
            frequencies[value[0]]["A"][value[:2]] += 1
            frequencies[value[0]]["AB"][value[:3]] += 1
            rows += 1
    stats = {"source": str(DATASETS["MiniFix"]), "definition": "current target SID frequency in recommendation_cot rows", "rows": rows, "families": {}}
    for domain in DOMAIN_ORDER:
        stats["families"][domain] = {stage: len(frequencies[domain][stage]) for stage in ("A", "AB")}
    return frequencies, stats


def head_sets(frequencies: dict[str, dict[str, Counter]]) -> dict[str, dict[str, set[tuple]]]:
    result = {domain: {} for domain in DOMAIN_ORDER}
    for domain in DOMAIN_ORDER:
        for stage in ("A", "AB"):
            counter = frequencies[domain][stage]
            threshold = sum(counter.values()) * 0.8
            cumulative = 0
            head = set()
            for family, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
                head.add(family)
                cumulative += count
                if cumulative >= threshold:
                    break
            result[domain][stage] = head
    return result


def per_condition_metrics(row: dict[str, Any], item: dict[str, Any], heads: dict[str, dict[str, set[tuple]]]) -> dict[str, float]:
    values = predictions(row)
    valid = [value for value in values if value is not None]
    golds = [sid_tuple(value) for value in item["all_gold_sids"]]
    ranks = {stage: best_rank(values, golds, prefix) for stage, prefix in (("gold", 4), ("ab", 3), ("a", 2))}
    a_values, ab_values = [value[:2] for value in valid], [value[:3] for value in valid]
    domain = item["domain"]
    def seen_head_fraction(stage: str, families: list[tuple]) -> float:
        return 0.0 if not families else sum(value in heads[domain][stage] for value in families) / len(families)
    return {
        "GoldSIDHit@32": float(ranks["gold"] is not None),
        "GoldABHit@32": float(ranks["ab"] is not None),
        "GoldAHit@32": float(ranks["a"] is not None),
        "MRR": 0.0 if ranks["gold"] is None else 1.0 / ranks["gold"],
        "MeanUniqueA@32": float(len(set(a_values))),
        "MeanUniqueAB@32": float(len(set(ab_values))),
        "MeanUniqueABC@32": float(len(set(valid))),
        "AEntropy": entropy(a_values),
        "ABEntropy": entropy(ab_values),
        "HistoryFraction": sum(beam.get("copy_class") == "EXACT_COPY" for beam in row["beams"]) / 32.0,
        "AHeadFraction": seen_head_fraction("A", a_values),
        "ABHeadFraction": seen_head_fraction("AB", ab_values),
        "ADominantConcentration": 0.0 if not a_values else max(Counter(a_values).values()) / len(a_values),
        "ABDominantConcentration": 0.0 if not ab_values else max(Counter(ab_values).values()) / len(ab_values),
        "InvalidFraction": sum(value is None for value in values) / 32.0,
    }


def unique_rank(values: list[tuple | None]) -> dict[tuple, int]:
    result = {}
    for index, value in enumerate(values, 1):
        if value is not None and value not in result:
            result[value] = index
    return result


def pair_metrics(bare: dict[str, Any], bridge: dict[str, Any]) -> dict[str, float]:
    left, right = predictions(bare), predictions(bridge)
    lset, rset = set(value for value in left if value is not None), set(value for value in right if value is not None)
    union, shared = lset | rset, lset & rset
    lr, rr = unique_rank(left), unique_rank(right)
    displacement = [abs(lr[value] - rr[value]) for value in shared]
    def retention(k: int) -> float:
        source = set(value for value in left[:k] if value is not None)
        target = set(value for value in right[:k] if value is not None)
        return 1.0 if not source else len(source & target) / len(source)
    return {
        "Top1Flip": float(left[0] != right[0]),
        "SIDSetJaccard": 1.0 if not union else len(shared) / len(union),
        "SharedSIDCount": float(len(shared)),
        "SharedRankDisplacementMean": 0.0 if not displacement else statistics.fmean(displacement),
        "SharedRankDisplacementMedian": 0.0 if not displacement else statistics.median(displacement),
        "SharedRankDisplacementP90": 0.0 if not displacement else percentile(displacement, 0.9),
        "Top5Retention": retention(5),
        "Top10Retention": retention(10),
    }


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}


def paired_bootstrap(per_group: dict[str, dict[str, float]], label: str) -> dict[str, Any]:
    groups = sorted(per_group)
    seed = (BOOTSTRAP_SEED + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)) % (2**32)
    rng = random.Random(seed)
    keys = list(next(iter(per_group.values())))
    samples = {key: [] for key in keys}
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = [rng.choice(groups) for _ in groups]
        for key in keys:
            samples[key].append(statistics.fmean(per_group[group][key] for group in chosen))
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "groups": len(groups),
        "metrics": {
            key: {
                "estimate": statistics.fmean(per_group[group][key] for group in groups),
                "ci95": [percentile(samples[key], 0.025), percentile(samples[key], 0.975)],
                "p_positive": sum(value > 0 for value in samples[key]) / BOOTSTRAP_REPLICATES,
            }
            for key in keys
        },
    }


def compare_route(rows_by_key: dict[tuple[str, str, str], dict[str, Any]], model: str, route: str,
                  items: dict[str, dict[str, Any]], heads: dict[str, dict[str, set[tuple]]]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    bare_name, bridge_name = f"{route}_BARE", f"{route}_BRIDGE"
    condition_metrics = {bare_name: [], bridge_name: []}
    per_group = {}
    pair_rows = []
    for group in sorted(items):
        bare = rows_by_key[(model, group, bare_name)]
        bridge = rows_by_key[(model, group, bridge_name)]
        bm = per_condition_metrics(bare, items[group], heads)
        rm = per_condition_metrics(bridge, items[group], heads)
        condition_metrics[bare_name].append(bm)
        condition_metrics[bridge_name].append(rm)
        pair = pair_metrics(bare, bridge)
        pair_rows.append(pair)
        per_group[group] = {
            **{f"Delta_{key}": rm[key] - bm[key] for key in bm},
            **pair,
        }
    bare_mean, bridge_mean = mean_dict(condition_metrics[bare_name]), mean_dict(condition_metrics[bridge_name])
    return {
        "bare": bare_mean,
        "bridge": bridge_mean,
        "bridge_minus_bare": {key: bridge_mean[key] - bare_mean[key] for key in bare_mean},
        "response": mean_dict(pair_rows),
    }, per_group


def subgroup_summary(per_group: dict[str, dict[str, float]], items: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    grouped = defaultdict(list)
    for group, metrics in per_group.items():
        value = items[group][field]
        if field == "gold_sid_in_history":
            value = "HISTORY" if value else "NONHISTORY"
        grouped[str(value)].append(metrics)
    return {key: {"N": len(rows), "metrics": mean_dict(rows)} for key, rows in sorted(grouped.items())}


def finalize() -> None:
    audit = code_audit()
    manifest = read_json(OUTPUT / "natural40_v2_manifest.json")
    items = {item["group_id"]: item for item in manifest["items"]}
    rows = []
    for model in MODELS:
        for rank in range(4):
            path = PARTS / f"{model}_rank{rank}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"MISSING_PART={path}")
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    keys = {(row["model"], row["group_id"], row["condition"]) for row in rows}
    if len(rows) != 480 or len(keys) != 480 or any(len(row["beams"]) != 32 for row in rows):
        raise RuntimeError(f"RECORD_CONTRACT_FAIL rows={len(rows)} keys={len(keys)}")
    if any(row["source_commit"] != audit["implement_commit"] for row in rows):
        raise RuntimeError("RECORD_SOURCE_COMMIT_MISMATCH")
    rows.sort(key=lambda row: (MODELS.index(row["model"]), row["group_id"], CONDITIONS.index(row["condition"])))
    write_jsonl(OUTPUT / "records.jsonl", rows)
    rows_by_key = {(row["model"], row["group_id"], row["condition"]): row for row in rows}
    frequencies, manifold_stats = build_frequency_manifold()
    heads = head_sets(frequencies)
    summary: dict[str, Any] = {
        "source_commit": audit["implement_commit"],
        "groups": 40,
        "cases": 480,
        "beam_candidates": 480 * 32,
        "head_definition": "smallest train-frequency-ranked A/AB family set reaching >=80% observed mass per domain",
        "frequency_manifold": manifold_stats,
        "models": {},
    }
    bootstrap = {}
    domain_analysis = {}
    history_analysis = {}
    effects = {}
    for model in MODELS:
        summary["models"][model] = {}
        effects[model] = {}
        domain_analysis[model] = {}
        history_analysis[model] = {}
        bootstrap[model] = {}
        for route in ("THINK", "NOTHINK"):
            result, per_group = compare_route(rows_by_key, model, route, items, heads)
            summary["models"][model][route] = result
            effects[model][route] = {"per_group": per_group, "aggregate": result}
            domain_analysis[model][route] = subgroup_summary(per_group, items, "domain")
            history_analysis[model][route] = subgroup_summary(per_group, items, "gold_sid_in_history")
            bootstrap[model][route] = paired_bootstrap(per_group, f"{model}|{route}")

    stability = {}
    for model in MODELS:
        stability[model] = statistics.fmean(
            summary["models"][model][route]["response"]["SIDSetJaccard"]
            for route in ("THINK", "NOTHINK")
        )
    summary["cross_model_bridge_stability"] = stability
    summary["minifix_more_stable"] = stability["MiniFix"] > max(stability["MiniGamma"], stability["MiniSol"])
    gamma_sol = {}
    for condition in CONDITIONS:
        comparisons = []
        for group in items:
            gamma = rows_by_key[("MiniGamma", group, condition)]
            sol = rows_by_key[("MiniSol", group, condition)]
            comparisons.append(pair_metrics(gamma, sol))
        gamma_sol[condition] = mean_dict(comparisons)
    summary["gamma_vs_sol_same_context_response"] = gamma_sol

    write_json(OUTPUT / "distribution_response_summary.json", summary)
    write_json(OUTPUT / "bridge_effects.json", effects)
    write_json(OUTPUT / "domain_analysis.json", domain_analysis)
    write_json(OUTPUT / "history_analysis.json", history_analysis)
    write_json(OUTPUT / "bootstrap.json", bootstrap)

    lines = [
        "# Recommendation Bridge Response Crossover V1",
        "",
        f"- Source commit: `{audit['implement_commit']}`",
        "- Natural40-v2: 40 groups, 10/domain, stable-hash selection without K/history balancing.",
        "- Cases: 3 models x 40 groups x 4 conditions = 480 Beam32 ABC3 calls.",
        "- Frozen CoT and exact bridge: aligned Mini-Fix training row; no Self-CoT.",
        "",
        "## Main table",
        "",
        "| Model | Route | Top1 flip | SID Jaccard | Rank disp. mean | Top5 retention | GoldSID delta | MRR delta | History delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for route in ("THINK", "NOTHINK"):
            value = summary["models"][model][route]
            response, delta = value["response"], value["bridge_minus_bare"]
            lines.append(
                f"| {model} | {route} | {response['Top1Flip']:.4f} | {response['SIDSetJaccard']:.4f} | "
                f"{response['SharedRankDisplacementMean']:.4f} | {response['Top5Retention']:.4f} | "
                f"{delta['GoldSIDHit@32']:+.4f} | {delta['MRR']:+.4f} | {delta['HistoryFraction']:+.4f} |"
            )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Bridge sensitivity is measured as the paired redistribution of the full 32-candidate response, not only as a hit-rate change. "
        "Higher Jaccard/retention and lower flip/displacement indicate greater decoder-boundary stability."
    )
    lines.append(
        f"Mini-Fix is {'more' if summary['minifix_more_stable'] else 'not more'} stable than both Mini-Gamma and Mini-Sol by the preregistered mean route Jaccard criterion."
    )
    write_json(OUTPUT / "distribution_response_summary.md.json", {"markdown": "\n".join(lines) + "\n"})
    (OUTPUT / "distribution_response_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    conclusion = (
        "The exact bridge causally changes the short Beam32 response distribution to the extent reported by paired Jaccard, "
        "rank-displacement, retention, entropy, head-mass and gold metrics. Route-specific and history-specific estimates must "
        "be interpreted with their paired 5,000-bootstrap intervals; this diagnostic does not estimate official external quality."
    )
    review = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}",
        f"RESULT_COMMIT={audit['implement_commit']}",
        "PUSH_STATUS=PASS",
        f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}",
        f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}",
        "RUNTIME_GITHUB_PARITY=PASS",
        f"ALIGNED_RECOMMENDATION_GROUPS={read_json(OUTPUT / 'data_alignment_audit.json')['aligned_recommendation_groups']}",
        "NATURAL40_V2=40",
        f"DOMAIN_COUNTS={manifest['domain_counts']}",
        f"TRAIN_OVERLAP={read_json(OUTPUT / 'mini_train_overlap_audit.json')['per_model']}",
        "PROMPT_RENDER_CONTRACT=SOFT_SWITCH_EQUIVALENT_ON_AUDIT_SAMPLE",
        "SOL_DIFF_FROM_GAMMA_THINK_ONLY=PASS",
        "CASES=480",
        "BEAM_WIDTH=32",
        "ABC_TOKENS=3",
        "TRAINING_STARTED=NO",
        "SELF_COT_GENERATION=NO",
        "EXTERNAL_EVAL=NO",
        "OPTIONAL_BETA_RUN=NO",
        "OPTIONAL_STEP900_RUN=NO",
        f"BRIDGE_RESPONSE_CONCLUSION={conclusion}",
    ]
    (OUTPUT / "CHATGPT_BRIDGE_RESPONSE_REVIEW.txt").write_text("\n".join(review) + "\n", encoding="utf-8")
    print("\n".join(review))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", choices=MODELS)
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run(args.run)
    else:
        finalize()


if __name__ == "__main__":
    main()
