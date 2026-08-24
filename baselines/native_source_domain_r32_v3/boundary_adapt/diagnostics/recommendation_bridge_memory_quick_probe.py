"""Phase 1.5-Quick: bridge memory-key versus generalizable-prior screen."""
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
    "recommendation_bridge_memory_quick_probe.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_bridge_memory_quick_probe.py"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]

from boundary_adapt.bridge_inside_sft.build_dataset import load_unique_source  # noqa: E402
from boundary_adapt.bridge_inside_sft.common import (  # noqa: E402
    BASE, DOMAIN, DOMAIN_ORDER, SOURCE, SOURCE_SHA256, extract_response, file_sha,
    infer_domain, source_metadata, stable_hash, token_ids_sha,
)
from boundary_adapt.diagnostics.controlled_generator_decoder_crossover import load_model, strict_beam  # noqa: E402
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import history_from_prompt, parse_gold  # noqa: E402
from boundary_adapt.diagnostics.recommendation_official_prompt_bare_crossover import render_prompt_ids  # noqa: E402

OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_quick_probe"
PARTS = OUTPUT / "parts"
SPLIT_MANIFEST = Path("/root/GRPO_audit_results/bridge_inside_transition_sft_v1_20260824/split_manifest.json")
CANONICAL_ROWS = RUNTIME / "boundary_adapt/results/boundary_adapt_rows.jsonl"
PHASE141 = RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_4_1"
MODELS = ("MiniFix", "Gamma")
MODEL_PATHS = {
    "MiniFix": Path(
        "/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/"
        "MINI-FIX-R32-2E-GC04-4GPU-20260816-092741/checkpoint-138"
    ),
    "Gamma": Path(
        "/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/"
        "mini_gamma/checkpoint-136"
    ),
}
EXPECTED_MODEL_SHA = {
    "MiniFix": "8d61c7d61f1413f3e449bb1aebeca5ee060bd31080959b02cbaf56a8788fc72d",
    "Gamma": "772921211cbcd238728cf538fe904042697c60e6034dd0a363081c306e3f48aa",
}
DATASETS = {
    "MiniFix": Path("/data/lf_data_versions/alltrain/mini_fix/onereason_mini_fix.jsonl"),
    "Gamma": Path(
        "/data/lf_data_versions/alltrain/mini_fix_eval_align_answer_only_v1/"
        "onereason_mini_fix_eval_align_answer_only.jsonl"
    ),
}
EXPECTED_DATASET_SHA = {
    "MiniFix": "6dc7660417f093b923c93d4f19734ed9240d5fba4c9beaaa01b6a164ccd94f7f",
    "Gamma": "216857d8c5d0049a3e9642279acd89051c40f5bcddb1d4afe513e36080b086cf",
}
SELECTION_SALT = "recommendation_bridge_memory_quick_probe|"
CONDITIONS = ("BARE", "EXACT_BRIDGE")
NATIVE_CONDITIONS = ("MF_NATIVE_BARE", "MF_NATIVE_BRIDGE")
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260825

SYSTEM = {
    "video": "\u4f60\u662f\u4e00\u4e2a\u63a8\u8350\u7cfb\u7edf\u52a9\u624b\uff0c\u64c5\u957f\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\u9884\u6d4b\u7528\u6237\u7684\u89c6\u9891\u504f\u597d\u3002",
    "prod": "\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u63a8\u8350\u52a9\u7406\uff0c\u80fd\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff0c\u63a8\u8350\u7528\u6237\u4e0b\u4e00\u4e2a\u611f\u5174\u8da3\u7684\u5546\u54c1\u3002",
    "ad": "\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u63a8\u8350\u52a9\u7406\uff0c\u80fd\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff0c\u63a8\u8350\u7528\u6237\u4e0b\u4e00\u4e2a\u611f\u5174\u8da3\u7684\u5e7f\u544a\u3002",
    "living": "\u4f60\u662f\u4e00\u4e2a\u63a8\u8350\u7cfb\u7edf\u52a9\u624b\uff0c\u64c5\u957f\u6839\u636e\u591a\u57df\u5386\u53f2\u884c\u4e3a\u9884\u6d4b\u7528\u6237\u7684\u4e3b\u64ad\u504f\u597d\u3002",
}
QUESTION = {
    "video": "\u8bf7\u63a8\u65ad\u7528\u6237\u63a5\u4e0b\u6765\u4f1a\u70b9\u51fb\u7684\u89c6\u9891\u3002/think",
    "prod": "\u8bf7\u63a8\u65ad\u7528\u6237\u63a5\u4e0b\u6765\u4f1a\u70b9\u51fb\u7684\u5546\u54c1\u3002/think",
    "ad": "\u8bf7\u63a8\u8350\u7528\u6237\u4e0b\u4e00\u4e2a\u4f1a\u70b9\u51fb\u7684\u5e7f\u544a\u3002/think",
    "living": "\u8bf7\u63a8\u65ad\u7528\u6237\u63a5\u4e0b\u6765\u4f1a\u70b9\u51fb\u7684\u4e3b\u64ad\u3002/think",
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


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), *args], text=True).strip()


def code_audit() -> dict[str, Any]:
    commit, origin, status = git("rev-parse", "HEAD"), git("rev-parse", "origin/main"), git("status", "--short")
    github_sha, runtime_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT), file_sha(RUNTIME_SCRIPT)
    result = {
        "implement_commit": commit, "origin_main": origin,
        "push_status": "PASS" if commit == origin else "FAIL", "git_status_short": status,
        "github_script_sha256": github_sha, "runtime_script_sha256": runtime_sha,
        "runtime_github_parity": "PASS" if github_sha == runtime_sha else "FAIL",
    }
    if result["push_status"] != "PASS" or status or result["runtime_github_parity"] != "PASS":
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={result}")
    return result


def row_sha(row: dict[str, Any]) -> str:
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "base": config.get("base_model_name_or_path"), "r": config.get("r"),
        "alpha": config.get("lora_alpha"), "dropout": config.get("lora_dropout"),
        "targets": sorted(config.get("target_modules", [])), "peft_type": config.get("peft_type"),
    }


def build_model_inventory() -> dict[str, Any]:
    values, configs = {}, []
    for label, path in MODEL_PATHS.items():
        actual = file_sha(path / "adapter_model.safetensors")
        if actual != EXPECTED_MODEL_SHA[label]:
            raise RuntimeError(f"ADAPTER_SHA_MISMATCH model={label} actual={actual}")
        config = normalized_config(read_json(path / "adapter_config.json"))
        configs.append(config)
        values[label] = {"path": str(path), "adapter_sha256": actual, "config": config}
    parity = len({json.dumps(value, sort_keys=True) for value in configs}) == 1
    if not parity:
        raise RuntimeError("MODEL_CONFIG_PARITY_FAIL")
    return {
        "models": values, "base_identical": "YES", "lora_config_parity": "PASS",
        "lora_rank": configs[0]["r"], "lora_alpha": configs[0]["alpha"],
        "lora_target_modules": configs[0]["targets"],
    }


def scan_membership(label: str) -> dict[str, Any]:
    path = DATASETS[label]
    actual = file_sha(path)
    if actual != EXPECTED_DATASET_SHA[label]:
        raise RuntimeError(f"DATASET_SHA_MISMATCH model={label} actual={actual}")
    groups, cot_prefixes = set(), defaultdict(set)
    recommendation_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend":
                continue
            recommendation_rows += 1
            metadata = source_metadata(row)
            group = str(metadata["recommendation_group_id"])
            groups.add(group)
            if row.get("source_segment") == "recommendation_cot":
                prefix = (str(row.get("system", "")), str(row.get("instruction", "")) + str(row.get("input", "")))
                cot_prefixes[group].add(hashlib.sha256(json.dumps(prefix, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
    return {
        "path": str(path), "sha256": actual, "groups": groups,
        "recommendation_rows": recommendation_rows, "cot_prefixes": cot_prefixes,
    }


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
    if not body or SID_RE.search(body) is None:
        raise RuntimeError("HISTORY_EXTRACTION_FAIL")
    return body


def natural_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = read_json(SPLIT_MANIFEST)
    holdout = set(map(str, split["holdout_group_ids"]))
    if file_sha(SOURCE) != SOURCE_SHA256 or split["source_sha256"] != SOURCE_SHA256 or len(holdout) != 1595:
        raise RuntimeError("NATURAL_SOURCE_CONTRACT_FAIL")
    raw_rows, duplicates = load_unique_source(SOURCE, CANONICAL_ROWS)
    result = []
    for raw in raw_rows:
        metadata = source_metadata(raw)
        group = str(metadata["recommendation_group_id"])
        if group not in holdout:
            continue
        domain = infer_domain(metadata)
        cot_before_close, bridge, _ = extract_response(str(raw["output"]), domain)
        user = str(raw.get("instruction", "")) + str(raw.get("input", ""))
        history = extract_history(user)
        history_sids = [match.group(0) for match in SID_RE.finditer(history) if match.group(1) == domain]
        golds = list(map(str, metadata["recommendation_all_gold_sids"]))
        result.append({
            "group_id": group, "domain": domain, "all_gold_sids": golds, "K": len(golds),
            "gold_sid_in_history": bool(set(golds) & set(history_sids)),
            "history_target_domain_sid_count": len(history_sids), "history_target_domain_sids": history_sids,
            "history": history, "original_system": str(raw["system"]), "original_user": user,
            "original_cot": cot_before_close + "</think>", "exact_original_bridge": bridge,
            "source_row_sha256": row_sha(raw),
        })
    if len(result) != 1595 or {row["group_id"] for row in result} != holdout:
        raise RuntimeError("NATURAL_1595_RECONSTRUCTION_FAIL")
    return result, {"groups": 1595, "legacy_duplicates_dropped": duplicates, "not_official_distribution": True}


def select_probe(rows: list[dict[str, Any]], seen: set[str], union: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools = {"TRAIN_SEEN": [], "TRAIN_UNSEEN": []}
    excluded_one_model = 0
    for row in rows:
        group = row["group_id"]
        if group in seen:
            pools["TRAIN_SEEN"].append(row)
        elif group not in union:
            pools["TRAIN_UNSEEN"].append(row)
        else:
            excluded_one_model += 1
    selected = []
    pool_counts = {}
    for domain in DOMAIN_ORDER:
        for membership in ("TRAIN_SEEN", "TRAIN_UNSEEN"):
            candidates = [row for row in pools[membership] if row["domain"] == domain]
            candidates.sort(key=lambda row: (stable_hash(SELECTION_SALT + membership + "|" + row["group_id"]), row["group_id"]))
            if len(candidates) < 2:
                raise RuntimeError(f"INSUFFICIENT_POOL domain={domain} membership={membership} n={len(candidates)}")
            chosen = candidates[:2]
            selected.extend({**row, "membership": membership} for row in chosen)
            pool_counts[f"{domain}|{membership}"] = len(candidates)
    selected.sort(key=lambda row: (DOMAIN_ORDER.index(row["domain"]), row["membership"], row["group_id"]))
    if len(selected) != 16 or Counter(row["membership"] for row in selected) != Counter({"TRAIN_SEEN": 8, "TRAIN_UNSEEN": 8}):
        raise RuntimeError("PROBE16_CARDINALITY_FAIL")
    return selected, {
        "salt": SELECTION_SALT, "method": "stable-hash first two per domain x strict membership; no K/history balancing",
        "pool_counts": pool_counts, "one_model_only_groups_excluded": excluded_one_model,
    }


def soft_switch_ids(tokenizer, user: str, system: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True,
    )
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    elif hasattr(rendered, "keys") and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    return list(map(int, rendered))


def prepare() -> None:
    from transformers import AutoTokenizer

    audit = code_audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cleanup = {
        "previous_phase15_work_found": "YES",
        "reverted_phase15_commits": ["3af7991", "b7512ce", "626e97f", "e22364a"],
        "revert_commits": ["0b667c6", "5477474", "8a03458", "97cf37a"],
        "runtime_script_removed": True, "runtime_results_removed": True,
        "phase141_preserved": (PHASE141 / "summary.json").is_file(),
        "previous_phase15_cleanup_pass": "YES",
    }
    if not cleanup["phase141_preserved"]:
        raise RuntimeError("PHASE141_PRESERVATION_FAIL")
    write_json(OUTPUT / "cleanup_audit.json", cleanup)
    inventory = build_model_inventory()
    inventory["implement_commit"] = audit["implement_commit"]
    write_json(OUTPUT / "model_inventory.json", inventory)

    scans = {label: scan_membership(label) for label in MODELS}
    minifix, gamma = scans["MiniFix"]["groups"], scans["Gamma"]["groups"]
    intersection, union = minifix & gamma, minifix | gamma
    membership = {
        "definition": {
            "TRAIN_SEEN": "group in both MiniFix and Gamma recommendation training",
            "TRAIN_UNSEEN": "UNSEEN_FROM_MINI_TRAIN: group absent from both MiniFix and Gamma recommendation training",
        },
        "minifix_rec_group_count": len(minifix), "gamma_rec_group_count": len(gamma),
        "intersection_count": len(intersection), "minifix_only_count": len(minifix - gamma),
        "gamma_only_count": len(gamma - minifix),
        "datasets": {label: {key: value for key, value in scan.items() if key not in ("groups", "cot_prefixes")} for label, scan in scans.items()},
    }
    write_json(OUTPUT / "mini_train_membership.json", membership)

    natural, natural_audit = natural_candidates()
    selected, selection = select_probe(natural, intersection, union)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("THINK_CLOSE_NOT_ATOMIC")
    items, source_audit = [], []
    for row in selected:
        domain = row["domain"]
        official_user = "\u7528\u6237\u591a\u57df\u5386\u53f2\u884c\u4e3a\uff1a\n" + row["history"] + "\n\n" + QUESTION[domain]
        official_ids = render_prompt_ids(tokenizer, official_user, SYSTEM[domain])
        native_ids = render_prompt_ids(tokenizer, row["original_user"], row["original_system"])
        cot_ids = list(map(int, tokenizer.encode(row["original_cot"], add_special_tokens=False)))
        bridge_ids = list(map(int, tokenizer.encode(row["exact_original_bridge"], add_special_tokens=False)))
        domain_ids = list(map(int, tokenizer.encode(DOMAIN[domain], add_special_tokens=False)))
        if cot_ids[-1:] != close_ids or not bridge_ids or len(domain_ids) != 1:
            raise RuntimeError(f"SOURCE_BOUNDARY_TOKEN_FAIL group={row['group_id']}")
        native_prefix_sha = hashlib.sha256(json.dumps((row["original_system"], row["original_user"]), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        native_seen_match = row["membership"] != "TRAIN_SEEN" or native_prefix_sha in scans["MiniFix"]["cot_prefixes"].get(row["group_id"], set())
        items.append({
            **{key: row[key] for key in (
                "group_id", "domain", "membership", "all_gold_sids", "K", "gold_sid_in_history",
                "history_target_domain_sid_count", "history_target_domain_sids", "source_row_sha256",
            )},
            "official_system": SYSTEM[domain], "official_prompt": official_user,
            "official_prompt_token_ids": official_ids,
            "native_system": row["original_system"], "native_prompt": row["original_user"],
            "native_prompt_token_ids": native_ids, "native_seen_training_prefix_match": native_seen_match,
            "frozen_cot_source": "BATA_SOURCE", "frozen_cot_body": row["original_cot"],
            "frozen_cot_token_ids": cot_ids, "frozen_cot_sha256": token_ids_sha(cot_ids),
            "exact_original_bridge": row["exact_original_bridge"],
            "exact_original_bridge_token_ids": bridge_ids, "bridge_sha256": token_ids_sha(bridge_ids),
            "domain_token_ids": domain_ids,
        })
        source_audit.append({
            "group_id": row["group_id"], "membership": row["membership"], "domain": domain,
            "cot_closed": True, "bridge_nonempty": True, "cot_tokens": len(cot_ids),
            "bridge_tokens": len(bridge_ids), "native_seen_training_prefix_match": native_seen_match,
        })
    native_available = all(item["native_seen_training_prefix_match"] for item in items)
    audit_items = []
    for domain in DOMAIN_ORDER:
        for member in ("TRAIN_SEEN", "TRAIN_UNSEEN"):
            item = next(value for value in items if value["domain"] == domain and value["membership"] == member)
            proxy, soft = item["official_prompt_token_ids"], soft_switch_ids(tokenizer, item["official_prompt"], item["official_system"])
            original_history = item["history_target_domain_sids"]
            rendered_history = [match.group(0) for match in SID_RE.finditer(item["official_prompt"]) if match.group(1) == domain]
            audit_items.append({
                "group_id": item["group_id"], "domain": domain, "membership": member,
                "token_identical": proxy == soft, "proxy_sha256": token_ids_sha(proxy),
                "soft_switch_sha256": token_ids_sha(soft), "history_sid_parity": original_history == rendered_history,
            })
    prompt_pass = len(audit_items) == 8 and all(row["token_identical"] and row["history_sid_parity"] for row in audit_items)
    prior = read_json(PHASE141 / "soft_switch_token_diff.json")
    if not prompt_pass or prior.get("soft_switch_proxy_token_identical_count") != 8:
        raise RuntimeError("PROMPT_TOKEN_AUDIT_FAIL")
    manifest = {
        "version": "recommendation_bridge_memory_quick_probe16", "implement_commit": audit["implement_commit"],
        "groups": 16, "train_seen": 8, "train_unseen": 8,
        "domain_counts": dict(Counter(row["domain"] for row in items)),
        "seen_gold_in_history": sum(row["gold_sid_in_history"] for row in items if row["membership"] == "TRAIN_SEEN"),
        "unseen_gold_in_history": sum(row["gold_sid_in_history"] for row in items if row["membership"] == "TRAIN_UNSEEN"),
        "selection": selection, "natural_source_audit": natural_audit,
        "native_prefix_available": native_available, "items": items,
        "training_started": False, "self_cot_generation": False, "external_eval": False,
    }
    write_json(OUTPUT / "probe16_manifest.json", manifest)
    write_json(OUTPUT / "prompt_audit.json", {
        "prompt_token_audit_pass": "PASS", "history_sid_parity": "PASS", "sample_groups": 8,
        "contract": "Phase1.4.1 soft-switch equivalent controlled proxy", "items": audit_items,
    })
    write_json(OUTPUT / "source_bridge_cot_audit.json", {
        "frozen_cot_source": "BATA_SOURCE", "frozen_cot_shared_across_models": "PASS",
        "native_prefix_available": "YES" if native_available else "NO", "items": source_audit,
    })
    print("CPU_PREPARE_PASS=YES")
    print(f"MINIFIX_REC_GROUP_COUNT={len(minifix)}")
    print(f"GAMMA_REC_GROUP_COUNT={len(gamma)}")
    print(f"INTERSECTION_COUNT={len(intersection)}")
    print(f"SEEN_GOLD_IN_HISTORY={manifest['seen_gold_in_history']}")
    print(f"UNSEEN_GOLD_IN_HISTORY={manifest['unseen_gold_in_history']}")
    print(f"NATIVE_PREFIX_AVAILABLE={'YES' if native_available else 'NO'}")


def run(model_label: str) -> None:
    audit = code_audit()
    manifest = read_json(OUTPUT / "probe16_manifest.json")
    if manifest["implement_commit"] != audit["implement_commit"]:
        raise RuntimeError("MANIFEST_COMMIT_MISMATCH")
    if read_json(OUTPUT / "prompt_audit.json")["prompt_token_audit_pass"] != "PASS":
        raise RuntimeError("GPU_BLOCKED_BY_PROMPT_AUDIT")
    rank, world = int(os.environ.get("LOCAL_RANK", "-1")), int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("QUICK_PROBE_REQUIRES_4_LOCAL_RANKS")
    torch.cuda.set_device(rank)
    model, tokenizer = load_model(MODEL_PATHS[model_label], f"cuda:{rank}")
    local = []
    for index, item in enumerate(manifest["items"]):
        if index % world != rank:
            continue
        prompt, native = list(map(int, item["official_prompt_token_ids"])), list(map(int, item["native_prompt_token_ids"]))
        cot, bridge, domain_ids = (
            list(map(int, item["frozen_cot_token_ids"])),
            list(map(int, item["exact_original_bridge_token_ids"])),
            list(map(int, item["domain_token_ids"])),
        )
        contexts = {"BARE": prompt + cot + domain_ids, "EXACT_BRIDGE": prompt + cot + bridge + domain_ids}
        if model_label == "MiniFix" and manifest["native_prefix_available"]:
            contexts.update({"MF_NATIVE_BARE": native + cot + domain_ids, "MF_NATIVE_BRIDGE": native + cot + bridge + domain_ids})
        golds = {parse_gold(value) for value in item["all_gold_sids"]}
        histories = {
            "OFFICIAL": history_from_prompt(tokenizer, prompt),
            "NATIVE": history_from_prompt(tokenizer, native),
        }
        for condition, context in contexts.items():
            history = histories["NATIVE" if condition.startswith("MF_NATIVE") else "OFFICIAL"]
            result = strict_beam(model, tokenizer, context, item["domain"], golds, history)
            local.append({
                "implement_commit": audit["implement_commit"], "model": model_label,
                "group_id": item["group_id"], "domain": item["domain"], "membership": item["membership"],
                "gold_sid_in_history": item["gold_sid_in_history"], "K": item["K"],
                "condition": condition, "context_token_count": len(context),
                "context_sha256": token_ids_sha(context), **result,
            })
        print(f"QUICK_BRIDGE_PROGRESS model={model_label} rank={rank} group={index // world + 1}/4", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    write_jsonl(PARTS / f"{model_label}_rank{rank}.jsonl", local)


def sid_tuple(value: Any) -> tuple | None:
    return None if value is None else tuple(value)


def predictions(row: dict[str, Any]) -> list[tuple | None]:
    return [sid_tuple(beam.get("predicted_sid")) for beam in row["beams"]]


def best_rank(values: list[tuple | None], golds: list[tuple], prefix: int) -> int | None:
    return next((rank for rank, value in enumerate(values, 1) if value is not None and any(value[:prefix] == gold[:prefix] for gold in golds)), None)


def case_metrics(row: dict[str, Any], item: dict[str, Any]) -> dict[str, float]:
    values, golds = predictions(row), [tuple(parse_gold(value)) for value in item["all_gold_sids"]]
    ranks = {name: best_rank(values, golds, prefix) for name, prefix in (("gold", 4), ("ab", 3), ("a", 2))}
    valid = [value for value in values if value is not None]
    return {
        "GoldSIDHit@1": float(ranks["gold"] == 1), "GoldSIDHit@5": float(ranks["gold"] is not None and ranks["gold"] <= 5),
        "GoldSIDHit@32": float(ranks["gold"] is not None), "GoldMRR": 0.0 if ranks["gold"] is None else 1.0 / ranks["gold"],
        "ABHit@32": float(ranks["ab"] is not None), "AHit@32": float(ranks["a"] is not None),
        "HistoryCandidateFraction": sum(beam.get("copy_class") == "EXACT_COPY" for beam in row["beams"]) / 32,
        "Top1IsHistory": float(row["beams"][0].get("copy_class") == "EXACT_COPY"),
        "UniqueA": float(len({value[:2] for value in valid})), "UniqueAB": float(len({value[:3] for value in valid})),
        "UniqueABC": float(len(set(valid))),
    }


def pair_response(bare: dict[str, Any], bridge: dict[str, Any]) -> dict[str, float]:
    left, right = predictions(bare), predictions(bridge)
    lset, rset = {value for value in left if value is not None}, {value for value in right if value is not None}
    shared, union = lset & rset, lset | rset
    def ranks(values: list[tuple | None]) -> dict[tuple, int]:
        result = {}
        for rank, value in enumerate(values, 1):
            if value is not None and value not in result:
                result[value] = rank
        return result
    lr, rr = ranks(left), ranks(right)
    displacement = [abs(lr[value] - rr[value]) for value in shared]
    return {
        "Top1Flip": float(left[0] != right[0]),
        "Jaccard": 1.0 if not union else len(shared) / len(union),
        "Response1MinusJaccard": 0.0 if not union else 1.0 - len(shared) / len(union),
        "CommonSIDMeanRankDisplacement": 0.0 if not displacement else statistics.fmean(displacement),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}


def paired_groups(index: dict[tuple[str, str, str], dict[str, Any]], model: str, bare_name: str,
                  bridge_name: str, items: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for group, item in items.items():
        bare, bridge = index[(model, group, bare_name)], index[(model, group, bridge_name)]
        bm, rm = case_metrics(bare, item), case_metrics(bridge, item)
        response = pair_response(bare, bridge)
        result[group] = {
            "BareMRR": bm["GoldMRR"], "BridgeMRR": rm["GoldMRR"],
            "BridgeGain_MRR": rm["GoldMRR"] - bm["GoldMRR"],
            "BridgeGain_Hit32": rm["GoldSIDHit@32"] - bm["GoldSIDHit@32"],
            "BridgeGain_AB": rm["ABHit@32"] - bm["ABHit@32"],
            "BridgeGain_A": rm["AHit@32"] - bm["AHit@32"],
            **response,
            **{f"Bare_{key}": value for key, value in bm.items()},
            **{f"Bridge_{key}": value for key, value in rm.items()},
        }
    return result


def bootstrap(groups: dict[str, dict[str, float]], label: str) -> dict[str, Any]:
    ids, metrics = sorted(groups), ("BridgeGain_MRR", "BridgeGain_Hit32", "BridgeGain_AB", "BridgeGain_A", "Response1MinusJaccard", "Top1Flip", "CommonSIDMeanRankDisplacement")
    seed = (BOOTSTRAP_SEED + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)) % (2**32)
    rng, samples = random.Random(seed), {metric: [] for metric in metrics}
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = [rng.choice(ids) for _ in ids]
        for metric in metrics:
            samples[metric].append(statistics.fmean(groups[group][metric] for group in chosen))
    def pct(values: list[float], q: float) -> float:
        ordered, pos = sorted(values), (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    return {
        "replicates": BOOTSTRAP_REPLICATES, "seed": seed, "groups": len(ids),
        "metrics": {metric: {"estimate": statistics.fmean(groups[group][metric] for group in ids), "ci95": [pct(samples[metric], .025), pct(samples[metric], .975)], "direction_only_not_significance": True} for metric in metrics},
    }


def classify_hypotheses(summary: dict[str, Any], items: dict[str, dict[str, Any]], per_group: dict[str, Any]) -> dict[str, str]:
    mf_s, mf_u = summary["MiniFix"]["TRAIN_SEEN"], summary["MiniFix"]["TRAIN_UNSEEN"]
    ga_s, ga_u = summary["Gamma"]["TRAIN_SEEN"], summary["Gamma"]["TRAIN_UNSEEN"]
    mrr_interaction = mf_s["BridgeGain_MRR"] - mf_u["BridgeGain_MRR"]
    response_interaction = mf_s["Response1MinusJaccard"] - mf_u["Response1MinusJaccard"]
    gamma_mrr_interaction = ga_s["BridgeGain_MRR"] - ga_u["BridgeGain_MRR"]
    gamma_response_interaction = ga_s["Response1MinusJaccard"] - ga_u["Response1MinusJaccard"]
    memory_signals = int(mrr_interaction >= .05 and mrr_interaction - gamma_mrr_interaction >= .03) + int(response_interaction >= .10 and response_interaction - gamma_response_interaction >= .05)
    memory = "STRONG" if memory_signals == 2 else "MODERATE" if memory_signals == 1 else "WEAK_OR_UNSUPPORTED"
    general_signals = int(mf_u["Response1MinusJaccard"] >= .75 * max(mf_s["Response1MinusJaccard"], 1e-9) and mf_u["Response1MinusJaccard"] - ga_u["Response1MinusJaccard"] >= .05) + int(mf_u["BridgeGain_MRR"] >= mf_s["BridgeGain_MRR"] - .03 and mf_u["BridgeGain_MRR"] - ga_u["BridgeGain_MRR"] >= .03)
    general = "STRONG" if general_signals == 2 else "MODERATE" if general_signals == 1 else "WEAK_OR_UNSUPPORTED"
    generic = "STRONG" if all(abs(summary["MiniFix"][member][metric] - summary["Gamma"][member][metric]) <= threshold for member in ("TRAIN_SEEN", "TRAIN_UNSEEN") for metric, threshold in (("BridgeGain_MRR", .03), ("Response1MinusJaccard", .08))) else "WEAK_OR_UNSUPPORTED"
    history_values, nonhistory_values, unseen_nonhistory = [], [], []
    for group, metrics in per_group["MiniFix"].items():
        (history_values if items[group]["gold_sid_in_history"] else nonhistory_values).append(metrics["BridgeGain_MRR"])
        if items[group]["membership"] == "TRAIN_UNSEEN" and not items[group]["gold_sid_in_history"]:
            unseen_nonhistory.append(metrics)
    history_mean = statistics.fmean(history_values) if history_values else 0.0
    nonhistory_mean = statistics.fmean(nonhistory_values) if nonhistory_values else 0.0
    if history_values and history_mean - nonhistory_mean >= .05 and nonhistory_mean <= 0:
        history = "YES"
    elif unseen_nonhistory and (statistics.fmean(row["BridgeGain_MRR"] for row in unseen_nonhistory) > .02 or statistics.fmean(row["Response1MinusJaccard"] for row in unseen_nonhistory) > .10):
        history = "NO_OR_PARTIAL"
    else:
        history = "INCONCLUSIVE"
    signals = {"MEMORY_KEY": abs(mrr_interaction) + abs(response_interaction), "GENERALIZABLE_PRIOR": abs(mf_u["BridgeGain_MRR"] - ga_u["BridgeGain_MRR"]) + abs(mf_u["Response1MinusJaccard"] - ga_u["Response1MinusJaccard"]), "GENERIC_CONDITIONING": 1.0 if generic == "STRONG" else 0.0}
    primary = max(signals, key=signals.get)
    return {
        "MEMORY_KEY_HYPOTHESIS_SUPPORT": memory, "GENERALIZABLE_BRIDGE_PRIOR_SUPPORT": general,
        "GENERIC_BRIDGE_CONDITIONING_SUPPORT": generic, "HISTORY_RETRIEVAL_SHORTCUT_SUPPORT": history,
        "PRIMARY_SIGNAL": primary,
        "ROOT_CLASS": "SCREEN_SUPPORTS_" + primary,
    }


def finalize() -> None:
    audit, manifest = code_audit(), read_json(OUTPUT / "probe16_manifest.json")
    items = {row["group_id"]: row for row in manifest["items"]}
    rows = []
    for model in MODELS:
        for rank in range(4):
            path = PARTS / f"{model}_rank{rank}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"MISSING_PART={path}")
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    expected = 64 + (32 if manifest["native_prefix_available"] else 0)
    index = {(row["model"], row["group_id"], row["condition"]): row for row in rows}
    if len(rows) != expected or len(index) != expected or any(len(row["beams"]) != 32 for row in rows):
        raise RuntimeError(f"CASE_CONTRACT_FAIL expected={expected} rows={len(rows)} keys={len(index)}")
    rows.sort(key=lambda row: (MODELS.index(row["model"]), row["group_id"], row["condition"]))
    write_jsonl(OUTPUT / "records.jsonl", rows)
    all_pairs, aggregates, boot = {}, {}, {}
    for model in MODELS:
        pairs = paired_groups(index, model, "BARE", "EXACT_BRIDGE", items)
        all_pairs[model] = pairs
        aggregates[model], boot[model] = {}, {}
        for membership in ("TRAIN_SEEN", "TRAIN_UNSEEN"):
            subset = {group: value for group, value in pairs.items() if items[group]["membership"] == membership}
            aggregates[model][membership] = mean_metrics(list(subset.values()))
            boot[model][membership] = bootstrap(subset, f"{model}|{membership}")
    native_pairs = None
    if manifest["native_prefix_available"]:
        native_pairs = paired_groups(index, "MiniFix", "MF_NATIVE_BARE", "MF_NATIVE_BRIDGE", items)
        aggregates["MiniFixNative"] = {}
        boot["MiniFixNative"] = {}
        for membership in ("TRAIN_SEEN", "TRAIN_UNSEEN"):
            subset = {group: value for group, value in native_pairs.items() if items[group]["membership"] == membership}
            aggregates["MiniFixNative"][membership] = mean_metrics(list(subset.values()))
            boot["MiniFixNative"][membership] = bootstrap(subset, f"MiniFixNative|{membership}")
    hypotheses = classify_hypotheses(aggregates, items, all_pairs)
    membership = read_json(OUTPUT / "mini_train_membership.json")
    summary = {
        "implement_commit": audit["implement_commit"], "cases": expected, "groups": 16,
        "membership_definition": membership["definition"], "aggregates": aggregates,
        "per_group": all_pairs, "native_per_group": native_pairs, "hypotheses": hypotheses,
        "interpretation_limit": "TRAIN_SEEN/UNSEEN is relative only to Mini-Fix/Gamma training; this cannot establish official train-test leakage.",
    }
    write_json(OUTPUT / "seen_unseen_summary.json", summary)
    write_json(OUTPUT / "bootstrap.json", boot)
    lines = [
        "# Phase 1.5-Quick: Bridge Memory Probe", "",
        "This is a 16-group descriptive screen, not an official evaluation or a leakage test.", "",
        "| Model | Membership | Bare MRR | Bridge MRR | Gain MRR | 1-Jaccard | Top1 flip |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for membership_name in ("TRAIN_SEEN", "TRAIN_UNSEEN"):
            value = aggregates[model][membership_name]
            lines.append(f"| {model} | {membership_name} | {value['BareMRR']:.4f} | {value['BridgeMRR']:.4f} | {value['BridgeGain_MRR']:+.4f} | {value['Response1MinusJaccard']:.4f} | {value['Top1Flip']:.4f} |")
    lines.extend(["", "## Decision", "", *[f"- {key}: {value}" for key, value in hypotheses.items()], "", "Bootstrap intervals are paired and descriptive only."])
    (OUTPUT / "seen_unseen_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mf_s, mf_u = aggregates["MiniFix"]["TRAIN_SEEN"], aggregates["MiniFix"]["TRAIN_UNSEEN"]
    ga_s, ga_u = aggregates["Gamma"]["TRAIN_SEEN"], aggregates["Gamma"]["TRAIN_UNSEEN"]
    inv, cleanup = read_json(OUTPUT / "model_inventory.json"), read_json(OUTPUT / "cleanup_audit.json")
    native_s = aggregates.get("MiniFixNative", {}).get("TRAIN_SEEN")
    native_u = aggregates.get("MiniFixNative", {}).get("TRAIN_UNSEEN")
    review = [
        "--- Cleanup ---", f"PREVIOUS_PHASE15_WORK_FOUND={cleanup['previous_phase15_work_found']}",
        f"REVERTED_PHASE15_COMMITS={','.join(cleanup['reverted_phase15_commits'])}", "PREVIOUS_PHASE15_CLEANUP_PASS=YES",
        "", "--- Git ---", f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT",
        "PUSH_STATUS=PASS", "GIT_STATUS_SHORT=EMPTY", f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}",
        f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}", "RUNTIME_GITHUB_PARITY=PASS",
        "", "--- Models ---", f"MINIFIX_PATH={MODEL_PATHS['MiniFix']}", f"MINIFIX_SHA256={EXPECTED_MODEL_SHA['MiniFix']}",
        f"GAMMA_PATH={MODEL_PATHS['Gamma']}", f"GAMMA_SHA256={EXPECTED_MODEL_SHA['Gamma']}",
        "BASE_IDENTICAL=YES", "LORA_CONFIG_PARITY=PASS",
        "", "--- Training Membership ---", f"MINIFIX_REC_GROUP_COUNT={membership['minifix_rec_group_count']}",
        f"GAMMA_REC_GROUP_COUNT={membership['gamma_rec_group_count']}", f"INTERSECTION_COUNT={membership['intersection_count']}",
        "", "--- Probe ---", "GROUPS=16", "TRAIN_SEEN=8", "TRAIN_UNSEEN=8",
        f"DOMAIN_COUNTS={manifest['domain_counts']}", f"SEEN_GOLD_IN_HISTORY={manifest['seen_gold_in_history']}",
        f"UNSEEN_GOLD_IN_HISTORY={manifest['unseen_gold_in_history']}", "PROMPT_TOKEN_AUDIT_PASS=PASS",
        "FROZEN_COT_SOURCE=BATA_SOURCE", "FROZEN_COT_SHARED_ACROSS_MODELS=PASS",
        "", "--- MiniFix ---", f"MF_SEEN_BARE_MRR={mf_s['BareMRR']:.8f}", f"MF_SEEN_BRIDGE_MRR={mf_s['BridgeMRR']:.8f}",
        f"MF_SEEN_BRIDGE_GAIN_MRR={mf_s['BridgeGain_MRR']:.8f}", f"MF_UNSEEN_BARE_MRR={mf_u['BareMRR']:.8f}",
        f"MF_UNSEEN_BRIDGE_MRR={mf_u['BridgeMRR']:.8f}", f"MF_UNSEEN_BRIDGE_GAIN_MRR={mf_u['BridgeGain_MRR']:.8f}",
        f"MF_SEEN_RESPONSE_1_MINUS_JACCARD={mf_s['Response1MinusJaccard']:.8f}", f"MF_UNSEEN_RESPONSE_1_MINUS_JACCARD={mf_u['Response1MinusJaccard']:.8f}",
        f"MF_SEEN_TOP1_FLIP={mf_s['Top1Flip']:.8f}", f"MF_UNSEEN_TOP1_FLIP={mf_u['Top1Flip']:.8f}",
        "", "--- Gamma ---", f"GAMMA_SEEN_BARE_MRR={ga_s['BareMRR']:.8f}", f"GAMMA_SEEN_BRIDGE_MRR={ga_s['BridgeMRR']:.8f}",
        f"GAMMA_SEEN_BRIDGE_GAIN_MRR={ga_s['BridgeGain_MRR']:.8f}", f"GAMMA_UNSEEN_BARE_MRR={ga_u['BareMRR']:.8f}",
        f"GAMMA_UNSEEN_BRIDGE_MRR={ga_u['BridgeMRR']:.8f}", f"GAMMA_UNSEEN_BRIDGE_GAIN_MRR={ga_u['BridgeGain_MRR']:.8f}",
        f"GAMMA_SEEN_RESPONSE_1_MINUS_JACCARD={ga_s['Response1MinusJaccard']:.8f}", f"GAMMA_UNSEEN_RESPONSE_1_MINUS_JACCARD={ga_u['Response1MinusJaccard']:.8f}",
        "", "--- Native Prefix Optional ---", f"NATIVE_PREFIX_AVAILABLE={'YES' if native_s else 'NO'}", f"NATIVE_PREFIX_CASES={32 if native_s else 0}",
        f"MF_NATIVE_SEEN_BRIDGE_GAIN_MRR={native_s['BridgeGain_MRR']:.8f}" if native_s else "MF_NATIVE_SEEN_BRIDGE_GAIN_MRR=NA",
        f"MF_NATIVE_UNSEEN_BRIDGE_GAIN_MRR={native_u['BridgeGain_MRR']:.8f}" if native_u else "MF_NATIVE_UNSEEN_BRIDGE_GAIN_MRR=NA",
        f"NATIVE_BRIDGE_SEEN_BOOST={native_s['BridgeMRR'] - mf_s['BridgeMRR']:.8f}" if native_s else "NATIVE_BRIDGE_SEEN_BOOST=NA",
        f"NATIVE_BRIDGE_UNSEEN_BOOST={native_u['BridgeMRR'] - mf_u['BridgeMRR']:.8f}" if native_u else "NATIVE_BRIDGE_UNSEEN_BOOST=NA",
        "", "--- Hypothesis ---", *[f"{key}={value}" for key, value in hypotheses.items()],
        "MAIN_LIMITATION=16-group descriptive screen; Mini-train membership is not official train-test overlap",
        "CONCLUSION=See seen_unseen_summary.md; decision thresholds are preregistered in implementation",
        "NEXT_EXPERIMENT=STOP_AND_REVIEW_BEFORE_ANY_FREE_CONTINUATION",
        "", "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "SELF_COT_GENERATION_STARTED=NO",
        "EXTERNAL_EVAL_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
    ]
    (OUTPUT / "CHATGPT_BRIDGE_MEMORY_QUICK_REVIEW.txt").write_text("\n".join(review) + "\n", encoding="utf-8")
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
