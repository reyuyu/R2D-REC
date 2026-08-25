"""TrueRec-GRPO Phase 0.1 CPU-only source and canonical-group audit."""
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


SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
RUNTIME = Path("/data/GRPO")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/truerec_grpo/data/build_canonical_groups.py"
)
RUNTIME_SCRIPT = RUNTIME / "truerec_grpo/data/build_canonical_groups.py"
OUTPUT = RUNTIME / "truerec_grpo/results/phase0_1"
BATA_SOURCE = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
GAMMA_SOURCE = Path(
    "/data/lf_data_versions/alltrain/mini_fix_eval_align_answer_only_v1/"
    "onereason_mini_fix_eval_align_answer_only.jsonl"
)
BETA_CHECKPOINT = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
GAMMA_CHECKPOINT = Path(
    "/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/"
    "mini_gamma/checkpoint-136"
)
EXPECTED_BATA_SHA = "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca"
EXPECTED_GAMMA_SHA = "216857d8c5d0049a3e9642279acd89051c40f5bcddb1d4afe513e36080b086cf"
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
ROUTES = {"recommendation_cot": "think", "recommendation_nocot": "nothink"}
DOMAINS = ("video", "prod", "ad", "living")
REQUIRED_ROW_FIELDS = {
    "data_source", "source_segment", "system", "instruction", "input", "output",
    "history", "aux_metadata_json",
}
REQUIRED_METADATA_FIELDS = {
    "recommendation_group_id", "recommendation_group_size",
    "recommendation_current_gold_sid", "recommendation_all_gold_sids",
}


class SchemaError(RuntimeError):
    pass


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(raw)
            digest.update(raw)
    return digest.hexdigest()


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


def parse_sid(value: str | list[Any] | tuple[Any, ...]) -> tuple[str, int, int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            raise ValueError(f"invalid SID {value!r}")
        return str(value[0]), int(value[1]), int(value[2]), int(value[3])
    match = SID_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"invalid SID {value!r}")
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def extract_history_text(instruction: str, input_text: str) -> str:
    """Return behavior history without route/task-template text."""
    prompt = (instruction + input_text).strip()
    prompt = re.sub(r"/(?:no_)?think\s*$", "", prompt).rstrip()
    blocks = prompt.split("\n\n")
    sid_blocks = [index for index, block in enumerate(blocks) if SID_RE.search(block)]
    if not sid_blocks:
        raise SchemaError("HISTORY_EXTRACTION_NO_SID")
    return "\n\n".join(blocks[: sid_blocks[-1] + 1]).strip()


def classify_answer_suffix(output: str) -> dict[str, Any]:
    if "</think>" not in output:
        return {"class": "MISSING_THINK_CLOSE", "tail": output, "bridge": None, "sid": None}
    tail = output.split("</think>", 1)[1].strip()
    direct = SID_RE.fullmatch(tail)
    if direct:
        return {"class": "DIRECT_DOMAIN_SID", "tail": tail, "bridge": "", "sid": direct.group(0)}
    match = SID_RE.search(tail)
    if match and not tail[match.end():].strip():
        bridge = tail[: match.start()].strip()
        if bridge:
            return {"class": "NATURAL_LANGUAGE_BRIDGE", "tail": tail, "bridge": bridge, "sid": match.group(0)}
    return {"class": "OTHER_AFTER_THINK", "tail": tail, "bridge": None, "sid": None}


def normalize_recommendation_row(row: dict[str, Any], row_number: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing = REQUIRED_ROW_FIELDS - set(row)
    if missing:
        raise SchemaError(f"ROW_SCHEMA_MISSING row={row_number} fields={sorted(missing)}")
    segment = str(row["source_segment"])
    if segment not in ROUTES:
        raise SchemaError(f"UNKNOWN_RECOMMENDATION_ROUTE row={row_number} route={segment}")
    try:
        info = json.loads(row["aux_metadata_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"INVALID_AUX_METADATA row={row_number}") from exc
    missing_meta = REQUIRED_METADATA_FIELDS - set(info)
    if missing_meta:
        raise SchemaError(f"METADATA_SCHEMA_MISSING row={row_number} fields={sorted(missing_meta)}")
    if not isinstance(info["recommendation_all_gold_sids"], list) or not info["recommendation_all_gold_sids"]:
        raise SchemaError(f"ALL_GOLD_NOT_NONEMPTY_LIST row={row_number}")

    errors = []
    parsed_gold = []
    for value in info["recommendation_all_gold_sids"]:
        try:
            parsed_gold.append(parse_sid(str(value)))
        except ValueError:
            errors.append({"row_number": row_number, "kind": "INVALID_GOLD_SID", "value": value})
    try:
        current = parse_sid(str(info["recommendation_current_gold_sid"]))
    except ValueError:
        current = None
        errors.append({"row_number": row_number, "kind": "INVALID_CURRENT_GOLD_SID", "value": info["recommendation_current_gold_sid"]})
    target_domain = current[0] if current else None
    for value in parsed_gold:
        if target_domain is not None and value[0] != target_domain:
            errors.append({"row_number": row_number, "kind": "GOLD_DOMAIN_MISMATCH", "value": list(value), "target_domain": target_domain})
    history_text = extract_history_text(str(row["instruction"]), str(row["input"]))
    history_sids = [parse_sid(match.group(0)) for match in SID_RE.finditer(history_text)]
    suffix = classify_answer_suffix(str(row["output"]))
    normalized = {
        "row_number": row_number,
        "group_id": str(info["recommendation_group_id"]),
        "route": ROUTES[segment],
        "source_segment": segment,
        "target_domain": target_domain,
        "current_gold_sid": current,
        "all_gold_sids": tuple(sorted(set(parsed_gold))),
        "declared_group_size": int(info["recommendation_group_size"]),
        "history_text_sha256": hashlib.sha256(history_text.encode("utf-8")).hexdigest(),
        "history_sids": tuple(history_sids),
        "answer_suffix_class": suffix["class"],
        "answer_bridge": suffix["bridge"],
    }
    return normalized, errors


def aggregate_normalized(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    canonical, conflicts = [], {
        "target_domain": [], "all_gold": [], "current_gold": [], "history": [],
    }
    for group in sorted(grouped):
        values = grouped[group]
        routes = sorted({row["route"] for row in values})
        domains = sorted({row["target_domain"] for row in values if row["target_domain"] is not None})
        gold_contracts = {row["all_gold_sids"] for row in values}
        history_hashes = sorted({row["history_text_sha256"] for row in values})
        route_current = {
            route: sorted({row["current_gold_sid"] for row in values if row["route"] == route and row["current_gold_sid"] is not None})
            for route in routes
        }
        if len(domains) != 1:
            conflicts["target_domain"].append({"group_id": group, "values": domains})
        if len(gold_contracts) != 1:
            conflicts["all_gold"].append({"group_id": group, "digests": sorted(stable_sha([list(sid) for sid in contract]) for contract in gold_contracts)})
        if set(routes) == {"think", "nothink"} and route_current["think"] != route_current["nothink"]:
            conflicts["current_gold"].append({"group_id": group, "think": [list(sid) for sid in route_current["think"]], "nothink": [list(sid) for sid in route_current["nothink"]]})
        if len(history_hashes) != 1:
            conflicts["history"].append({"group_id": group, "history_sha256": history_hashes})
        all_gold = sorted(set().union(*(set(contract) for contract in gold_contracts)))
        all_current = sorted(set().union(*(set(current) for current in route_current.values())))
        history_sids = sorted(set().union(*(set(row["history_sids"]) for row in values)))
        canonical.append({
            "recommendation_group_id": group,
            "target_domain": domains[0] if len(domains) == 1 else None,
            "source_row_count": len(values),
            "routes_present": routes,
            "all_gold_sids": [list(sid) for sid in all_gold],
            "current_gold_sid": list(all_current[0]) if len(all_current) == 1 else None,
            "current_gold_sids": [list(sid) for sid in all_current],
            "history_sids": [list(sid) for sid in history_sids],
            "history_text_sha256": history_hashes[0] if len(history_hashes) == 1 else None,
            "think_row_count": sum(row["route"] == "think" for row in values),
            "nothink_row_count": sum(row["route"] == "nothink" for row in values),
            "route_current_gold_sids": {route: [list(sid) for sid in current] for route, current in route_current.items()},
        })
    return canonical, conflicts


def scan_source(path: Path, label: str, build_groups: bool) -> dict[str, Any]:
    total = recommendation = 0
    normalized_rows = []
    parser_errors = []
    source_segments = Counter()
    answer_classes = Counter()
    bridge_texts = Counter()
    row_keys, metadata_keys = set(), set()
    group_gold: dict[str, tuple[tuple[str, int, int, int], ...]] = {}
    group_gold_conflicts = []
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            total += 1
            row = json.loads(line)
            if row.get("data_source") != "recommend":
                continue
            recommendation += 1
            row_keys.update(row)
            info = json.loads(row.get("aux_metadata_json") or "{}")
            metadata_keys.update(info)
            normalized, errors = normalize_recommendation_row(row, row_number)
            parser_errors.extend(errors)
            source_segments[normalized["source_segment"]] += 1
            answer_classes[normalized["answer_suffix_class"]] += 1
            if normalized["answer_bridge"]:
                bridge_texts[normalized["answer_bridge"]] += 1
            group = normalized["group_id"]
            previous = group_gold.get(group)
            if previous is not None and previous != normalized["all_gold_sids"]:
                group_gold_conflicts.append(group)
            group_gold[group] = normalized["all_gold_sids"]
            if build_groups:
                normalized_rows.append(normalized)
    if group_gold_conflicts:
        raise SchemaError(f"SOURCE_GROUP_GOLD_CONFLICT={label}:{sorted(set(group_gold_conflicts))[:10]}")
    return {
        "label": label,
        "path": str(path),
        "sha256": file_sha(path),
        "total_rows": total,
        "recommendation_rows": recommendation,
        "recommendation_groups": len(group_gold),
        "row_detection": "data_source == recommend",
        "row_keys": sorted(row_keys),
        "metadata_keys": sorted(metadata_keys),
        "route_field": "source_segment",
        "route_mapping": ROUTES,
        "group_field": "aux_metadata_json.recommendation_group_id",
        "target_domain_field": None,
        "target_domain_derivation": "domain parsed from aux_metadata_json.recommendation_current_gold_sid",
        "all_gold_field": "aux_metadata_json.recommendation_all_gold_sids",
        "current_gold_field": "aux_metadata_json.recommendation_current_gold_sid",
        "history_source": "instruction + input; retain through last paragraph containing a SID; remove task template and route marker",
        "source_segments": dict(source_segments),
        "answer_suffix_classes": dict(answer_classes),
        "natural_language_bridge_counts": dict(bridge_texts.most_common()),
        "parser_errors": parser_errors,
        "normalized_rows": normalized_rows,
        "group_gold_contract": group_gold,
    }


def public_schema(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in ("normalized_rows", "group_gold_contract", "parser_errors")}


def render_review(audit: dict[str, Any], beta: dict[str, Any], gamma: dict[str, Any], canonical_audit: dict[str, Any], domain_counts: dict[str, int], canonical_sha: str, test_status: str) -> str:
    conflicts = canonical_audit["conflict_counts"]
    routes = canonical_audit["route_group_counts"]
    lines = [
        "--- Git ---", "", f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT", "",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short'] or 'EMPTY'}", "",
        f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}", f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}",
        f"RUNTIME_GITHUB_PARITY={audit['runtime_github_parity']}", "", "--- Source ---", "",
        f"SOURCE_PATH={beta['path']}", f"SOURCE_SHA256={beta['sha256']}", f"TOTAL_ROWS={beta['total_rows']}",
        f"RECOMMENDATION_ROWS={beta['recommendation_rows']}", f"UNIQUE_RECOMMENDATION_GROUPS={beta['recommendation_groups']}",
        f"VIDEO_GROUPS={domain_counts['video']}", f"PROD_GROUPS={domain_counts['prod']}", f"AD_GROUPS={domain_counts['ad']}", f"LIVING_GROUPS={domain_counts['living']}",
        "", "--- Beta/Gamma Template ---", "", f"BETA_RECOMMENDATION_SOURCE={beta['path']}", f"GAMMA_RECOMMENDATION_SOURCE={gamma['path']}",
        f"GAMMA_SOURCE_SHA256={gamma['sha256']}", f"BETA_AFTER_THINK_DIRECT_SID_ROWS={beta['answer_suffix_classes'].get('DIRECT_DOMAIN_SID', 0)}",
        f"BETA_AFTER_THINK_BRIDGE_ROWS={beta['answer_suffix_classes'].get('NATURAL_LANGUAGE_BRIDGE', 0)}",
        f"GAMMA_AFTER_THINK_DIRECT_SID_ROWS={gamma['answer_suffix_classes'].get('DIRECT_DOMAIN_SID', 0)}",
        f"GAMMA_AFTER_THINK_BRIDGE_ROWS={gamma['answer_suffix_classes'].get('NATURAL_LANGUAGE_BRIDGE', 0)}",
        f"GAMMA_AFTER_THINK_TEMPLATE=DIRECT_DOMAIN_PLUS_SID_NO_NATURAL_LANGUAGE_BRIDGE", "", "--- Canonical Contract ---", "",
        f"THINK_ONLY_GROUPS={routes['think_only']}", f"NOTHINK_ONLY_GROUPS={routes['nothink_only']}", f"THINK_NOTHINK_GROUPS={routes['think_nothink']}",
        f"TARGET_DOMAIN_CONFLICT_GROUPS={conflicts['target_domain']}", f"ALL_GOLD_CONFLICT_GROUPS={conflicts['all_gold']}",
        f"CURRENT_GOLD_CONFLICT_GROUPS={conflicts['current_gold']}", f"HISTORY_CONFLICT_GROUPS={conflicts['history']}",
        f"INVALID_GOLD_SID_COUNT={canonical_audit['invalid_gold_sid_count']}", f"GOLD_DOMAIN_MISMATCH_COUNT={canonical_audit['gold_domain_mismatch_count']}",
        f"CANONICAL_GROUPS_SHA256={canonical_sha}", f"TEST_STATUS={test_status}", "", "GPU_INFERENCE_STARTED=NO", "MODEL_FORWARD_STARTED=NO",
        "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "BEAM_STARTED=NO", "EXTERNAL_EVAL_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
    ]
    return "\n".join(lines) + "\n"


def run(test_status: str) -> None:
    if "torch" in sys.modules:
        raise RuntimeError("TORCH_ALREADY_IMPORTED")
    if test_status != "PASS":
        raise RuntimeError("TEST_STATUS_GATE_FAIL")
    audit = code_audit()
    beta = scan_source(BATA_SOURCE, "Beta/BATA", build_groups=True)
    gamma = scan_source(GAMMA_SOURCE, "Gamma answer-only", build_groups=False)
    if beta["sha256"] != EXPECTED_BATA_SHA or gamma["sha256"] != EXPECTED_GAMMA_SHA:
        raise RuntimeError("SOURCE_SHA_MISMATCH")
    canonical, conflicts = aggregate_normalized(beta["normalized_rows"])
    if len(canonical) != beta["recommendation_groups"] or len({row["recommendation_group_id"] for row in canonical}) != len(canonical):
        raise RuntimeError("CANONICAL_GROUP_COUNT_OR_UNIQUENESS_FAIL")
    domain_counts = Counter(row["target_domain"] for row in canonical)
    route_counts = Counter(
        "think_nothink" if set(row["routes_present"]) == {"think", "nothink"}
        else "think_only" if row["routes_present"] == ["think"] else "nothink_only"
        for row in canonical
    )
    parser_errors = beta["parser_errors"]
    canonical_audit = {
        "canonical_groups": len(canonical),
        "unique_recommendation_group_ids": beta["recommendation_groups"],
        "canonical_group_uniqueness": "PASS",
        "route_group_counts": {key: route_counts[key] for key in ("think_only", "nothink_only", "think_nothink")},
        "conflict_counts": {key: len(value) for key, value in conflicts.items()},
        "invalid_gold_sid_count": sum(error["kind"] in ("INVALID_GOLD_SID", "INVALID_CURRENT_GOLD_SID") for error in parser_errors),
        "gold_domain_mismatch_count": sum(error["kind"] == "GOLD_DOMAIN_MISMATCH" for error in parser_errors),
        "test_status": test_status,
    }
    beta_groups, gamma_groups = set(beta["group_gold_contract"]), set(gamma["group_gold_contract"])
    intersection = beta_groups & gamma_groups
    beta_gamma = {
        "beta_checkpoint": str(BETA_CHECKPOINT),
        "gamma_checkpoint": str(GAMMA_CHECKPOINT),
        "beta_source": str(BATA_SOURCE),
        "gamma_source": str(GAMMA_SOURCE),
        "gamma_groups_subset_of_beta": gamma_groups <= beta_groups,
        "group_intersection": len(intersection),
        "gold_contract_parity_on_intersection": all(beta["group_gold_contract"][group] == gamma["group_gold_contract"][group] for group in intersection),
        "checkpoint_to_dataset_evidence": "Direct corpus scan plus repository diagnostic model/dataset mapping; checkpoint training_args.bin does not embed dataset path.",
        "beta_after_think_contract": "NATURAL_LANGUAGE_BRIDGE_THEN_DOMAIN_SID",
        "gamma_after_think_contract": "DIRECT_DOMAIN_SID",
        "gamma_has_natural_language_bridge": gamma["answer_suffix_classes"].get("NATURAL_LANGUAGE_BRIDGE", 0) > 0,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    canonical_sha = write_jsonl(OUTPUT / "canonical_groups.jsonl", canonical)
    schema_audit = {
        "beta": public_schema(beta),
        "gamma": public_schema(gamma),
        "beta_gamma_provenance": beta_gamma,
        "schema_status": "PASS",
        "cpu_only": True,
        "torch_imported": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(OUTPUT / "source_schema_audit.json", schema_audit)
    write_json(OUTPUT / "canonical_group_audit.json", canonical_audit)
    write_json(OUTPUT / "domain_counts.json", {domain: domain_counts[domain] for domain in DOMAINS})
    write_json(OUTPUT / "conflict_groups.json", conflicts)
    write_json(OUTPUT / "dataset_sha256.json", {
        "beta_bata_source": beta["sha256"], "gamma_source": gamma["sha256"],
        "canonical_groups_jsonl": canonical_sha,
    })
    review = render_review(audit, beta, gamma, canonical_audit, domain_counts, canonical_sha, test_status)
    (OUTPUT / "CHATGPT_PHASE0_1_REVIEW.txt").write_text(review, encoding="utf-8")
    print(review, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-status", choices=("PASS", "FAIL"), required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
    run(args.test_status)


if __name__ == "__main__":
    main()
