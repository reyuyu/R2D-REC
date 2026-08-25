"""TrueRec-GRPO Phase 0.5 fixed-domain ABC record and tokenizer audit."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Iterable


DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from audit_history_overlap import (  # noqa: E402
    AuditError,
    DOMAINS,
    EXPECTED_GROUPS,
    EXPECTED_SOURCE_SHA,
    SID_RE,
    SOURCE,
    classify_overlap,
    extract_history_text,
    k_bucket,
    normalize_gold,
    parse_sid,
)
from beta_gamma_renderer import BetaGammaRenderer, EMPTY_THINK, join_alpaca_user  # noqa: E402


SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
RUNTIME = Path("/data/GRPO")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/truerec_grpo/data/build_fixed_domain_abc_records.py"
)
RUNTIME_SCRIPT = RUNTIME / "truerec_grpo/data/build_fixed_domain_abc_records.py"
DEPENDENCIES = (
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/data/beta_gamma_renderer.py"),
    Path("baselines/native_source_domain_r32_v3/truerec_grpo/data/audit_history_overlap.py"),
)
SPLIT_DIR = RUNTIME / "truerec_grpo/data/splits"
RECORD_DIR = RUNTIME / "truerec_grpo/data/fixed_domain_abc"
OUTPUT = RUNTIME / "truerec_grpo/results/phase0_5"
RUN_PATH = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BETA-GAMMA-R32-2E-GC04-4GPU-20260825-054127"
)
FROZEN_MANIFEST_SHA = {
    "train_pool_group_ids.json": "ef645cf62cf50f619b62f2d0fbfb267dd4f8df6c5cfc89c70a7977df3aecf724",
    "dev_group_ids.json": "70341b476c57e4d7dc961ccf423ab3991a42375af1989bad040be4134e8f4fe4",
    "final_group_ids.json": "e6be46befa8ed320fb0762e6df23f553c6646297bdf5649003e332c04b2e61d0",
    "probe20_group_ids.json": "7193f540ee229ef53b7ca09e541f065396fe212a8cfd43893cceff499d83abdf",
}
EXPECTED_SPLIT_COUNTS = {"train_pool": 17971, "dev": 512, "final": 2048, "probe": 20}
DOMAIN_TOKEN = {domain: f"<|{domain}_begin|>" for domain in DOMAINS}
BRIDGES = (
    "该用户最近喜欢的视频有:",
    "该用户最近点击了商品:",
    "该用户最近感兴趣的广告有:",
    "该用户最近首次打赏了主播:",
)
ROUTE_RE = re.compile(r"/(think|no_think)$")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(raw)
            digest.update(raw)
            count += 1
    return digest.hexdigest(), count


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args], text=True, encoding="utf-8"
    ).strip()


def git_blob_sha(head: str, relative: Path) -> str:
    blob = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{head}:{relative.as_posix()}"]
    )
    return hashlib.sha256(blob).hexdigest()


def code_audit() -> dict[str, Any]:
    head, origin, status = git("rev-parse", "HEAD"), git("rev-parse", "origin/main"), git("status", "--short")
    parity = {}
    for relative, runtime in ((RELATIVE_SCRIPT, RUNTIME_SCRIPT),) + tuple(
        (relative, RUNTIME / "truerec_grpo/data" / relative.name) for relative in DEPENDENCIES
    ):
        github_sha = git_blob_sha(head, relative)
        source_sha = file_sha(SOURCE_REPO / relative)
        runtime_sha = file_sha(runtime)
        parity[relative.name] = {
            "github": github_sha, "source": source_sha, "runtime": runtime_sha,
            "pass": github_sha == source_sha == runtime_sha,
        }
    if head != origin or status or not all(value["pass"] for value in parity.values()):
        raise AuditError(f"CODE_AUDIT_GATE_FAIL head={head} origin={origin} status={status!r} parity={parity}")
    return {
        "implement_commit": head,
        "push_status": "PASS",
        "git_status_short": "EMPTY",
        "github_runtime_parity": "PASS",
        "files": parity,
    }


def sid_text(sid: tuple[str, int, int, int]) -> str:
    return f"<|{sid[0]}_begin|><s_a_{sid[1]}><s_b_{sid[2]}><s_c_{sid[3]}>"


def abc_text(sid: tuple[str, int, int, int]) -> str:
    return f"<s_a_{sid[1]}><s_b_{sid[2]}><s_c_{sid[3]}>"


def canonicalize_route(user_content: str) -> tuple[str, str]:
    match = ROUTE_RE.search(user_content)
    if not match:
        raise AuditError("TERMINAL_ROUTE_MARKER_MISSING")
    original_marker = "/" + match.group(1)
    canonical = user_content[: match.start()] + "/no_think"
    if user_content[: match.start()] + original_marker != user_content:
        raise AuditError("ROUTE_MARKER_ONLY_TRANSFORMATION_FAIL")
    return canonical, original_marker


def ordered_unique_sids(text: str) -> tuple[tuple[str, int, int, int], ...]:
    seen = set()
    result = []
    for match in SID_RE.finditer(text):
        sid = parse_sid(match.group(0))
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return tuple(result)


def has_bridge(system: str, user_content: str) -> bool:
    combined = system + "\n" + user_content
    return any(bridge in combined for bridge in BRIDGES)


def merge_source_row(
    groups: dict[str, dict[str, Any]],
    group_id: str,
    system: str,
    canonical_user: str,
    original_marker: str,
    target_domain: str,
    gold: tuple[tuple[str, int, int, int], ...],
    history: tuple[tuple[str, int, int, int], ...],
) -> None:
    target_history = tuple(sid for sid in history if sid[0] == target_domain)
    current = groups.get(group_id)
    if current is None:
        groups[group_id] = {
            "recommendation_group_id": group_id,
            "systems": {system},
            "users": {canonical_user},
            "target_domain": target_domain,
            "gold": gold,
            "history": history,
            "target_history": target_history,
            "markers": {original_marker},
            "source_rows": 1,
            "real_nothink_rows": int(original_marker == "/no_think"),
        }
        return
    current["systems"].add(system)
    current["users"].add(canonical_user)
    current["markers"].add(original_marker)
    current["source_rows"] += 1
    current["real_nothink_rows"] += int(original_marker == "/no_think")
    if current["target_domain"] != target_domain:
        raise AuditError(f"GROUP_DOMAIN_CONFLICT={group_id}")
    if current["gold"] != gold:
        raise AuditError(f"GROUP_GOLD_CONFLICT={group_id}")
    if current["history"] != history:
        raise AuditError(f"GROUP_HISTORY_CONFLICT={group_id}")


def load_canonical_source(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    parity_rows = []
    total_rows = recommendation_rows = 0
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            total_rows += 1
            row = json.loads(line)
            if row.get("data_source") != "recommend":
                continue
            recommendation_rows += 1
            try:
                metadata = json.loads(row["aux_metadata_json"])
                group_id = str(metadata["recommendation_group_id"])
                current_gold = parse_sid(str(metadata["recommendation_current_gold_sid"]))
                target_domain = current_gold[0]
                gold = normalize_gold(metadata["recommendation_all_gold_sids"], target_domain)
                user = join_alpaca_user(str(row["instruction"]), str(row["input"]))
                canonical_user, marker = canonicalize_route(user)
                history_text = extract_history_text(str(row["instruction"]), str(row["input"]))
                history = ordered_unique_sids(history_text)
                system = str(row["system"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise AuditError(f"SOURCE_SCHEMA_ERROR_ROW={row_number}") from exc
            merge_source_row(
                groups, group_id, system, canonical_user, marker, target_domain, gold, history
            )
            if marker == "/no_think":
                parity_rows.append({
                    "row_number": row_number,
                    "recommendation_group_id": group_id,
                    "system": system,
                    "user_content": user,
                    "target_domain": target_domain,
                    "output": str(row["output"]),
                    "current_gold": current_gold,
                })
    conflicts = sorted(
        group_id for group_id, group in groups.items()
        if len(group["systems"]) != 1 or len(group["users"]) != 1
    )
    result = []
    for group_id in sorted(groups):
        group = groups[group_id]
        overlap = classify_overlap(group["gold"], group["target_history"])
        result.append({
            "recommendation_group_id": group_id,
            "system": sorted(group["systems"])[0] if len(group["systems"]) == 1 else None,
            "user_content_nothink": sorted(group["users"])[0] if len(group["users"]) == 1 else None,
            "target_domain": group["target_domain"],
            "gold": group["gold"],
            "history": group["history"],
            "novelty": overlap["novelty"],
            "K": len(group["gold"]),
            "K_bucket": k_bucket(len(group["gold"])),
            "real_nothink_rows": group["real_nothink_rows"],
            "markers": sorted(group["markers"]),
        })
    return result, parity_rows, {
        "path": str(path), "sha256": file_sha(path), "total_rows": total_rows,
        "recommendation_rows": recommendation_rows,
        "canonical_prompt_conflict_groups": len(conflicts),
        "canonical_prompt_conflict_examples": conflicts[:20],
        "think_derived_nothink_groups": sum(group["real_nothink_rows"] == 0 for group in result),
        "route_marker_only_transformation": "PASS",
    }


def build_record(group: dict[str, Any], split: str) -> dict[str, Any]:
    gold_sids = [sid_text(sid) for sid in group["gold"]]
    gold_abc = [abc_text(sid) for sid in group["gold"]]
    return {
        "recommendation_group_id": group["recommendation_group_id"],
        "split": split,
        "target_domain": group["target_domain"],
        "novelty": group["novelty"],
        "K": group["K"],
        "K_bucket": group["K_bucket"],
        "system": group["system"],
        "user_content_nothink": group["user_content_nothink"],
        "fixed_domain_token": DOMAIN_TOKEN[group["target_domain"]],
        "all_gold_sids": gold_sids,
        "all_gold_abc": gold_abc,
        "history_sids": [sid_text(sid) for sid in group["history"]],
    }


def component_tokens(sid: tuple[str, int, int, int]) -> tuple[str, str, str]:
    return f"<s_a_{sid[1]}>", f"<s_b_{sid[2]}>", f"<s_c_{sid[3]}>"


def audit_token_contract(
    groups: list[dict[str, Any]], encode: Callable[[str], list[int]]
) -> dict[str, Any]:
    domain_ids = {domain: encode(token) for domain, token in DOMAIN_TOKEN.items()}
    components = {"A": set(), "B": set(), "C": set()}
    abc_actions = set()
    for group in groups:
        for sid in group["gold"]:
            a, b, c = component_tokens(sid)
            components["A"].add(a)
            components["B"].add(b)
            components["C"].add(c)
            abc_actions.add(a + b + c)
    component_failures = {
        level: sorted(token for token in tokens if len(encode(token)) != 1)
        for level, tokens in components.items()
    }
    abc_failures = sorted(action for action in abc_actions if len(encode(action)) != 3)
    domain_pass = all(len(ids) == 1 for ids in domain_ids.values())
    return {
        "domain_token_ids": {domain: ids for domain, ids in domain_ids.items()},
        "domain_token_single_token": "PASS" if domain_pass else "FAIL",
        "component_unique_counts": {level: len(tokens) for level, tokens in components.items()},
        "component_failure_counts": {level: len(values) for level, values in component_failures.items()},
        "component_failure_examples": {level: values[:20] for level, values in component_failures.items()},
        "A_component_single_token": "PASS" if not component_failures["A"] else "FAIL",
        "B_component_single_token": "PASS" if not component_failures["B"] else "FAIL",
        "C_component_single_token": "PASS" if not component_failures["C"] else "FAIL",
        "abc_unique_actions": len(abc_actions),
        "abc_failure_count": len(abc_failures),
        "abc_failure_examples": abc_failures[:20],
        "abc_action_exactly_3_tokens": "PASS"
        if domain_pass and not any(component_failures.values()) and not abc_failures else "FAIL",
    }


def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    for index in range(start, len(haystack) - len(needle) + 1):
        if haystack[index: index + len(needle)] == needle:
            return index
    return -1


def prefix_parity(
    prompt_ids: list[int], response_ids: list[int], rl_context_ids: list[int], full_sid_ids: list[int]
) -> tuple[bool, int]:
    full_ids = prompt_ids + response_ids
    position = find_subsequence(full_ids, full_sid_ids, start=len(prompt_ids))
    if position < 0 or len(full_sid_ids) != 4:
        return False, position
    prefix_end = position + 1
    passed = (
        rl_context_ids == full_ids[:prefix_end]
        and full_ids[prefix_end: prefix_end + 3] == full_sid_ids[1:]
    )
    return passed, position


def run_prefix_audit(
    parity_rows: list[dict[str, Any]], renderer: BetaGammaRenderer
) -> dict[str, Any]:
    prompt_cache: dict[str, list[int]] = {}
    context_cache: dict[str, list[int]] = {}
    passed = 0
    failures = []
    for row in parity_rows:
        group_id = row["recommendation_group_id"]
        if group_id not in prompt_cache:
            prompt_cache[group_id] = renderer.prompt_ids(row["system"], row["user_content"])
            context_cache[group_id] = renderer.rl_context_ids(
                row["system"], row["user_content"], DOMAIN_TOKEN[row["target_domain"]]
            )
        prompt_ids = prompt_cache[group_id]
        response_ids = renderer.assistant_ids(row["output"])
        sid = row["current_gold"]
        full_sid_ids = renderer.encode(DOMAIN_TOKEN[sid[0]] + abc_text(sid))
        ok, position = prefix_parity(
            prompt_ids, response_ids, context_cache[group_id], full_sid_ids
        )
        if ok:
            passed += 1
        elif len(failures) < 20:
            failures.append({
                "row_number": row["row_number"], "recommendation_group_id": group_id,
                "full_sid": sid_text(sid), "match_position": position,
                "prompt_tokens": len(prompt_ids), "context_tokens": len(context_cache[group_id]),
            })
    return {
        "sft_prefix_parity_rows": len(parity_rows),
        "sft_prefix_parity_pass": passed,
        "sft_prefix_parity_fail": len(parity_rows) - passed,
        "failure_examples": failures,
        "real_nothink_unique_prompt_cache": len(prompt_cache),
        "contract": "RL_CONTEXT_IDS == SFT full input_ids before first A action token",
    }


def load_frozen_splits() -> tuple[dict[str, set[str]], dict[str, Any]]:
    for name, expected_sha in FROZEN_MANIFEST_SHA.items():
        actual = file_sha(SPLIT_DIR / name)
        if actual != expected_sha:
            raise AuditError(f"FROZEN_MANIFEST_SHA_FAIL={name}:{actual}")
    train = set(json.loads((SPLIT_DIR / "train_pool_group_ids.json").read_text(encoding="utf-8")))
    dev = set(json.loads((SPLIT_DIR / "dev_group_ids.json").read_text(encoding="utf-8")))
    final = set(json.loads((SPLIT_DIR / "final_group_ids.json").read_text(encoding="utf-8")))
    probe_payload = json.loads((SPLIT_DIR / "probe20_group_ids.json").read_text(encoding="utf-8"))
    probe = {row["recommendation_group_id"] for row in probe_payload}
    splits = {"train_pool": train, "dev": dev, "final": final, "probe": probe}
    counts = {name: len(values) for name, values in splits.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise AuditError(f"FROZEN_SPLIT_COUNT_FAIL={counts}")
    if train & dev or train & final or dev & final or not probe <= dev:
        raise AuditError("FROZEN_SPLIT_SET_CONTRACT_FAIL")
    return splits, {"sha256": FROZEN_MANIFEST_SHA, "counts": counts}


def assign_records(
    groups: list[dict[str, Any]], splits: dict[str, set[str]]
) -> dict[str, list[dict[str, Any]]]:
    by_id = {group["recommendation_group_id"]: group for group in groups}
    full_ids = set(by_id)
    if splits["train_pool"] | splits["dev"] | splits["final"] != full_ids:
        raise AuditError("FROZEN_SPLIT_FULL_GROUP_MATCH_FAIL")
    records = {
        name: [build_record(by_id[group_id], name) for group_id in sorted(splits[name])]
        for name in ("train_pool", "dev", "final")
    }
    dev_by_id = {row["recommendation_group_id"]: row for row in records["dev"]}
    records["probe"] = [
        {**dev_by_id[group_id], "record_view": "dev_online_probe", "optimizer_eligible": False}
        for group_id in sorted(splits["probe"])
    ]
    return records


def render_review(
    audit: dict[str, Any], source_audit: dict[str, Any], record_audit: dict[str, Any],
    token_audit: dict[str, Any], prefix_audit: dict[str, Any], dataset_sha: dict[str, str]
) -> str:
    ids = token_audit["domain_token_ids"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT", "",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short']}",
        f"GITHUB_RUNTIME_PARITY={audit['github_runtime_parity']}", "",
        f"SOURCE_GROUPS={record_audit['source_groups']}", "",
        f"TRAIN_RECORDS={record_audit['train_records']}", f"DEV_RECORDS={record_audit['dev_records']}",
        f"FINAL_RECORDS={record_audit['final_records']}", f"PROBE_RECORDS={record_audit['probe_records']}", "",
        f"CANONICAL_PROMPT_CONFLICT_GROUPS={source_audit['canonical_prompt_conflict_groups']}",
        f"THINK_DERIVED_NOTHINK_GROUPS={source_audit['think_derived_nothink_groups']}",
        f"BRIDGE_CONTAMINATED_RECORDS={record_audit['bridge_contaminated_records']}", "",
        "DOMAIN_IS_CONTEXT=YES", "DOMAIN_IS_RL_ACTION=NO", "",
        f"VIDEO_BEGIN_TOKEN_ID={ids['video'][0]}", f"PROD_BEGIN_TOKEN_ID={ids['prod'][0]}",
        f"AD_BEGIN_TOKEN_ID={ids['ad'][0]}", f"LIVING_BEGIN_TOKEN_ID={ids['living'][0]}", "",
        f"DOMAIN_TOKEN_SINGLE_TOKEN={token_audit['domain_token_single_token']}",
        f"A_COMPONENT_SINGLE_TOKEN={token_audit['A_component_single_token']}",
        f"B_COMPONENT_SINGLE_TOKEN={token_audit['B_component_single_token']}",
        f"C_COMPONENT_SINGLE_TOKEN={token_audit['C_component_single_token']}",
        f"ABC_ACTION_EXACTLY_3_TOKENS={token_audit['abc_action_exactly_3_tokens']}", "",
        f"SFT_PREFIX_PARITY_ROWS={prefix_audit['sft_prefix_parity_rows']}",
        f"SFT_PREFIX_PARITY_PASS={prefix_audit['sft_prefix_parity_pass']}",
        f"SFT_PREFIX_PARITY_FAIL={prefix_audit['sft_prefix_parity_fail']}", "",
        f"EMPTY_GOLD_RECORDS={record_audit['empty_gold_records']}",
        f"GOLD_DOMAIN_MISMATCH_RECORDS={record_audit['gold_domain_mismatch_records']}", "",
        f"PROBE_SUBSET_OF_DEV={record_audit['probe_subset_of_dev']}",
        f"TRAIN_DEV_FINAL_MATCH_FROZEN_MANIFESTS={record_audit['train_dev_final_match_frozen_manifests']}", "",
        f"TRAIN_DATASET_SHA256={dataset_sha['train_records.jsonl']}",
        f"DEV_DATASET_SHA256={dataset_sha['dev_records.jsonl']}",
        f"FINAL_DATASET_SHA256={dataset_sha['final_records.jsonl']}",
        f"PROBE_DATASET_SHA256={dataset_sha['probe20_records.jsonl']}", "",
        "TEST_STATUS=PASS", "", "GPU_INFERENCE_STARTED=NO", "MODEL_FORWARD_STARTED=NO",
        "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "PROBE_STARTED=NO",
        "NEXT_EXPERIMENT_STARTED=NO",
    ]
    return "\n".join(lines) + "\n"


def run(test_status: str) -> None:
    if test_status != "PASS":
        raise AuditError("TEST_STATUS_GATE_FAIL")
    audit = code_audit()
    groups, parity_rows, source_audit = load_canonical_source(SOURCE)
    if source_audit["sha256"] != EXPECTED_SOURCE_SHA or len(groups) != EXPECTED_GROUPS:
        raise AuditError("SOURCE_CONTRACT_FAIL")
    if source_audit["canonical_prompt_conflict_groups"]:
        raise AuditError("CANONICAL_PROMPT_CONFLICT_GATE_FAIL")
    splits, split_audit = load_frozen_splits()
    records = assign_records(groups, splits)
    renderer = BetaGammaRenderer()
    token_audit = audit_token_contract(groups, renderer.encode)
    prefix_audit = run_prefix_audit(parity_rows, renderer)
    bridge_count = sum(has_bridge(group["system"], group["user_content_nothink"]) for group in groups)
    empty_gold = sum(not group["gold"] for group in groups)
    mismatch = sum(any(sid[0] != group["target_domain"] for sid in group["gold"]) for group in groups)
    record_audit = {
        "source_groups": len(groups),
        "train_records": len(records["train_pool"]), "dev_records": len(records["dev"]),
        "final_records": len(records["final"]), "probe_records": len(records["probe"]),
        "canonical_prompt_conflict_groups": source_audit["canonical_prompt_conflict_groups"],
        "bridge_contaminated_records": bridge_count,
        "domain_is_context": "YES", "domain_is_rl_action": "NO",
        "rl_action_levels": ["A", "B", "C"], "rl_action_token_count": 3,
        "empty_gold_records": empty_gold, "gold_domain_mismatch_records": mismatch,
        "probe_subset_of_dev": "PASS" if splits["probe"] <= splits["dev"] else "FAIL",
        "train_dev_final_match_frozen_manifests": "PASS",
        "probe_optimizer_eligible": False,
        "frozen_split": split_audit,
        "renderer": renderer.audit(),
    }
    hard_values = [
        bridge_count, empty_gold, mismatch, prefix_audit["sft_prefix_parity_fail"],
        int(token_audit["domain_token_single_token"] != "PASS"),
        int(token_audit["A_component_single_token"] != "PASS"),
        int(token_audit["B_component_single_token"] != "PASS"),
        int(token_audit["C_component_single_token"] != "PASS"),
        int(token_audit["abc_action_exactly_3_tokens"] != "PASS"),
    ]
    if any(hard_values):
        raise AuditError(f"RECORD_OR_TOKEN_HARD_GATE_FAIL={hard_values}")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "train_records.jsonl": records["train_pool"], "dev_records.jsonl": records["dev"],
        "final_records.jsonl": records["final"], "probe20_records.jsonl": records["probe"],
    }
    dataset_sha = {}
    dataset_counts = {}
    for name, rows in files.items():
        dataset_sha[name], dataset_counts[name] = write_jsonl(RECORD_DIR / name, rows)
    if dataset_counts != {
        "train_records.jsonl": 17971, "dev_records.jsonl": 512,
        "final_records.jsonl": 2048, "probe20_records.jsonl": 20,
    }:
        raise AuditError(f"WRITTEN_RECORD_COUNT_FAIL={dataset_counts}")
    source_audit["renderer_contract"] = renderer.audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "record_contract_audit.json", {"source": source_audit, "records": record_audit})
    write_json(OUTPUT / "tokenizer_contract_audit.json", token_audit)
    write_json(OUTPUT / "sft_prefix_parity_audit.json", prefix_audit)
    write_json(OUTPUT / "dataset_sha256.json", {"files": dataset_sha, "counts": dataset_counts})
    review = render_review(audit, source_audit, record_audit, token_audit, prefix_audit, dataset_sha)
    (OUTPUT / "CHATGPT_PHASE0_5_REVIEW.txt").write_text(review, encoding="utf-8")
    print(review, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-status", required=True, choices=("PASS", "FAIL"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise AuditError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
    run(args.test_status)


if __name__ == "__main__":
    main()
