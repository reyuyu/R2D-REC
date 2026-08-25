"""TrueRec-GRPO Phase 0.3 CPU-only History-Gold overlap census."""
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
from typing import Any


SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
RUNTIME = Path("/data/GRPO")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/truerec_grpo/data/audit_history_overlap.py"
)
RUNTIME_SCRIPT = RUNTIME / "truerec_grpo/data/audit_history_overlap.py"
OUTPUT = RUNTIME / "truerec_grpo/results/phase0_3"
SOURCE = Path("/data/lf_data_versions/alltrain/beta_gamma_v1/onereason_beta_gamma.jsonl")
EXPECTED_SOURCE_SHA = "72170e142a3db1ee7d0dd5b76ffb884f143a9fbf074906ea9fc91c9e8883ad28"
EXPECTED_GROUPS = 20531
DOMAINS = ("video", "prod", "ad", "living")
NOVELTY_CLASSES = ("H", "N2", "N1", "N0")
K_BUCKETS = ("K=1", "K=2", "K=3-5", "K=6-10", "K=11+")
SID_RE = re.compile(
    r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)


class AuditError(RuntimeError):
    pass


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args], text=True, encoding="utf-8"
    ).strip()


def code_audit() -> dict[str, str]:
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    status = git("status", "--short")
    source_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT)
    runtime_sha = file_sha(RUNTIME_SCRIPT)
    blob = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{head}:{RELATIVE_SCRIPT.as_posix()}"]
    )
    github_sha = hashlib.sha256(blob).hexdigest()
    if head != origin or status or github_sha != source_sha or source_sha != runtime_sha:
        raise AuditError(
            f"CODE_AUDIT_GATE_FAIL head={head} origin={origin} status={status!r} "
            f"github={github_sha} source={source_sha} runtime={runtime_sha}"
        )
    return {
        "implement_commit": head,
        "push_status": "PASS",
        "git_status_short": "EMPTY",
        "github_runtime_parity": "PASS",
        "github_script_sha256": github_sha,
        "runtime_script_sha256": runtime_sha,
    }


def parse_sid(value: str) -> tuple[str, int, int, int]:
    match = SID_RE.fullmatch(value)
    if not match:
        raise AuditError(f"INVALID_SID={value!r}")
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def extract_history_text(instruction: str, input_text: str) -> str:
    prompt = (instruction + input_text).strip()
    prompt = re.sub(r"/(?:no_)?think\s*$", "", prompt).rstrip()
    blocks = prompt.split("\n\n")
    sid_blocks = [index for index, block in enumerate(blocks) if SID_RE.search(block)]
    if not sid_blocks:
        raise AuditError("HISTORY_EXTRACTION_NO_SID")
    return "\n\n".join(blocks[: sid_blocks[-1] + 1]).strip()


def normalize_gold(values: Any, target_domain: str) -> tuple[tuple[str, int, int, int], ...]:
    if not isinstance(values, list) or not values:
        raise AuditError("ALL_GOLD_MUST_BE_NONEMPTY_LIST")
    unique = tuple(sorted({parse_sid(str(value)) for value in values}))
    if any(sid[0] != target_domain for sid in unique):
        raise AuditError(f"GOLD_DOMAIN_MISMATCH target={target_domain}")
    return unique


def merge_group(
    groups: dict[str, dict[str, Any]],
    group_id: str,
    route: str,
    target_domain: str,
    gold: tuple[tuple[str, int, int, int], ...],
    history: tuple[tuple[str, int, int, int], ...],
) -> None:
    current = groups.get(group_id)
    if current is None:
        groups[group_id] = {
            "recommendation_group_id": group_id,
            "target_domain": target_domain,
            "gold": gold,
            "history": history,
            "routes": {route},
            "source_rows": 1,
        }
        return
    if current["target_domain"] != target_domain:
        raise AuditError(f"GROUP_DOMAIN_CONFLICT={group_id}")
    if current["gold"] != gold:
        raise AuditError(f"GROUP_GOLD_CONFLICT={group_id}")
    if current["history"] != history:
        raise AuditError(f"GROUP_HISTORY_CONFLICT={group_id}")
    current["routes"].add(route)
    current["source_rows"] += 1


def load_groups(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    total_rows = recommendation_rows = 0
    routes: Counter[str] = Counter()
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
                current = parse_sid(str(metadata["recommendation_current_gold_sid"]))
                target_domain = current[0]
                gold = normalize_gold(metadata["recommendation_all_gold_sids"], target_domain)
                history_text = extract_history_text(str(row["instruction"]), str(row["input"]))
                history = tuple(sorted({
                    parse_sid(match.group(0))
                    for match in SID_RE.finditer(history_text)
                    if match.group(1) == target_domain
                }))
                route = str(row["source_segment"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise AuditError(f"SOURCE_SCHEMA_ERROR_ROW={row_number}") from exc
            routes[route] += 1
            merge_group(groups, group_id, route, target_domain, gold, history)
    result = sorted(groups.values(), key=lambda row: row["recommendation_group_id"])
    return result, {
        "path": str(path),
        "sha256": file_sha(path),
        "total_rows": total_rows,
        "recommendation_rows": recommendation_rows,
        "source_route_rows": dict(sorted(routes.items())),
    }


def classify_overlap(
    gold: tuple[tuple[str, int, int, int], ...],
    history: tuple[tuple[str, int, int, int], ...],
) -> dict[str, Any]:
    gold_abc, history_abc = set(gold), set(history)
    gold_ab = {sid[:3] for sid in gold}
    history_ab = {sid[:3] for sid in history}
    gold_a = {sid[:2] for sid in gold}
    history_a = {sid[:2] for sid in history}
    abc = bool(gold_abc & history_abc)
    ab = bool(gold_ab & history_ab)
    a = bool(gold_a & history_a)
    if abc:
        novelty = "H"
    elif ab:
        novelty = "N2"
    elif a:
        novelty = "N1"
    else:
        novelty = "N0"
    return {
        "GoldAInHistory": a,
        "GoldABInHistory": ab,
        "GoldABCInHistory": abc,
        "novelty": novelty,
    }


def k_bucket(k: int) -> str:
    if k == 1:
        return "K=1"
    if k == 2:
        return "K=2"
    if 3 <= k <= 5:
        return "K=3-5"
    if 6 <= k <= 10:
        return "K=6-10"
    if k >= 11:
        return "K=11+"
    raise AuditError(f"INVALID_K={k}")


def annotate(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for group in groups:
        overlap = classify_overlap(group["gold"], group["history"])
        annotated.append({
            "target_domain": group["target_domain"],
            "K": len(group["gold"]),
            "K_bucket": k_bucket(len(group["gold"])),
            **overlap,
        })
    return annotated


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = Counter(row["novelty"] for row in rows)
    if not n:
        return {
            "N": 0,
            "GoldAInHistory_rate": 0.0,
            "GoldABInHistory_rate": 0.0,
            "GoldABCInHistory_rate": 0.0,
            "novelty": {name: {"count": 0, "rate": 0.0} for name in NOVELTY_CLASSES},
        }
    return {
        "N": n,
        "GoldAInHistory_rate": round(sum(row["GoldAInHistory"] for row in rows) / n, 8),
        "GoldABInHistory_rate": round(sum(row["GoldABInHistory"] for row in rows) / n, 8),
        "GoldABCInHistory_rate": round(sum(row["GoldABCInHistory"] for row in rows) / n, 8),
        "novelty": {
            name: {"count": counts[name], "rate": round(counts[name] / n, 8)}
            for name in NOVELTY_CLASSES
        },
    }


def build_census(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    overall = summarize(rows)
    domains = {
        domain: summarize([row for row in rows if row["target_domain"] == domain])
        for domain in DOMAINS
    }
    if sum(value["N"] for value in domains.values()) != overall["N"]:
        raise AuditError("DOMAIN_COUNT_SUM_FAIL")
    class_sum = sum(overall["novelty"][name]["count"] for name in NOVELTY_CLASSES)
    exclusive = all(row["novelty"] in NOVELTY_CLASSES for row in rows)
    exhaustive = class_sum == overall["N"]
    if not exclusive or not exhaustive:
        raise AuditError("NOVELTY_PARTITION_FAIL")
    census = {
        "overall": overall,
        "domains": domains,
        "class_exclusive": "PASS",
        "class_exhaustive": "PASS",
        "definitions": {
            "H": "GoldABCInHistory=true",
            "N2": "GoldABCInHistory=false and GoldABInHistory=true",
            "N1": "GoldABInHistory=false and GoldAInHistory=true",
            "N0": "GoldAInHistory=false",
        },
    }
    table = {
        "domains": {
            domain: {
                bucket: summarize([
                    row for row in rows
                    if row["target_domain"] == domain and row["K_bucket"] == bucket
                ])
                for bucket in K_BUCKETS
            }
            for domain in DOMAINS
        }
    }
    return census, table


def render_review(audit: dict[str, str], census: dict[str, Any]) -> str:
    overall = census["overall"]
    novelty = overall["novelty"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}",
        "RESULT_COMMIT=PENDING_REPORT_COMMIT", "",
        f"PUSH_STATUS={audit['push_status']}",
        f"GIT_STATUS_SHORT={audit['git_status_short']}",
        f"GITHUB_RUNTIME_PARITY={audit['github_runtime_parity']}", "",
        f"GROUPS={overall['N']}", "",
        f"GOLD_A_IN_HISTORY_RATE={overall['GoldAInHistory_rate']}",
        f"GOLD_AB_IN_HISTORY_RATE={overall['GoldABInHistory_rate']}",
        f"GOLD_ABC_IN_HISTORY_RATE={overall['GoldABCInHistory_rate']}", "",
        f"H_N={novelty['H']['count']}", f"N2_N={novelty['N2']['count']}",
        f"N1_N={novelty['N1']['count']}", f"N0_N={novelty['N0']['count']}", "",
        f"VIDEO_H_RATE={census['domains']['video']['novelty']['H']['rate']}",
        f"PROD_H_RATE={census['domains']['prod']['novelty']['H']['rate']}",
        f"AD_H_RATE={census['domains']['ad']['novelty']['H']['rate']}",
        f"LIVING_H_RATE={census['domains']['living']['novelty']['H']['rate']}", "",
        f"CLASS_EXCLUSIVE={census['class_exclusive']}",
        f"CLASS_EXHAUSTIVE={census['class_exhaustive']}", "",
        "TEST_STATUS=PASS", "", "GPU_INFERENCE_STARTED=NO", "MODEL_FORWARD_STARTED=NO",
        "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "PROBE_STARTED=NO",
        "NEXT_EXPERIMENT_STARTED=NO",
    ]
    return "\n".join(lines) + "\n"


def run(test_status: str) -> None:
    if "torch" in sys.modules:
        raise AuditError("TORCH_ALREADY_IMPORTED")
    if test_status != "PASS":
        raise AuditError("TEST_STATUS_GATE_FAIL")
    audit = code_audit()
    groups, source = load_groups(SOURCE)
    if source["sha256"] != EXPECTED_SOURCE_SHA:
        raise AuditError("SOURCE_SHA_FAIL")
    if len(groups) != EXPECTED_GROUPS:
        raise AuditError(f"GROUP_COUNT_FAIL={len(groups)}")
    rows = annotate(groups)
    census, table = build_census(rows)
    census["source"] = source
    census["group_key"] = "recommendation_group_id"
    census["gold_contract"] = "unique recommendation_all_gold_sids"
    census["history_contract"] = "target-domain SID only from behavior-history blocks"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "history_overlap_census.json", census)
    write_json(OUTPUT / "domain_k_novelty_table.json", table)
    review = render_review(audit, census)
    (OUTPUT / "CHATGPT_PHASE0_3_REVIEW.txt").write_text(review, encoding="utf-8")
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
