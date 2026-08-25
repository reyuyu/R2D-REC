"""TrueRec-GRPO Phase 0.2 CPU-only Gold-K census."""
from __future__ import annotations

from collections import Counter
import argparse
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


SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
RUNTIME = Path("/data/GRPO")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/truerec_grpo/data/audit_gold_k_distribution.py"
)
RUNTIME_SCRIPT = RUNTIME / "truerec_grpo/data/audit_gold_k_distribution.py"
OUTPUT = RUNTIME / "truerec_grpo/results/phase0_2"
FULL_SOURCE = Path(
    "/data/lf_data_versions/alltrain/beta_gamma_v1/onereason_beta_gamma.jsonl"
)
OLD_SOURCE = Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl")
OLD_MANIFEST = Path("/data/GRPO/data/rec_mp_grpo_v2/manifest.json")
EXPECTED_FULL_SHA = "72170e142a3db1ee7d0dd5b76ffb884f143a9fbf074906ea9fc91c9e8883ad28"
EXPECTED_OLD_SHA = "791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc"
EXPECTED_FULL_GROUPS = 20531
EXPECTED_OLD_GROUPS = 1549
DOMAINS = ("video", "prod", "ad", "living")
THRESHOLDS = (1, 2, 3, 4, 5, 8, 10)
BUCKETS = ("K=1", "K=2", "K=3-5", "K=6-10", "K=11+")
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


def code_audit() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    status = git("status", "--short")
    source_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT)
    runtime_sha = file_sha(RUNTIME_SCRIPT)
    blob = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{head}:{RELATIVE_SCRIPT.as_posix()}"]
    )
    github_sha = hashlib.sha256(blob).hexdigest()
    parity = github_sha == source_sha == runtime_sha
    if head != origin or status or not parity:
        raise AuditError(
            f"CODE_AUDIT_GATE_FAIL head={head} origin={origin} status={status!r} "
            f"github={github_sha} source={source_sha} runtime={runtime_sha}"
        )
    return {
        "implement_commit": head,
        "push_status": "PASS",
        "git_status_short": "EMPTY",
        "github_script_sha256": github_sha,
        "runtime_script_sha256": runtime_sha,
        "github_runtime_parity": "PASS",
    }


def parse_sid(value: str) -> tuple[str, int, int, int]:
    match = SID_RE.fullmatch(value)
    if not match:
        raise AuditError(f"INVALID_GOLD_SID={value!r}")
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def normalize_gold(values: Any, target_domain: str) -> tuple[tuple[str, int, int, int], ...]:
    if not isinstance(values, list) or not values:
        raise AuditError("ALL_GOLD_MUST_BE_NONEMPTY_LIST")
    unique = tuple(sorted({parse_sid(str(value)) for value in values}))
    if any(sid[0] != target_domain for sid in unique):
        raise AuditError(f"GOLD_DOMAIN_MISMATCH target={target_domain} gold={unique}")
    return unique


def merge_group(
    groups: dict[str, dict[str, Any]],
    group_id: str,
    route: str,
    target_domain: str,
    gold: tuple[tuple[str, int, int, int], ...],
) -> None:
    if target_domain not in DOMAINS:
        raise AuditError(f"INVALID_TARGET_DOMAIN={target_domain!r}")
    current = groups.get(group_id)
    if current is None:
        groups[group_id] = {
            "recommendation_group_id": group_id,
            "target_domain": target_domain,
            "gold": gold,
            "routes": {route},
            "source_rows": 1,
        }
        return
    if current["target_domain"] != target_domain:
        raise AuditError(f"GROUP_DOMAIN_CONFLICT={group_id}")
    if current["gold"] != gold:
        raise AuditError(f"GROUP_GOLD_CONFLICT={group_id}")
    current["routes"].add(route)
    current["source_rows"] += 1


def load_full(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                gold = normalize_gold(metadata["recommendation_all_gold_sids"], current[0])
                route = str(row["source_segment"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise AuditError(f"FULL_SCHEMA_ERROR_ROW={row_number}") from exc
            routes[route] += 1
            merge_group(groups, group_id, route, current[0], gold)
    result = sorted(groups.values(), key=lambda row: row["recommendation_group_id"])
    return result, {
        "path": str(path),
        "sha256": file_sha(path),
        "total_rows": total_rows,
        "recommendation_rows": recommendation_rows,
        "source_route_rows": dict(sorted(routes.items())),
    }


def load_old(path: Path, expected_groups: int = EXPECTED_OLD_GROUPS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    total_rows = 0
    routes: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            total_rows += 1
            row = json.loads(line)
            try:
                group_id = str(row["recommendation_group_id"])
                route = str(row["route"])
                target_domain = str(row["target_domain"])
                gold = normalize_gold(row["all_gold_sids"], target_domain)
            except (KeyError, TypeError) as exc:
                raise AuditError(f"OLD_SCHEMA_ERROR_ROW={row_number}") from exc
            routes[route] += 1
            merge_group(groups, group_id, route, target_domain, gold)
    if len(groups) != expected_groups:
        raise AuditError(f"OLD_UNIQUE_GROUPS_EXPECTED={expected_groups}_ACTUAL={len(groups)}")
    if any(group["routes"] != {"think", "no_think"} for group in groups.values()):
        raise AuditError("OLD_ROUTE_PAIR_CONTRACT_FAIL")
    result = sorted(groups.values(), key=lambda row: row["recommendation_group_id"])
    return result, {
        "path": str(path),
        "sha256": file_sha(path),
        "total_rows": total_rows,
        "source_route_rows": dict(sorted(routes.items())),
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


def nearest_rank(values: list[int], probability: float) -> int:
    if not values:
        raise AuditError("EMPTY_PERCENTILE_INPUT")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summarize_values(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "N": 0,
            "K_mean": None,
            "K_median": None,
            "K_p75": None,
            "K_p90": None,
            "K_max": None,
            "buckets": {
                bucket: {"count": 0, "rate": 0.0}
                for bucket in BUCKETS
            },
        }
    counts = Counter(k_bucket(value) for value in values)
    n = len(values)
    return {
        "N": n,
        "K_mean": round(statistics.fmean(values), 6),
        "K_median": statistics.median(values),
        "K_p75": nearest_rank(values, 0.75),
        "K_p90": nearest_rank(values, 0.90),
        "K_max": max(values),
        "buckets": {
            bucket: {"count": counts[bucket], "rate": round(counts[bucket] / n, 8)}
            for bucket in BUCKETS
        },
    }


def census(groups: list[dict[str, Any]]) -> dict[str, Any]:
    overall = summarize_values([len(group["gold"]) for group in groups])
    domains = {
        domain: summarize_values(
            [len(group["gold"]) for group in groups if group["target_domain"] == domain]
        )
        for domain in DOMAINS
    }
    if sum(value["N"] for value in domains.values()) != overall["N"]:
        raise AuditError("DOMAIN_COUNT_SUM_FAIL")
    return {
        "overall": overall,
        "domains": domains,
        "percentile_method": "median=standard midpoint; p75/p90=nearest-rank",
    }


def compare_census(full: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    def comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {
            "old_minus_full_K_mean": round(right["K_mean"] - left["K_mean"], 6),
            "old_minus_full_K_median": right["K_median"] - left["K_median"],
            "bucket_percentage_point_drift_old_minus_full": {
                bucket: round(
                    100 * (right["buckets"][bucket]["rate"] - left["buckets"][bucket]["rate"]),
                    6,
                )
                for bucket in BUCKETS
            },
        }
    return {
        "direction": "OLD_GRPO_1549 minus FULL_BETA_GAMMA",
        "overall": comparison(full["overall"], old["overall"]),
        "domains": {
            domain: comparison(full["domains"][domain], old["domains"][domain])
            for domain in DOMAINS
        },
    }


def threshold_counts(groups: list[dict[str, Any]]) -> dict[str, Any]:
    def count_for(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
        materialized = list(rows)
        return {
            f"K>={threshold}": sum(len(group["gold"]) >= threshold for group in materialized)
            for threshold in THRESHOLDS
        }
    return {
        "overall": count_for(groups),
        "domains": {
            domain: count_for(group for group in groups if group["target_domain"] == domain)
            for domain in DOMAINS
        },
    }


def render_review(
    audit: dict[str, Any], full: dict[str, Any], old: dict[str, Any], thresholds: dict[str, Any]
) -> str:
    f, o = full["overall"], old["overall"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}",
        "RESULT_COMMIT=PENDING_REPORT_COMMIT",
        "",
        f"PUSH_STATUS={audit['push_status']}",
        f"GIT_STATUS_SHORT={audit['git_status_short']}",
        f"GITHUB_RUNTIME_PARITY={audit['github_runtime_parity']}",
        "",
        f"FULL_GROUPS={f['N']}",
        f"OLD_GRPO_GROUPS={o['N']}",
        "",
        f"FULL_K_MEAN={f['K_mean']}", f"FULL_K_MEDIAN={f['K_median']}",
        f"FULL_K_P90={f['K_p90']}", f"FULL_K_MAX={f['K_max']}", "",
    ]
    for domain in DOMAINS:
        value = full["domains"][domain]
        lines.extend([
            f"{domain.upper()}_K_MEAN={value['K_mean']}",
            f"{domain.upper()}_K_MEDIAN={value['K_median']}", "",
        ])
    lines.extend([
        f"FULL_K1_RATE={f['buckets']['K=1']['rate']}",
        f"FULL_K2_RATE={f['buckets']['K=2']['rate']}",
        f"FULL_K3_5_RATE={f['buckets']['K=3-5']['rate']}",
        f"FULL_K6_10_RATE={f['buckets']['K=6-10']['rate']}",
        f"FULL_K11PLUS_RATE={f['buckets']['K=11+']['rate']}", "",
        f"OLD_K_MEAN={o['K_mean']}", f"OLD_K_MEDIAN={o['K_median']}", "",
        f"OLD_K1_RATE={o['buckets']['K=1']['rate']}",
        f"OLD_K2_RATE={o['buckets']['K=2']['rate']}",
        f"OLD_K3_5_RATE={o['buckets']['K=3-5']['rate']}",
        f"OLD_K6_10_RATE={o['buckets']['K=6-10']['rate']}",
        f"OLD_K11PLUS_RATE={o['buckets']['K=11+']['rate']}", "",
    ])
    for threshold in (2, 3, 4, 5, 8, 10):
        lines.append(f"K_GE_{threshold}_GROUPS={thresholds['overall'][f'K>={threshold}']}")
    lines.extend([
        "", "TEST_STATUS=PASS", "", "GPU_INFERENCE_STARTED=NO",
        "MODEL_FORWARD_STARTED=NO", "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0",
        "PROBE_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
    ])
    return "\n".join(lines) + "\n"


def run(test_status: str) -> None:
    if "torch" in sys.modules:
        raise AuditError("TORCH_ALREADY_IMPORTED")
    if test_status != "PASS":
        raise AuditError("TEST_STATUS_GATE_FAIL")
    audit = code_audit()
    full_groups, full_source = load_full(FULL_SOURCE)
    old_groups, old_source = load_old(OLD_SOURCE)
    if full_source["sha256"] != EXPECTED_FULL_SHA:
        raise AuditError("FULL_SOURCE_SHA_FAIL")
    if old_source["sha256"] != EXPECTED_OLD_SHA:
        raise AuditError("OLD_SOURCE_SHA_FAIL")
    if len(full_groups) != EXPECTED_FULL_GROUPS:
        raise AuditError(f"FULL_GROUP_COUNT_FAIL={len(full_groups)}")
    manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("grpo_train_records") != old_source["total_rows"]:
        raise AuditError("OLD_MANIFEST_ROW_COUNT_FAIL")
    full = census(full_groups)
    old = census(old_groups)
    comparison = compare_census(full, old)
    thresholds = threshold_counts(full_groups)
    full["source"] = full_source
    full["group_key"] = "recommendation_group_id"
    full["K_definition"] = "count(unique valid recommendation_all_gold_sids)"
    old["source"] = old_source
    old["manifest"] = str(OLD_MANIFEST)
    old["group_key"] = "recommendation_group_id"
    old["K_definition"] = "count(unique valid all_gold_sids)"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "full_k_census.json", full)
    write_json(OUTPUT / "old_grpo_k_census.json", old)
    write_json(OUTPUT / "full_vs_old_k_comparison.json", comparison)
    write_json(OUTPUT / "k_threshold_counts.json", thresholds)
    review = render_review(audit, full, old, thresholds)
    (OUTPUT / "CHATGPT_PHASE0_2_REVIEW.txt").write_text(review, encoding="utf-8")
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
