"""TrueRec-GRPO Phase 0.4 CPU-only frozen RL split and diagnostic probe builder."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable


DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from audit_history_overlap import (  # noqa: E402
    AuditError,
    DOMAINS,
    EXPECTED_GROUPS,
    EXPECTED_SOURCE_SHA,
    NOVELTY_CLASSES,
    SOURCE,
    annotate,
    load_groups,
)


SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
RUNTIME = Path("/data/GRPO")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/truerec_grpo/data/build_frozen_rl_split.py"
)
RUNTIME_SCRIPT = RUNTIME / "truerec_grpo/data/build_frozen_rl_split.py"
DEPENDENCY_RELATIVE = Path(
    "baselines/native_source_domain_r32_v3/truerec_grpo/data/audit_history_overlap.py"
)
DEPENDENCY_RUNTIME = RUNTIME / "truerec_grpo/data/audit_history_overlap.py"
SPLIT_DIR = RUNTIME / "truerec_grpo/data/splits"
OUTPUT = RUNTIME / "truerec_grpo/results/phase0_4"
SEED = "20260825"
TRAIN_SIZE = 17971
DEV_SIZE = 512
FINAL_SIZE = 2048
PROBE_SIZE = 20
K_BUCKETS = ("K=1", "K=2", "K=3-5", "K=6-10", "K=11+")
PROBE_TARGET = {"N0": 2, "N1": 1, "N2": 1, "H": 1}
MANIFEST_NAMES = (
    "train_pool_group_ids.json",
    "dev_group_ids.json",
    "final_group_ids.json",
    "probe20_group_ids.json",
    "split_manifest.json",
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def value_sha(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args], text=True, encoding="utf-8"
    ).strip()


def git_blob_sha(head: str, relative: Path) -> str:
    blob = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{head}:{relative.as_posix()}"]
    )
    return hashlib.sha256(blob).hexdigest()


def code_audit() -> dict[str, str]:
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    status = git("status", "--short")
    github_sha = git_blob_sha(head, RELATIVE_SCRIPT)
    source_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT)
    runtime_sha = file_sha(RUNTIME_SCRIPT)
    dependency_github = git_blob_sha(head, DEPENDENCY_RELATIVE)
    dependency_source = file_sha(SOURCE_REPO / DEPENDENCY_RELATIVE)
    dependency_runtime = file_sha(DEPENDENCY_RUNTIME)
    parity = (
        github_sha == source_sha == runtime_sha
        and dependency_github == dependency_source == dependency_runtime
    )
    if head != origin or status or not parity:
        raise AuditError(
            f"CODE_AUDIT_GATE_FAIL head={head} origin={origin} status={status!r} "
            f"script={github_sha},{source_sha},{runtime_sha} "
            f"dependency={dependency_github},{dependency_source},{dependency_runtime}"
        )
    return {
        "implement_commit": head,
        "push_status": "PASS",
        "git_status_short": "EMPTY",
        "github_runtime_parity": "PASS",
        "github_script_sha256": github_sha,
        "runtime_script_sha256": runtime_sha,
        "dependency_sha256": dependency_github,
    }


def stable_score(group_id: str, seed: str = SEED) -> str:
    return hashlib.sha256((seed + group_id).encode("utf-8")).hexdigest()


def largest_remainder(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target < 0 or target > total or not counts:
        raise AuditError(f"INVALID_ALLOCATION_TARGET={target}_TOTAL={total}")
    exact = {key: counts[key] * target / total for key in counts}
    allocation = {key: int(exact[key]) for key in counts}
    remaining = target - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != target or any(allocation[key] > counts[key] for key in counts):
        raise AuditError("LARGEST_REMAINDER_ALLOCATION_FAIL")
    return allocation


def enrich_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlap_rows = annotate(groups)
    enriched = []
    for group, overlap in zip(groups, overlap_rows, strict=True):
        enriched.append({
            "recommendation_group_id": group["recommendation_group_id"],
            "target_domain": overlap["target_domain"],
            "novelty": overlap["novelty"],
            "K": overlap["K"],
            "K_bucket": overlap["K_bucket"],
        })
    return enriched


def build_split(
    groups: list[dict[str, Any]],
    final_size: int = FINAL_SIZE,
    dev_size: int = DEV_SIZE,
    seed: str = SEED,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    strata: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        key = f"{group['target_domain']}|{group['novelty']}"
        strata.setdefault(key, []).append(group)
    counts = {key: len(rows) for key, rows in strata.items()}
    final_allocation = largest_remainder(counts, final_size)
    dev_allocation = largest_remainder(counts, dev_size)
    if any(final_allocation[key] + dev_allocation[key] > counts[key] for key in counts):
        raise AuditError("STRATUM_CAPACITY_FAIL")
    split = {"train": [], "dev": [], "final": []}
    allocations: dict[str, dict[str, int]] = {}
    for key in sorted(strata):
        ordered = sorted(
            strata[key],
            key=lambda row: (stable_score(row["recommendation_group_id"], seed), row["recommendation_group_id"]),
        )
        final_n = final_allocation[key]
        dev_n = dev_allocation[key]
        split["final"].extend(ordered[:final_n])
        split["dev"].extend(ordered[final_n: final_n + dev_n])
        split["train"].extend(ordered[final_n + dev_n:])
        allocations[key] = {
            "full": len(ordered),
            "final": final_n,
            "dev": dev_n,
            "train": len(ordered) - final_n - dev_n,
        }
    for name in split:
        split[name].sort(key=lambda row: row["recommendation_group_id"])
    return split, allocations


def select_probe(
    dev: list[dict[str, Any]], seed: str = SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    fallbacks: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for novelty, wanted in PROBE_TARGET.items():
            candidates = sorted(
                [
                    row for row in dev
                    if row["target_domain"] == domain and row["novelty"] == novelty
                ],
                key=lambda row: (
                    row["K"],
                    stable_score(row["recommendation_group_id"], seed),
                    row["recommendation_group_id"],
                ),
            )
            chosen = candidates[:wanted]
            selected.extend(chosen)
            selected_ids.update(row["recommendation_group_id"] for row in chosen)
            if len(chosen) < wanted:
                fallbacks.append({
                    "target_domain": domain,
                    "novelty": novelty,
                    "requested": wanted,
                    "available": len(chosen),
                    "shortfall": wanted - len(chosen),
                })
        domain_selected = sum(row["target_domain"] == domain for row in selected)
        if domain_selected < 5:
            candidates = sorted(
                [
                    row for row in dev
                    if row["target_domain"] == domain
                    and row["recommendation_group_id"] not in selected_ids
                ],
                key=lambda row: (
                    row["K"],
                    stable_score(row["recommendation_group_id"], seed),
                    row["recommendation_group_id"],
                ),
            )
            extra = candidates[: 5 - domain_selected]
            selected.extend(extra)
            selected_ids.update(row["recommendation_group_id"] for row in extra)
    selected.sort(
        key=lambda row: (
            DOMAINS.index(row["target_domain"]),
            tuple(PROBE_TARGET).index(row["novelty"]),
            row["K"],
            stable_score(row["recommendation_group_id"], seed),
        )
    )
    if len(selected) != PROBE_SIZE or len(selected_ids) != PROBE_SIZE:
        raise AuditError(f"PROBE_SIZE_OR_UNIQUENESS_FAIL={len(selected)},{len(selected_ids)}")
    return selected, fallbacks


def id_set(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {row["recommendation_group_id"] for row in rows}


def validate_split(
    full: list[dict[str, Any]], split: dict[str, list[dict[str, Any]]], probe: list[dict[str, Any]]
) -> dict[str, Any]:
    train_ids, dev_ids, final_ids = id_set(split["train"]), id_set(split["dev"]), id_set(split["final"])
    full_ids, probe_ids = id_set(full), id_set(probe)
    gates = {
        "source_groups": len(full_ids),
        "train_pool_groups": len(train_ids),
        "dev_groups": len(dev_ids),
        "final_groups": len(final_ids),
        "probe_groups": len(probe_ids),
        "train_dev_overlap": len(train_ids & dev_ids),
        "train_final_overlap": len(train_ids & final_ids),
        "dev_final_overlap": len(dev_ids & final_ids),
        "probe_subset_of_dev": "PASS" if probe_ids <= dev_ids else "FAIL",
        "all_groups_covered_once": "PASS"
        if train_ids | dev_ids | final_ids == full_ids
        and len(train_ids) + len(dev_ids) + len(final_ids) == len(full_ids)
        else "FAIL",
        "k1_removed_from_train": "NO" if any(row["K"] == 1 for row in split["train"]) else "YES",
    }
    expected = {
        "source_groups": EXPECTED_GROUPS,
        "train_pool_groups": TRAIN_SIZE,
        "dev_groups": DEV_SIZE,
        "final_groups": FINAL_SIZE,
        "probe_groups": PROBE_SIZE,
        "train_dev_overlap": 0,
        "train_final_overlap": 0,
        "dev_final_overlap": 0,
        "probe_subset_of_dev": "PASS",
        "all_groups_covered_once": "PASS",
        "k1_removed_from_train": "NO",
    }
    if gates != expected:
        raise AuditError(f"SPLIT_GATE_FAIL actual={gates} expected={expected}")
    for name in ("train", "dev", "final"):
        domains = Counter(row["target_domain"] for row in split[name])
        novelty = Counter(row["novelty"] for row in split[name])
        if any(domains[domain] == 0 for domain in DOMAINS):
            raise AuditError(f"ZERO_DOMAIN_IN_SPLIT={name}")
        if any(novelty[value] == 0 for value in NOVELTY_CLASSES):
            raise AuditError(f"ZERO_NOVELTY_IN_SPLIT={name}")
    return gates


def categorical(rows: list[dict[str, Any]], key_fn, categories: list[str]) -> dict[str, Any]:
    counts = Counter(key_fn(row) for row in rows)
    n = len(rows)
    return {
        category: {"count": counts[category], "rate": round(counts[category] / n, 8)}
        for category in categories
    }


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    k_values = [row["K"] for row in rows]
    domain_novelty = [f"{domain}|{novelty}" for domain in DOMAINS for novelty in NOVELTY_CLASSES]
    domain_k = [f"{domain}|{bucket}" for domain in DOMAINS for bucket in K_BUCKETS]
    return {
        "N": len(rows),
        "domain": categorical(rows, lambda row: row["target_domain"], list(DOMAINS)),
        "novelty": categorical(rows, lambda row: row["novelty"], list(NOVELTY_CLASSES)),
        "K_mean": round(statistics.fmean(k_values), 6),
        "K_median": statistics.median(k_values),
        "K_bucket": categorical(rows, lambda row: row["K_bucket"], list(K_BUCKETS)),
        "domain_novelty": categorical(
            rows, lambda row: f"{row['target_domain']}|{row['novelty']}", domain_novelty
        ),
        "domain_K_bucket": categorical(
            rows, lambda row: f"{row['target_domain']}|{row['K_bucket']}", domain_k
        ),
    }


def drift(full: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "K_mean_difference": round(candidate["K_mean"] - full["K_mean"], 6),
        "K_median_difference": candidate["K_median"] - full["K_median"],
    }
    for field in ("domain", "novelty", "K_bucket", "domain_novelty", "domain_K_bucket"):
        result[f"{field}_percentage_point_drift"] = {
            key: round(100 * (candidate[field][key]["rate"] - full[field][key]["rate"]), 6)
            for key in full[field]
        }
    return result


def manifest_payloads(
    split: dict[str, list[dict[str, Any]]],
    probe: list[dict[str, Any]],
    allocations: dict[str, dict[str, int]],
    source_meta: dict[str, Any],
    seed: str = SEED,
) -> dict[str, Any]:
    train_ids = sorted(id_set(split["train"]))
    dev_ids = sorted(id_set(split["dev"]))
    final_ids = sorted(id_set(split["final"]))
    probe_rows = [
        {
            "recommendation_group_id": row["recommendation_group_id"],
            "target_domain": row["target_domain"],
            "novelty": row["novelty"],
            "K": row["K"],
            "K_bucket": row["K_bucket"],
        }
        for row in probe
    ]
    split_manifest = {
        "name": "truerec_grpo_frozen_rl_split_v1",
        "seed": seed,
        "hash_contract": "sha256(seed + recommendation_group_id)",
        "stratum": "target_domain x novelty",
        "allocation": "proportional largest remainder; Final then Dev in stable stratum order",
        "source": source_meta,
        "counts": {
            "full": len(train_ids) + len(dev_ids) + len(final_ids),
            "train_pool": len(train_ids),
            "dev": len(dev_ids),
            "final_rl_heldout": len(final_ids),
            "probe20_dev_subset": len(probe_rows),
        },
        "stratum_allocations": allocations,
        "heldout_semantics": "Dev and Final never enter RL optimizer; RL-heldout, not SFT-unseen",
    }
    return {
        "train_pool_group_ids.json": train_ids,
        "dev_group_ids.json": dev_ids,
        "final_group_ids.json": final_ids,
        "probe20_group_ids.json": probe_rows,
        "split_manifest.json": split_manifest,
    }


def render_review(
    audit: dict[str, str], gates: dict[str, Any], probe_audit: dict[str, Any], hashes: dict[str, str]
) -> str:
    domain = probe_audit["domain_counts"]
    novelty = probe_audit["novelty_counts"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT", "",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short']}",
        f"GITHUB_RUNTIME_PARITY={audit['github_runtime_parity']}", "",
        f"SOURCE_GROUPS={gates['source_groups']}", "",
        f"TRAIN_POOL_GROUPS={gates['train_pool_groups']}", f"DEV_GROUPS={gates['dev_groups']}",
        f"FINAL_GROUPS={gates['final_groups']}", f"PROBE_GROUPS={gates['probe_groups']}", "",
        f"TRAIN_DEV_OVERLAP={gates['train_dev_overlap']}",
        f"TRAIN_FINAL_OVERLAP={gates['train_final_overlap']}",
        f"DEV_FINAL_OVERLAP={gates['dev_final_overlap']}", "",
        f"PROBE_VIDEO={domain['video']}", f"PROBE_PROD={domain['prod']}",
        f"PROBE_AD={domain['ad']}", f"PROBE_LIVING={domain['living']}", "",
        f"PROBE_H={novelty['H']}", f"PROBE_N2={novelty['N2']}",
        f"PROBE_N1={novelty['N1']}", f"PROBE_N0={novelty['N0']}", "",
        f"PROBE_SUBSET_OF_DEV={gates['probe_subset_of_dev']}",
        f"ALL_GROUPS_COVERED_ONCE={gates['all_groups_covered_once']}",
        f"K1_REMOVED_FROM_TRAIN={gates['k1_removed_from_train']}",
        "SPLIT_DETERMINISTIC=PASS", "",
        f"TRAIN_MANIFEST_SHA256={hashes['train_pool_group_ids.json']}",
        f"DEV_MANIFEST_SHA256={hashes['dev_group_ids.json']}",
        f"FINAL_MANIFEST_SHA256={hashes['final_group_ids.json']}",
        f"PROBE_MANIFEST_SHA256={hashes['probe20_group_ids.json']}", "",
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
    raw_groups, source_meta = load_groups(SOURCE)
    if source_meta["sha256"] != EXPECTED_SOURCE_SHA or len(raw_groups) != EXPECTED_GROUPS:
        raise AuditError("SOURCE_CONTRACT_FAIL")
    groups = enrich_groups(raw_groups)
    split, allocations = build_split(groups)
    probe, fallbacks = select_probe(split["dev"])
    gates = validate_split(groups, split, probe)
    payloads = manifest_payloads(split, probe, allocations, source_meta)
    split_again, allocations_again = build_split(groups)
    probe_again, fallbacks_again = select_probe(split_again["dev"])
    payloads_again = manifest_payloads(split_again, probe_again, allocations_again, source_meta)
    hashes = {name: value_sha(value) for name, value in payloads.items()}
    hashes_again = {name: value_sha(value) for name, value in payloads_again.items()}
    if hashes != hashes_again or fallbacks != fallbacks_again:
        raise AuditError("SPLIT_DETERMINISM_FAIL")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        write_json(SPLIT_DIR / name, value)
        if file_sha(SPLIT_DIR / name) != hashes[name]:
            raise AuditError(f"WRITTEN_MANIFEST_SHA_FAIL={name}")
    summaries = {"full": summarize_split(groups)}
    summaries.update({name: summarize_split(split[name]) for name in ("train", "dev", "final")})
    split_audit = {
        "gates": {**gates, "split_deterministic": "PASS"},
        "summaries": summaries,
        "full_vs_split_drift": {
            name: drift(summaries["full"], summaries[name]) for name in ("train", "dev", "final")
        },
        "stratum_allocations": allocations,
        "source": source_meta,
    }
    probe_audit = {
        "groups": len(probe),
        "subset_of_dev": "PASS" if id_set(probe) <= id_set(split["dev"]) else "FAIL",
        "domain_counts": dict(Counter(row["target_domain"] for row in probe)),
        "novelty_counts": dict(Counter(row["novelty"] for row in probe)),
        "K_bucket_counts": dict(Counter(row["K_bucket"] for row in probe)),
        "target": {"domain_each": 5, "novelty": PROBE_TARGET},
        "fallbacks": fallbacks,
        "selection_contract": "within domain x novelty: lowest K, then sha256(seed + group_id)",
        "entries": payloads["probe20_group_ids.json"],
    }
    split_sha = {"files": hashes, "rerun_files": hashes_again, "deterministic": "PASS"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "split_audit.json", split_audit)
    write_json(OUTPUT / "probe20_audit.json", probe_audit)
    write_json(OUTPUT / "split_sha256.json", split_sha)
    review = render_review(audit, gates, probe_audit, hashes)
    (OUTPUT / "CHATGPT_PHASE0_4_REVIEW.txt").write_text(review, encoding="utf-8")
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
