"""TrueRec-GRPO Phase 0.6 deterministic signal-aware Pilot4096 builder."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable


SOURCE_REPO = Path("/data/tmp_inspect/truerec_phase02_worktree")
RUNTIME = Path("/data/GRPO")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/truerec_grpo/data/build_pilot4096.py"
)
RUNTIME_SCRIPT = RUNTIME / "truerec_grpo/data/build_pilot4096.py"
FIXED_RECORD_DIR = RUNTIME / "truerec_grpo/data/fixed_domain_abc"
SPLIT_DIR = RUNTIME / "truerec_grpo/data/splits"
PILOT_DIR = RUNTIME / "truerec_grpo/data/pilot4096"
OUTPUT = RUNTIME / "truerec_grpo/results/phase0_6"
OLD_GRPO = RUNTIME / "data/rec_mp_grpo_v2/train.jsonl"
TRAIN_SOURCE = FIXED_RECORD_DIR / "train_records.jsonl"
EXPECTED_TRAIN_SHA = "97edc2d3c600dbd073f6c694a3d497c5dba10e42ce0350628b63eddbca79ca7e"
EXPECTED_OLD_SHA = "791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc"
EXPECTED_SPLIT_SHA = {
    "dev_group_ids.json": "70341b476c57e4d7dc961ccf423ab3991a42375af1989bad040be4134e8f4fe4",
    "final_group_ids.json": "e6be46befa8ed320fb0762e6df23f553c6646297bdf5649003e332c04b2e61d0",
    "probe20_group_ids.json": "7193f540ee229ef53b7ca09e541f065396fe212a8cfd43893cceff499d83abdf",
}
TRAIN_GROUPS = 17971
PILOT_GROUPS = 4096
DOMAIN_QUOTA = 1024
PILOT_SEED = "20260825"
DOMAINS = ("video", "prod", "ad", "living")
NOVELTY_CLASSES = ("H", "N2", "N1", "N0")
K_BUCKETS = ("K=1", "K=2", "K=3-5", "K=6-10", "K=11+")


class PilotError(RuntimeError):
    pass


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


def write_json(path: Path, value: Any) -> str:
    raw = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(raw)
            digest.update(raw)
            count += 1
    return digest.hexdigest(), count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args], text=True, encoding="utf-8"
    ).strip()


def code_audit() -> dict[str, str]:
    head, origin, status = git("rev-parse", "HEAD"), git("rev-parse", "origin/main"), git("status", "--short")
    blob = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{head}:{RELATIVE_SCRIPT.as_posix()}"]
    )
    github_sha = hashlib.sha256(blob).hexdigest()
    source_sha, runtime_sha = file_sha(SOURCE_REPO / RELATIVE_SCRIPT), file_sha(RUNTIME_SCRIPT)
    if head != origin or status or github_sha != source_sha or source_sha != runtime_sha:
        raise PilotError(
            f"CODE_AUDIT_GATE_FAIL head={head} origin={origin} status={status!r} "
            f"sha={github_sha},{source_sha},{runtime_sha}"
        )
    return {
        "implement_commit": head, "push_status": "PASS", "git_status_short": "EMPTY",
        "github_runtime_parity": "PASS", "script_sha256": github_sha,
    }


def k_weight(k: int) -> float:
    if k < 1:
        raise PilotError(f"INVALID_K={k}")
    return min(2.0, math.sqrt(k))


def deterministic_uniform(group_id: str, seed: str = PILOT_SEED) -> float:
    digest = hashlib.sha256((seed + "|" + group_id).encode("utf-8")).digest()
    value_52bit = int.from_bytes(digest[:7], "big") >> 4
    uniform = (value_52bit + 0.5) / (2**52)
    if not 0.0 < uniform < 1.0:
        raise PilotError(f"UNIFORM_OPEN_INTERVAL_FAIL={uniform}")
    return uniform


def weighted_priority(record: dict[str, Any], seed: str = PILOT_SEED) -> float:
    return -math.log(deterministic_uniform(record["recommendation_group_id"], seed)) / k_weight(int(record["K"]))


def largest_remainder(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if not counts or target < 0 or target > total:
        raise PilotError(f"INVALID_ALLOCATION={target}/{total}")
    exact = {key: counts[key] * target / total for key in counts}
    allocation = {key: int(exact[key]) for key in counts}
    remaining = target - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != target:
        raise PilotError("LARGEST_REMAINDER_EXACT_SIZE_FAIL")
    return allocation


def validate_train_records(records: list[dict[str, Any]]) -> None:
    required = {
        "recommendation_group_id", "split", "target_domain", "novelty", "K", "K_bucket",
        "system", "user_content_nothink", "fixed_domain_token", "all_gold_sids",
        "all_gold_abc", "history_sids",
    }
    ids = set()
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise PilotError(f"TRAIN_SCHEMA_MISSING row={index} fields={sorted(missing)}")
        if record["split"] != "train_pool":
            raise PilotError(f"NON_TRAIN_RECORD_IN_SOURCE={record['recommendation_group_id']}")
        if record["target_domain"] not in DOMAINS or record["novelty"] not in NOVELTY_CLASSES:
            raise PilotError(f"INVALID_DOMAIN_OR_NOVELTY={record['recommendation_group_id']}")
        group_id = record["recommendation_group_id"]
        if group_id in ids:
            raise PilotError(f"DUPLICATE_TRAIN_GROUP={group_id}")
        ids.add(group_id)


def select_pilot(
    train_records: list[dict[str, Any]],
    domain_quota: int = DOMAIN_QUOTA,
    seed: str = PILOT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select solely from Train; heldout IDs are intentionally not an argument."""
    quotas: dict[str, dict[str, int]] = {}
    selected = []
    for domain in DOMAINS:
        domain_rows = [row for row in train_records if row["target_domain"] == domain]
        counts = Counter(row["novelty"] for row in domain_rows)
        if any(counts[name] == 0 for name in NOVELTY_CLASSES):
            raise PilotError(f"EMPTY_TRAIN_NOVELTY_CELL={domain}:{counts}")
        allocation = largest_remainder(
            {name: counts[name] for name in NOVELTY_CLASSES}, domain_quota
        )
        quotas[domain] = {
            name: allocation[name] for name in NOVELTY_CLASSES
        }
        for novelty in NOVELTY_CLASSES:
            candidates = [
                row for row in domain_rows if row["novelty"] == novelty
            ]
            candidates.sort(
                key=lambda row: (
                    weighted_priority(row, seed), row["recommendation_group_id"]
                )
            )
            selected.extend(candidates[: allocation[novelty]])
    selected.sort(key=lambda row: row["recommendation_group_id"])
    quota_audit = {
        "policy": "PRESERVE_WITHIN_DOMAIN_TRAIN_DISTRIBUTION",
        "domain_quota": domain_quota,
        "train_cell_counts": {
            domain: {
                novelty: sum(
                    row["target_domain"] == domain and row["novelty"] == novelty
                    for row in train_records
                )
                for novelty in NOVELTY_CLASSES
            }
            for domain in DOMAINS
        },
        "pilot_cell_quota": quotas,
        "K_weighting_scope": "within fixed target_domain x novelty cells only",
    }
    return selected, quota_audit


def record_payload_sha(record: dict[str, Any]) -> str:
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def id_set(records: Iterable[dict[str, Any]]) -> set[str]:
    return {record["recommendation_group_id"] for record in records}


def safety_audit(
    train: list[dict[str, Any]], pilot: list[dict[str, Any]],
    dev_ids: set[str], final_ids: set[str], probe_ids: set[str],
    quota_audit: dict[str, Any], expected_pilot: int = PILOT_GROUPS,
    expected_domain: int = DOMAIN_QUOTA,
) -> dict[str, Any]:
    train_ids, pilot_ids = id_set(train), id_set(pilot)
    domain_counts = Counter(row["target_domain"] for row in pilot)
    novelty_counts = Counter(row["novelty"] for row in pilot)
    actual_cells = {
        domain: {
            novelty: sum(
                row["target_domain"] == domain and row["novelty"] == novelty
                for row in pilot
            )
            for novelty in NOVELTY_CLASSES
        }
        for domain in DOMAINS
    }
    train_by_id = {row["recommendation_group_id"]: row for row in train}
    mutation_count = sum(
        record_payload_sha(row) != record_payload_sha(train_by_id[row["recommendation_group_id"]])
        for row in pilot
    )
    result = {
        "train_pool_groups": len(train_ids), "pilot_groups": len(pilot),
        "pilot_unique_groups": len(pilot_ids),
        "domain_counts": {domain: domain_counts[domain] for domain in DOMAINS},
        "novelty_counts": {name: novelty_counts[name] for name in NOVELTY_CLASSES},
        "pilot_subset_of_train": "PASS" if pilot_ids <= train_ids else "FAIL",
        "pilot_dev_overlap": len(pilot_ids & dev_ids),
        "pilot_final_overlap": len(pilot_ids & final_ids),
        "pilot_probe_overlap": len(pilot_ids & probe_ids),
        "domain_quota_exact": "PASS"
        if all(domain_counts[domain] == expected_domain for domain in DOMAINS) else "FAIL",
        "novelty_quota_exact": "PASS"
        if actual_cells == quota_audit["pilot_cell_quota"] else "FAIL",
        "K1_present_in_pilot": "PASS" if any(row["K"] == 1 for row in pilot) else "FAIL",
        "N0_present_in_every_domain": "PASS"
        if all(actual_cells[domain]["N0"] > 0 for domain in DOMAINS) else "FAIL",
        "pilot_record_mutation_count": mutation_count,
    }
    if len(pilot) != expected_pilot or len(pilot_ids) != expected_pilot:
        raise PilotError(f"PILOT_SIZE_OR_UNIQUENESS_FAIL={len(pilot)},{len(pilot_ids)}")
    hard = {
        "pilot_subset_of_train": "PASS", "pilot_dev_overlap": 0,
        "pilot_final_overlap": 0, "pilot_probe_overlap": 0,
        "domain_quota_exact": "PASS", "novelty_quota_exact": "PASS",
        "K1_present_in_pilot": "PASS", "N0_present_in_every_domain": "PASS",
        "pilot_record_mutation_count": 0,
    }
    if any(result[key] != expected for key, expected in hard.items()):
        raise PilotError(f"PILOT_SAFETY_GATE_FAIL={result}")
    return result


def categorical(
    records: list[dict[str, Any]], key_fn, categories: list[str]
) -> dict[str, Any]:
    counts = Counter(key_fn(row) for row in records)
    n = len(records)
    return {
        category: {"count": counts[category], "rate": round(counts[category] / n, 8)}
        for category in categories
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    k_values = [int(row["K"]) for row in records]
    domain_novelty = [f"{domain}|{novelty}" for domain in DOMAINS for novelty in NOVELTY_CLASSES]
    summary = {
        "N": len(records),
        "domain": categorical(records, lambda row: row["target_domain"], list(DOMAINS)),
        "novelty": categorical(records, lambda row: row["novelty"], list(NOVELTY_CLASSES)),
        "domain_novelty": categorical(
            records, lambda row: f"{row['target_domain']}|{row['novelty']}", domain_novelty
        ),
        "K_mean": round(statistics.fmean(k_values), 6),
        "K_median": statistics.median(k_values),
        "K_bucket": categorical(records, lambda row: row["K_bucket"], list(K_BUCKETS)),
        "domain_K": {},
    }
    for domain in DOMAINS:
        rows = [row for row in records if row["target_domain"] == domain]
        values = [int(row["K"]) for row in rows]
        summary["domain_K"][domain] = {
            "N": len(rows), "K_mean": round(statistics.fmean(values), 6),
            "K_median": statistics.median(values),
            "K_bucket": categorical(rows, lambda row: row["K_bucket"], list(K_BUCKETS)),
        }
    return summary


def distribution_drift(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "K_mean_difference": round(candidate["K_mean"] - reference["K_mean"], 6),
        "K_median_difference": candidate["K_median"] - reference["K_median"],
    }
    for field in ("domain", "novelty", "domain_novelty", "K_bucket"):
        result[f"{field}_percentage_point"] = {
            key: round(100 * (candidate[field][key]["rate"] - reference[field][key]["rate"]), 6)
            for key in reference[field]
        }
    return result


def load_heldout_ids() -> tuple[set[str], set[str], set[str]]:
    for name, expected in EXPECTED_SPLIT_SHA.items():
        actual = file_sha(SPLIT_DIR / name)
        if actual != expected:
            raise PilotError(f"FROZEN_SPLIT_SHA_FAIL={name}:{actual}")
    dev = set(json.loads((SPLIT_DIR / "dev_group_ids.json").read_text(encoding="utf-8")))
    final = set(json.loads((SPLIT_DIR / "final_group_ids.json").read_text(encoding="utf-8")))
    probe_payload = json.loads((SPLIT_DIR / "probe20_group_ids.json").read_text(encoding="utf-8"))
    probe = {row["recommendation_group_id"] for row in probe_payload}
    return dev, final, probe


def load_old_reference(train_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load heldout records only after selection, solely to annotate old-GRPO reference IDs."""
    full = {row["recommendation_group_id"]: row for row in train_records}
    for name in ("dev_records.jsonl", "final_records.jsonl"):
        for row in read_jsonl(FIXED_RECORD_DIR / name):
            full[row["recommendation_group_id"]] = row
    old_ids = set()
    for row in read_jsonl(OLD_GRPO):
        old_ids.add(str(row["recommendation_group_id"]))
    missing = sorted(old_ids - set(full))
    if len(old_ids) != 1549 or missing:
        raise PilotError(f"OLD_REFERENCE_GROUP_CONTRACT_FAIL={len(old_ids)},{missing[:10]}")
    return [full[group_id] for group_id in sorted(old_ids)]


def make_manifest(
    group_ids_sha: str, records_sha: str, quota_audit: dict[str, Any],
    safety: dict[str, Any], distribution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": "truerec_grpo_signal_aware_pilot4096_v1",
        "seed": PILOT_SEED,
        "source": {"path": str(TRAIN_SOURCE), "sha256": EXPECTED_TRAIN_SHA, "groups": TRAIN_GROUPS},
        "groups": PILOT_GROUPS,
        "domain_policy": "exact 1024 per domain",
        "novelty_policy": "PRESERVE_WITHIN_DOMAIN_TRAIN_DISTRIBUTION",
        "K_weight": "min(2.0, sqrt(K))",
        "weighted_priority": "-log(SHA256-derived open-interval uniform) / K_weight",
        "K_weighting_constraint": "applied only within fixed Domain x Novelty cells; cannot change novelty composition",
        "performance_claim": "none; this manifest records a sampling contract only",
        "quota": quota_audit,
        "safety": safety,
        "pilot_distribution": distribution,
        "pilot_group_ids_sha256": group_ids_sha,
        "pilot_records_sha256": records_sha,
    }


def render_review(
    audit: dict[str, str], safety: dict[str, Any], pilot_summary: dict[str, Any],
    manifest_sha: str, records_sha: str, rerun_sha: str,
) -> str:
    domain, novelty, buckets = safety["domain_counts"], safety["novelty_counts"], pilot_summary["K_bucket"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT", "",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short']}",
        f"GITHUB_RUNTIME_PARITY={audit['github_runtime_parity']}", "",
        f"TRAIN_POOL_GROUPS={safety['train_pool_groups']}", f"PILOT_GROUPS={safety['pilot_groups']}",
        f"PILOT_UNIQUE_GROUPS={safety['pilot_unique_groups']}", "",
        f"PILOT_VIDEO={domain['video']}", f"PILOT_PROD={domain['prod']}",
        f"PILOT_AD={domain['ad']}", f"PILOT_LIVING={domain['living']}", "",
        f"PILOT_H={novelty['H']}", f"PILOT_N2={novelty['N2']}",
        f"PILOT_N1={novelty['N1']}", f"PILOT_N0={novelty['N0']}", "",
        f"PILOT_K_MEAN={pilot_summary['K_mean']}", f"PILOT_K_MEDIAN={pilot_summary['K_median']}", "",
        f"PILOT_K1_RATE={buckets['K=1']['rate']}", f"PILOT_K2_RATE={buckets['K=2']['rate']}",
        f"PILOT_K3_5_RATE={buckets['K=3-5']['rate']}",
        f"PILOT_K6_10_RATE={buckets['K=6-10']['rate']}",
        f"PILOT_K11PLUS_RATE={buckets['K=11+']['rate']}", "",
        f"PILOT_SUBSET_OF_TRAIN={safety['pilot_subset_of_train']}", "",
        f"PILOT_DEV_OVERLAP={safety['pilot_dev_overlap']}",
        f"PILOT_FINAL_OVERLAP={safety['pilot_final_overlap']}",
        f"PILOT_PROBE_OVERLAP={safety['pilot_probe_overlap']}", "",
        f"DOMAIN_QUOTA_EXACT={safety['domain_quota_exact']}",
        f"NOVELTY_QUOTA_EXACT={safety['novelty_quota_exact']}",
        f"K1_PRESENT_IN_PILOT={safety['K1_present_in_pilot']}",
        f"N0_PRESENT_IN_EVERY_DOMAIN={safety['N0_present_in_every_domain']}", "",
        f"PILOT_RECORD_MUTATION_COUNT={safety['pilot_record_mutation_count']}", "",
        f"PILOT_MANIFEST_SHA256={manifest_sha}", f"PILOT_RECORDS_SHA256={records_sha}",
        f"RERUN_MANIFEST_SHA256={rerun_sha}", "SPLIT_DETERMINISTIC=PASS", "",
        "TEST_STATUS=PASS", "", "GPU_INFERENCE_STARTED=NO", "MODEL_FORWARD_STARTED=NO",
        "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "PROBE_STARTED=NO",
        "NEXT_EXPERIMENT_STARTED=NO",
    ]
    return "\n".join(lines) + "\n"


def run(test_status: str) -> None:
    if "torch" in sys.modules:
        raise PilotError("TORCH_ALREADY_IMPORTED")
    if test_status != "PASS":
        raise PilotError("TEST_STATUS_GATE_FAIL")
    audit = code_audit()
    if file_sha(TRAIN_SOURCE) != EXPECTED_TRAIN_SHA:
        raise PilotError("TRAIN_SOURCE_SHA_FAIL")
    train = read_jsonl(TRAIN_SOURCE)
    validate_train_records(train)
    if len(train) != TRAIN_GROUPS:
        raise PilotError(f"TRAIN_GROUP_COUNT_FAIL={len(train)}")
    pilot, quota_audit = select_pilot(train)
    pilot_rerun, quota_rerun = select_pilot(list(reversed(train)))
    if [row["recommendation_group_id"] for row in pilot] != [row["recommendation_group_id"] for row in pilot_rerun] or quota_audit != quota_rerun:
        raise PilotError("IN_PROCESS_DETERMINISM_FAIL")
    dev_ids, final_ids, probe_ids = load_heldout_ids()
    safety = safety_audit(train, pilot, dev_ids, final_ids, probe_ids, quota_audit)
    pilot_summary = summarize(pilot)
    k1_rate = pilot_summary["K_bucket"]["K=1"]["rate"]
    k11_rate = pilot_summary["K_bucket"]["K=11+"]["rate"]
    if k1_rate < 0.40 or k11_rate > 0.10:
        raise PilotError(f"OLD_GRPO_COLLAPSE_GATE_FAIL K1={k1_rate} K11={k11_rate}")
    train_summary = summarize(train)
    if file_sha(OLD_GRPO) != EXPECTED_OLD_SHA:
        raise PilotError("OLD_GRPO_SHA_FAIL")
    old = load_old_reference(train)
    old_summary = summarize(old)
    group_ids = [row["recommendation_group_id"] for row in pilot]
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    group_ids_sha = write_json(PILOT_DIR / "pilot4096_group_ids.json", group_ids)
    records_sha, records_count = write_jsonl(PILOT_DIR / "pilot4096_records.jsonl", pilot)
    if records_count != PILOT_GROUPS:
        raise PilotError(f"PILOT_RECORD_WRITE_COUNT_FAIL={records_count}")
    manifest = make_manifest(group_ids_sha, records_sha, quota_audit, safety, pilot_summary)
    manifest_sha = write_json(PILOT_DIR / "pilot4096_manifest.json", manifest)
    rerun_group_sha = value_sha([row["recommendation_group_id"] for row in pilot_rerun])
    rerun_records_raw = hashlib.sha256()
    for row in pilot_rerun:
        rerun_records_raw.update((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    rerun_manifest = make_manifest(
        rerun_group_sha, rerun_records_raw.hexdigest(), quota_rerun, safety, summarize(pilot_rerun)
    )
    rerun_manifest_sha = value_sha(rerun_manifest)
    if rerun_manifest_sha != manifest_sha:
        raise PilotError(f"MANIFEST_DETERMINISM_FAIL={manifest_sha},{rerun_manifest_sha}")
    comparison = {
        "train_pool": train_summary, "pilot4096": pilot_summary, "old_grpo_1549": old_summary,
        "pilot_minus_train": distribution_drift(train_summary, pilot_summary),
        "old_minus_train": distribution_drift(train_summary, old_summary),
        "old_reference_role": "distribution reference only; never used for Pilot selection",
        "heldout_read_order": "Dev/Final metadata and IDs loaded only after Pilot selection",
    }
    sampling_audit = {
        **safety,
        "seed": PILOT_SEED,
        "uniform_contract": "first 52 SHA256 bits mapped to strict 0<u<1",
        "weight_contract": "min(2.0, sqrt(K)) within fixed Domain x Novelty cell",
        "K1_rate_gate": {"minimum": 0.40, "actual": k1_rate, "pass": True},
        "K11plus_rate_gate": {"maximum": 0.10, "actual": k11_rate, "pass": True},
    }
    dataset_sha = {
        "train_source": EXPECTED_TRAIN_SHA,
        "pilot4096_group_ids.json": group_ids_sha,
        "pilot4096_records.jsonl": records_sha,
        "pilot4096_manifest.json": manifest_sha,
        "rerun_manifest_sha256": rerun_manifest_sha,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "pilot_sampling_audit.json", sampling_audit)
    write_json(OUTPUT / "pilot_domain_novelty_quota.json", quota_audit)
    write_json(OUTPUT / "pilot_k_distribution.json", pilot_summary)
    write_json(OUTPUT / "pilot_vs_train_vs_old.json", comparison)
    write_json(OUTPUT / "dataset_sha256.json", dataset_sha)
    review = render_review(audit, safety, pilot_summary, manifest_sha, records_sha, rerun_manifest_sha)
    (OUTPUT / "CHATGPT_PHASE0_6_REVIEW.txt").write_text(review, encoding="utf-8")
    print(review, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-status", required=True, choices=("PASS", "FAIL"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise PilotError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")
    run(args.test_status)


if __name__ == "__main__":
    main()
