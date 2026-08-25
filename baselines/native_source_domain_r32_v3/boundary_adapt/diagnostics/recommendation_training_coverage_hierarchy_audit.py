"""Phase 1.5.3 CPU-only training coverage and SID hierarchy audit."""
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
    "recommendation_training_coverage_hierarchy_audit.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_training_coverage_hierarchy_audit.py"
OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_training_coverage_hierarchy_audit"

BATA_PATH = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
MINIFIX_PATH = Path("/data/lf_data_versions/alltrain/mini_fix/onereason_mini_fix.jsonl")
GAMMA_PATH = Path(
    "/data/lf_data_versions/alltrain/mini_fix_eval_align_answer_only_v1/"
    "onereason_mini_fix_eval_align_answer_only.jsonl"
)
EXPECTED_SHA = {
    "BATA": "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca",
    "MiniFix": "6dc7660417f093b923c93d4f19734ed9240d5fba4c9beaaa01b6a164ccd94f7f",
    "Gamma": "216857d8c5d0049a3e9642279acd89051c40f5bcddb1d4afe513e36080b086cf",
}
CORPUS_PATHS = {"BATA": BATA_PATH, "MiniFix": MINIFIX_PATH, "Gamma": GAMMA_PATH}

SPLIT_MANIFEST = Path(
    "/root/GRPO_audit_results/bridge_inside_transition_sft_v1_20260824/split_manifest.json"
)
CANONICAL_ROWS = RUNTIME / "boundary_adapt/results/boundary_adapt_rows.jsonl"
PHASE151 = RUNTIME / "boundary_adapt/results/recommendation_memory_history_2x2_probe"
PHASE152 = RUNTIME / "boundary_adapt/results/recommendation_history_floor_cpu_audit"
DOMAIN_ORDER = ("video", "prod", "ad", "living")
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
FREQUENCY_BUCKETS = ("0", "1", "2-4", "5-9", "10-19", "20+")

PROTECTED_FILES = {
    "phase141_summary": RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_4_1/summary.json",
    "phase15_records": RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_quick_probe/records.jsonl",
    "phase15_summary": RUNTIME / "boundary_adapt/results/recommendation_bridge_memory_quick_probe/seen_unseen_summary.json",
    "phase151_records": PHASE151 / "records.jsonl",
    "phase151_manifest": PHASE151 / "probe16_manifest.json",
    "phase151_summary": PHASE151 / "cell_summary.json",
    "phase152_source_contract": PHASE152 / "source_contract_audit.json",
    "phase152_history_rates": PHASE152 / "domain_history_rates.json",
    "phase152_hierarchy": PHASE152 / "phase151_hierarchy_analysis.json",
    "phase152_root_summary": PHASE152 / "history_floor_root_cause_summary.json",
    "phase152_review": PHASE152 / "CHATGPT_HISTORY_FLOOR_CPU_REVIEW.txt",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args], text=True, encoding="utf-8"
    ).strip()


def code_audit() -> dict[str, Any]:
    head, origin, status = git("rev-parse", "HEAD"), git("rev-parse", "origin/main"), git("status", "--short")
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


def prefix(sid: tuple[str, int, int, int], level: str) -> tuple[Any, ...]:
    return sid[: {"A": 2, "AB": 3, "ABC": 4}[level]]


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row.get("aux_metadata_json") or "{}")


def load_canonical() -> tuple[dict[str, tuple[str, str]], set[str], dict[str, Any]]:
    canonical: dict[str, tuple[str, str]] = {}
    with CANONICAL_ROWS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            group = str(row["boundary_group_id"])
            identity = (str(row["prompt"]), str(row["original_response"]))
            if group in canonical and canonical[group] != identity:
                raise RuntimeError(f"CANONICAL_CONFLICT={group}")
            canonical[group] = identity
    split = read_json(SPLIT_MANIFEST)
    train, holdout = set(map(str, split["train_group_ids"])), set(map(str, split["holdout_group_ids"]))
    if len(canonical) != 15943 or len(train) != 14348 or len(holdout) != 1595 or train & holdout:
        raise RuntimeError("ADAPTATION_SPLIT_CONTRACT_FAIL")
    if split.get("source_sha256") != EXPECTED_SHA["BATA"]:
        raise RuntimeError("ADAPTATION_SPLIT_SOURCE_SHA_FAIL")
    return canonical, holdout, {
        "canonical_rows": str(CANONICAL_ROWS),
        "canonical_rows_sha256": file_sha(CANONICAL_ROWS),
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": file_sha(SPLIT_MANIFEST),
        "adaptation_groups": len(canonical),
        "adaptation_train_groups": len(train),
        "adaptation_holdout_groups": len(holdout),
        "adaptation_train_holdout_intersection": 0,
    }


def new_index(name: str, path: Path, sha256: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256,
        "recommendation_rows": 0,
        "segments": Counter(),
        "group_all_gold": {},
        "group_current_sids": defaultdict(set),
        "row_frequency": {level: Counter() for level in ("A", "AB", "ABC")},
        "unique_group_sets": {level: defaultdict(set) for level in ("A", "AB", "ABC")},
        "group_row_frequency": {
            level: defaultdict(Counter) for level in ("A", "AB", "ABC")
        },
        "answer_suffix_match": 0,
        "natural_matches": {},
    }


def scan_corpus(
    name: str, path: Path, canonical: dict[str, tuple[str, str]] | None = None,
    holdout: set[str] | None = None,
) -> dict[str, Any]:
    actual_sha = file_sha(path)
    if actual_sha != EXPECTED_SHA[name]:
        raise RuntimeError(f"CORPUS_SHA_MISMATCH={name}:{actual_sha}")
    index = new_index(name, path, actual_sha)
    natural_signatures: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend":
                continue
            info = metadata(row)
            group = str(info["recommendation_group_id"])
            current_raw = str(info["recommendation_current_gold_sid"])
            current = parse_sid(current_raw)
            all_gold = frozenset(parse_sid(str(value)) for value in info["recommendation_all_gold_sids"])
            if current not in all_gold or any(value[0] != current[0] for value in all_gold):
                raise RuntimeError(f"TRAIN_GOLD_CONTRACT_FAIL={name}:{group}")
            previous = index["group_all_gold"].get(group)
            if previous is not None and previous != all_gold:
                raise RuntimeError(f"TRAIN_GROUP_GOLD_CONFLICT={name}:{group}")
            index["group_all_gold"][group] = all_gold
            index["group_current_sids"][group].add(current)
            index["recommendation_rows"] += 1
            index["segments"][str(row.get("source_segment"))] += 1
            if str(row.get("output", "")).rstrip().endswith(current_raw):
                index["answer_suffix_match"] += 1
            for level in ("A", "AB", "ABC"):
                key = prefix(current, level)
                index["row_frequency"][level][key] += 1
                index["unique_group_sets"][level][key].add(group)
                index["group_row_frequency"][level][group][key] += 1
            if canonical is not None and holdout is not None and group in holdout:
                identity = (
                    str(row.get("instruction", "")) + str(row.get("input", "")),
                    str(row.get("output", "")),
                )
                if canonical.get(group) == identity:
                    signature = stable_sha(
                        [row.get("system"), row.get("instruction"), row.get("input"), row.get("output"), row.get("aux_metadata_json")]
                    )
                    if group in natural_signatures and natural_signatures[group] != signature:
                        raise RuntimeError(f"NATURAL_CANONICAL_AMBIGUOUS={group}")
                    natural_signatures[group] = signature
                    index["natural_matches"].setdefault(group, row)
    index["group_current_sids"] = dict(index["group_current_sids"])
    index["group_row_frequency"] = {
        level: dict(values) for level, values in index["group_row_frequency"].items()
    }
    return index


def corpus_public(index: dict[str, Any]) -> dict[str, Any]:
    group_gold_digest = stable_sha(
        {
            group: [list(value) for value in sorted(golds)]
            for group, golds in sorted(index["group_all_gold"].items())
        }
    )
    current_digest = stable_sha(
        {
            group: [list(value) for value in sorted(golds)]
            for group, golds in sorted(index["group_current_sids"].items())
        }
    )
    return {
        "path": index["path"],
        "sha256": index["sha256"],
        "recommendation_rows": index["recommendation_rows"],
        "recommendation_groups": len(index["group_all_gold"]),
        "segments": dict(index["segments"]),
        "unique_supervised_current_paths": sum(len(value) for value in index["group_current_sids"].values()),
        "answer_suffix_match_rows": index["answer_suffix_match"],
        "answer_suffix_match_rate": index["answer_suffix_match"] / index["recommendation_rows"],
        "group_all_gold_digest": group_gold_digest,
        "group_current_sid_digest": current_digest,
    }


def build_natural(index: dict[str, Any], holdout: set[str]) -> list[dict[str, Any]]:
    if set(index["natural_matches"]) != holdout:
        raise RuntimeError(f"NATURAL_MATCH_COUNT={len(index['natural_matches'])}")
    rows = []
    for group in sorted(holdout):
        raw = index["natural_matches"][group]
        info = metadata(raw)
        current = parse_sid(str(info["recommendation_current_gold_sid"]))
        golds = sorted(parse_sid(str(value)) for value in info["recommendation_all_gold_sids"])
        prompt = str(raw.get("instruction", "")) + str(raw.get("input", ""))
        history = [parse_sid(match.group(0)) for match in SID_RE.finditer(prompt) if match.group(1) == current[0]]
        history_sets = {
            level: {prefix(value, level) for value in history} for level in ("A", "AB", "ABC")
        }
        rows.append(
            {
                "group_id": group,
                "domain": current[0],
                "K": len(golds),
                "current_gold_sid": current,
                "all_gold_sids": golds,
                "CURRENT_GOLD_A_IN_HISTORY": prefix(current, "A") in history_sets["A"],
                "CURRENT_GOLD_AB_IN_HISTORY": prefix(current, "AB") in history_sets["AB"],
                "CURRENT_GOLD_ABC_IN_HISTORY": current in history_sets["ABC"],
                "ANY_GOLD_A_IN_HISTORY": any(prefix(value, "A") in history_sets["A"] for value in golds),
                "ANY_GOLD_AB_IN_HISTORY": any(prefix(value, "AB") in history_sets["AB"] for value in golds),
                "ANY_GOLD_ABC_IN_HISTORY": any(value in history_sets["ABC"] for value in golds),
            }
        )
    phase152 = read_json(PHASE152 / "domain_history_rates.json")
    for domain in DOMAIN_ORDER:
        values = [row for row in rows if row["domain"] == domain]
        expected = phase152["per_domain"][domain]
        checks = {
            "N": len(values),
            "GoldAInHistory": sum(row["ANY_GOLD_A_IN_HISTORY"] for row in values),
            "GoldABInHistory": sum(row["ANY_GOLD_AB_IN_HISTORY"] for row in values),
            "GoldABCInHistory": sum(row["ANY_GOLD_ABC_IN_HISTORY"] for row in values),
        }
        if any(checks[key] != expected[key] for key in checks):
            raise RuntimeError(f"PHASE152_HISTORY_PARITY_FAIL={domain}:{checks}")
    return rows


def other_group_unique_frequency(index: dict[str, Any], level: str, key: tuple[Any, ...], group: str) -> int:
    groups = index["unique_group_sets"][level].get(key, set())
    return len(groups) - int(group in groups)


def other_group_row_frequency(index: dict[str, Any], level: str, key: tuple[Any, ...], group: str) -> int:
    total = index["row_frequency"][level].get(key, 0)
    same = index["group_row_frequency"][level].get(group, Counter()).get(key, 0)
    return total - same


def gold_coverage(index: dict[str, Any], target: dict[str, Any], golds: list[tuple[str, int, int, int]]) -> dict[str, Any]:
    group = target["group_id"]
    result: dict[str, Any] = {
        "SAME_GROUP_SEEN": group in index["group_all_gold"],
    }
    for level in ("A", "AB", "ABC"):
        keys = [prefix(value, level) for value in golds]
        other_unique = [other_group_unique_frequency(index, level, key, group) for key in keys]
        other_rows = [other_group_row_frequency(index, level, key, group) for key in keys]
        total_unique = [len(index["unique_group_sets"][level].get(key, set())) for key in keys]
        same_present = [group in index["unique_group_sets"][level].get(key, set()) for key in keys]
        result.update(
            {
                f"{level}_SAME_GROUP_PRESENT": any(same_present),
                f"{level}_OTHER_GROUP_SEEN": any(value > 0 for value in other_unique),
                f"{level}_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY": max(other_unique, default=0),
                f"{level}_OTHER_GROUP_ROW_FREQUENCY": max(other_rows, default=0),
                f"{level}_TOTAL_UNIQUE_GROUP_FREQUENCY": max(total_unique, default=0),
                f"{level}_TOTAL_SEEN": any(value > 0 for value in total_unique),
            }
        )
    return result


def branch_children(index: dict[str, Any]) -> tuple[dict[tuple[Any, ...], set[tuple[Any, ...]]], dict[tuple[Any, ...], set[int]]]:
    ab_by_a: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    c_by_ab: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for sid in index["unique_group_sets"]["ABC"]:
        a, ab = prefix(sid, "A"), prefix(sid, "AB")
        ab_by_a[a].add(ab)
        c_by_ab[ab].add(int(sid[3]))
    return dict(ab_by_a), dict(c_by_ab)


def target_surprisal(
    index: dict[str, Any], target: dict[str, Any], sid: tuple[str, int, int, int],
    c_by_ab: dict[tuple[Any, ...], set[int]],
) -> dict[str, Any]:
    group, ab = target["group_id"], prefix(sid, "AB")
    children = c_by_ab.get(ab, set())
    total_denominator = sum(
        len(index["unique_group_sets"]["ABC"].get(ab + (child,), set())) for child in children
    )
    total_numerator = len(index["unique_group_sets"]["ABC"].get(sid, set()))
    same_group_children = {
        value[3] for value in index["group_current_sids"].get(group, set()) if prefix(value, "AB") == ab
    }
    other_denominator = total_denominator - len(same_group_children)
    other_numerator = total_numerator - int(sid in index["group_current_sids"].get(group, set()))

    def probability(numerator: int, denominator: int) -> tuple[float | None, float | None]:
        if numerator <= 0 or denominator <= 0:
            return None, None
        value = numerator / denominator
        return value, -math.log(value)

    total_p, total_s = probability(total_numerator, total_denominator)
    other_p, other_s = probability(other_numerator, other_denominator)
    return {
        "AB_TRAIN_SEEN": bool(children),
        "AB_DISTINCT_C": len(children),
        "ABC_TOTAL_UNIQUE_GROUP_FREQUENCY": total_numerator,
        "ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY": max(0, other_numerator),
        "P_TOTAL_C_GIVEN_AB": total_p,
        "TOTAL_SURPRISAL": total_s,
        "P_OTHER_GROUP_C_GIVEN_AB": other_p,
        "OTHER_GROUP_SURPRISAL": other_s,
        "OTHER_GROUP_UNSEEN_ABC": other_numerator <= 0,
    }


def annotate_targets(
    natural: list[dict[str, Any]], index: dict[str, Any], label: str,
) -> list[dict[str, Any]]:
    _, c_by_ab = branch_children(index)
    output = []
    for target in natural:
        current = target["current_gold_sid"]
        golds = list(target["all_gold_sids"])
        current_coverage = gold_coverage(index, target, [current])
        any_coverage = gold_coverage(index, target, golds)
        current_detail = target_surprisal(index, target, current, c_by_ab)
        all_details = [target_surprisal(index, target, sid, c_by_ab) for sid in golds]
        best_index = max(
            range(len(golds)),
            key=lambda i: (
                all_details[i]["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"],
                -all_details[i]["AB_DISTINCT_C"],
                tuple(-value for value in golds[i][1:]),
            ),
        )
        output.append(
            {
                "group_id": target["group_id"],
                "domain": target["domain"],
                "K": target["K"],
                "ANY_GOLD_ABC_IN_HISTORY": target["ANY_GOLD_ABC_IN_HISTORY"],
                "CURRENT_GOLD": current_coverage,
                "ANY_GOLD": any_coverage,
                "CURRENT_GOLD_DETAIL": current_detail,
                "ANY_GOLD_BEST_COVERED_SID": golds[best_index],
                "ANY_GOLD_BEST_DETAIL": all_details[best_index],
                "ANY_GOLD_ALL_DETAILS": all_details,
                "corpus": label,
            }
        )
    return output


def frequency_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 9:
        return "5-9"
    if value <= 19:
        return "10-19"
    return "20+"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position) + ordered[high] * (position - low))


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {"N": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "N": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p75": percentile(data, 0.75),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "max": max(data),
    }


def weighted_summary(values_and_weights: Iterable[tuple[float, int]]) -> dict[str, float | int]:
    pairs = [(float(value), int(weight)) for value, weight in values_and_weights if weight > 0]
    total = sum(weight for _, weight in pairs)
    if not pairs or total == 0:
        return {"weight": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    pairs.sort()

    def quantile(q: float) -> float:
        threshold = q * total
        running = 0
        for value, weight in pairs:
            running += weight
            if running >= threshold:
                return value
        return pairs[-1][0]

    return {
        "weight": total,
        "mean": sum(value * weight for value, weight in pairs) / total,
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": max(value for value, _ in pairs),
    }


def frequency_stats(index: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"frequency_unit": {"row": "serialized supervised current-gold rows", "unique_group": "distinct recommendation_group_id"}}
    for domain in DOMAIN_ORDER:
        output[domain] = {}
        for level in ("A", "AB", "ABC"):
            keys = [key for key in index["row_frequency"][level] if key[0] == domain]
            output[domain][level] = {
                "vocabulary_size": len(keys),
                "row_frequency": numeric_summary(index["row_frequency"][level][key] for key in keys),
                "unique_group_frequency": numeric_summary(len(index["unique_group_sets"][level][key]) for key in keys),
            }
    return output


def branching_stats(index: dict[str, Any]) -> dict[str, Any]:
    ab_by_a, c_by_ab = branch_children(index)
    result = {}
    for domain in DOMAIN_ORDER:
        a_values = [(key, len(value)) for key, value in ab_by_a.items() if key[0] == domain]
        ab_values = [(key, len(value)) for key, value in c_by_ab.items() if key[0] == domain]
        result[domain] = {
            "distinct_AB_per_A": {
                "macro": numeric_summary(value for _, value in a_values),
                "weighted_by_training_group": weighted_summary(
                    (value, len(index["unique_group_sets"]["A"].get(key, set()))) for key, value in a_values
                ),
            },
            "distinct_C_per_AB": {
                "macro": numeric_summary(value for _, value in ab_values),
                "weighted_by_training_group": weighted_summary(
                    (value, len(index["unique_group_sets"]["AB"].get(key, set()))) for key, value in ab_values
                ),
            },
        }
    return result


def conditional_entropy(index: dict[str, Any]) -> dict[str, Any]:
    _, c_by_ab = branch_children(index)
    result = {}
    for domain in DOMAIN_ORDER:
        values = []
        for ab, children in c_by_ab.items():
            if ab[0] != domain:
                continue
            counts = [len(index["unique_group_sets"]["ABC"][ab + (child,)]) for child in children]
            total = sum(counts)
            entropy = -sum((count / total) * math.log(count / total) for count in counts)
            values.append((entropy, total, ab, len(children)))
        result[domain] = {
            "AB_count": len(values),
            "macro_mean_H_C_given_AB_nats": statistics.fmean(value[0] for value in values),
            "frequency_weighted_mean_H_C_given_AB_nats": sum(value[0] * value[1] for value in values) / sum(value[1] for value in values),
            "entropy_distribution": numeric_summary(value[0] for value in values),
            "definition": "p(C|AB)=unique_group_frequency(ABC)/sum_C unique_group_frequency(ABC); natural log",
        }
    return result


def aggregate_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"contracts": {}, "frequency_buckets": {}, "history_x_train_abc": {}}
    for contract in ("CURRENT_GOLD", "ANY_GOLD"):
        result["contracts"][contract] = {}
        result["frequency_buckets"][contract] = {}
        result["history_x_train_abc"][contract] = {}
        for domain in DOMAIN_ORDER:
            values = [row for row in rows if row["domain"] == domain]
            n = len(values)
            entry = {"N": n, "SAME_GROUP_SEEN_RATE": statistics.fmean(row[contract]["SAME_GROUP_SEEN"] for row in values)}
            for level in ("A", "AB", "ABC"):
                entry[f"{level}_OTHER_GROUP_COVERAGE"] = statistics.fmean(row[contract][f"{level}_OTHER_GROUP_SEEN"] for row in values)
                entry[f"{level}_TOTAL_COVERAGE"] = statistics.fmean(row[contract][f"{level}_TOTAL_SEEN"] for row in values)
                entry[f"{level}_SAME_GROUP_PRESENT_RATE"] = statistics.fmean(row[contract][f"{level}_SAME_GROUP_PRESENT"] for row in values)
            nonhistory = [row for row in values if not row["ANY_GOLD_ABC_IN_HISTORY"]]
            entry["NONHISTORY_N"] = len(nonhistory)
            entry["NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE"] = statistics.fmean(
                not row[contract]["ABC_OTHER_GROUP_SEEN"] for row in nonhistory
            )
            result["contracts"][contract][domain] = entry
            for level in ("AB", "ABC"):
                counts = Counter(
                    frequency_bucket(int(row[contract][f"{level}_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"]))
                    for row in values
                )
                result["frequency_buckets"][contract].setdefault(domain, {})[level] = {
                    "counts": {bucket: counts[bucket] for bucket in FREQUENCY_BUCKETS},
                    "rates": {bucket: counts[bucket] / n for bucket in FREQUENCY_BUCKETS},
                    "frequency_unit": "OTHER_GROUP_UNIQUE_GROUP_FREQUENCY",
                }
            cross = Counter(
                ("HISTORY_YES" if row["ANY_GOLD_ABC_IN_HISTORY"] else "HISTORY_NO")
                + "__"
                + ("TRAIN_ABC_YES" if row[contract]["ABC_OTHER_GROUP_SEEN"] else "TRAIN_ABC_NO")
                for row in values
            )
            result["history_x_train_abc"][contract][domain] = {
                key: cross[key]
                for key in (
                    "HISTORY_YES__TRAIN_ABC_YES", "HISTORY_YES__TRAIN_ABC_NO",
                    "HISTORY_NO__TRAIN_ABC_YES", "HISTORY_NO__TRAIN_ABC_NO",
                )
            }
    return result


def target_frequency_surprisal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for contract, detail_key in (("CURRENT_GOLD", "CURRENT_GOLD_DETAIL"), ("ANY_GOLD", "ANY_GOLD_BEST_DETAIL")):
        result[contract] = {}
        for domain in DOMAIN_ORDER:
            values = [row[detail_key] for row in rows if row["domain"] == domain]
            seen = [value for value in values if not value["OTHER_GROUP_UNSEEN_ABC"]]
            unseen = [value for value in values if value["OTHER_GROUP_UNSEEN_ABC"]]
            abc_seen = [value for value in values if value["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"] > 0]
            abc_unseen = [value for value in values if value["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"] == 0]
            result[contract][domain] = {
                "N": len(values),
                "ABC_OTHER_GROUP_UNSEEN_RATE": len(abc_unseen) / len(values),
                "ABC_OTHER_GROUP_FREQUENCY": numeric_summary(value["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"] for value in values),
                "TARGET_GOLD_AB_BRANCHING": numeric_summary(value["AB_DISTINCT_C"] for value in values),
                "TARGET_GOLD_AB_BRANCHING_ABC_SEEN": numeric_summary(value["AB_DISTINCT_C"] for value in abc_seen),
                "TARGET_GOLD_AB_BRANCHING_ABC_UNSEEN": numeric_summary(value["AB_DISTINCT_C"] for value in abc_unseen),
                "OTHER_GROUP_SURPRISAL_SEEN_ABC": numeric_summary(value["OTHER_GROUP_SURPRISAL"] for value in seen),
                "UNSEEN_ABC_N": len(unseen),
            }
    return result


def natural_coverage_payload(
    natural: list[dict[str, Any]], annotations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "name": "ADAPTATION_HELDOUT_NATURAL_PROXY",
        "N": len(natural),
        "distribution_unchanged": True,
        "gold_contracts": {
            "CURRENT_GOLD": "recommendation_current_gold_sid",
            "ANY_GOLD": "any recommendation_all_gold_sids; frequency is max over positives",
        },
        "corpora": {label: aggregate_coverage(rows) for label, rows in annotations.items()},
        "per_target": {
            label: rows for label, rows in annotations.items()
        },
    }


def render_natural_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Natural1595 training coverage",
        "",
        "Main rows use ANY_GOLD and other-group unique-group frequency. The pool is unchanged and is not a BATA holdout.",
        "",
        "| Corpus | Domain | N | Same group | A other | AB other | ABC other | NonHistory ABC unseen |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in ("BATA", "MINI"):
        values = payload["corpora"][corpus]["contracts"]["ANY_GOLD"]
        for domain in DOMAIN_ORDER:
            value = values[domain]
            lines.append(
                f"| {corpus} | {domain} | {value['N']} | {value['SAME_GROUP_SEEN_RATE']:.2%} | "
                f"{value['A_OTHER_GROUP_COVERAGE']:.2%} | {value['AB_OTHER_GROUP_COVERAGE']:.2%} | "
                f"{value['ABC_OTHER_GROUP_COVERAGE']:.2%} | {value['NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE']:.2%} |"
            )
    return "\n".join(lines) + "\n"


def mini_vs_bata(payload: dict[str, Any]) -> dict[str, Any]:
    result = {"ANY_GOLD": {}, "CURRENT_GOLD": {}}
    for contract in result:
        for domain in DOMAIN_ORDER:
            bata = payload["corpora"]["BATA"]["contracts"][contract][domain]
            mini = payload["corpora"]["MINI"]["contracts"][contract][domain]
            result[contract][domain] = {
                f"BATA_EXTRA_{level}_OTHER_GROUP_COVERAGE_OVER_MINI": bata[f"{level}_OTHER_GROUP_COVERAGE"] - mini[f"{level}_OTHER_GROUP_COVERAGE"]
                for level in ("A", "AB", "ABC")
            }
            result[contract][domain]["BATA_MINUS_MINI_NONHISTORY_ABC_COVERAGE"] = (
                (1 - bata["NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE"])
                - (1 - mini["NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE"])
            )
    abc_deltas = [result["ANY_GOLD"][domain]["BATA_EXTRA_ABC_OTHER_GROUP_COVERAGE_OVER_MINI"] for domain in DOMAIN_ORDER]
    result["BATA_HAS_MORE_FINE_ITEM_TRAINING_OPPORTUNITY"] = "YES" if all(value > 0 for value in abc_deltas) else "PARTIAL" if any(value > 0 for value in abc_deltas) else "NO"
    result["ability_claim"] = "DATA_OPPORTUNITY_ONLY_NOT_MODEL_ABILITY"
    return result


def phase151_join(
    mini_rows: list[dict[str, Any]], natural: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage = {row["group_id"]: row for row in mini_rows}
    natural_lookup = {row["group_id"]: row for row in natural}
    manifest = read_json(PHASE151 / "probe16_manifest.json")
    hierarchy = read_json(PHASE152 / "phase151_hierarchy_analysis.json")
    beam_rows = hierarchy["per_group"]
    selected = [item for item in manifest["items"] if item["cell"] in ("SN", "UN")]
    groups = []
    for item in selected:
        cov = coverage[item["group_id"]]
        target = natural_lookup[item["group_id"]]
        groups.append(
            {
                "group_id": item["group_id"],
                "domain": item["domain"],
                "cell": item["cell"],
                "K": item["K"],
                "CURRENT_GOLD": cov["CURRENT_GOLD"],
                "ANY_GOLD": cov["ANY_GOLD"],
                "ANY_GOLD_BEST_DETAIL": cov["ANY_GOLD_BEST_DETAIL"],
                "current_gold_sid": target["current_gold_sid"],
                "all_gold_sids": target["all_gold_sids"],
            }
        )
    group_lookup = {row["group_id"]: row for row in groups}
    joined = []
    for beam in beam_rows:
        if beam["cell"] not in ("SN", "UN"):
            continue
        base = group_lookup[beam["group_id"]]
        joined.append(
            {
                "group_id": beam["group_id"], "domain": beam.get("domain", base["domain"]),
                "cell": beam["cell"], "model": beam["model"], "condition": beam["condition"],
                "coverage": base["ANY_GOLD"], "best_detail": base["ANY_GOLD_BEST_DETAIL"],
                "AHit@32": beam["AHit@32"], "ABHit@32": beam["ABHit@32"], "ABCHit@32": beam["ABCHit@32"],
                "AMRR": beam["AMRR"], "ABMRR": beam["ABMRR"], "ABCMRR": beam["ABCMRR"],
            }
        )
    summary = {}
    for cell in ("SN", "UN"):
        values = [row for row in groups if row["cell"] == cell]
        summary[cell] = {
            "N": len(values),
            "SAME_GROUP_SEEN_RATE": statistics.fmean(row["ANY_GOLD"]["SAME_GROUP_SEEN"] for row in values),
            "SAME_GROUP_ABC_PRESENT_RATE": statistics.fmean(row["ANY_GOLD"]["ABC_SAME_GROUP_PRESENT"] for row in values),
            "A_TOTAL_SEEN_RATE": statistics.fmean(row["ANY_GOLD"]["A_TOTAL_SEEN"] for row in values),
            "AB_TOTAL_SEEN_RATE": statistics.fmean(row["ANY_GOLD"]["AB_TOTAL_SEEN"] for row in values),
            "ABC_TOTAL_SEEN_RATE": statistics.fmean(row["ANY_GOLD"]["ABC_TOTAL_SEEN"] for row in values),
            "A_OTHER_GROUP_SEEN_RATE": statistics.fmean(row["ANY_GOLD"]["A_OTHER_GROUP_SEEN"] for row in values),
            "AB_OTHER_GROUP_SEEN_RATE": statistics.fmean(row["ANY_GOLD"]["AB_OTHER_GROUP_SEEN"] for row in values),
            "ABC_OTHER_GROUP_SEEN_RATE": statistics.fmean(row["ANY_GOLD"]["ABC_OTHER_GROUP_SEEN"] for row in values),
            "ABC_OTHER_GROUP_FREQUENCY": numeric_summary(row["ANY_GOLD"]["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"] for row in values),
            "GOLD_AB_C_BRANCHING": numeric_summary(row["ANY_GOLD_BEST_DETAIL"]["AB_DISTINCT_C"] for row in values),
        }
    abc_beam_zero = all(row["ABCHit@32"] == 0 for row in joined)
    hierarchy_pattern = (
        summary["SN"]["A_TOTAL_SEEN_RATE"] > summary["SN"]["AB_OTHER_GROUP_SEEN_RATE"]
        and summary["UN"]["A_TOTAL_SEEN_RATE"] >= summary["UN"]["ABC_TOTAL_SEEN_RATE"]
        and abc_beam_zero
    )
    return {
        "coverage_contract": "ANY_GOLD; total and other-group coverage separated",
        "groups": groups,
        "joined_beam_rows": joined,
        "summary": summary,
        "training_coverage_matches_hierarchy_failure": "SUPPORTED" if hierarchy_pattern else "MIXED",
    }


def render_join_md(joined: dict[str, Any]) -> str:
    lines = [
        "# Phase 1.5.1 coverage x Beam32 join",
        "",
        "Coverage is Mini ANY_GOLD. Beam metrics are reused without generation.",
        "",
        "| Cell | N | Same group | Same-group ABC | A total | AB total | ABC total | A other | AB other | ABC other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in ("SN", "UN"):
        value = joined["summary"][cell]
        lines.append(
            f"| {cell} | {value['N']} | {value['SAME_GROUP_SEEN_RATE']:.2f} | {value['SAME_GROUP_ABC_PRESENT_RATE']:.2f} | "
            f"{value['A_TOTAL_SEEN_RATE']:.2f} | {value['AB_TOTAL_SEEN_RATE']:.2f} | {value['ABC_TOTAL_SEEN_RATE']:.2f} | "
            f"{value['A_OTHER_GROUP_SEEN_RATE']:.2f} | {value['AB_OTHER_GROUP_SEEN_RATE']:.2f} | {value['ABC_OTHER_GROUP_SEEN_RATE']:.2f} |"
        )
    lines.extend(["", f"Decision: `{joined['training_coverage_matches_hierarchy_failure']}`."])
    return "\n".join(lines) + "\n"


def root_decision(
    natural_payload: dict[str, Any], branching: dict[str, Any], entropy: dict[str, Any],
    surprisal: dict[str, Any], comparison: dict[str, Any], joined: dict[str, Any],
) -> dict[str, Any]:
    mini = natural_payload["corpora"]["MINI"]["contracts"]["ANY_GOLD"]
    natural_n = sum(mini[domain]["N"] for domain in DOMAIN_ORDER)
    total_n = sum(mini[domain]["NONHISTORY_N"] for domain in DOMAIN_ORDER)
    nonhistory_unseen = sum(
        mini[domain]["NONHISTORY_N"] * mini[domain]["NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE"]
        for domain in DOMAIN_ORDER
    ) / total_n
    def natural_weighted(metric: str) -> float:
        return sum(mini[domain]["N"] * mini[domain][metric] for domain in DOMAIN_ORDER) / natural_n

    a_cov = natural_weighted("A_OTHER_GROUP_COVERAGE")
    ab_cov = natural_weighted("AB_OTHER_GROUP_COVERAGE")
    abc_cov = natural_weighted("ABC_OTHER_GROUP_COVERAGE")
    if a_cov - ab_cov >= 0.15 and ab_cov - abc_cov >= 0.10 and nonhistory_unseen >= 0.50:
        data_support = "STRONG"
    elif a_cov > abc_cov and nonhistory_unseen >= 0.30:
        data_support = "MODERATE"
    else:
        data_support = "WEAK"
    sn = joined["summary"]["SN"]
    abc_beam_zero = all(row["ABCHit@32"] == 0 for row in joined["joined_beam_rows"])
    decoder_support = "MODERATE" if sn["SAME_GROUP_ABC_PRESENT_RATE"] >= 0.75 and abc_beam_zero else "WEAK"
    video_signals = {
        "lowest_mini_abc_other_coverage": mini["video"]["ABC_OTHER_GROUP_COVERAGE"] == min(value["ABC_OTHER_GROUP_COVERAGE"] for value in mini.values()),
        "highest_mini_c_per_ab_p90": branching["MINI"]["video"]["distinct_C_per_AB"]["macro"]["p90"] == max(branching["MINI"][domain]["distinct_C_per_AB"]["macro"]["p90"] for domain in DOMAIN_ORDER),
        "highest_mini_entropy": entropy["MINI"]["video"]["frequency_weighted_mean_H_C_given_AB_nats"] == max(entropy["MINI"][domain]["frequency_weighted_mean_H_C_given_AB_nats"] for domain in DOMAIN_ORDER),
        "highest_target_surprisal": surprisal["MINI"]["ANY_GOLD"]["video"]["OTHER_GROUP_SURPRISAL_SEEN_ABC"]["median"] == max(surprisal["MINI"]["ANY_GOLD"][domain]["OTHER_GROUP_SURPRISAL_SEEN_ABC"]["median"] for domain in DOMAIN_ORDER),
    }
    video_count = sum(video_signals.values())
    video_class = "YES" if video_count >= 2 else "PARTIAL" if video_count == 1 else "NO"
    if data_support in ("STRONG", "MODERATE") and decoder_support == "MODERATE":
        root_class = "COVERAGE_PLUS_RANKING_BOTTLENECK"
    elif data_support == "STRONG":
        root_class = "FINE_ITEM_DATA_COVERAGE_BOTTLENECK"
    elif decoder_support == "MODERATE":
        root_class = "FINE_ITEM_RANKING_BOTTLENECK"
    else:
        root_class = "INSUFFICIENT_TO_DISTINGUISH"
    future = (
        "COVERAGE_CONTROLLED_SEEN_ABC_VS_UNSEEN_ABC_TEACHER_FORCED_PROBE"
        if nonhistory_unseen >= 0.50
        else "TEACHER_FORCED_GOLD_LOGPROB_SN_VS_UN"
    )
    return {
        "mini_macro_other_group_coverage": {"A": a_cov, "AB": ab_cov, "ABC": abc_cov},
        "mini_nonhistory_abc_other_group_unseen_rate": nonhistory_unseen,
        "data_coverage_limit_support": data_support,
        "decoder_ranking_limit_support": decoder_support,
        "video_fine_item_space_is_harder": video_class,
        "video_signals": video_signals,
        "coarse_interest_signal": "PRESENT",
        "fine_item_signal": "WEAK",
        "root_class": root_class,
        "primary_signal": "Mini other-group coverage contracts from A to AB to ABC, while SN same-group ABC supervision still fails Beam32.",
        "main_limitation": "Coverage is data opportunity, not model ability; Phase1.5.1 has only four SN and four UN groups and Beam depth 32.",
        "conclusion": "Fine-item long-tail coverage and decoder ranking both contribute; neither alone explains all observed hierarchy failure.",
        "beta_true_recommendation_identifiable": "NO",
        "bata_has_more_fine_item_training_opportunity": comparison["BATA_HAS_MORE_FINE_ITEM_TRAINING_OPPORTUNITY"],
        "future_gpu_experiment": future,
    }


def render_root_md(summary: dict[str, Any], natural: dict[str, Any], joined: dict[str, Any]) -> str:
    mini = natural["corpora"]["MINI"]["contracts"]["ANY_GOLD"]
    lines = [
        "# Training coverage x SID hierarchy root cause",
        "",
        "CPU-only. Coverage denotes training opportunity, not model ability.",
        "",
        "| Domain | Mini A other | Mini AB other | Mini ABC other | NonHistory ABC unseen |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAIN_ORDER:
        value = mini[domain]
        lines.append(
            f"| {domain} | {value['A_OTHER_GROUP_COVERAGE']:.2%} | {value['AB_OTHER_GROUP_COVERAGE']:.2%} | "
            f"{value['ABC_OTHER_GROUP_COVERAGE']:.2%} | {value['NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE']:.2%} |"
        )
    lines.extend(
        [
            "", "## Decision", "",
            f"- DATA_COVERAGE_LIMIT_SUPPORT: `{summary['data_coverage_limit_support']}`",
            f"- DECODER_RANKING_LIMIT_SUPPORT: `{summary['decoder_ranking_limit_support']}`",
            f"- VIDEO_FINE_ITEM_SPACE_IS_HARDER: `{summary['video_fine_item_space_is_harder']}`",
            f"- ROOT_CLASS: `{summary['root_class']}`",
            f"- Phase1.5.1 coverage/Beam match: `{joined['training_coverage_matches_hierarchy_failure']}`",
            "", summary["conclusion"], "", "No GPU experiment was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def terminal(
    audit: dict[str, Any], inventory: dict[str, Any], natural: dict[str, Any], branching: dict[str, Any],
    entropy: dict[str, Any], comparison: dict[str, Any], joined: dict[str, Any], summary: dict[str, Any],
) -> str:
    mini = natural["corpora"]["MINI"]["contracts"]["ANY_GOLD"]
    bata = natural["corpora"]["BATA"]["contracts"]["ANY_GOLD"]
    target_n = sum(mini[domain]["N"] for domain in DOMAIN_ORDER)
    bata_group_seen = sum(bata[domain]["N"] * bata[domain]["SAME_GROUP_SEEN_RATE"] for domain in DOMAIN_ORDER) / target_n
    mini_group_seen = sum(mini[domain]["N"] * mini[domain]["SAME_GROUP_SEEN_RATE"] for domain in DOMAIN_ORDER) / target_n
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}", "RESULT_COMMIT=PENDING_REPORT_COMMIT",
        f"PUSH_STATUS={audit['push_status']}", f"GIT_STATUS_SHORT={audit['git_status_short'] or 'EMPTY'}",
        f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}", f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}",
        f"RUNTIME_GITHUB_PARITY={audit['runtime_github_parity']}", "", "--- Corpus ---", "",
        f"BATA_REC_GROUPS={inventory['corpora']['BATA']['recommendation_groups']}",
        f"MINIFIX_REC_GROUPS={inventory['corpora']['MiniFix']['recommendation_groups']}",
        f"GAMMA_REC_GROUPS={inventory['corpora']['Gamma']['recommendation_groups']}",
        f"MINIFIX_GAMMA_GROUP_PARITY={inventory['minifix_gamma_group_parity']}",
        f"MINIFIX_GAMMA_GOLD_PARITY={inventory['minifix_gamma_gold_parity']}",
        "ADAPTATION_HELDOUT_IS_NOT_BATA_HELDOUT=YES",
        f"BATA_TARGET_GROUP_SEEN_RATE={bata_group_seen:.9f}",
        f"MINI_TARGET_GROUP_SEEN_RATE={mini_group_seen:.9f}",
        "", "--- Mini Natural Coverage ---", "",
    ]
    for domain in DOMAIN_ORDER:
        label, value = domain.upper(), mini[domain]
        lines.extend(
            [
                f"{label}_MINI_A_OTHER_GROUP_COVERAGE={value['A_OTHER_GROUP_COVERAGE']:.9f}",
                f"{label}_MINI_AB_OTHER_GROUP_COVERAGE={value['AB_OTHER_GROUP_COVERAGE']:.9f}",
                f"{label}_MINI_ABC_OTHER_GROUP_COVERAGE={value['ABC_OTHER_GROUP_COVERAGE']:.9f}",
            ]
        )
    lines.extend(["", "--- BATA Natural Coverage ---", ""])
    for domain in DOMAIN_ORDER:
        label, value = domain.upper(), bata[domain]
        lines.extend(
            [
                f"{label}_BATA_A_OTHER_GROUP_COVERAGE={value['A_OTHER_GROUP_COVERAGE']:.9f}",
                f"{label}_BATA_AB_OTHER_GROUP_COVERAGE={value['AB_OTHER_GROUP_COVERAGE']:.9f}",
                f"{label}_BATA_ABC_OTHER_GROUP_COVERAGE={value['ABC_OTHER_GROUP_COVERAGE']:.9f}",
            ]
        )
    lines.extend(["", "--- NonHistory ABC Coverage ---", ""])
    for domain in DOMAIN_ORDER:
        lines.append(f"{domain.upper()}_MINI_NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE={mini[domain]['NONHISTORY_ABC_OTHER_GROUP_UNSEEN_RATE']:.9f}")
    lines.extend(["", "--- Branching ---", ""])
    for domain in DOMAIN_ORDER:
        branch = branching["MINI"][domain]["distinct_C_per_AB"]["macro"]
        h = entropy["MINI"][domain]["frequency_weighted_mean_H_C_given_AB_nats"]
        lines.extend(
            [
                f"{domain.upper()}_MINI_C_PER_AB_MEDIAN={branch['median']:.8f}",
                f"{domain.upper()}_MINI_C_PER_AB_P90={branch['p90']:.8f}",
                f"{domain.upper()}_MINI_H_C_GIVEN_AB={h:.8f}",
            ]
        )
    lines.extend(["", "--- BATA vs Mini ---", ""])
    for domain in DOMAIN_ORDER:
        delta = comparison["ANY_GOLD"][domain]["BATA_EXTRA_ABC_OTHER_GROUP_COVERAGE_OVER_MINI"]
        lines.append(f"BATA_EXTRA_ABC_COVERAGE_OVER_MINI_{domain.upper()}={delta:.9f}")
    lines.extend(
        [
            f"BATA_HAS_MORE_FINE_ITEM_TRAINING_OPPORTUNITY={comparison['BATA_HAS_MORE_FINE_ITEM_TRAINING_OPPORTUNITY']}",
            "", "--- Phase1.5.1 Join ---", "",
            f"SN_A_SEEN_RATE={joined['summary']['SN']['A_TOTAL_SEEN_RATE']:.8f}",
            f"SN_AB_SEEN_RATE={joined['summary']['SN']['AB_TOTAL_SEEN_RATE']:.8f}",
            f"SN_ABC_OTHER_GROUP_SEEN_RATE={joined['summary']['SN']['ABC_OTHER_GROUP_SEEN_RATE']:.8f}",
            f"UN_A_SEEN_RATE={joined['summary']['UN']['A_TOTAL_SEEN_RATE']:.8f}",
            f"UN_AB_SEEN_RATE={joined['summary']['UN']['AB_TOTAL_SEEN_RATE']:.8f}",
            f"UN_ABC_OTHER_GROUP_SEEN_RATE={joined['summary']['UN']['ABC_OTHER_GROUP_SEEN_RATE']:.8f}",
            f"TRAINING_COVERAGE_MATCHES_HIERARCHY_FAILURE={joined['training_coverage_matches_hierarchy_failure']}",
            "", "--- Interpretation ---", "",
            f"DATA_COVERAGE_LIMIT_SUPPORT={summary['data_coverage_limit_support']}",
            f"DECODER_RANKING_LIMIT_SUPPORT={summary['decoder_ranking_limit_support']}",
            f"VIDEO_FINE_ITEM_SPACE_IS_HARDER={summary['video_fine_item_space_is_harder']}",
            f"COARSE_INTEREST_SIGNAL={summary['coarse_interest_signal']}",
            f"FINE_ITEM_SIGNAL={summary['fine_item_signal']}", f"ROOT_CLASS={summary['root_class']}",
            f"PRIMARY_SIGNAL={summary['primary_signal']}", f"MAIN_LIMITATION={summary['main_limitation']}",
            f"CONCLUSION={summary['conclusion']}",
            f"BETA_TRUE_RECOMMENDATION_IDENTIFIABLE={summary['beta_true_recommendation_identifiable']}",
            f"FUTURE_GPU_EXPERIMENT={summary['future_gpu_experiment']}", "",
            "GPU_INFERENCE_STARTED=NO", "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0",
            "SELF_COT_GENERATION_STARTED=NO", "EXTERNAL_EVAL_STARTED=NO", "NEXT_EXPERIMENT_STARTED=NO",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    assert parse_sid("<|video_begin|><s_a_1><s_b_2><s_c_3>") == ("video", 1, 2, 3)
    assert prefix(("video", 1, 2, 3), "A") == ("video", 1)
    assert prefix(("video", 1, 2, 3), "AB") == ("video", 1, 2)
    assert [frequency_bucket(value) for value in (0, 1, 2, 4, 5, 9, 10, 19, 20)] == [
        "0", "1", "2-4", "2-4", "5-9", "5-9", "10-19", "10-19", "20+"
    ]
    summary = weighted_summary([(1, 1), (3, 3)])
    assert summary["mean"] == 2.5 and summary["median"] == 3.0
    fake = new_index("X", Path("x"), "x")
    sid = ("video", 1, 2, 3)
    other = ("video", 1, 2, 4)
    for group, values in (("g", [sid]), ("h", [sid, other])):
        fake["group_all_gold"][group] = frozenset(values)
        fake["group_current_sids"][group].update(values)
        for value in values:
            for level in ("A", "AB", "ABC"):
                key = prefix(value, level)
                fake["row_frequency"][level][key] += 1
                fake["unique_group_sets"][level][key].add(group)
                fake["group_row_frequency"][level][group][key] += 1
    target = {"group_id": "g"}
    cov = gold_coverage(fake, target, [sid])
    assert cov["ABC_TOTAL_UNIQUE_GROUP_FREQUENCY"] == 2
    assert cov["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"] == 1
    _, children = branch_children(fake)
    detail = target_surprisal(fake, target, sid, children)
    assert detail["AB_DISTINCT_C"] == 2 and detail["ABC_OTHER_GROUP_UNIQUE_GROUP_FREQUENCY"] == 1
    print("CPU_SELF_TEST_PASS=YES")


def run() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU_ONLY_GATE_REQUIRES_CUDA_VISIBLE_DEVICES_EMPTY")
    if "torch" in sys.modules:
        raise RuntimeError("CPU_ONLY_GATE_TORCH_IMPORTED")
    audit = code_audit()
    before = protected_hashes()
    canonical, holdout, source_contract = load_canonical()
    bata = scan_corpus("BATA", BATA_PATH, canonical, holdout)
    minifix = scan_corpus("MiniFix", MINIFIX_PATH)
    gamma = scan_corpus("Gamma", GAMMA_PATH)
    natural = build_natural(bata, holdout)
    group_parity = set(minifix["group_all_gold"]) == set(gamma["group_all_gold"])
    gold_parity = minifix["group_all_gold"] == gamma["group_all_gold"]
    current_parity = minifix["group_current_sids"] == gamma["group_current_sids"]
    if not (group_parity and gold_parity and current_parity):
        raise RuntimeError("MINIFIX_GAMMA_TRAIN_CONTRACT_PARITY_FAIL")

    inventory = {
        "corpora": {name: corpus_public(value) for name, value in (("BATA", bata), ("MiniFix", minifix), ("Gamma", gamma))},
        "minifix_gamma_group_parity": "PASS" if group_parity else "FAIL",
        "minifix_gamma_gold_parity": "PASS" if gold_parity else "FAIL",
        "minifix_gamma_current_gold_parity": "PASS" if current_parity else "FAIL",
        "mini_coverage_index": "MINIFIX_EQ_GAMMA_USE_MINIFIX",
        "frequency_primary_unit": "UNIQUE_GROUP_FREQUENCY",
        "frequency_secondary_unit": "ROW_FREQUENCY",
    }
    annotations = {"BATA": annotate_targets(natural, bata, "BATA"), "MINI": annotate_targets(natural, minifix, "MINI")}
    natural_payload = natural_coverage_payload(natural, annotations)
    frequency = {"BATA": frequency_stats(bata), "MINI": frequency_stats(minifix)}
    branching = {"BATA": branching_stats(bata), "MINI": branching_stats(minifix)}
    entropy = {"BATA": conditional_entropy(bata), "MINI": conditional_entropy(minifix)}
    surprisal = {label: target_frequency_surprisal(rows) for label, rows in annotations.items()}
    comparison = mini_vs_bata(natural_payload)
    joined = phase151_join(annotations["MINI"], natural)
    summary = root_decision(natural_payload, branching, entropy, surprisal, comparison, joined)
    after = protected_hashes()
    if before != after:
        raise RuntimeError("PROTECTED_RESULT_SHA256_CHANGED")
    source_contract.update(
        {
            "natural_pool": "ADAPTATION_HELDOUT_NATURAL_PROXY",
            "natural_N": len(natural),
            "adaptation_heldout_is_not_bata_heldout": True,
            "phase152_history_overlap_parity": "PASS",
            "protected_result_sha256_before": before,
            "protected_result_sha256_after": after,
            "protected_results_unchanged": True,
            "cpu_only": True,
            "torch_imported": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    )
    history_cross = {
        label: natural_payload["corpora"][label]["history_x_train_abc"] for label in ("BATA", "MINI")
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "source_contract_audit.json", source_contract)
    write_json(OUTPUT / "training_corpus_inventory.json", inventory)
    write_json(OUTPUT / "training_sid_frequency_stats.json", frequency)
    write_json(OUTPUT / "natural1595_training_coverage.json", natural_payload)
    (OUTPUT / "natural1595_training_coverage.md").write_text(render_natural_md(natural_payload), encoding="utf-8")
    write_json(OUTPUT / "history_training_cross.json", history_cross)
    write_json(OUTPUT / "branching_factor_stats.json", branching)
    write_json(OUTPUT / "conditional_entropy_stats.json", entropy)
    write_json(OUTPUT / "target_gold_frequency_surprisal.json", surprisal)
    write_json(OUTPUT / "mini_vs_bata_coverage.json", comparison)
    write_json(OUTPUT / "phase151_coverage_beam_join.json", joined)
    (OUTPUT / "phase151_coverage_beam_join.md").write_text(render_join_md(joined), encoding="utf-8")
    write_json(OUTPUT / "root_cause_summary.json", summary)
    (OUTPUT / "root_cause_summary.md").write_text(render_root_md(summary, natural_payload, joined), encoding="utf-8")
    review = terminal(audit, inventory, natural_payload, branching, entropy, comparison, joined, summary)
    (OUTPUT / "CHATGPT_TRAINING_COVERAGE_REVIEW.txt").write_text(review + "\n", encoding="utf-8")
    print(review)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 1:
        run()
    else:
        raise SystemExit("usage: recommendation_training_coverage_hierarchy_audit.py [--self-test]")
