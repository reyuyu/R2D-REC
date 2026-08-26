"""Build the CPU-only TrueRec V2 Curriculum2048 dataset.

The builder consumes only the frozen Phase 0.5 Train records.  It validates the
fixed-domain/no-bridge ABC contract, ranks learnability within each domain, and
constructs a smooth four-stage epoch-1 order with exact 16/domain balance in
every consecutive block of 64 groups.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Sequence


DOMAINS = ("video", "prod", "ad", "living")
DOMAIN_TOKEN = {domain: f"<|{domain}_begin|>" for domain in DOMAINS}
TOTAL_GROUPS = 2048
DOMAIN_GROUPS = 512
STAGE_GROUPS = 512
STAGE_DOMAIN_GROUPS = 128
BLOCK_GROUPS = 64
BLOCK_DOMAIN_GROUPS = 16
STAGES = ("stage1", "stage2", "stage3", "stage4")
SEED = "truerec-v2-curriculum2048-epoch1-20260827"
MAX_POSITION_EMBEDDINGS = 131072
EXPECTED_TRAIN_GROUPS = 17971
EXPECTED_TRAIN_SHA256 = (
    "97edc2d3c600dbd073f6c694a3d497c5dba10e42ce0350628b63eddbca79ca7e"
)
EXPECTED_SPLIT_SHA256 = {
    "train_pool_group_ids.json": "ef645cf62cf50f619b62f2d0fbfb267dd4f8df6c5cfc89c70a7977df3aecf724",
    "dev_group_ids.json": "70341b476c57e4d7dc961ccf423ab3991a42375af1989bad040be4134e8f4fe4",
    "final_group_ids.json": "e6be46befa8ed320fb0762e6df23f553c6646297bdf5649003e332c04b2e61d0",
    "probe20_group_ids.json": "7193f540ee229ef53b7ca09e541f065396fe212a8cfd43893cceff499d83abdf",
}

RUNTIME_ROOT = Path("/data/GRPO")
DATA_ROOT = RUNTIME_ROOT / "truerec_grpo/data"
TRAIN_RECORDS = DATA_ROOT / "fixed_domain_abc/train_records.jsonl"
SPLIT_DIR = DATA_ROOT / "splits"
OUTPUT_DIR = DATA_ROOT / "curriculum2048_v2"
MODEL_CONFIG = Path("/data/models/onereason-8b-pretrain-competition/config.json")

ABC_RE = re.compile(r"^<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$")
SID_RE = re.compile(
    r"^<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$"
)
BRIDGES = (
    "该用户最近喜欢的视频有:",
    "该用户最近点击了商品:",
    "该用户最近感兴趣的广告有:",
    "该用户最近首次打赏了主播:",
)
K_BUCKETS = ("1", "2", "3", "4", "5+")
HIERARCHY_CLASSES = ("A_RICH", "B_RICH", "C_RICH", "SINGLETON", "OTHER")
STAGE_CATEGORY_QUOTAS = {
    "video": {
        "stage1": {"A_RICH": 128, "B_RICH": 0, "C_RICH": 0, "SINGLETON": 0},
        "stage2": {"A_RICH": 80, "B_RICH": 40, "C_RICH": 0, "SINGLETON": 8},
        "stage3": {"A_RICH": 24, "B_RICH": 45, "C_RICH": 9, "SINGLETON": 50},
        "stage4": {"A_RICH": 8, "B_RICH": 6, "C_RICH": 4, "SINGLETON": 110},
    },
    "prod": {
        "stage1": {"A_RICH": 128, "B_RICH": 0, "C_RICH": 0, "SINGLETON": 0},
        "stage2": {"A_RICH": 96, "B_RICH": 28, "C_RICH": 0, "SINGLETON": 4},
        "stage3": {"A_RICH": 24, "B_RICH": 32, "C_RICH": 4, "SINGLETON": 68},
        "stage4": {"A_RICH": 4, "B_RICH": 4, "C_RICH": 2, "SINGLETON": 118},
    },
    "ad": {
        "stage1": {"A_RICH": 128, "B_RICH": 0, "C_RICH": 0, "SINGLETON": 0},
        "stage2": {"A_RICH": 112, "B_RICH": 12, "C_RICH": 0, "SINGLETON": 4},
        "stage3": {"A_RICH": 24, "B_RICH": 14, "C_RICH": 2, "SINGLETON": 88},
        "stage4": {"A_RICH": 4, "B_RICH": 3, "C_RICH": 2, "SINGLETON": 119},
    },
    "living": {
        "stage1": {"A_RICH": 128, "B_RICH": 0, "C_RICH": 0, "SINGLETON": 0},
        "stage2": {"A_RICH": 10, "B_RICH": 5, "C_RICH": 0, "SINGLETON": 113},
        "stage3": {"A_RICH": 3, "B_RICH": 6, "C_RICH": 4, "SINGLETON": 115},
        "stage4": {"A_RICH": 2, "B_RICH": 2, "C_RICH": 3, "SINGLETON": 121},
    },
}
REJECTION_REASONS = (
    "duplicate_group_id", "duplicate_group_metadata_conflict", "missing_group_id",
    "not_in_frozen_train_pool", "split_not_train_pool", "invalid_target_domain",
    "invalid_fixed_domain_token", "invalid_system", "invalid_nothink_prompt",
    "bridge_contamination", "empty_all_gold_abc", "gold_sid_abc_cardinality_mismatch",
    "malformed_gold_sid_or_abc", "gold_domain_mismatch", "gold_sid_abc_mismatch",
    "empty_gold_after_dedup", "nothink_renderer_error", "empty_rendered_context",
    "context_requires_truncation",
)


class CurriculumError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


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
            raw = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            handle.write(raw)
            digest.update(raw)
            count += 1
    return digest.hexdigest(), count


def stable_hash(group_id: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).hexdigest()


def load_json_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload and isinstance(payload[0], dict):
        return {str(row["recommendation_group_id"]) for row in payload}
    return {str(value) for value in payload}


def parse_gold(record: dict[str, Any]) -> tuple[list[tuple[int, int, int]], list[str]]:
    reasons: list[str] = []
    raw_abc = record.get("all_gold_abc")
    raw_sids = record.get("all_gold_sids")
    if not isinstance(raw_abc, list) or not raw_abc:
        return [], ["empty_all_gold_abc"]
    if not isinstance(raw_sids, list) or len(raw_sids) != len(raw_abc):
        return [], ["gold_sid_abc_cardinality_mismatch"]
    domain = record.get("target_domain")
    parsed: list[tuple[int, int, int]] = []
    for abc_text, sid_text in zip(raw_abc, raw_sids):
        abc_match = ABC_RE.fullmatch(str(abc_text))
        sid_match = SID_RE.fullmatch(str(sid_text))
        if abc_match is None or sid_match is None:
            reasons.append("malformed_gold_sid_or_abc")
            continue
        abc = tuple(int(value) for value in abc_match.groups())
        sid_domain = sid_match.group(1)
        sid_abc = tuple(int(value) for value in sid_match.groups()[1:])
        if sid_domain != domain:
            reasons.append("gold_domain_mismatch")
        if sid_abc != abc:
            reasons.append("gold_sid_abc_mismatch")
        parsed.append(abc)
    unique = list(dict.fromkeys(parsed))
    if not unique:
        reasons.append("empty_gold_after_dedup")
    return unique, sorted(set(reasons))


def hierarchy_class(k_a: int, k_ab: int, k_abc: int) -> str:
    if k_a >= 2:
        return "A_RICH"
    if k_a == 1 and k_ab >= 2:
        return "B_RICH"
    if k_ab == 1 and k_abc >= 2:
        return "C_RICH"
    if (k_a, k_ab, k_abc) == (1, 1, 1):
        return "SINGLETON"
    return "OTHER"


def static_eligibility(record: dict[str, Any], train_ids: set[str]) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    group_id = str(record.get("recommendation_group_id", ""))
    domain = record.get("target_domain")
    if not group_id:
        reasons.append("missing_group_id")
    elif group_id not in train_ids:
        reasons.append("not_in_frozen_train_pool")
    if record.get("split") != "train_pool":
        reasons.append("split_not_train_pool")
    if domain not in DOMAINS:
        reasons.append("invalid_target_domain")
    if domain in DOMAINS and record.get("fixed_domain_token") != DOMAIN_TOKEN[domain]:
        reasons.append("invalid_fixed_domain_token")
    system = record.get("system")
    user = record.get("user_content_nothink")
    if not isinstance(system, str) or not system:
        reasons.append("invalid_system")
    if not isinstance(user, str) or not user.endswith("/no_think"):
        reasons.append("invalid_nothink_prompt")
    if isinstance(system, str) and isinstance(user, str) and any(bridge in system + "\n" + user for bridge in BRIDGES):
        reasons.append("bridge_contamination")
    gold, gold_reasons = parse_gold(record)
    reasons.extend(gold_reasons)
    if reasons:
        return None, sorted(set(reasons))
    unique_a = {abc[0] for abc in gold}
    unique_ab = {abc[:2] for abc in gold}
    enriched = dict(record)
    enriched["all_gold_abc"] = [f"<s_a_{a}><s_b_{b}><s_c_{c}>" for a, b, c in gold]
    enriched["K_A"] = len(unique_a)
    enriched["K_AB"] = len(unique_ab)
    enriched["K_ABC"] = len(gold)
    enriched["hierarchy_class"] = hierarchy_class(len(unique_a), len(unique_ab), len(gold))
    enriched["gold_duplicate_count"] = len(record["all_gold_abc"]) - len(gold)
    enriched["prefix_rich"] = bool(len(unique_a) >= 2 or len(unique_ab) >= 2 or len(gold) >= 2)
    return enriched, []


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        raise CurriculumError("EMPTY_PERCENTILE_INPUT")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def annotate_context_quality(records: list[dict[str, Any]]) -> None:
    for domain in DOMAINS:
        rows = [row for row in records if row["target_domain"] == domain]
        lengths = [int(row["context_token_count"]) for row in rows]
        median = statistics.median(lengths)
        p01, p99 = percentile(lengths, 0.01), percentile(lengths, 0.99)
        scale = max(1.0, percentile(lengths, 0.90) - percentile(lengths, 0.10))
        ordered = sorted(lengths)
        for row in rows:
            length = int(row["context_token_count"])
            rank = sum(value < length for value in ordered)
            row["context_length_percentile"] = round(rank / max(1, len(ordered) - 1), 8)
            row["context_distance_from_domain_median"] = round(abs(length - median) / scale, 8)
            row["context_outlier"] = bool(length < p01 or length > p99)


def quality_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(record["K_A"]),
        -int(record["K_AB"]),
        -int(record["K_ABC"]),
        bool(record["context_outlier"]),
        float(record["context_distance_from_domain_median"]),
        int(record["gold_duplicate_count"]),
        stable_hash(record["recommendation_group_id"]),
        record["recommendation_group_id"],
    )


def assign_domain_stages(records: list[dict[str, Any]], domain: str) -> dict[str, list[dict[str, Any]]]:
    pools = {
        name: sorted((row for row in records if row["hierarchy_class"] == name), key=quality_key)
        for name in HIERARCHY_CLASSES
    }
    required = Counter()
    for stage in STAGES:
        quotas = STAGE_CATEGORY_QUOTAS[domain][stage]
        if sum(quotas.values()) != STAGE_DOMAIN_GROUPS:
            raise CurriculumError(f"STAGE_QUOTA_SUM_FAIL={domain},{stage},{quotas}")
        required.update(quotas)
    for name, count in required.items():
        if len(pools[name]) < count:
            raise CurriculumError(f"HIERARCHY_CAPACITY_FAIL={domain},{name},{len(pools[name])},{count}")
    result: dict[str, list[dict[str, Any]]] = {}
    cursors = Counter()
    for stage_index, stage in enumerate(STAGES):
        rows = []
        for name in ("A_RICH", "B_RICH", "C_RICH", "SINGLETON"):
            count = STAGE_CATEGORY_QUOTAS[domain][stage][name]
            start = cursors[name]
            chosen = pools[name][start: start + count]
            cursors[name] += count
            for rank, row in enumerate(chosen, start=start):
                row["selection_tier"] = name
                row["quality_rank_within_hierarchy_class"] = rank
            rows.extend(chosen)
        rows.sort(key=quality_key)
        for row in rows:
            row["stage"] = stage
            row["stage_index"] = stage_index
        result[stage] = rows
    if any(len(result[stage]) != STAGE_DOMAIN_GROUPS for stage in STAGES):
        raise CurriculumError(f"STAGE_ASSIGNMENT_SIZE_FAIL={domain}")
    if any(row["hierarchy_class"] != "A_RICH" for row in result["stage1"]):
        raise CurriculumError(f"STAGE1_NOT_ALL_A_RICH={domain}")
    return result


def build_epoch1_order(staged: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(STAGES):
        for block in range(STAGE_GROUPS // BLOCK_GROUPS):
            domain_order = DOMAINS[block % len(DOMAINS):] + DOMAINS[: block % len(DOMAINS)]
            for domain in domain_order:
                rows = staged[domain][stage]
                ordered.extend(rows[block * BLOCK_DOMAIN_GROUPS: (block + 1) * BLOCK_DOMAIN_GROUPS])
    for epoch_index, row in enumerate(ordered):
        row["epoch1_index"] = epoch_index
    return ordered


def bucket(value: int) -> str:
    return str(value) if value in (1, 2, 3, 4) else "5+"


def distribution(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    result = {}
    for domain in DOMAINS:
        counts = Counter(bucket(int(row[field])) for row in records if row["target_domain"] == domain)
        result[domain] = {name: counts[name] for name in ("1", "2", "3", "4", "5+")}
    return result


def joint_hierarchy_census(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"groups": len(records), "domains": {}}
    for domain in DOMAINS:
        rows = [row for row in records if row["target_domain"] == domain]
        categories = Counter(row["hierarchy_class"] for row in rows)
        combinations = Counter((int(row["K_A"]), int(row["K_AB"]), int(row["K_ABC"])) for row in rows)
        result["domains"][domain] = {
            "N": len(rows),
            "hierarchy_classes": {name: categories[name] for name in HIERARCHY_CLASSES},
            "joint_K_A_K_AB_K_ABC": [
                {"K_A": values[0], "K_AB": values[1], "K_ABC": values[2], "count": count}
                for values, count in sorted(combinations.items())
            ],
            "K_A": distribution(rows, "K_A")[domain],
            "K_AB": distribution(rows, "K_AB")[domain],
            "K_ABC": distribution(rows, "K_ABC")[domain],
        }
    return result


def stage_domain_census(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for stage in STAGES:
        result[stage] = {}
        for domain in DOMAINS:
            rows = [row for row in records if row["stage"] == stage and row["target_domain"] == domain]
            categories = Counter(row["hierarchy_class"] for row in rows)
            result[stage][domain] = {
                "N": len(rows),
                "hierarchy_classes": {name: categories[name] for name in HIERARCHY_CLASSES},
                "K_A": distribution(rows, "K_A")[domain],
                "K_AB": distribution(rows, "K_AB")[domain],
                "K_ABC": distribution(rows, "K_ABC")[domain],
            }
    return result


def validate_order(
    ordered: list[dict[str, Any]], train_ids: set[str], probe_ids: set[str], dev_ids: set[str], final_ids: set[str]
) -> dict[str, Any]:
    selected_ids = {row["recommendation_group_id"] for row in ordered}
    domain_counts = Counter(row["target_domain"] for row in ordered)
    stage_counts = Counter(row["stage"] for row in ordered)
    every64 = []
    for start in range(0, len(ordered), BLOCK_GROUPS):
        counts = Counter(row["target_domain"] for row in ordered[start: start + BLOCK_GROUPS])
        every64.append({"start": start, "end": start + BLOCK_GROUPS - 1, "domain_counts": dict(counts)})
    audit = {
        "selected_groups": len(ordered),
        "selected_unique_groups": len(selected_ids),
        "selected_subset_of_train": "PASS" if selected_ids <= train_ids else "FAIL",
        "domain_counts": {domain: domain_counts[domain] for domain in DOMAINS},
        "stage_counts": {stage: stage_counts[stage] for stage in STAGES},
        "every64_domain_balance": "PASS" if all(
            item["domain_counts"] == {domain: BLOCK_DOMAIN_GROUPS for domain in DOMAINS} for item in every64
        ) else "FAIL",
        "every64_blocks": every64,
        "probe_overlap": len(selected_ids & probe_ids),
        "dev_overlap": len(selected_ids & dev_ids),
        "final_overlap": len(selected_ids & final_ids),
    }
    expected = {
        "selected_groups": TOTAL_GROUPS,
        "selected_unique_groups": TOTAL_GROUPS,
        "selected_subset_of_train": "PASS",
        "domain_counts": {domain: DOMAIN_GROUPS for domain in DOMAINS},
        "stage_counts": {stage: STAGE_GROUPS for stage in STAGES},
        "every64_domain_balance": "PASS",
        "probe_overlap": 0,
        "dev_overlap": 0,
        "final_overlap": 0,
    }
    if any(audit[key] != value for key, value in expected.items()):
        raise CurriculumError(f"ORDER_AUDIT_FAIL={audit}")
    return audit


def compact_quality(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": record["recommendation_group_id"],
        "domain": record["target_domain"],
        "K_A": record["K_A"],
        "K_AB": record["K_AB"],
        "K_ABC": record["K_ABC"],
        "stage": record["stage"],
        "epoch1_index": record["epoch1_index"],
        "quality_features": {
            "hierarchy_class": record["hierarchy_class"],
            "prefix_rich": record["prefix_rich"],
            "selection_tier": record["selection_tier"],
            "quality_rank_within_hierarchy_class": record["quality_rank_within_hierarchy_class"],
            "context_token_count": record["context_token_count"],
            "context_length_percentile": record["context_length_percentile"],
            "context_outlier": record["context_outlier"],
            "context_distance_from_domain_median": record["context_distance_from_domain_median"],
            "gold_duplicate_count": record["gold_duplicate_count"],
        },
    }


def load_and_filter(renderer: Any) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, set[str]]]:
    if file_sha256(TRAIN_RECORDS) != EXPECTED_TRAIN_SHA256:
        raise CurriculumError("TRAIN_SOURCE_SHA256_FAIL")
    split_ids = {}
    for name, expected_sha in EXPECTED_SPLIT_SHA256.items():
        path = SPLIT_DIR / name
        if file_sha256(path) != expected_sha:
            raise CurriculumError(f"FROZEN_SPLIT_SHA256_FAIL={name}")
        split_ids[name] = load_json_ids(path)
    train_ids = split_ids["train_pool_group_ids.json"]
    rejection = Counter({reason: 0 for reason in REJECTION_REASONS})
    eligible = []
    seen: dict[str, dict[str, Any]] = {}
    with TRAIN_RECORDS.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            group_id = str(raw.get("recommendation_group_id", ""))
            if group_id in seen:
                rejection["duplicate_group_id"] += 1
                if raw != seen[group_id]:
                    rejection["duplicate_group_metadata_conflict"] += 1
                continue
            seen[group_id] = raw
            row, reasons = static_eligibility(raw, train_ids)
            if reasons:
                rejection.update(reasons)
                continue
            try:
                context_ids = renderer.rl_context_ids(row["system"], row["user_content_nothink"], row["fixed_domain_token"])
            except Exception:
                rejection["nothink_renderer_error"] += 1
                continue
            if not context_ids:
                rejection["empty_rendered_context"] += 1
                continue
            if len(context_ids) + 3 > MAX_POSITION_EMBEDDINGS:
                rejection["context_requires_truncation"] += 1
                continue
            row["context_token_count"] = len(context_ids)
            row["context_truncation_applied"] = False
            eligible.append(row)
    if len(seen) != EXPECTED_TRAIN_GROUPS or set(seen) != train_ids:
        raise CurriculumError(f"TRAIN_GROUP_IDENTITY_FAIL={len(seen)},{len(train_ids)}")
    annotate_context_quality(eligible)
    return eligible, dict(sorted(rejection.items())), {
        "train": train_ids,
        "dev": split_ids["dev_group_ids.json"],
        "final": split_ids["final_group_ids.json"],
        "probe": split_ids["probe20_group_ids.json"],
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if int(json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))["max_position_embeddings"]) != MAX_POSITION_EMBEDDINGS:
        raise CurriculumError("MODEL_CONTEXT_CONTRACT_FAIL")
    data_dir = Path(__file__).resolve().parent
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    from beta_gamma_renderer import BetaGammaRenderer

    renderer = BetaGammaRenderer()
    eligible, rejection, split_ids = load_and_filter(renderer)
    selected_by_domain = {}
    staged = {}
    for domain in DOMAINS:
        domain_rows = [row for row in eligible if row["target_domain"] == domain]
        staged[domain] = assign_domain_stages(domain_rows, domain)
        selected_by_domain[domain] = [row for stage in STAGES for row in staged[domain][stage]]
    ordered = build_epoch1_order(staged)
    order_audit = validate_order(ordered, split_ids["train"], split_ids["probe"], split_ids["dev"], split_ids["final"])

    full_records = []
    for row in ordered:
        payload = dict(row)
        payload["K"] = row["K_ABC"]
        full_records.append(payload)
    records_sha, records_count = write_jsonl(output_dir / "records.jsonl", full_records)
    order_payload = [row["recommendation_group_id"] for row in ordered]
    order_sha = write_json(output_dir / "epoch1_order.json", order_payload)
    metadata_sha, metadata_count = write_jsonl(output_dir / "records_metadata.jsonl", (compact_quality(row) for row in ordered))

    selected = [row for domain in DOMAINS for row in selected_by_domain[domain]]
    pool_census = joint_hierarchy_census(eligible)
    selected_categories = {
        domain: {
            name: sum(row["target_domain"] == domain and row["hierarchy_class"] == name for row in selected)
            for name in HIERARCHY_CLASSES
        }
        for domain in DOMAINS
    }
    selected_census = {
        "groups": len(selected),
        "domain_counts": dict(Counter(row["target_domain"] for row in selected)),
        "hierarchy_classes": selected_categories,
        "K_ABC": distribution(selected, "K_ABC"),
        "K_A": distribution(selected, "K_A"),
        "K_AB": distribution(selected, "K_AB"),
        "stage_domain": stage_domain_census(selected),
    }
    pool_video_tail = pool_census["domains"]["video"]["K_ABC"]["5+"] / pool_census["domains"]["video"]["N"]
    selected_video_tail = selected_census["K_ABC"]["video"]["5+"] / DOMAIN_GROUPS
    video_tail_gate = "PASS" if selected_census["K_ABC"]["video"]["5+"] > 0 and selected_video_tail >= pool_video_tail else "FAIL"
    if video_tail_gate != "PASS":
        raise CurriculumError("VIDEO_HIGH_K_TAIL_NOT_PRESERVED")
    if sum(
        selected_census["stage_domain"]["stage1"][domain]["hierarchy_classes"]["A_RICH"]
        for domain in DOMAINS
    ) != STAGE_GROUPS:
        raise CurriculumError("STAGE1_A_RICH_512_GATE_FAIL")
    census = {"eligible_pool": pool_census, "selected": selected_census}
    selection_report = {
        "seed": SEED,
        "ranking": ["hierarchy curriculum class", "K_A descending", "K_AB descending", "K_ABC descending", "context outlier avoidance", "context median distance", "stable SHA256 tie-break"],
        "domain_specific_policy": "Select for hierarchical credit exposure; do not match the source natural K distribution and do not impose a shared cross-domain K distribution.",
        "curriculum_policy": "Stage1 all A_RICH; Stage2 A/B; Stage3 B/C with rehearsal; Stage4 C/singleton with A/B rehearsal. Scarce B/C groups are fully used.",
        "stage_category_quotas": STAGE_CATEGORY_QUOTAS,
        "capacity_note": "After Stage1, living has only 15 A_RICH and 20 total B_RICH+C_RICH in the full pool; singleton fill is mechanically necessary.",
        "source_train_groups": EXPECTED_TRAIN_GROUPS,
        "eligible_groups": len(eligible),
        "rejection_reason_counts": rejection,
        "order_audit": order_audit,
        "stage_domain_census": selected_census["stage_domain"],
        "video_high_k_tail": {
            "eligible_rate": round(pool_video_tail, 8),
            "selected_rate": round(selected_video_tail, 8),
            "gate": video_tail_gate,
        },
        "epoch2_order_generated": False,
    }
    census_sha = write_json(output_dir / "joint_hierarchy_census.json", census)
    write_json(output_dir / "census.json", census)
    report_sha = write_json(output_dir / "selection_report.json", selection_report)
    manifest = {
        "name": "truerec_v2_curriculum2048",
        "seed": SEED,
        "source_pool": {
            "path": str(TRAIN_RECORDS),
            "sha256": EXPECTED_TRAIN_SHA256,
            "groups": EXPECTED_TRAIN_GROUPS,
            "identity": "Phase0.5 frozen fixed-domain ABC Train pool",
        },
        "frozen_split_sha256": EXPECTED_SPLIT_SHA256,
        "records": {"path": str(output_dir / "records.jsonl"), "sha256": records_sha, "count": records_count},
        "records_metadata": {"path": str(output_dir / "records_metadata.jsonl"), "sha256": metadata_sha, "count": metadata_count},
        "epoch1_order": {"path": str(output_dir / "epoch1_order.json"), "sha256": order_sha, "count": len(order_payload)},
        "joint_hierarchy_census_sha256": census_sha,
        "selection_report_sha256": report_sha,
        "selected_unique_2048": "PASS" if order_audit["selected_unique_groups"] == TOTAL_GROUPS else "FAIL",
        "domain_512_each": "PASS" if order_audit["domain_counts"] == {domain: DOMAIN_GROUPS for domain in DOMAINS} else "FAIL",
        "every64_domain_balance": order_audit["every64_domain_balance"],
        "probe_overlap": order_audit["probe_overlap"],
        "dev_overlap": order_audit["dev_overlap"],
        "final_overlap": order_audit["final_overlap"],
        "video_high_k_tail_preserved": video_tail_gate,
        "stage_domain_hierarchy_census": selected_census["stage_domain"],
        "epoch2_order_generated": False,
        "gpu_inference_started": False,
        "training_started": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    return {"manifest": manifest, "census": census, "selection_report": selection_report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({
        "SOURCE_TRAIN_GROUPS": EXPECTED_TRAIN_GROUPS,
        "ELIGIBLE_GROUPS": result["selection_report"]["eligible_groups"],
        "SELECTED_GROUPS": result["manifest"]["records"]["count"],
        "DOMAIN_COUNTS": result["selection_report"]["order_audit"]["domain_counts"],
        "STAGE_COUNTS": result["selection_report"]["order_audit"]["stage_counts"],
        "EVERY_64_DOMAIN_BALANCE": result["manifest"]["every64_domain_balance"],
        "VIDEO_HIGH_K_TAIL_PRESERVED": result["manifest"]["video_high_k_tail_preserved"],
        "PROBE_OVERLAP": result["manifest"]["probe_overlap"],
        "DEV_OVERLAP": result["manifest"]["dev_overlap"],
        "FINAL_OVERLAP": result["manifest"]["final_overlap"],
        "RECORDS_SHA256": result["manifest"]["records"]["sha256"],
        "EPOCH1_ORDER_SHA256": result["manifest"]["epoch1_order"]["sha256"],
        "GPU_INFERENCE_STARTED": "NO",
        "TRAINING_STARTED": "NO",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
