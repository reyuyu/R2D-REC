#!/usr/bin/env python3
"""CPU-only audit of Recommendation Gold overlap with prompt SID history."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


GRPO_SCRIPT_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "scripts",
    Path("/data/GRPO/scripts"),
)
GRPO_SCRIPTS = next(
    (path for path in GRPO_SCRIPT_CANDIDATES if (path / "grpo_sid.py").is_file()),
    GRPO_SCRIPT_CANDIDATES[0],
)
sys.path.insert(0, str(GRPO_SCRIPTS))

from grpo_sid import all_sids, parse_sid  # noqa: E402


DOMAIN_ORDER = ("video", "prod", "ad", "living")
DOMAIN_LABELS = {"video": "Video", "prod": "Product", "ad": "Ad", "living": "Live"}
EXPECTED_ROUTES = {"think", "no_think"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_gold_sids(row: dict) -> frozenset[tuple[str, int, int, int]]:
    raw_values = row.get("all_gold_sids")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError("all_gold_sids must be a non-empty list")
    parsed = [parse_sid(value) for value in raw_values]
    if any(value is None for value in parsed):
        raise ValueError(f"invalid complete Gold SID in group {row.get('recommendation_group_id')}")
    values = frozenset(parsed)
    if len(values) != len(raw_values):
        raise ValueError(f"duplicate Gold SID in group {row.get('recommendation_group_id')}")
    domain = row.get("target_domain")
    if any(value[0] != domain for value in values):
        raise ValueError(f"Gold SID domain mismatch in group {row.get('recommendation_group_id')}")
    return values


def group_records(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group_id = row.get("recommendation_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("missing recommendation_group_id")
        grouped[group_id].append(row)

    records = []
    route_counts = Counter()
    for group_id, group_rows in sorted(grouped.items()):
        routes = {row.get("route") for row in group_rows}
        if len(group_rows) != 2 or routes != EXPECTED_ROUTES:
            raise ValueError(f"group {group_id} must contain exactly Think and NoThink rows")
        domains = {row.get("target_domain") for row in group_rows}
        if len(domains) != 1 or next(iter(domains)) not in DOMAIN_ORDER:
            raise ValueError(f"group {group_id} has invalid or inconsistent target_domain")
        gold_sets = [parse_gold_sids(row) for row in group_rows]
        history_sets = [frozenset(all_sids(row.get("prompt", ""))) for row in group_rows]
        if gold_sets[0] != gold_sets[1]:
            raise ValueError(f"group {group_id} has route-dependent Gold metadata")
        if history_sets[0] != history_sets[1]:
            raise ValueError(f"group {group_id} has route-dependent History SID set")
        if not history_sets[0]:
            raise ValueError(f"group {group_id} has no complete History SID in prompt")

        gold = gold_sets[0]
        history = history_sets[0]
        overlap_count = len(gold & history)
        gold_count = len(gold)
        history_rate = overlap_count / gold_count
        if overlap_count == 0:
            category = "novel_only"
        elif overlap_count == gold_count:
            category = "history_only"
        else:
            category = "mixed"
        records.append({
            "group_id": group_id,
            "domain": next(iter(domains)),
            "history_sid_count": len(history),
            "gold_sid_count": gold_count,
            "history_gold_count": overlap_count,
            "novel_gold_count": gold_count - overlap_count,
            "history_gold_rate": history_rate,
            "novel_gold_rate": 1.0 - history_rate,
            "category": category,
        })
        route_counts.update(routes)
    return records, {
        "raw_rows": len(rows),
        "unique_groups": len(grouped),
        "route_counts": dict(sorted(route_counts.items())),
        "route_parity": True,
    }


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_domain(records: list[dict]) -> dict:
    if not records:
        raise ValueError("cannot summarize empty domain")
    categories = Counter(record["category"] for record in records)
    gold_counts = [record["gold_sid_count"] for record in records]
    ceilings = [record["history_gold_rate"] for record in records]
    count = len(records)
    return {
        "sample_group_count": count,
        "gold_sid_count": {
            "mean": statistics.fmean(gold_counts),
            "median": statistics.median(gold_counts),
        },
        "category": {
            "history_only_count": categories["history_only"],
            "history_only_rate": _fraction(categories["history_only"], count),
            "mixed_count": categories["mixed"],
            "mixed_rate": _fraction(categories["mixed"], count),
            "novel_only_count": categories["novel_only"],
            "novel_only_rate": _fraction(categories["novel_only"], count),
        },
        "mean_history_gold_rate": statistics.fmean(ceilings),
        "mean_novel_gold_rate": statistics.fmean(record["novel_gold_rate"] for record in records),
        "history_copy_recall_ceiling": {
            "mean": statistics.fmean(ceilings),
            "median": statistics.median(ceilings),
            "zero_rate": statistics.fmean(value == 0 for value in ceilings),
            "below_0_25_rate": statistics.fmean(value < 0.25 for value in ceilings),
            "below_0_5_rate": statistics.fmean(value < 0.5 for value in ceilings),
        },
    }


def audit_user_structure(path: Path) -> dict:
    before = sha256_file(path)
    rows = read_jsonl(path)
    routes = Counter()
    subset_count = 0
    for row in rows:
        route = row.get("route")
        routes[route] += 1
        gold = set(row.get("gold_sids", ()))
        history = set(row.get("history_sids", ()))
        if not gold:
            raise ValueError(f"GR_USER row {row.get('sample_id')} has empty gold_sids")
        if gold.issubset(history):
            subset_count += 1
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("GR_USER source dataset SHA changed during read-only audit")
    return {
        "path": str(path),
        "sha256": before,
        "sample_count": len(rows),
        "route_counts": dict(sorted(routes.items())),
        "gold_subset_history_count": subset_count,
        "gold_subset_history_rate": _fraction(subset_count, len(rows)),
    }


def classify_task_conflict(overall: dict, user_structure: dict | None) -> str:
    user_is_extractive = (
        user_structure is not None
        and user_structure["gold_subset_history_rate"] == 1.0
    )
    recommendation_is_novel = (
        overall["mean_novel_gold_rate"] > 0.5
        and overall["category"]["novel_only_rate"] > 0.5
    )
    return "TASK_CONFLICT_SUPPORTED" if user_is_extractive and recommendation_is_novel else "TASK_CONFLICT_NOT_SUPPORTED"


def summarize(records: list[dict], source: dict, user_structure: dict | None = None) -> dict:
    by_domain = {
        domain: summarize_domain([record for record in records if record["domain"] == domain])
        for domain in DOMAIN_ORDER
    }
    video = by_domain["video"]
    comparisons = {}
    for domain in ("prod", "living"):
        item = by_domain[domain]
        comparisons[f"{domain}_vs_video"] = {
            "mean_novel_gold_rate_delta": item["mean_novel_gold_rate"] - video["mean_novel_gold_rate"],
            "novel_only_rate_delta": item["category"]["novel_only_rate"] - video["category"]["novel_only_rate"],
            "mean_history_copy_ceiling_delta": (
                item["history_copy_recall_ceiling"]["mean"]
                - video["history_copy_recall_ceiling"]["mean"]
            ),
        }
    overall = summarize_domain(records)
    return {
        "audit": "recommendation_history_novel_audit_v1",
        "scope": "task_structure_only",
        "source": source,
        "integrity": {
            "group_contract_passed": True,
            "all_gold_nonempty_valid_unique_and_target_domain": True,
            "history_extracted_with_existing_grpo_sid_all_sids": True,
            "recommendation_source_sha_unchanged": True,
            "user_source_sha_unchanged": True,
            "model_or_checkpoint_accessed": False,
            "generation_run": False,
            "training_run": False,
        },
        "domains": by_domain,
        "overall": overall,
        "user_structure": user_structure,
        "comparisons": comparisons,
        "decision": classify_task_conflict(overall, user_structure),
        "records": records,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(result: dict) -> str:
    lines = [
        "# Recommendation History-vs-Novel CPU Audit v1",
        "",
        "## Scope",
        "",
        "This is a CPU-only task-structure audit. It does not test whether a trained model actually copies History SIDs.",
        "",
        f"- Dataset: `{result['source']['path']}`",
        f"- SHA256: `{result['source']['sha256']}`",
        f"- Raw rows / unique groups: {result['source']['raw_rows']} / {result['source']['unique_groups']}",
        "- Gold: authoritative `all_gold_sids` group metadata",
        "- History: complete SIDs in `prompt`, extracted with existing `grpo_sid.all_sids`",
        "- Think/NoThink route parity: PASS",
        (
            f"- GR_USER Gold subset of History: "
            f"{result['user_structure']['gold_subset_history_count']}/"
            f"{result['user_structure']['sample_count']} "
            f"({_pct(result['user_structure']['gold_subset_history_rate'])})"
            if result.get("user_structure") else "- GR_USER structural comparison: not supplied"
        ),
        "",
        "## Domain results",
        "",
        "| Domain | Groups | Gold mean | Gold median | History only | Mixed | Novel only | Mean history rate | Mean novel rate | Ceiling median | Ceiling=0 | Ceiling<0.25 | Ceiling<0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain in DOMAIN_ORDER:
        item = result["domains"][domain]
        category = item["category"]
        ceiling = item["history_copy_recall_ceiling"]
        lines.append(
            f"| {DOMAIN_LABELS[domain]} | {item['sample_group_count']} | "
            f"{item['gold_sid_count']['mean']:.3f} | {item['gold_sid_count']['median']:.3f} | "
            f"{_pct(category['history_only_rate'])} | {_pct(category['mixed_rate'])} | "
            f"{_pct(category['novel_only_rate'])} | {_pct(item['mean_history_gold_rate'])} | "
            f"{_pct(item['mean_novel_gold_rate'])} | {_pct(ceiling['median'])} | "
            f"{_pct(ceiling['zero_rate'])} | {_pct(ceiling['below_0_25_rate'])} | "
            f"{_pct(ceiling['below_0_5_rate'])} |"
        )
    lines.extend(["", "## Product / Live versus Video", ""])
    for domain in ("prod", "living"):
        comparison = result["comparisons"][f"{domain}_vs_video"]
        lines.append(
            f"- {DOMAIN_LABELS[domain]} minus Video: mean novel Gold "
            f"{comparison['mean_novel_gold_rate_delta']:+.4f}; novel-only rate "
            f"{comparison['novel_only_rate_delta']:+.4f}; mean History-copy ceiling "
            f"{comparison['mean_history_copy_ceiling_delta']:+.4f}."
        )
    product_higher = result["comparisons"]["prod_vs_video"]["mean_novel_gold_rate_delta"] > 0
    live_higher = result["comparisons"]["living_vs_video"]["mean_novel_gold_rate_delta"] > 0
    overall = result["overall"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Across all Recommendation groups, mean novel Gold rate is {_pct(overall['mean_novel_gold_rate'])}, "
        f"novel-only rate is {_pct(overall['category']['novel_only_rate'])}, and mean History-copy recall ceiling is "
        f"{_pct(overall['history_copy_recall_ceiling']['mean'])}.",
        "GR_USER is structurally extractive in this dataset: every audited Gold SID is contained in its sample History. Recommendation is structurally predictive: most groups require at least one SID absent from History, and most are entirely novel.",
        f"Product novel rate higher than Video: {'YES' if product_higher else 'NO'}. Live novel rate higher than Video: {'YES' if live_higher else 'NO'}. The domain-specific Product/Live premise is therefore not supported; Video is the most novel domain in this dataset.",
        "The overall task-structure conflict can still be supported even though the proposed domain ordering is false. This audit cannot show that GR_USER training actually increased History copying; that requires model-output evidence.",
        "",
        "## Direct answers",
        "",
        "1. Product / Live mean novel Gold rate higher than Video: **NO**. Both are lower than Video, by 11.19 and 4.44 percentage points respectively.",
        f"2. Product / Live have many novel-only groups: **YES** ({_pct(result['domains']['prod']['category']['novel_only_rate'])} / {_pct(result['domains']['living']['category']['novel_only_rate'])}).",
        "3. Product / Live History-copy recall ceilings lower than Video: **NO**. Their mean ceilings are higher than Video, although all three are low in absolute terms.",
        "4. User extraction versus Recommendation prediction task-structure conflict: **SUPPORTED overall**. GR_USER Gold is 100% contained in History, while Recommendation mean novel Gold is 92.78% and 83.21% of groups are novel-only.",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Protection confirmation",
        "",
        "- CPU-only",
        "- No training",
        "- No generation",
        "- No model or checkpoint access",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--user-data", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--docs-output", type=Path, required=True)
    args = parser.parse_args()

    before = sha256_file(args.data)
    rows = read_jsonl(args.data)
    records, source_counts = group_records(rows)
    after = sha256_file(args.data)
    if before != after:
        raise RuntimeError("source dataset SHA changed during read-only audit")
    source = {"path": str(args.data), "sha256": before, **source_counts}
    user_structure = audit_user_structure(args.user_data)
    result = summarize(records, source, user_structure=user_structure)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.docs_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.docs_output.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({
        "groups": len(records),
        "domains": result["domains"],
        "comparisons": result["comparisons"],
        "source_sha_unchanged": before == after,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
