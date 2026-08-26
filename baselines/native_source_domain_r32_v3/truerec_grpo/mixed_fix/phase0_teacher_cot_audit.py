"""CPU-only Teacher-CoT feasibility audit for TrueRec Mixed-Fix V1.

This module deliberately has no torch, transformers, model, or trainer imports.
It joins the frozen Curriculum2048 records to original BATA recommendation CoT
rows by exact recommendation_group_id and fails closed on any provenance issue.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
from typing import Any, Iterable


DOMAINS = ("video", "prod", "ad", "living")
STATUSES = ("UNIQUE", "MISSING", "AMBIGUOUS", "INVALID_NO_THINK_CLOSE")
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
DEFAULT_SOURCE = Path("/data/lf_data/onereason_recommendation_cot.jsonl")
DEFAULT_LINEAGE = Path("/data/lf_data_versions/task_pools/懂推荐/beta版/recommendation_beta.jsonl")
DEFAULT_CURRICULUM = Path("/data/GRPO/truerec_grpo/data/curriculum2048_v2/records.jsonl")
DEFAULT_SPLITS = Path("/data/GRPO/truerec_grpo/data/splits")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/phase0_teacher_cot_audit"


class AuditError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_lineage_teacher_rows(source: Path, lineage: Path) -> Iterable[dict[str, Any]]:
    """Attach immutable group metadata to the row-aligned original CoT source.

    The original registered file predates aux_metadata_json. The BATA task-pool
    manifest declares its recommendation file a byte-preserving row projection.
    We still independently gate every row on history SID equality and answer
    current-gold containment before accepting the positional lineage.
    """
    with source.open(encoding="utf-8") as source_handle, lineage.open(encoding="utf-8") as lineage_handle:
        for row_number, pair in enumerate(zip_longest(source_handle, lineage_handle), 1):
            source_line, lineage_line = pair
            if source_line is None or lineage_line is None:
                raise AuditError(f"SOURCE_LINEAGE_ROW_COUNT_MISMATCH_AT={row_number}")
            raw, projected = json.loads(source_line), json.loads(lineage_line)
            try:
                metadata = json.loads(projected.get("aux_metadata_json") or "{}")
            except (TypeError, json.JSONDecodeError) as exc:
                raise AuditError(f"SOURCE_LINEAGE_METADATA_INVALID_AT={row_number}") from exc
            raw_user = joined_user(raw)
            projected_user = joined_user(projected)
            if ordered_sids(raw_user) != ordered_sids(projected_user):
                raise AuditError(f"SOURCE_LINEAGE_HISTORY_MISMATCH_AT={row_number}")
            current_gold = str(metadata.get("recommendation_current_gold_sid", ""))
            suffix = str(raw.get("output", "")).split("</think>", 1)[-1]
            if not current_gold or current_gold not in suffix:
                raise AuditError(f"SOURCE_LINEAGE_CURRENT_GOLD_MISMATCH_AT={row_number}")
            yield {
                "source_segment": "recommendation_cot",
                "system": projected.get("system", ""),
                "instruction": raw_user.rstrip() + "/think",
                "input": "",
                "output": raw.get("output", ""),
                "aux_metadata_json": projected["aux_metadata_json"],
                "_lineage_row_number": row_number,
            }


def load_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item.get("recommendation_group_id")) if isinstance(item, dict) else str(item)
        for item in payload
    }


def joined_user(row: dict[str, Any]) -> str:
    return "\n".join(str(value) for value in (row.get("instruction", ""), row.get("input", "")) if value)


def ordered_sids(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in SID_RE.finditer(text)))


def extract_teacher_cot(output: str) -> str | None:
    """Return assistant start through the first closing think tag, inclusive."""
    close = output.find("</think>")
    return None if close < 0 else output[: close + len("</think>")]


def cot_status(cot_rows: list[dict[str, Any]]) -> tuple[str, str | None, int]:
    if not cot_rows:
        return "MISSING", None, 0
    extracted = [extract_teacher_cot(str(row.get("output", ""))) for row in cot_rows]
    invalid = sum(value is None for value in extracted)
    if invalid:
        return "INVALID_NO_THINK_CLOSE", None, invalid
    variants = set(extracted)
    if len(variants) != 1:
        return "AMBIGUOUS", None, 0
    return "UNIQUE", next(iter(variants)), 0


def hierarchy_key(record: dict[str, Any]) -> str:
    return str(record.get("hierarchy_class") or "UNKNOWN")


def empty_coverage() -> dict[str, Any]:
    return {status: 0 for status in STATUSES} | {"total": 0, "ready": 0}


def add_coverage(target: dict[str, Any], status: str, ready: bool) -> None:
    target["total"] += 1
    target[status] += 1
    target["ready"] += int(ready)


def context_snippet(cot: str, match: str, radius: int = 80) -> str:
    pos = cot.find(match)
    return cot[max(0, pos - radius) : pos + len(match) + radius]


def audit_records(
    curriculum: list[dict[str, Any]],
    source_rows: Iterable[dict[str, Any]],
    split_ids: dict[str, set[str]],
    *,
    source_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_id = {str(row["recommendation_group_id"]): row for row in curriculum}
    if len(by_id) != len(curriculum):
        raise AuditError("CURRICULUM_GROUP_IDS_NOT_UNIQUE")
    source_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if row.get("source_segment") != "recommendation_cot":
            continue
        try:
            metadata = json.loads(row.get("aux_metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        group_id = str(metadata.get("recommendation_group_id", ""))
        if group_id in by_id:
            copied = dict(row)
            copied["_metadata"] = metadata
            source_by_id[group_id].append(copied)

    selected_ids = set(by_id)
    overlaps = {
        name: len(selected_ids & ids)
        for name, ids in split_ids.items()
        if name in {"dev", "final", "probe"}
    }
    train_missing = len(selected_ids - split_ids.get("train", set()))
    overlap_clean = not any(overlaps.values()) and train_missing == 0

    coverage = {
        "overall": empty_coverage(),
        "domain": {domain: empty_coverage() for domain in DOMAINS},
        "stage": defaultdict(empty_coverage),
        "hierarchy": defaultdict(empty_coverage),
        "stage_hierarchy": defaultdict(lambda: defaultdict(empty_coverage)),
    }
    conflict_counts = Counter()
    leak_groups: list[dict[str, Any]] = []
    problem_groups: list[dict[str, Any]] = []
    eligible: list[str] = []
    group_results: list[dict[str, Any]] = []

    for group_id, record in by_id.items():
        rows = source_by_id.get(group_id, [])
        status, cot, invalid_rows = cot_status(rows)
        conflicts: set[str] = set()

        no_think_user = str(record.get("user_content_nothink", ""))
        domain = str(record.get("target_domain", ""))
        gold = set(map(str, record.get("all_gold_sids", [])))
        history = list(map(str, record.get("history_sids", [])))
        if not no_think_user.endswith("/no_think"):
            conflicts.add("route")
        if record.get("fixed_domain_token") != f"<|{domain}_begin|>" or domain not in DOMAINS:
            conflicts.add("domain")
        if not gold or any(not sid.startswith(f"<|{domain}_begin|>") for sid in gold):
            conflicts.add("gold")

        for row in rows:
            metadata = row["_metadata"]
            user = joined_user(row)
            source_gold = set(map(str, metadata.get("recommendation_all_gold_sids", [])))
            current_gold = str(metadata.get("recommendation_current_gold_sid", ""))
            if not user.endswith("/think"):
                conflicts.add("route")
            if (source_gold and source_gold != gold) or current_gold not in gold:
                conflicts.add("gold")
            source_domains = {match.group(1) for sid in source_gold for match in [SID_RE.fullmatch(sid)] if match}
            if source_domains != {domain}:
                conflicts.add("domain")
            if ordered_sids(user) != history:
                conflicts.add("history")

        matched_leaks: list[str] = []
        if cot is not None:
            metadata_gold = set()
            for row in rows:
                metadata_gold.update(map(str, row["_metadata"].get("recommendation_all_gold_sids", [])))
                current = str(row["_metadata"].get("recommendation_current_gold_sid", ""))
                if current:
                    metadata_gold.add(current)
            # A Gold SID may also be historical evidence. Its mere occurrence in
            # the CoT is not target leakage; flag only non-history Gold SIDs.
            matched_leaks = sorted(sid for sid in metadata_gold if sid and sid not in history and sid in cot)
            if matched_leaks:
                leak_groups.append({
                    "recommendation_group_id": group_id,
                    "domain": domain,
                    "stage": record.get("stage"),
                    "hierarchy_class": hierarchy_key(record),
                    "matched_sids": matched_leaks,
                    "context_snippets": [context_snippet(cot, sid) for sid in matched_leaks[:3]],
                })

        for name in conflicts:
            conflict_counts[name] += 1
        ready = status == "UNIQUE" and not conflicts and not matched_leaks and overlap_clean
        add_coverage(coverage["overall"], status, ready)
        add_coverage(coverage["domain"][domain], status, ready)
        add_coverage(coverage["stage"][str(record.get("stage"))], status, ready)
        add_coverage(coverage["hierarchy"][hierarchy_key(record)], status, ready)
        add_coverage(coverage["stage_hierarchy"][str(record.get("stage"))][hierarchy_key(record)], status, ready)
        if ready:
            eligible.append(group_id)
        reasons = ([status] if status != "UNIQUE" else []) + sorted(f"{name.upper()}_CONFLICT" for name in conflicts)
        if matched_leaks:
            reasons.append("COT_GOLD_SID_LEAK_SUSPECT")
        if not ready:
            problem_groups.append({
                "recommendation_group_id": group_id,
                "domain": domain,
                "stage": record.get("stage"),
                "hierarchy_class": hierarchy_key(record),
                "status": status,
                "source_cot_rows": len(rows),
                "invalid_no_think_close_rows": invalid_rows,
                "reasons": reasons,
            })
        group_results.append({
            "recommendation_group_id": group_id,
            "domain": domain,
            "stage": record.get("stage"),
            "hierarchy_class": hierarchy_key(record),
            "status": status,
            "source_cot_rows": len(rows),
            "ready": ready,
        })

    stage1_a = coverage["stage_hierarchy"].get("stage1", {}).get("A_RICH", empty_coverage())
    all_a = coverage["hierarchy"].get("A_RICH", empty_coverage())
    direct_reuse = len(eligible) == len(curriculum) and overlap_clean
    return {
        "contract": {
            "join_key": "aux_metadata_json.recommendation_group_id",
            "join_method": "exact equality; no fuzzy matching",
            "teacher_cot_slice": "assistant output start through first </think>, inclusive",
            "teacher_cot_suffix_included": False,
            "gpu_started": False,
            "model_loaded": False,
            "generation_started": False,
            "backward_started": False,
            "optimizer_steps": 0,
        },
        "source": source_identity or {},
        "counts": {
            "curriculum_groups": len(curriculum),
            "teacher_cot_unique": coverage["overall"]["UNIQUE"],
            "teacher_cot_missing": coverage["overall"]["MISSING"],
            "teacher_cot_ambiguous": coverage["overall"]["AMBIGUOUS"],
            "teacher_cot_invalid_no_think_close": coverage["overall"]["INVALID_NO_THINK_CLOSE"],
            "eligible": len(eligible),
            "problem": len(problem_groups),
        },
        "coverage": {
            "overall": coverage["overall"],
            "domain": coverage["domain"],
            "stage": dict(coverage["stage"]),
            "hierarchy": dict(coverage["hierarchy"]),
            "stage_hierarchy": {key: dict(value) for key, value in coverage["stage_hierarchy"].items()},
        },
        "special_coverage": {
            "stage1_a_rich_total": stage1_a["total"],
            "stage1_a_rich_teacher_cot_ready": stage1_a["ready"],
            "all_a_rich_total": all_a["total"],
            "all_a_rich_teacher_cot_ready": all_a["ready"],
        },
        "split_audit": {
            "train_missing": train_missing,
            "train_dev_overlap": overlaps.get("dev", 0),
            "train_final_overlap": overlaps.get("final", 0),
            "train_probe_overlap": overlaps.get("probe", 0),
            "pass": overlap_clean,
        },
        "conflicts": {name: conflict_counts[name] for name in ("route", "domain", "gold", "history")},
        "cot_gold_sid_leak_suspect_count": len(leak_groups),
        "cot_gold_sid_leak_suspects": leak_groups,
        "curriculum2048_direct_reuse": direct_reuse,
        "eligible_group_ids": sorted(eligible),
        "problem_groups": problem_groups,
        "groups": group_results,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_review(report: dict[str, Any], source_commit: str) -> str:
    c, s, x = report["counts"], report["special_coverage"], report["split_audit"]
    conflicts = report["conflicts"]
    values = {
        "SOURCE_MAIN_COMMIT": source_commit,
        "CURRICULUM2048_GROUPS": c["curriculum_groups"],
        "TEACHER_COT_UNIQUE": c["teacher_cot_unique"],
        "TEACHER_COT_MISSING": c["teacher_cot_missing"],
        "TEACHER_COT_AMBIGUOUS": c["teacher_cot_ambiguous"],
        "TEACHER_COT_INVALID_NO_THINK_CLOSE": c["teacher_cot_invalid_no_think_close"],
        "STAGE1_A_RICH_TOTAL": s["stage1_a_rich_total"],
        "STAGE1_A_RICH_TEACHER_COT_READY": s["stage1_a_rich_teacher_cot_ready"],
        "ALL_A_RICH_TOTAL": s["all_a_rich_total"],
        "ALL_A_RICH_TEACHER_COT_READY": s["all_a_rich_teacher_cot_ready"],
        "TRAIN_DEV_OVERLAP": x["train_dev_overlap"],
        "TRAIN_FINAL_OVERLAP": x["train_final_overlap"],
        "TRAIN_PROBE_OVERLAP": x["train_probe_overlap"],
        "ROUTE_CONFLICT_GROUPS": conflicts["route"],
        "DOMAIN_CONFLICT_GROUPS": conflicts["domain"],
        "GOLD_CONFLICT_GROUPS": conflicts["gold"],
        "HISTORY_CONFLICT_GROUPS": conflicts["history"],
        "COT_GOLD_SID_LEAK_SUSPECT_COUNT": report["cot_gold_sid_leak_suspect_count"],
        "CURRICULUM2048_DIRECT_REUSE": "YES" if report["curriculum2048_direct_reuse"] else "NO",
        "GPU_STARTED": "NO", "MODEL_LOADED": "NO", "GENERATION_STARTED": "NO",
        "BACKWARD_STARTED": "NO", "OPTIMIZER_STEPS": 0,
    }
    return "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"


def run(source: Path, lineage: Path, curriculum_path: Path, split_dir: Path, output_dir: Path, source_commit: str) -> dict[str, Any]:
    curriculum = load_jsonl(curriculum_path)
    splits = {
        "train": load_ids(split_dir / "train_pool_group_ids.json"),
        "dev": load_ids(split_dir / "dev_group_ids.json"),
        "final": load_ids(split_dir / "final_group_ids.json"),
        "probe": load_ids(split_dir / "probe20_group_ids.json"),
    }
    report = audit_records(
        curriculum, iter_lineage_teacher_rows(source, lineage), splits,
        source_identity={
            "path": str(source), "sha256": file_sha256(source),
            "lineage_path": str(lineage), "lineage_sha256": file_sha256(lineage),
            "lineage_method": "strict row alignment gated by exact ordered history SIDs and current-gold suffix",
        },
    )
    write_json(output_dir / "teacher_cot_audit.json", report)
    write_json(output_dir / "teacher_cot_eligible_group_ids.json", report["eligible_group_ids"])
    write_json(output_dir / "teacher_cot_problem_groups.json", report["problem_groups"])
    (output_dir / "REVIEW.txt").write_text(render_review(report, source_commit), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--lineage", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    report = run(args.source, args.lineage, args.curriculum, args.split_dir, args.output_dir, args.source_commit)
    print(render_review(report, args.source_commit), end="")


if __name__ == "__main__":
    main()
