#!/usr/bin/env python3
"""Build Exp-A Strict-Clean three-task data without mutating baseline files."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAIN_ROOT = Path("/data/LLm-8B/code/train")
SOURCE = TRAIN_ROOT / "data/material140k_userrec_full_v1"
OUTPUT = TRAIN_ROOT / "data/material140k_expA_strict_three_task_v1"
REPORT_DIR = TRAIN_ROOT / "reports/material140k_expA_strict_three_task_v1"
SEED = 20260806

MATERIAL_FILES = (
    "material_no_think_semantic_to_sid.jsonl",
    "material_no_think_sid_to_semantic.jsonl",
    "material_think_semantic_to_sid.jsonl",
    "material_think_sid_to_semantic.jsonl",
)
TRUNCATION_TAILS = frozenset({"(", "（", "、", "如", ",", "，", "`"})
SID_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_\d+><s_b_\d+><s_c_\d+>"
)
DATE_RE = re.compile(r"^【(\d{4}-\d{2}-\d{2})】$")
TIMELINE_ACTION_RE = re.compile(r"^\s*(?:\d{2}:\d{2}|--:--)\s+\[([^\]]+)\]\s+(.+?)\s*$")
EVENT_ACTION_RE = re.compile(r"^\[([^\]]+)\]\s+(.+?)\s*$")

R1_SYSTEM = (
    "你是一个推荐系统推理助手，擅长根据用户的多域真实历史行为，归纳用户兴趣偏好、"
    "识别近期行为信号并分析兴趣演化。推理必须以给定历史行为为依据，不得臆造用户行为或"
    "未来目标。请仅输出推荐推理过程，不输出最终物料 SID。"
)
DOMAIN_LABEL = {"video": "视频", "prod": "商品", "ad": "广告", "living": "直播"}
R1_INSTRUCTION = {
    domain: (
        f"请根据以上多域历史行为，分析用户在{label}场景下的当前兴趣偏好、近期需求及兴趣演化，"
        "并归纳最可能的下一步兴趣方向。请仅输出推理过程，不要输出最终推荐 SID。/think"
    )
    for domain, label in DOMAIN_LABEL.items()
}
R2_SYSTEM = {
    "video": "你需要根据用户真实历史行为形成的兴趣归纳，预测用户未来最可能交互的视频 SID。",
    "prod": "你需要根据用户真实历史行为形成的兴趣归纳，预测用户未来最可能交互的商品 SID。",
    "ad": "你需要根据用户真实历史行为形成的兴趣归纳，预测用户未来最可能交互的广告 SID。",
    "living": "你需要根据用户真实历史行为形成的兴趣归纳，预测用户未来最可能交互的直播 SID。",
}

# 42,900 is close to the strict-clean population (42,922) and permits an exact
# 20:20:15:45 task split with an exact 10:1:1:1 domain split inside every task.
SAMPLED_TOTAL = 42_900
TASK_TOTALS = {"U": 8_580, "R1": 8_580, "R2": 6_435, "R3": 19_305}
DOMAIN_MULTIPLIERS = {"video": 10, "ad": 1, "prod": 1, "living": 1}


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_record(row: dict[str, Any]) -> dict[str, str]:
    value = {key: row[key] for key in ("system", "prompt", "response")}
    if any(not isinstance(value[key], str) for key in value):
        raise ValueError("system/prompt/response must be strings")
    return value


def split_response(response: str, row_id: str) -> tuple[str, str]:
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        raise ValueError(f"{row_id}: response must contain one complete think block")
    thought, answer = response.split("</think>", 1)
    if not thought.startswith("<think>"):
        raise ValueError(f"{row_id}: response must start with <think>")
    thought = thought[len("<think>") :].strip()
    answer = answer.strip()
    if not thought or not answer:
        raise ValueError(f"{row_id}: empty thought or final answer")
    return thought, answer


def extract_target(answer: str, row_id: str) -> tuple[str, str]:
    matches = list(SID_RE.finditer(answer))
    if len(matches) != 1:
        raise ValueError(f"{row_id}: expected one final target SID, found {len(matches)}")
    return matches[0].group(0), matches[0].group("domain")


def replace_last_instruction(prompt: str, instruction: str) -> str:
    stripped = prompt.rstrip()
    if not stripped.endswith("/think"):
        raise ValueError("recommendation prompt must end with /think")
    head, separator, _last_line = stripped.rpartition("\n")
    if not separator:
        raise ValueError("recommendation prompt has no replaceable final instruction line")
    return head + "\n" + instruction


def to_no_think_prompt(prompt: str) -> str:
    stripped = prompt.rstrip()
    if not stripped.endswith("/think"):
        raise ValueError("recommendation prompt must end with /think")
    return stripped[: -len("/think")] + "/no_think"


def parse_timeline(prompt: str) -> list[tuple[str, str, str]]:
    current_date: str | None = None
    events: list[tuple[str, str, str]] = []
    for line in prompt.splitlines():
        date_match = DATE_RE.match(line.strip())
        if date_match:
            current_date = date_match.group(1)
            continue
        action_match = TIMELINE_ACTION_RE.match(line)
        if current_date and action_match:
            events.append((current_date, action_match.group(1).strip(), action_match.group(2).strip()))
    return events


def is_chain_record(row: dict[str, str]) -> bool:
    text = row["prompt"] + "\n" + row["response"]
    return "logic_chain" in text or "行为逻辑链" in text


def parse_chain_json(response: str) -> dict[str, Any]:
    tail = response.split("</think>", 1)[1].strip() if "</think>" in response else response.strip()
    value = json.loads(tail)
    chain = value["logic_chain"]
    if not isinstance(chain, dict) or not isinstance(chain.get("name"), str):
        raise ValueError("missing logic_chain.name")
    events = chain.get("events")
    if not isinstance(events, list):
        raise ValueError("logic_chain.events is not a list")
    for event in events:
        if not isinstance(event, dict) or any(key not in event for key in ("date", "action", "logic")):
            raise ValueError("event missing date/action/logic")
        if any(not isinstance(event[key], str) for key in ("date", "action", "logic")):
            raise ValueError("event date/action/logic must be strings")
    return value


def chain_error_reason(row: dict[str, str]) -> str | None:
    try:
        value = parse_chain_json(row["response"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return "invalid_json"

    timeline = parse_timeline(row["prompt"])
    exact = set(timeline)
    by_object: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for date, action_type, obj in timeline:
        by_object[obj].append((date, action_type))

    for event in value["logic_chain"]["events"]:
        match = EVENT_ACTION_RE.match(event["action"].strip())
        if not match:
            return "invalid_json"
        date = event["date"].strip()
        action_type = match.group(1).strip()
        obj = match.group(2).strip()
        if (date, action_type, obj) in exact:
            continue
        candidates = by_object.get(obj, [])
        date_matches = any(candidate_date == date for candidate_date, _ in candidates)
        action_matches = any(candidate_action == action_type for _, candidate_action in candidates)
        if action_matches and not date_matches:
            return "date_mismatch"
        if date_matches and not action_matches:
            return "action_type_mismatch"
        return "date_and_action_mismatch"
    return None


def build_user(temporary: Path) -> dict[str, Any]:
    kept: list[dict[str, str]] = []
    removed: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    raw_chain_count = 0
    for filename in ("user_think.jsonl", "user_no_think.jsonl"):
        with (SOURCE / filename).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = training_record(json.loads(line))
                if not is_chain_record(row):
                    kept.append(row)
                    continue
                raw_chain_count += 1
                reason = chain_error_reason(row)
                if reason is None:
                    kept.append(row)
                else:
                    counts[reason] += 1
                    removed.append(
                        {
                            "source_file": filename,
                            "source_line": line_number,
                            "reason": reason,
                            "record": row,
                        }
                    )
    dump_jsonl(temporary / "user_clean.jsonl", kept)
    dump_jsonl(temporary / "user_chain_removed.jsonl", removed)
    report = {
        "raw_rows": 8_339 + 24_509,
        "output_rows": len(kept),
        "raw_chain_count": raw_chain_count,
        "kept_chain_count": raw_chain_count - len(removed),
        "removed_chain_count": len(removed),
        "invalid_json_count": counts["invalid_json"],
        "date_mismatch_count": counts["date_mismatch"],
        "action_type_mismatch_count": counts["action_type_mismatch"],
        "date_and_action_mismatch_count": counts["date_and_action_mismatch"],
    }
    (temporary / "user_chain_clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def canonical_quality(row: dict[str, Any]) -> tuple[int, int]:
    return (len(row["cot_text"]), -int(row["source_line"]))


def build_recommendation(temporary: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_path = SOURCE / "recommendation_think.jsonl"
    parsed: list[dict[str, Any]] = []
    prompt_counts_raw: Counter[str] = Counter()
    prompt_targets_raw: defaultdict[str, set[str]] = defaultdict(set)
    target_in_prompt_raw = 0
    target_only_in_cot_raw = 0
    overlap = 0

    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = training_record(json.loads(line))
            row_id = f"recommendation:{line_number}"
            cot, answer = split_response(row["response"], row_id)
            target, domain = extract_target(answer, row_id)
            prompt_key = row["prompt"].strip()
            truncated = cot[-1] in TRUNCATION_TAILS
            in_prompt = target in row["prompt"]
            in_cot = target in cot
            prompt_counts_raw[prompt_key] += 1
            prompt_targets_raw[prompt_key].add(target)
            target_in_prompt_raw += in_prompt
            # TASK_V2's historical field name says "only_in_cot", while its
            # audited count (339) is the exact target-in-CoT count regardless
            # of whether the same target also occurs in history.
            target_only_in_cot_raw += in_cot
            overlap += truncated and in_prompt
            parsed.append(
                {
                    **row,
                    "source_line": line_number,
                    "source_row_id": row_id,
                    "cot_text": cot,
                    "final_answer_tail": answer,
                    "target_sid": target,
                    "target_domain": domain,
                    "prompt_key": prompt_key,
                    "is_truncated": truncated,
                    "is_target_in_history": in_prompt,
                    "target_in_cot": in_cot,
                }
            )

    raw_repeat_keys = sum(count >= 2 for count in prompt_counts_raw.values())
    raw_repeat_rows = sum(count for count in prompt_counts_raw.values() if count >= 2)
    raw_stats = {
        "raw_rows": len(parsed),
        "truncated_removed": sum(row["is_truncated"] for row in parsed),
        "target_in_prompt_raw": target_in_prompt_raw,
        "target_in_prompt_overlap_truncated": overlap,
        "target_in_prompt_after_truncation": sum(
            row["is_target_in_history"] and not row["is_truncated"] for row in parsed
        ),
        "raw_unique_prompt_keys": len(prompt_counts_raw),
        "raw_repeat_prompt_keys": raw_repeat_keys,
        "raw_repeat_rows": raw_repeat_rows,
        "raw_extra_repeat_rows": sum(count - 1 for count in prompt_counts_raw.values()),
        "raw_multi_sid_repeat_prompt_keys": sum(
            prompt_counts_raw[key] >= 2 and len(targets) >= 2
            for key, targets in prompt_targets_raw.items()
        ),
        "raw_max_prompt_group_size": max(prompt_counts_raw.values()),
        "target_sid_only_in_cot_raw": target_only_in_cot_raw,
    }
    expected = {
        "raw_rows": 48_269,
        "truncated_removed": 3_680,
        "target_in_prompt_raw": 1_784,
        "target_in_prompt_overlap_truncated": 117,
        "target_in_prompt_after_truncation": 1_667,
        "raw_unique_prompt_keys": 16_623,
        "raw_repeat_prompt_keys": 9_010,
        "raw_repeat_rows": 40_656,
        "raw_extra_repeat_rows": 31_646,
        "raw_multi_sid_repeat_prompt_keys": 8_911,
        "raw_max_prompt_group_size": 23,
        "target_sid_only_in_cot_raw": 339,
    }
    if raw_stats != expected:
        raise RuntimeError(f"raw recommendation statistics mismatch: {raw_stats} != {expected}")

    quarantine = [row for row in parsed if not row["is_truncated"] and row["is_target_in_history"]]
    strict = [row for row in parsed if not row["is_truncated"] and not row["is_target_in_history"]]
    if len(strict) != 42_922 or len(quarantine) != 1_667:
        raise RuntimeError("strict-clean or quarantine row count mismatch")

    # Deduplicate only identical (prompt, final target) supervision and retain
    # the row with the most complete CoT, then regroup on the canonical rows.
    by_prompt_target: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in strict:
        by_prompt_target[(row["prompt_key"], row["target_sid"])].append(row)
    canonical = [max(group, key=canonical_quality) for group in by_prompt_target.values()]
    canonical.sort(key=lambda row: row["source_line"])
    prompt_counts_clean = Counter(row["prompt_key"] for row in canonical)
    prompt_targets_clean: defaultdict[str, set[str]] = defaultdict(set)
    for row in canonical:
        prompt_targets_clean[row["prompt_key"]].add(row["target_sid"])

    unique_rows = [row for row in canonical if prompt_counts_clean[row["prompt_key"]] == 1]
    repeat_rows = [row for row in canonical if prompt_counts_clean[row["prompt_key"]] >= 2]
    anchors: list[dict[str, str]] = []
    r1_rows: list[dict[str, str]] = []
    r2_rows: list[dict[str, str]] = []
    r3_rows: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []

    for row in canonical:
        group_id = hashlib.sha256(row["prompt_key"].encode()).hexdigest()[:16]
        common_meta = {
            "source_row_id": row["source_row_id"],
            "source_line": row["source_line"],
            "target_sid": row["target_sid"],
            "target_domain": row["target_domain"],
            "prompt_group_id": group_id,
            "is_repeat_prompt": prompt_counts_clean[row["prompt_key"]] >= 2,
            "is_target_in_history": False,
        }
        if not common_meta["is_repeat_prompt"]:
            anchors.append(training_record(row))
            metadata.append({**common_meta, "task_type": "U"})
            continue

        clean_cot = row["cot_text"].replace(row["target_sid"], "相关目标内容")
        r1 = {
            "system": R1_SYSTEM,
            "prompt": replace_last_instruction(
                row["prompt"], R1_INSTRUCTION[row["target_domain"]]
            ),
            "response": f"<think>{clean_cot}</think>",
        }
        cot_for_sid = row["cot_text"].replace(row["target_sid"], "[目标物料已遮蔽]")
        r2 = {
            "system": R2_SYSTEM[row["target_domain"]],
            "prompt": cot_for_sid.rstrip() + "/no_think",
            "response": f"<think>\n</think>\n{row['final_answer_tail']}",
        }
        r3 = {
            "system": row["system"],
            "prompt": to_no_think_prompt(row["prompt"]),
            "response": f"<think>\n</think>\n{row['final_answer_tail']}",
        }
        if row["target_sid"] in r1["response"] or row["target_sid"] in r2["prompt"]:
            raise RuntimeError(f"target leak survived R1/R2 for {row['source_row_id']}")
        r1_rows.append(r1)
        r2_rows.append(r2)
        r3_rows.append(r3)
        metadata.extend(
            {**common_meta, "task_type": task_type}
            for task_type in ("R1", "R2", "R3")
        )

    dump_jsonl(temporary / "rec_unique_anchor.jsonl", anchors)
    dump_jsonl(temporary / "rec_repeat_history_to_cot.jsonl", r1_rows)
    dump_jsonl(temporary / "rec_repeat_cot_to_sid.jsonl", r2_rows)
    dump_jsonl(temporary / "rec_repeat_history_to_sid_nothink.jsonl", r3_rows)
    dump_jsonl(
        temporary / "rec_repeat_signal_quarantine.jsonl",
        [
            {
                "source_row_id": row["source_row_id"],
                "target_sid": row["target_sid"],
                "target_domain": row["target_domain"],
                "reason": "final_target_sid_in_prompt_after_truncation_filter",
                "record": training_record(row),
            }
            for row in quarantine
        ],
    )
    dump_jsonl(temporary / "rec_metadata.jsonl", metadata)

    clean_stats = {
        "strict_clean_rows": len(strict),
        "canonical_prompt_sid_rows": len(canonical),
        "duplicate_prompt_sid_rows_removed": len(strict) - len(canonical),
        "clean_unique_prompt_rows": len(unique_rows),
        "clean_repeat_prompt_keys": sum(count >= 2 for count in prompt_counts_clean.values()),
        "clean_repeat_rows": len(repeat_rows),
        "clean_multi_sid_prompt_keys": sum(
            len(targets) >= 2 for targets in prompt_targets_clean.values()
        ),
        "clean_prompt_sid_pairs": len(by_prompt_target),
        "R1_rows": len(r1_rows),
        "R2_rows": len(r2_rows),
        "R3_rows": len(r3_rows),
        "R2_target_leak_before_mask": sum(row["target_in_cot"] for row in repeat_rows),
        "R2_target_leak_after_mask": 0,
        "video_rows": sum(row["target_domain"] == "video" for row in canonical),
        "ad_rows": sum(row["target_domain"] == "ad" for row in canonical),
        "product_rows": sum(row["target_domain"] == "prod" for row in canonical),
        "live_rows": sum(row["target_domain"] == "living" for row in canonical),
    }
    stats = {**raw_stats, **clean_stats}
    (temporary / "rec_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pools: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for task, rows_source, source_meta in (
        ("U", anchors, [row for row in canonical if prompt_counts_clean[row["prompt_key"]] == 1]),
        ("R1", r1_rows, repeat_rows),
        ("R2", r2_rows, repeat_rows),
        ("R3", r3_rows, repeat_rows),
    ):
        if len(rows_source) != len(source_meta):
            raise RuntimeError(f"{task}: derived rows and metadata are misaligned")
        for derived, original in zip(rows_source, source_meta):
            pools[(task, original["target_domain"])].append(derived)

    rng = random.Random(SEED)
    sampled: list[dict[str, str]] = []
    sampling_counts: dict[str, dict[str, int]] = {}
    for task, task_total in TASK_TOTALS.items():
        unit = task_total // 13
        if unit * 13 != task_total:
            raise RuntimeError(f"task total is not divisible by 13: {task}")
        sampling_counts[task] = {}
        for domain, multiplier in DOMAIN_MULTIPLIERS.items():
            quota = unit * multiplier
            source_pool = pools[(task, domain)]
            if not source_pool:
                raise RuntimeError(f"empty sampling pool for {task}/{domain}")
            shuffled = list(source_pool)
            rng.shuffle(shuffled)
            selected = [shuffled[index % len(shuffled)] for index in range(quota)]
            sampled.extend(selected)
            sampling_counts[task][domain] = quota
    rng.shuffle(sampled)
    if len(sampled) != SAMPLED_TOTAL:
        raise RuntimeError(f"sampled total mismatch: {len(sampled)}")
    dump_jsonl(temporary / "rec_train_expA_strict_clean.jsonl", sampled)
    return {**stats, "sampled_rows": len(sampled), "sampling_counts": sampling_counts}, metadata


def verify_derived(temporary: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for filename in (
        "rec_unique_anchor.jsonl",
        "rec_repeat_history_to_cot.jsonl",
        "rec_repeat_cot_to_sid.jsonl",
        "rec_repeat_history_to_sid_nothink.jsonl",
        "rec_train_expA_strict_clean.jsonl",
    ):
        rows = [json.loads(line) for line in (temporary / filename).open(encoding="utf-8")]
        for row in rows:
            training_record(row)
        checks[filename] = len(rows)
    r1 = [json.loads(line) for line in (temporary / "rec_repeat_history_to_cot.jsonl").open(encoding="utf-8")]
    if any(not row["prompt"].endswith("/think") or not row["response"].endswith("</think>") for row in r1):
        raise RuntimeError("R1 format validation failed")
    r2 = [json.loads(line) for line in (temporary / "rec_repeat_cot_to_sid.jsonl").open(encoding="utf-8")]
    r3 = [json.loads(line) for line in (temporary / "rec_repeat_history_to_sid_nothink.jsonl").open(encoding="utf-8")]
    if any(not row["prompt"].endswith("/no_think") or not row["response"].startswith("<think>\n</think>\n") for row in r2 + r3):
        raise RuntimeError("R2/R3 format validation failed")
    return checks


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing dataset: {OUTPUT}")
    temporary = OUTPUT.with_name(OUTPUT.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for filename in MATERIAL_FILES:
            shutil.copy2(SOURCE / filename, temporary / filename)
        user_report = build_user(temporary)
        rec_report, _metadata = build_recommendation(temporary)
        verification = verify_derived(temporary)
        dataset_info = {
            Path(filename).stem: {
                "file_name": filename,
                "columns": {"prompt": "prompt", "response": "response", "system": "system"},
            }
            for filename in (*MATERIAL_FILES, "user_clean.jsonl", "rec_train_expA_strict_clean.jsonl")
        }
        (temporary / "dataset_info.json").write_text(
            json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        hashes = {
            path.name: sha256(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(SOURCE),
            "output_dataset": str(OUTPUT),
            "seed": SEED,
            "user": user_report,
            "recommendation": rec_report,
            "verification": verification,
            "sha256": hashes,
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(temporary / "user_chain_clean_report.json", REPORT_DIR / "user_chain_clean_report.json")
        shutil.copy2(temporary / "rec_stats.json", REPORT_DIR / "rec_stats.json")
        temporary.rename(OUTPUT)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
