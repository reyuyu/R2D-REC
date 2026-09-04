#!/usr/bin/env python3
"""Build TASK V3 Full/R1/R2 curriculum data with a leakage-safe 95/5 split.

Material and user examples are copied byte-for-byte at the JSON object level from
Exp-A. Recommendation Full examples are copied from the original 48,269-row
source; only the auxiliary R1/R2 pools apply the established truncation filter.
Recommendation train/validation assignment is made at prompt-group granularity.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/data/LLm-8B/code/train")
EXP_A = ROOT / "data/material140k_expA_strict_three_task_v1"
RAW = ROOT / "data/material140k_userrec_full_v1/recommendation_think.jsonl"
OUT = ROOT / "data/material140k_userclean_rec_curriculum_fullft_v3"
REPORT_DIR = ROOT / "reports/material140k_userclean_rec_curriculum_fullft_v3"
SEED = 19260817
VAL_FRACTION = 0.05

MATERIAL_FILES = (
    "material_no_think_semantic_to_sid.jsonl",
    "material_no_think_sid_to_semantic.jsonl",
    "material_think_semantic_to_sid.jsonl",
    "material_think_sid_to_semantic.jsonl",
)
BASE_FILES = MATERIAL_FILES + ("user_clean.jsonl",)
TRUNCATION_TAILS = frozenset({"(", "（", "、", "如", ",", "，", "`"})
SID_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_\d+><s_b_\d+><s_c_\d+>"
)
DOMAIN_LABEL = {"video": "视频", "prod": "商品", "ad": "广告", "living": "直播"}

R1_SYSTEM = (
    "你是一个推荐系统推理助手，擅长根据用户的多域真实历史行为，归纳用户兴趣偏好、识别近期行为信号并分析兴趣演化。"
    "推理必须严格以给定历史行为为依据，不得臆造用户行为、物料或未来目标。"
    "请仅输出用于后续推荐决策的推理过程，不输出最终推荐 SID。"
)
R1_INSTRUCTION = {
    domain: (
        f"请根据以上多域历史行为，分析用户在{label}场景下的当前兴趣偏好、近期需求和兴趣演化过程，"
        "并归纳最可能的下一步兴趣方向。请仅输出推理过程，不要输出最终推荐 SID。/think"
    )
    for domain, label in DOMAIN_LABEL.items()
}
R2_SYSTEM = {
    "video": "你是一个推荐系统决策助手。请根据由用户真实历史行为归纳得到的兴趣偏好、近期需求和兴趣演化信息，预测用户未来最可能交互的视频 SID。请只输出对应的视频 SID，不重新生成用户画像，不补充额外解释。",
    "prod": "你是一个推荐系统决策助手。请根据由用户真实历史行为归纳得到的兴趣偏好、近期需求和兴趣演化信息，预测用户未来最可能交互的商品 SID。请只输出对应的商品 SID，不重新生成用户画像，不补充额外解释。",
    "ad": "你是一个推荐系统决策助手。请根据由用户真实历史行为归纳得到的兴趣偏好、近期需求和兴趣演化信息，预测用户未来最可能交互的广告 SID。请只输出对应的广告 SID，不重新生成用户画像，不补充额外解释。",
    "living": "你是一个推荐系统决策助手。请根据由用户真实历史行为归纳得到的兴趣偏好、近期需求和兴趣演化信息，预测用户未来最可能交互的直播 SID。请只输出对应的直播 SID，不重新生成用户画像，不补充额外解释。",
}


def stable_digest(text: str) -> str:
    return hashlib.sha256(f"{SEED}\0{text}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def training_record(row: dict[str, Any]) -> dict[str, str]:
    output = {key: row[key] for key in ("system", "prompt", "response")}
    if any(not isinstance(value, str) for value in output.values()):
        raise ValueError("system/prompt/response must all be strings")
    return output


def split_response(response: str, row_id: str) -> tuple[str, str]:
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        raise ValueError(f"{row_id}: expected exactly one complete think block")
    prefix, answer = response.split("</think>", 1)
    if not prefix.startswith("<think>"):
        raise ValueError(f"{row_id}: response must start with <think>")
    cot = prefix[len("<think>") :].strip()
    answer = answer.strip()
    if not cot or not answer:
        raise ValueError(f"{row_id}: empty CoT or answer tail")
    return cot, answer


def extract_target(answer: str, row_id: str) -> tuple[str, str]:
    matches = list(SID_RE.finditer(answer))
    if len(matches) != 1:
        raise ValueError(f"{row_id}: expected one final SID, found {len(matches)}")
    return matches[0].group(0), matches[0].group("domain")


def replace_instruction(prompt: str, instruction: str) -> str:
    prompt = prompt.rstrip()
    if not prompt.endswith("/think"):
        raise ValueError("recommendation prompt must end in /think")
    body, separator, _ = prompt.rpartition("\n")
    if not separator:
        raise ValueError("recommendation prompt has no replaceable last instruction")
    return body + "\n" + instruction


def sanitize_cot(cot: str, target: str) -> str:
    sanitized = cot.replace(target, "[目标物料已遮蔽]").strip()
    if target in sanitized or not sanitized:
        raise ValueError("failed to remove final target SID from CoT")
    return sanitized


def exact_val_keys(keys: list[str], namespace: str) -> set[str]:
    selected = round(len(keys) * VAL_FRACTION)
    ordered = sorted(keys, key=lambda key: stable_digest(f"{namespace}\0{key}"))
    return set(ordered[:selected])


def split_base_file(source: Path, temporary: Path) -> dict[str, Any]:
    rows = load_jsonl(source)
    val_indices = exact_val_keys([str(index) for index in range(len(rows))], source.name)
    train_rows = [row for index, row in enumerate(rows) if str(index) not in val_indices]
    val_rows = [row for index, row in enumerate(rows) if str(index) in val_indices]
    stem = source.stem
    train_path = temporary / f"{stem}_train.jsonl"
    val_path = temporary / f"{stem}_val.jsonl"
    dump_jsonl(train_path, train_rows)
    dump_jsonl(val_path, val_rows)
    if Counter(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) != Counter(
        json.dumps(row, sort_keys=True, ensure_ascii=False) for row in train_rows + val_rows
    ):
        raise AssertionError(f"{source.name}: split changed source examples")
    return {
        "source": str(source),
        "source_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_fraction": len(val_rows) / len(rows),
        "source_sha256": file_sha256(source),
        "train_sha256": file_sha256(train_path),
        "val_sha256": file_sha256(val_path),
        "content_unchanged": True,
    }


def build_recommendation(temporary: Path) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    prompt_counts: Counter[str] = Counter()
    prompt_targets: defaultdict[str, set[str]] = defaultdict(set)
    with RAW.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            original = training_record(json.loads(line))
            cot, answer = split_response(original["response"], f"recommendation:{line_number}")
            target, domain = extract_target(answer, f"recommendation:{line_number}")
            prompt_key = original["prompt"].strip()
            prompt_counts[prompt_key] += 1
            prompt_targets[prompt_key].add(target)
            parsed.append(
                {
                    "original": original,
                    "line": line_number,
                    "prompt_key": prompt_key,
                    "group_id": stable_digest("prompt-group\0" + prompt_key)[:24],
                    "cot": cot,
                    "answer": answer,
                    "target": target,
                    "domain": domain,
                    "truncated": cot[-1] in TRUNCATION_TAILS,
                }
            )

    if len(parsed) != 48_269:
        raise AssertionError(f"raw Full count changed: {len(parsed)} != 48269")
    repeated_keys = {key for key, count in prompt_counts.items() if count >= 2}
    if len(repeated_keys) != 9_010:
        raise AssertionError(f"repeated prompt group count changed: {len(repeated_keys)} != 9010")

    val_prompt_keys = exact_val_keys(list(prompt_counts), "recommendation-prompt-groups")
    full_train = [row["original"] for row in parsed if row["prompt_key"] not in val_prompt_keys]
    full_val = [row["original"] for row in parsed if row["prompt_key"] in val_prompt_keys]
    dump_jsonl(temporary / "rec_full_train.jsonl", full_train)
    dump_jsonl(temporary / "rec_full_val.jsonl", full_val)

    aux_rows = [
        row for row in parsed if row["prompt_key"] in repeated_keys and not row["truncated"]
    ]
    r1_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    r2_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in aux_rows:
        sanitized = sanitize_cot(row["cot"], row["target"])
        r1_prompt = replace_instruction(row["original"]["prompt"], R1_INSTRUCTION[row["domain"]])
        r1_response = f"<think>\n{sanitized}\n</think>"
        if row["target"] in r1_response:
            raise AssertionError("target SID leaked into R1 response")
        r1_key = (row["prompt_key"], sanitized, row["domain"])
        r1_by_key.setdefault(
            r1_key,
            {"system": R1_SYSTEM, "prompt": r1_prompt, "response": r1_response},
        )

        r2_prompt = (
            "以下是根据用户真实多域历史行为归纳得到的兴趣、近期需求与兴趣演化：\n\n"
            f"{sanitized}\n\n"
            f"请根据以上信息预测用户下一步最可能交互的{DOMAIN_LABEL[row['domain']]} SID。/no_think"
        )
        if row["target"] in r2_prompt:
            raise AssertionError("target SID leaked into R2 prompt")
        r2_key = (row["group_id"], sanitized, row["target"])
        r2_by_key.setdefault(
            r2_key,
            {
                "system": R2_SYSTEM[row["domain"]],
                "prompt": r2_prompt,
                "response": f"<think>\n</think>\n{row['answer']}",
                "prompt_group_id": row["group_id"],
                "sanitized_cot": sanitized,
                "final_target_sid": row["target"],
                "_prompt_key": row["prompt_key"],
            },
        )

    group_sids: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for (group_id, sanitized, target) in r2_by_key:
        group_sids[(group_id, sanitized)].add(target)
    r2_rows: list[dict[str, Any]] = []
    for (group_id, sanitized, target), row in r2_by_key.items():
        k = len(group_sids[(group_id, sanitized)])
        row["sample_weight"] = 1.0 / k
        row["group_unique_sid_count"] = k
        r2_rows.append(row)

    r1_train: list[dict[str, str]] = []
    r1_val: list[dict[str, str]] = []
    for (prompt_key, _sanitized, _domain), row in r1_by_key.items():
        (r1_val if prompt_key in val_prompt_keys else r1_train).append(row)
    r2_train: list[dict[str, Any]] = []
    r2_val: list[dict[str, Any]] = []
    for row in r2_rows:
        prompt_key = row.pop("_prompt_key")
        (r2_val if prompt_key in val_prompt_keys else r2_train).append(row)

    dump_jsonl(temporary / "rec_r1_train.jsonl", r1_train)
    dump_jsonl(temporary / "rec_r1_val.jsonl", r1_val)
    dump_jsonl(temporary / "rec_r2_train.jsonl", r2_train)
    dump_jsonl(temporary / "rec_r2_val.jsonl", r2_val)

    train_prompts = {row["prompt_key"] for row in parsed if row["prompt_key"] not in val_prompt_keys}
    val_prompts = {row["prompt_key"] for row in parsed if row["prompt_key"] in val_prompt_keys}
    if train_prompts & val_prompts:
        raise AssertionError("recommendation prompt groups leaked across train/validation")

    original_counter = Counter(
        json.dumps(row["original"], ensure_ascii=False, sort_keys=True) for row in parsed
    )
    split_counter = Counter(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in full_train + full_val
    )
    if original_counter != split_counter:
        raise AssertionError("Full pool does not exactly preserve the raw source")

    group_sizes = Counter(row["prompt_key"] for row in parsed if row["prompt_key"] in repeated_keys)
    r2_group_sizes = Counter((row["prompt_group_id"], row["sanitized_cot"]) for row in r2_rows)
    return {
        "raw_full_think_count": len(parsed),
        "unique_prompt_group_count": len(prompt_counts),
        "repeat_prompt_group_count": len(repeated_keys),
        "repeat_rows_raw": sum(group_sizes.values()),
        "max_raw_prompt_group_size": max(group_sizes.values()),
        "multi_sid_prompt_group_count": sum(len(targets) > 1 for targets in prompt_targets.values()),
        "truncated_rows_raw": sum(row["truncated"] for row in parsed),
        "auxiliary_source_rows_after_truncation_filter": len(aux_rows),
        "validation_prompt_groups": len(val_prompt_keys),
        "validation_prompt_group_fraction": len(val_prompt_keys) / len(prompt_counts),
        "full_train_count": len(full_train),
        "full_val_count": len(full_val),
        "r1_train_count": len(r1_train),
        "r1_val_count": len(r1_val),
        "r2_train_count": len(r2_train),
        "r2_val_count": len(r2_val),
        "r2_group_count": len(r2_group_sizes),
        "r2_max_unique_sid_per_group": max(r2_group_sizes.values()),
        "full_original_text_unchanged": True,
        "train_val_prompt_overlap": 0,
        "r1_target_leak_count": 0,
        "r2_target_leak_count": 0,
        "history_to_sid_artificial_pool_count": 0,
    }


def dataset_info() -> dict[str, Any]:
    names = [
        *(f"{Path(name).stem}_{split}" for name in BASE_FILES for split in ("train", "val")),
        *(f"rec_{task}_{split}" for task in ("full", "r1", "r2") for split in ("train", "val")),
    ]
    return {
        name: {
            "file_name": f"{name}.jsonl",
            "columns": {"prompt": "prompt", "response": "response", "system": "system"},
        }
        for name in names
    }


def main() -> None:
    if not EXP_A.is_dir() or not RAW.is_file():
        raise FileNotFoundError("required Exp-A or raw recommendation source is missing")
    temporary = OUT.with_name(OUT.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    base_reports = {name: split_base_file(EXP_A / name, temporary) for name in BASE_FILES}
    rec_report = build_recommendation(temporary)
    (temporary / "dataset_info.json").write_text(
        json.dumps(dataset_info(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reuse_note = {
        "material_and_user_source": str(EXP_A),
        "recommendation_full_source": str(RAW),
        "reuse_policy": (
            "Material/user JSON objects are not transformed; only deterministic 95/5 assignment is applied. "
            "Full recommendation system/prompt/response are unchanged. R1/R2 are newly generated."
        ),
        "auxiliary_cleaning": (
            "The established incomplete-CoT tail filter is applied only to R1/R2. It is not applied to Full."
        ),
    }
    (temporary / "REUSE_AND_SPLIT.json").write_text(
        json.dumps(reuse_note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "seed": SEED,
        "validation_fraction_requested": VAL_FRACTION,
        "base": base_reports,
        "recommendation": rec_report,
    }
    (temporary / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    temporary.rename(OUT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "build_report.json", REPORT_DIR / "build_report.json")
    shutil.copy2(OUT / "REUSE_AND_SPLIT.json", REPORT_DIR / "REUSE_AND_SPLIT.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
