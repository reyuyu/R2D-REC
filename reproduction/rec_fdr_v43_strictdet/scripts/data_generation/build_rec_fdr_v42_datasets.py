#!/usr/bin/env python3
"""Build the two V4.2 recommendation views from the frozen V4.1 dataset."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/data/LLm-8B/code/train")
SOURCE = ROOT / "data/rec_fdr_curriculum_v4"
OUT_A = ROOT / "data/rec_fdr_v42_interestonly"
OUT_B = ROOT / "data/rec_fdr_v42_cot_uncot_domainratio"
SEED = 19260817
RATIOS = {"video": 0.55, "prod": 0.95, "ad": 0.25, "living": 0.55}
FULL_TRAIN = [f"rec_full_{bucket}_train.jsonl" for bucket in ("k1", "k2", "k3")]
RECOMMENDATION_FILES = [
    *FULL_TRAIN,
    "rec_r1_train.jsonl",
    "rec_r2_k1_train.jsonl",
    "rec_r2_k2_train.jsonl",
    "rec_r2_k3_train.jsonl",
    "rec_full_val.jsonl",
    "rec_r1_val.jsonl",
    "rec_r2_val.jsonl",
]
AUXILIARY_R2_K1 = [f"rec_r2_k1_pack_seed_{seed}_train.jsonl" for seed in range(4)]
A_FILES = RECOMMENDATION_FILES + AUXILIARY_R2_K1
MARKER = {
    name: re.compile(
        rf"^\s*(?:#+\s*)?(?:(?:\d+|[一二三四五六七八九十]+)[.、]\s*)?"
        rf"(?:\*{{0,2}}\s*)?【?\s*(?:\*{{0,2}}\s*)?{name}"
        rf"\s*(?:\*{{0,2}}\s*)?】?\s*(?:\*{{0,2}}\s*)?(?:[:：]\s*)?"
    )
    for name in ("兴趣归纳", "行为模式", "预测总结")
}
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interest_only(cot: str) -> str:
    lines = cot.splitlines()
    start: int | None = None
    first_content = ""
    stop = len(lines)
    for index, line in enumerate(lines):
        match = MARKER["兴趣归纳"].match(line)
        if match and start is None:
            start = index + 1
            first_content = line[match.end():]
            continue
        if start is not None and (MARKER["行为模式"].match(line) or MARKER["预测总结"].match(line)):
            stop = index
            break
    if start is None:
        raise ValueError("interest marker not found")
    content = ([first_content] if first_content.strip() else []) + lines[start:stop]
    body = "\n".join(content).strip()
    if not body:
        raise ValueError("interest section is empty")
    return f"【兴趣归纳】\n{body}"


def replace_think(response: str, replacement: str) -> tuple[str, str]:
    match = THINK.search(response)
    if match is None:
        raise ValueError("complete think block not found")
    tail = response[match.end():]
    return response[:match.start()] + f"<think>\n{replacement}\n</think>" + tail, tail


def prepare_dir(path: Path) -> Path:
    temporary = path.with_name(path.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    return temporary


def link_shared(source: Path, target: Path, excluded: set[str]) -> None:
    for path in source.iterdir():
        if path.is_file() and path.name not in excluded:
            os.link(path, target / path.name)


def finalize(temporary: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    temporary.rename(output)


def transform_a_row(row: dict[str, Any], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    output = dict(row)
    if name.startswith(("rec_full", "rec_r1")):
        match = THINK.search(row["response"])
        if match is None:
            raise ValueError(f"{name}: missing think block")
        original = match.group(1)
        interest = interest_only(original)
        output["response"], tail = replace_think(row["response"], interest)
        if tail != row["response"][match.end():]:
            raise AssertionError("answer tail changed")
    elif name.startswith("rec_r2"):
        original = row["sanitized_cot"]
        interest = interest_only(original)
        if original not in row["prompt"]:
            raise ValueError(f"{name}: sanitized_cot is not an exact prompt substring")
        output["prompt"] = row["prompt"].replace(original, interest, 1)
        output["sanitized_cot"] = interest
    else:
        raise ValueError(name)
    return output, {
        "original_chars": len(original),
        "interest_chars": len(interest),
        "removed_chars": len(original) - len(interest),
    }


def build_a() -> dict[str, Any]:
    temporary = prepare_dir(OUT_A)
    link_shared(SOURCE, temporary, set(A_FILES))
    totals = Counter()
    per_file: dict[str, Any] = {}
    before_groups: dict[str, list[str]] = {}
    after_groups: dict[str, list[str]] = {}
    for name in A_FILES:
        source_path = SOURCE / name
        target_path = temporary / name
        file_stats = Counter()
        old_group_ids: list[str] = []
        new_group_ids: list[str] = []
        with target_path.open("w", encoding="utf-8") as output_handle:
            for row in rows(source_path):
                logical_row = name not in AUXILIARY_R2_K1
                totals["total_reco_rows"] += int(logical_row)
                if logical_row and THINK.search(row.get("response", "")):
                    totals["has_think_rows"] += 1
                original_text = row.get("sanitized_cot", row.get("response", ""))
                totals["has_interest_marker_rows"] += int(logical_row and "兴趣归纳" in original_text)
                totals["behavior_marker_rows"] += int(logical_row and "行为模式" in original_text)
                totals["prediction_marker_rows"] += int(logical_row and "预测总结" in original_text)
                try:
                    transformed, stats = transform_a_row(row, name)
                except ValueError:
                    totals["interest_parse_fail_rows"] += int(logical_row)
                    raise
                totals["interest_parse_success_rows"] += int(logical_row)
                file_stats.update(stats)
                if "prompt_group_id" in row:
                    old_group_ids.append(str(row["prompt_group_id"]))
                    new_group_ids.append(str(transformed["prompt_group_id"]))
                for key in ("final_target_sid", "positive_future_sids", "rec_group_index", "rec_group_key"):
                    if row.get(key) != transformed.get(key):
                        raise AssertionError(f"{name}: {key} changed")
                output_handle.write(json.dumps(transformed, ensure_ascii=False, separators=(",", ":")) + "\n")
        before_groups[name] = old_group_ids
        after_groups[name] = new_group_ids
        per_file[name] = {"rows": totals_for_file(source_path), **dict(file_stats)}
    success = totals["interest_parse_success_rows"] / max(1, totals["total_reco_rows"])
    if success < 0.995:
        raise RuntimeError(f"interest parse success {success:.6f} is below 0.995")
    report = {
        "schema_version": 1,
        "interest_parse_fail_rows": 0,
        **dict(totals),
        "interest_parse_success_rate": success,
        "same_row_count": all(totals_for_file(SOURCE / n) == totals_for_file(temporary / n) for n in A_FILES),
        "same_prompt_group_ids": before_groups == after_groups,
        "same_final_target_sids": True,
        "same_positive_sets": True,
        "same_train_val_split": True,
        "only_expected_change": "recommendation_cot_content",
        "per_file": per_file,
    }
    (temporary / "interestonly_equivalence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    token_report = interest_token_report(temporary)
    (temporary / "interest_token_report.json").write_text(
        json.dumps(token_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(temporary, "v4.2-a-interest-only")
    finalize(temporary, OUT_A)
    return report


def interest_token_report(transformed_root: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT.parent / "OneReason-8B-pretrain-competition", trust_remote_code=True, local_files_only=True
    )
    original_lengths: list[int] = []
    interest_lengths: list[int] = []
    original_batch: list[str] = []
    interest_batch: list[str] = []

    def flush() -> None:
        if not original_batch:
            return
        original_lengths.extend(tokenizer(original_batch, add_special_tokens=False, return_length=True)["length"])
        interest_lengths.extend(tokenizer(interest_batch, add_special_tokens=False, return_length=True)["length"])
        original_batch.clear()
        interest_batch.clear()

    for name in RECOMMENDATION_FILES:
        for original_row, transformed_row in zip(rows(SOURCE / name), rows(transformed_root / name), strict=True):
            if name.startswith(("rec_full", "rec_r1")):
                original_match = THINK.search(original_row["response"])
                transformed_match = THINK.search(transformed_row["response"])
                if original_match is None or transformed_match is None:
                    raise ValueError(f"{name}: missing think block during token audit")
                original_batch.append(original_match.group(1).strip())
                interest_batch.append(transformed_match.group(1).strip())
            else:
                original_batch.append(original_row["sanitized_cot"].strip())
                interest_batch.append(transformed_row["sanitized_cot"].strip())
            if len(original_batch) >= 128:
                flush()
    flush()
    removed = [old - new for old, new in zip(original_lengths, interest_lengths)]

    def percentile(values: list[int], fraction: float) -> float:
        ordered = sorted(values)
        return float(ordered[round((len(ordered) - 1) * fraction)])

    old_mean = sum(original_lengths) / len(original_lengths)
    new_mean = sum(interest_lengths) / len(interest_lengths)
    return {
        "schema_version": 1,
        "rows": len(original_lengths),
        "original_cot_tokens_mean": old_mean,
        "interest_tokens_mean": new_mean,
        "interest_tokens_p50": percentile(interest_lengths, 0.50),
        "interest_tokens_p90": percentile(interest_lengths, 0.90),
        "removed_cot_tokens_mean": sum(removed) / len(removed),
        "cot_compression_ratio": new_mean / old_mean,
    }


def totals_for_file(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def assign_modes(indexed: list[tuple[int, dict[str, Any]]]) -> tuple[dict[int, str], dict[str, Any]]:
    strata: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in indexed:
        domain = str(row["target_domain"])
        bucket = str(row["future_entropy_bucket"])
        strata[(domain, bucket)].append(index)
    assignments: dict[int, str] = {}
    audit: dict[str, Any] = {}
    for (domain, bucket), indices in sorted(strata.items()):
        shuffled = list(indices)
        random.Random(f"{SEED}:{domain}:{bucket}").shuffle(shuffled)
        target = RATIOS[domain]
        cot_count = round(target * len(shuffled))
        cot_indices = set(shuffled[:cot_count])
        assignments.update({idx: ("cot" if idx in cot_indices else "uncot") for idx in indices})
        achieved = cot_count / len(indices)
        audit[f"{domain}_{bucket}"] = {
            "rows": len(indices), "cot_rows": cot_count, "uncot_rows": len(indices) - cot_count,
            "target_cot_ratio": target, "achieved_cot_ratio": achieved,
            "absolute_error_percentage_points": abs(achieved - target) * 100,
        }
    return assignments, audit


def make_uncot(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    match = THINK.search(row["response"])
    if match is None:
        raise ValueError("Full response lacks think block")
    tail = row["response"][match.end():]
    output["response"] = row["response"][:match.start()] + "<think>\n</think>" + tail
    if output["prompt"].endswith("/think"):
        output["prompt"] = output["prompt"][:-len("/think")] + "/no_think"
    else:
        raise ValueError("Full prompt does not end in the established /think switch")
    output["response_mode"] = "uncot"
    return output


def build_b() -> dict[str, Any]:
    temporary = prepare_dir(OUT_B)
    transformed = set(FULL_TRAIN + ["rec_full_val.jsonl"])
    link_shared(SOURCE, temporary, transformed)
    ratio_audit: dict[str, Any] = {}
    examples: dict[str, str] = {}
    total_cot = total_uncot = 0
    for name in FULL_TRAIN + ["rec_full_val.jsonl"]:
        indexed = list(enumerate(rows(SOURCE / name)))
        assignments, audit = assign_modes(indexed)
        ratio_audit[name] = audit
        with (temporary / name).open("w", encoding="utf-8") as output_handle:
            for index, row in indexed:
                mode = assignments[index]
                output = dict(row)
                output["response_mode"] = "cot"
                if mode == "uncot":
                    output = make_uncot(row)
                    total_uncot += 1
                else:
                    total_cot += 1
                for key in ("final_target_sid", "positive_future_sids", "prompt_group_id", "rec_group_index", "rec_group_key"):
                    if row[key] != output[key]:
                        raise AssertionError(f"{name}: {key} changed")
                if mode == "cot" and (row["prompt"] != output["prompt"] or row["response"] != output["response"]):
                    raise AssertionError("CoT text changed")
                examples.setdefault(f"{mode}_prompt_example", output["prompt"])
                examples.setdefault(f"{mode}_response_example", output["response"])
                output_handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "schema_version": 1,
        "same_full_row_count": all(totals_for_file(SOURCE / n) == totals_for_file(temporary / n) for n in transformed),
        "same_prompt_group_ids": True,
        "same_final_target_sids": True,
        "same_positive_sets": True,
        "same_r1_r2": all(
            os.stat(SOURCE / n).st_ino == os.stat(temporary / n).st_ino
            for n in RECOMMENDATION_FILES if n.startswith(("rec_r1", "rec_r2"))
        ),
        "only_expected_change": "full_response_mode_ratio",
        "duplicate_rows": False,
        "seed": SEED,
        "total_cot_rows": total_cot,
        "total_uncot_rows": total_uncot,
        "strata": ratio_audit,
    }
    (temporary / "cot_uncot_ratio_equivalence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (temporary / "mode_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(temporary, "v4.2-b-domain-ratio")
    finalize(temporary, OUT_B)
    return report


def write_manifest(root: Path, variant: str) -> None:
    files = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "dataset_manifest.json":
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (root / "dataset_manifest.json").write_text(
        json.dumps({"schema_version": 1, "variant": variant, "source": str(SOURCE), "files": files}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    report_a = build_a()
    report_b = build_b()
    print(json.dumps({"interest_only": report_a, "cot_uncot": report_b}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
