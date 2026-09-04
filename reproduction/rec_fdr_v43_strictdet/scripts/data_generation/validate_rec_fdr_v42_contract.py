#!/usr/bin/env python3
"""Fail-fast validation for both V4.2 experiment contracts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml
from build_rec_fdr_v42_datasets import MARKER


ROOT = Path("/data/LLm-8B/code/train")
SOURCE = ROOT / "data/rec_fdr_curriculum_v4"
A = ROOT / "data/rec_fdr_v42_interestonly"
B = ROOT / "data/rec_fdr_v42_cot_uncot_domainratio"
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        yield from (json.loads(line) for line in handle if line.strip())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_a() -> None:
    report = json.loads((A / "interestonly_equivalence_report.json").read_text(encoding="utf-8"))
    for key in ("same_row_count", "same_prompt_group_ids", "same_final_target_sids", "same_positive_sets", "same_train_val_split"):
        require(report.get(key) is True, f"A equivalence failed: {key}")
    require(report["interest_parse_success_rate"] >= 0.995, "A parser success below 99.5%")
    for path in A.glob("rec_*.jsonl"):
        for index, row in enumerate(rows(path), 1):
            if path.name.startswith(("rec_full", "rec_r1")):
                match = THINK.search(row["response"])
                require(match is not None, f"{path.name}:{index} missing think")
                cot = match.group(1)
            elif path.name.startswith("rec_r2"):
                cot = row["sanitized_cot"]
                require(row["final_target_sid"] not in row["prompt"], f"{path.name}:{index} target leaked into prompt")
            else:
                continue
            require("兴趣归纳" in cot, f"{path.name}:{index} missing interest")
            require(not any(MARKER["行为模式"].match(line) for line in cot.splitlines()), f"{path.name}:{index} behavior title remains")
            require(not any(MARKER["预测总结"].match(line) for line in cot.splitlines()), f"{path.name}:{index} prediction title remains")
    config = yaml.safe_load((ROOT / "configs/rec_fdr_v42_interestonly_cot_from0_fullft_32k_2gpu.yaml").read_text())
    require(config["gradient_accumulation_steps"] == 8, "A accumulation must preserve global batch 16")
    require(str(config["output_dir"]).startswith("/mnt/"), "A model output is not on /mnt")


def validate_b() -> None:
    report = json.loads((B / "cot_uncot_ratio_equivalence_report.json").read_text(encoding="utf-8"))
    for key in ("same_full_row_count", "same_prompt_group_ids", "same_final_target_sids", "same_positive_sets", "same_r1_r2"):
        require(report.get(key) is True, f"B equivalence failed: {key}")
    for file_name, strata in report["strata"].items():
        for name, values in strata.items():
            if file_name != "rec_full_val.jsonl" or values["rows"] >= 200:
                require(values["absolute_error_percentage_points"] <= 0.5, f"B ratio miss: {file_name}/{name}")
    for name in ("rec_r1_train.jsonl", "rec_r1_val.jsonl", "rec_r2_k1_train.jsonl", "rec_r2_k2_train.jsonl", "rec_r2_k3_train.jsonl", "rec_r2_val.jsonl"):
        require(os.stat(SOURCE / name).st_ino == os.stat(B / name).st_ino, f"B {name} is not byte-identical hard link")
    for path in [*(B / f"rec_full_{bucket}_train.jsonl" for bucket in ("k1", "k2", "k3")), B / "rec_full_val.jsonl"]:
        for index, row in enumerate(rows(path), 1):
            match = THINK.search(row["response"])
            require(match is not None, f"{path.name}:{index} missing think")
            if row["response_mode"] == "cot":
                require(row["prompt"].endswith("/think"), f"{path.name}:{index} CoT prompt switch changed")
                require(match.group(1).strip(), f"{path.name}:{index} CoT reasoning empty")
            else:
                require(row["response_mode"] == "uncot", f"{path.name}:{index} invalid mode")
                require(row["prompt"].endswith("/no_think"), f"{path.name}:{index} unCoT prompt switch wrong")
                require(not match.group(1).strip(), f"{path.name}:{index} unCoT reasoning not empty")
    config = yaml.safe_load((ROOT / "configs/rec_fdr_v42_cot_uncot_domainratio_from0_fullft_32k_2gpu.yaml").read_text())
    require(config["gradient_accumulation_steps"] == 8, "B accumulation must preserve global batch 16")
    require(str(config["output_dir"]).startswith("/mnt/"), "B model output is not on /mnt")


def main() -> None:
    catalog = json.loads((SOURCE / "rec_group_catalog.json").read_text(encoding="utf-8"))
    train = {item["prompt_group_id"] for item in catalog["train_groups"]}
    val = {item["prompt_group_id"] for item in catalog["validation_groups"]}
    require(train.isdisjoint(val), "V4.1 train/validation prompt groups overlap")
    validate_a()
    validate_b()
    print(json.dumps({"status": "ok", "train_val_prompt_group_overlap": 0, "variants": ["a", "b"]}))


if __name__ == "__main__":
    main()
