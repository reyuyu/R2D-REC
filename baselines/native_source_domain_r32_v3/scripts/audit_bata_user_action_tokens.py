#!/usr/bin/env python3
"""Read-only raw-token distribution audit for bata_baseline user_action."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


SID = re.compile(r"<\|(ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
THRESHOLDS = (1024, 2048, 4096, 6144, 8192)


def summary(values: list[int]) -> dict[str, object]:
    ordered = sorted(values)
    size = len(ordered)
    def quantile(q: float) -> int:
        return ordered[max(0, min(size - 1, math.ceil(q * size) - 1))]
    return {
        "count": size,
        "mean": round(sum(ordered) / size, 2),
        "quantiles": {f"p{int(q * 100):02d}": quantile(q) for q in QUANTILES},
        "over_threshold": {str(limit): sum(value > limit for value in ordered) for limit in THRESHOLDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    input_tokens, output_tokens, field_tokens, answer_sid_counts = [], [], [], []
    with args.dataset.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("data_source") != "understand_user" or row.get("source_segment") != "user_action":
                continue
            prompt = "".join(str(row.get(key, "")) for key in ("system", "instruction", "input"))
            output = str(row.get("output", ""))
            prompt_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            output_count = len(tokenizer(output, add_special_tokens=False)["input_ids"])
            input_tokens.append(prompt_count)
            output_tokens.append(output_count)
            field_tokens.append(prompt_count + output_count)
            answer_sid_counts.append(len(SID.findall(output)))

    report = {
        "scope": "data_source=understand_user AND source_segment=user_action",
        "tokenizer": args.model,
        "token_counting": "raw concatenated system+instruction+input and raw output, add_special_tokens=false; chat-template framing is excluded",
        "input_prompt_tokens": summary(input_tokens),
        "output_tokens": summary(output_tokens),
        "raw_prompt_plus_output_tokens": summary(field_tokens),
        "answer_sid_count": summary(answer_sid_counts),
        "answer_sid_count_histogram": {
            str(count): answer_sid_counts.count(count) for count in range(1, max(answer_sid_counts) + 1)
        },
        "answer_sid_count_bands": {
            "1": sum(count == 1 for count in answer_sid_counts),
            "2": sum(count == 2 for count in answer_sid_counts),
            "3": sum(count == 3 for count in answer_sid_counts),
            "4": sum(count == 4 for count in answer_sid_counts),
            "5": sum(count == 5 for count in answer_sid_counts),
            "6-10": sum(6 <= count <= 10 for count in answer_sid_counts),
            "11-20": sum(11 <= count <= 20 for count in answer_sid_counts),
            "21-30": sum(21 <= count <= 30 for count in answer_sid_counts),
            "31-40": sum(31 <= count <= 40 for count in answer_sid_counts),
            "41-50": sum(41 <= count <= 50 for count in answer_sid_counts),
            "51+": sum(count >= 51 for count in answer_sid_counts),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
