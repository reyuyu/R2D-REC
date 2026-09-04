#!/usr/bin/env python3
"""Export classified competition data in reference and LLaMA-Factory JSONL formats."""

from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_INPUT = Path("/data/LLm-8B/code/train/data/processed")
DEFAULT_OUTPUT = Path("/data/LLm-8B/code/train/data/classified_jsonl")
DEFAULT_MARKER = Path("/data/LLm-8B/code/train/data/.classified_success.json")
WORLD_GROUPS = ("world_500", "world_819")

GROUPS = (
    "material_think_sid_to_semantic",
    "material_think_semantic_to_sid",
    "material_no_think_sid_to_semantic",
    "material_no_think_semantic_to_sid",
    "user_think",
    "user_no_think",
    "recommendation_think",
    "recommendation_no_think",
)
CONTROL_SUFFIX = re.compile(r"\s*/(?:no_)?think\s*$")
THINK_BLOCK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)
SID_PATTERN = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def split_dialog(messages: list[dict[str, str]]) -> tuple[str, str, str]:
    system = ""
    if messages and messages[0]["role"] == "system":
        system = messages[0]["content"]
        messages = messages[1:]
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError(f"expected one user/assistant turn, got roles: {[x['role'] for x in messages]}")
    return system, messages[0]["content"], messages[1]["content"]


def with_control(prompt: str, mode: str) -> str:
    prompt = CONTROL_SUFFIX.sub("", prompt).rstrip()
    return prompt + ("/think" if mode == "think" else "/no_think")


def think_response(response: str) -> str:
    match = THINK_BLOCK.match(response)
    if not match or not match.group(1).strip():
        raise ValueError("think sample has no non-empty <think> block")
    return response.strip()


def no_think_response(response: str) -> str:
    match = THINK_BLOCK.match(response)
    answer = response[match.end() :] if match else response
    return "<think>\n</think>\n" + answer.lstrip("\n").strip()


def classify(source_dataset: str, prompt: str, response: str) -> list[tuple[str, str]]:
    if source_dataset.startswith("light_SID2Caption_"):
        mode = "no_think" if source_dataset.endswith("_uncot") else "think"
        answer = response.split("</think>", 1)[-1]
        prompt_has_sid = SID_PATTERN.search(prompt) is not None
        answer_has_sid = SID_PATTERN.search(answer) is not None
        if prompt_has_sid == answer_has_sid:
            raise ValueError(
                f"cannot uniquely determine material direction: prompt_has_sid={prompt_has_sid}, "
                f"answer_has_sid={answer_has_sid}"
            )
        direction = "sid_to_semantic" if prompt_has_sid else "semantic_to_sid"
        return [(f"material_{mode}_{direction}", mode)]
    if source_dataset == "OneRec_Foundation_Multi_Hop_Think":
        return [("user_think", "think")]
    if source_dataset == "OneRec_Foundation_Multi_Hop_Nothink":
        return [("user_no_think", "no_think")]
    if source_dataset == "OneRec_Foundation_UserProfile_CoT_Sft_Demodata_Video":
        return [("recommendation_think", "think"), ("recommendation_no_think", "no_think")]
    raise ValueError(f"unmapped source dataset: {source_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    args = parser.parse_args()

    sources = sorted(args.input.glob("*.parquet"))
    if not sources:
        raise SystemExit(f"No converted Parquet files found under {args.input}")

    reference_dir = args.output / "reference"
    factory_dir = args.output / "llamafactory"
    reference_dir.mkdir(parents=True, exist_ok=True)
    factory_dir.mkdir(parents=True, exist_ok=True)
    counts = {group: 0 for group in GROUPS}

    temp_paths: list[Path] = []
    final_paths: dict[tuple[str, str], Path] = {}
    with ExitStack() as stack:
        handles = {}
        for group in GROUPS:
            for kind, directory in (("reference", reference_dir), ("llamafactory", factory_dir)):
                final = directory / f"{group}.jsonl"
                temp = final.with_suffix(".jsonl.tmp")
                temp.unlink(missing_ok=True)
                temp_paths.append(temp)
                final_paths[(kind, group)] = final
                handles[(kind, group)] = stack.enter_context(temp.open("w", encoding="utf-8"))

        try:
            for index, source in enumerate(sources, start=1):
                table = pq.read_table(source, columns=["messages", "source_dataset"])
                for row in table.to_pylist():
                    system, prompt, response = split_dialog(row["messages"])
                    for group, mode in classify(row["source_dataset"], prompt, response):
                        record = {
                            "system": system,
                            "prompt": with_control(prompt, mode),
                            "response": think_response(response) if mode == "think" else no_think_response(response),
                        }
                        plain = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        handles[("llamafactory", group)].write(plain + "\n")
                        handles[("reference", group)].write("[" + plain + "]\n")
                        counts[group] += 1
                if index % 100 == 0 or index == len(sources):
                    print(f"[{index}/{len(sources)}] exported", flush=True)
        except Exception:
            for handle in handles.values():
                handle.close()
            for temp in temp_paths:
                temp.unlink(missing_ok=True)
            raise

    for key, final in final_paths.items():
        os.replace(final.with_suffix(".jsonl.tmp"), final)

    world_counts: dict[str, int] = {}
    for group in WORLD_GROUPS:
        reference = reference_dir / f"{group}.jsonl"
        if not reference.is_file():
            raise FileNotFoundError(f"missing world reference dataset: {reference}")
        final = factory_dir / f"{group}.jsonl"
        temp = final.with_suffix(".jsonl.tmp")
        temp.unlink(missing_ok=True)
        count = 0
        try:
            with reference.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
                for line_number, line in enumerate(source, start=1):
                    wrapped = json.loads(line)
                    if not isinstance(wrapped, list) or len(wrapped) != 1 or not isinstance(wrapped[0], dict):
                        raise ValueError(f"{reference}:{line_number}: expected a single-element JSON array")
                    record = wrapped[0]
                    if set(record) != {"system", "prompt", "response"}:
                        raise ValueError(f"{reference}:{line_number}: invalid fields {sorted(record)}")
                    target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        os.replace(temp, final)
        world_counts[group] = count

    marker = {
        "input": str(args.input),
        "output": str(args.output),
        "source_files": len(sources),
        "counts": counts,
        "world_counts": world_counts,
        "total_reference_rows": sum(counts.values()),
    }
    args.marker.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
