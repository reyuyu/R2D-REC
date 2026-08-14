#!/usr/bin/env python3
"""Create an unregistered recommendation pool with incomplete CoT converted to NoThink."""

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_SECTIONS = ("【兴趣归纳】", "【行为模式】", "【预测总结】")
RECOMMENDATION_SEGMENTS = {"recommendation_cot", "recommendation_nocot"}


def cot_is_complete(output: str) -> bool:
    if "<think>" not in output or "</think>" not in output:
        return False
    thought = output.split("<think>", 1)[1].split("</think>", 1)[0]
    return all(section in thought for section in REQUIRED_SECTIONS)


def convert_to_nothink(row: dict) -> dict:
    output = str(row["output"])
    if "</think>" not in output:
        raise ValueError("Cannot convert an unclosed CoT output safely.")
    final_answer = output.split("</think>", 1)[1].strip()
    if not final_answer:
        raise ValueError("Cannot convert a CoT row without a final answer.")
    converted = dict(row)
    instruction = str(converted.get("instruction", ""))
    if not instruction.rstrip().endswith("/think"):
        raise ValueError("Incomplete CoT instruction lacks terminal /think marker.")
    converted["instruction"] = instruction.rstrip()[: -len("/think")] + "/no_think"
    converted["output"] = "<think>\n\n</think>\n\n" + final_answer
    converted["source_segment"] = "recommendation_nocot"
    return converted


def write_jsonl(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exceptions-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() or args.exceptions_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing task-pool or exception directory.")
    args.output_dir.mkdir(parents=True)
    args.exceptions_dir.mkdir(parents=True)

    output_paths = {
        "cot_complete": args.output_dir / "recommendation_cot_complete.jsonl",
        "nocot_original": args.output_dir / "recommendation_nocot_original.jsonl",
        "nocot_converted": args.output_dir / "recommendation_nocot_from_incomplete_cot.jsonl",
        "all": args.output_dir / "recommendation_all_after_incomplete_cot_filter.jsonl",
    }
    exception_path = args.exceptions_dir / "recommendation_cot_incomplete_original.jsonl"
    handles = {key: path.open("x", encoding="utf-8") for key, path in output_paths.items()}
    exception_handle = exception_path.open("x", encoding="utf-8")
    counts = Counter()
    missing_section_patterns = Counter()

    try:
        with args.dataset.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                if row.get("data_source") != "recommend" or row.get("source_segment") not in RECOMMENDATION_SEGMENTS:
                    continue
                counts["recommendation_input_rows"] += 1
                segment = row["source_segment"]
                if segment == "recommendation_nocot":
                    write_jsonl(handles["nocot_original"], row)
                    write_jsonl(handles["all"], row)
                    counts["nocot_original_rows"] += 1
                    continue

                output = str(row.get("output", ""))
                if cot_is_complete(output):
                    write_jsonl(handles["cot_complete"], row)
                    write_jsonl(handles["all"], row)
                    counts["cot_complete_rows"] += 1
                    continue

                write_jsonl(exception_handle, row)
                converted = convert_to_nothink(row)
                write_jsonl(handles["nocot_converted"], converted)
                write_jsonl(handles["all"], converted)
                counts["cot_incomplete_converted_rows"] += 1
                thought = output.split("<think>", 1)[1].split("</think>", 1)[0] if "<think>" in output and "</think>" in output else ""
                missing = tuple(section for section in REQUIRED_SECTIONS if section not in thought)
                missing_section_patterns[" + ".join(missing) if missing else "missing_or_unclosed_think"] += 1
                counts["exception_original_rows"] += 1
    finally:
        for handle in handles.values():
            handle.close()
        exception_handle.close()

    # One original incomplete CoT row is written exactly once to exceptions.
    if counts["exception_original_rows"] != counts["cot_incomplete_converted_rows"]:
        raise AssertionError("Exception and converted-row counts differ.")
    if counts["recommendation_input_rows"] != counts["cot_complete_rows"] + counts["nocot_original_rows"] + counts["cot_incomplete_converted_rows"]:
        raise AssertionError("Recommendation route counts do not balance.")

    manifest = {
        "kind": "unregistered_task_pool",
        "name": "不完整过滤",
        "source_dataset": str(args.dataset),
        "scope": "data_source=recommend from beta-baseline-v1",
        "rule": {
            "complete_cot": "<think> contains 【兴趣归纳】, 【行为模式】, and 【预测总结】",
            "incomplete_cot": "original row is retained only in exceptions; task pool converts it to recommendation_nocot",
            "conversion": "instruction terminal /think -> /no_think; output becomes empty think block plus original final answer after </think>; metadata unchanged",
        },
        "counts": dict(counts),
        "missing_section_patterns": dict(missing_section_patterns),
        "files": {key: path.name for key, path in output_paths.items()},
        "exceptions": {"directory": str(args.exceptions_dir), "original_incomplete_cot": exception_path.name},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.exceptions_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
