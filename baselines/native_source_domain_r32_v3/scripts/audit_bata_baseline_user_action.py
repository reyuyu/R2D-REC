#!/usr/bin/env python3
"""Read-only integrity audit for bata_baseline_v1 user_action rows."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SID = re.compile(r"<\|(ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def digest_text(*parts: object) -> str:
    content = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def examples(groups: dict[str, list[int]], counts: Counter, limit: int = 10) -> list[dict[str, object]]:
    result = []
    for key, total in counts.most_common(limit):
        if total > 1:
            result.append({"sha256": key, "rows": total, "line_numbers": groups[key][:20]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    rows = 0
    no_answer_sid = 0
    malformed_output_sid = 0
    answer_sid_total = 0
    answer_sid_in_history = 0
    answer_sid_outside_history = 0
    rows_with_outside_history = 0
    outside_examples = []
    output_duplicate_sid_rows = 0
    input_hash_counts = Counter()
    input_hash_lines = defaultdict(list)
    full_prompt_hash_counts = Counter()
    full_prompt_hash_lines = defaultdict(list)

    with args.dataset.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            row = json.loads(line)
            if row.get("data_source") != "understand_user" or row.get("source_segment") != "user_action":
                continue
            rows += 1
            input_text = str(row.get("input", ""))
            output_text = str(row.get("output", ""))
            input_sids = set(match.group(0) for match in SID.finditer(input_text))
            output_sids = [match.group(0) for match in SID.finditer(output_text)]
            answer_sid_total += len(output_sids)
            if not output_sids:
                no_answer_sid += 1
            if len(output_sids) != len(set(output_sids)):
                output_duplicate_sid_rows += 1
            # Every partial SID marker in output is malformed even if no complete SID was matched.
            marker_count = output_text.count("<|")
            if marker_count != len(output_sids):
                malformed_output_sid += 1

            outside = sorted(set(output_sids) - input_sids)
            answer_sid_in_history += len(output_sids) - sum(sid not in input_sids for sid in output_sids)
            answer_sid_outside_history += sum(sid not in input_sids for sid in output_sids)
            if outside:
                rows_with_outside_history += 1
                if len(outside_examples) < 20:
                    outside_examples.append(
                        {
                            "line_number": line_no,
                            "outside_history_sids": outside,
                            "history_sid_count": len(input_sids),
                            "answer_sid_count": len(output_sids),
                        }
                    )

            input_key = digest_text(input_text)
            full_prompt_key = digest_text(
                row.get("instruction", ""), input_text, row.get("history", []), row.get("system", "")
            )
            input_hash_counts[input_key] += 1
            full_prompt_hash_counts[full_prompt_key] += 1
            if len(input_hash_lines[input_key]) < 20:
                input_hash_lines[input_key].append(line_no)
            if len(full_prompt_hash_lines[full_prompt_key]) < 20:
                full_prompt_hash_lines[full_prompt_key].append(line_no)

    duplicate_input_groups = sum(count > 1 for count in input_hash_counts.values())
    duplicate_full_prompt_groups = sum(count > 1 for count in full_prompt_hash_counts.values())
    report = {
        "scope": "data_source=understand_user AND source_segment=user_action",
        "rows": rows,
        "answer_sid_audit": {
            "answer_sid_total": answer_sid_total,
            "answer_sid_in_history": answer_sid_in_history,
            "answer_sid_outside_history": answer_sid_outside_history,
            "outside_history_rate": answer_sid_outside_history / answer_sid_total if answer_sid_total else 0.0,
            "rows_with_outside_history": rows_with_outside_history,
            "rows_with_no_answer_sid": no_answer_sid,
            "rows_with_duplicate_answer_sid": output_duplicate_sid_rows,
            "rows_with_malformed_output_sid_marker": malformed_output_sid,
            "outside_history_examples": outside_examples,
        },
        "duplicate_prompts": {
            "exact_input": {
                "duplicate_groups": duplicate_input_groups,
                "duplicate_rows_including_first": sum(count for count in input_hash_counts.values() if count > 1),
                "extra_duplicate_rows": sum(count - 1 for count in input_hash_counts.values() if count > 1),
                "examples": examples(input_hash_lines, input_hash_counts),
            },
            "exact_full_training_prompt": {
                "fields": ["instruction", "input", "history", "system"],
                "duplicate_groups": duplicate_full_prompt_groups,
                "duplicate_rows_including_first": sum(count for count in full_prompt_hash_counts.values() if count > 1),
                "extra_duplicate_rows": sum(count - 1 for count in full_prompt_hash_counts.values() if count > 1),
                "examples": examples(full_prompt_hash_lines, full_prompt_hash_counts),
            },
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
