#!/usr/bin/env python3
"""Materialize clean-pool removals as original rows grouped by concrete issue."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SID_PATTERN = r"<\|(ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
SID_RE = re.compile(SID_PATTERN)
DATE_RE = re.compile(r"\u3010(\d{4}-\d{2}-\d{2})\u3011")
EVENT_RE = re.compile(r"(?:^|\n)\s*[^\s]+\s+\[([^\]]+)\]\s+(" + SID_PATTERN + r")")
FILES = {
    "length_gt_8192": "length_gt_8192.jsonl",
    "json_invalid": "json_invalid.jsonl",
    "date_mismatch": "date_mismatch.jsonl",
    "action_mismatch": "action_mismatch.jsonl",
    "date_and_action_mismatch": "date_and_action_mismatch.jsonl",
    "ambiguous_duplicate_history_event": "ambiguous_duplicate_history_event.jsonl",
}


def parse_history(text):
    events = set()
    matches = list(DATE_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        for event in EVENT_RE.finditer(text[match.end():end]):
            events.add((match.group(1), event.group(1), event.group(2)))
    return events


def parse_final_events(output):
    try:
        payload = json.loads(output.split("</think>", 1)[-1].strip())
        events = payload["logic_chain"]["events"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], True
    if not isinstance(events, list):
        return [], True
    parsed = []
    for event in events:
        if not isinstance(event, dict):
            return [], True
        date, action_text = event.get("date"), event.get("action")
        sid = SID_RE.search(str(action_text))
        action = re.search(r"\[([^\]]+)\]", str(action_text))
        if not isinstance(date, str) or sid is None or action is None:
            return [], True
        parsed.append((date, action.group(1), sid.group(0)))
    return parsed, False


def detailed_categories(row, removal_reasons):
    categories = set()
    if "total_tokens_gt_8192" in removal_reasons:
        categories.add("length_gt_8192")
    if row.get("source_segment") == "user_action":
        if any(reason in {"invalid_final_json", "invalid_event_shape"} for reason in removal_reasons):
            categories.add("json_invalid")
        return categories

    final_events, invalid = parse_final_events(str(row.get("output", "")))
    if invalid:
        categories.add("json_invalid")
        return categories
    history = parse_history(str(row.get("input", "")))
    for date, action, sid in final_events:
        if (date, action, sid) in history:
            continue
        sid_events = [event for event in history if event[2] == sid]
        same_date = any(event[0] == date for event in sid_events)
        same_action = any(event[1] == action for event in sid_events)
        if same_date and same_action:
            categories.add("ambiguous_duplicate_history_event")
        elif same_date:
            categories.add("action_mismatch")
        elif same_action:
            categories.add("date_mismatch")
        else:
            categories.add("date_and_action_mismatch")
    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--removed-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = {int(item["source_line"]): item for item in map(json.loads, args.removed_audit.read_text(encoding="utf-8").splitlines())}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    handles = {name: (args.output_dir / path).open("w", encoding="utf-8") for name, path in FILES.items()}
    stats = Counter()
    with args.source_dataset.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            item = audit.get(line_number)
            if item is None:
                continue
            row = json.loads(line)
            for category in detailed_categories(row, item["reasons"]):
                handles[category].write(line)
                stats[category] += 1
    for handle in handles.values():
        handle.close()

    manifest = {
        "scope": "understand_user removed samples, original rows retained",
        "source_dataset": str(args.source_dataset),
        "removed_audit": str(args.removed_audit),
        "files": {FILES[name]: stats[name] for name in FILES},
        "notes": "Rows matching more than one issue are intentionally copied into each matching issue file.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
