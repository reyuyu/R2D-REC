#!/usr/bin/env python3
"""Create an unregistered, independently reusable clean understand_user task pool."""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


SID_PATTERN = r"<\|(ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
SID_RE = re.compile(SID_PATTERN)
DATE_RE = re.compile(r"\u3010(\d{4}-\d{2}-\d{2})\u3011")
EVENT_RE = re.compile(r"(?:^|\n)\s*[^\s]+\s+\[([^\]]+)\]\s+(" + SID_PATTERN + r")")
SEGMENTS = ("user_action", "user_chain_cot", "user_chain_nocot")
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)


def summary(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return {"count": 0, "mean": None, "quantiles": {}}

    def quantile(value):
        return ordered[max(0, min(count - 1, math.ceil(value * count) - 1))]

    return {
        "count": count,
        "mean": round(sum(ordered) / count, 2),
        "quantiles": {f"p{int(value * 100):02d}": quantile(value) for value in QUANTILES},
    }


def history_events(text):
    triples = set()
    sid_set = set()
    dates = list(DATE_RE.finditer(text))
    for index, match in enumerate(dates):
        date = match.group(1)
        end = dates[index + 1].start() if index + 1 < len(dates) else len(text)
        for event in EVENT_RE.finditer(text[match.end():end]):
            action, sid = event.group(1), event.group(2)
            triples.add((date, action, sid))
            sid_set.add(sid)
    return triples, sid_set


def final_chain_events(output):
    """Return (events, error); events are (date, action, sid)."""
    payload_text = output.split("</think>", 1)[-1].strip()
    try:
        payload = json.loads(payload_text)
        events = payload["logic_chain"]["events"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], "invalid_final_json"
    if not isinstance(events, list):
        return [], "invalid_event_shape"

    parsed = []
    for event in events:
        if not isinstance(event, dict):
            return [], "invalid_event_shape"
        date, action_text = event.get("date"), event.get("action")
        sid_match = SID_RE.search(str(action_text))
        action_match = re.search(r"\[([^\]]+)\]", str(action_text))
        if not isinstance(date, str) or sid_match is None or action_match is None:
            return [], "invalid_event_shape"
        parsed.append((date, action_match.group(1), sid_match.group(0)))
    return parsed, None


def action_answer_is_valid(output, history_sid_set):
    payload_text = output.split("</think>", 1)[-1].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return False, "invalid_final_json"
    if not isinstance(payload, list) or not all(isinstance(item, str) and SID_RE.fullmatch(item) for item in payload):
        return False, "invalid_event_shape"
    if len(payload) != len(set(payload)):
        return False, "duplicate_answer_sid"
    if any(item not in history_sid_set for item in payload):
        return False, "answer_sid_outside_history"
    return True, None


def row_reasons(row, total_tokens):
    reasons = []
    if total_tokens > 8192:
        reasons.append("total_tokens_gt_8192")
    segment = row["source_segment"]
    history, history_sid_set = history_events(str(row.get("input", "")))
    output = str(row.get("output", ""))

    if segment == "user_action":
        _, error = action_answer_is_valid(output, history_sid_set)
        if error:
            reasons.append(error)
        return reasons

    events, error = final_chain_events(output)
    if error:
        reasons.append(error)
        return reasons
    for date, action, sid in events:
        if (date, action, sid) not in history:
            reasons.append("date_or_action_mismatch")
            break
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    outputs = {segment: (args.output_dir / f"{segment}.jsonl").open("w", encoding="utf-8") for segment in SEGMENTS}
    merged = (args.output_dir / "understand_user_clean.jsonl").open("w", encoding="utf-8")
    removed = (args.output_dir / "removed_samples.jsonl").open("w", encoding="utf-8")
    stats = {segment: {"before": 0, "after": 0, "removed": Counter(), "total_tokens_after": []} for segment in SEGMENTS}

    try:
        with args.dataset.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                row = json.loads(line)
                if row.get("data_source") != "understand_user" or row.get("source_segment") not in SEGMENTS:
                    continue
                segment = row["source_segment"]
                stats[segment]["before"] += 1
                prompt = "".join(str(row.get(key, "")) for key in ("system", "instruction", "input"))
                output = str(row.get("output", ""))
                total_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + len(tokenizer(output, add_special_tokens=False)["input_ids"])
                reasons = row_reasons(row, total_tokens)
                if reasons:
                    stats[segment]["removed"].update(reasons)
                    removed.write(json.dumps({"source_line": line_number, "source_segment": segment, "total_tokens": total_tokens, "reasons": reasons}, ensure_ascii=False) + "\n")
                    continue
                outputs[segment].write(line)
                merged.write(line)
                stats[segment]["after"] += 1
                stats[segment]["total_tokens_after"].append(total_tokens)
    finally:
        for handle in outputs.values():
            handle.close()
        merged.close()
        removed.close()

    total_before = sum(item["before"] for item in stats.values())
    total_after = sum(item["after"] for item in stats.values())
    manifest = {
        "kind": "unregistered_task_data_pool",
        "source_dataset": str(args.dataset),
        "scope": "data_source=understand_user",
        "tokenizer": args.model,
        "token_rule": "raw system+instruction+input plus raw output, add_special_tokens=false; remove total > 8192",
        "chain_rule": "remove malformed final JSON/event shapes and any final event whose exact (date, action, SID) is absent from input history",
        "action_rule": "remove malformed JSON-array outputs, duplicate output SIDs, and output SIDs absent from input history",
        "outputs": {"merged": "understand_user_clean.jsonl", "segments": {segment: f"{segment}.jsonl" for segment in SEGMENTS}, "removed_audit": "removed_samples.jsonl"},
        "total_before": total_before,
        "total_after": total_after,
        "total_removed": total_before - total_after,
        "segments": {
            segment: {
                "before": item["before"],
                "after": item["after"],
                "removed_unique_rows": item["before"] - item["after"],
                "share_after": round(item["after"] / total_after, 8) if total_after else 0,
                "removal_reason_hits": dict(sorted(item["removed"].items())),
                "total_token_distribution_after": summary(item["total_tokens_after"]),
            }
            for segment, item in stats.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
