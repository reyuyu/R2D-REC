#!/usr/bin/env python3
"""Read-only audit of bata_baseline_v1 user_chain_cot samples."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


SID = r"<\|(ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
SID_RE = re.compile(SID)
DATE_RE = re.compile(r"【([^】]+)】")
EVENT_RE = re.compile(r"(?:^|\n)\s*([^\s]+)\s+\[([^\]]+)\]\s+(" + SID + r")")
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def quantile_summary(values: list[int]) -> dict[str, object]:
    ordered = sorted(values)
    size = len(ordered)
    def q(value: float) -> int:
        return ordered[max(0, min(size - 1, math.ceil(value * size) - 1))]
    return {
        "count": size,
        "mean": round(sum(ordered) / size, 2),
        "quantiles": {f"p{int(value * 100):02d}": q(value) for value in QUANTILES},
        "over_threshold": {str(limit): sum(item > limit for item in ordered) for limit in (4096, 6144, 8192)},
    }


def parse_history_events(text: str) -> tuple[set[str], set[tuple[str, str, str]], Counter]:
    sid_set: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    parse_stats = Counter()
    date_matches = list(DATE_RE.finditer(text))
    for index, date_match in enumerate(date_matches):
        date = date_match.group(1)
        section_end = date_matches[index + 1].start() if index + 1 < len(date_matches) else len(text)
        section = text[date_match.end():section_end]
        for event in EVENT_RE.finditer(section):
            time, action, sid = event.group(1), event.group(2), event.group(3)
            sid_set.add(sid)
            triples.add((date, action, sid))
            parse_stats["history_events"] += 1
            parse_stats["history_sids"] += 1
    return sid_set, triples, parse_stats


def parse_final_events(output: str) -> tuple[list[tuple[str, str, str]], str | None]:
    """Read the structured answer after the CoT block, not prose inside it."""
    candidate = output.split("</think>", 1)[-1].strip()
    try:
        payload = json.loads(candidate)
        events = payload["logic_chain"]["events"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], "invalid_final_json"
    parsed = []
    for event in events:
        date, action_text = event.get("date"), event.get("action")
        sid_match = SID_RE.search(str(action_text))
        action_match = re.search(r"\[([^\]]+)\]", str(action_text))
        if not isinstance(date, str) or sid_match is None or action_match is None:
            return [], "invalid_event_shape"
        parsed.append((date, action_match.group(1), sid_match.group(0)))
    return parsed, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--segment", choices=("user_chain_cot", "user_chain_nocot"), default="user_chain_cot")
    parser.add_argument("--show-samples", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    lengths = {"input": [], "output": [], "total": [], "answer_sids": []}
    sid_stats, event_stats = Counter(), Counter()
    samples = []
    rows = 0
    with args.dataset.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if row.get("data_source") != "understand_user" or row.get("source_segment") != args.segment:
                continue
            rows += 1
            input_text, output = str(row.get("input", "")), str(row.get("output", ""))
            input_ids = tokenizer("".join(str(row.get(key, "")) for key in ("system", "instruction", "input")), add_special_tokens=False)["input_ids"]
            output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
            lengths["input"].append(len(input_ids)); lengths["output"].append(len(output_ids)); lengths["total"].append(len(input_ids) + len(output_ids))
            history_sids, history_events, parsed = parse_history_events(input_text)
            event_stats.update(parsed)
            output_matches = list(SID_RE.finditer(output))
            final_events, event_parse_error = parse_final_events(output)
            lengths["answer_sids"].append(len(final_events))
            if not final_events:
                sid_stats["rows_without_answer_sid"] += 1
            sid_stats["answer_sid_total"] += len(output_matches)
            outside = []
            mismatched_events = []
            for match in output_matches:
                sid = match.group(0)
                if sid in history_sids:
                    sid_stats["answer_sid_in_history"] += 1
                else:
                    sid_stats["answer_sid_outside_history"] += 1; outside.append(sid)
            if event_parse_error:
                event_stats[event_parse_error] += 1
            for date, action, sid in final_events:
                event_stats["final_answer_events"] += 1
                if (date, action, sid) in history_events:
                    event_stats["answer_events_exactly_match_history"] += 1
                else:
                    event_stats["answer_events_parseable_but_not_in_history"] += 1
                    sid_events = [event for event in history_events if event[2] == sid]
                    same_date = any(event[0] == date for event in sid_events)
                    same_action = any(event[1] == action for event in sid_events)
                    if same_date and same_action:
                        event_stats["event_mismatch_nonidentical_duplicate_history_event"] += 1
                    elif same_date:
                        event_stats["event_mismatch_action_only"] += 1
                    elif same_action:
                        event_stats["event_mismatch_date_only"] += 1
                    else:
                        event_stats["event_mismatch_date_and_action"] += 1
                    mismatched_events.append(
                        {
                            "answer": {"date": date, "action": action, "sid": sid},
                            "history_candidates_for_sid": [
                                {"date": item[0], "action": item[1]} for item in sorted(sid_events)
                            ],
                        }
                    )
            if outside: sid_stats["rows_with_sid_outside_history"] += 1
            if mismatched_events: event_stats["rows_with_event_mismatch"] += 1
            if (outside or mismatched_events) and len(samples) < 20:
                samples.append({"line_number": line_number, "outside_sids": sorted(set(outside)), "mismatched_events": mismatched_events})

    report = {
        "scope": f"data_source=understand_user AND source_segment={args.segment}",
        "rows": rows,
        "token_counting": "raw system+instruction+input and raw output, add_special_tokens=false; chat-template framing excluded",
        "token_distribution": {name: quantile_summary(values) for name, values in lengths.items() if name != "answer_sids"},
        "answer_sid_count": quantile_summary(lengths["answer_sids"]),
        "answer_sid_histogram": {str(count): lengths["answer_sids"].count(count) for count in range(max(lengths["answer_sids"]) + 1)},
        "sid_membership": dict(sid_stats),
        "event_alignment": dict(event_stats),
        "mismatch_examples": samples,
    }
    if args.show_samples:
        report["sample_outputs"] = samples
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
