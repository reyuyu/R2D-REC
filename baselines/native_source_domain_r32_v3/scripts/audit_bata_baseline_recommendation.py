#!/usr/bin/env python3
"""Read-only audit for beta-baseline-v1 recommendation rows."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


SID_RE = re.compile(r"<\|(?P<domain>ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
DOMAINS = ("ad", "prod", "living", "video")
REQUIRED_COT_SECTIONS = ("【兴趣归纳】", "【行为模式】", "【预测总结】")
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def summary(values):
    values = sorted(values)
    count = len(values)
    if not count:
        return {"count": 0, "mean": None, "quantiles": {}, "over_threshold": {}}

    def q(value):
        return values[max(0, min(count - 1, math.ceil(value * count) - 1))]

    return {
        "count": count,
        "mean": round(sum(values) / count, 2),
        "quantiles": {f"p{int(value * 100):02d}": q(value) for value in QUANTILES},
        "over_threshold": {str(limit): sum(item > limit for item in values) for limit in (4096, 6144, 8192)},
    }


def mode(row):
    return "cot" if row["source_segment"] == "recommendation_cot" else "nocot"


def canonical_history(row):
    # Instruction is the user-history sequence; only its mode suffix differs across routes.
    instruction = re.sub(r"/(?:think|no_think)\s*$", "", str(row.get("instruction", ""))).rstrip()
    return json.dumps([row.get("system", ""), instruction, row.get("input", ""), row.get("history", [])], ensure_ascii=False, separators=(",", ":"))


def cot_status(output):
    if "<think>" not in output or "</think>" not in output:
        return "missing_or_unclosed_think"
    thought = output.split("<think>", 1)[1].split("</think>", 1)[0]
    missing = [section for section in REQUIRED_COT_SECTIONS if section not in thought]
    if missing:
        return "missing_sections:" + ",".join(missing)
    return "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows_by_mode = Counter()
    answer_domain = defaultdict(Counter)
    current_gold_domain = defaultdict(Counter)
    gold_domain = defaultdict(Counter)
    answer_sid_counts = defaultdict(list)
    token_lengths = defaultdict(lambda: {"input": [], "output": [], "total": []})
    history_sid = {}
    group_records = defaultdict(list)
    metadata_groups = defaultdict(list)
    cot_row_status = Counter()
    cot_bad_history_groups = set()
    cot_bad_metadata_groups = set()
    sid_membership = defaultdict(Counter)
    current_gold_membership = defaultdict(Counter)
    final_output_alignment = defaultdict(Counter)
    metadata_issues = Counter()

    with args.dataset.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("data_source") != "recommend":
                continue
            current_mode = mode(row)
            rows_by_mode[current_mode] += 1
            output = str(row.get("output", ""))
            prompt = "".join(str(row.get(key, "")) for key in ("system", "instruction", "input"))
            input_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            output_tokens = len(tokenizer(output, add_special_tokens=False)["input_ids"])
            for label, value in (("input", input_tokens), ("output", output_tokens), ("total", input_tokens + output_tokens)):
                token_lengths[current_mode][label].append(value)

            answer_sids = list(SID_RE.finditer(output))
            answer_sid_counts[current_mode].append(len(answer_sids))
            for match in answer_sids:
                answer_domain[current_mode][match.group("domain")] += 1

            hist_key = canonical_history(row)
            if hist_key not in history_sid:
                history_text = str(row.get("instruction", "")) + str(row.get("input", ""))
                history_sid[hist_key] = {match.group(0) for match in SID_RE.finditer(history_text)}
            answer_sid_texts = [match.group(0) for match in answer_sids]
            sid_membership[current_mode]["answer_sid_total"] += len(answer_sid_texts)
            sid_membership[current_mode]["answer_sid_in_history"] += sum(sid in history_sid[hist_key] for sid in answer_sid_texts)
            sid_membership[current_mode]["rows_with_any_answer_sid_in_history"] += int(any(sid in history_sid[hist_key] for sid in answer_sid_texts))
            sid_membership[current_mode]["rows_with_answer_sid"] += int(bool(answer_sid_texts))

            try:
                metadata = json.loads(row.get("aux_metadata_json", ""))
                all_gold = metadata["recommendation_all_gold_sids"]
                current_gold = metadata["recommendation_current_gold_sid"]
                metadata_group_id = metadata["recommendation_group_id"]
                declared_size = metadata["recommendation_group_size"]
                if not isinstance(all_gold, list) or not all_gold or current_gold not in all_gold:
                    raise ValueError("invalid_gold_list")
                if len(set(all_gold)) != len(all_gold):
                    metadata_issues["duplicate_sid_in_all_gold"] += 1
                if declared_size != len(all_gold):
                    metadata_issues["declared_group_size_mismatch"] += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                metadata_issues["invalid_or_missing_metadata"] += 1
                all_gold, current_gold, metadata_group_id = [], None, None

            for sid in all_gold:
                domain_match = SID_RE.fullmatch(sid)
                if domain_match:
                    gold_domain[current_mode][domain_match.group("domain")] += 1
                else:
                    metadata_issues["malformed_gold_sid"] += 1
            if current_gold:
                domain = SID_RE.fullmatch(current_gold)
                domain_name = domain.group("domain") if domain else "unknown"
                current_gold_domain[current_mode][domain_name] += 1
                current_gold_membership[current_mode]["current_gold_total"] += 1
                current_gold_membership[current_mode]["current_gold_in_history"] += int(current_gold in history_sid[hist_key])
                final_answer_text = output.split("</think>", 1)[-1]
                final_answer_sids = {match.group(0) for match in SID_RE.finditer(final_answer_text)}
                final_output_alignment[current_mode]["rows"] += 1
                final_output_alignment[current_mode]["current_gold_present_in_final_output"] += int(current_gold in final_answer_sids)
                group_records[(hist_key, domain_name)].append({"line": line_number, "mode": current_mode, "gold": current_gold, "all_gold": tuple(all_gold), "metadata_group_id": metadata_group_id})
                metadata_groups[metadata_group_id].append((hist_key, domain_name, current_gold, tuple(all_gold), current_mode, line_number))

            if current_mode == "cot":
                status = cot_status(output)
                cot_row_status[status] += 1
                if status != "complete":
                    cot_bad_history_groups.add((hist_key, domain_name))
                    if metadata_group_id:
                        cot_bad_metadata_groups.add(metadata_group_id)

    group_size = defaultdict(list)
    group_stats = defaultdict(Counter)
    metadata_group_conflicts = Counter()
    for (hist_key, domain), records in group_records.items():
        unique_gold = {item["gold"] for item in records}
        modes = {item["mode"] for item in records}
        group_size["all"].append(len(unique_gold))
        group_stats["all"]["groups"] += 1
        group_stats["all"]["rows"] += len(records)
        group_stats["all"]["singleton_groups"] += int(len(unique_gold) == 1)
        group_stats["all"]["multi_positive_groups"] += int(len(unique_gold) > 1)
        group_size[domain].append(len(unique_gold))
        group_stats[domain]["groups"] += 1
        group_stats[domain]["rows"] += len(records)
        group_stats[domain]["singleton_groups"] += int(len(unique_gold) == 1)
        group_stats[domain]["multi_positive_groups"] += int(len(unique_gold) > 1)
        if len(modes) > 1:
            group_stats["all"]["cross_mode_groups"] += 1
        declared_sets = {item["all_gold"] for item in records}
        declared_ids = {item["metadata_group_id"] for item in records}
        if len(declared_sets) != 1:
            metadata_group_conflicts["same_history_domain_has_inconsistent_all_gold"] += 1
        if len(declared_ids) != 1:
            metadata_group_conflicts["same_history_domain_has_multiple_metadata_group_id"] += 1

    metadata_id_stats = Counter()
    for group_id, records in metadata_groups.items():
        metadata_id_stats["groups"] += 1
        canonical_keys = {(item[0], item[1]) for item in records}
        declared_sets = {item[3] for item in records}
        if len(canonical_keys) != 1:
            metadata_group_conflicts["metadata_group_id_maps_multiple_history_domain"] += 1
        if len(declared_sets) != 1:
            metadata_group_conflicts["metadata_group_id_has_inconsistent_all_gold"] += 1

    def groups_report():
        result = {}
        for domain, values in group_size.items():
            count = len(values)
            singles = sum(value == 1 for value in values)
            result[domain] = {
                "groups": count,
                "positive_count_distribution": dict(sorted(Counter(values).items())),
                "mean_positive_count": round(sum(values) / count, 4) if count else 0,
                "singleton_groups": singles,
                "singleton_ratio": round(singles / count, 8) if count else 0,
                "multi_positive_groups": count - singles,
            }
        return result

    report = {
        "scope": "data_source=recommend in beta-baseline-v1",
        "rows": dict(rows_by_mode),
        "answer_sid_domain_distribution": {name: {domain: answer_domain[name][domain] for domain in DOMAINS} for name in ("cot", "nocot")},
        "current_gold_domain_distribution": {name: {domain: current_gold_domain[name][domain] for domain in DOMAINS} for name in ("cot", "nocot")},
        "metadata_all_gold_domain_distribution": {name: {domain: gold_domain[name][domain] for domain in DOMAINS} for name in ("cot", "nocot")},
        "token_counting": "raw system+instruction+input and raw output, add_special_tokens=false; chat-template framing excluded",
        "token_distribution": {name: {part: summary(values) for part, values in token_lengths[name].items()} for name in ("cot", "nocot")},
        "answer_sid_count_per_row": {name: summary(answer_sid_counts[name]) for name in ("cot", "nocot")},
        "deduplicated_history_domain_groups": groups_report(),
        "metadata_group_ids": dict(metadata_id_stats),
        "metadata_issues": dict(metadata_issues),
        "metadata_group_conflicts": dict(metadata_group_conflicts),
        "cot_completeness": {
            "rule": list(REQUIRED_COT_SECTIONS),
            "row_status": dict(cot_row_status),
            "incomplete_rows": sum(count for status, count in cot_row_status.items() if status != "complete"),
            "incomplete_deduplicated_history_domain_groups": len(cot_bad_history_groups),
            "incomplete_metadata_groups": len(cot_bad_metadata_groups),
        },
        "answer_sid_from_history": {
            "primary_current_gold": {
                name: {
                    **dict(current_gold_membership[name]),
                    "ratio": round(current_gold_membership[name]["current_gold_in_history"] / current_gold_membership[name]["current_gold_total"], 8) if current_gold_membership[name]["current_gold_total"] else 0,
                }
                for name in ("cot", "nocot")
            },
            "current_gold_present_in_final_output": {
                name: {
                    **dict(final_output_alignment[name]),
                    "ratio": round(final_output_alignment[name]["current_gold_present_in_final_output"] / final_output_alignment[name]["rows"], 8) if final_output_alignment[name]["rows"] else 0,
                }
                for name in ("cot", "nocot")
            },
            "all_output_sids_auxiliary_includes_cot_prose": {
            name: {
                **dict(sid_membership[name]),
                "sid_ratio": round(sid_membership[name]["answer_sid_in_history"] / sid_membership[name]["answer_sid_total"], 8) if sid_membership[name]["answer_sid_total"] else 0,
                "row_ratio": round(sid_membership[name]["rows_with_any_answer_sid_in_history"] / sid_membership[name]["rows_with_answer_sid"], 8) if sid_membership[name]["rows_with_answer_sid"] else 0,
            }
            for name in ("cot", "nocot")
            },
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
