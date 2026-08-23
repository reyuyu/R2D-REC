"""Four-GPU inference-only audit of history SID copying in two Beam modes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import torch

from .probe1_beam_anatomy import (
    ADAPTER,
    BASE,
    DOMAINS,
    closed_cot,
    encode_prompt,
    generate_batch,
    load_model,
    parse_gold,
)
from grpo_sid import final_sid

GRPO_ROOT = Path(__file__).resolve().parents[3]
PROBES = Path(
    "/data/GRPO/runs/GR-REC-THINK-COMPOSITE-INTEREST-V1-ABC3-"
    "FORMAL716-20260823/probes.jsonl"
)
DEFAULT_PARTS = Path("/data/GRPO/results/history_sid_copy_audit_20260823_parts")
DEFAULT_OUTPUT = GRPO_ROOT / (
    "results/gr_rec_think_composite_interest_v1_history_sid_copy_audit_20260823.json"
)
STAGE_PATTERNS = tuple(re.compile(rf"<s_{stage}_(\d+)>").fullmatch for stage in "abc")


def load_step0_probes():
    rows = []
    with PROBES.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("step") == 0:
                rows.append(row)
    rows.sort(key=lambda row: (int(row["probe_round"]), tuple(DOMAINS).index(row["target_domain"])))
    domains = Counter(row["target_domain"] for row in rows)
    if len(rows) != 12 or domains != Counter({domain: 3 for domain in DOMAINS}):
        raise RuntimeError(f"STEP0_PROBE_CONTRACT_INVALID rows={len(rows)} domains={domains}")
    if any(len(row["think"]["candidates"]) != 4 for row in rows):
        raise RuntimeError("STEP0_COT_CONTRACT_INVALID")
    return rows


class RawSidParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.domain_by_id = {}
        for domain, token in DOMAINS.items():
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"DOMAIN_TOKEN_NOT_ATOMIC domain={domain} ids={ids}")
            self.domain_by_id[ids[0]] = domain

    def scan(self, ids):
        found = []
        for start in range(max(0, len(ids) - 3)):
            domain = self.domain_by_id.get(ids[start])
            if domain is None:
                continue
            tokens = self.tokenizer.convert_ids_to_tokens(
                ids[start + 1 : start + 4], skip_special_tokens=False
            )
            if isinstance(tokens, str):
                tokens = [tokens]
            matches = [pattern(token) for pattern, token in zip(STAGE_PATTERNS, tokens)]
            if len(tokens) != 3 or any(match is None for match in matches):
                continue
            sid = (domain, *(int(match.group(1)) for match in matches))
            found.append({"sid": sid, "start": start, "end": start + 3})
        return found

    def parse_abc3(self, domain, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        if isinstance(tokens, str):
            tokens = [tokens]
        matches = [pattern(token) for pattern, token in zip(STAGE_PATTERNS, tokens)]
        if len(ids) != 3 or len(tokens) != 3 or any(match is None for match in matches):
            return None
        return (domain, *(int(match.group(1)) for match in matches))


def sid_list(sid):
    return list(sid) if sid is not None else None


def history_from_prompt(parser, prompt_ids):
    occurrences = parser.scan(prompt_ids)
    if not occurrences:
        raise RuntimeError("MODEL_VISIBLE_PROMPT_HAS_NO_HISTORY_SID")
    grouped = {}
    order = []
    for item in occurrences:
        sid = item["sid"]
        if sid not in grouped:
            order.append(sid)
            grouped[sid] = []
        grouped[sid].append(item["start"])
    records = []
    for sid in order:
        positions = grouped[sid]
        records.append({
            "sid": sid_list(sid),
            "occurrence_count": len(positions),
            "positions": positions,
            "last_position": positions[-1],
            "distance_from_history_end": len(prompt_ids) - 1 - positions[-1],
        })
    last = occurrences[-1]["sid"]
    lookup = {tuple(row["sid"]): row for row in records}
    return {
        "history_sid_occurrences": [
            {"sid": sid_list(item["sid"]), "position": item["start"]}
            for item in occurrences
        ],
        "history_sid_unique_order": [sid_list(sid) for sid in order],
        "history_sid_unique_records": records,
        "history_sid_unique_count": len(order),
        "most_recent_history_sid": sid_list(last),
        "lookup": lookup,
        "unique_set": set(order),
    }


def copy_class(sid, history_set):
    if sid is None:
        return None
    if sid in history_set:
        return "EXACT_COPY"
    if any(old[:3] == sid[:3] for old in history_set):
        return "AB_COPY"
    if any(old[:2] == sid[:2] for old in history_set):
        return "A_COPY"
    return "NOVEL"


def classify_prediction(sid, history, gold_set):
    category = copy_class(sid, history["unique_set"])
    is_gold = sid in gold_set if sid is not None else False
    is_recent = sid == tuple(history["most_recent_history_sid"]) if sid is not None else False
    if is_gold and category == "EXACT_COPY":
        cross = "GOLD_AND_HISTORY"
    elif is_gold:
        cross = "GOLD_NOT_HISTORY"
    elif category == "EXACT_COPY":
        cross = "HISTORY_NOT_GOLD"
    else:
        cross = "NEITHER"
    history_record = history["lookup"].get(sid) if category == "EXACT_COPY" else None
    return {
        "predicted_sid": sid_list(sid),
        "copy_class": category,
        "is_gold_exact": is_gold,
        "is_most_recent_exact_copy": is_recent,
        "gold_history_class": cross,
        "history_occurrence_count": (
            history_record["occurrence_count"] if history_record else None
        ),
        "history_distance_from_end": (
            history_record["distance_from_history_end"] if history_record else None
        ),
    }


def ordinary_beams(model, tokenizer, parser, context, target_domain, history, gold_set):
    texts, raw_ids = generate_batch(
        model,
        tokenizer,
        [context],
        max_new_tokens=128,
        do_sample=False,
        num_beams=32,
        num_return_sequences=32,
        return_ids=True,
    )
    rows = []
    for index, (text, ids) in enumerate(zip(texts, raw_ids)):
        generated = parser.scan(ids)
        first = generated[0] if generated else None
        first_target = next(
            (item for item in generated if item["sid"][0] == target_domain),
            None,
        )
        legacy = final_sid(text)
        rows.append({
            "beam_index": index,
            "raw_generated_token_ids": ids,
            "all_generated_sids": [
                {"sid": sid_list(item["sid"]), "start": item["start"], "end": item["end"]}
                for item in generated
            ],
            "first_any_sid": {
                **classify_prediction(first["sid"] if first else None, history, gold_set),
                "start_token_position": first["start"] if first else None,
                "end_token_position": first["end"] if first else None,
            },
            "first_target_domain_sid": classify_prediction(
                first_target["sid"] if first_target else None, history, gold_set
            ),
            "legacy_final_sid": classify_prediction(legacy, history, gold_set),
        })
    return rows


def fixed_beams(model, tokenizer, parser, context, domain, history, gold_set):
    domain_ids = tokenizer.encode(DOMAINS[domain], add_special_tokens=False)
    texts, raw_ids = generate_batch(
        model,
        tokenizer,
        [context + domain_ids],
        min_new_tokens=3,
        max_new_tokens=3,
        do_sample=False,
        num_beams=32,
        num_return_sequences=32,
        return_ids=True,
    )
    return [
        {
            "beam_index": index,
            "raw_generated_token_ids": ids,
            **classify_prediction(parser.parse_abc3(domain, ids), history, gold_set),
        }
        for index, ids in enumerate(raw_ids)
    ]


def serializable_history(history):
    return {
        key: value
        for key, value in history.items()
        if key not in {"lookup", "unique_set"}
    }


def run_rank(rank, parts_dir):
    probes = load_step0_probes()
    torch.cuda.set_device(rank)
    model, tokenizer, _ = load_model(device=f"cuda:{rank}")
    parser = RawSidParser(tokenizer)
    part_path = parts_dir / f"rank{rank}.json"
    groups = []
    started = time.perf_counter()
    for probe_index, probe in enumerate(probes):
        candidate = probe["think"]["candidates"][rank]
        prompt_ids = encode_prompt(tokenizer, probe["think_prompt"])
        history = history_from_prompt(parser, prompt_ids)
        gold_set = {parse_gold(value) for value in probe["gold_sids"]}
        gold_in_history = gold_set & history["unique_set"]
        cot_ids = tokenizer.encode(
            closed_cot(candidate["completion"]), add_special_tokens=False
        )
        context = prompt_ids + cot_ids
        with torch.inference_mode():
            ordinary = ordinary_beams(
                model, tokenizer, parser, context, probe["target_domain"], history, gold_set
            )
            fixed = fixed_beams(
                model,
                tokenizer,
                parser,
                context,
                probe["target_domain"],
                history,
                gold_set,
            )
        groups.append({
            "probe_index": probe_index,
            "probe_round": probe["probe_round"],
            "recommendation_group_id": probe["group_id"],
            "target_domain": probe["target_domain"],
            "candidate_id": rank,
            "completion_sha256": candidate["completion_sha256"],
            "prompt_token_count": len(prompt_ids),
            "cot_token_count": len(cot_ids),
            "history": serializable_history(history),
            "gold_count": len(gold_set),
            "gold_in_history_count": len(gold_in_history),
            "gold_in_history_sids": [sid_list(sid) for sid in sorted(gold_in_history)],
            "ordinary_beams": ordinary,
            "fixed_abc3_beams": fixed,
        })
        payload = {
            "rank": rank,
            "device": f"cuda:{rank}",
            "model_parent": {"base": BASE, "adapter": ADAPTER, "fresh_original_bata": True},
            "groups": groups,
            "elapsed_sec": time.perf_counter() - started,
            "training_started": False,
            "optimizer_created": False,
        }
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(
            f"COPY_AUDIT_PROGRESS rank={rank} group={probe_index + 1}/12 "
            f"domain={probe['target_domain']} elapsed={payload['elapsed_sec']:.1f}s",
            flush=True,
        )


def flatten(parts, mode, selector=None):
    rows = []
    for part in parts:
        for group in part["groups"]:
            for beam in group[mode]:
                prediction = selector(beam) if selector else beam
                rows.append({
                    **prediction,
                    "target_domain": group["target_domain"],
                    "candidate_key": (
                        group["recommendation_group_id"], group["candidate_id"]
                    ),
                })
    return rows


def ratio(value, denominator):
    return value / denominator if denominator else None


def aggregate(rows):
    total = len(rows)
    valid = [row for row in rows if row["predicted_sid"] is not None]
    classes = Counter(row["copy_class"] for row in valid)
    cross = Counter(row["gold_history_class"] for row in rows)
    recent = sum(row["is_most_recent_exact_copy"] for row in valid)
    unique = {tuple(row["predicted_sid"]) for row in valid}
    unique_exact = {
        tuple(row["predicted_sid"])
        for row in valid
        if row["copy_class"] == "EXACT_COPY"
    }
    candidate_keys = {row["candidate_key"] for row in rows}
    copy_candidate_keys = {
        row["candidate_key"]
        for row in rows
        if row["copy_class"] == "EXACT_COPY"
    }
    return {
        "total_beams": total,
        "valid_predictions": len(valid),
        "valid_rate": ratio(len(valid), total),
        "exact_copy_count": classes["EXACT_COPY"],
        "exact_copy_rate_all": ratio(classes["EXACT_COPY"], total),
        "exact_copy_rate_valid": ratio(classes["EXACT_COPY"], len(valid)),
        "ab_copy_rate_valid": ratio(classes["AB_COPY"], len(valid)),
        "a_copy_rate_valid": ratio(classes["A_COPY"], len(valid)),
        "novel_rate_valid": ratio(classes["NOVEL"], len(valid)),
        "most_recent_copy_rate": ratio(recent, len(valid)),
        "unique_predicted_sid_count": len(unique),
        "unique_exact_copy_sid_count": len(unique_exact),
        "unique_sid_exact_copy_rate": ratio(len(unique_exact), len(unique)),
        "gold_and_history": ratio(cross["GOLD_AND_HISTORY"], total),
        "gold_not_history": ratio(cross["GOLD_NOT_HISTORY"], total),
        "history_not_gold": ratio(cross["HISTORY_NOT_GOLD"], total),
        "neither": ratio(cross["NEITHER"], total),
        "candidate_count": len(candidate_keys),
        "candidate_any_exact_copy_count": len(copy_candidate_keys),
        "any_copy_beam32": ratio(len(copy_candidate_keys), len(candidate_keys)),
    }


def candidate_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["candidate_key"], []).append(row)
    output = []
    for (group_id, candidate_id), values in sorted(grouped.items()):
        counts = Counter(row["copy_class"] for row in values if row["copy_class"])
        output.append({
            "recommendation_group_id": group_id,
            "candidate_id": candidate_id,
            "exact_copy_count": counts["EXACT_COPY"],
            "ab_copy_count": counts["AB_COPY"],
            "a_copy_count": counts["A_COPY"],
            "novel_count": counts["NOVEL"],
            "invalid_count": sum(row["predicted_sid"] is None for row in values),
            "candidate_any_exact_copy": counts["EXACT_COPY"] > 0,
        })
    return output


def ordinary_position_and_count(parts):
    beams = [
        beam
        for part in parts
        for group in part["groups"]
        for beam in group["ordinary_beams"]
    ]
    positions = [
        beam["first_any_sid"]["start_token_position"]
        for beam in beams
        if beam["first_any_sid"]["start_token_position"] is not None
    ]
    sid_counts = [len(beam["all_generated_sids"]) for beam in beams]
    buckets = Counter("0" if value == 0 else "1" if value == 1 else "2+" for value in sid_counts)
    return {
        "first_sid_start_token_position": {
            "mean": statistics.mean(positions) if positions else None,
            "median": statistics.median(positions) if positions else None,
        },
        "first_sid_end_token_position": {
            "mean": statistics.mean(value + 3 for value in positions) if positions else None,
            "median": statistics.median(value + 3 for value in positions) if positions else None,
        },
        "sid_count_per_beam": {
            key: {"count": buckets[key], "rate": ratio(buckets[key], len(beams))}
            for key in ("0", "1", "2+")
        },
    }


def merge(parts_dir, output):
    parts = [
        json.loads((parts_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(4)
    ]
    if any(len(part["groups"]) != 12 for part in parts):
        raise RuntimeError("INCOMPLETE_RANK_PART")
    ordinary_first = flatten(parts, "ordinary_beams", lambda beam: beam["first_any_sid"])
    ordinary_target = flatten(parts, "ordinary_beams", lambda beam: beam["first_target_domain_sid"])
    ordinary_final = flatten(parts, "ordinary_beams", lambda beam: beam["legacy_final_sid"])
    fixed = flatten(parts, "fixed_abc3_beams")
    gold_count = sum(group["gold_count"] for group in parts[0]["groups"])
    gold_in_history_count = sum(
        group["gold_in_history_count"] for group in parts[0]["groups"]
    )
    per_domain = {}
    for domain in DOMAINS:
        per_domain[domain] = {
            "ordinary_first_sid": aggregate(
                [row for row in ordinary_first if row["target_domain"] == domain]
            ),
            "fixed_abc3": aggregate(
                [row for row in fixed if row["target_domain"] == domain]
            ),
        }
    ordinary_summary = aggregate(ordinary_first)
    fixed_summary = aggregate(fixed)
    comparison_keys = (
        "valid_rate", "exact_copy_rate_all", "exact_copy_rate_valid",
        "ab_copy_rate_valid", "a_copy_rate_valid", "novel_rate_valid",
        "most_recent_copy_rate", "unique_sid_exact_copy_rate",
        "gold_and_history", "gold_not_history", "history_not_gold",
        "any_copy_beam32",
    )
    result = {
        "type": "history_sid_copy_audit",
        "diagnostic_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=GRPO_ROOT, text=True
        ).strip(),
        "model_parent": parts[0]["model_parent"],
        "probe_source": str(PROBES),
        "groups": 12,
        "cots": 48,
        "beams_per_mode": 1536,
        "history_extraction_source": "MODEL_VISIBLE_PROMPT_ONLY",
        "gold_history_overlap": {
            "gold_count": gold_count,
            "gold_in_history_count": gold_in_history_count,
            "gold_in_history_rate": ratio(gold_in_history_count, gold_count),
            "per_probe": [
                {
                    "recommendation_group_id": group["recommendation_group_id"],
                    "target_domain": group["target_domain"],
                    "gold_count": group["gold_count"],
                    "gold_in_history_count": group["gold_in_history_count"],
                    "gold_in_history_rate": ratio(group["gold_in_history_count"], group["gold_count"]),
                }
                for group in parts[0]["groups"]
            ],
        },
        "ordinary_first_sid": ordinary_summary,
        "ordinary_first_target_domain_sid": aggregate(ordinary_target),
        "ordinary_legacy_final_sid": aggregate(ordinary_final),
        "fixed_abc3": fixed_summary,
        "fixed_minus_ordinary_first_sid": {
            key: fixed_summary[key] - ordinary_summary[key]
            for key in comparison_keys
            if fixed_summary[key] is not None and ordinary_summary[key] is not None
        },
        "per_domain": per_domain,
        "ordinary_generation_shape": ordinary_position_and_count(parts),
        "candidate_summary": {
            "ordinary_first_sid": candidate_summary(ordinary_first),
            "fixed_abc3": candidate_summary(fixed),
        },
        "rank_parts": parts,
        "gpu_used": 4,
        "training_started": False,
        "production_code_changed": False,
    }
    if len(ordinary_first) != 1536 or len(fixed) != 1536:
        raise RuntimeError("BEAM_COUNT_CONTRACT_INVALID")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "gold_history_overlap": result["gold_history_overlap"],
        "ordinary_first_sid": ordinary_summary,
        "ordinary_legacy_final_sid": result["ordinary_legacy_final_sid"],
        "fixed_abc3": fixed_summary,
        "fixed_minus_ordinary_first_sid": result["fixed_minus_ordinary_first_sid"],
        "ordinary_generation_shape": result["ordinary_generation_shape"],
    }, ensure_ascii=False, indent=2))
    print(f"HISTORY_COPY_AUDIT_RESULT={output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args.parts_dir, args.output)
        return
    rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    if world != 4 or rank not in range(4):
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    run_rank(rank, args.parts_dir)


if __name__ == "__main__":
    main()
