"""CPU-only controlled audit for GR_USER_v1 rewards and span attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from user_action_reward import score_action
from user_chain_reward import score_chain
from user_common import SID_RE
from user_span_attribution import TokenSpanMapper, sid_component_token_spans


EXPECTED_SHA = {
    "train_3000.jsonl": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "pilot_600.jsonl": "1b846a0d434d529136b4d9784be3b913d7aaf3d0c8c2c87ce470ba53679a4803",
    "probe_v1.jsonl": "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61",
}
HALL_SID = "<|video_begin|><s_a_999999><s_b_999998><s_c_999997>"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_action(sids):
    return json.dumps(list(sids), ensure_ascii=False)


def dump_chain(events, name="controlled audit"):
    return json.dumps({"logic_chain": {"name": name, "events": events}}, ensure_ascii=False)


def action_variants(row):
    gold = list(row["gold_sids"])
    wrong = next((sid for sid in row["history_sids"] if sid not in set(gold)), None)
    drop_count = max(1, round(len(gold) * 0.2))
    output = {
        "clean": dump_action(gold),
        "mild_drop_20pct": dump_action(gold[:-drop_count]),
        "severe_empty": "[]",
        "hallucinated_sid": dump_action(gold + [HALL_SID]),
        "duplicate_sid": dump_action(gold + ([gold[0]] if gold else [])),
        "malformed": dump_action(gold) + " trailing",
    }
    if wrong:
        output["wrong_history_selection"] = dump_action(gold + [wrong])
        output["random_history_sid"] = dump_action([wrong])
    return output


def chain_variants(row):
    gold = [dict(item) for item in row["gold_events"]]
    severe = [dict(item, action="完全无关动作", logic="完全无关逻辑") for item in gold]
    hall = [dict(item) for item in gold]
    if hall:
        hall[0]["action"] += " " + HALL_SID
    wrong_date = [dict(item) for item in gold]
    if wrong_date:
        wrong_date[0]["date"] = "2099-12-31"
    output = {
        "clean": dump_chain(gold),
        "mild_drop_one": dump_chain(gold[:-1]),
        "severe_unrelated": dump_chain(severe),
        "hallucinated_sid": dump_chain(hall),
        "wrong_date": dump_chain(wrong_date),
        "duplicate_event": dump_chain(gold + ([dict(gold[-1])] if gold else [])),
        "malformed": dump_chain(gold) + " trailing",
    }
    if gold:
        weak_logic = [dict(item, logic="相关需求进一步发展") for item in gold]
        output["weak_logic"] = dump_chain(weak_logic)
        other = next((item for item in row["history_events"] if item.get("raw") != gold[0]["action"]), None)
        if other:
            changed_action = [dict(item) for item in gold]
            changed_action[0]["action"] = other["raw"]
            output["other_history_action"] = dump_chain(changed_action)
    if len(gold) >= 2:
        swapped = [dict(item) for item in gold]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        output["swapped_order"] = dump_chain(swapped)
    if gold:
        excess = [dict(gold[i % len(gold)], date=f"2098-01-{i + 1:02d}", action=f"[搜索] excess-{i}") for i in range(6)]
        output["excess_events"] = dump_chain(excess)
    return output


def summarize_controlled(rows):
    values = defaultdict(list)
    examples = {"action": [], "chain": []}
    ordering = defaultdict(lambda: {"pass": 0, "total": 0})
    for row in rows:
        variants = action_variants(row) if row["route"] == "action" else chain_variants(row)
        scorer = score_action if row["route"] == "action" else score_chain
        results = {}
        for name, completion in variants.items():
            result = scorer(completion, row)
            results[name] = result
            values[(row["route"], name)].append(result.reward)
        severe_name = "severe_empty" if row["route"] == "action" else "severe_unrelated"
        ordering[row["route"]]["total"] += 1
        mild_name = "mild_drop_20pct" if row["route"] == "action" else "mild_drop_one"
        ordering[row["route"]]["pass"] += int(results["clean"].reward > results[mild_name].reward > results[severe_name].reward)
        if len(examples[row["route"]]) < 5:
            examples[row["route"]].append(
                {
                    "sample_id": row["sample_id"],
                    "source": "probe" if row.get("_source") == "probe" else "pilot",
                    "gold_count": len(row["gold_sids"] if row["route"] == "action" else row["gold_events"]),
                    "variant_rewards": {name: round(result.reward, 6) for name, result in results.items()},
                    "variant_violation_kinds": {
                        name: sorted({item.kind for item in result.violations}) for name, result in results.items()
                    },
                }
            )
    aggregates = {}
    for (route, name), scores in sorted(values.items()):
        aggregates.setdefault(route, {})[name] = {
            "count": len(scores),
            "mean_reward": statistics.fmean(scores),
            "min_reward": min(scores),
            "max_reward": max(scores),
        }
    return aggregates, dict(ordering), examples


def locality_audit(rows, tokenizer, route, perturbation, count=50):
    selected = [row for row in rows if row["route"] == route][:count]
    records = []
    for row in selected:
        if route == "action" and perturbation == "hallucination":
            target_value = HALL_SID
            completion = dump_action(list(row["gold_sids"]) + [target_value])
            result = score_action(completion, row, tokenizer)
            target_kind = "hallucinated_sid"
        elif route == "action":
            target_value = row["gold_sids"][0]
            completion = dump_action(list(row["gold_sids"]) + [target_value])
            result = score_action(completion, row, tokenizer)
            target_kind = "duplicate_sid"
        elif perturbation == "hallucination":
            events = [dict(item) for item in row["gold_events"]]
            events[0]["action"] += " " + HALL_SID
            completion = dump_chain(events)
            result = score_chain(completion, row, tokenizer)
            target_value = HALL_SID
            target_kind = "hallucinated_sid"
        else:
            events = [dict(item) for item in row["gold_events"]]
            target_value = dict(events[-1])
            completion = dump_chain(events + [target_value])
            result = score_chain(completion, row, tokenizer)
            target_kind = "duplicate_event"
        mapper = TokenSpanMapper(tokenizer, completion)
        local = [item for item in result.violations if item.kind == target_kind]
        penalized = set()
        for item in local:
            penalized.update(range(item.token_start or 0, item.token_end or 0))
        exact = False
        if len(local) == 1:
            raw_target = completion[local[0].char_start:local[0].char_end]
            exact = raw_target == target_value if isinstance(target_value, str) else json.loads(raw_target) == target_value
        records.append(
            {
                "sample_id": row["sample_id"],
                "completion_tokens": len(mapper.input_ids),
                "attributed_tokens": len(penalized),
                "fraction": len(penalized) / max(1, len(mapper.input_ids)),
                "only_target_occurrence_span": exact,
            }
        )
    fractions = [item["fraction"] for item in records]
    return {
        "count": len(records),
        "perturbation": perturbation,
        "violation_kind": target_kind,
        "all_local_span_exact": all(item["only_target_occurrence_span"] for item in records),
        "mean_attributed_fraction": statistics.fmean(fractions),
        "max_attributed_fraction": max(fractions),
        "examples": records[:5],
    }


def tokenizer_audit(rows, tokenizer):
    unique = []
    seen = set()
    for row in rows:
        for sid in row["history_sids"]:
            if sid not in seen:
                unique.append(sid)
                seen.add(sid)
            if len(unique) == 100:
                break
        if len(unique) == 100:
            break
    records = []
    for sid in unique:
        spans = sid_component_token_spans(tokenizer, sid)
        full = tokenizer(sid, add_special_tokens=False, return_offsets_mapping=True)
        records.append(
            {
                "sid": sid,
                "token_count": len(full["input_ids"]),
                "roundtrip": tokenizer.decode(full["input_ids"]) == sid,
                "component_token_counts": {name: span.end - span.start for name, span in spans.items()},
                "component_nonempty": all(span.end > span.start for span in spans.values()),
            }
        )
    return {
        "count": len(records),
        "all_roundtrip": all(item["roundtrip"] for item in records),
        "all_components_nonempty": all(item["component_nonempty"] for item in records),
        "token_count_distribution": dict(sorted({count: sum(item["token_count"] == count for item in records) for count in {item["token_count"] for item in records}}.items())),
        "examples": records[:5],
    }


def benchmark(rows, tokenizer):
    output = {}
    for route, scorer in (("action", score_action), ("chain", score_chain)):
        selected = [row for row in rows if row["route"] == route][:300]
        start = time.perf_counter()
        for row in selected:
            scorer(row["raw_gold_output"], row)
        score_seconds = time.perf_counter() - start
        span_selected = selected[:100]
        start = time.perf_counter()
        for row in span_selected:
            scorer(row["raw_gold_output"], row, tokenizer)
        span_seconds = time.perf_counter() - start
        output[route] = {
            "score_count": len(selected),
            "score_total_seconds": score_seconds,
            "score_ms_per_sample": score_seconds * 1000 / len(selected),
            "score_samples_per_second": len(selected) / score_seconds,
            "score_plus_span_count": len(span_selected),
            "score_plus_span_ms_per_sample": span_seconds * 1000 / len(span_selected),
            "span_attribution_incremental_ms_per_sample": max(
                0.0,
                span_seconds * 1000 / len(span_selected) - score_seconds * 1000 / len(selected),
            ),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    sha_before = {name: sha256(args.data_dir / name) for name in EXPECTED_SHA}
    if sha_before != EXPECTED_SHA:
        raise SystemExit(f"frozen SHA mismatch: {sha_before}")
    pilot = read_jsonl(args.data_dir / "pilot_600.jsonl")
    probe = read_jsonl(args.data_dir / "probe_v1.jsonl")
    for row in pilot:
        row["_source"] = "pilot"
    for row in probe:
        row["_source"] = "probe"
    tokenizer = AutoTokenizer.from_pretrained(args.parent_checkpoint, local_files_only=True)
    if not tokenizer.is_fast:
        raise SystemExit("parent tokenizer must expose exact offset mappings")

    controlled, ordering, examples = summarize_controlled(probe + pilot)
    result = {
        "contract_version": "gr_user_reward_v1",
        "execution": {"cpu_only": True, "generation": False, "training": False},
        "evaluator_evidence": {
            "official_implementation_found": False,
            "known": "Task name and EvolutionTopicGenEvaluator class name only.",
            "approximate": "Normalized action token-set F1, action-driven monotonic DP, and 0.5 token-set F1 + 0.5 ROUGE-L-F1 logic similarity.",
        },
        "frozen_sha_before": sha_before,
        "controlled_perturbations": controlled,
        "clean_mild_severe_ordering": ordering,
        "real_examples": examples,
        "penalty_locality": {
            "action": {
                "hallucination": locality_audit(pilot, tokenizer, "action", "hallucination", 50),
                "duplicate": locality_audit(pilot, tokenizer, "action", "duplicate", 50),
            },
            "chain": {
                "hallucination": locality_audit(pilot, tokenizer, "chain", "hallucination", 50),
                "duplicate": locality_audit(pilot, tokenizer, "chain", "duplicate", 50),
            },
        },
        "tokenizer_audit": tokenizer_audit(pilot + probe, tokenizer),
        "cpu_benchmark": benchmark(pilot, tokenizer),
    }
    sha_after = {name: sha256(args.data_dir / name) for name in EXPECTED_SHA}
    result["frozen_sha_after"] = sha_after
    result["frozen_sha_unchanged"] = sha_before == sha_after == EXPECTED_SHA
    if not result["frozen_sha_unchanged"]:
        raise SystemExit("frozen data changed during read-only audit")
    if result["tokenizer_audit"]["count"] != 100 or not result["tokenizer_audit"]["all_roundtrip"]:
        raise SystemExit("tokenizer audit failed")
    locality_items = [result["penalty_locality"][route][kind] for route in ("action", "chain") for kind in ("hallucination", "duplicate")]
    if any(item["count"] < 50 for item in locality_items):
        raise SystemExit("penalty locality sample count failed")
    if not all(item["all_local_span_exact"] for item in locality_items):
        raise SystemExit("penalty locality span failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "contract_version": result["contract_version"],
        "execution": result["execution"],
        "frozen_sha_unchanged": result["frozen_sha_unchanged"],
        "clean_mild_severe_ordering": ordering,
        "penalty_locality": result["penalty_locality"],
        "tokenizer_audit": result["tokenizer_audit"],
        "cpu_benchmark": result["cpu_benchmark"],
        "evaluator_evidence": result["evaluator_evidence"],
        "real_examples": examples,
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("frozen_sha_unchanged", "clean_mild_severe_ordering", "cpu_benchmark")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
