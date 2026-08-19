"""CPU-only real-data locality audit for the User GRPO penalty-mask compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from user_action_reward import score_action
from user_chain_reward import score_chain
from user_common import SID_RE, SID_PART_RE
from user_penalty_mask import SID_COMPONENT_RE, compile_penalty_mask
from user_span_attribution import TokenSpanMapper


EXPECTED_SHA = {
    "train_3000.jsonl": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "pilot_600.jsonl": "1b846a0d434d529136b4d9784be3b913d7aaf3d0c8c2c87ce470ba53679a4803",
    "probe_v1.jsonl": "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_json(sids):
    return json.dumps(list(sids), ensure_ascii=False)


def chain_json(events):
    return json.dumps({"logic_chain": {"name": "penalty locality audit", "events": events}}, ensure_ascii=False)


def mask_for_spans(tokenizer, completion, spans):
    mapper = TokenSpanMapper(tokenizer, completion)
    mask = [False] * len(mapper.input_ids)
    for char_start, char_end in spans:
        token_span = mapper.map(char_start, char_end)
        for index in range(token_span.start, token_span.end):
            mask[index] = True
    return mask


def c_component_span(completion: str, sid: str):
    occurrence = next(match for match in SID_RE.finditer(completion) if match.group() == sid)
    component = SID_COMPONENT_RE.fullmatch(sid)
    if component is None:
        raise AssertionError(f"non-canonical real SID: {sid}")
    start, end = component.span("c")
    return occurrence.start() + start, occurrence.start() + end


def percentile(values, quantile):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def summarize(records):
    counts = [item["masked_tokens"] for item in records]
    fractions = [item["masked_fraction"] for item in records]
    return {
        "count": len(records),
        "mean_masked_tokens": statistics.fmean(counts),
        "p50_masked_tokens": percentile(counts, 0.50),
        "p95_masked_tokens": percentile(counts, 0.95),
        "max_masked_tokens": max(counts),
        "mean_masked_fraction": statistics.fmean(fractions),
        "exact_target_count": sum(item["exact_target"] for item in records),
        "exact_target_rate": sum(item["exact_target"] for item in records) / len(records),
        "examples": records[:5],
    }


def record_case(category, row, completion, result, compiled, expected_mask, **metadata):
    included = [item for item in compiled["records"] if item["included"]]
    return {
        "category": category,
        "sample_id": row["sample_id"],
        "token_count": compiled["token_count"],
        "masked_tokens": compiled["masked_token_count"],
        "masked_fraction": compiled["masked_token_fraction"],
        "exact_target": compiled["penalty_mask"] == expected_mask,
        "compiled_kinds": sorted({item["kind"] for item in included}),
        "source_violation_kinds": sorted({item.kind for item in result.violations}),
        **metadata,
    }


def build_real_sid_pool(rows, tokenizer):
    pool = sorted({sid for row in rows for sid in row["history_sids"]})
    four_token = []
    for start in range(0, len(pool), 2048):
        batch = pool[start : start + 2048]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        four_token.extend(sid for sid, token_ids in zip(batch, encoded) if len(token_ids) == 4)
    if len(four_token) != len(pool):
        raise AssertionError("pilot contains a real SID that is not four parent-tokenizer tokens")
    by_prefix = defaultdict(list)
    for sid in four_token:
        match = SID_PART_RE.fullmatch(sid)
        if match:
            domain, a, b, _ = match.groups()
            by_prefix[(domain, a, b)].append(sid)
    return four_token, by_prefix


def choose_c_only_candidate(row, by_prefix):
    history = set(row["history_sids"])
    for source_sid in row["history_sids"]:
        match = SID_PART_RE.fullmatch(source_sid)
        if not match:
            continue
        domain, a, b, _ = match.groups()
        for candidate in by_prefix[(domain, a, b)]:
            if candidate not in history:
                return candidate
    return None


def choose_outside_history(row, real_sid_pool):
    history = set(row["history_sids"])
    return next(candidate for candidate in real_sid_pool if candidate not in history)


def audit_action_hallucination(rows, tokenizer, by_prefix, count=50):
    records = []
    for row in rows:
        if row["route"] != "action":
            continue
        candidate = choose_c_only_candidate(row, by_prefix)
        if candidate is None:
            continue
        completion = action_json([*row["gold_sids"], candidate])
        result = score_action(completion, row, tokenizer)
        compiled = compile_penalty_mask(completion, result.violations, tokenizer, "action")
        target = c_component_span(completion, candidate)
        expected = mask_for_spans(tokenizer, completion, [target])
        hall = next(item for item in result.violations if item.kind == "hallucinated_sid" and item.metadata.get("sid") == candidate)
        records.append(
            record_case(
                "action_hallucination_c_only",
                row,
                completion,
                result,
                compiled,
                expected,
                candidate_sid=candidate,
                candidate_from_real_pool=True,
                candidate_token_count=len(tokenizer.encode(candidate, add_special_tokens=False)),
                first_invalid_component=hall.metadata["first_invalid_component"],
            )
        )
        if len(records) == count:
            break
    return records


def audit_action_duplicate(rows, tokenizer, count=50):
    records = []
    for row in rows:
        if row["route"] != "action" or not row["gold_sids"]:
            continue
        sid = row["gold_sids"][0]
        completion = action_json([*row["gold_sids"], sid])
        result = score_action(completion, row, tokenizer)
        compiled = compile_penalty_mask(completion, result.violations, tokenizer, "action")
        target = [match.span() for match in SID_RE.finditer(completion) if match.group() == sid][-1]
        expected = mask_for_spans(tokenizer, completion, [target])
        records.append(record_case("action_duplicate", row, completion, result, compiled, expected, duplicated_sid=sid))
        if len(records) == count:
            break
    return records


def audit_chain_hallucination(rows, tokenizer, real_sid_pool, count=50):
    records = []
    for row in rows:
        if row["route"] != "chain":
            continue
        candidate = choose_outside_history(row, real_sid_pool)
        events = [dict(item) for item in row["gold_events"]]
        events[0]["action"] += f"；[受控审计] {candidate}"
        completion = chain_json(events)
        result = score_chain(completion, row, tokenizer)
        compiled = compile_penalty_mask(completion, result.violations, tokenizer, "chain")
        target = next(match.span() for match in SID_RE.finditer(completion) if match.group() == candidate)
        expected = mask_for_spans(tokenizer, completion, [target])
        records.append(
            record_case(
                "chain_hallucination",
                row,
                completion,
                result,
                compiled,
                expected,
                candidate_sid=candidate,
                candidate_from_real_pool=True,
                candidate_token_count=len(tokenizer.encode(candidate, add_special_tokens=False)),
            )
        )
        if len(records) == count:
            break
    return records


def audit_chain_date_mismatch(rows, tokenizer, count=50):
    records = []
    for row in rows:
        if row["route"] != "chain":
            continue
        events = [dict(item) for item in row["gold_events"]]
        events[0]["date"] = "1900-01-01"
        completion = chain_json(events)
        result = score_chain(completion, row, tokenizer)
        compiled = compile_penalty_mask(completion, result.violations, tokenizer, "chain")
        span = result.event_spans[0]["fields"]["date"]
        target = (span["char_start"], span["char_end"])
        expected = mask_for_spans(tokenizer, completion, [target])
        records.append(record_case("chain_date_mismatch", row, completion, result, compiled, expected))
        if len(records) == count:
            break
    return records


def controlled_wrong_part(action: str):
    parts = [part.strip() for part in action.split("；")]
    original = parts[-1]
    sids = SID_RE.findall(original)
    wrong = f"[受控错误行为] {' '.join(sids)}" if sids else original + "（受控错误）"
    parts[-1] = wrong
    return "；".join(parts), wrong


def audit_chain_action_mismatch(rows, tokenizer, count=50):
    records = []
    for row in rows:
        if row["route"] != "chain":
            continue
        events = [dict(item) for item in row["gold_events"]]
        events[0]["action"], wrong_part = controlled_wrong_part(events[0]["action"])
        completion = chain_json(events)
        result = score_chain(completion, row, tokenizer)
        compiled = compile_penalty_mask(completion, result.violations, tokenizer, "chain")
        start = completion.index(wrong_part)
        expected = mask_for_spans(tokenizer, completion, [(start, start + len(wrong_part))])
        records.append(
            record_case(
                "chain_action_mismatch",
                row,
                completion,
                result,
                compiled,
                expected,
                merged_action=len(events[0]["action"].split("；")) > 1,
            )
        )
        if len(records) == count:
            break
    return records


def audit_chain_duplicate(rows, tokenizer, count=50):
    records = []
    for row in rows:
        if row["route"] != "chain" or len(row["gold_events"]) >= 5:
            continue
        events = [dict(item) for item in row["gold_events"]]
        events.append(dict(events[-1]))
        completion = chain_json(events)
        result = score_chain(completion, row, tokenizer)
        compiled = compile_penalty_mask(completion, result.violations, tokenizer, "chain")
        span = result.event_spans[-1]["event"]
        expected = mask_for_spans(tokenizer, completion, [(span["char_start"], span["char_end"])])
        records.append(record_case("chain_duplicate", row, completion, result, compiled, expected))
        if len(records) == count:
            break
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    before = {name: sha256(args.data_dir / name) for name in EXPECTED_SHA}
    if before != EXPECTED_SHA:
        raise SystemExit(f"frozen SHA mismatch before audit: {before}")
    pilot = read_jsonl(args.data_dir / "pilot_600.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.parent_checkpoint, local_files_only=True)
    if not tokenizer.is_fast:
        raise SystemExit("parent tokenizer must expose exact offsets")
    real_sid_pool, by_prefix = build_real_sid_pool(pilot, tokenizer)

    category_records = {
        "action_hallucination": audit_action_hallucination(pilot, tokenizer, by_prefix),
        "action_duplicate": audit_action_duplicate(pilot, tokenizer),
        "chain_hallucination": audit_chain_hallucination(pilot, tokenizer, real_sid_pool),
        "chain_date_mismatch": audit_chain_date_mismatch(pilot, tokenizer),
        "chain_action_mismatch": audit_chain_action_mismatch(pilot, tokenizer),
        "chain_duplicate": audit_chain_duplicate(pilot, tokenizer),
    }
    if any(len(records) != 50 for records in category_records.values()):
        raise SystemExit({name: len(records) for name, records in category_records.items()})
    summaries = {name: summarize(records) for name, records in category_records.items()}
    if any(item["exact_target_rate"] != 1.0 for item in summaries.values()):
        raise SystemExit("locality exact-target audit failed")
    c_records = category_records["action_hallucination"]
    if not all(
        item["candidate_from_real_pool"]
        and item["candidate_token_count"] == 4
        and item["first_invalid_component"] == "c"
        and item["masked_tokens"] == 1
        for item in c_records
    ):
        raise SystemExit("C-only real-SID hallucination fixture failed")

    after = {name: sha256(args.data_dir / name) for name in EXPECTED_SHA}
    if after != before:
        raise SystemExit("frozen data changed during audit")
    result = {
        "contract_version": "gr_user_penalty_mask_v1",
        "execution": {"cpu_only": True, "generation": False, "training": False, "trainer_integration": False},
        "fixture": {
            "real_sid_pool_size": len(real_sid_pool),
            "all_real_sids_four_tokens": True,
            "synthetic_out_of_vocab_sid_used": False,
        },
        "frozen_sha_before": before,
        "frozen_sha_after": after,
        "frozen_sha_unchanged": before == after == EXPECTED_SHA,
        "locality": summaries,
        "records": category_records,
    }
    summary = {key: value for key, value in result.items() if key != "records"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
