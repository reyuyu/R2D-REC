"""Deterministic held-out probes for GR_USER_v1 training diagnostics."""

from __future__ import annotations

import random
import statistics
import time
from collections import defaultdict
from typing import Callable

import torch
import torch.distributed as dist

from user_action_reward import score_action
from user_chain_reward import score_chain
from user_common import SID_RE, normalize_text


PROBE_ROUTES = ("action", "chain")


def probe_due(step: int, max_steps: int, every_steps: int) -> bool:
    if every_steps <= 0:
        return False
    return step == 0 or step == max_steps or step % every_steps == 0


def validate_probe_rows(rows: list[dict]) -> None:
    action = [row for row in rows if row.get("route") == "action"]
    chain = [row for row in rows if row.get("route") == "chain"]
    counts = (len(action), len(chain))
    if counts not in {(12, 8), (3, 3)} or len(rows) != sum(counts):
        raise ValueError("User fixed probe must contain 12/8 or light 3/3 Action/Chain rows")
    sample_ids = [row.get("sample_id") for row in rows]
    if None in sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("User fixed probe sample IDs must be present and unique")


def partition_probe_rows(rows: list[dict], rank: int, world_size: int = 4) -> dict[str, list[dict]]:
    if world_size != 4 or rank not in range(world_size):
        raise ValueError("User fixed probe requires four ranks")
    validate_probe_rows(rows)
    return {
        route: sorted(
            [row for row in rows if row["route"] == route],
            key=lambda row: row["sample_id"],
        )[rank::world_size]
        for route in PROBE_ROUTES
    }


def _unique_spans(spans: list[dict]) -> list[dict]:
    output = []
    seen = set()
    for span in sorted(spans, key=lambda item: (item["start"], item["end"], item["kind"])):
        key = (span["start"], span["end"], span["kind"])
        if span["end"] > span["start"] and key not in seen:
            output.append(span)
            seen.add(key)
    return output


def correct_match_spans(score, sample: dict, route: str) -> list[dict]:
    """Return exact source spans that are safe to render as correct matches."""
    spans: list[dict] = []
    if route == "action":
        true_positive = set(score.true_positive_sids)
        for occurrence in score.sid_occurrences:
            if occurrence["sid"] in true_positive:
                spans.append({
                    "start": occurrence["char_start"],
                    "end": occurrence["char_end"],
                    "kind": "correct_sid",
                })
        return _unique_spans(spans)

    gold_events = sample.get("gold_events", [])
    gold_sids = set(sample.get("gold_sids", []))
    for event_span in score.event_spans:
        for sid_span in event_span.get("sids", []):
            if sid_span.get("sid") in gold_sids:
                spans.append({
                    "start": sid_span["char_start"],
                    "end": sid_span["char_end"],
                    "kind": "correct_sid",
                })
    for match in score.matches:
        predicted = score.predicted_events[match.predicted_index]
        gold = gold_events[match.gold_index]
        if (
            predicted.get("date") == gold.get("date")
            and normalize_text(predicted.get("action", "")) == normalize_text(gold.get("action", ""))
        ):
            event_span = score.event_spans[match.predicted_index]["event"]
            spans.append({
                "start": event_span["char_start"],
                "end": event_span["char_end"],
                "kind": "correct_event",
            })
    return _unique_spans(spans)


def _masked_spans(compiled: dict) -> list[dict]:
    spans = []
    for record in compiled.get("records", []):
        if not record.get("included"):
            continue
        for span in record.get("token_spans", []):
            start, end = span.get("char_start"), span.get("char_end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                spans.append({"start": start, "end": end, "kind": record["kind"]})
    return _unique_spans(spans)


def score_probe_candidate(completion: str, sample: dict, tokenizer) -> dict:
    from user_penalty_mask import compile_penalty_mask

    route = sample["route"]
    score = score_action(completion, sample, tokenizer) if route == "action" else score_chain(
        completion, sample, tokenizer
    )
    compiled = compile_penalty_mask(completion, score.violations, tokenizer, route)
    record = {
        "completion": completion,
        "completion_length": len(compiled["penalty_mask"]),
        "reward": float(score.reward),
        "violations": [item.kind for item in score.violations],
        "penalty_kinds": sorted({item["kind"] for item in compiled["records"] if item.get("included")}),
        "masked_spans": _masked_spans(compiled),
        "masked_token_count": int(sum(compiled["penalty_mask"])),
        "match_spans": correct_match_spans(score, sample, route),
    }
    if route == "action":
        record.update(
            f1=float(score.f1),
            precision=float(score.precision),
            recall=float(score.recall),
            exact_match=bool(score.exact_set_match),
            gold_sids=list(score.gold_sids),
            pred_sids=list(score.pred_sids_unique),
        )
    else:
        record.update(
            total_reward=float(score.total_reward),
            action_alignment=float(score.action_f1),
            logic_alignment=float(score.logic_f1),
            gold_sids=list(sample.get("gold_sids", [])),
            gold_events=list(sample.get("gold_events", [])),
            predicted_events=list(score.predicted_events),
        )
    return record


def build_probe_event(
    sample: dict,
    candidates: list[dict],
    *,
    step: int,
    reason: str,
    seed: int,
    probe_wall_sec: float,
) -> dict:
    rewards = [float(item["reward"]) for item in candidates]
    event = {
        "step": int(step),
        "reason": reason,
        "seed": int(seed),
        "group_id": sample["sample_id"],
        "route": sample["route"],
        "bucket": sample.get("bucket"),
        "gold_sids": list(sample.get("gold_sids", [])),
        "gold_events": list(sample.get("gold_events", [])),
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "probe_wall_sec": float(probe_wall_sec),
        "candidates": candidates,
    }
    if sample["route"] == "action":
        event["action"] = {
            "f1_mean": statistics.fmean(item["f1"] for item in candidates),
            "precision_mean": statistics.fmean(item["precision"] for item in candidates),
            "recall_mean": statistics.fmean(item["recall"] for item in candidates),
            "exact_match_rate": statistics.fmean(float(item["exact_match"]) for item in candidates),
        }
    else:
        event["chain"] = {
            "total_reward_mean": statistics.fmean(item["total_reward"] for item in candidates),
            "action_alignment_mean": statistics.fmean(item["action_alignment"] for item in candidates),
            "logic_alignment_mean": statistics.fmean(item["logic_alignment"] for item in candidates),
        }
    return event


def aggregate_probe_steps(events: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        grouped[int(event["step"])].append(event)
    output = []
    for step, rows in sorted(grouped.items()):
        action = [row for row in rows if row["route"] == "action"]
        chain = [row for row in rows if row["route"] == "chain"]
        output.append({
            "step": step,
            "action_f1": statistics.fmean(row["action"]["f1_mean"] for row in action),
            "action_precision": statistics.fmean(row["action"]["precision_mean"] for row in action),
            "action_recall": statistics.fmean(row["action"]["recall_mean"] for row in action),
            "action_exact": statistics.fmean(row["action"]["exact_match_rate"] for row in action),
            "chain_reward": statistics.fmean(row["chain"]["total_reward_mean"] for row in chain),
            "chain_action_alignment": statistics.fmean(row["chain"]["action_alignment_mean"] for row in chain),
            "chain_logic_alignment": statistics.fmean(row["chain"]["logic_alignment_mean"] for row in chain),
        })
    return output


def evaluate_user_fixed_probe(
    model,
    tokenizer,
    rows: list[dict],
    device,
    *,
    rank: int,
    world_size: int,
    step: int,
    reason: str,
    seed: int,
    generate_fn: Callable,
) -> list[dict]:
    """Run fixed-seed generation and return all probe events on rank 0."""
    partitions = partition_probe_rows(rows, rank, world_size)
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    was_training = model.training
    local_events = []
    started = time.perf_counter()
    try:
        model.eval()
        for route_index, route in enumerate(PROBE_ROUTES):
            local_rows = partitions[route]
            if not local_rows:
                continue
            route_seed = seed + 1000 * route_index + rank
            random.seed(route_seed)
            torch.manual_seed(route_seed)
            torch.cuda.manual_seed_all(route_seed)
            completion_ids = generate_fn(model, tokenizer, local_rows, device)
            decoded = [tokenizer.decode(ids, skip_special_tokens=False) for ids in completion_ids]
            for row_index, sample in enumerate(local_rows):
                candidates = [
                    score_probe_candidate(decoded[row_index * 4 + candidate], sample, tokenizer)
                    for candidate in range(4)
                ]
                for candidate_id, candidate in enumerate(candidates):
                    candidate["candidate_id"] = candidate_id
                local_events.append((sample, candidates))
        torch.cuda.synchronize(device)
        local_wall = time.perf_counter() - started
        gathered = [None] * world_size
        dist.all_gather_object(gathered, {"events": local_events, "wall": local_wall})
        if rank != 0:
            return []
        wall = max(payload["wall"] for payload in gathered)
        events = []
        for payload in gathered:
            for sample, candidates in payload["events"]:
                events.append(build_probe_event(
                    sample,
                    candidates,
                    step=step,
                    reason=reason,
                    seed=seed,
                    probe_wall_sec=wall,
                ))
        return sorted(events, key=lambda event: (event["route"], event["group_id"]))
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)
        model.train(was_training)
