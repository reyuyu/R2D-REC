"""Paired, inference-only checkpoint evaluation for Recommendation GRPO."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALIDATION_DATASET = Path(
    "/data/lf_data_versions/alltrain/alpha_mini_v1_validation_filtered_v1/dev.jsonl"
)
LEAKAGE_AUDIT = VALIDATION_DATASET.parent / "leakage_audit.json"
TRAIN_DATASET = Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl")
DOMAINS = ("video", "prod", "ad", "living")
DOMAIN_PROMPTS = {
    "video": "请阅读该用户的对话行为记录，推断该用户下一个可能感兴趣的视频。",
    "prod": "请阅读该用户的对话行为记录，推断该用户下一个可能感兴趣的商品。",
    "ad": "请阅读该用户的对话行为记录，推断该用户下一个可能感兴趣的广告。",
    "living": "请阅读该用户的对话行为记录，推断该用户下一个可能观看的直播。",
}
SID_RE = re.compile(r"<\|(?P<domain>video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
ROUTE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sid(value: str) -> tuple[str, int, int, int] | None:
    match = SID_RE.fullmatch(str(value).strip())
    return (
        (match.group("domain"), int(match.group(2)), int(match.group(3)), int(match.group(4)))
        if match else None
    )


def extract_sids(value: str) -> set[tuple[str, int, int, int]]:
    return {
        (
            match.group("domain"),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        for match in SID_RE.finditer(str(value))
    }


def _base_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("instruction") or "")
    if row.get("input"):
        prompt += ("\n" if prompt else "") + str(row["input"])
    return ROUTE_SUFFIX_RE.sub("", prompt).strip()


def route_prompt(base_prompt: str, domain: str, route: str) -> str:
    if domain not in DOMAIN_PROMPTS or route not in {"think", "no_think"}:
        raise ValueError("unsupported validation domain or route")
    remainder = base_prompt.split("\n", 1)[1].lstrip() if "\n" in base_prompt else base_prompt
    return f"{DOMAIN_PROMPTS[domain]}\n\n{remainder.rstrip()}/{route}"


def load_validation_pool(
    validation_path: Path = VALIDATION_DATASET,
    leakage_path: Path = LEAKAGE_AUDIT,
    train_path: Path = TRAIN_DATASET,
) -> list[dict[str, Any]]:
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    if not leakage.get("validation_safe") or any(
        int(leakage.get("dev", {}).get(key, -1)) != 0
        for key in (
            "recommendation_group_id_overlap", "recommendation_history_domain_overlap",
            "exact_full_row_overlap", "canonical_prompt_overlap",
        )
    ):
        raise RuntimeError("validation leakage audit is not clean")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    with validation_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source_segment") not in {"recommendation_cot", "recommendation_nocot"}:
                continue
            metadata = json.loads(row.get("aux_metadata_json") or "{}")
            group_id = metadata.get("recommendation_group_id")
            if isinstance(group_id, str) and re.fullmatch(r"[0-9a-f]{64}", group_id):
                grouped[group_id].append((row, metadata))

    with train_path.open(encoding="utf-8") as handle:
        train_ids = {
            json.loads(line)["recommendation_group_id"]
            for line in handle if line.strip()
        }
    if train_ids.intersection(grouped):
        raise RuntimeError("validation group IDs overlap the GRPO training dataset")

    result = []
    for group_id, rows in sorted(grouped.items()):
        prompts = {_base_prompt(row) for row, _ in rows}
        gold_values = {
            sid for _, metadata in rows
            for sid in metadata.get("recommendation_all_gold_sids", [])
            if parse_sid(sid) is not None
        }
        parsed = {parse_sid(sid) for sid in gold_values}
        domains = {sid[0] for sid in parsed if sid is not None}
        if len(prompts) != 1 or len(domains) != 1 or not gold_values:
            continue
        base_prompt = next(iter(prompts))
        result.append({
            "group_id": group_id,
            "domain": next(iter(domains)),
            "base_prompt": base_prompt,
            "gold_sids": sorted(gold_values),
            "history_sids": sorted(extract_sids(base_prompt)),
        })
    return result


def _allocation(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {domain: sample_size * counts.get(domain, 0) / total for domain in DOMAINS}
    allocated = {domain: min(counts.get(domain, 0), int(math.floor(raw[domain]))) for domain in DOMAINS}
    remaining = sample_size - sum(allocated.values())
    order = sorted(DOMAINS, key=lambda domain: (raw[domain] - allocated[domain], counts.get(domain, 0), domain), reverse=True)
    while remaining:
        changed = False
        for domain in order:
            if allocated[domain] < counts.get(domain, 0):
                allocated[domain] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            raise ValueError("sample size exceeds validation pool")
    return allocated


def build_cohort(pool: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size < 1 or sample_size > len(pool):
        raise ValueError(f"sample_size must be in [1, {len(pool)}]")
    by_domain = {domain: [row for row in pool if row["domain"] == domain] for domain in DOMAINS}
    counts = {domain: len(rows) for domain, rows in by_domain.items()}
    allocated = _allocation(counts, sample_size)
    chosen = []
    for domain in DOMAINS:
        ranked = sorted(
            by_domain[domain],
            key=lambda row: hashlib.sha256(f"{seed}:{row['group_id']}".encode()).hexdigest(),
        )
        chosen.extend(ranked[:allocated[domain]])
    return sorted(chosen, key=lambda row: hashlib.sha256(f"order:{seed}:{row['group_id']}".encode()).hexdigest())


def wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def validation_catalog() -> dict[str, Any]:
    pool = load_validation_pool()
    counts = collections.Counter(row["domain"] for row in pool)
    return {
        "dataset": str(VALIDATION_DATASET),
        "dataset_sha256": sha256_file(VALIDATION_DATASET),
        "leakage_audit": str(LEAKAGE_AUDIT),
        "validation_safe": True,
        "train_group_overlap": 0,
        "pool_size": len(pool),
        "domain_counts": dict(counts),
        "sample_presets": [64, 128, 256],
        "recommended_sample_size": 128,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _route_outcome(
    predicted: list[tuple[str, int, int, int] | None],
    gold: set[tuple],
    history: set[tuple[str, int, int, int]],
) -> dict[str, Any]:
    valid = [sid for sid in predicted if sid is not None]
    exact = any(sid in gold for sid in valid)
    ab = any(any(sid[:3] == target[:3] for target in gold) for sid in valid)
    a = any(any(sid[:2] == target[:2] for target in gold) for sid in valid)
    history_count = sum(sid in history for sid in valid)
    novel_count = len(valid) - history_count
    unique_valid = set(valid)
    unique_history_count = len(unique_valid.intersection(history))
    unique_novel_count = len(unique_valid) - unique_history_count
    return {
        "hit": exact,
        "ab_hit": ab,
        "a_hit": a,
        "invalid_count": len(predicted) - len(valid),
        "valid_prediction_count": len(valid),
        "history_copy_prediction_count": history_count,
        "novel_prediction_count": novel_count,
        "history_copy_rate": _rate(history_count, len(valid)),
        "novel_prediction_rate": _rate(novel_count, len(valid)),
        "unique_valid_sid_count": len(unique_valid),
        "unique_history_copy_sid_count": unique_history_count,
        "unique_novel_sid_count": unique_novel_count,
        "unique_history_copy_rate": _rate(unique_history_count, len(unique_valid)),
        "unique_novel_rate": _rate(unique_novel_count, len(unique_valid)),
    }


def _aggregate(rows: list[dict[str, Any]], route: str) -> dict[str, Any]:
    total = len(rows)
    hits = sum(bool(row[route]["hit"]) for row in rows)
    valid_prediction_count = sum(int(row[route]["valid_prediction_count"]) for row in rows)
    history_copy_prediction_count = sum(
        int(row[route]["history_copy_prediction_count"]) for row in rows
    )
    novel_prediction_count = sum(int(row[route]["novel_prediction_count"]) for row in rows)
    unique_valid_sid_count = sum(int(row[route]["unique_valid_sid_count"]) for row in rows)
    unique_history_copy_sid_count = sum(
        int(row[route]["unique_history_copy_sid_count"]) for row in rows
    )
    unique_novel_sid_count = sum(int(row[route]["unique_novel_sid_count"]) for row in rows)
    result = {
        "n": total, "hits": hits, "hit_rate": hits / total if total else None,
        "hit_rate_ci95": wilson_interval(hits, total),
        "ab_hit_rate": sum(bool(row[route]["ab_hit"]) for row in rows) / total if total else None,
        "a_hit_rate": sum(bool(row[route]["a_hit"]) for row in rows) / total if total else None,
        "invalid_rate": sum(int(row[route]["invalid_count"]) for row in rows) / (32 * total) if total else None,
        "valid_prediction_count": valid_prediction_count,
        "history_copy_prediction_count": history_copy_prediction_count,
        "novel_prediction_count": novel_prediction_count,
        "history_copy_rate": _rate(history_copy_prediction_count, valid_prediction_count),
        "novel_prediction_rate": _rate(novel_prediction_count, valid_prediction_count),
        "unique_valid_sid_count": unique_valid_sid_count,
        "unique_history_copy_sid_count": unique_history_copy_sid_count,
        "unique_novel_sid_count": unique_novel_sid_count,
        "unique_history_copy_rate": _rate(
            unique_history_copy_sid_count, unique_valid_sid_count
        ),
        "unique_novel_rate": _rate(unique_novel_sid_count, unique_valid_sid_count),
    }
    if route == "think":
        result["closure_rate"] = sum(bool(row[route]["closed"]) for row in rows) / total if total else None
    return result


def run_evaluation(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from grpo_model import BASE, encode_prompt, generate_batch
    from grpo_sid import final_sid

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        # Ranks only exchange small Python result objects. CPU coordination avoids
        # coupling independent GPU inference to the host's NCCL network setup.
        dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    output = Path(args.output_dir).resolve()
    checkpoints = json.loads(Path(args.config).read_text(encoding="utf-8"))["checkpoints"]
    pool = load_validation_pool()
    cohort = build_cohort(pool, args.sample_size, args.seed)
    if rank == 0:
        atomic_json(output / "cohort.json", {
            **validation_catalog(), "seed": args.seed, "sample_size": len(cohort),
            "domain_counts": dict(collections.Counter(row["domain"] for row in cohort)),
            "group_ids": [row["group_id"] for row in cohort],
        })
        atomic_json(output / "status.json", {
            "state": "running", "started_at": utc_now(), "sample_size": len(cohort),
            "checkpoint_count": len(checkpoints), "completed_checkpoints": 0,
            "phase": "loading_base_model",
        })
    if world > 1:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=f"cuda:{local_rank}",
        trust_remote_code=True, attn_implementation="flash_attention_2",
    )
    all_summaries = []
    started_all = time.time()
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        if rank == 0:
            for progress_path in output.glob("rank*-progress.json"):
                progress_path.unlink(missing_ok=True)
            atomic_json(output / "status.json", {
                "state": "running", "started_at": utc_now(), "sample_size": len(cohort),
                "checkpoint_count": len(checkpoints), "completed_checkpoints": checkpoint_index,
                "current_checkpoint": checkpoint["name"], "phase": "loading_checkpoint",
            })
        if world > 1:
            dist.barrier()
        model = PeftModel.from_pretrained(base_model, checkpoint["path"], is_trainable=False)
        model.eval()
        if rank == 0:
            atomic_json(output / "status.json", {
                "state": "running", "started_at": utc_now(), "sample_size": len(cohort),
                "checkpoint_count": len(checkpoints), "completed_checkpoints": checkpoint_index,
                "current_checkpoint": checkpoint["name"], "phase": "evaluating",
            })
        local_rows = []
        for cohort_index in range(rank, len(cohort), world):
            sample = cohort[cohort_index]
            gold = {parse_sid(value) for value in sample["gold_sids"]}
            gold.discard(None)
            history = {tuple(sid) for sid in sample["history_sids"]}
            sample_started = time.time()

            think_prompt = route_prompt(sample["base_prompt"], sample["domain"], "think")
            torch.manual_seed(args.seed + cohort_index)
            torch.cuda.manual_seed_all(args.seed + cohort_index)
            cot_text, cot_ids = generate_batch(
                model, tokenizer, [encode_prompt(tokenizer, think_prompt)], max_new_tokens=2048,
                do_sample=True, temperature=0.9, top_p=0.95, num_beams=1,
                num_return_sequences=1, return_ids=True,
            )
            cot_text, cot_ids = cot_text[0], cot_ids[0]
            close_at = cot_text.find("</think>")
            closed = close_at >= 0
            if closed:
                cot_text = cot_text[: close_at + len("</think>")]
                cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
            think_beams = generate_batch(
                model, tokenizer, [encode_prompt(tokenizer, think_prompt) + cot_ids],
                max_new_tokens=128, do_sample=False, num_beams=32, num_return_sequences=32,
            )
            think_sids = [final_sid(text) for text in think_beams]

            no_prompt = route_prompt(sample["base_prompt"], sample["domain"], "no_think")
            no_beams = generate_batch(
                model, tokenizer, [encode_prompt(tokenizer, no_prompt)], max_new_tokens=128,
                do_sample=False, num_beams=32, num_return_sequences=32,
            )
            no_sids = [final_sid(text) for text in no_beams]
            local_rows.append({
                "cohort_index": cohort_index, "group_id": sample["group_id"], "domain": sample["domain"],
                "gold_sids": sample["gold_sids"],
                "history_sid_count": len(history),
                "think": {**_route_outcome(think_sids, gold, history), "closed": closed},
                "no_think": _route_outcome(no_sids, gold, history),
                "wall_sec": time.time() - sample_started,
            })
            atomic_json(output / f"rank{rank}-progress.json", {
                "rank": rank, "checkpoint": checkpoint["name"],
                "completed_samples": len(local_rows), "assigned_samples": len(range(rank, len(cohort), world)),
                "updated_at": utc_now(),
            })

        gathered = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(local_rows, gathered, dst=0)
        else:
            gathered = [local_rows]
        if rank == 0:
            rows = sorted((row for shard in gathered for row in shard), key=lambda row: row["cohort_index"])
            summary = {
                "checkpoint": checkpoint["name"], "step": checkpoint["step"], "n": len(rows),
                "think": _aggregate(rows, "think"), "no_think": _aggregate(rows, "no_think"),
                "domains": {
                    domain: {
                        "think": _aggregate([row for row in rows if row["domain"] == domain], "think"),
                        "no_think": _aggregate([row for row in rows if row["domain"] == domain], "no_think"),
                    } for domain in DOMAINS
                },
                "wall_sec": sum(float(row["wall_sec"]) for row in rows) / world,
                "examples": rows[:8],
            }
            all_summaries.append(summary)
            atomic_json(output / "results.json", {
                "run_id": args.run_id, "seed": args.seed, "sample_size": len(cohort),
                "paired_cohort": True, "checkpoints": all_summaries,
            })
            atomic_json(output / "status.json", {
                "state": "running", "started_at": utc_now(), "sample_size": len(cohort),
                "checkpoint_count": len(checkpoints), "completed_checkpoints": checkpoint_index + 1,
                "current_checkpoint": checkpoint["name"], "phase": "checkpoint_complete",
            })
        base_model = model.unload()
        del model
        torch.cuda.empty_cache()
        if world > 1:
            dist.barrier()

    if rank == 0:
        atomic_json(output / "status.json", {
            "state": "completed", "completed_at": utc_now(), "sample_size": len(cohort),
            "checkpoint_count": len(checkpoints), "completed_checkpoints": len(checkpoints),
            "phase": "completed",
            "wall_sec": time.time() - started_all,
        })
    if world > 1:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference-only paired checkpoint evaluator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--catalog", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.catalog:
        print(json.dumps(validation_catalog(), ensure_ascii=False, indent=2))
        return
    run_evaluation(args)


if __name__ == "__main__":
    main()
