"""Four-GPU Step0/Step300 Self-CoT health gate for Boundary Adaptation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RUNTIME = Path("/data/GRPO")
GRPO = RUNTIME / "baselines/native_source_domain_r32_v3/grpo"
sys.path[:0] = [str(RUNTIME / "scripts"), str(GRPO)]
from grpo_model import BASE, encode_prompt  # noqa: E402
from grpo_probe import FixedProbeEvaluator  # noqa: E402
from run_grpo_trl_smoke import make_beam32_fn, make_grpo_config  # noqa: E402
from ablations.gr_rec_think_composite_interest_v1.composite_trainer import (  # noqa: E402
    ThinkCompositeInterestRecGRPOTrainer,
)
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import (  # noqa: E402
    PROBES, classify, history_from_prompt, parse_gold,
)

SEED = 20260818
STEP0 = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
STEP300 = Path("/data/outputs/boundary_adapt/formal_300step/checkpoint-300")
CHECKPOINTS = {"0": STEP0, "300": STEP300}
RESULTS = RUNTIME / "boundary_adapt/results/self_cot_health_20260824_parts"
FINAL = RUNTIME / "boundary_adapt/results/boundary_adapt_self_cot_health_gate_20260824.json"
DOMAIN_ORDER = ("video", "prod", "ad", "living")


class CaptureMonitor:
    enabled = True

    def write_beam_detail(self, _row):
        pass


def completion_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def frozen_records():
    rows = []
    with PROBES.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("step", -1)) == 0:
                rows.append(row)
    rows.sort(key=lambda row: (int(row["probe_round"]), DOMAIN_ORDER.index(row["target_domain"])))
    if len(rows) != 12 or Counter(row["target_domain"] for row in rows) != Counter({d: 3 for d in DOMAIN_ORDER}):
        raise RuntimeError("FROZEN_12_PROBE_CONTRACT_INVALID")
    records = {}
    rounds = []
    for round_index in range(3):
        chunk = rows[round_index * 4:(round_index + 1) * 4]
        rounds.append([row["group_id"] for row in chunk])
        for row in chunk:
            records[row["group_id"]] = {"think": {
                "prompt": row["think_prompt"], "all_gold_sids": row["gold_sids"],
                "target_domain": row["target_domain"],
            }}
    return records, rounds


def load_model(adapter, device):
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tokenizer


def dummy_reward(completions, **_kwargs):
    return [0.0] * len(completions)


def run_checkpoint(label):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    random.seed(SEED + rank); torch.manual_seed(SEED + rank); torch.cuda.manual_seed(SEED + rank)
    records, rounds = frozen_records()
    model, tokenizer = load_model(CHECKPOINTS[label], f"cuda:{rank}")
    cfg = make_grpo_config(str(RESULTS / f"trainer_{label}"), 1, 0.0, SEED)
    seed_rows = [{"prompt": next(iter(records.values()))["think"]["prompt"], "route": "think"}] * 16
    monitor = CaptureMonitor()
    trainer = ThinkCompositeInterestRecGRPOTrainer(
        model=model, args=cfg, processing_class=tokenizer,
        train_dataset=Dataset.from_list(seed_rows), reward_funcs=[dummy_reward],
        monitor_writer=None,
    )
    beam32 = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
    local = []
    for round_index, group_ids in enumerate(rounds):
        evaluator = FixedProbeEvaluator(trainer, records, group_ids, beam32, monitor, SEED, 1)
        evaluator._set_seed()
        item = evaluator._think()
        item["probe_round"] = round_index
        item["origin_rank"] = rank
        local.append(item)
    gathered = [None] * 4
    dist.all_gather_object(gathered, local)
    if rank == 0:
        items = [item for rank_items in gathered for item in rank_items]
        if len(items) != 12:
            raise RuntimeError("SELF_COT_GROUP_COUNT_INVALID")
        records_out = []
        for item in items:
            history = history_from_prompt(tokenizer, encode_prompt(tokenizer, item["prompt"]))
            gold_set = {parse_gold(value) for value in item["gold_sids"]}
            candidate_rows = []
            for candidate_id, candidate in enumerate(item["candidates"]):
                text = candidate["completion"]
                ids = tokenizer.encode(text, add_special_tokens=False)
                beams = [classify(tuple(sid) if sid else None, history, gold_set) for sid in candidate["beam_sids"]]
                candidate_rows.append({
                    **{key: candidate[key] for key in ("completion_length", "closed", "reward", "exact", "ab", "a", "invalid")},
                    "candidate_id": candidate_id, "completion_sha256": candidate["completion_sha256"],
                    "empty": len(ids) == 0 or not text.strip(), "closure_position": ids.index(tokenizer.encode("</think>", add_special_tokens=False)[0]) if candidate["closed"] else None,
                    "literal_sid_occurrences": text.count("<s_a_"), "history_beams": beams,
                })
            records_out.append({
                "probe_round": item["probe_round"], "origin_rank": item["origin_rank"],
                "group_id": item["group_id"], "target_domain": item["target_domain"],
                "prompt_token_count": len(encode_prompt(tokenizer, item["prompt"])),
                "generation_wall_sec": item["generation_wall_sec"], "wall_sec": item["wall_sec"],
                "candidates": candidate_rows,
            })
        candidates = [candidate for row in records_out for candidate in row["candidates"]]
        beams = [beam for candidate in candidates for beam in candidate["history_beams"]]
        valid_beams = [beam for beam in beams if beam["predicted_sid"] is not None]
        summary = {
            "checkpoint": int(label), "adapter": str(CHECKPOINTS[label]),
            "probe_groups": 12, "self_cot_count": len(candidates), "total_beams": len(beams),
            "closed_count": sum(candidate["closed"] for candidate in candidates),
            "closed_rate": sum(candidate["closed"] for candidate in candidates) / len(candidates),
            "empty_count": sum(candidate["empty"] for candidate in candidates),
            "valid_cot_count": sum(candidate["closed"] and not candidate["empty"] for candidate in candidates),
            "completion_token_mean": statistics.fmean(candidate["completion_length"] for candidate in candidates),
            "completion_token_min": min(candidate["completion_length"] for candidate in candidates),
            "completion_token_max": max(candidate["completion_length"] for candidate in candidates),
            "literal_sid_occurrences": sum(candidate["literal_sid_occurrences"] for candidate in candidates),
            "bare_raw_mean": statistics.fmean(float(candidate["reward"]) for candidate in candidates),
            "exact": sum(int(candidate["exact"]) for candidate in candidates),
            "ab": sum(int(candidate["ab"]) for candidate in candidates),
            "a": sum(int(candidate["a"]) for candidate in candidates),
            "invalid": sum(int(candidate["invalid"]) for candidate in candidates),
            "abc_invalid_rate": sum(int(candidate["invalid"]) for candidate in candidates) / len(beams),
            "history_exact_copy_rate_valid": sum(beam["copy_class"] == "EXACT_COPY" for beam in valid_beams) / len(valid_beams),
            "history_not_gold_rate": sum(beam["gold_history_class"] == "HISTORY_NOT_GOLD" for beam in beams) / len(beams),
            "generation_wall_mean_sec": statistics.fmean(row["generation_wall_sec"] for row in records_out),
            "records": records_out,
        }
        (RESULTS / f"checkpoint_{label}.json").parent.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"checkpoint_{label}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"SELF_COT_CHECKPOINT_RESULT={RESULTS / f'checkpoint_{label}.json'}", flush=True)


def finalize():
    step0 = json.loads((RESULTS / "checkpoint_0.json").read_text(encoding="utf-8"))
    step300 = json.loads((RESULTS / "checkpoint_300.json").read_text(encoding="utf-8"))
    failures = []
    if step300["closed_rate"] < 0.95: failures.append("STEP300_CLOSED_RATE_LT_0.95")
    if step300["valid_cot_count"] < step0["valid_cot_count"] - 2: failures.append("STEP300_VALID_COT_DROP_GE_2")
    if step300["bare_raw_mean"] < 0.9 * step0["bare_raw_mean"]: failures.append("STEP300_BARE_RAW_LT_90PCT_STEP0")
    if step300["abc_invalid_rate"] > 0.01: failures.append("STEP300_ABC_INVALID_GT_1PCT")
    if step300["empty_count"]: failures.append("STEP300_EMPTY_COT")
    result = {
        "type": "boundary_adaptation_self_cot_health_gate", "seed": SEED,
        "production_generation_path": "ThinkCompositeInterestRecGRPOTrainer._generate_single_turn via FixedProbeEvaluator._think",
        "checkpoints": {"0": step0, "300": step300},
        "phase_a_pass": not failures, "hard_stop_reasons": failures,
        "training_started": False, "optimizer_created": False,
    }
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    FINAL.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE_A_PASS={'YES' if not failures else 'NO'}")
    print(f"FINAL_RESULT={FINAL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.finalize):
        raise SystemExit("choose exactly one operation")
    run_checkpoint(args.checkpoint) if args.checkpoint else finalize()
