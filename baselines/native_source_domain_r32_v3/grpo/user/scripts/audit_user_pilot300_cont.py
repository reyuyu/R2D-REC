#!/usr/bin/env python3
"""CPU-only preflight for the Phase 5C continuation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_user_grpo_smoke import EXPECTED_DATA_SHA, dataset_sha, read_jsonl
from run_user_pilot150 import build_training_plan, load_resume_contract, select_pilot_rows


def bucket_inventory(rows, consumed_ids):
    consumed_ids = set(consumed_ids)
    inventory = {"action": {}, "chain": {}}
    action_specs = (("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20),
                    ("21-30", 21, 30), ("31-40", 31, 40), ("41+", 41, 10**9))
    for label, low, high in action_specs:
        pool = [row for row in rows if row["route"] == "action" and low <= int(row["gold_sid_count"]) <= high]
        consumed = sum(row["sample_id"] in consumed_ids for row in pool)
        inventory["action"][label] = {"total": len(pool), "consumed": consumed, "remaining": len(pool) - consumed}
    for event_count in (2, 3, 4, 5):
        pool = [row for row in rows if row["route"] == "chain" and int(row["gold_event_count"]) == event_count]
        consumed = sum(row["sample_id"] in consumed_ids for row in pool)
        inventory["chain"][str(event_count)] = {"total": len(pool), "consumed": consumed, "remaining": len(pool) - consumed}
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    resume = load_resume_contract(config)
    if resume is None:
        raise RuntimeError("Pilot300 preflight requires a resume contract")
    data_sha = dataset_sha(Path(config["dataset"]).parent)
    if data_sha != EXPECTED_DATA_SHA:
        raise RuntimeError("frozen GR_USER_v1 data SHA mismatch")
    dataset_rows = read_jsonl(config["dataset"])
    inventory = bucket_inventory(dataset_rows, resume["first_flat_ids"])
    print(json.dumps({"bucket_inventory": inventory}, ensure_ascii=False), flush=True)
    selected, selection_audit = select_pilot_rows(
        dataset_rows,
        config["seed"] + resume["config"]["cumulative_start_step"],
        resume["first_flat_ids"],
    )
    current_ids = [row["sample_id"] for route in ("action", "chain") for row in selected[route]]
    overlap = set(current_ids) & set(resume["first_flat_ids"])
    plan = build_training_plan(selected, resume["config"]["cumulative_start_step"])

    optimizer_state = torch.load(resume["optimizer_path"], map_location="cpu", weights_only=True)
    optimizer_steps = sorted({
        int(state["step"])
        for state in optimizer_state["state"].values()
        if "step" in state
    })
    result = {
        "status": "PASS",
        "dataset_sha": data_sha,
        "starting_processed_prompts": resume["metadata"]["processed_unique_prompts"],
        "starting_optimizer_step": resume["metadata"]["optimizer_steps"],
        "optimizer_state_count": len(optimizer_state["state"]),
        "optimizer_param_group_count": len(optimizer_state["param_groups"]),
        "optimizer_internal_steps": optimizer_steps,
        "bucket_inventory": inventory,
        "second150_overlap_with_first150": len(overlap),
        "second150_unique": len(set(current_ids)),
        "cumulative_unique": len(set(current_ids) | set(resume["first_flat_ids"])),
        "selection_audit": selection_audit,
        "plan_steps": [item["step"] for item in plan],
        "plan_routes": [item["route"] for item in plan],
    }
    if optimizer_steps != [20] or len(overlap) != 0 or result["cumulative_unique"] != 300:
        raise RuntimeError(f"Pilot300 continuation preflight failed: {result}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
