#!/usr/bin/env python3
"""CPU-only static preflight for the User GRPO full-epoch continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch


EXPECTED_DATA_SHA = {
    "train": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
    "pilot": "1b846a0d434d529136b4d9784be3b913d7aaf3d0c8c2c87ce470ba53679a4803",
    "probe": "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_sha(data_dir: Path) -> dict[str, str]:
    paths = {
        "train": data_dir / "train_3000.jsonl",
        "pilot": data_dir / "pilot_600.jsonl",
        "probe": data_dir / "probe_v1.jsonl",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


ROUTES = ("action", "chain")


def select_remaining(rows: list[dict], consumed_ids: list[str], seed: int) -> dict[str, list[dict]]:
    consumed = set(consumed_ids)
    selected = {
        route: [row for row in rows if row["route"] == route and row["sample_id"] not in consumed]
        for route in ROUTES
    }
    random.Random(seed + 1).shuffle(selected["action"])
    random.Random(seed + 2).shuffle(selected["chain"])
    return selected


def build_formal_plan(selected: dict[str, list[dict]], starting_step: int = 40) -> list[dict]:
    chunks = {route: [selected[route][index:index + 8] for index in range(0, len(selected[route]), 8)] for route in ROUTES}
    if {route: [len(chunk) for chunk in values] for route, values in chunks.items()} != {
        "action": [8] * 168 + [6], "chain": [8] * 168 + [6]
    }:
        raise ValueError("formal route chunks must be 168x8 + 1x6")
    plan = []
    route_offset = {route: 0 for route in ROUTES}
    for chunk_index in range(169):
        for route in ROUTES:
            rows = chunks[route][chunk_index]
            plan.append({
                "step": starting_step + len(plan) + 1,
                "route": route,
                "rows": rows,
                "route_start": route_offset[route],
            })
            route_offset[route] += len(rows)
    if len(plan) != 338 or plan[-1]["step"] != 378 or route_offset != {"action": 1350, "chain": 1350}:
        raise AssertionError("formal training plan coverage drift")
    return plan


def distribute_formal_step_rows(rows: list[dict], rank: int, world_size: int = 4):
    if world_size != 4 or rank not in range(world_size) or len(rows) not in (6, 8):
        raise ValueError("formal DDP supports global batches of 8 or tail batches of 6")
    if len(rows) == 8:
        return rows[rank * 2:(rank + 1) * 2], False, 1.0
    counts = (2, 2, 1, 1)
    start = sum(counts[:rank])
    local = rows[start:start + counts[rank]]
    return local, False, world_size * len(local) / len(rows)


def load_resume(checkpoint: Path) -> dict:
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("global_optimizer_step") != 40 or metadata.get("processed_unique_prompts") != 300:
        raise RuntimeError("Pilot300 resume cursor must be step40/processed300")
    consumed = metadata.get("consumed_sample_ids", [])
    if len(consumed) != 300 or len(set(consumed)) != 300:
        raise RuntimeError("Pilot300 consumed IDs must contain 300 unique samples")
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer_steps = sorted({int(state["step"]) for state in optimizer["state"].values() if "step" in state})
    if len(optimizer["state"]) != 504 or optimizer_steps != [40]:
        raise RuntimeError("Pilot300 optimizer state contract failed")
    rng = []
    for rank in range(4):
        path = checkpoint / f"rng_rank{rank}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("rank") != rank or state.get("global_optimizer_step") != 40 or state.get("processed_unique_prompts") != 300:
            raise RuntimeError(f"rank {rank} RNG resume cursor mismatch")
        rng.append(path.name)
    return {"metadata": metadata, "consumed_ids": consumed, "optimizer_state_count": len(optimizer["state"]), "optimizer_steps": optimizer_steps, "rng_files": rng}


def run_preflight(config: dict) -> dict:
    data_path = Path(config["dataset"])
    data_sha = dataset_sha(data_path.parent)
    if data_sha != EXPECTED_DATA_SHA:
        raise RuntimeError("frozen GR_USER_v1 data SHA mismatch")
    resume = load_resume(Path(config["resume_checkpoint"]))
    rows = read_jsonl(data_path)
    if len(rows) != 3000 or len({row["sample_id"] for row in rows}) != 3000:
        raise RuntimeError("train_3000 must contain 3000 unique rows")
    by_id = {row["sample_id"]: row for row in rows}
    if not set(resume["consumed_ids"]).issubset(by_id):
        raise RuntimeError("Pilot300 consumed IDs are not a subset of train_3000")
    consumed_counts = {route: sum(by_id[sample_id]["route"] == route for sample_id in resume["consumed_ids"]) for route in ROUTES}
    selected = select_remaining(rows, resume["consumed_ids"], int(config["seed"]))
    remaining_ids = [row["sample_id"] for route in ROUTES for row in selected[route]]
    plan = build_formal_plan(selected, 40)
    union = set(resume["consumed_ids"]) | set(remaining_ids)
    probe_path = Path(config["fixed_probe"]["dataset"])
    probe_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    checks = {
        "starting_global_step": 40,
        "starting_processed_prompts": 300,
        "consumed_counts": consumed_counts,
        "remaining_total": len(remaining_ids),
        "remaining_action": len(selected["action"]),
        "remaining_chain": len(selected["chain"]),
        "remaining_consumed_overlap": len(set(remaining_ids) & set(resume["consumed_ids"])),
        "union_equals_train": union == set(by_id),
        "union_count": len(union),
        "plan_steps": len(plan),
        "expected_final_step": plan[-1]["step"],
        "probe_light_sha": probe_sha,
        "probe_light_sha_matches": probe_sha == config["fixed_probe"]["sha256"],
        "optimizer_state_count": resume["optimizer_state_count"],
        "optimizer_steps": resume["optimizer_steps"],
        "rng_files": resume["rng_files"],
        "dataset_sha": data_sha,
    }
    expected = {
        "consumed_counts": {"action": 150, "chain": 150},
        "remaining_total": 2700,
        "remaining_action": 1350,
        "remaining_chain": 1350,
        "remaining_consumed_overlap": 0,
        "union_equals_train": True,
        "union_count": 3000,
        "plan_steps": 338,
        "expected_final_step": 378,
        "probe_light_sha_matches": True,
    }
    failures = {key: (checks[key], value) for key, value in expected.items() if checks[key] != value}
    if failures:
        raise RuntimeError(f"full-epoch preflight failed: {failures}")
    return {"status": "PASS", **checks, "remaining_sample_ids": {route: [row["sample_id"] for row in selected[route]] for route in ROUTES}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "remaining_sample_ids"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
