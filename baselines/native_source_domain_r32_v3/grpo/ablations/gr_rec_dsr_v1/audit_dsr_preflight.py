#!/usr/bin/env python3
"""CPU-only reproducibility audit for the DSR formal data/probe contract."""
from __future__ import annotations

import hashlib
import json
import sys

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

import run_grpo_trl_train as baseline_train

from gr_rec_dsr_v1.dsr_contract import (
    PROBE_EVERY_STEPS,
    PROBE_GROUP_IDS,
    PROBE_SEED,
    TRAIN_SEED,
    enforce_formal_contract,
)


def _sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _descriptor(plan):
    rows = list(plan["dataset"])
    sampler = baseline_train.RouteAwareRepeatSampler(
        plan["dataset"], generation_batch_size=16, repeat_count=2, shuffle=False
    )
    candidate_order = []
    seen = set()
    for row in rows:
        group_id = row["recommendation_group_id"]
        if group_id not in seen:
            seen.add(group_id)
            candidate_order.append(group_id)
    generation_schedule = []
    optimizer_schedule = []
    trained = set()
    for route, indices in sampler._chunks:
        group_ids = [rows[index]["recommendation_group_id"] for index in indices]
        trained.update(group_ids)
        generation_schedule.append({"route": route, "group_ids": group_ids})
        for policy_iteration in range(sampler.repeat_count):
            optimizer_schedule.append({
                "route": route,
                "group_ids": group_ids,
                "policy_iteration": policy_iteration,
            })
    return {
        "candidate_order": candidate_order,
        "candidate_order_sha256": _sha256(candidate_order),
        "trained_group_ids": sorted(trained),
        "trained_group_ids_sha256": _sha256(sorted(trained)),
        "generation_schedule": generation_schedule,
        "generation_schedule_sha256": _sha256(generation_schedule),
        "optimizer_schedule": optimizer_schedule,
        "schedule_sha256": _sha256(optimizer_schedule),
    }


def main():
    base = ["--run-id", "GR-REC-DSR-V1-PREFLIGHT"]
    dsr_argv = enforce_formal_contract(base)
    dsr_args = baseline_train.build_arg_parser().parse_args(dsr_argv)
    dsr_plan = baseline_train.prepare_run_plan(dsr_args)

    baseline_argv = [
        "--run-id", "GR-REC-V1-PREFLIGHT",
        "--seed", TRAIN_SEED,
        "--n-groups", "all",
        "--probe-groups", "4",
        "--probe-seed", PROBE_SEED,
        "--probe-every-steps", PROBE_EVERY_STEPS,
    ]
    for group_id in PROBE_GROUP_IDS:
        baseline_argv.extend(["--probe-group-id", group_id])
    baseline_args = baseline_train.build_arg_parser().parse_args(baseline_argv)
    baseline_plan = baseline_train.prepare_run_plan(baseline_args)

    dsr = _descriptor(dsr_plan)
    baseline = _descriptor(baseline_plan)
    probe_domains = [
        dsr_plan["probe_records"][group_id]["think"]["target_domain"]
        for group_id in dsr_plan["probe_group_ids"]
    ]
    audit = dsr_plan["audit"]
    result = {
        "train_seed": int(TRAIN_SEED),
        "probe": {
            "group_ids": dsr_plan["probe_group_ids"],
            "domains": probe_domains,
            "seed": int(PROBE_SEED),
            "every_steps": int(PROBE_EVERY_STEPS),
            "think_g": baseline_train.M_THINK,
            "nothink_g": baseline_train.M_NO,
            "excluded_from_training": not set(PROBE_GROUP_IDS).intersection(dsr["candidate_order"]),
        },
        "raw_groups": dsr_plan["raw_groups"],
        "excluded_probe_groups": len(PROBE_GROUP_IDS),
        "candidate_train_groups": audit["selected_groups"],
        "dropped_tail_groups": audit["dropped_groups"],
        "dropped_group_ids": audit["dropped_group_ids"],
        "actual_trained_groups": audit["trained_groups"],
        "think_rollouts": audit["think_rollouts"],
        "nothink_rollouts": audit["nothink_rollouts"],
        "optimizer_steps": audit["optimizer_steps"],
        "generation_route_preview": [item["route"] for item in dsr["generation_schedule"][:24]],
        "optimizer_route_preview": [item["route"] for item in dsr["optimizer_schedule"][:24]],
        "candidate_order_sha256": dsr["candidate_order_sha256"],
        "trained_group_ids_sha256": dsr["trained_group_ids_sha256"],
        "schedule_sha256": dsr["schedule_sha256"],
        "generation_schedule_sha256": dsr["generation_schedule_sha256"],
        "baseline_schedule_sha256": baseline["schedule_sha256"],
        "baseline_candidate_order_sha256": baseline["candidate_order_sha256"],
        "exact_same_schedule_as_gr_rec_v1": dsr == baseline,
    }
    expected_domains = ["video", "living", "prod", "ad"]
    if probe_domains != expected_domains:
        raise AssertionError(f"probe domain order drift: {probe_domains}")
    if not result["probe"]["excluded_from_training"]:
        raise AssertionError("probe group leaked into the DSR training dataset")
    if not result["exact_same_schedule_as_gr_rec_v1"]:
        raise AssertionError("DSR and GR_REC_v1 schedules differ")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
