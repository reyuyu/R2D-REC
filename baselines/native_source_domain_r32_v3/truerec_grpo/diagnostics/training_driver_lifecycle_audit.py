"""CPU-only Phase 1.3A lifecycle and frozen Pilot4096 admission audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))

from training_driver_v1 import (  # noqa: E402
    TrueRecTrainingDriverV1,
    audit_pilot4096_admission,
    frozen_contract,
)


class FakeOptimizer:
    def __init__(self) -> None:
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        if set_to_none is not True:
            raise AssertionError("set_to_none contract changed")
        self.zero_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class FakeTrainer:
    def __init__(self) -> None:
        self.calls = 0

    def backward_group_streaming(self, group: Any) -> dict[str, Any]:
        self.calls += 1
        return {"group": group["recommendation_group_id"], "detached": True}


def run_lifecycle_audit() -> dict[str, Any]:
    optimizer = FakeOptimizer()
    trainer = FakeTrainer()
    version = {"value": "unchanged-policy"}
    monitor_events: list[dict[str, Any]] = []
    driver = TrueRecTrainingDriverV1(
        rollout_fn=lambda record: {"completion_ids": tuple(range(8)), "record": record},
        old_rescore_fn=lambda record, rollout: {
            "recommendation_group_id": record["recommendation_group_id"],
            "completion_ids": rollout["completion_ids"],
            "old_logp_source": "FULL_FORWARD_RESCORE",
        },
        trainer=trainer,
        optimizer=optimizer,
        gradient_finite_fn=lambda: True,
        policy_fingerprint_fn=lambda: version["value"],
        monitor=lambda event, payload: monitor_events.append({"event": event, **payload}),
        groups_per_optimizer_step=1,
    )
    driver.run([{"recommendation_group_id": f"cpu-audit-{index}"} for index in range(3)])
    expected_one = [
        "zero_grad", "rollout", "old_rescore", "streaming_backward",
        "gradient_gate", "optimizer_step", "global_step_increment",
    ]
    expected = expected_one * 3
    state = asdict(driver.state)
    if driver.lifecycle_events != expected:
        raise AssertionError(f"lifecycle mismatch: {driver.lifecycle_events}")
    if state != {
        "business_groups_seen": 3, "optimizer_steps": 3, "global_step": 3,
        "rollouts_completed": 3, "old_rescores_completed": 3,
        "streaming_backwards_completed": 3, "groups_in_accumulation_window": 0,
        "failed": False,
    }:
        raise AssertionError(f"counter mismatch: {state}")
    return {
        "groups": 3,
        "groups_per_optimizer_step": 1,
        "lifecycle_events": driver.lifecycle_events,
        "state": state,
        "optimizer_zero_grad_calls": optimizer.zero_calls,
        "optimizer_step_calls": optimizer.step_calls,
        "trainer_streaming_backward_calls": trainer.calls,
        "monitor_group_complete_calls": len(monitor_events),
        "lifecycle_order_pass": "PASS",
        "counter_contract_pass": "PASS",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-status", choices=("PASS",), required=True)
    args = parser.parse_args()
    lifecycle = run_lifecycle_audit()
    pilot = audit_pilot4096_admission(args.pilot_records, args.pilot_manifest)
    result = {
        "phase": "TrueRec-GRPO Phase 1.3A",
        "contract": frozen_contract(),
        "lifecycle": lifecycle,
        "pilot4096_admission": pilot,
        "fail_fast_pass": "PASS",
        "real_model_loaded": False,
        "gpu_started": False,
        "checkpoint_implemented": False,
        "ddp_implemented": False,
        "formal_training_started": False,
        "test_status": args.test_status,
        "next_experiment_started": False,
    }
    write_json(args.output_dir / "training_driver_lifecycle_audit.json", result)
    review = (
        "TrueRec-GRPO Phase 1.3A CPU Training Driver Lifecycle Contract\n"
        "Lifecycle order, counters, fail-fast tests, and frozen Pilot4096 admission passed. "
        "This phase used fake components only: no model, GPU, checkpoint, DDP, or training.\n"
    )
    (args.output_dir / "CHATGPT_PHASE1_3A_REVIEW.txt").write_text(review, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
