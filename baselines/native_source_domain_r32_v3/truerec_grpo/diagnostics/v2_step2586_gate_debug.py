"""Replay V2 checkpoint 2560 and diagnose the step2586 pre-optimizer gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from checkpoint_ddp_v1 import load_distributed_checkpoint  # noqa: E402
from distributed_trainer_v1 import distributed_fail_if  # noqa: E402
from monitoring_v1 import build_train_explain  # noqa: E402
from real_three_group_resume_smoke import LoraCheckpointState, load_runtime, optimizer_step_value  # noqa: E402
from run_truerec_pilot_ddp_v1 import gather_rank_objects, initialize, run_loaded_group  # noqa: E402
from run_truerec_v2_curriculum_ddp_v1 import (  # noqa: E402
    EPOCH1_ORDER_SHA256, EPOCH_STEPS, RECORDS_SHA256, TOTAL_STEPS,
    _load_epoch2_order, _read_jsonl, driver_state_for_cursor, load_curriculum, order_metadata,
)


ORIGINAL_RUN = Path("/root/GRPO/truerec_grpo/runs/TRUEREC-V2-V1STEP4096-CURRICULUM2048-2E-4GPU")
CHECKPOINT = ORIGINAL_RUN / "checkpoints/checkpoint-step-2560"
DEFAULT_OUTPUT = Path("/root/GRPO/truerec_grpo/debug/v2_step2586_gate")
CHECKPOINT_CURSOR = 2560
FAILED_GROUP_INDEX = 2585
FAILED_GLOBAL_STEP = 2586
EPOCH2_LOCAL_INDEX = 537
EXPECTED_FAILED_GROUP_ID = "6005f443f291be018ce1ec7dbf3a1e512e21b477a7ff9acc10803752e2c4edcd"


def json_write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def trajectory_payload(explain: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in explain.items()
        if key not in {"global_step", "group_index", "mode", "optimizer_update"}
    }


def zero_gradient_assessment(
    diagnostic: dict[str, Any], explain: dict[str, Any],
) -> dict[str, Any]:
    action_tokens = [
        token for candidate in explain["candidates"] for token in candidate["action_tokens"]
    ]
    frontier_all_zero = all(float(token["frontier_token_credit"]) == 0 for token in action_tokens)
    format_all_zero = all(float(token["format_token_credit"]) == 0 for token in action_tokens)
    hpr_none = explain["hpr_trigger"] in {"HPR_NONE", "NONE"}
    has_hpr_sites = bool(explain["hpr_sites"])
    loss = diagnostic["loss_gate"]
    loss_strict_zero = all(float(loss[name]) == 0 for name in ("frontier", "hpr_raw", "hpr_weighted", "total"))
    theoretical_signal = not frontier_all_zero or not format_all_zero or has_hpr_sites
    unique_failure = diagnostic["EXACT_FAILED_SUBGATES"] == ["ZERO_GRADIENT"]
    if not unique_failure:
        classification = "NOT_APPLICABLE"
    elif loss_strict_zero and frontier_all_zero and format_all_zero and hpr_none and not theoretical_signal:
        classification = "ZERO_GRADIENT_EXPECTED"
    else:
        classification = "ZERO_GRADIENT_BUG"
    return {
        "unique_failed_subgate_is_zero_gradient": unique_failure,
        "loss_strict_zero": loss_strict_zero,
        "frontier_credits_all_zero": frontier_all_zero,
        "format_credits_all_zero": format_all_zero,
        "hpr_none": hpr_none,
        "hpr_site_count": len(explain["hpr_sites"]),
        "theoretical_gradient_signal_present": theoretical_signal,
        "classification": classification,
    }


def group_metadata(record: dict[str, Any], selected_mb: int) -> dict[str, Any]:
    return {
        "FAILED_GROUP_ID": record["recommendation_group_id"],
        "domain": record["target_domain"],
        "hierarchy_class": record["hierarchy_class"],
        "K_A": record["K_A"], "K_AB": record["K_AB"], "K_ABC": record["K_ABC"],
        "context_tokens": record["context_token_count"],
        "selected_mb": selected_mb,
    }


def run(output_dir: Path) -> None:
    rank, device, resource = initialize()
    try:
        records, epoch1_order, _ = load_curriculum()
        epoch2_order, epoch2_manifest = _load_epoch2_order(ORIGINAL_RUN, records)
        full_order = epoch1_order + epoch2_order
        failed_group_id = full_order[FAILED_GROUP_INDEX]
        if epoch2_order[EPOCH2_LOCAL_INDEX] != failed_group_id or failed_group_id != EXPECTED_FAILED_GROUP_ID:
            raise RuntimeError("failed group index contract mismatch")
        model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
        ddp = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False,
        )
        ddp.eval(); model.eval()
        dataset_identity = {
            "records_sha256": RECORDS_SHA256,
            "epoch1_order_sha256": EPOCH1_ORDER_SHA256,
            "record_count": TOTAL_STEPS,
            "unique_group_count": EPOCH_STEPS,
        }
        restored = load_distributed_checkpoint(
            CHECKPOINT, model=LoraCheckpointState(model), optimizer=optimizer,
            current_dataset_identity=dataset_identity, current_group_ids=full_order, device=device,
            allow_custom_dataset=True,
            expected_order=order_metadata(full_order, name="v2_two_epoch_order"),
        )
        if (
            int(restored["next_group_index"]) != CHECKPOINT_CURSOR
            or restored["driver_state"] != driver_state_for_cursor(CHECKPOINT_CURSOR)
        ):
            raise RuntimeError("checkpoint2560 cursor/driver restore mismatch")
        restore_rows = gather_rank_objects({
            "rank": rank,
            "model_restore_exact": restored["model_restore_exact"],
            "rng_restored": restored["rng_restored"],
            "optimizer_step": optimizer_step_value(optimizer),
            "resource": resource,
        })
        original_explain = _read_jsonl(ORIGINAL_RUN / "train_explain.jsonl") if rank == 0 else []
        replay_rows: list[dict[str, Any]] = []
        for group_index in range(CHECKPOINT_CURSOR, FAILED_GROUP_INDEX):
            group_id = full_order[group_index]
            report, group, backward, payloads = run_loaded_group(
                group_id, records[group_id], model, ddp, optimizer, renderer, device,
                provenance=provenance, dropout=dropout, trainability=trainability,
                optimizer_audit=optimizer_audit,
            )
            replay_match = True
            if rank == 0:
                explain = build_train_explain(
                    group, backward.runtime_monitoring_plan, payloads,
                    renderer.tokenizer.convert_ids_to_tokens,
                )
                expected = original_explain[group_index]
                replay_match = (
                    expected["recommendation_group_id"] == group_id
                    and trajectory_payload(explain) == trajectory_payload(expected)
                )
                replay_rows.append({
                    "group_index": group_index, "global_step": group_index + 1,
                    "recommendation_group_id": group_id,
                    "trajectory_payload_exact": replay_match,
                    "runtime_plan_hash": backward.runtime_plan_hash,
                    "loss": report["loss"], "gradient": report["gradient"],
                })
            distributed_fail_if(not replay_match, "debug replay trajectory differs from original run", device)

        def write_target_diagnostic(diagnostic, group, backward, payloads) -> None:
            explain = build_train_explain(
                group, backward.runtime_monitoring_plan, payloads,
                renderer.tokenizer.convert_ids_to_tokens,
            )
            explain.update({
                "mode": "DEBUG_PRE_OPTIMIZER", "optimizer_update": False,
                "global_step_if_success": FAILED_GLOBAL_STEP,
                "group_index": FAILED_GROUP_INDEX,
            })
            zero_gradient = zero_gradient_assessment(diagnostic, explain)
            metadata = group_metadata(records[failed_group_id], diagnostic["call_gate"]["selected_mb"])
            summary = {
                "status": "GATE_FAILURE_REPRODUCED" if diagnostic["EXACT_FAILED_SUBGATES"] else "GATE_PASS_UNEXPECTED",
                "checkpoint": str(CHECKPOINT), "checkpoint_cursor": CHECKPOINT_CURSOR,
                "failed_group_index": FAILED_GROUP_INDEX,
                "global_step_if_success": FAILED_GLOBAL_STEP,
                "epoch2_local_index": EPOCH2_LOCAL_INDEX,
                **metadata,
                "checkpoint_restore_by_rank": restore_rows,
                "epoch2_order_sha256": epoch2_manifest["epoch2_order_sha256"],
                "replay_optimizer_steps": len(replay_rows),
                "replay_range": [CHECKPOINT_CURSOR, FAILED_GROUP_INDEX - 1],
                "replay_trajectory_exact_count": sum(row["trajectory_payload_exact"] for row in replay_rows),
                "replay_trajectory_all_exact": all(row["trajectory_payload_exact"] for row in replay_rows),
                "target_optimizer_step": False,
                "gate_diagnostic": diagnostic,
                "EXACT_FAILED_SUBGATES": diagnostic["EXACT_FAILED_SUBGATES"],
                "zero_gradient_assessment": zero_gradient,
                "replay_rows": replay_rows,
            }
            json_write_atomic(output_dir / "debug_summary.json", summary)
            json_write_atomic(output_dir / "failed_group_explain.json", {
                "group_metadata": metadata,
                "gate_diagnostic": diagnostic,
                "zero_gradient_assessment": zero_gradient,
                "explain": explain,
            })
            review = (
                f"TrueRec V2 step2586 gate diagnostic: {summary['status']}\n"
                f"FAILED_GROUP_ID={failed_group_id}\n"
                f"EXACT_FAILED_SUBGATES={diagnostic['EXACT_FAILED_SUBGATES']}\n"
                f"REPLAY_TRAJECTORY_EXACT={summary['replay_trajectory_exact_count']}/25\n"
                f"ZERO_GRADIENT_CLASSIFICATION={zero_gradient['classification']}\n"
                "The target group stopped before optimizer.step; no reward, PPO, HPR, Frontier, or LR math was changed.\n"
            )
            (output_dir / "REVIEW.txt").write_text(review, encoding="utf-8")

        try:
            run_loaded_group(
                failed_group_id, records[failed_group_id], model, ddp, optimizer, renderer, device,
                provenance=provenance, dropout=dropout, trainability=trainability,
                optimizer_audit=optimizer_audit,
                pre_optimizer_diagnostic_callback=write_target_diagnostic,
                stop_before_optimizer=True,
            )
        except RuntimeError as exc:
            if str(exc) not in {
                "pre-optimizer distributed gate failed",
                "requested diagnostic stop before optimizer",
            }:
                raise
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
