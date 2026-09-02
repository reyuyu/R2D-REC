#!/usr/bin/env python3
"""Fail-closed summary and root-cause localization for one replay pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def one_event(events: list[dict[str, Any]], event: str, label: str | None = None) -> dict[str, Any]:
    matches = [
        row for row in events if row.get("event") == event and (label is None or row.get("label") == label)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {event}/{label} event, found {len(matches)}")
    return matches[0]


def optional_event(events: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    matches = [row for row in events if row.get("event") == event]
    if len(matches) > 1:
        raise RuntimeError(f"Expected at most one {event} event, found {len(matches)}")
    return matches[0] if matches else None


def rank_files(run_dir: Path) -> dict[int, Path]:
    files = {}
    for path in (run_dir / "evidence").glob("rank*.jsonl"):
        files[int(path.stem.removeprefix("rank"))] = path
    if sorted(files) != [0, 1, 2, 3]:
        raise RuntimeError(f"Expected ranks 0..3, found {sorted(files)}")
    return files


def _batch_contract_equal(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    contract_a = a.get("batch_contract")
    contract_b = b.get("batch_contract")
    if not contract_a or not contract_b:
        return False, {"equal": False, "reason": "missing batch contract"}
    source = contract_a.get("source")
    if source != contract_b.get("source"):
        return False, {"equal": False, "reason": "batch fingerprint source differs"}
    if source == "frozen_step554_contract":
        keys = (
            "contract_sha256",
            "rank_ordered_batch_fingerprint",
            "expected_microbatch_count",
            "observed_microbatch_count",
            "runtime_ordered_metadata_sha256",
            "runtime_metadata_fingerprint",
        )
        equal = all(contract_a.get(key) == contract_b.get(key) for key in keys)
        count_valid = (
            contract_a.get("observed_microbatch_count") == contract_a.get("expected_microbatch_count")
            and contract_b.get("observed_microbatch_count") == contract_b.get("expected_microbatch_count")
        )
        return equal and count_valid, {
            "equal": equal and count_valid,
            "source": source,
            "contract_sha256": contract_a.get("contract_sha256"),
            "rank_ordered_batch_fingerprint": contract_a.get("rank_ordered_batch_fingerprint"),
            "microbatch_count_and_order_equal": equal,
            "expected_count_observed": count_valid,
        }
    if source == "runtime_full_gpu_hash":
        equal = (
            a.get("ordered_batch_fingerprint") == b.get("ordered_batch_fingerprint")
            and a.get("ordered_microbatch_sha256") == b.get("ordered_microbatch_sha256")
            and a.get("microbatch_count") == b.get("microbatch_count")
        )
        return equal, {
            "equal": equal,
            "source": source,
            "rank_ordered_batch_fingerprint": a.get("ordered_batch_fingerprint"),
            "microbatch_count_and_order_equal": a.get("ordered_microbatch_sha256")
            == b.get("ordered_microbatch_sha256"),
        }
    return False, {"equal": False, "reason": f"unknown batch contract source: {source}"}


def _bucket_rows(events: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in events if row.get("event") == event),
        key=lambda row: row["bucket_call_index"],
    )


def _bucket_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "bucket_call_index",
            "bucket_index",
            "layout_sha256",
            "dtype",
            "numel",
            "shape",
            "is_last",
            "parameter_count",
            "parameter_numel",
        )
    }


def _compare_buckets(
    events_a: list[dict[str, Any]], events_b: list[dict[str, Any]]
) -> dict[str, Any] | None:
    pre_a = _bucket_rows(events_a, "ddp_pre_allreduce")
    pre_b = _bucket_rows(events_b, "ddp_pre_allreduce")
    post_a = _bucket_rows(events_a, "ddp_post_allreduce")
    post_b = _bucket_rows(events_b, "ddp_post_allreduce")
    if not pre_a and not pre_b and not post_a and not post_b:
        return None
    layout_a = [_bucket_metadata(row) for row in pre_a]
    layout_b = [_bucket_metadata(row) for row in pre_b]
    post_layout_a = [_bucket_metadata(row) for row in post_a]
    post_layout_b = [_bucket_metadata(row) for row in post_b]
    layout_equal = layout_a == layout_b == post_layout_a == post_layout_b
    pre_equal = layout_equal and [row["fingerprint"] for row in pre_a] == [
        row["fingerprint"] for row in pre_b
    ]
    post_equal = layout_equal and [row["fingerprint"] for row in post_a] == [
        row["fingerprint"] for row in post_b
    ]
    rows = []
    if layout_equal:
        for index in range(len(pre_a)):
            rows.append(
                {
                    **layout_a[index],
                    "pre_a": pre_a[index]["fingerprint"],
                    "pre_b": pre_b[index]["fingerprint"],
                    "pre_equal": pre_a[index]["fingerprint"] == pre_b[index]["fingerprint"],
                    "post_a": post_a[index]["fingerprint"],
                    "post_b": post_b[index]["fingerprint"],
                    "post_equal": post_a[index]["fingerprint"] == post_b[index]["fingerprint"],
                }
            )
    return {
        "bucket_count_a": len(pre_a),
        "bucket_count_b": len(pre_b),
        "layout_equal": layout_equal,
        "pre_allreduce_equal": pre_equal,
        "post_allreduce_equal": post_equal,
        "buckets": rows,
    }


def _compare_clip(events_a: list[dict[str, Any]], events_b: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows_a = [row for row in events_a if row.get("event") == "gradient_clip"]
    rows_b = [row for row in events_b if row.get("event") == "gradient_clip"]
    if not rows_a and not rows_b:
        return None
    by_stage_a = {row["stage"]: row for row in rows_a}
    by_stage_b = {row["stage"]: row for row in rows_b}
    if set(by_stage_a) != {"PRE_CLIP", "POST_CLIP"} or set(by_stage_b) != {"PRE_CLIP", "POST_CLIP"}:
        return {
            "complete": False,
            "pre_clip_equal": False,
            "post_clip_equal": False,
            "returned_grad_norm_equal": False,
        }
    pre_a, pre_b = by_stage_a["PRE_CLIP"], by_stage_b["PRE_CLIP"]
    post_a, post_b = by_stage_a["POST_CLIP"], by_stage_b["POST_CLIP"]
    return {
        "complete": True,
        "pre_clip_a": pre_a["gradients"]["sha256"],
        "pre_clip_b": pre_b["gradients"]["sha256"],
        "pre_clip_equal": pre_a["gradients"]["sha256"] == pre_b["gradients"]["sha256"],
        "post_clip_a": post_a["gradients"]["sha256"],
        "post_clip_b": post_b["gradients"]["sha256"],
        "post_clip_equal": post_a["gradients"]["sha256"] == post_b["gradients"]["sha256"],
        "returned_grad_norm_a": post_a.get("returned_grad_norm"),
        "returned_grad_norm_b": post_b.get("returned_grad_norm"),
        "returned_grad_norm_equal": post_a.get("returned_grad_norm") == post_b.get("returned_grad_norm"),
        "pre_clip_tensor_count": pre_a["gradients"]["tensor_count"],
        "post_clip_tensor_count": post_a["gradients"]["tensor_count"],
    }


def summarize_pair(run_a: Path, run_b: Path) -> dict[str, Any]:
    files_a, files_b = rank_files(run_a), rank_files(run_b)
    events_a = {rank: load_events(files_a[rank]) for rank in range(4)}
    events_b = {rank: load_events(files_b[rank]) for rank in range(4)}
    ranks = []
    contract_reasons = []
    gradient_evidence_complete = True

    for rank in range(4):
        optimizer_a = one_event(events_a[rank], "optimizer_step")
        optimizer_b = one_event(events_b[rank], "optimizer_step")
        batch_equal, batch = _batch_contract_equal(optimizer_a, optimizer_b)
        restored_a = optional_event(events_a[rank], "rng_restored")
        restored_b = optional_event(events_b[rank], "rng_restored")
        rng_equal = bool(restored_a and restored_b and restored_a.get("rng") == restored_b.get("rng"))
        if not batch_equal:
            contract_reasons.append(f"rank{rank}: batch contract mismatch")
        if not rng_equal:
            contract_reasons.append(f"rank{rank}: restored RNG mismatch or missing")
        micro_losses_equal = optimizer_a.get("rank_local_micro_losses") == optimizer_b.get(
            "rank_local_micro_losses"
        )
        local_loss_equal = optimizer_a.get("rank_local_loss_mean") == optimizer_b.get("rank_local_loss_mean")
        buckets = _compare_buckets(events_a[rank], events_b[rank])
        clip = _compare_clip(events_a[rank], events_b[rank])
        if buckets is None or clip is None or not clip.get("complete"):
            gradient_evidence_complete = False
        elif not buckets["layout_equal"]:
            contract_reasons.append(f"rank{rank}: DDP bucket layout mismatch")
        ranks.append(
            {
                "rank": rank,
                "batch": batch,
                "rng_fingerprint_a": canonical_hash(restored_a["rng"]) if restored_a else None,
                "rng_fingerprint_b": canonical_hash(restored_b["rng"]) if restored_b else None,
                "rng_equal": rng_equal,
                "microbatch_count_a": optimizer_a.get("microbatch_count"),
                "microbatch_count_b": optimizer_b.get("microbatch_count"),
                "micro_losses_equal": micro_losses_equal,
                "local_loss_a": optimizer_a.get("rank_local_loss_mean"),
                "local_loss_b": optimizer_b.get("rank_local_loss_mean"),
                "local_loss_equal": local_loss_equal,
                "grad_norm_a": optimizer_a.get("grad_norm"),
                "grad_norm_b": optimizer_b.get("grad_norm"),
                "grad_norm_equal": optimizer_a.get("grad_norm") == optimizer_b.get("grad_norm"),
                "ddp": buckets,
                "clip": clip,
            }
        )

    initial_a = one_event(events_a[0], "heavy_fingerprint", "initial553")
    initial_b = one_event(events_b[0], "heavy_fingerprint", "initial553")
    final_a = one_event(events_a[0], "heavy_fingerprint", "step554")
    final_b = one_event(events_b[0], "heavy_fingerprint", "step554")
    checkpoint_a = initial_a.get("checkpoint_sha", {})
    checkpoint_b = initial_b.get("checkpoint_sha", {})
    checkpoint_equal = bool(
        checkpoint_a.get("exact")
        and checkpoint_b.get("exact")
        and checkpoint_a.get("actual") == checkpoint_b.get("actual")
    )
    initial_lora_equal = initial_a.get("canonical_lora_sha256") == initial_b.get("canonical_lora_sha256")
    initial_optimizer_equal = initial_a.get("optimizer", {}).get("sha256") == initial_b.get(
        "optimizer", {}
    ).get("sha256")
    if not checkpoint_equal:
        contract_reasons.append("initial checkpoint fingerprint mismatch or missing")
    if not initial_lora_equal:
        contract_reasons.append("initial LoRA fingerprint mismatch")
    if not initial_optimizer_equal:
        contract_reasons.append("initial optimizer fingerprint mismatch")

    final = {
        "initial_checkpoint_equal": checkpoint_equal,
        "initial_checkpoint_fingerprint": canonical_hash(checkpoint_a.get("actual")) if checkpoint_equal else None,
        "initial_lora_equal": initial_lora_equal,
        "initial_optimizer_equal": initial_optimizer_equal,
        "lora_a": final_a["canonical_lora_sha256"],
        "lora_b": final_b["canonical_lora_sha256"],
        "lora_equal": final_a["canonical_lora_sha256"] == final_b["canonical_lora_sha256"],
        "effective_ba_a": final_a["effective_ba"]["sha256"],
        "effective_ba_b": final_b["effective_ba"]["sha256"],
        "effective_ba_equal": final_a["effective_ba"]["sha256"] == final_b["effective_ba"]["sha256"],
        "optimizer_a": final_a["optimizer"]["sha256"],
        "optimizer_b": final_b["optimizer"]["sha256"],
        "optimizer_equal": final_a["optimizer"]["sha256"] == final_b["optimizer"]["sha256"],
    }
    contract_equal = not contract_reasons
    losses_equal = all(row["micro_losses_equal"] and row["local_loss_equal"] for row in ranks)
    final_equal = all(final[key] for key in ("lora_equal", "effective_ba_equal", "optimizer_equal"))
    grad_norms_equal = all(row["grad_norm_equal"] for row in ranks)
    gradient_states_equal = gradient_evidence_complete and all(
        row["ddp"]["pre_allreduce_equal"]
        and row["ddp"]["post_allreduce_equal"]
        and row["clip"]["pre_clip_equal"]
        and row["clip"]["post_clip_equal"]
        and row["clip"]["returned_grad_norm_equal"]
        for row in ranks
    )
    post_update_equal = gradient_states_equal and grad_norms_equal and final_equal

    if not contract_equal:
        repeatability = "CONTRACT_MISMATCH"
    elif not losses_equal:
        repeatability = "D1_PRE_BACKWARD"
    elif post_update_equal:
        repeatability = "D2_REPEATABLE"
    else:
        repeatability = "D3_POST_BACKWARD"

    if not contract_equal:
        verdict = "CONTRACT_MISMATCH"
    elif not losses_equal or not gradient_evidence_complete:
        verdict = "UNRESOLVED"
    elif any(not row["ddp"]["pre_allreduce_equal"] for row in ranks):
        verdict = "LOCAL_BACKWARD"
    elif any(not row["ddp"]["post_allreduce_equal"] for row in ranks):
        verdict = "DDP_NCCL"
    elif any(not row["clip"]["pre_clip_equal"] for row in ranks):
        verdict = "UNRESOLVED"
    elif any(
        not row["clip"]["post_clip_equal"] or not row["clip"]["returned_grad_norm_equal"]
        for row in ranks
    ):
        verdict = "GRAD_CLIPPING"
    elif not final_equal:
        verdict = "OPTIMIZER_UPDATE"
    else:
        verdict = "STEP554_FULLY_REPEATABLE"

    return {
        "repeatability_classification": repeatability,
        "verdict": verdict,
        "contract_equal": contract_equal,
        "contract_mismatch_reasons": contract_reasons,
        "pre_backward_local_loss_repeatable": contract_equal and losses_equal,
        "gradient_evidence_complete": gradient_evidence_complete,
        "post_backward_state_repeatable": post_update_equal,
        "ranks": ranks,
        "step554": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_pair(args.run_a, args.run_b), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
