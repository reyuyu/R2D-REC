#!/usr/bin/env python3
"""Validate that a V4.3 run exercised the complete four-GPU smoke path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _required_metric_values(
    records: list[dict[str, object]], keys: tuple[str, ...], label: str
) -> list[float]:
    missing = [key for key in keys if not any(key in record for record in records)]
    if missing:
        raise AssertionError(f"smoke is missing {label} metrics: {missing}")
    values = [float(record[key]) for record in records for key in keys if key in record]
    if any(not math.isfinite(value) for value in values):
        raise AssertionError(f"smoke has non-finite {label}: {values}")
    return values


def validate_smoke(
    run_dir: Path,
    stage: str,
    *,
    allow_aux_review: bool = False,
) -> dict[str, object]:
    world_size = int((run_dir / "smoke_world_size.txt").read_text(encoding="utf-8").strip())
    if world_size != 4:
        raise AssertionError(f"expected four-GPU smoke, got world_size={world_size}")
    records = [
        json.loads(line)
        for line in (run_dir / "trainer_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_records = [record for record in records if "loss" in record and int(record.get("step", 0)) > 0]
    if not train_records or max(int(record["step"]) for record in train_records) < 2:
        raise AssertionError("smoke did not complete two optimizer steps")
    if any(not math.isfinite(float(record["loss"])) for record in train_records):
        raise AssertionError("non-finite smoke loss")
    if stage == "hcr_off":
        deltas = [
            float(value)
            for record in train_records
            for key, value in record.items()
            if key.endswith("hcr_off_loss_delta")
        ]
        if not deltas or any(value != 0.0 for value in deltas):
            raise AssertionError(f"HCR-off regression delta is not exactly zero: {deltas}")

    summary: dict[str, object] = {
        "status": "passed",
        "stage": stage,
        "steps": 2,
        "world_size": 4,
    }
    if stage == "s2":
        history_losses = _required_metric_values(
            train_records, ("train/history_fdr_total",), "history-FDR loss"
        )
        if not any(value > 0.0 for value in history_losses):
            raise AssertionError(f"S2 history-FDR loss was never active: {history_losses}")

        pair_counts = _required_metric_values(
            train_records,
            tuple(f"train/history_pair_count_{level}" for level in ("a", "b", "c")),
            "history pair-count",
        )
        if any(value < 0.0 for value in pair_counts) or sum(pair_counts) <= 0.0:
            raise AssertionError(f"S2 selected no history-FDR pairs: {pair_counts}")

        evidence = _required_metric_values(
            train_records,
            tuple(
                f"train/behavior_evidence_{side}_{level}"
                for side in ("pos", "neg")
                for level in ("A", "B", "C")
            ),
            "behavior evidence",
        )
        if any(value < 0.0 for value in evidence) or not any(value > 0.0 for value in evidence):
            raise AssertionError(f"S2 behavior evidence was never active: {evidence}")

        selected_support = _required_metric_values(
            train_records,
            ("train/selected_history_competitor_behavior_support",),
            "selected-competitor behavior support",
        )
        if any(value < 0.0 for value in selected_support) or not any(value > 0.0 for value in selected_support):
            raise AssertionError(
                f"S2 selected competitors have no behavior support: {selected_support}"
            )

        selected_frequency = _required_metric_values(
            train_records,
            ("train/selected_history_competitor_raw_frequency",),
            "selected-competitor raw frequency",
        )
        if any(value < 0.0 for value in selected_frequency) or not any(value > 0.0 for value in selected_frequency):
            raise AssertionError(
                f"S2 selected competitors have no raw-frequency evidence: {selected_frequency}"
            )
        summary["history_fdr"] = {
            "max_loss": max(history_losses),
            "pair_count_sum": sum(pair_counts),
            "max_behavior_evidence": max(evidence),
            "max_selected_behavior_support": max(selected_support),
            "max_selected_raw_frequency": max(selected_frequency),
        }
    if stage == "s3":
        topk_losses = _required_metric_values(
            train_records, ("train/hcr_topk_total",), "S1 TopK loss"
        )
        if not any(value > 0.0 for value in topk_losses):
            raise AssertionError(f"S3 did not retain active S1 TopK supervision: {topk_losses}")

        multi_losses = _required_metric_values(
            train_records, ("train/multi_a_loss",), "Video Multi-A loss"
        )
        multi_active = _required_metric_values(
            train_records, ("train/video_multi_a_active",), "Video Multi-A activation"
        )
        positive_counts = _required_metric_values(
            train_records, ("train/video_positive_a_count",), "Video positive-A count"
        )
        if not any(value > 0.0 for value in multi_losses):
            raise AssertionError(f"S3 Video Multi-A loss was never active: {multi_losses}")
        if not any(value > 0.0 for value in multi_active):
            raise AssertionError(f"S3 selected no eligible Video Multi-A group: {multi_active}")
        if not any(value >= 2.0 for value in positive_counts):
            raise AssertionError(f"S3 saw no group with at least two positive A modes: {positive_counts}")

        disabled_history_keys = (
            "train/history_fdr_total",
            *(f"train/history_pair_count_{level}" for level in ("a", "b", "c")),
            "train/selected_history_competitor_behavior_support",
            "train/selected_history_competitor_raw_frequency",
        )
        disabled_history = _required_metric_values(
            train_records, disabled_history_keys, "disabled History-FDR"
        )
        if any(value != 0.0 for value in disabled_history):
            raise AssertionError(
                f"S3 must keep History-FDR exactly disabled, got {disabled_history}"
            )

        scale_keys = (
            "train/hcr_aux_total",
            "train/rec_base_v42",
            "train/weighted_multi_a",
        )
        scale_records = [
            record for record in train_records if all(key in record for key in scale_keys)
        ]
        if not scale_records:
            raise AssertionError(
                f"S3 smoke has no complete auxiliary-scale record: {scale_keys}"
            )
        aux_ratios: list[float] = []
        multi_ratios: list[float] = []
        for record in scale_records:
            base = float(record["train/rec_base_v42"])
            aux = float(record["train/hcr_aux_total"])
            weighted_multi = float(record["train/weighted_multi_a"])
            if not all(math.isfinite(value) for value in (base, aux, weighted_multi)):
                raise AssertionError("S3 auxiliary-scale metrics must be finite")
            if base <= 0.0 or aux < 0.0 or weighted_multi < 0.0:
                raise AssertionError(
                    f"invalid S3 auxiliary scale: base={base}, aux={aux}, multi={weighted_multi}"
                )
            aux_ratios.append(aux / base)
            multi_ratios.append(weighted_multi / base)
        max_aux_ratio = max(aux_ratios)
        max_multi_ratio = max(multi_ratios)
        if max_aux_ratio > 0.30:
            raise AssertionError(
                f"S3 HCR auxiliary exceeds the 30% hard limit: {max_aux_ratio:.6f}"
            )
        if max_aux_ratio >= 0.20 and not allow_aux_review:
            raise AssertionError(
                "S3 HCR auxiliary is in the 20%-30% manual-review band: "
                f"{max_aux_ratio:.6f}; rerun only with --allow-aux-review after review"
            )

        repetition = _required_metric_values(
            train_records,
            (
                "train/video_multi_a_active_instances_per_pack",
                "train/video_multi_a_unique_eligible_groups_per_pack",
                "train/video_multi_a_repeat_factor",
            ),
            "Video Multi-A repetition diagnostics",
        )
        repeat_records = [
            record
            for record in train_records
            if "train/video_multi_a_active_instances_per_pack" in record
            and "train/video_multi_a_unique_eligible_groups_per_pack" in record
        ]
        effective_repeat_factors = [
            float(record["train/video_multi_a_active_instances_per_pack"])
            / float(record["train/video_multi_a_unique_eligible_groups_per_pack"])
            for record in repeat_records
            if float(record["train/video_multi_a_unique_eligible_groups_per_pack"]) > 0.0
        ]
        summary["s1_video_multi_a"] = {
            "max_topk_loss": max(topk_losses),
            "max_multi_a_loss": max(multi_losses),
            "active_count": sum(multi_active),
            "max_video_positive_a_count": max(positive_counts),
            "history_fdr_exact_zero": True,
            "max_hcr_aux_over_base_rec": max_aux_ratio,
            "max_weighted_multi_a_over_base_rec": max_multi_ratio,
            "auxiliary_gate": (
                "manual_review_approved" if max_aux_ratio >= 0.20 else "automatic_pass"
            ),
            "repetition_diagnostics": repetition,
            "max_effective_repeat_factor": (
                max(effective_repeat_factors) if effective_repeat_factors else 0.0
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--allow-aux-review", action="store_true")
    args = parser.parse_args()

    print(
        json.dumps(
            validate_smoke(
                args.run_dir,
                args.stage,
                allow_aux_review=args.allow_aux_review,
            )
        )
    )


if __name__ == "__main__":
    main()
