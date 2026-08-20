#!/usr/bin/env python3
"""Attach matched C0/C40/Final Chain Probe v2 evidence to a full-epoch result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from diagnose_user_chain import (
    METRICS,
    SEED,
    _checkpoint_summary,
    _group_means,
    bootstrap_mean_ci,
    paired_deltas,
    read_jsonl,
)


LABELS = ("C0", "C40", "Final")
DOCS_MARKER = "## Final Chain Probe v2"


def readiness(training_summary: dict, comparisons: dict) -> tuple[str, list[str]]:
    integrity = training_summary["integrity"]
    failures = []
    required = {
        "base_frozen": integrity.get("base_frozen"),
        "lora_updated": integrity.get("lora_updated"),
        "dataset_sha_unchanged": integrity.get("dataset_sha_unchanged"),
        "sample_coverage_exact": integrity.get("sample_coverage_exact"),
        "nan_or_inf_false": not integrity.get("nan_or_inf", True),
    }
    failures.extend(name for name, passed in required.items() if not passed)
    final_vs_c40 = comparisons["Final-C40"]["total_reward"]
    ci = final_vs_c40["bootstrap_95pct_ci"]
    if final_vs_c40["mean"] <= -0.01 and ci[1] < 0:
        failures.append("significant_chain_total_collapse_vs_c40")
    return (
        "INTERNAL_READY_FOR_EXTERNAL_EVAL" if not failures else "INTERNAL_NOT_READY",
        failures,
    )


def validate_final_integrity(integrity: dict) -> None:
    required = {
        "label_final": integrity.get("label") == "Final",
        "candidate_count_160": integrity.get("candidate_count") == 160,
        "lora_checksum_unchanged": integrity.get("lora_checksum_unchanged"),
        "rng_restored": integrity.get("rng_restored"),
        "requires_grad_zero": integrity.get("requires_grad_parameter_count") == 0,
    }
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise RuntimeError(f"Final Probe v2 integrity contract failed: {failures}")


def finalize(
    training_summary: dict,
    rows_by_label: dict[str, list[dict]],
    final_integrity: dict | None = None,
) -> dict:
    if final_integrity is not None:
        validate_final_integrity(final_integrity)
    expected_ids = None
    groups = {}
    for label in LABELS:
        rows = rows_by_label[label]
        if len(rows) != 160 or any(row.get("checkpoint") != label for row in rows):
            raise RuntimeError(f"{label} must contain 40 matched G4 groups with the true label")
        sample_ids = {row["sample_id"] for row in rows}
        if len(sample_ids) != 40:
            raise RuntimeError(f"{label} does not contain 40 unique sample IDs")
        expected_ids = sample_ids if expected_ids is None else expected_ids
        if sample_ids != expected_ids:
            raise RuntimeError("C0/C40/Final Probe v2 sample IDs are not matched")
        groups[label] = _group_means(rows)

    comparisons = {}
    for comparison_index, (name, left, right) in enumerate((
        ("C40-C0", "C0", "C40"),
        ("Final-C0", "C0", "Final"),
        ("Final-C40", "C40", "Final"),
    )):
        comparison = paired_deltas(groups, left, right)
        for metric_index, metric in enumerate(METRICS):
            values = [
                groups[right][sample_id][metric] - groups[left][sample_id][metric]
                for sample_id in sorted(expected_ids)
            ]
            comparison[metric]["bootstrap_95pct_ci"] = bootstrap_mean_ci(
                values,
                seed=SEED + comparison_index * len(METRICS) + metric_index,
            )
        comparisons[name] = comparison
    checkpoints = {label: _checkpoint_summary(rows_by_label[label]) for label in LABELS}
    if not all(math.isfinite(checkpoints[label][metric]) for label in LABELS for metric in ("Total", "Action", "Logic")):
        raise RuntimeError("non-finite Final Probe v2 aggregate")
    status, failures = readiness(training_summary, comparisons)
    training_summary["final_chain_probe_v2"] = {
        "matched_samples": 40,
        "G": 4,
        "checkpoints": checkpoints,
        "comparisons": comparisons,
        "inference_integrity": final_integrity,
    }
    training_summary["internal_status"] = status
    training_summary["internal_readiness_failures"] = failures
    return training_summary


def render_docs(existing: str, summary: dict) -> str:
    probe = summary["final_chain_probe_v2"]
    lines = [
        DOCS_MARKER,
        "",
        "Matched inference-only probe: 40 Chain samples, G=4 (160 candidates).",
        "",
        "| Checkpoint | Total | Action | Logic |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        metrics = probe["checkpoints"][label]
        lines.append(
            f"| {label} | {metrics['Total']:.6f} | {metrics['Action']:.6f} | {metrics['Logic']:.6f} |"
        )
    lines.extend(["", "| Comparison | Metric | Mean delta | Bootstrap 95% CI |", "| --- | --- | ---: | ---: |"])
    for name in ("C40-C0", "Final-C0", "Final-C40"):
        for metric in METRICS:
            item = probe["comparisons"][name][metric]
            ci = item["bootstrap_95pct_ci"]
            lines.append(f"| {name} | {metric} | {item['mean']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    lines.extend(["", f"Internal status: `{summary['internal_status']}`.", ""])
    prefix = existing.split(DOCS_MARKER, 1)[0].rstrip()
    return (prefix + "\n\n" if prefix else "") + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--c0", type=Path, required=True)
    parser.add_argument("--c40", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--final-integrity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-summary-output", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--docs-output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    final_integrity = (
        json.loads(args.final_integrity.read_text(encoding="utf-8"))
        if args.final_integrity else None
    )
    summary = finalize(summary, {
        "C0": read_jsonl(args.c0),
        "C40": read_jsonl(args.c40),
        "Final": read_jsonl(args.final),
    }, final_integrity=final_integrity)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    if args.run_summary_output:
        args.run_summary_output.write_text(serialized, encoding="utf-8")
    if args.run_manifest:
        manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest["internal_status"] = summary["internal_status"]
        manifest["final_chain_probe_v2"] = {
            "label": "Final",
            "sample_count": 40,
            "candidate_count": 160,
            "integrity_passed": True,
        }
        args.run_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.docs_output:
        existing = args.docs_output.read_text(encoding="utf-8") if args.docs_output.exists() else ""
        args.docs_output.write_text(render_docs(existing, summary), encoding="utf-8")
    print(json.dumps({
        "status": summary["internal_status"],
        "failures": summary["internal_readiness_failures"],
        "checkpoints": summary["final_chain_probe_v2"]["checkpoints"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
