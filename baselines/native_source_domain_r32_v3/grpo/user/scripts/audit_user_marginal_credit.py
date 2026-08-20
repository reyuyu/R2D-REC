"""CPU-only real-data audit for evaluator-aligned marginal credit."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from user_marginal_credit import action_marginal_credit, chain_marginal_credit


SEED = 20260821
EXPECTED_TRAIN_SHA256 = "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801"
CONSISTENCY_TOLERANCE = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_value(sample_id: str, salt: str) -> int:
    payload = f"{SEED}:{salt}:{sample_id}".encode()
    return int(hashlib.sha256(payload).hexdigest(), 16)


def select_rows(rows: Sequence[dict[str, Any]], route: str, count: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("route") == route]
    selected = sorted(
        candidates,
        key=lambda row: (stable_value(row["sample_id"], f"select:{route}"), row["sample_id"]),
    )[:count]
    if len(selected) != count:
        raise ValueError(f"expected {count} {route} rows, found {len(selected)}")
    return selected


def action_json(sids: Sequence[str]) -> str:
    return json.dumps(list(sids), ensure_ascii=False, separators=(",", ":"))


def chain_json(events: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        {"logic_chain": {"name": "MC real-data audit", "events": list(events)}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def choose(values: Sequence[Any], sample_id: str, salt: str) -> Any:
    if not values:
        raise ValueError(f"empty candidate set for {sample_id}:{salt}")
    return values[stable_value(sample_id, salt) % len(values)]


def choose_outside_history(row: Mapping[str, Any], sid_pool: Sequence[str]) -> str:
    history = set(row["history_sids"])
    start = stable_value(row["sample_id"], "action:history-out") % len(sid_pool)
    for offset in range(len(sid_pool)):
        candidate = sid_pool[(start + offset) % len(sid_pool)]
        if candidate not in history:
            return candidate
    raise ValueError(f"no History-out SID for {row['sample_id']}")


def action_variants(row: Mapping[str, Any], sid_pool: Sequence[str]) -> dict[str, dict[str, Any]]:
    gold = list(row["gold_sids"])
    history_non_gold = sorted(set(row["history_sids"]) - set(gold))
    dropped = choose(gold, row["sample_id"], "action:drop")
    duplicated = choose(gold, row["sample_id"], "action:duplicate")
    history_in_fp = choose(history_non_gold, row["sample_id"], "action:history-in")
    history_out_fp = choose_outside_history(row, sid_pool)
    return {
        "clean_gold": {"completion": action_json(gold)},
        "drop_gold": {
            "completion": action_json([sid for sid in gold if sid != dropped]),
            "target_sid": dropped,
        },
        "history_in_fp": {
            "completion": action_json([*gold, history_in_fp]),
            "target_sid": history_in_fp,
        },
        "history_out_fp": {
            "completion": action_json([*gold, history_out_fp]),
            "target_sid": history_out_fp,
        },
        "duplicate_gold": {
            "completion": action_json([*gold, duplicated]),
            "target_sid": duplicated,
        },
    }


def unrelated_text(sample_id: str, field: str) -> str:
    digest = hashlib.sha256(f"{sample_id}:{field}".encode()).hexdigest()[:24]
    return f"mc_unrelated_{field}_{digest}"


def chain_variants(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    gold = [dict(event) for event in row["gold_events"]]
    drop_index = stable_value(row["sample_id"], "chain:drop") % len(gold)
    action_index = stable_value(row["sample_id"], "chain:action") % len(gold)
    logic_index = stable_value(row["sample_id"], "chain:logic") % len(gold)

    unrelated_event = {
        "date": "9999-12-31",
        "action": unrelated_text(row["sample_id"], "action"),
        "logic": unrelated_text(row["sample_id"], "logic"),
    }
    changed_action = [dict(event) for event in gold]
    changed_action[action_index]["action"] = unrelated_text(row["sample_id"], "action_edit")
    changed_logic = [dict(event) for event in gold]
    changed_logic[logic_index]["logic"] = unrelated_text(row["sample_id"], "logic_edit")
    return {
        "clean_gold": {"completion": chain_json(gold)},
        "drop_event": {
            "completion": chain_json(gold[:drop_index] + gold[drop_index + 1 :]),
            "target_event_index": drop_index,
        },
        "add_unrelated_event": {
            "completion": chain_json([*gold, unrelated_event]),
            "target_event_index": len(gold),
        },
        "modify_action": {
            "completion": chain_json(changed_action),
            "target_event_index": action_index,
        },
        "modify_logic": {
            "completion": chain_json(changed_logic),
            "target_event_index": logic_index,
        },
    }


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def anomaly(
    reason: str,
    row: Mapping[str, Any],
    variant: str,
    completion: str,
    credit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "sample_id": row["sample_id"],
        "variant": variant,
        "completion": completion,
        "credit": dict(credit) if credit is not None else None,
        "sample": dict(row),
    }


def audit_action(rows: Sequence[dict[str, Any]], sid_pool: Sequence[str]) -> dict[str, Any]:
    tp_total = tp_positive = 0
    fp_total = fp_negative = 0
    duplicate_total = duplicate_zero = 0
    consistency_total = consistency_exact = 0
    consistency_errors: list[float] = []
    span_total = span_errors = 0
    variant_types: dict[str, Counter[str]] = defaultdict(Counter)
    anomalies: list[dict[str, Any]] = []

    for row in rows:
        results: dict[str, dict[str, Any]] = {}
        variants = action_variants(row, sid_pool)
        for variant, fixture in variants.items():
            completion = fixture["completion"]
            result = action_marginal_credit(completion, row)
            results[variant] = result
            if not result["valid"]:
                anomalies.append(anomaly("invalid_variant", row, variant, completion))
                continue
            for credit in result["credits"]:
                variant_types[variant][credit["credit_type"]] += 1
                start, end = credit["char_start"], credit["char_end"]
                span_total += 1
                if completion[start:end] != credit["sid"]:
                    span_errors += 1
                    anomalies.append(anomaly("action_span_error", row, variant, completion, credit))

                if credit["occurrence"] == 1 and credit["is_gold"]:
                    tp_total += 1
                    tp_positive += int(credit["delta"] > 0.0)
                    if credit["delta"] <= 0.0:
                        anomalies.append(anomaly("gold_sid_not_positive", row, variant, completion, credit))
                elif not credit["is_gold"]:
                    fp_total += 1
                    fp_negative += int(credit["delta"] < 0.0)
                    if credit["delta"] >= 0.0:
                        anomalies.append(anomaly("fp_sid_not_negative", row, variant, completion, credit))

            if variant == "duplicate_gold":
                target = fixture["target_sid"]
                duplicate_credits = [
                    credit
                    for credit in result["credits"]
                    if credit["sid"] == target and credit["occurrence"] > 1
                ]
                duplicate_total += len(duplicate_credits)
                for credit in duplicate_credits:
                    duplicate_zero += int(credit["delta"] == 0.0)
                    if credit["delta"] != 0.0:
                        anomalies.append(anomaly("duplicate_not_zero", row, variant, completion, credit))

        in_target = variants["history_in_fp"]["target_sid"]
        out_target = variants["history_out_fp"]["target_sid"]
        in_credit = next(credit for credit in results["history_in_fp"]["credits"] if credit["sid"] == in_target)
        out_credit = next(credit for credit in results["history_out_fp"]["credits"] if credit["sid"] == out_target)
        difference = abs(in_credit["delta"] - out_credit["delta"])
        consistency_total += 1
        consistency_errors.append(difference)
        is_consistent = difference <= CONSISTENCY_TOLERANCE and in_credit["credit_type"] == out_credit["credit_type"]
        consistency_exact += int(is_consistent)
        if not is_consistent:
            anomalies.append(
                anomaly(
                    "history_in_out_fp_inconsistent",
                    row,
                    "history_in_fp_vs_history_out_fp",
                    json.dumps(
                        {
                            "history_in": variants["history_in_fp"]["completion"],
                            "history_out": variants["history_out_fp"]["completion"],
                        },
                        ensure_ascii=False,
                    ),
                    {"history_in": in_credit, "history_out": out_credit},
                )
            )

    return {
        "sample_count": len(rows),
        "variant_count": len(rows) * 5,
        "tp": {"count": tp_total, "positive_count": tp_positive, "positive_rate": tp_positive / tp_total},
        "fp": {"count": fp_total, "negative_count": fp_negative, "negative_rate": fp_negative / fp_total},
        "duplicate": {
            "count": duplicate_total,
            "zero_count": duplicate_zero,
            "zero_rate": duplicate_zero / duplicate_total,
        },
        "history_in_out_fp_consistency": {
            "count": consistency_total,
            "consistent_count": consistency_exact,
            "rate": consistency_exact / consistency_total,
            "max_delta_difference": max(consistency_errors, default=0.0),
        },
        "span": {"count": span_total, "error_count": span_errors},
        "credit_types_by_variant": {
            name: dict(sorted(counts.items())) for name, counts in sorted(variant_types.items())
        },
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def audit_chain(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    credit_types: Counter[str] = Counter()
    variant_types: dict[str, Counter[str]] = defaultdict(Counter)
    delta_total: list[float] = []
    delta_action: list[float] = []
    delta_logic: list[float] = []
    consistency_errors: list[float] = []
    span_total = span_errors = 0
    invalid_variants = 0
    anomalies: list[dict[str, Any]] = []

    for row in rows:
        for variant, fixture in chain_variants(row).items():
            completion = fixture["completion"]
            completion_events = json.loads(completion)["logic_chain"]["events"]
            result = chain_marginal_credit(completion, row)
            if not result["valid"]:
                invalid_variants += 1
                anomalies.append(anomaly("invalid_variant", row, variant, completion))
                continue
            for credit in result["credits"]:
                credit_types[credit["credit_type"]] += 1
                variant_types[variant][credit["credit_type"]] += 1
                delta_total.append(credit["delta_total"])
                delta_action.append(credit["delta_action_alignment"])
                delta_logic.append(credit["delta_logic_alignment"])

                direct_delta = credit["full_reward"] - credit["reward_without_event"]
                error = abs(credit["delta_total"] - direct_delta)
                consistency_errors.append(error)
                if error > CONSISTENCY_TOLERANCE:
                    anomalies.append(anomaly("chain_reward_delta_mismatch", row, variant, completion, credit))

                start, end = credit["char_start"], credit["char_end"]
                span_total += 1
                try:
                    span_matches = (
                        json.loads(completion[start:end])
                        == completion_events[credit["event_index"]]
                    )
                except (json.JSONDecodeError, IndexError):
                    span_matches = False
                if not span_matches:
                    span_errors += 1
                    anomalies.append(anomaly("chain_span_error", row, variant, completion, credit))

    for kind in ("positive", "negative", "zero"):
        credit_types.setdefault(kind, 0)
    return {
        "sample_count": len(rows),
        "variant_count": len(rows) * 5,
        "credit_count": sum(credit_types.values()),
        "credit_types": dict(sorted(credit_types.items())),
        "credit_types_by_variant": {
            name: dict(sorted(counts.items())) for name, counts in sorted(variant_types.items())
        },
        "delta_total": distribution(delta_total),
        "delta_action": distribution(delta_action),
        "delta_logic": distribution(delta_logic),
        "reward_delta_consistency": {
            "tolerance": CONSISTENCY_TOLERANCE,
            "max_error": max(consistency_errors, default=0.0),
            "error_count": sum(error > CONSISTENCY_TOLERANCE for error in consistency_errors),
        },
        "span": {"count": span_total, "error_count": span_errors},
        "invalid_variant_count": invalid_variants,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
def render_markdown(result: Mapping[str, Any]) -> str:
    action = result["action"]
    chain = result["chain"]
    return "\n".join(
        [
            "# MC_USER_v1 Real-data Marginal Credit Audit",
            "",
            f"Status: **{result['status']}**",
            "",
            "## Scope",
            "",
            f"- Seed: `{result['selection']['seed']}`",
            f"- Frozen train SHA256: `{result['dataset']['sha256_before']}`",
            "- Selection: 100 Action + 100 Chain from train_3000.jsonl",
            "- Execution: CPU-only; no generation, training, checkpoint, or GR_USER_v1 changes",
            "- Credit source: leave-one-out calls to the existing Action/Chain scorers only",
            "",
            "## Action",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| TP positive rate | {action['tp']['positive_rate']:.6%} ({action['tp']['positive_count']}/{action['tp']['count']}) |",
            f"| FP negative rate | {action['fp']['negative_rate']:.6%} ({action['fp']['negative_count']}/{action['fp']['count']}) |",
            f"| Duplicate-zero rate | {action['duplicate']['zero_rate']:.6%} ({action['duplicate']['zero_count']}/{action['duplicate']['count']}) |",
            f"| History-in/out FP consistency | {action['history_in_out_fp_consistency']['rate']:.6%} |",
            f"| Span errors | {action['span']['error_count']} / {action['span']['count']} |",
            f"| Anomalies | {action['anomaly_count']} |",
            "",
            "## Chain",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Positive / negative / zero | {chain['credit_types']['positive']} / {chain['credit_types']['negative']} / {chain['credit_types']['zero']} |",
            f"| delta_total mean / min / max | {chain['delta_total']['mean']:.12f} / {chain['delta_total']['min']:.12f} / {chain['delta_total']['max']:.12f} |",
            f"| delta_action mean | {chain['delta_action']['mean']:.12f} |",
            f"| delta_logic mean | {chain['delta_logic']['mean']:.12f} |",
            f"| Reward-delta max error | {chain['reward_delta_consistency']['max_error']:.3e} |",
            f"| Span errors | {chain['span']['error_count']} / {chain['span']['count']} |",
            f"| Invalid variants | {chain['invalid_variant_count']} |",
            "",
            "## Interpretation",
            "",
            "Marginal credit is determined only by the existing evaluator-aligned reward change.",
            "Constraint and grounding violations are diagnostics and are not used to alter any credit.",
            "Complete anomaly records are retained in the JSON artifact when present.",
            "",
        ]
    )


def run_audit(train_path: Path) -> dict[str, Any]:
    before = sha256_file(train_path)
    if before != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(f"frozen train SHA mismatch: {before}")
    rows = read_jsonl(train_path)
    if len(rows) != 3000:
        raise RuntimeError(f"expected 3000 train rows, found {len(rows)}")

    selected_action = select_rows(rows, "action", 100)
    selected_chain = select_rows(rows, "chain", 100)
    sid_pool = sorted({sid for row in rows for sid in row["history_sids"]})
    action = audit_action(selected_action, sid_pool)
    chain = audit_chain(selected_chain)

    after = sha256_file(train_path)
    frozen_unchanged = before == after == EXPECTED_TRAIN_SHA256
    passed = (
        action["tp"]["positive_rate"] == 1.0
        and action["fp"]["negative_rate"] == 1.0
        and action["duplicate"]["zero_rate"] == 1.0
        and action["history_in_out_fp_consistency"]["rate"] == 1.0
        and action["span"]["error_count"] == 0
        and action["anomaly_count"] == 0
        and chain["reward_delta_consistency"]["max_error"] <= CONSISTENCY_TOLERANCE
        and chain["span"]["error_count"] == 0
        and chain["invalid_variant_count"] == 0
        and frozen_unchanged
    )
    return {
        "audit": "mc_user_v1_credit_audit",
        "status": "MC_CREDIT_AUDIT_PASS" if passed else "MC_CREDIT_AUDIT_FAIL",
        "execution": {
            "cpu_only": True,
            "training": False,
            "generation": False,
            "checkpoint_modified": False,
            "gr_user_v1_modified": False,
        },
        "dataset": {
            "path": str(train_path),
            "row_count": len(rows),
            "sha256_before": before,
            "sha256_after": after,
            "frozen_sha_unchanged": frozen_unchanged,
            "real_sid_pool_size": len(sid_pool),
        },
        "selection": {
            "seed": SEED,
            "method": "SHA256(seed, route, sample_id) stable ordering",
            "action_count": len(selected_action),
            "chain_count": len(selected_chain),
            "action_sample_ids": [row["sample_id"] for row in selected_action],
            "chain_sample_ids": [row["sample_id"] for row in selected_chain],
        },
        "action": action,
        "chain": chain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    result = run_audit(args.train)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "action": result["action"], "chain": result["chain"]}, ensure_ascii=False))
    if result["status"] != "MC_CREDIT_AUDIT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
