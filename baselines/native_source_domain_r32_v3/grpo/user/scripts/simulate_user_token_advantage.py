"""Offline token-advantage simulation over frozen GR_USER_v1 G=4 rollouts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


EPSILON = 1e-4
LAMBDAS = (0.10, 0.25, 0.50, 1.00)
STRATEGIES = ("fixed", "sqrt")
WHITELISTS = {
    "action": frozenset({"hallucinated_sid", "duplicate_sid"}),
    "chain": frozenset(
        {
            "hallucinated_sid",
            "date_mismatch",
            "action_mismatch",
            "duplicate_event",
            "chronology_violation",
            "excess_event",
        }
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def population_advantages(rewards: list[float], epsilon: float = EPSILON) -> tuple[list[float], float, float]:
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std == 0.0:
        return [0.0] * len(rewards), mean, std
    return [(reward - mean) / (std + epsilon) for reward in rewards], mean, std


def _effective_lambda(base_lambda: float, span_length: int, strategy: str) -> float:
    if span_length <= 0:
        raise ValueError("violation span must contain at least one token")
    if strategy == "fixed":
        return base_lambda
    if strategy == "sqrt":
        return base_lambda / math.sqrt(span_length)
    raise ValueError(f"unknown strategy: {strategy}")


def compile_effective_penalties(
    candidate: dict,
    strategy: str,
    kind_lambdas: dict[str, float],
) -> tuple[list[float], list[set[str]]]:
    """Return strongest per-token penalty and winning kinds without summation."""
    token_count = candidate["completion_token_count"]
    penalties = [0.0] * token_count
    winners = [set() for _ in range(token_count)]
    whitelist = WHITELISTS[candidate["route"]]
    for record in candidate["penalty"]["records"]:
        if not record["included"] or record["kind"] not in whitelist:
            continue
        indices = sorted(set(record["masked_token_indices"]))
        if not indices:
            raise ValueError("included violation has an empty token span")
        if indices[-1] >= token_count:
            raise ValueError("violation token index exceeds completion length")
        effective = _effective_lambda(kind_lambdas[record["kind"]], len(indices), strategy)
        for token_index in indices:
            if effective > penalties[token_index] + 1e-15:
                penalties[token_index] = effective
                winners[token_index] = {record["kind"]}
            elif math.isclose(effective, penalties[token_index], rel_tol=0.0, abs_tol=1e-15):
                winners[token_index].add(record["kind"])
    return penalties, winners


def simulate_token_advantages(
    sequence_advantage: float,
    penalties: list[float],
) -> list[float]:
    return [
        min(sequence_advantage, -penalty) if penalty > 0.0 else sequence_advantage
        for penalty in penalties
    ]


def _candidate_static_kind_stats(candidate: dict) -> dict[str, dict]:
    records_by_kind = collections.defaultdict(list)
    for record in candidate["penalty"]["records"]:
        if record["included"] and record["kind"] in WHITELISTS[candidate["route"]]:
            records_by_kind[record["kind"]].append(sorted(set(record["masked_token_indices"])))
    output = {}
    for kind, spans in records_by_kind.items():
        output[kind] = {
            "violation_count": len(spans),
            "span_lengths": [len(span) for span in spans],
            "masked_indices": sorted({index for span in spans for index in span}),
        }
    return output


def _assign_advantages(candidates: list[dict], groups: list[dict]) -> tuple[dict, dict]:
    candidates_by_sample = collections.defaultdict(dict)
    for candidate in candidates:
        candidates_by_sample[candidate["sample_id"]][candidate["candidate_index"]] = candidate
    advantage_by_key = {}
    consistency_errors = []
    zero_std = {"action": {"count": 0, "rescued": 0}, "chain": {"count": 0, "rescued": 0}}
    for group in groups:
        rewards = list(group["rewards"])
        advantages, mean, std = population_advantages(rewards)
        consistency_errors.append(abs(mean - group["reward_mean"]))
        consistency_errors.append(abs(std - group["reward_population_std"]))
        route = group["route"]
        group_candidates = candidates_by_sample[group["sample_id"]]
        if len(group_candidates) != 4:
            raise ValueError(f"group {group['sample_id']} does not have four candidates")
        if std == 0.0:
            zero_std[route]["count"] += 1
            if any(any(candidate["penalty"]["penalty_mask"]) for candidate in group_candidates.values()):
                zero_std[route]["rescued"] += 1
        for candidate_index, reward, advantage in zip(group["candidate_indices"], rewards, advantages):
            candidate = group_candidates[candidate_index]
            if not math.isclose(candidate["reward"], reward, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("candidate reward differs from its group reward")
            advantage_by_key[(group["sample_id"], candidate_index)] = advantage
    if len(advantage_by_key) != len(candidates):
        raise ValueError("not every candidate received one group advantage")
    return advantage_by_key, {
        "epsilon": EPSILON,
        "max_existing_group_stat_error": max(consistency_errors, default=0.0),
        "population_std_consistent": max(consistency_errors, default=0.0) <= 1e-12,
        "zero_std_groups": zero_std,
    }


def _impact_ratio(sequence_advantage: float, shift: float) -> float:
    if sequence_advantage == 0.0:
        return math.inf if shift > 0.0 else 0.0
    return shift / abs(sequence_advantage)


def simulate_route(
    candidates: list[dict],
    advantages: dict,
    route: str,
    strategy: str,
    base_lambda: float,
) -> tuple[dict, dict]:
    selected = [candidate for candidate in candidates if candidate["route"] == route]
    whitelist = WHITELISTS[route]
    kind_lambdas = {kind: base_lambda for kind in whitelist}
    total_tokens = 0
    masked_tokens = 0
    masked_candidates = 0
    positive_masked_tokens = 0
    flipped_positive_masked_tokens = 0
    task_abs = []
    masked_token_abs = []
    negative_mass = 0.0
    positive_mass = 0.0
    mean_abs_shifts = []
    impact_counts = {"gt_10pct": 0, "gt_25pct": 0, "gt_50pct": 0}
    positive_trajectory_flips = 0
    candidate_details = {}
    per_kind_static = collections.defaultdict(
        lambda: {"candidate_count": 0, "violation_count": 0, "masked_token_count": 0, "span_lengths": []}
    )
    per_kind_negative_mass = collections.Counter()
    total_penalty_negative_mass = 0.0

    for candidate in selected:
        key = (candidate["sample_id"], candidate["candidate_index"])
        sequence_advantage = advantages[key]
        penalties, winners = compile_effective_penalties(candidate, strategy, kind_lambdas)
        token_advantages = simulate_token_advantages(sequence_advantage, penalties)
        if len(token_advantages) != candidate["completion_token_count"]:
            raise AssertionError("token advantage length drift")
        for index, penalty in enumerate(penalties):
            if penalty == 0.0 and token_advantages[index] != sequence_advantage:
                raise AssertionError("an unmasked token differs from sequence advantage")

        token_count = len(token_advantages)
        masked_indices = [index for index, penalty in enumerate(penalties) if penalty > 0.0]
        total_tokens += token_count
        masked_tokens += len(masked_indices)
        masked_candidates += bool(masked_indices)
        task_abs.append(abs(sequence_advantage))
        masked_token_abs.extend(abs(token_advantages[index]) for index in masked_indices)
        negative_mass += sum(abs(value) for value in token_advantages if value < 0.0)
        positive_mass += sum(value for value in token_advantages if value > 0.0)
        positive_masked_tokens += sum(sequence_advantage > 0.0 for _ in masked_indices)
        flipped_positive_masked_tokens += sum(
            sequence_advantage > 0.0 and token_advantages[index] < 0.0 for index in masked_indices
        )

        token_mean = statistics.fmean(token_advantages) if token_advantages else sequence_advantage
        shift = abs(token_mean - sequence_advantage)
        mean_abs_shifts.append(shift)
        impact = _impact_ratio(sequence_advantage, shift)
        for threshold, label in ((0.10, "gt_10pct"), (0.25, "gt_25pct"), (0.50, "gt_50pct")):
            impact_counts[label] += impact > threshold
        positive_trajectory_flips += sequence_advantage > 0.0 and token_mean < 0.0

        baseline_negative = max(-sequence_advantage, 0.0)
        for index in masked_indices:
            incremental_mass = max(-token_advantages[index], 0.0) - baseline_negative
            if incremental_mass < -1e-12:
                raise AssertionError("penalty reduced negative mass")
            incremental_mass = max(incremental_mass, 0.0)
            total_penalty_negative_mass += incremental_mass
            if winners[index]:
                share = incremental_mass / len(winners[index])
                for kind in winners[index]:
                    per_kind_negative_mass[kind] += share

        static = _candidate_static_kind_stats(candidate)
        for kind, values in static.items():
            per_kind_static[kind]["candidate_count"] += 1
            per_kind_static[kind]["violation_count"] += values["violation_count"]
            per_kind_static[kind]["masked_token_count"] += len(values["masked_indices"])
            per_kind_static[kind]["span_lengths"].extend(values["span_lengths"])

        span_details = {}
        for record in candidate["penalty"]["records"]:
            if not record["included"] or record["kind"] not in whitelist:
                continue
            indices = sorted(set(record["masked_token_indices"]))
            values = [token_advantages[index] for index in indices]
            span_details.setdefault(record["kind"], []).append(
                {
                    "token_count": len(indices),
                    "effective_lambda": _effective_lambda(base_lambda, len(indices), strategy),
                    "token_advantage_mean": statistics.fmean(values),
                    "token_advantage_min": min(values),
                    "token_advantage_max": max(values),
                }
            )
        candidate_details[key] = {
            "sequence_advantage": sequence_advantage,
            "token_advantage_mean": token_mean,
            "token_advantage_min": min(token_advantages) if token_advantages else sequence_advantage,
            "overridden_token_count": len(masked_indices),
            "overridden_token_fraction": len(masked_indices) / token_count if token_count else 0.0,
            "absolute_mean_shift": shift,
            "relative_mean_shift": impact,
            "span_details": span_details,
        }

    per_kind = {}
    for kind in sorted(whitelist):
        values = per_kind_static[kind]
        mass = per_kind_negative_mass[kind]
        per_kind[kind] = {
            "candidate_count": values["candidate_count"],
            "violation_count": values["violation_count"],
            "masked_token_count": values["masked_token_count"],
            "average_span_length": (
                statistics.fmean(values["span_lengths"]) if values["span_lengths"] else 0.0
            ),
            "penalty_negative_mass": mass,
            "penalty_negative_mass_share": (
                mass / total_penalty_negative_mass if total_penalty_negative_mass else 0.0
            ),
        }

    count = len(selected)
    output = {
        "candidate_count": count,
        "masked_candidate_rate": masked_candidates / count,
        "masked_token_rate": masked_tokens / total_tokens,
        "positive_masked_token_flip_rate": (
            flipped_positive_masked_tokens / positive_masked_tokens if positive_masked_tokens else 0.0
        ),
        "average_abs_task_advantage": statistics.fmean(task_abs),
        "average_masked_abs_token_advantage": (
            statistics.fmean(masked_token_abs) if masked_token_abs else 0.0
        ),
        "negative_token_mass": negative_mass,
        "positive_token_mass": positive_mass,
        "negative_positive_mass_ratio": negative_mass / positive_mass if positive_mass else math.inf,
        "average_absolute_token_advantage_mean_shift": statistics.fmean(mean_abs_shifts),
        "completion_impact": {
            **{label: value for label, value in impact_counts.items()},
            **{f"{label}_rate": value / count for label, value in impact_counts.items()},
            "positive_sequence_to_negative_mean_count": positive_trajectory_flips,
            "positive_sequence_to_negative_mean_rate": positive_trajectory_flips / count,
        },
        "total_penalty_negative_mass": total_penalty_negative_mass,
        "per_kind": per_kind,
    }
    return output, candidate_details


def _representative_candidates(run_summary: dict) -> dict[str, tuple[str, int]]:
    examples = run_summary["representative_examples"]
    mapping = {}
    for label, source in (
        ("high_f1_hallucination", "action_high_reward_hallucination"),
        ("high_f1_duplicate", "action_high_reward_duplicate"),
        ("high_chain_reward_mismatch", "chain_high_reward_date_or_action_mismatch"),
    ):
        if examples[source]:
            item = examples[source][0]
            mapping[label] = (item["sample_id"], item["candidate_index"])
    return mapping


def _markdown(summary: dict) -> str:
    lines = [
        "# GR_USER_v1 token advantage offline simulation",
        "",
        f"Source run: `{summary['source_run_id']}`",
        "",
        "No generation, GPU, model loading, training, or Trainer integration was used.",
        "",
        "## Formula",
        "",
        "`A_i = (R_i - group_mean) / (group_population_std + 1e-4)`. Zero-std groups use `A_i=0`.",
        "Unmasked tokens retain `A_i`; masked tokens use `min(A_i, -lambda_eff)`.",
        "Fixed uses `lambda_eff=lambda`; sqrt uses `lambda_eff=lambda/sqrt(span_tokens)`.",
        "Overlaps use the strongest effective lambda and are never summed.",
        "Positive/negative mass is `sum(abs(token_advantage))` over tokens of that sign.",
        "Penalty negative mass is the incremental negative mass beyond the original task advantage.",
        "",
        "## Strategy comparison",
        "",
        "| Strategy | Lambda | Action neg/pos | Chain neg/pos | Action mean shift | Chain mean shift | duplicate_event share | Zero-std rescued |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rescued = sum(
        values["rescued"] for values in summary["normalization_audit"]["zero_std_groups"].values()
    )
    for item in summary["schemes"]:
        action = item["routes"]["action"]
        chain = item["routes"]["chain"]
        duplicate_share = chain["per_kind"]["duplicate_event"]["penalty_negative_mass_share"]
        lines.append(
            f"| {item['strategy']} | {item['lambda']:.2f} | "
            f"{action['negative_positive_mass_ratio']:.4f} | {chain['negative_positive_mass_ratio']:.4f} | "
            f"{action['average_absolute_token_advantage_mean_shift']:.6f} | "
            f"{chain['average_absolute_token_advantage_mean_shift']:.6f} | "
            f"{duplicate_share:.2%} | {rescued} |"
        )
    lines.extend(
        [
            "",
            "## Route metrics",
            "",
            "| Strategy | Lambda | Route | Masked candidates | Masked tokens | Positive masked flips | Avg abs task A | Avg masked abs token A | Negative mass | Positive mass |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["schemes"]:
        for route in ("action", "chain"):
            values = item["routes"][route]
            lines.append(
                f"| {item['strategy']} | {item['lambda']:.2f} | {route} | "
                f"{values['masked_candidate_rate']:.2%} | {values['masked_token_rate']:.2%} | "
                f"{values['positive_masked_token_flip_rate']:.2%} | "
                f"{values['average_abs_task_advantage']:.6f} | "
                f"{values['average_masked_abs_token_advantage']:.6f} | "
                f"{values['negative_token_mass']:.3f} | {values['positive_token_mass']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Completion-level impact",
            "",
            "Relative shift is `abs(mean_token_A - sequence_A) / abs(sequence_A)`. "
            "For zero sequence advantage, any nonzero shift is treated as infinite.",
            "",
            "| Strategy | Lambda | Action >10% / >25% / >50% | Chain >10% / >25% / >50% | Action positive-to-negative | Chain positive-to-negative |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["schemes"]:
        action = item["routes"]["action"]["completion_impact"]
        chain = item["routes"]["chain"]["completion_impact"]
        lines.append(
            f"| {item['strategy']} | {item['lambda']:.2f} | "
            f"{action['gt_10pct_rate']:.2%} / {action['gt_25pct_rate']:.2%} / {action['gt_50pct_rate']:.2%} | "
            f"{chain['gt_10pct_rate']:.2%} / {chain['gt_25pct_rate']:.2%} / {chain['gt_50pct_rate']:.2%} | "
            f"{action['positive_sequence_to_negative_mean_rate']:.2%} | "
            f"{chain['positive_sequence_to_negative_mean_rate']:.2%} |"
        )
    recommended = next(
        item for item in summary["schemes"] if item["strategy"] == "sqrt" and item["lambda"] == 0.50
    )
    lines.extend(
        [
            "",
            "## Violation kinds at recommended setting",
            "",
            "| Route | Kind | Candidates | Violations | Masked tokens | Avg span | Penalty negative mass | Share |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for route in ("action", "chain"):
        for kind, values in recommended["routes"][route]["per_kind"].items():
            lines.append(
                f"| {route} | `{kind}` | {values['candidate_count']} | {values['violation_count']} | "
                f"{values['masked_token_count']} | {values['average_span_length']:.2f} | "
                f"{values['penalty_negative_mass']:.3f} | {values['penalty_negative_mass_share']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## High-reward violation examples",
            "",
            "| Strategy | Lambda | Example | Sequence A | Span tokens / token A | Completion mean token A |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for item in summary["schemes"]:
        for label, example in item["representative_samples"].items():
            spans = []
            for kind, values in example["span_details"].items():
                for value in values:
                    spans.append(
                        f"{kind}:{value['token_count']}@{value['token_advantage_mean']:.4f}"
                    )
            lines.append(
                f"| {item['strategy']} | {item['lambda']:.2f} | `{label}` | "
                f"{example['sequence_advantage']:.6f} | {'; '.join(spans)} | "
                f"{example['token_advantage_mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Zero-std groups",
            "",
            f"- Action: {summary['normalization_audit']['zero_std_groups']['action']['count']} total, "
            f"{summary['normalization_audit']['zero_std_groups']['action']['rescued']} with constraint signal",
            f"- Chain: {summary['normalization_audit']['zero_std_groups']['chain']['count']} total, "
            f"{summary['normalization_audit']['zero_std_groups']['chain']['rescued']} with constraint signal",
            "",
            "## Span audit",
            "",
            "The real rollout contains no 1-token Action hallucination. Observed Action hallucination spans are 2-3 tokens.",
            "Action duplicate spans are 4 tokens. Chain duplicate_event spans average 79.75 tokens (max 86).",
            "",
            "## Recommendation",
            "",
            f"Use **{summary['recommendation']['strategy']}** with initial lambda "
            f"**{summary['recommendation']['lambda']:.2f}**.",
            "Sqrt normalization keeps short SID corrections meaningful while preventing roughly 80-token duplicate events "
            "from receiving linear per-token amplification. At lambda 0.50 it keeps duplicate_event at about 5.07% "
            "of Chain penalty negative mass, with 1.5% of Chain completions shifting by more than 25% and 0.5% "
            "shifting by more than 50%.",
            "",
            "## Integrity",
            "",
            f"- Input SHA unchanged: {summary['integrity']['input_sha_unchanged']}",
            f"- Existing group population statistics consistent: {summary['normalization_audit']['population_std_consistent']}",
            f"- Candidates/groups: {summary['input_counts']['candidates']}/{summary['input_counts']['groups']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--docs-output", type=Path, required=True)
    args = parser.parse_args()

    candidate_path = args.run_dir / "candidates.jsonl"
    group_path = args.run_dir / "groups.jsonl"
    run_summary_path = args.run_dir / "summary.json"
    input_paths = (candidate_path, group_path, run_summary_path)
    sha_before = {path.name: sha256_file(path) for path in input_paths}
    candidates = read_jsonl(candidate_path)
    groups = read_jsonl(group_path)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if len(candidates) != 400 or len(groups) != 100:
        raise ValueError("Phase 3B source must contain exactly 400 candidates and 100 groups")
    if run_summary["run_id"] != "GR-USER-G4-AUDIT-4GPU-20260819-181843":
        raise ValueError("unexpected Phase 3B source run")

    advantages, normalization_audit = _assign_advantages(candidates, groups)
    representatives = _representative_candidates(run_summary)
    schemes = []
    for strategy in STRATEGIES:
        for base_lambda in LAMBDAS:
            routes = {}
            detail_by_route = {}
            for route in ("action", "chain"):
                routes[route], detail_by_route[route] = simulate_route(
                    candidates, advantages, route, strategy, base_lambda
                )
            representative_output = {}
            for label, key in representatives.items():
                candidate = next(
                    item
                    for item in candidates
                    if item["sample_id"] == key[0] and item["candidate_index"] == key[1]
                )
                representative_output[label] = {
                    "sample_id": key[0],
                    "candidate_index": key[1],
                    "route": candidate["route"],
                    "reward": candidate["reward"],
                    **detail_by_route[candidate["route"]][key],
                }
            schemes.append(
                {
                    "strategy": strategy,
                    "lambda": base_lambda,
                    "routes": routes,
                    "representative_samples": representative_output,
                }
            )

    sha_after = {path.name: sha256_file(path) for path in input_paths}
    summary = {
        "contract_version": "gr_user_token_advantage_sim_v1",
        "source_run_id": run_summary["run_id"],
        "input_counts": {"candidates": len(candidates), "groups": len(groups)},
        "formula": {
            "epsilon": EPSILON,
            "task_advantage": "(reward - group_mean) / (group_population_std + epsilon)",
            "zero_std_task_advantage": 0.0,
            "masked_token_advantage": "min(task_advantage, -lambda_eff)",
            "fixed_lambda_eff": "lambda",
            "sqrt_lambda_eff": "lambda / sqrt(violation_span_token_count)",
            "overlap": "maximum effective lambda; never sum",
            "mass": "sum(abs(token_advantage)) by sign",
            "penalty_negative_mass": "incremental negative mass beyond the unpenalized task advantage",
        },
        "normalization_audit": normalization_audit,
        "schemes": schemes,
        "recommendation": {
            "strategy": "sqrt",
            "lambda": 0.50,
            "basis": (
                "Balances short-span correction strength with bounded long-event mass; "
                "duplicate_event contributes about 5% of Chain penalty negative mass."
            ),
        },
        "integrity": {
            "input_sha_before": sha_before,
            "input_sha_after": sha_after,
            "input_sha_unchanged": sha_before == sha_after,
            "gpu_used": False,
            "generation_used": False,
            "training_used": False,
            "model_loaded": False,
        },
    }
    if not summary["integrity"]["input_sha_unchanged"]:
        raise RuntimeError("Phase 3B rollout files changed during simulation")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.docs_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.docs_output.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": str(args.summary_output), "docs": str(args.docs_output)}))


if __name__ == "__main__":
    main()
