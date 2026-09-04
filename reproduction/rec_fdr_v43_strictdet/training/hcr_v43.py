#!/usr/bin/env python3
"""History-conditioned hierarchical ranking helpers for V4.3.

The helpers operate only on teacher-forced A/B/C logits already produced by
the V4.2 recommendation objective.  They never run an additional Transformer
forward and never impose a generation-time catalog mask.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


Sid = tuple[int, int, int]
LEVELS = ("a", "b", "c")
NOVELTY_TYPES = ("T0", "T1", "T2", "T3")
DOMAINS = ("video", "prod", "ad", "living")
HIT_KS = (1, 3, 5, 10, 16, 32)


def _metric_schema() -> tuple[str, ...]:
    names = {
        "hcr_aux_total",
        "hcr_off_loss_delta",
        "hcr_topk_total",
        "history_fdr_total",
        "multi_a_loss",
        "weighted_multi_a",
        "selected_history_competitor_behavior_support",
        "selected_history_competitor_raw_frequency",
        "video_multi_a_active",
        "video_multi_a_eligible",
        "video_multi_a_active_instances_per_pack",
        "video_multi_a_unique_eligible_groups_per_pack",
        "video_multi_a_repeat_factor",
        "video_positive_a_count",
    }
    for level in LEVELS:
        upper = level.upper()
        names.update(
            {
                f"hcr_topk_{level}",
                f"hcr_top16_{level}",
                f"hcr_top32_{level}",
                f"top16_active_{level}",
                f"top32_active_{level}",
                f"history_fdr_{level}",
                f"history_pair_count_{level}",
                f"history_competitor_outrank_rate_{level}",
                f"behavior_evidence_pos_{upper}",
                f"behavior_evidence_neg_{upper}",
                f"behavior_margin_adjustment_{level}",
                f"catalog_rank_mean_{level}",
                f"catalog_rank_pack_median_{level}",
            }
        )
        names.update(f"teacher_forced_{level}_hit{k}" for k in HIT_KS)
    names.update(f"{novelty}_count" for novelty in NOVELTY_TYPES)
    names.update(f"{domain}_{novelty}_count" for domain in DOMAINS for novelty in NOVELTY_TYPES)
    names.update(f"frag_{bucket}_count" for bucket in ("le40", "41_80", "81_120", "gt120"))
    return tuple(sorted(names))


HCR_METRIC_NAMES = _metric_schema()


def parse_sid(value: str) -> Sid:
    """Parse one canonical SID without accepting partial or cross-domain text."""
    import re

    match = re.fullmatch(
        r"<\|(?:video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>",
        value,
    )
    if match is None:
        raise ValueError(f"invalid canonical SID: {value!r}")
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def novelty_type(anchor: Sid, history: Iterable[Sid]) -> str:
    history_set = set(history)
    if anchor[0] not in {item[0] for item in history_set}:
        return "T0"
    if anchor[:2] not in {item[:2] for item in history_set}:
        return "T1"
    if anchor not in history_set:
        return "T2"
    return "T3"


def safe_negative_indices(
    level: str,
    anchor: Sid,
    positives: Iterable[Sid],
    train_inventory: Iterable[Sid],
) -> list[int]:
    """Return observed train-only local competitors, excluding every positive."""
    positive_set = set(positives)
    if level == "a":
        blocked = {item[0] for item in positive_set}
        values = {item[0] for item in train_inventory if item[0] not in blocked}
    elif level == "b":
        blocked = {item[1] for item in positive_set if item[0] == anchor[0]}
        values = {
            item[1]
            for item in train_inventory
            if item[0] == anchor[0] and item[1] not in blocked
        }
    elif level == "c":
        blocked = {item[2] for item in positive_set if item[:2] == anchor[:2]}
        values = {
            item[2]
            for item in train_inventory
            if item[:2] == anchor[:2] and item[2] not in blocked
        }
    else:
        raise ValueError(f"unknown hierarchy level: {level}")
    return sorted(values)


def positive_indices(level: str, anchor: Sid, positives: Iterable[Sid]) -> list[int]:
    values = set(positives)
    if level == "a":
        result = {item[0] for item in values}
    elif level == "b":
        result = {item[1] for item in values if item[0] == anchor[0]}
    elif level == "c":
        result = {item[2] for item in values if item[:2] == anchor[:2]}
    else:
        raise ValueError(f"unknown hierarchy level: {level}")
    if not result:
        raise ValueError(f"empty positive set at hierarchy level {level}")
    return sorted(result)


def topk_boundary_loss(
    logits: torch.Tensor,
    positives: list[int],
    negatives: list[int],
    k: int,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, bool]:
    """Teacher-forced positive-vs-Kth-safe-negative boundary loss."""
    zero = logits.sum() * 0.0
    if len(negatives) < k:
        return zero, False
    negative_index = torch.tensor(negatives, device=logits.device, dtype=torch.long)
    positive_index = torch.tensor(positives, device=logits.device, dtype=torch.long)
    boundary = torch.topk(logits.index_select(0, negative_index), k=k).values[-1]
    terms = torch.nn.functional.softplus(
        (boundary + float(margin) - logits.index_select(0, positive_index)) / float(temperature)
    )
    return terms.mean(), True


def _level_value(sid: Sid, level: str) -> str:
    if level == "a":
        return str(sid[0])
    if level == "b":
        return f"{sid[0]},{sid[1]}"
    if level == "c":
        return f"{sid[0]},{sid[1]},{sid[2]}"
    raise ValueError(level)


def _token_value(sid: Sid, level: str) -> int:
    return sid[{"a": 0, "b": 1, "c": 2}[level]]


def _zero_metrics(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {name: zero for name in HCR_METRIC_NAMES}


@dataclass(frozen=True)
class GroupMetadata:
    group_index: int
    target_domain: str
    history: tuple[Sid, ...]
    target_history_n_sa: int
    level_counts: Mapping[str, Mapping[str, int]]
    behavior_level_counts: Mapping[str, Mapping[str, Mapping[str, int]]]


@dataclass(frozen=True)
class HCRMetadata:
    groups: Mapping[int, GroupMetadata]
    behavior_reliability: Mapping[str, Mapping[str, Mapping[str, float]]]

    @classmethod
    def load(cls, path: Path) -> "HCRMetadata":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if int(raw.get("schema_version", 0)) != 1:
            raise ValueError("hcr_group_metadata.json must use schema_version=1")
        groups: dict[int, GroupMetadata] = {}
        records = raw.get("history_records")
        if not isinstance(records, dict):
            raise ValueError("hcr_group_metadata.json is missing compact history_records")
        for key, record_key in raw["groups"].items():
            group_index = int(key)
            item = records[str(record_key)]
            level_counts = item["target_history_level_counts"]
            history = tuple(
                tuple(int(value) for value in sid_key.split(","))
                for sid_key in level_counts["c"]
            )
            groups[group_index] = GroupMetadata(
                group_index=group_index,
                target_domain=str(item["target_domain"]),
                history=history,  # type: ignore[arg-type]
                target_history_n_sa=int(item["target_history_n_sa"]),
                level_counts=level_counts,
                behavior_level_counts=item["behavior_level_counts"],
            )
        return cls(groups=groups, behavior_reliability=raw["behavior_reliability"])


def _candidate_evidence(
    metadata: GroupMetadata,
    reliability: Mapping[str, Mapping[str, Mapping[str, float]]],
    level: str,
    key: str,
) -> float:
    total = 0.0
    domain_weights = reliability.get(metadata.target_domain, {})
    for behavior, level_values in metadata.behavior_level_counts.get(level, {}).items():
        count = int(level_values.get(key, 0))
        weight = float(domain_weights.get(behavior, {}).get(level, 0.0))
        total += weight * math.log1p(count)
    return total


def _normalized(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lower = min(values.values())
    upper = max(values.values())
    if upper - lower <= 1e-8:
        return {key: 0.5 for key in values}
    return {key: (value - lower) / (upper - lower) for key, value in values.items()}


def _history_candidates(
    level: str,
    anchor: Sid,
    positives: list[Sid],
    metadata: GroupMetadata,
) -> dict[str, tuple[int, int]]:
    """Map hierarchy key to (local token index, raw history frequency)."""
    blocked = {_level_value(item, level) for item in positives if level == "a" or (level == "b" and item[0] == anchor[0]) or (level == "c" and item[:2] == anchor[:2])}
    values: dict[str, tuple[int, int]] = {}
    for key, count in metadata.level_counts[level].items():
        parts = tuple(int(value) for value in key.split(","))
        if key in blocked:
            continue
        if level == "a":
            token = parts[0]
        elif level == "b" and parts[:1] == anchor[:1]:
            token = parts[1]
        elif level == "c" and parts[:2] == anchor[:2]:
            token = parts[2]
        else:
            continue
        values[key] = (token, int(count))
    return values


def history_fdr_loss(
    stage_logits: Mapping[str, torch.Tensor],
    anchor: Sid,
    positives: list[Sid],
    metadata: GroupMetadata,
    reliability: Mapping[str, Mapping[str, Mapping[str, float]]],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = next(iter(stage_logits.values())).sum() * 0.0
    metrics: dict[str, torch.Tensor] = {}
    losses: list[torch.Tensor] = []
    selected_behavior_support: list[float] = []
    selected_raw_frequency: list[float] = []
    max_pairs = int(config.get("max_history_hard_pairs_per_level", 8))
    hard_fraction = float(config.get("history_hard_fraction", 0.5))
    eta_f = float(config.get("eta_frequency", 0.35))
    eta_e = float(config.get("eta_evidence", 0.45))
    eta_z = float(config.get("eta_logit", 0.20))
    for level in LEVELS:
        candidates = _history_candidates(level, anchor, positives, metadata)
        if not candidates:
            for name in (
                f"history_fdr_{level}", f"history_pair_count_{level}",
                f"history_competitor_outrank_rate_{level}",
                f"behavior_evidence_pos_{level.upper()}", f"behavior_evidence_neg_{level.upper()}",
                f"behavior_margin_adjustment_{level}",
            ):
                metrics[name] = zero
            continue
        frequency = {key: math.log1p(value[1]) for key, value in candidates.items()}
        evidence = {
            key: _candidate_evidence(metadata, reliability, level, key)
            for key in candidates
        }
        model_values = {
            key: float(stage_logits[level][token].detach().item())
            for key, (token, _count) in candidates.items()
        }
        norm_frequency = _normalized(frequency)
        norm_evidence = _normalized(evidence)
        norm_model = _normalized(model_values)
        scored = []
        for key, (token, _count) in candidates.items():
            scored.append((eta_f * norm_frequency[key] + eta_e * norm_evidence[key] + eta_z * norm_model[key], key))
        scored.sort(reverse=True)
        total = min(max_pairs, len(scored))
        hard_count = min(total, max(1, int(math.ceil(total * hard_fraction))))
        selected = scored[:hard_count]
        remaining = scored[hard_count:]
        rng = random.Random(seed + {"a": 101, "b": 211, "c": 307}[level])
        rng.shuffle(remaining)
        selected.extend(remaining[: total - hard_count])

        positive_key = _level_value(anchor, level)
        positive_evidence = _candidate_evidence(metadata, reliability, level, positive_key)
        selected_evidence = {key: evidence[key] for _score, key in selected}
        selected_evidence[positive_key] = positive_evidence
        normalized_selected_evidence = _normalized(selected_evidence)
        positive_hat = normalized_selected_evidence[positive_key]
        pos_token = _token_value(anchor, level)
        terms = []
        adjustments = []
        negative_evidence = []
        outrank = []
        for _score, key in selected:
            neg_token = candidates[key][0]
            neg_hat = normalized_selected_evidence[key]
            adjustment = float(config.get("gamma_pos", 0.05)) * positive_hat - float(config.get("gamma_neg", 0.05)) * neg_hat
            margin = max(
                float(config.get("margin_min", 0.02)),
                min(float(config.get("margin_max", 0.20)), float(config.get("margin", 0.10)) + adjustment),
            )
            gap = stage_logits[level][pos_token] - stage_logits[level][neg_token]
            terms.append(torch.nn.functional.softplus((margin - gap) / float(config.get("temperature", 1.0))))
            adjustments.append(adjustment)
            negative_evidence.append(evidence[key])
            selected_behavior_support.append(evidence[key])
            selected_raw_frequency.append(float(candidates[key][1]))
            outrank.append((gap < 0).float())
        local = torch.stack(terms).mean()
        losses.append(local)
        metrics[f"history_fdr_{level}"] = local
        metrics[f"history_pair_count_{level}"] = torch.tensor(float(len(terms)), device=local.device)
        metrics[f"history_competitor_outrank_rate_{level}"] = torch.stack(outrank).mean()
        metrics[f"behavior_evidence_pos_{level.upper()}"] = torch.tensor(positive_evidence, device=local.device)
        metrics[f"behavior_evidence_neg_{level.upper()}"] = torch.tensor(sum(negative_evidence) / len(negative_evidence), device=local.device)
        metrics[f"behavior_margin_adjustment_{level}"] = torch.tensor(sum(adjustments) / len(adjustments), device=local.device)
    total_loss = torch.stack(losses).mean() if losses else zero
    metrics["history_fdr_total"] = total_loss
    metrics["selected_history_competitor_behavior_support"] = (
        torch.tensor(
            sum(selected_behavior_support) / len(selected_behavior_support),
            device=zero.device,
            dtype=zero.dtype,
        )
        if selected_behavior_support else zero
    )
    metrics["selected_history_competitor_raw_frequency"] = (
        torch.tensor(
            sum(selected_raw_frequency) / len(selected_raw_frequency),
            device=zero.device,
            dtype=zero.dtype,
        )
        if selected_raw_frequency else zero
    )
    return total_loss, metrics


def compute_hcr_loss(
    stage_logits: Mapping[str, torch.Tensor],
    anchor: Sid,
    positives: list[Sid],
    train_inventory: list[Sid],
    metadata: GroupMetadata,
    behavior_reliability: Mapping[str, Mapping[str, Mapping[str, float]]],
    config: Mapping[str, Any],
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute enabled V4.3 auxiliaries and dense, reduction-safe metrics."""
    reference = next(iter(stage_logits.values()))
    metrics = _zero_metrics(reference)
    zero = reference.sum() * 0.0
    if not bool(config.get("enabled", False)):
        return zero, metrics
    observed_novelty = novelty_type(anchor, metadata.history)
    metrics[f"{observed_novelty}_count"] = torch.ones((), device=reference.device)
    metrics[f"{metadata.target_domain}_{observed_novelty}_count"] = torch.ones((), device=reference.device)
    frag = "le40" if metadata.target_history_n_sa <= 40 else "41_80" if metadata.target_history_n_sa <= 80 else "81_120" if metadata.target_history_n_sa <= 120 else "gt120"
    metrics[f"frag_{frag}_count"] = torch.ones((), device=reference.device)

    topk_cfg = config.get("novelty_topk", {})
    topk_total = zero
    if bool(topk_cfg.get("enabled", False)):
        matrix = topk_cfg["novelty_weights"]
        level_losses = []
        for level in LEVELS:
            # Novelty is defined for this row's anchor. Other same-history
            # positives are protected from negatives but are not pushed with
            # the anchor's novelty weights; their own rows supervise them.
            positives_at_level = [_token_value(anchor, level)]
            negatives = safe_negative_indices(level, anchor, positives, train_inventory)
            local_parts = []
            active_weight = 0.0
            for k, alpha in ((16, float(topk_cfg.get("alpha_16", 0.35))), (32, float(topk_cfg.get("alpha_32", 0.65)))):
                value, active = topk_boundary_loss(
                    stage_logits[level], positives_at_level, negatives, k,
                    float(topk_cfg.get("margin", {}).get(level, 0.10)),
                    float(topk_cfg.get("temperature", {}).get(level, 1.0)),
                )
                metrics[f"hcr_top{k}_{level}"] = value
                metrics[f"top{k}_active_{level}"] = torch.tensor(float(active), device=reference.device)
                if active:
                    local_parts.append(alpha * value)
                    active_weight += alpha
            local = sum(local_parts) / active_weight if local_parts else zero
            metrics[f"hcr_topk_{level}"] = local
            level_losses.append(float(matrix[observed_novelty][level]) * local)

            truth_score = stage_logits[level][_token_value(anchor, level)]
            if negatives:
                neg_index = torch.tensor(negatives, device=reference.device, dtype=torch.long)
                rank = 1.0 + (stage_logits[level].index_select(0, neg_index) > truth_score).float().sum()
            else:
                rank = torch.ones((), device=reference.device)
            metrics[f"catalog_rank_mean_{level}"] = rank
            metrics[f"catalog_rank_pack_median_{level}"] = rank
            for k in HIT_KS:
                metrics[f"teacher_forced_{level}_hit{k}"] = (rank <= k).float()
        topk_total = sum(level_losses)
        metrics["hcr_topk_total"] = topk_total

    history_cfg = config.get("history_fdr", {})
    history_total = zero
    if bool(history_cfg.get("enabled", False)):
        history_total, history_metrics = history_fdr_loss(
            stage_logits, anchor, positives, metadata, behavior_reliability, history_cfg, seed
        )
        metrics.update(history_metrics)

    multi_cfg = config.get("multi_a", {})
    multi_total = zero
    positive_a = positive_indices("a", anchor, positives)
    metrics["video_positive_a_count"] = torch.tensor(float(len(positive_a)), device=reference.device)
    multi_eligible = (
        bool(multi_cfg.get("enabled", False))
        and metadata.target_domain == "video"
        and len(positive_a) >= 2
    )
    metrics["video_multi_a_eligible"] = torch.tensor(
        float(multi_eligible), device=reference.device
    )
    if multi_eligible:
        negatives = safe_negative_indices("a", anchor, positives, train_inventory)
        if len(negatives) >= 32:
            positive_index = torch.tensor(positive_a, device=reference.device, dtype=torch.long)
            negative_index = torch.tensor(negatives, device=reference.device, dtype=torch.long)
            second_positive = torch.topk(stage_logits["a"].index_select(0, positive_index), k=2).values[-1]
            boundary = torch.topk(stage_logits["a"].index_select(0, negative_index), k=32).values[-1]
            multi_total = torch.nn.functional.softplus(
                (boundary + float(multi_cfg.get("margin", 0.10)) - second_positive)
                / float(multi_cfg.get("temperature", 1.0))
            )
            metrics["multi_a_loss"] = multi_total
            metrics["video_multi_a_active"] = torch.ones((), device=reference.device)

    metrics["weighted_multi_a"] = float(multi_cfg.get("lambda", 0.05)) * multi_total

    auxiliary = (
        float(topk_cfg.get("lambda", 0.10)) * topk_total
        + float(history_cfg.get("lambda", 0.05)) * history_total
        + float(multi_cfg.get("lambda", 0.05)) * multi_total
    )
    metrics["hcr_aux_total"] = auxiliary
    return auxiliary, metrics


def validate_hcr_config(config: Mapping[str, Any]) -> None:
    if not bool(config.get("enabled", False)):
        return
    topk = config.get("novelty_topk", {})
    if bool(topk.get("enabled", False)):
        if topk.get("positive_mode") != "anchor_only":
            raise ValueError("S1 novelty_topk.positive_mode must be anchor_only")
        if topk.get("known_positive_negative_policy") != "exclude_entire_group":
            raise ValueError("all group positives must be excluded from S1 negatives")
        matrix = topk.get("novelty_weights", {})
        if set(matrix) != set(NOVELTY_TYPES):
            raise ValueError("novelty_weights must define exactly T0/T1/T2/T3")
        for novelty, weights in matrix.items():
            if set(weights) != set(LEVELS):
                raise ValueError(f"{novelty} weights must define a/b/c")
            if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-8:
                raise ValueError(f"{novelty} novelty weights do not sum to 1")
    history = config.get("history_fdr", {})
    if bool(history.get("enabled", False)):
        for flag in ("behavior_conditioned", "hierarchy_specific", "plausible_negative_protection"):
            if not bool(history.get(flag, False)):
                raise ValueError(f"S2 safety flag must be true: {flag}")


def validate_hcr_stage_contract(stage: str, config: Mapping[str, Any]) -> None:
    """Reject inherited or accidentally combined HCR modules for formal stages."""
    expected = {
        "s1": (True, False, False),
        "s2": (True, True, False),
        "s3": (True, False, True),
    }
    if stage not in expected:
        return
    actual = (
        bool(config.get("novelty_topk", {}).get("enabled", False)),
        bool(config.get("history_fdr", {}).get("enabled", False)),
        bool(config.get("multi_a", {}).get("enabled", False)),
    )
    if actual != expected[stage]:
        raise ValueError(
            f"{stage} HCR modules must be novelty_topk/history_fdr/multi_a="
            f"{expected[stage]}, got {actual}"
        )
