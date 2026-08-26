"""Deterministic, family-independent aggregation of non-official diagnostics."""
from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Sequence

from eval_result import EvaluationBusinessSample
from scoring_adapter import INTERNAL_DIAGNOSTICS, SCORING_LABEL, internal_group_diagnostics


DOMAINS = ("video", "prod", "ad", "living")


def _block(samples: Sequence[EvaluationBusinessSample]) -> dict:
    rows = [internal_group_diagnostics(item.candidates, item.all_gold_abc) for item in samples]
    return {
        "N": len(rows),
        "score_label": SCORING_LABEL,
        **{key: fmean(float(row[key]) for row in rows) if rows else 0.0 for key in INTERNAL_DIAGNOSTICS},
    }


def aggregate_results(samples: Sequence[EvaluationBusinessSample]) -> dict:
    ordered = sorted(samples, key=lambda item: item.recommendation_group_id)
    if len({item.recommendation_group_id for item in ordered}) != len(ordered):
        raise ValueError("duplicate evaluation business group")
    by_domain = defaultdict(list)
    for sample in ordered:
        by_domain[sample.target_domain].append(sample)
    return {"overall": _block(ordered), "per_domain": {domain: _block(by_domain[domain]) for domain in DOMAINS}}
