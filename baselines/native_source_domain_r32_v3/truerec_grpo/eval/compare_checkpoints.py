"""Uniform checkpoint comparison rows for Beta, Beta-Gamma and TrueRec."""
from __future__ import annotations

from model_spec import ModelSpec
from official_aligned_contract import CONTRACT_ID


def checkpoint_comparison_row(model: ModelSpec, split: str, aggregate: dict) -> dict:
    return {
        "checkpoint_name": model.name,
        "model_family": model.family,
        "split": split,
        "eval_contract_id": CONTRACT_ID,
        "overall_internal_diagnostics": aggregate["overall"],
        "per_domain": aggregate["per_domain"],
    }
