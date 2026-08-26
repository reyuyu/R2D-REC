"""Internal diagnostics only; no official composite formula is invented."""
from __future__ import annotations

from pathlib import Path
import sys

from beam32_contract import BeamCandidate

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from rollout_metrics import split_abc  # noqa: E402


OFFICIAL_COMPOSITE_FORMULA_AVAILABLE = False
SCORING_LABEL = "NOT_OFFICIAL_COMPOSITE_SCORE"
INTERNAL_DIAGNOSTICS = ("GoldSIDHit@32", "GoldABHit@32", "GoldAHit@32", "SIDValid@32")


def internal_group_diagnostics(candidates: tuple[BeamCandidate, ...], all_gold_abc: tuple[str, ...]) -> dict[str, float | str]:
    valid = [item for item in candidates if item.format_valid]
    gold_parts = [split_abc(value) for value in all_gold_abc]
    parsed_parts = [split_abc(item.parsed_abc) for item in valid]
    exact = any(parts in gold_parts for parts in parsed_parts)
    ab = any(any(parts[:2] == gold[:2] for gold in gold_parts) for parts in parsed_parts)
    a = any(any(parts[0] == gold[0] for gold in gold_parts) for parts in parsed_parts)
    return {
        "score_label": SCORING_LABEL,
        "GoldSIDHit@32": float(exact),
        "GoldABHit@32": float(ab),
        "GoldAHit@32": float(a),
        "SIDValid@32": len(valid) / 32.0,
    }
