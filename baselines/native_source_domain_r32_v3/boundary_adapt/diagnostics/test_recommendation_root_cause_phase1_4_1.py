"""Focused CPU tests for Phase 1.4.1 statistical helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("recommendation_root_cause_phase1_4_1.py")
spec = importlib.util.spec_from_file_location("phase141", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def candidate(predicted, copy="NOVEL"):
    return {"predicted_sid": predicted, "copy_class": copy}


def test_metrics() -> None:
    item = {"all_gold_sids": ["<|video_begin|><s_a_1><s_b_2><s_c_3>"]}
    beams = [candidate(["video", 9, 9, 9]), candidate(["video", 1, 2, 3], "EXACT_COPY")]
    beams += [candidate(["video", 8, 8, index]) for index in range(30)]
    result = module.row_metrics({"beams": beams}, item)
    assert result == {"mrr": 0.5, "hit32": 1.0, "ab32": 1.0, "a32": 1.0, "history_fraction": 1 / 32}


def test_paired_bootstrap() -> None:
    diffs = {
        "a": {metric: 1.0 for metric in module.METRICS},
        "b": {metric: -0.5 for metric in module.METRICS},
    }
    first = module.paired_bootstrap(diffs, "fixture")
    second = module.paired_bootstrap(diffs, "fixture")
    assert first == second
    assert first["metrics"]["mrr"]["estimate"] == 0.25
    assert first["replicates"] == 10_000


def test_direction_and_prefix() -> None:
    assert module.direction(1) == "BETA_GT_STEP900"
    assert module.direction(-1) == "STEP900_GT_BETA"
    assert module.direction(0) == "TIE"
    assert module.common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert module.common_suffix([1, 2, 3, 7], [1, 2, 4, 7], 2) == 1


if __name__ == "__main__":
    test_metrics()
    test_paired_bootstrap()
    test_direction_and_prefix()
    print("PHASE1_4_1_TARGETED_TESTS=PASS")
