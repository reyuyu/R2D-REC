from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
try:
    from .advantage_adapter import (
        _formal_source_roots,
        formal_available,
        reconstruct_nothink_group,
        reconstruct_think_group,
    )
    from .server import create_app
except ImportError:
    from advantage_adapter import (
        _formal_source_roots,
        formal_available,
        reconstruct_nothink_group,
        reconstruct_think_group,
    )
    from server import create_app


GOLD = [["prod", 1, 2, 3]]


def candidate_for_reward(index: int, reward: float) -> dict:
    sid = {
        0.0: ["prod", 99, 0, 0],
        0.5: ["prod", 1, 99, 0],
        2.0: ["prod", 1, 2, 99],
        8.0: ["prod", 1, 2, 3],
    }[reward]
    return {
        "candidate_id": index,
        "completion": f"candidate {index} <|prod_begin|><s_a_{sid[1]}><s_b_{sid[2]}><s_c_{sid[3]}>",
        "completion_length": 12,
        "parsed_sid": sid,
        "reward": reward,
    }


def alignment(_candidate: dict, _index: int) -> dict:
    return {
        "valid": True,
        "eligible": True,
        "mode": "branch",
        "failure": None,
        "positions": [1, 2, 3, 4],
        "spans": [
            {"start": 0, "end": 1, "text": name}
            for name in ("点击", "A", "B", "C")
        ],
    }


def no_think(pattern: list[float]) -> dict:
    trace = {
        "step": 10,
        "rollout_id": 5,
        "group_id": "synthetic",
        "route": "no_think",
        "gold_sids": GOLD,
        "candidates": [candidate_for_reward(index, reward) for index, reward in enumerate(pattern)],
    }
    return reconstruct_nothink_group(trace, aligner=alignment)


def assert_close(actual: float | None, expected: float) -> None:
    assert actual is not None and math.isclose(actual, expected, abs_tol=1e-12)


def test_formal_source_root_discovery() -> None:
    roots = _formal_source_roots()
    assert roots
    assert len(roots) == len(set(roots))


def test_think_exact_clamp_display_contract() -> None:
    if not formal_available():
        return
    trace = {
        "step": 2,
        "route": "think",
        "candidates": [
            {"candidate_id": index, "completion": f"reasoning {index}", "completion_length": 20, "reward": reward}
            for index, reward in enumerate([8, 8, 8, 12])
        ],
    }
    result = reconstruct_think_group(trace)
    assert result["raw_advantages"] == [-0.125, -0.125, -0.125, 0.375]
    assert result["final_advantages"] == [0.0, 0.0, 0.0, 0.375]
    assert result["clamped_count"] == 3
    assert result["candidates"][0]["provenance"]["reward"]["source"] == "captured"
    assert result["candidates"][0]["provenance"]["final_advantage"]["source"] == "reconstructed"


def test_nothink_singleton_credit_patterns() -> None:
    if not formal_available():
        return
    a = no_think([0.0] * 7 + [0.5])
    assert_close(a["candidates"][0]["stages"][1]["credit"], -0.0078125)
    assert_close(a["candidates"][7]["stages"][1]["credit"], 0.0546875)

    b = no_think([0.0] * 7 + [2.0])
    assert b["b_singleton"] is True
    assert_close(b["candidates"][7]["stages"][2]["credit"], 0.1640625)

    c = no_think([0.0] * 7 + [8.0])
    assert c["b_singleton"] is True and c["c_singleton"] is True
    assert_close(c["candidates"][7]["stages"][3]["credit"], 0.65625)

    ab = no_think([0.5] * 7 + [2.0])
    assert_close(ab["candidates"][0]["stages"][2]["credit"], -0.0234375)
    assert_close(ab["candidates"][7]["stages"][2]["credit"], 0.1640625)

    exact = no_think([2.0] * 7 + [8.0])
    assert_close(exact["candidates"][0]["stages"][3]["credit"], -0.09375)
    assert_close(exact["candidates"][7]["stages"][3]["credit"], 0.65625)


def test_zero_bridge_and_gated_are_distinct() -> None:
    if not formal_available():
        return
    result = no_think([0.0] * 8)
    assert result["taxonomy"] == "DEAD_ZERO_BRIDGE"
    assert result["bridge"]["active"] is True
    assert result["candidates"][0]["stages"][0]["credit"] == 0.0
    assert result["candidates"][0]["stages"][1]["credit"] == 0.0
    assert result["candidates"][0]["stages"][2]["eligible"] is False
    assert result["candidates"][0]["stages"][2]["credit"] is None


def test_read_only_api_and_legacy_run() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "current"
        (run / "traces").mkdir(parents=True)
        (run / "manifest.json").write_text('{"run_id":"current"}', encoding="utf-8")
        trace = {
            "step": 2,
            "route": "think",
            "candidates": [
                {"candidate_id": index, "completion": "x", "reward": reward}
                for index, reward in enumerate([8, 8, 8, 12])
            ],
        }
        (run / "traces" / "traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
        legacy = root / "legacy"
        legacy.mkdir()
        (legacy / "manifest.json").write_text('{"run_id":"legacy"}', encoding="utf-8")
        client = TestClient(create_app(runs_dir=root))
        payload = client.get("/api/advantages?run_id=current").json()
        assert payload["read_only"] is True
        if formal_available():
            assert payload["groups"][0]["clamped_count"] == 3
            assert payload["groups"][0]["provenance"]["source"] == "reconstructed"
        else:
            assert payload["groups"][0]["valid"] is False
        assert client.get("/api/advantages?run_id=legacy").json()["groups"] == []


if __name__ == "__main__":
    test_formal_source_root_discovery()
    test_think_exact_clamp_display_contract()
    test_nothink_singleton_credit_patterns()
    test_zero_bridge_and_gated_are_distinct()
    test_read_only_api_and_legacy_run()
    print("ADVANTAGE ADAPTER CPU TESTS PASSED")
