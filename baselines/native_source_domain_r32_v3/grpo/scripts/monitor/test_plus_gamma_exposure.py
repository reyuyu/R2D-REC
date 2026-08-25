import json
from pathlib import Path

from monitor.plus_gamma_exposure import annotate, build_index, exposure, load_index


def _row(group_id: str, repeat: int = 1) -> str:
    metadata = {"recommendation_group_id": group_id}
    return "\n".join(json.dumps({
        "source_segment": "recommendation_cot",
        "aux_metadata_json": json.dumps(metadata),
    }) for _ in range(repeat)) + "\n"


def test_exact_epoch_membership_and_counts(tmp_path: Path):
    first = "1" * 64
    second = "2" * 64
    epoch1, epoch2 = tmp_path / "epoch1.jsonl", tmp_path / "epoch2.jsonl"
    epoch1.write_text(_row(first, 2), encoding="utf-8")
    epoch2.write_text(_row(second), encoding="utf-8")
    index = build_index((("epoch1", epoch1), ("epoch2", epoch2)))
    assert exposure(first, index) == {
        "status": "seen", "seen": True, "epochs": ["epoch1"], "row_count": 2,
        "match_key": "recommendation_group_id",
    }
    assert exposure("3" * 64, index)["status"] == "unseen"


def test_unknown_is_not_reported_as_unseen():
    assert exposure(None, {"groups": {}})["seen"] is None
    assert exposure("1" * 64, None)["status"] == "unknown"


def test_group_and_candidates_receive_same_exposure():
    group_id = "a" * 64
    rows = [{"group_id": group_id, "candidates": [{"candidate_id": 0}]}]
    annotate(rows, {"groups": {group_id: {"epochs": ["epoch2"], "row_count": 3}}})
    assert rows[0]["plus_gamma_exposure"]["epochs"] == ["epoch2"]
    assert rows[0]["candidates"][0]["plus_gamma_exposure"]["row_count"] == 3


def test_invalid_cache_fails_closed(tmp_path: Path):
    cache = tmp_path / "bad.json"
    cache.write_text("{}", encoding="utf-8")
    assert load_index(cache) is None
