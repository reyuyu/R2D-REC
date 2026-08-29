import json
from pathlib import Path

from fastapi.testclient import TestClient

from .exact_sharpen_v4_adapter import adapt_events, captured_payload, is_exact_sharpen_v4_manifest
from .server import create_app


def _cot(free_raw, official_raw, free_saturated=True, official_saturated=False):
    def branch(raw, saturated):
        saturation = [0.0 if value == 0.5 and saturated else value for value in raw]
        return saturation, [0.0] * 8, list(range(8))
    free, free_penalty, free_adv = branch(free_raw, free_saturated)
    official, official_penalty, official_adv = branch(official_raw, official_saturated)
    return {
        "cot_text": "reasoning", "cot_reward": 2.0,
        "cot_coverage": {"unique_exact": 0, "unique_uncovered_ab": 1, "unique_uncovered_a": 0},
        "free_raw_rewards": free_raw, "free_rewards": free,
        "free_duplicate_penalties": free_penalty, "free_advantages": free_adv,
        "free_sids": [["video", index, 2, 3] for index in range(8)],
        "free_candidate_ids": [[1, 2, 3, 4]] * 8, "free_saturated": free_saturated,
        "free_parser_statuses": ["complete_sid"] * 8,
        "free_action_spans": [[0, 4]] * 8,
        "official_raw_rewards": official_raw, "official_rewards": official,
        "official_duplicate_penalties": official_penalty, "official_advantages": official_adv,
        "official_sids": [["video", index, 2, 3] for index in range(8)],
        "official_candidate_ids": [[2, 3, 4]] * 8, "official_saturated": official_saturated,
    }


def test_v4_adapter_exposes_two_g8_branches_and_removed_a_reward():
    cot = _cot([0.5] + [8.0] * 7, [0.5] + [0.0] * 7)
    event = {
        "type": "exact_sharpen_v4", "step": 2, "rollout_id": 1,
        "recommendation_group_id": "group", "rollout_fingerprint": "fp",
        "free_saturated": True, "official_saturated": False,
        "free_a_plus_count": 32, "official_a_plus_count": 1,
        "normalization_topology": "4 independent G8 per branch", "cots": [cot] * 4,
    }
    group = adapt_events(
        [event],
        source_index={"group": {"prompt": "model input", "all_gold_sids": ["gold"]}},
        decode_token_ids=lambda ids: "decoded:" + ",".join(map(str, ids)),
    )[0]
    assert len(group["cots"]) == 4
    assert len(group["cots"][0]["free"]) == len(group["cots"][0]["official"]) == 8
    assert group["cots"][0]["free"][0]["a_reward_removed"] is True
    assert group["cots"][0]["free"][0]["saturation_reward"] == 0.0
    assert group["cots"][0]["official"][0]["a_reward_removed"] is False
    assert group["cots"][0]["official"][0]["saturation_reward"] == 0.5
    assert group["cots"][0]["free"][0]["masked_action_token_count"] == 4
    assert group["cots"][0]["official"][0]["masked_action_token_count"] == 3
    assert group["cots"][0]["free_summary"]["parsed_count"] == 8
    assert group["cots"][0]["branch_consistency"]["exact_sid_overlap"] == 8
    assert group["branch_consistency"]["cot_count"] == 4
    assert group["input_prompt"] == "model input"
    assert group["gold_sids"] == ["gold"]
    assert group["cots"][0]["free"][0]["completion_text"] == "decoded:1,2,3,4"
    assert captured_payload([group])["formula"] == "exact_sharpen_v4"
    assert is_exact_sharpen_v4_manifest({"experiment": "GR_REC_ThinkExactSharpen_v4"})


def test_v4_monitor_endpoint_and_frontend_contract(tmp_path):
    cot = _cot([0.5] + [0.0] * 7, [8.0] + [0.0] * 7)
    event = {
        "type": "exact_sharpen_v4", "step": 4, "rollout_id": 2,
        "recommendation_group_id": "g", "free_saturated": True,
        "official_saturated": False, "free_a_plus_count": 9,
        "official_a_plus_count": 1, "cots": [cot] * 4,
    }
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "v4", "experiment": "GR_REC_ThinkExactSharpen_v4",
    }), encoding="utf-8")
    (tmp_path / "exact_sharpen_v4.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    capabilities = client.get("/api/capabilities").json()
    assert capabilities["exact_sharpen_v4"] is True
    payload = client.get("/api/advantages").json()
    assert payload["formula"] == "exact_sharpen_v4"
    assert payload["groups"][0]["cots"][0]["free"][0]["a_reward_removed"] is True
    dashboard = (Path(__file__).parent / "static" / "exact_sharpen_v4_dashboard.js").read_text(encoding="utf-8")
    for contract in (
        "A 已删除", "Free Sample8", "Official Sample8", "duplicate_penalty",
        "probe_free", "probe_official", "v4ProbeFreeRewardChart",
        "v4ProbeOfficialRewardChart", "展开 32 条 Beam SID", "完整 Free 采样",
        "branch_consistency", "A+ rate 绝对差", "采样与解析明细",
        "模型输入原文", "completion_text", "var(--ink)",
    ):
        assert contract in dashboard
