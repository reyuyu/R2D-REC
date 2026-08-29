import json
from pathlib import Path

from fastapi.testclient import TestClient

from .server import create_app
from .video_official_anticopy_v5_adapter import adapt_events, captured_payload

def _event():
    candidates = []
    for i in range(8):
        raw = [0.5, 2.0, 8.0, 0.0, 0.5, 0.0, -1.0, 0.0][i]
        copied = i < 4
        final = 8.0 if raw == 8.0 else 0.0 if copied else raw
        candidates.append({
            "candidate_sid": ["video", i, i + 1, i + 2],
            "candidate_text": f"<s_a_{i}><s_b_{i+1}><s_c_{i+2}>",
            "raw_q_reward": raw, "reward_level": "Exact" if raw == 8 else "AB" if raw == 2 else "A" if raw == .5 else "domain",
            "is_history_copy": copied, "final_sid_reward": final,
            "sid_advantage": float(i - 3.5),
        })
    cot = {
        "cot_text": "<think>reason</think>", "cot_length": 7,
        "candidate_ids": [[i, i + 1, i + 2] for i in range(8)],
        "candidate_details": candidates, "cot_contributions": [0, 0, 0, 0, .5, 0, -1, 0],
        "cot_reward": -.5, "cot_advantage": .25, "history_sids": [["video", 0, 1, 2]],
        "gold_sids": [["video", 2, 3, 4]], "official_domain_prefix": "<|video_begin|>",
    }
    return {
        "type": "video_official_anticopy_v5", "step": 10, "rollout_id": 6,
        "recommendation_group_id": "g", "rollout_fingerprint": "fp",
        "normalization_topology": "G4 + 4 independent Official G8; never G32",
        "cots": [dict(cot) for _ in range(4)],
    }


def test_v5_adapter_preserves_captured_anticopy_score_flow_and_text():
    groups, optimization = adapt_events([_event()], source_index={"g": {"prompt": "model input", "target_domain": "video"}})
    assert not optimization
    assert len(groups) == 1 and groups[0]["valid"]
    assert groups[0]["input_prompt"] == "model input"
    assert len(groups[0]["cots"]) == 4
    rows = groups[0]["cots"][0]["candidates"]
    assert len(rows) == 8 and all(row["masked_action_token_count"] == 3 for row in rows)
    assert rows[0]["raw_q_reward"] == .5 and rows[0]["final_sid_reward"] == 0 and rows[0]["anti_copy_removed"]
    assert rows[1]["raw_q_reward"] == 2 and rows[1]["final_sid_reward"] == 0 and rows[1]["anti_copy_removed"]
    assert rows[2]["raw_q_reward"] == 8 and rows[2]["final_sid_reward"] == 8 and rows[2]["copied_exact_preserved"]
    assert rows[2]["cot_contribution"] == 0
    assert rows[4]["raw_q_reward"] == rows[4]["final_sid_reward"] == .5
    assert rows[0]["completion_text"].startswith("<s_a_")


def test_v5_adapter_exposes_both_policy_iterations_without_reconstruction():
    optimization = [
        {"type": "optimization", "step": 9, "rollout_id": 5, "policy_iteration": 1, "sid_ratio_mean": 1.0},
        {"type": "optimization", "step": 10, "rollout_id": 5, "policy_iteration": 2, "sid_ratio_mean": .98},
    ]
    groups, rows = adapt_events([*optimization, _event()])
    payload = captured_payload(groups, rows)
    assert payload["supported"] and payload["formula"] == "video_official_anticopy_v5"
    assert [row["policy_iteration"] for row in payload["optimization"]] == [1, 2]
    assert payload["groups"][0]["normalization_topology"].endswith("never G32")


def test_v5_dashboard_preserves_details_and_visualizes_advantage_sign():
    source = (Path(__file__).parent / "static" / "video_official_anticopy_v5_dashboard.js").read_text(
        encoding="utf-8"
    )
    assert "const detailState = new Map()" in source
    assert "data-v5-detail" in source
    assert "document.addEventListener('toggle'" in source
    assert "cotOverview(selected.cots)" in source
    assert "v5-cot-score positive" not in source  # tone remains data-driven
    assert "v5-adv-positive" in source
    assert "v5-adv-negative" in source
    assert "v5-adv-zero" in source
    assert "v5-adv-pill" in source


def test_v5_monitor_endpoint_serves_captured_groups(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "v5", "experiment": "GR_REC_VideoOfficialAntiCopy_v5",
        "source_dataset_path": str(tmp_path / "missing.jsonl"),
        "source_dataset_sha256": "0" * 64,
    }), encoding="utf-8")
    (tmp_path / "video_official_anticopy_v5.jsonl").write_text(
        json.dumps(_event()) + "\n", encoding="utf-8"
    )
    client = TestClient(create_app(tmp_path))
    capabilities = client.get("/api/capabilities").json()
    assert capabilities["video_official_anticopy_v5"] is True
    assert capabilities["advantage_source"] == "captured"
    payload = client.get("/api/advantages").json()
    assert payload["formula"] == "video_official_anticopy_v5"
    assert payload["groups"][0]["cots"][0]["candidates"][0]["is_history_copy"] is True
