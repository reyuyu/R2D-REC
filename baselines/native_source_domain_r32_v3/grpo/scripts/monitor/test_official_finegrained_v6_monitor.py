from pathlib import Path

from .official_finegrained_v6_adapter import (
    adapt_events,
    captured_payload,
    is_official_finegrained_v6_manifest,
)


def _cot(raw, copied):
    hit_a = [int(value >= 0.5) for value in raw]
    hit_b = [int(value >= 2.0) for value in raw]
    hit_c = [int(value == 8.0) for value in raw]
    return {
        "cot_text": "reasoning",
        "cot_reward": sum(raw),
        "cot_advantage": 1.0,
        "candidate_sids": [["ad", index, 2, 3] for index in range(8)],
        "candidate_texts": [f"candidate {index}" for index in range(8)],
        "raw_q_rewards": raw,
        "is_history_copy": copied,
        "hit_A": hit_a,
        "hit_B": hit_b,
        "hit_C": hit_c,
        "token_advantages": [[1.0 if a else -1.0, 2.0 if b else -0.5, 3.0 if c else -0.25] for a, b, c in zip(hit_a, hit_b, hit_c)],
        "token_masks": [[1, a, b] for a, b in zip(hit_a, hit_b)],
        "active_A_tokens": 8,
        "active_B_tokens": sum(hit_a),
        "active_C_tokens": sum(hit_b),
        "zero_std_A": False,
        "zero_std_B": False,
        "zero_std_C": False,
        "history_sids": [["ad", 0, 2, 3]],
        "gold_sids": [["ad", 1, 2, 3]],
    }


def test_v6_adapter_preserves_token_credit_gate_and_copy_diagnostics():
    raw = [0.0, 0.5, 2.0, 8.0, 0.0, 0.0, 0.0, 0.0]
    copied = [True, False, False, True, False, False, False, False]
    event = {
        "type": "official_finegrained_v6",
        "step": 8,
        "rollout_id": 4,
        "fresh_rollout_index": 4,
        "target_domain": "ad",
        "recommendation_group_id": "group",
        "rollout_fingerprint": "fingerprint",
        "cots": [_cot(raw, copied) for _ in range(4)],
    }
    optimization = {
        "type": "optimization", "step": 9, "rollout_id": 4,
        "policy_iteration": 1, "sid_A_ratio_mean": 1.0,
    }
    groups, rows = adapt_events(
        [event, optimization],
        source_index={"group": {"prompt": "full prompt", "target_domain": "ad"}},
    )
    assert is_official_finegrained_v6_manifest({"experiment": "GR_REC_OfficialFineGrained_v6A"})
    assert groups[0]["valid"] and groups[0]["input_prompt"] == "full prompt"
    candidate = groups[0]["cots"][0]["candidates"][2]
    assert candidate["hit_A"] == 1 and candidate["hit_B"] == 1 and candidate["hit_C"] == 0
    assert candidate["token_advantages"] == [1.0, 2.0, -0.25]
    assert candidate["token_masks"] == [1, 1, 1]
    assert groups[0]["cots"][0]["candidates"][0]["is_history_copy"]
    assert not groups[0]["cots"][0]["candidates"][0]["copy_discounted"]
    assert rows == [optimization]
    assert captured_payload(groups, rows)["formula"] == "official_finegrained_v6"


def test_v6_dashboard_keeps_open_details_and_shows_layerwise_credit():
    source = (Path(__file__).with_name("static") / "video_official_anticopy_v5_dashboard.js").read_text(encoding="utf-8")
    assert "captureDetailState()" in source
    assert "data-v5-detail" in source
    assert "token_advantages" in source and "token_masks" in source
    assert "sid_A_ratio_mean" in source and "sid_B_approx_kl" in source and "sid_C_approx_kl" in source
    assert "Heldout Probe4" in source and "Training-overlap Diagnostic4" in source
    assert "copy 仅诊断，不改训练信号" in source
