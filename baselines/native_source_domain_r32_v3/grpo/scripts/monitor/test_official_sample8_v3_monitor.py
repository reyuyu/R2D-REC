from pathlib import Path

from .official_sample8_v3_adapter import (
    adapt_events, captured_payload, extract_history_sids, is_official_sample8_v3_manifest,
)


def _cot(offset: float = 0.0):
    rewards = [0.0, 0.5, 2.0, 8.0, 0.0, -0.25, -1.0, 0.0]
    advantages = [-0.4, 0.1, 0.6, 1.8, -0.4, -0.5, -0.7, -0.4]
    return {
        "cot_text": "model reasoning",
        "cot_length": 123,
        "cot_reward": sum(rewards) + offset,
        "cot_advantage": 0.75,
        "candidate_ids": [[10 + index, 20 + index, 30 + index] for index in range(8)],
        "candidate_texts": [f"ABC-{index}" for index in range(8)],
        "candidate_sids": [["ad", index, 2, 3] for index in range(8)],
        "q_rewards": rewards,
        "sid_advantages": advantages,
        "candidate_details": [
            {"q_reward": reward, "sid_advantage": advantage, "reward_level": level}
            for reward, advantage, level in zip(
                rewards, advantages, ["domain", "a", "ab", "exact", "domain", "wrong_domain", "invalid", "domain"]
            )
        ],
    }


def test_v3_official_adapter_preserves_captured_g4_g8_and_optimization():
    event = {
        "type": "official_sample8",
        "step": 8,
        "rollout_id": 4,
        "target_domain": "ad",
        "recommendation_group_id": "group",
        "rollout_fingerprint": "fingerprint",
        "normalization_topology": "G4 CoT + 4 independent Official G8; never G32",
        "cots": [_cot(float(index)) for index in range(4)],
    }
    optimization = {
        "type": "optimization", "step": 9, "rollout_id": 4, "policy_iteration": 1,
        "cot_loss": 0.1, "sid_loss": 0.2, "total_loss": 0.3,
        "cot_ratio_mean": 1.0, "sid_ratio_mean": 1.0,
        "cot_clip_fraction": 0.0, "sid_clip_fraction": 0.0,
        "cot_approx_kl": 0.0, "sid_approx_kl": 0.0,
        "total_lora_grad_norm": 2.5,
    }
    groups, rows = adapt_events(
        [event, optimization],
        source_index={"group": {
            "prompt": "instruction\n<|ad_begin|><s_a_3><s_b_2><s_c_3> /think example "
                      "<|ad_begin|><s_a_4><s_b_2><s_c_3>",
            "target_domain": "ad",
        }},
    )
    assert is_official_sample8_v3_manifest({"experiment": "GR_REC_ThinkOfficialSample8_v3"})
    assert groups[0]["valid"] and "<s_a_3>" in groups[0]["input_prompt"]
    assert len(groups[0]["cots"]) == 4
    assert all(len(cot["candidates"]) == 8 for cot in groups[0]["cots"])
    candidate = groups[0]["cots"][0]["candidates"][3]
    assert candidate["raw_q_reward"] == candidate["final_sid_reward"] == candidate["cot_contribution"] == 8.0
    assert candidate["sid_advantage"] == 1.8 and candidate["masked_action_token_count"] == 3
    assert candidate["is_history_copy"] and candidate["copied_history_sid"] == ["ad", 3, 2, 3]
    assert not groups[0]["cots"][0]["candidates"][4]["is_history_copy"]
    assert rows == [optimization]
    assert captured_payload(groups, rows)["formula"] == "official_sample8_v3"


def test_v3_official_copy_parser_is_full_sid_target_domain_and_history_only():
    prompt = (
        "instruction <|video_begin|><s_a_9><s_b_9><s_c_9>\n"
        "<|video_begin|><s_a_1><s_b_2><s_c_3> "
        "<|video_begin|><s_a_1><s_b_8><s_c_8> "
        "<|ad_begin|><s_a_1><s_b_2><s_c_3> /think "
        "<|video_begin|><s_a_7><s_b_7><s_c_7>"
    )
    assert extract_history_sids(prompt, "video") == {
        ("video", 1, 2, 3), ("video", 1, 8, 8),
    }


def test_v3_official_dashboard_persists_details_and_exposes_ppo_diagnostics():
    source = (Path(__file__).with_name("static") / "video_official_anticopy_v5_dashboard.js").read_text(encoding="utf-8")
    assert "captureDetailState()" in source and "data-v5-detail" in source
    assert "isOfficialV3" in source and "official_sample8" in source
    assert "cot_ratio_mean" in source and "sid_ratio_mean" in source
    assert "cot_clip_fraction" in source and "sid_approx_kl" in source
    assert "total_lora_grad_norm" in source
