from pathlib import Path

from .video_official_anticopy_v5_adapter import (
    adapt_events,
    captured_payload,
    is_official_anticopy_mixed_v5_manifest,
    is_official_anticopy_v5_manifest,
)


def _cot(domain: str, raw: list[float], final: list[float], copies: list[bool]):
    return {
        "cot_text": f"<{domain}>reason</{domain}>",
        "cot_length": 12,
        "official_domain_prefix": f"<|{domain}_begin|>",
        "candidate_ids": [[index, index + 1, index + 2] for index in range(8)],
        "candidate_details": [
            {
                "candidate_sid": [domain, index, index + 1, index + 2],
                "candidate_text": f"<s_a_{index}><s_b_{index + 1}><s_c_{index + 2}>",
                "raw_q_reward": raw[index],
                "final_sid_reward": final[index],
                "reward_level": "Exact" if raw[index] == 8 else "AB" if raw[index] == 2 else "A" if raw[index] == .5 else "domain",
                "is_history_copy": copies[index],
                "sid_advantage": index - 3.5,
            }
            for index in range(8)
        ],
        "cot_contributions": final,
        "cot_reward": sum(final),
        "cot_advantage": .5,
        "history_sids": [[domain, 0, 1, 2]],
        "gold_sids": [[domain, 2, 3, 4]],
    }


def _event(domain: str, rollout_index: int, raw: list[float], final: list[float]):
    copies = [True, True, True, False, False, False, False, False]
    cot = _cot(domain, raw, final, copies)
    return {
        "type": "official_anticopy_mixed_v5",
        "step": rollout_index * 2,
        "rollout_id": rollout_index + 1,
        "fresh_rollout_index": rollout_index,
        "reward_stage": "nonvideo_copy_a_warmup" if rollout_index < 50 else "steady",
        "target_domain": domain,
        "recommendation_group_id": f"group-{domain}",
        "rollout_fingerprint": f"fp-{domain}",
        "cots": [dict(cot) for _ in range(4)],
    }


def test_mixed_manifest_and_four_domain_captured_rows():
    manifest = {"experiment": "GR_REC_OfficialAntiCopy_Mixed_v5"}
    assert is_official_anticopy_mixed_v5_manifest(manifest)
    assert is_official_anticopy_v5_manifest(manifest)
    events = []
    for index, domain in enumerate(("video", "ad", "prod", "living")):
        events.append(_event(domain, index, [0, .5, 8, 2, 0, 0, 0, 0], [0, .25, 6, 2, 0, 0, 0, 0]))
    source = {f"group-{domain}": {"prompt": f"input-{domain}", "target_domain": domain} for domain in ("video", "ad", "prod", "living")}
    groups, _ = adapt_events(events, source_index=source)
    assert [group["target_domain"] for group in groups] == ["video", "ad", "prod", "living"]
    assert all(group["valid"] and group["input_prompt"].startswith("input-") for group in groups)
    assert captured_payload(groups, [])["formula"] == "official_anticopy_mixed_v5"


def test_mixed_copy_sid_and_soft_discount_are_explicit():
    event = _event("ad", 49, [0, .5, 8, 2, 0, 0, 0, 0], [0, .25, 6, 2, 0, 0, 0, 0])
    groups, _ = adapt_events([event])
    rows = groups[0]["cots"][0]["candidates"]
    assert rows[1]["is_history_copy"] and rows[1]["copied_history_sid"] == rows[1]["sid"]
    assert rows[1]["copy_discounted"] and rows[1]["reward_delta"] == -.25
    assert rows[2]["copy_discounted"] and not rows[2]["copied_exact_preserved"]
    assert rows[3]["target_domain"] == "ad" and rows[3]["masked_action_token_count"] == 3
    assert groups[0]["fresh_rollout_index"] == 49
    assert groups[0]["reward_stage"] == "nonvideo_copy_a_warmup"


def test_dashboard_has_four_domain_copy_curves_and_g8_copy_sid_markers():
    source = (Path(__file__).with_name("static") / "video_official_anticopy_v5_dashboard.js").read_text(encoding="utf-8")
    for token in ("video", "ad", "prod", "living"):
        assert token in source
    assert "四域训练 History Copy Rate" in source
    assert "Official Probe History Copy Rate · 四域" in source
    assert "历史抄写 SID" in source
    assert "matched history SID" in source
    assert "4 × G8" in source and "32 仅总候选数" in source
    assert "const detailState = new Map()" in source
