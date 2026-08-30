import pytest
import torch

from ablations.gr_rec_video_official_anticopy_v5.video_official_anticopy_trainer import (
    shape_sid_reward as original_v5_shape_sid_reward,
)

from .official_probe import beam_copy_details, official_probe_summary
from .run_official_anticopy_mixed_train import (
    EXPECTED_AFTER_PROBE4,
    EXPECTED_AFTER_PROBE16,
    EXPECTED_DOMAIN_GROUPS,
    EXPECTED_RAW_GROUPS,
    EXPECTED_STEPS,
    EXPECTED_TRAIN_GROUPS,
    FINAL_GROUP_ID_LIST_SHA256,
    FINAL_TRAIN_CANONICAL_SHA256,
    FIXED_PROBE4_IDS,
    FIXED_PROBE16_IDS,
    FORBIDDEN_POSITIVE_DATASET,
    PARENT_ADAPTER_SHA256,
    SOURCE_DATASET,
    SOURCE_DATASET_SHA256,
    build_v5_parser,
    prepare_v5_run_plan,
    validate_parent_adapter,
    validate_source_dataset,
)
from .official_anticopy_mixed_trainer import (
    COT_G,
    SID_G,
    assert_iteration_reuse,
    audit_v5_sampler,
    combine_losses,
    cot_candidate_contribution,
    domain_rollout_summary,
    extract_history_sids,
    finalize_global_records,
    fresh_rollout_index_from_step,
    independent_official_advantages,
    official_action_mask,
    reward_stage,
    rollout_fingerprint,
    shape_sid_reward,
)


def _args():
    values = [
        "--run-id", "V5-MIXED-CPU-TEST", "--n-groups", "all",
        "--max-steps", "2", "--probe-groups", "16",
    ]
    for gid in FIXED_PROBE16_IDS:
        values.extend(["--probe-group-id", gid])
    return build_v5_parser().parse_args(values)


@pytest.fixture(scope="module")
def plan():
    return prepare_v5_run_plan(_args())


def test_source_is_exact_v3_dataset_not_positive_dataset():
    audit = validate_source_dataset()
    assert audit == {
        "path": str(SOURCE_DATASET), "baseline_data_path": str(SOURCE_DATASET),
        "sha256": SOURCE_DATASET_SHA256,
        "rows": 3098, "business_groups": EXPECTED_RAW_GROUPS,
    }
    assert SOURCE_DATASET != FORBIDDEN_POSITIVE_DATASET
    assert "positive_groups_1946" not in str(SOURCE_DATASET)


def test_parent_checkpoint_and_adapter_sha():
    assert validate_parent_adapter()["adapter_sha256"] == PARENT_ADAPTER_SHA256


def test_dataset_guards_and_four_domain_distribution(plan):
    guard = plan["dataset_guard"]
    assert guard["source_business_groups"] == EXPECTED_RAW_GROUPS
    assert guard["groups_after_probe4"] == EXPECTED_AFTER_PROBE4
    assert guard["groups_after_probe16"] == EXPECTED_AFTER_PROBE16
    assert guard["train_rows"] == guard["unique_groups"] == EXPECTED_TRAIN_GROUPS
    assert guard["domain_group_counts"] == EXPECTED_DOMAIN_GROUPS
    assert guard["canonical_train_sha256"] == FINAL_TRAIN_CANONICAL_SHA256
    assert guard["group_id_list_sha256"] == FINAL_GROUP_ID_LIST_SHA256
    assert guard["probe4_overlap"] == 0
    assert guard["probe16_overlap"] == 0


def test_dataset_is_think_only_unique_and_probe16_excluded(plan):
    rows = list(plan["dataset"])
    gids = {row["recommendation_group_id"] for row in rows}
    assert len(rows) == len(gids) == EXPECTED_TRAIN_GROUPS
    assert {row["route"] for row in rows} == {"think"}
    assert {row["target_domain"] for row in rows} == set(EXPECTED_DOMAIN_GROUPS)
    assert not gids.intersection(FIXED_PROBE16_IDS)


def test_probe16_is_four_fixed_groups_per_domain(plan):
    domains = [plan["probe_records"][gid]["think"]["target_domain"]
               for gid in FIXED_PROBE16_IDS]
    assert domains == ["video", "living", "prod", "ad"] * 4
    assert len(FIXED_PROBE4_IDS) == 4


def test_history_parser_is_complete_domain_aware_and_user_history_only():
    prompt = (
        "instruction <|video_begin|><s_a_9><s_b_9><s_c_9>\n"
        "history <|video_begin|><s_a_1><s_b_2><s_c_3>, "
        "<|ad_begin|><s_a_4><s_b_5><s_c_6>, "
        "<|prod_begin|><s_a_7><s_b_8>, "
        "<s_a_1><s_b_2><s_c_4>./think"
        "</think><|living_begin|><s_a_7><s_b_8><s_c_9>"
    )
    assert extract_history_sids(prompt) == {
        ("video", 1, 2, 3), ("ad", 4, 5, 6),
    }
    assert extract_history_sids(prompt, "video") == {("video", 1, 2, 3)}
    assert extract_history_sids(prompt, "ad") == {("ad", 4, 5, 6)}
    assert extract_history_sids(prompt, "prod") == set()


def test_full_same_domain_sid_equality_is_required_for_copy():
    history = {("ad", 1, 2, 3)}
    assert ("ad", 1, 2, 3) in history
    assert ("ad", 1, 2, 4) not in history
    assert ("video", 1, 2, 3) not in history


@pytest.mark.parametrize(
    "sid",
    [None, ("video", 9, 9, 9), ("video", 1, 9, 9),
     ("video", 1, 2, 9), ("video", 1, 2, 3)],
)
def test_video_reward_has_exact_original_v5_parity(sid):
    gold = {("video", 1, 2, 3)}
    history = {sid} if sid is not None else set()
    assert shape_sid_reward(sid, gold, history, "video", 0) == (
        original_v5_shape_sid_reward(sid, gold, history)
    )


@pytest.mark.parametrize(
    "sid,raw",
    [(("ad", 9, 9, 9), 0.0), (("ad", 1, 9, 9), 0.5),
     (("ad", 1, 2, 9), 2.0), (("ad", 1, 2, 3), 8.0)],
)
def test_nonvideo_noncopy_uses_raw_q_reward(sid, raw):
    assert shape_sid_reward(sid, {("ad", 1, 2, 3)}, set(), "ad", 0) == (
        raw, False, raw
    )


@pytest.mark.parametrize(
    "domain", ["ad", "prod", "living"]
)
@pytest.mark.parametrize(
    "suffix,expected49,expected50",
    [((9, 9, 9), 0.0, 0.0), ((1, 9, 9), 0.25, 0.0),
     ((1, 2, 9), 1.0, 1.0), ((1, 2, 3), 6.0, 6.0)],
)
def test_nonvideo_copy_boundary_rollout49_to50(domain, suffix, expected49, expected50):
    sid = (domain, *suffix)
    gold = {(domain, 1, 2, 3)}
    history = {sid}
    assert shape_sid_reward(sid, gold, history, domain, 49)[2] == expected49
    assert shape_sid_reward(sid, gold, history, domain, 50)[2] == expected50
    assert reward_stage(domain, 49) == "nonvideo_copy_a_warmup"
    assert reward_stage(domain, 50) == "nonvideo_copy_a_off"


def test_optimizer_step_boundary_maps_to_frozen_fresh_rollout_index():
    assert fresh_rollout_index_from_step(98) == 49
    assert fresh_rollout_index_from_step(100) == 50
    with pytest.raises(RuntimeError):
        fresh_rollout_index_from_step(99)


def test_cot_contribution_uses_domain_contract():
    assert cot_candidate_contribution("video", 8.0, True, 8.0) == 0.0
    assert cot_candidate_contribution("video", 2.0, False, 2.0) == 2.0
    assert cot_candidate_contribution("ad", 8.0, True, 6.0) == 6.0
    assert cot_candidate_contribution("living", 0.5, True, 0.25) == 0.25


def _record(index, domain="ad"):
    candidates = [
        (domain, 1, 2, 3), (domain, 1, 2, 9),
        (domain, 1, 9, 9), (domain, 2 + index, 3, 4),
        (domain, 5, 6, 7), (domain, 8, 9, 10),
        (domain, 11, 12, 13), None,
    ]
    return {
        "recommendation_group_id": "g", "target_domain": domain,
        "cot_ids": [100 + index], "cot_text": f"cot-{index}", "cot_length": 1,
        "candidate_ids": [[1, 2, 3] for _ in range(SID_G)],
        "candidate_texts": [f"candidate-{i}" for i in range(SID_G)],
        "candidate_sids": candidates, "gold_sids": [(domain, 1, 2, 3)],
        "history_sids": candidates[:3],
    }


def test_finalize_freezes_stage_reward_and_four_independent_g8():
    records = finalize_global_records([_record(i) for i in range(COT_G)], 49)
    fingerprint = rollout_fingerprint(records)
    assert all(row["fresh_rollout_index"] == 49 for row in records)
    assert all(row["reward_stage"] == "nonvideo_copy_a_warmup" for row in records)
    assert rollout_fingerprint(records) == fingerprint
    assert assert_iteration_reuse(fingerprint, fingerprint, 1, 4, 4)
    assert assert_iteration_reuse(fingerprint, fingerprint, 2, 4, 4)
    advantages = independent_official_advantages([row["sid_rewards"] for row in records])
    assert advantages.shape == (COT_G, SID_G)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(COT_G), atol=1e-6)
    summary = domain_rollout_summary(records)
    assert summary["domain"] == "ad"
    assert summary["reward_stage"] == "nonvideo_copy_a_warmup"
    assert "copy_positive_advantage_count" in summary


def test_official_mask_and_loss_contract():
    assert official_action_mask([[1, 2, 3] for _ in range(SID_G)]) == [
        [1, 1, 1] for _ in range(SID_G)
    ]
    with pytest.raises(ValueError):
        official_action_mask([[1, 2, 3, 4] for _ in range(SID_G)])
    assert combine_losses(torch.tensor(2.0), torch.tensor(3.0)).item() == 5.0


def test_sampler_audit_manifest_contract(plan):
    audit = audit_v5_sampler(plan["dataset"], object())
    assert audit["selected_groups"] == EXPECTED_TRAIN_GROUPS
    assert audit["optimizer_steps"] == EXPECTED_STEPS
    assert audit["official_g8_groups"] == 4
    assert audit["domain_group_counts"] == EXPECTED_DOMAIN_GROUPS
    assert audit["nothink_optimizer_steps"] == 0


def test_probe_uses_production_reward_and_only_adds_copy_anatomy():
    gold = {("prod", 1, 2, 3)}
    history = {("prod", 1, 2, 3), ("prod", 1, 2, 9)}
    details = beam_copy_details(
        [("prod", 1, 2, 3), ("prod", 1, 2, 9), ("prod", 1, 9, 9)],
        gold, history,
    )
    candidates = [{
        "reward": 8.0, "closed": True, "exact": 1, "ab": 0, "a": 0,
        "invalid": 0, "completion_length": 32,
        "beam_candidate_details": details,
    }]
    summary = official_probe_summary(candidates)
    assert summary["reward_mean"] == 8.0
    assert "no training copy discount" in summary["reward_semantics"]
    assert summary["copy_Exact"] == 1 and summary["copy_AB"] == 1
    assert summary["noncopy_A"] == 1
