import json
from pathlib import Path

import pytest
import torch

from .official_probe import (
    beam_copy_details, heldout_video_probe_due, official_probe_summary,
    restore_beam_stats,
)
from .run_video_official_anticopy_train import (
    EXPECTED_AFTER_PROBE4, EXPECTED_RAW_GROUPS, EXPECTED_STEPS,
    EXPECTED_TRAIN_GROUPS, EXPECTED_VIDEO_GROUPS,
    FINAL_GROUP_ID_LIST_SHA256, FINAL_TRAIN_CANONICAL_SHA256,
    FIXED_PROBE4_IDS, FORBIDDEN_POSITIVE_DATASET, HELDOUT_VIDEO_PROBE_IDS,
    PARENT_ADAPTER_SHA256, SOURCE_DATASET, SOURCE_DATASET_SHA256,
    build_v5_parser, prepare_v5_run_plan, validate_parent_adapter,
    validate_source_dataset,
)
from .video_official_anticopy_trainer import (
    COT_G, SID_G, assert_iteration_reuse, audit_v5_sampler, combine_losses,
    cot_candidate_contribution, extract_history_sids, finalize_global_records,
    independent_official_advantages, official_action_mask, shape_sid_reward,
    rollout_fingerprint, warmup_reward_contract,
)


def _args():
    values = [
        "--run-id", "V5-CPU-TEST", "--n-groups", "all",
        "--max-steps", "2", "--probe-groups", "4",
    ]
    for gid in FIXED_PROBE4_IDS:
        values.extend(["--probe-group-id", gid])
    return build_v5_parser().parse_args(values)


def test_warmup_launcher_has_independent_identity():
    launcher = Path(__file__).with_name("launch_video_official_anticopy_train.sh")
    source = launcher.read_text(encoding="utf-8")
    assert "GR-REC-VIDEO-OFFICIAL-ANTICOPY-V5-WARMUP-FORMAL" in source
    assert "-m ablations.gr_rec_video_official_anticopy_v5_warmup." in source


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


def test_dataset_selection_and_sha_guards(plan):
    guard = plan["dataset_guard"]
    assert guard["source_business_groups"] == EXPECTED_RAW_GROUPS
    assert guard["groups_after_probe4"] == EXPECTED_AFTER_PROBE4
    assert guard["video_think_groups_before_v5_heldout"] == EXPECTED_VIDEO_GROUPS
    assert guard["train_rows"] == EXPECTED_TRAIN_GROUPS
    assert guard["canonical_train_sha256"] == FINAL_TRAIN_CANONICAL_SHA256
    assert guard["group_id_list_sha256"] == FINAL_GROUP_ID_LIST_SHA256
    assert guard["probe4_overlap"] == guard["heldout_video_overlap"] == 0


def test_dataset_is_think_video_unique_and_heldout(plan):
    rows = list(plan["dataset"])
    gids = {row["recommendation_group_id"] for row in rows}
    assert len(rows) == len(gids) == EXPECTED_TRAIN_GROUPS
    assert {row["route"] for row in rows} == {"think"}
    assert {row["target_domain"] for row in rows} == {"video"}
    assert not gids.intersection(FIXED_PROBE4_IDS)
    assert not gids.intersection(HELDOUT_VIDEO_PROBE_IDS)


def test_history_parser_uses_only_complete_user_history():
    prompt = (
        "instruction example <|video_begin|><s_a_9><s_b_9><s_c_9>\n"
        "user history <|video_begin|><s_a_1><s_b_2><s_c_3>, "
        "<|video_begin|><s_a_1><s_b_2>, <s_a_1><s_b_2><s_c_4>./think"
        "</think><|video_begin|><s_a_7><s_b_8><s_c_9>"
    )
    assert extract_history_sids(prompt) == {("video", 1, 2, 3)}


def test_full_sid_equality_is_required_for_copy():
    history = {("video", 1, 2, 3)}
    assert ("video", 1, 2, 3) in history
    assert ("video", 1, 2, 4) not in history
    assert ("video", 1, 9, 3) not in history


@pytest.mark.parametrize(
    "sid,gold,history,expected",
    [
        (None, {("video", 1, 2, 3)}, set(), (-1.0, False, -1.0)),
        (("video", 1, 9, 9), {("video", 1, 2, 3)}, {("video", 1, 9, 9)}, (0.5, True, 0.0)),
        (("video", 1, 2, 9), {("video", 1, 2, 3)}, {("video", 1, 2, 9)}, (2.0, True, 0.0)),
        (("video", 1, 2, 3), {("video", 1, 2, 3)}, {("video", 1, 2, 3)}, (8.0, True, 8.0)),
        (("video", 1, 9, 8), {("video", 1, 2, 3)}, set(), (0.5, False, 0.5)),
        (("video", 1, 2, 8), {("video", 1, 2, 3)}, set(), (2.0, False, 2.0)),
        (("video", 1, 2, 3), {("video", 1, 2, 3)}, set(), (8.0, False, 8.0)),
    ],
)
def test_sid_reward_order(sid, gold, history, expected):
    assert shape_sid_reward(sid, gold, history) == expected


def test_copied_exact_gets_no_cot_credit():
    assert cot_candidate_contribution(8.0, True) == 0.0
    assert cot_candidate_contribution(8.0, False) == 8.0
    assert cot_candidate_contribution(2.0, False) == 2.0


@pytest.mark.parametrize(
    "index,stage,sid_bonus,cot_bonus",
    [
        (19, "sid_only", 0.25, 0.0),
        (20, "sid_and_cot", 0.25, 0.25),
        (39, "sid_and_cot", 0.25, 0.25),
        (40, "off", 0.0, 0.0),
    ],
)
def test_warmup_boundaries(index, stage, sid_bonus, cot_bonus):
    assert warmup_reward_contract(index) == (stage, sid_bonus, cot_bonus)
    sid = ("video", 99, 98, 97)
    gold = {("video", 1, 2, 3)}
    raw, copied, final = shape_sid_reward(sid, gold, set(), sid_bonus)
    assert (raw, copied, final) == (0.0, False, sid_bonus)
    assert cot_candidate_contribution(raw, copied, cot_bonus) == cot_bonus


def test_warmup_never_rewards_copies_and_preserves_exact():
    gold = {("video", 1, 2, 3)}
    copied_zero = ("video", 99, 98, 97)
    copied_exact = ("video", 1, 2, 3)
    assert shape_sid_reward(copied_zero, gold, {copied_zero}, 0.25) == (
        0.0, True, 0.0,
    )
    assert shape_sid_reward(copied_exact, gold, {copied_exact}, 0.25) == (
        8.0, True, 8.0,
    )
    assert cot_candidate_contribution(0.0, True, 0.25) == 0.0
    assert cot_candidate_contribution(8.0, True, 0.25) == 0.0


def _record(index):
    candidates = [
        ("video", 1, 2, 3), ("video", 1, 2, 9),
        ("video", 1, 9, 9), ("video", 2 + index, 3, 4),
        ("video", 5, 6, 7), ("video", 8, 9, 10),
        ("video", 11, 12, 13), None,
    ]
    return {
        "recommendation_group_id": "g", "target_domain": "video",
        "cot_ids": [100 + index], "cot_text": f"cot-{index}", "cot_length": 1,
        "candidate_ids": [[1, 2, 3] for _ in range(SID_G)],
        "candidate_texts": [f"candidate-{i}" for i in range(SID_G)],
        "candidate_sids": candidates, "gold_sids": [("video", 1, 2, 3)],
        "history_sids": [
            ("video", 1, 2, 3), ("video", 1, 2, 9), ("video", 1, 9, 9),
        ],
    }


def test_finalize_records_monitor_and_four_independent_g8():
    records = finalize_global_records(
        [_record(i) for i in range(COT_G)], fresh_rollout_index=40
    )
    groups = [row["sid_rewards"] for row in records]
    advantages = independent_official_advantages(groups)
    assert advantages.shape == (COT_G, SID_G)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(COT_G), atol=1e-6)
    row = records[0]
    required = {
        "cot_length", "history_sid_count", "gold_history_exact_overlap",
        "candidate_details", "copy_count", "copy_rate", "copy_A", "copy_AB",
        "copy_Exact", "noncopy_A", "noncopy_AB", "noncopy_Exact",
        "copy_positive_advantage_count", "copy_mean_advantage",
        "cot_reward", "cot_advantage", "noncopy_positive_reward_count",
        "fresh_rollout_index", "warmup_stage", "noncopy_zero_bonus",
        "noncopy_zero_count",
    }
    assert required.issubset(row)
    copied_exact = row["candidate_details"][0]
    assert copied_exact["raw_q_reward"] == 8.0
    assert copied_exact["final_sid_reward"] == 8.0
    assert row["cot_contributions"][0] == 0.0


def test_official_mask_is_exact_abc3():
    assert official_action_mask([[1, 2, 3] for _ in range(SID_G)]) == [
        [1, 1, 1] for _ in range(SID_G)
    ]
    with pytest.raises(ValueError):
        official_action_mask([[1, 2, 3, 4] for _ in range(SID_G)])


def test_loss_is_one_to_one():
    assert combine_losses(torch.tensor(2.0), torch.tensor(3.0)).item() == 5.0


def test_iteration2_reuses_fingerprint_and_rollout():
    assert assert_iteration_reuse("same", "same", 1, 7, 7)
    assert assert_iteration_reuse("same", "same", 2, 7, 7)
    with pytest.raises(RuntimeError):
        assert_iteration_reuse("same", "different", 2, 7, 7)
    with pytest.raises(RuntimeError):
        assert_iteration_reuse("same", "same", 2, 7, 8)


def test_iteration2_keeps_frozen_warmup_reward():
    records = finalize_global_records(
        [_record(i) for i in range(COT_G)], fresh_rollout_index=19
    )
    fingerprint = rollout_fingerprint(records)
    frozen = json.dumps(records, sort_keys=True)
    assert assert_iteration_reuse(fingerprint, rollout_fingerprint(records), 1, 4, 4)
    assert assert_iteration_reuse(fingerprint, rollout_fingerprint(records), 2, 4, 4)
    assert json.dumps(records, sort_keys=True) == frozen
    next_stage_records = finalize_global_records(
        [_record(i) for i in range(COT_G)], fresh_rollout_index=20
    )
    assert rollout_fingerprint(next_stage_records) != fingerprint


def test_sampler_audit_manifest_contract(plan):
    audit = audit_v5_sampler(plan["dataset"], object())
    required = {
        "selected_groups", "trained_groups", "dropped_groups",
        "think_rollouts", "nothink_rollouts",
        "unique_groups_per_global_rollout", "nothink_optimizer_steps",
        "official_g8_groups", "optimizer_steps",
    }
    assert required.issubset(audit)
    assert audit["official_g8_groups"] == 4
    assert audit["optimizer_steps"] == EXPECTED_STEPS


def test_probe_copy_anatomy_and_denominator():
    gold = {("video", 1, 2, 3)}
    history = {("video", 1, 2, 3), ("video", 1, 2, 9)}
    details = beam_copy_details(
        [("video", 1, 2, 3), ("video", 1, 2, 9), ("video", 1, 9, 9)],
        gold, history,
    )
    candidates = [{
        "reward": 8.0, "closed": True, "exact": 1, "ab": 0, "a": 0,
        "invalid": 0, "completion_length": 32,
        "beam_candidate_details": details,
    }]
    summary = official_probe_summary(candidates)
    assert summary["candidate_count"] == 1
    assert summary["beam_candidate_count"] == 3
    assert summary["copy_Exact"] == 1
    assert summary["copy_AB"] == 1
    assert summary["noncopy_A"] == 1


def test_heldout_video_probe_cadence_is_step0_every100_and_final():
    assert heldout_video_probe_due(0, "baseline")
    assert not heldout_video_probe_due(0, "interval")
    assert not heldout_video_probe_due(50, "interval")
    assert heldout_video_probe_due(100, "interval")
    assert heldout_video_probe_due(200, "interval")
    assert heldout_video_probe_due(1074, "final")
    assert heldout_video_probe_due(100, "final")


def test_beam_stats_restore_contract():
    class Model:
        pass
    model = Model()
    restore_beam_stats(model, False, None)
    assert not hasattr(model, "_beam_stats")
    original = {"kept": True}
    model._beam_stats = {"probe": True}
    restore_beam_stats(model, True, original)
    assert model._beam_stats is original
