from types import SimpleNamespace

import pytest
import torch

from grpo_probe import probe_group_batches

from .official_probe import beam_copy_details, official_probe_summary
from .official_finegrained_trainer import (
    COT_G, SID_G, assert_iteration_reuse, audit_v6_sampler, combine_losses,
    extract_history_sids, finalize_global_records, finegrained_g8_signal,
    independent_finegrained_signals, official_action_mask, reward_hits,
    rollout_fingerprint, ThinkOfficialFineGrainedTrainer,
)
from .run_official_finegrained_train import (
    EXPECTED_AFTER_PROBE4, EXPECTED_AFTER_PROBE8, EXPECTED_DOMAIN_GROUPS,
    EXPECTED_RAW_GROUPS, EXPECTED_STEPS, EXPECTED_TRAIN_GROUPS,
    EXTRA_PROBE4_IDS, FINAL_GROUP_ID_LIST_SHA256, FINAL_TRAIN_CANONICAL_SHA256,
    FIXED_PROBE4_IDS, FIXED_PROBE8_IDS, FORBIDDEN_POSITIVE_DATASET,
    HISTORY_COPY_PROBE4_IDS, HISTORY_COPY_PROBE4_OVERLAP, SOURCE_DATASET,
    SOURCE_DATASET_SHA256, build_v6_parser, prepare_v6_run_plan,
    validate_history_copy_probe, validate_source_dataset,
)


def _args():
    values = [
        "--run-id", "V6-A-CPU-TEST", "--n-groups", "all",
        "--max-steps", "2", "--probe-groups", "8",
        "--secondary-probe-suite", "history_copy_exact",
    ]
    for gid in FIXED_PROBE8_IDS:
        values.extend(["--probe-group-id", gid])
    for gid in HISTORY_COPY_PROBE4_IDS:
        values.extend(["--secondary-probe-group-id", gid])
    return build_v6_parser().parse_args(values)


@pytest.fixture(scope="module")
def plan():
    return prepare_v6_run_plan(_args())


def _record(candidate_sids, *, history_sids=(), rank=0):
    return {
        "recommendation_group_id": "g", "target_domain": "video",
        "cot_ids": [100 + rank, 200], "cot_text": f"cot-{rank}",
        "cot_length": 2, "candidate_ids": [[1, 2, 3] for _ in range(SID_G)],
        "candidate_texts": [f"candidate-{index}" for index in range(SID_G)],
        "candidate_sids": list(candidate_sids),
        "gold_sids": [("video", 1, 2, 3)],
        "history_sids": list(history_sids),
    }


def test_source_is_exact_v3_dataset_not_positive_dataset():
    audit = validate_source_dataset()
    assert audit == {
        "path": str(SOURCE_DATASET), "baseline_data_path": str(SOURCE_DATASET),
        "sha256": SOURCE_DATASET_SHA256,
        "rows": 3098, "business_groups": EXPECTED_RAW_GROUPS,
    }
    assert SOURCE_DATASET != FORBIDDEN_POSITIVE_DATASET
    assert "positive_groups_1946" not in str(SOURCE_DATASET)


def test_dataset_guards_and_domain_distribution(plan):
    guard = plan["dataset_guard"]
    assert guard["source_business_groups"] == EXPECTED_RAW_GROUPS
    assert guard["groups_after_probe4"] == EXPECTED_AFTER_PROBE4
    assert guard["groups_after_probe8"] == EXPECTED_AFTER_PROBE8
    assert guard["train_rows"] == guard["unique_groups"] == EXPECTED_TRAIN_GROUPS
    assert guard["domain_group_counts"] == EXPECTED_DOMAIN_GROUPS
    assert guard["canonical_train_sha256"] == FINAL_TRAIN_CANONICAL_SHA256
    assert guard["group_id_list_sha256"] == FINAL_GROUP_ID_LIST_SHA256
    assert guard["heldout_probe4_overlap"] == 0
    assert guard["probe8_training_overlap"] == 4
    assert plan["max_steps"] == 2 and EXPECTED_STEPS == 3090


def test_dataset_is_think_only_unique_and_only_original_probe4_is_held_out(plan):
    rows = list(plan["dataset"])
    gids = {row["recommendation_group_id"] for row in rows}
    assert len(rows) == len(gids) == EXPECTED_TRAIN_GROUPS
    assert {row["route"] for row in rows} == {"think"}
    assert {row["target_domain"] for row in rows} == set(EXPECTED_DOMAIN_GROUPS)
    assert not gids.intersection(FIXED_PROBE4_IDS)
    assert gids.intersection(EXTRA_PROBE4_IDS) == set(EXTRA_PROBE4_IDS)


def test_primary_probe8_is_two_per_domain_and_history_probe4_is_preserved(plan):
    domains = [
        plan["probe_records"][gid]["think"]["target_domain"]
        for gid in FIXED_PROBE8_IDS
    ]
    assert domains == ["video", "living", "prod", "ad"] * 2
    assert probe_group_batches(FIXED_PROBE8_IDS) == [
        list(FIXED_PROBE8_IDS[:4]), list(FIXED_PROBE8_IDS[4:]),
    ]
    assert tuple(plan["secondary_probe_group_ids"]) == HISTORY_COPY_PROBE4_IDS
    assert validate_history_copy_probe(plan["secondary_probe_records"]) == (
        HISTORY_COPY_PROBE4_OVERLAP
    )


def test_history_parser_requires_complete_same_domain_sid():
    prompt = (
        "instruction <|video_begin|><s_a_99><s_b_99><s_c_99>\n"
        "history <|video_begin|><s_a_1><s_b_2><s_c_3> "
        "partial <|video_begin|><s_a_4><s_b_5> "
        "other <|ad_begin|><s_a_6><s_b_7><s_c_8>\n/think"
        "<|video_begin|><s_a_10><s_b_11><s_c_12>"
    )
    assert extract_history_sids(prompt, "video") == {("video", 1, 2, 3)}
    assert extract_history_sids(prompt, "ad") == {("ad", 6, 7, 8)}


def test_reward_hits_are_cumulative():
    assert reward_hits(-1) == reward_hits(-0.25) == reward_hits(0) == (0, 0, 0)
    assert reward_hits(0.5) == (1, 0, 0)
    assert reward_hits(2) == (1, 1, 0)
    assert reward_hits(8) == (1, 1, 1)


def test_one_A_produces_nonzero_A_advantage():
    signal = finegrained_g8_signal([0.5] + [0] * 7)
    assert signal["advantages"][0, 0] > 0
    assert torch.all(signal["advantages"][1:, 0] < 0)
    assert signal["zero_std"].tolist() == [False, True, True]


def test_one_AB_produces_nonzero_B_advantage():
    signal = finegrained_g8_signal([2] + [0] * 7)
    assert signal["advantages"][0, 1] > 0
    assert torch.all(signal["advantages"][1:, 1] < 0)
    assert signal["masks"][:, 1].tolist() == [1] + [0] * 7


def test_AB_and_A_only_frontier_gate_values():
    signal = finegrained_g8_signal([2, 0.5] + [0] * 6)
    assert signal["advantages"][0, 1] > 0
    assert signal["advantages"][1, 1] < 0
    assert signal["masks"][:, 1].tolist() == [1, 1] + [0] * 6
    assert signal["masks"][2, 1] == 0


def test_Exact_and_AB_only_frontier_gate_values():
    signal = finegrained_g8_signal([8, 2] + [0] * 6)
    assert signal["advantages"][0, 2] > 0
    assert signal["advantages"][1, 2] < 0
    assert signal["masks"][:, 2].tolist() == [1, 1] + [0] * 6
    assert signal["masks"][2, 2] == 0


def test_no_AB_or_Exact_has_zero_std_and_zero_advantage():
    signal = finegrained_g8_signal([0.5, 0.5] + [0] * 6)
    assert signal["zero_std"].tolist() == [False, True, True]
    assert torch.count_nonzero(signal["advantages"][:, 1:]) == 0


def test_four_G8_are_normalized_independently_never_G32():
    groups = [[0.5] + [0] * 7, [2] + [0] * 7, [8] + [0] * 7, [0] * 8]
    signal = independent_finegrained_signals(groups)
    assert signal["advantages"].shape == (COT_G, SID_G, 3)
    assert signal["zero_std"].tolist() == [
        [False, True, True], [False, False, True],
        [False, False, False], [True, True, True],
    ]


def test_official_mask_contract_is_three_tokens_and_loss_is_one_to_one():
    assert official_action_mask([[1, 2, 3] for _ in range(SID_G)]) == (
        [[1, 1, 1] for _ in range(SID_G)]
    )
    with pytest.raises(ValueError, match="ABC3"):
        official_action_mask([[1, 2] for _ in range(SID_G)])
    assert combine_losses(torch.tensor(2.0), torch.tensor(3.0)).item() == 5.0


def test_sid_ppo_averages_only_active_token_advantages():
    current = torch.zeros((2, 3), requires_grad=True)
    fake = SimpleNamespace(
        epsilon_low=0.2, epsilon_high=0.2,
        current_gradient_accumulation_steps=1,
        _get_per_token_logps_and_entropies=lambda *args, **kwargs: (current, None),
    )
    action = torch.tensor([[1, 1, 0], [1, 0, 0]])
    advantages = torch.tensor([[2.0, 4.0, 99.0], [-2.0, -99.0, -99.0]])
    loss, stats = ThinkOfficialFineGrainedTrainer._branch_loss(
        fake, None, None, None, action, torch.zeros_like(current),
        advantages, "sid",
    )
    assert loss.item() == pytest.approx(-(2.0 + 4.0 - 2.0) / 3.0)
    assert stats["sid_action_tokens"] == 3
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 2] == 0 and current.grad[1, 1] == 0


def test_cot_reward_is_exact_sum_of_eight_raw_q_rewards():
    candidates = [
        ("video", 1, 2, 3), ("video", 1, 2, 4),
        ("video", 1, 5, 6), ("video", 9, 9, 9),
        ("ad", 1, 2, 3), None, ("video", 9, 9, 8), ("video", 0, 0, 0),
    ]
    records = finalize_global_records(
        [_record(candidates, rank=rank) for rank in range(COT_G)], 0
    )
    assert records[0]["cot_reward"] == pytest.approx(
        sum(records[0]["raw_q_rewards"])
    )
    assert records[0]["cot_contributions"] == records[0]["raw_q_rewards"]


def test_history_copy_changes_monitor_only_not_training_signal_or_fingerprint():
    candidates = [("video", 1, 2, 3), ("video", 1, 2, 4)] + [
        ("video", 9, 9, index) for index in range(6)
    ]
    plain = finalize_global_records(
        [_record(candidates, rank=rank) for rank in range(COT_G)], 0
    )
    copied = finalize_global_records([
        _record(candidates, history_sids=candidates, rank=rank)
        for rank in range(COT_G)
    ], 0)
    for left, right in zip(plain, copied):
        assert left["raw_q_rewards"] == right["raw_q_rewards"]
        assert left["token_advantages"] == right["token_advantages"]
        assert left["token_masks"] == right["token_masks"]
        assert left["cot_reward"] == right["cot_reward"]
    assert rollout_fingerprint(plain) == rollout_fingerprint(copied)
    assert plain[0]["copy_count"] == 0 and copied[0]["copy_count"] == 8


def test_iteration2_reuses_fingerprint_reward_masks_and_old_rollout():
    assert assert_iteration_reuse("same", "same", 1, 4, 8)
    assert assert_iteration_reuse("same", "same", 2, 8, 8)
    with pytest.raises(RuntimeError, match="resampled"):
        assert_iteration_reuse("same", "same", 2, 8, 9)
    with pytest.raises(RuntimeError, match="fingerprint"):
        assert_iteration_reuse("old", "new", 2, 8, 8)


def test_sampler_audit_has_baseline_manifest_contract(plan):
    audit = audit_v6_sampler(plan["dataset"], SimpleNamespace())
    assert audit["think_rollouts"] == EXPECTED_TRAIN_GROUPS
    assert audit["nothink_rollouts"] == 0
    assert audit["dropped_groups"] == 0
    assert audit["unique_groups_per_global_rollout"] == 1
    assert audit["nothink_optimizer_steps"] == 0
    assert audit["official_g8_groups"] == 4


def test_probe_uses_production_reward_and_copy_is_diagnostic_only():
    details = beam_copy_details(
        [("video", 1, 2, 3), ("video", 1, 2, 4)],
        {("video", 1, 2, 3)}, {("video", 1, 2, 3)},
    )
    assert [row["is_history_copy"] for row in details] == [True, False]
    assert [row["raw_q_reward"] for row in details] == [8.0, 2.0]
    candidates = [
        {"reward": 8.0, "closed": True, "a": 0, "ab": 0, "exact": 1,
         "invalid": 0, "completion_length": 17,
         "beam_candidate_details": details,
         "history_sid_count": 1, "gold_history_exact_overlap": 1},
        {"reward": 2.0, "closed": True, "a": 0, "ab": 1, "exact": 0,
         "invalid": 0, "completion_length": 17,
         "beam_candidate_details": []},
    ]
    summary = official_probe_summary(candidates)
    assert summary["reward_mean"] == 5.0
    assert summary["history_copy_rate"] == 0.5
    assert summary["cot_length_mean"] == 17
