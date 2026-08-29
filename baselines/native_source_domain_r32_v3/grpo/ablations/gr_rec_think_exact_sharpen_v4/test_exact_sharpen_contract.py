from pathlib import Path

import pytest
import torch

from . import run_exact_sharpen_train as runner
from .dual_probe import probe_summary
from .exact_sharpen_trainer import (
    COT_G, SID_G, branch_is_saturated, duplicate_penalties,
    finalize_global_records, independent_g8_advantages, population_advantages,
    shape_g8, strict_unique_cot_reward,
)


def _groups(a_plus):
    values = [0.5] * a_plus + [0.0] * (32 - a_plus)
    return [values[index:index + 8] for index in range(0, 32, 8)]


def _record(index, free_rewards=None, official_rewards=None):
    free_rewards = free_rewards or [0.0] * 8
    official_rewards = official_rewards or [0.0] * 8
    def sids(rewards):
        return [("video", index + 1, j + 1, 1) if value >= 0 else None
                for j, value in enumerate(rewards)]
    return {
        "recommendation_group_id": "g", "target_domain": "video",
        "cot_ids": [index], "free_candidate_ids": [[1, 2, 3, 4]] * 8,
        "official_candidate_ids": [[2, 3, 4]] * 8,
        "free_sids": sids(free_rewards), "official_sids": sids(official_rewards),
        "free_raw_rewards": list(free_rewards),
        "official_raw_rewards": list(official_rewards),
    }


def test_dataset_guard_exact_contract():
    result = runner.validate_dataset()
    assert result["rows"] == result["unique_groups"] == 611
    assert result["think_only"] and result["probe4_overlap"] == []
    assert result["sha256"] == runner.DATASET_SHA256


def test_run_plan_replaces_training_data_but_retains_probe4():
    args = runner.build_v4_parser().parse_args([
        "--run-id", "V4-PLAN-CPU-TEST", "--n-groups", "all", "--probe-groups", "4",
    ])
    plan = runner.prepare_v4_run_plan(args)
    assert len(plan["dataset"]) == 611 and plan["max_steps"] == 1222
    assert tuple(plan["probe_group_ids"]) == runner.FIXED_PROBE4_IDS
    assert plan["probe_train_overlap"] == []


def test_parent_checkpoint_and_sha_guard():
    assert runner.validate_parent_adapter() == {
        "path": str(runner.PARENT_ADAPTER),
        "adapter_sha256": runner.PARENT_ADAPTER_SHA256,
    }


def test_saturation_is_strictly_more_than_8_of_32():
    assert branch_is_saturated(_groups(8)) is False
    assert branch_is_saturated(_groups(9)) is True


def test_saturation_requires_exact_branch_shape():
    with pytest.raises(ValueError):
        branch_is_saturated([[0.5] * 8] * 3)


def test_free_and_official_saturation_are_independent():
    records = [_record(i, _groups(9)[i], _groups(8)[i]) for i in range(4)]
    finalize_global_records(records)
    assert all(row["free_saturated"] for row in records)
    assert not any(row["official_saturated"] for row in records)
    assert records[0]["free_a_plus_count"] == 9
    assert records[0]["official_a_plus_count"] == 8


def test_saturated_reward_values():
    sids = [("video", 1, i, 1) for i in range(8)]
    shaped, _ = shape_g8(sids, [-1, -0.25, 0, 0.5, 2, 8, 0, 0], True)
    assert shaped[:6] == [-1.0, -0.25, 0.0, 0.0, 1.5, 7.5]


def test_duplicate_penalty_thresholds_apply_within_g8():
    x = ("video", 1, 2, 3)
    y = ("video", 2, 3, 4)
    penalties = duplicate_penalties([x, x, x, y, y, y, y, None],
                                    [0.5, 0.5, 0.5, 2, 2, 2, 2, -1])
    assert penalties == [-0.5] * 3 + [-1.0] * 4 + [0.0]


def test_exact_duplicates_are_never_penalized():
    sid = ("video", 1, 2, 3)
    assert duplicate_penalties([sid] * 8, [8.0] * 8) == [0.0] * 8


def test_invalid_has_no_sid_identity_penalty():
    assert duplicate_penalties([None] * 8, [-1.0] * 8) == [0.0] * 8


def test_strict_unique_coverage_deduplicates_and_covers_prefixes():
    exact = ("video", 1, 2, 3)
    covered_ab = ("video", 1, 2, 9)
    uncovered_ab = ("video", 1, 4, 9)
    covered_a = ("video", 1, 8, 9)
    uncovered_a = ("video", 7, 8, 9)
    sids = [exact, exact, covered_ab, uncovered_ab, uncovered_ab,
            covered_a, uncovered_a, uncovered_a]
    raw = [8, 8, 2, 2, 2, 0.5, 0.5, 0.5]
    reward, counts = strict_unique_cot_reward(sids, raw, False)
    assert reward == 8 + 2 + 0.5
    assert counts == {"unique_exact": 1, "unique_uncovered_ab": 1,
                      "unique_uncovered_a": 1}


def test_saturated_cot_reward_drops_a_and_sharpens_exact_ab():
    sids = [("video", 1, 2, 3), ("video", 1, 4, 5),
            ("video", 7, 8, 9)] + [None] * 5
    reward, counts = strict_unique_cot_reward(sids, [8, 2, 0.5] + [-1] * 5, True)
    assert reward == 9.0
    assert counts["unique_uncovered_a"] == 0


def test_official_does_not_enter_cot_reward():
    records = [_record(i, [0] * 8, [8] * 8) for i in range(4)]
    finalize_global_records(records)
    assert [row["cot_reward"] for row in records] == [0.0] * 4
    assert all(row["official_saturated"] for row in records)


def test_eight_independent_g8_advantages():
    free = [[0] * 8, [8] * 8, [0, 8] * 4, [2, 0] * 4]
    official = [[-1, 0, 0.5, 2, 8, 0, 0.5, 2]] * 4
    f, o = independent_g8_advantages(free), independent_g8_advantages(official)
    assert f.shape == o.shape == (COT_G, SID_G)
    assert f[0].tolist() == f[1].tolist() == [0.0] * 8
    assert not torch.allclose(f.flatten(), population_advantages(sum(free, [])))


def test_free_and_official_masks_are_four_and_three_tokens():
    source = Path(__file__).with_name("exact_sharpen_trainer.py").read_text()
    assert "FREE_SID_TOKENS = 4" in source
    assert "OFFICIAL_SID_TOKENS = 3" in source
    assert 'action = [[1] * OFFICIAL_SID_TOKENS' in source
    assert 'row[offset + start:offset + end] = [1] * FREE_SID_TOKENS' in source


def test_loss_weights_are_frozen_one_half_half():
    source = Path(__file__).with_name("exact_sharpen_trainer.py").read_text()
    assert "self.lambda_cot, self.lambda_free, self.lambda_official = 1.0, 0.5, 0.5" in source
    assert "self.lambda_free * losses[\"free\"]" in source
    assert "self.lambda_official * losses[\"official\"]" in source


def test_iteration2_reuses_cached_rollout_without_resampling():
    source = Path(__file__).with_name("exact_sharpen_trainer.py").read_text()
    assert "self._v4_policy_epoch >= 2" in source
    assert "self._v4_policy_epoch += 1" in source
    assert source.count("self.model.generate(") == 1
    assert 'free_sample_calls' in source and 'official_sample_calls' in source


def test_official_sampling_is_stochastic_exact_abc3():
    source = Path(__file__).with_name("exact_sharpen_trainer.py").read_text()
    for value in ("do_sample=True", "temperature=1.0", "top_p=1.0", "top_k=0"):
        assert value in source
    assert "max_tokens = OFFICIAL_SID_TOKENS if official" in source
    assert "parse_strict_abc3_ids" in source


def test_free_and_official_probe_routes_are_not_confused():
    source = Path(__file__).with_name("dual_probe.py").read_text()
    assert '"probe_free"' in source and '"probe_official"' in source
    assert "self._free_sample8" in source
    assert "self.beam32_fn" in source
    assert '"fixed_domain_prefix": False' in source
    assert '"fixed_domain_prefix": True' in source


def test_probe_summary_reports_hierarchical_counts():
    free = probe_summary([{"reward": 0.5}, {"reward": 2.0}, {"reward": 8.0}, {"reward": -1.0}])
    assert (free["a_count"], free["ab_count"], free["exact_count"], free["invalid_count"]) == (1, 1, 1, 1)
    official = probe_summary([{"reward": 10.0, "a": 1, "ab": 2, "exact": 3, "invalid": 4}])
    assert (official["a_count"], official["ab_count"], official["exact_count"], official["invalid_count"]) == (1, 2, 3, 4)


def test_config_and_optimizer_are_frozen():
    cfg = runner.v4_config_kwargs()
    assert cfg["num_iterations"] == 2 and cfg["weight_decay"] == 0.0
    assert cfg["lr_scheduler_type"] == "constant"


def test_no_gpu_preflight_is_added():
    assert not Path(__file__).with_name("gpu_preflight.py").exists()


def test_launcher_declares_four_gpu_formal_but_is_not_a_test_launch():
    source = Path(__file__).with_name("launch_exact_sharpen_train.sh").read_text()
    assert "--nproc_per_node=4" in source
    assert "gpu_preflight" not in source
