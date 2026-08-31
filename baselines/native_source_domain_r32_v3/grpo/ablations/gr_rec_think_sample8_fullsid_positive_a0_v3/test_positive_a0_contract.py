import hashlib
import json
from pathlib import Path

import pytest
import torch

from .a0_reward import q_reward_without_a_only, remove_a_only_reward
from . import run_positive_a0_train as runner
from ablations.gr_rec_think_sample8_fullsid_v3 import sample8_fullsid_trainer as v3


def _fixture_rows():
    rows = []
    for domain, count in runner.EXPECTED_DOMAIN_COUNTS.items():
        for index in range(count):
            rows.append({
                "prompt": f"prompt-{domain}-{index}",
                "route": "think",
                "target_domain": domain,
                "recommendation_group_id": f"{domain}-{index}",
                "all_gold_sids": [],
            })
    return rows


def _write_jsonl(path, rows):
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_a_only_reward_changes():
    original = [-1, -0.25, 0, 0.5, 2, 8]
    assert [remove_a_only_reward(value) for value in original] == [
        -1, -0.25, 0, 0, 2, 8,
    ]


def test_ab_and_exact_do_not_lose_their_a_prefix_reward():
    assert remove_a_only_reward(2.0) == 2.0
    assert remove_a_only_reward(8.0) == 8.0


def test_real_sid_match_classes_apply_a0_only_to_a_only():
    gold = {("video", 1, 2, 3)}
    assert q_reward_without_a_only(None, gold) == -1.0
    assert q_reward_without_a_only(("ad", 1, 2, 3), gold) == -0.25
    assert q_reward_without_a_only(("video", 9, 9, 9), gold) == 0.0
    assert q_reward_without_a_only(("video", 1, 9, 9), gold) == 0.0
    assert q_reward_without_a_only(("video", 1, 2, 9), gold) == 2.0
    assert q_reward_without_a_only(("video", 1, 2, 3), gold) == 8.0


def test_a0_reward_changes_sid_and_cot_advantages():
    rewards = [-1, -0.25, 0, 0.5, 2, 8, 0.5, 0]
    shaped = [remove_a_only_reward(value) for value in rewards]
    assert v3.cot_reward_from_sid_rewards(shaped) == 8.75
    expected = v3.population_advantages(shaped)
    result = v3.independent_sid_advantages([shaped] * 4)
    assert result.shape == (4, 8)
    assert torch.allclose(result[0], expected)


def test_dataset_guard_611_think_unique_four_domains_no_probe_overlap(tmp_path):
    path = tmp_path / "train.jsonl"
    sha = _write_jsonl(path, _fixture_rows())
    audit = runner.validate_dataset(path, expected_sha=sha)
    assert audit["rows"] == 611
    assert audit["unique_groups"] == 611
    assert audit["think_only"] is True
    assert audit["domain_counts"] == runner.EXPECTED_DOMAIN_COUNTS
    assert audit["probe4_overlap"] == []


def test_dataset_guard_rejects_probe_overlap(tmp_path):
    rows = _fixture_rows()
    rows[0]["recommendation_group_id"] = runner.FIXED_PROBE4_IDS[0]
    path = tmp_path / "train.jsonl"
    sha = _write_jsonl(path, rows)
    with pytest.raises(RuntimeError, match="PROBE4_OVERLAP"):
        runner.validate_dataset(path, expected_sha=sha)


def test_dataset_guard_rejects_non_think(tmp_path):
    rows = _fixture_rows()
    rows[0]["route"] = "nothink"
    path = tmp_path / "train.jsonl"
    sha = _write_jsonl(path, rows)
    with pytest.raises(RuntimeError, match="NOT_THINK_ONLY"):
        runner.validate_dataset(path, expected_sha=sha)


def test_parent_contract_is_recorded_best_v3_checkpoint():
    assert runner.PARENT_ADAPTER.name == "checkpoint-250"
    assert runner.PARENT_ADAPTER_SHA256 == (
        "64e1a85500b68f48d1996e6a215ea7a0bfe0dfdb925d6d53a86579ea23650be5"
    )
    assert runner.PARENT_RECORDED_EXTERNAL_SCORE == 1.3579


def test_positive_dataset_contract_and_full_epoch_steps():
    assert runner.DATASET.name == "train.jsonl"
    assert "positive_groups_1946_20260829" in str(runner.DATASET)
    assert runner.DATASET_SHA256 == (
        "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"
    )
    assert runner.EXPECTED_ROWS == 611
    assert runner.EXPECTED_STEPS == 1222


def test_v3_framework_objects_are_reused_not_copied():
    source = Path(runner.__file__).read_text()
    assert "ThinkSample8FullSIDTrainer" in source
    assert "ThinkG4SingleGroupSampler" in source
    assert "q_reward_without_a_only" in source
    assert not Path(runner.__file__).with_name("sample8_fullsid_trainer.py").exists()


def test_v3_loss_sampling_and_cache_contract_remain_unchanged():
    source = Path(v3.__file__).read_text()
    assert "self.lambda_cot, self.lambda_sid = 1.0, 1.0" in source
    assert "_sample_policy_epoch >= 2" in source
    assert source.count("self.model.generate(") == 1
    assert "num_return_sequences=SID_G" in source
    assert "min_new_tokens=FULL_SID_TOKENS" in source
    assert "max_new_tokens=SAMPLE_MAX_NEW_TOKENS" in source


def test_config_and_checkpoint_defaults_are_frozen():
    cfg = runner.positive_a0_config_kwargs()
    assert cfg["num_iterations"] == 2
    assert cfg["weight_decay"] == 0.0
    assert cfg["lr_scheduler_type"] == "constant"
    args = runner.build_positive_a0_parser().parse_args(["--run-id", "CPU-CONTRACT"])
    assert args.save_steps == 50
    assert args.probe_every_steps == 50
    assert args.save_total_limit == 64


def test_resume_is_forbidden_to_guarantee_fresh_adamw(monkeypatch):
    args = runner.build_positive_a0_parser().parse_args([
        "--run-id", "CPU-CONTRACT",
        "--resume-from-checkpoint", "/tmp/other/checkpoint-100",
    ])
    with pytest.raises(ValueError, match="fresh-start"):
        runner.prepare_positive_a0_run_plan(args)


def test_partial_dataset_selection_is_forbidden():
    args = runner.build_positive_a0_parser().parse_args([
        "--run-id", "CPU-CONTRACT", "--n-groups", "100",
    ])
    with pytest.raises(ValueError, match="n-groups all"):
        runner.prepare_positive_a0_run_plan(args)


def test_launcher_is_four_gpu_but_not_auto_executed():
    source = Path(runner.__file__).with_name("launch_positive_a0_train.sh").read_text()
    assert "--nproc_per_node=4" in source
    assert "--max-steps 1222" in source
    assert "--save-steps 50" in source
    assert "--probe-every-steps 50" in source
