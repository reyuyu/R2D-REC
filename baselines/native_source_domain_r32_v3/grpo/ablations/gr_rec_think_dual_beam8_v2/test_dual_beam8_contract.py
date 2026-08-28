from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from . import run_dual_beam8_train as runner
from .dual_beam8_trainer import (
    COT_G, SID_G, FinalStepSaveCallback, ThinkG4SingleGroupSampler,
    assert_one_global_group, audit_dual_sampler, independent_sid_advantages,
    population_advantages, rollout_fingerprint,
)


@pytest.fixture(scope="module")
def plan():
    args = runner.baseline_runner.build_arg_parser().parse_args([
        "--run-id", "DUAL-BEAM8-CPU-CONTRACT", "--n-groups", "all",
        "--probe-groups", "4", "--probe-every-steps", "200",
        "--save-steps", "250", "--save-total-limit", "8",
    ])
    return runner.prepare_dual_run_plan(args)


def test_probe4_exact_ids(plan):
    assert tuple(plan["probe_group_ids"]) == runner.FIXED_PROBE4_IDS


def test_probe4_train_zero_overlap(plan):
    assert not set(plan["probe_group_ids"]) & set(plan["dataset"]["recommendation_group_id"])


def test_runtime_topology_1549_1545_3090(plan):
    assert plan["raw_groups"] == 1549
    assert len(plan["dataset"]) == 1545
    assert plan["audit"]["optimizer_steps"] == 3090


def test_think_only_one_row_per_group(plan):
    assert set(plan["dataset"]["route"]) == {"think"}
    gids = plan["dataset"]["recommendation_group_id"]
    assert len(gids) == len(set(gids))


def test_g4_sampler_rank_alignment():
    sampler = ThinkG4SingleGroupSampler([0, 1], repeat_count=2)
    assert list(sampler) == [0] * 8 + [1] * 8


def test_g4_sampler_hard_num_iterations():
    with pytest.raises(ValueError):
        ThinkG4SingleGroupSampler([0], repeat_count=1)


def test_g4_single_group_guard():
    assert_one_global_group(["x"] * 4)
    with pytest.raises(RuntimeError):
        assert_one_global_group(["x", "x", "y", "x"])


def test_population_advantage_hand_calculation():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    expected = (values - values.mean()) / (values.std(correction=0) + 1e-4)
    assert torch.allclose(population_advantages(values), expected)


def test_zero_std_means_zero_advantage():
    assert population_advantages([2.0] * 8).tolist() == [0.0] * 8


def test_four_sid_g8_are_independent():
    groups = [[-1, 0, 0.5, 2, 8, 0, 0.5, 2], [8] * 8,
              [0, 0, 0, 0, 0.5, 0.5, 2, 8], [-0.25, 0, 0.5, 2, 8, 8, 2, 0]]
    result = independent_sid_advantages(groups)
    assert result.shape == (4, 8)
    assert result[1].tolist() == [0.0] * 8
    for index, group in enumerate(groups):
        assert torch.allclose(result[index], population_advantages(group))


def test_wrong_g32_normalization_is_detected():
    groups = [[0] * 8, [8] * 8, [0, 8] * 4, [2] * 8]
    correct = independent_sid_advantages(groups).flatten()
    wrong = population_advantages([value for group in groups for value in group])
    assert not torch.allclose(correct, wrong)


def test_invalid_sid_topology_rejected():
    with pytest.raises(ValueError):
        independent_sid_advantages([[0] * 8] * 3)


def test_sampler_audit_fresh_rollout_semantics():
    rows = [{"route": "think", "recommendation_group_id": "a"},
            {"route": "think", "recommendation_group_id": "b"}]
    audit = audit_dual_sampler(rows, ThinkG4SingleGroupSampler(rows))
    assert audit["fresh_rollout_count"] == 2
    assert audit["optimizer_steps"] == 4


def test_no_reroll_in_source():
    source = Path(__file__).with_name("dual_beam8_trainer.py").read_text()
    assert "resample_decision" not in source and "zero_std_reroll" not in source


def test_loss_weights_are_one_to_one():
    source = Path(__file__).with_name("dual_beam8_trainer.py").read_text()
    assert "self.lambda_cot, self.lambda_sid = 1.0, 1.0" in source


def test_iteration2_reuses_cache_without_rebeam():
    source = Path(__file__).with_name("dual_beam8_trainer.py").read_text()
    assert "_dual_policy_epoch >= 2" in source
    assert source.count("generate_batch(") == 1


def test_rollout_fingerprint_is_stable():
    row = {"recommendation_group_id": "x", "cot_ids": [1],
           "beam_candidate_ids": [[1, 2, 3]] * 8, "cot_reward": 1,
           "sid_rewards": [0] * 8}
    assert rollout_fingerprint([row] * 4) == rollout_fingerprint([row] * 4)


def test_parent_sha_fail_closed(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"wrong")
    monkeypatch.setattr(runner, "PARENT_ADAPTER", tmp_path)
    with pytest.raises(RuntimeError, match="PARENT_SHA_MISMATCH"):
        runner.validate_parent_adapter()


def test_runtime_import_provenance_passes():
    assert runner.assert_runtime_import_provenance()["runtime_import_provenance"] == "PASS"


def test_runtime_import_provenance_stale_path_fails():
    from ablations.gr_rec_think_suffix_sid_v1.runtime_import_provenance import evaluate_runtime_import_provenance
    fake = SimpleNamespace(__file__="/stale/scripts/fake.py", RecGRPOTrainer=object())
    result = evaluate_runtime_import_provenance(fake, fake, fake, fake,
                                                expected_scripts_dir=Path("/current/scripts"))
    assert result["runtime_import_provenance"] == "FAIL"


def test_training_beam8_probe_beam32_separation():
    source = Path(__file__).with_name("run_dual_beam8_train.py").read_text()
    assert '"num_beams": SID_G' in source
    assert "production-shaped Probe4/Beam32" in source


def test_launcher_uses_four_gpus():
    source = Path(__file__).with_name("launch_dual_beam8_train.sh").read_text()
    assert "--nproc_per_node=4" in source


def test_launcher_uses_nccl_loopback():
    source = Path(__file__).with_name("launch_dual_beam8_train.sh").read_text()
    assert 'NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"' in source


def test_final_step_explicit_save():
    callback = FinalStepSaveCallback()
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(SimpleNamespace(max_steps=3090),
                         SimpleNamespace(global_step=3090), control)
    assert control.should_save


def test_nonfinal_step_not_forced_to_save():
    callback = FinalStepSaveCallback()
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(SimpleNamespace(max_steps=3090),
                         SimpleNamespace(global_step=3089), control)
    assert not control.should_save


def test_config_is_frozen():
    cfg = runner.dual_config_kwargs()
    assert cfg["num_iterations"] == 2 and cfg["beta"] == 0.0
    assert cfg["epsilon"] == 0.2 and cfg["weight_decay"] == 0.0
    assert cfg["lr_scheduler_type"] == "constant"


def test_sid_old_logp_is_full_forward_not_generate_scores():
    source = Path(__file__).with_name("dual_beam8_trainer.py").read_text()
    assert "SID_OLD_LOGP_CONTRACT_FAILED" in source
    assert "generate.scores" not in source


def test_action_boundaries_are_explicit():
    source = Path(__file__).with_name("dual_beam8_trainer.py").read_text()
    assert 'inputs["completion_mask"]' in source
    assert "torch.ones((SID_G, 3)" in source


def test_preflight_module_cold_import():
    from . import gpu_preflight
    assert gpu_preflight.COT_G == 4 and gpu_preflight.SID_G == 8


def test_domain_prefix_is_context_not_sid_action():
    source = Path(__file__).with_name("dual_beam8_trainer.py").read_text()
    assert 'list(record["beam_context_ids"]) + list(candidate)' in source
    assert "action_mask.size(1)" in source
