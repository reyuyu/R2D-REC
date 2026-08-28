from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from . import run_sample8_fullsid_train as runner
from .sample8_fullsid_trainer import (
    COT_G, SID_G, SAMPLE_MAX_NEW_TOKENS, FinalStepSaveCallback, ThinkG4SingleGroupSampler,
    assert_one_global_group, audit_sample8_sampler, independent_sid_advantages,
    cot_reward_from_sid_rewards, parse_full_sid_ids, scan_full_sid_ids, population_advantages,
    rollout_fingerprint,
)


@pytest.fixture(scope="module")
def plan():
    args = runner.build_sample8_arg_parser().parse_args([
        "--run-id", "SAMPLE8-FULLSID-CPU-CONTRACT", "--n-groups", "all",
        "--probe-groups", "4",
    ])
    return runner.prepare_sample8_run_plan(args)


def test_formal_checkpoint_probe_defaults_are_aligned_on_root():
    args = runner.build_sample8_arg_parser().parse_args([
        "--run-id", "SAMPLE8-FULLSID-CHECKPOINT-CONTRACT",
    ])
    assert args.output_dir == "/root/GRPO-checkpoints"
    assert args.save_steps == 50
    assert args.probe_every_steps == args.save_steps
    assert args.save_total_limit == 64


def test_manifest_preserves_probe_schedule():
    class FakeWriter:
        rank = 0

        def write_manifest(self, payload):
            self.payload = payload

    writer = FakeWriter()
    wrapped = runner.Sample8ManifestWriter(writer)
    wrapped.write_manifest({"fixed_probe": {"seed": 20260818, "every_steps": 50}})
    assert writer.payload["fixed_probe"]["every_steps"] == 50
    assert writer.payload["fixed_probe"]["seed"] == 20260818
    assert writer.payload["fixed_probe"]["group_ids"] == list(runner.FIXED_PROBE4_IDS)


def test_trusted_resume_checkpoint_guard(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoint-250"
    checkpoint.mkdir(parents=True)
    required = [
        "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
        "scheduler.pt", "training_args.bin",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    ]
    for name in required:
        (checkpoint / name).write_bytes(b"trusted-test-fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 250}')
    audit = runner.validate_trusted_resume_checkpoint(checkpoint, output_root=tmp_path)
    assert audit == {
        "path": str(checkpoint.resolve()),
        "step": 250,
        "trusted_local_checkpoint": True,
    }


def test_trusted_resume_rejects_path_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside" / "checkpoint-250"
    outside.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="UNTRUSTED_RESUME_PATH"):
        runner.validate_trusted_resume_checkpoint(outside, output_root=root)


def test_trusted_resume_keeps_weights_only_numpy_allowlist(monkeypatch):
    import transformers.trainer as transformers_trainer

    monkeypatch.setattr(
        runner,
        "validate_trusted_resume_checkpoint",
        lambda path: {"path": str(path), "step": 250, "trusted_local_checkpoint": True},
    )
    runner.enable_trusted_torch_load_for_resume("checkpoint-250")
    assert transformers_trainer.check_torch_load_is_safe() is None
    with transformers_trainer.safe_globals():
        pass


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
    audit = audit_sample8_sampler(rows, ThinkG4SingleGroupSampler(rows))
    assert audit["fresh_rollout_count"] == 2
    assert audit["optimizer_steps"] == 4


def test_no_reroll_in_source():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    assert "resample_decision" not in source and "zero_std_reroll" not in source


def test_loss_weights_are_one_to_one():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    assert "self.lambda_cot, self.lambda_sid = 1.0, 1.0" in source


def test_iteration2_reuses_cache_without_resample():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    assert "_sample_policy_epoch >= 2" in source
    assert source.count("self.model.generate(") == 1


def test_rollout_fingerprint_is_stable():
    row = {"recommendation_group_id": "x", "cot_ids": [1],
           "sample_candidate_ids": [[1, 2, 3, 4]] * 8, "cot_reward": 1,
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


def test_training_sample8_probe_beam32_separation():
    source = Path(__file__).with_name("run_sample8_fullsid_train.py").read_text()
    assert '"do_sample": True' in source
    assert '"max_new_tokens": SAMPLE_MAX_NEW_TOKENS' in source
    assert "production-shaped Probe4/Beam32" in source


def test_launcher_uses_four_gpus():
    source = Path(__file__).with_name("launch_sample8_fullsid_train.sh").read_text()
    assert "--nproc_per_node=4" in source


def test_launcher_supports_safe_checkpoint_resume():
    source = Path(__file__).with_name("launch_sample8_fullsid_train.sh").read_text()
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-/root/GRPO-checkpoints}"' in source
    assert '--save-steps 50' in source
    assert '--probe-every-steps 50' in source
    assert '--save-total-limit 64' in source
    assert 'RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"' in source
    assert '--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}"' in source


def test_launcher_uses_nccl_loopback():
    source = Path(__file__).with_name("launch_sample8_fullsid_train.sh").read_text()
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
    cfg = runner.sample8_config_kwargs()
    assert cfg["num_iterations"] == 2 and cfg["beta"] == 0.0
    assert cfg["epsilon"] == 0.2 and cfg["weight_decay"] == 0.0
    assert cfg["lr_scheduler_type"] == "constant"


def test_sid_old_logp_is_full_forward_not_generate_scores():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    assert "SID_OLD_LOGP_CONTRACT_FAILED" in source
    assert "generate.scores" not in source


def test_action_boundaries_are_explicit():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    assert 'inputs["completion_mask"]' in source
    assert '"action_mask": action_mask' in source
    assert 'sid["action_mask"]' in source


def test_preflight_module_cold_import():
    from . import gpu_preflight
    assert gpu_preflight.COT_G == 4 and gpu_preflight.SID_G == 8


def test_preflight_group_is_train_not_probe(plan):
    from .gpu_preflight import PREFLIGHT_GROUP_ID
    assert PREFLIGHT_GROUP_ID in set(plan["dataset"]["recommendation_group_id"])
    assert PREFLIGHT_GROUP_ID not in runner.FIXED_PROBE4_IDS


def test_preflight_uses_real_sampled_advantages():
    trainer_source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    preflight_source = Path(__file__).with_name("gpu_preflight.py").read_text()
    assert "diagnostic_advantages" not in trainer_source
    assert 'production_cot_local = prepared["advantages"]' in preflight_source
    assert "sid_zero_std_production_advantage_zero" in preflight_source


def test_no_fixed_domain_or_language_bridge_and_full_sid_is_action():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    assert 'sample_context_ids = list(prompt_ids) + cot_trim_ids' in source
    assert "build_fixed_domain_beam_input" not in source
    assert "domain_prefix_ids" not in source
    assert '"natural_language_bridge": False' in source
    assert 'list(record["sample_context_ids"]) + list(candidate)' in source
    assert "action_mask.size(1)" in source


def test_sample8_contract_is_exact():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    for contract in (
        "do_sample=True", "temperature=1.0", "top_p=1.0", "top_k=0",
        "repetition_penalty=1.0", "num_return_sequences=SID_G",
        "min_new_tokens=FULL_SID_TOKENS", "max_new_tokens=SAMPLE_MAX_NEW_TOKENS",
    ):
        assert contract in source


def test_cot_reward_is_plain_sum_of_its_eight_sid_rewards():
    rewards = [0, 0.5, 0, 2, 0, 8, 0, -0.25]
    assert cot_reward_from_sid_rewards(rewards) == 10.25
    with pytest.raises(ValueError):
        cot_reward_from_sid_rewards(rewards[:-1])


def test_first_complete_sid_is_found_after_natural_language_prefix():
    class Tokenizer:
        TOKENS = {
            1: "<|video_begin|>", 2: "<s_a_12>", 3: "<s_b_34>",
            4: "<s_c_56>", 5: "not-a-domain", 6: "<|prod_begin|>",
            7: "<s_a_7>", 8: "<s_b_8>", 9: "<s_c_9>",
        }

        def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
            del skip_special_tokens
            return [self.TOKENS[value] for value in ids]

    tokenizer = Tokenizer()
    assert parse_full_sid_ids(tokenizer, [1, 2, 3, 4]) == ("video", 12, 34, 56)
    assert parse_full_sid_ids(tokenizer, [5, 5, 1, 2, 3, 4]) == ("video", 12, 34, 56)
    assert parse_full_sid_ids(tokenizer, [1, 2, 3]) is None


def test_multiple_sids_select_first_and_are_flagged():
    class Tokenizer:
        TOKENS = {
            0: "prefix", 1: "<|video_begin|>", 2: "<s_a_12>",
            3: "<s_b_34>", 4: "<s_c_56>", 5: "middle",
            6: "<|prod_begin|>", 7: "<s_a_7>", 8: "<s_b_8>", 9: "<s_c_9>",
        }

        def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
            del skip_special_tokens
            return [self.TOKENS[value] for value in ids]

    parsed = scan_full_sid_ids(Tokenizer(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert parsed["parsed_sid"] == ("video", 12, 34, 56)
    assert parsed["all_parsed_sids"] == [
        ("video", 12, 34, 56), ("prod", 7, 8, 9),
    ]
    assert parsed["first_sid_span"] == (1, 5)
    assert parsed["sid_count"] == 2
    assert parsed["multi_sid_output"] is True
    assert parsed["parser_status"] == "multiple_sids"


def test_long_continuation_budget_allows_sid_after_prefix():
    assert SAMPLE_MAX_NEW_TOKENS == 128


def test_q_reward_six_levels_remain_available():
    source = Path(__file__).with_name("sample8_fullsid_trainer.py").read_text()
    assert 'q_reward(sid, gold_set)' in source
    assert all(level in source for level in ("-1.0", "-0.25", "0.0", "0.5", "2.0", "8.0"))
