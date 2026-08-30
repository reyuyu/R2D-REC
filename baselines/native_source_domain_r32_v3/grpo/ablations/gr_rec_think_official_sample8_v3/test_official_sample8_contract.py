from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

from grpo_sid import q_reward

from . import official_sample8_trainer as trainer
from . import run_official_sample8_train as runner
from .official_probe import OfficialProbeEvaluator


@pytest.fixture(scope="module")
def formal_plan():
    args = runner.build_official_arg_parser().parse_args([
        "--run-id", "V3-OFFICIAL-CPU-CONTRACT",
        "--n-groups", "all",
        "--max-steps", "2",
        "--probe-groups", "4",
    ])
    return runner.prepare_official_run_plan(args)


def _record(index, rewards):
    candidates = {
        -1.0: None,
        -0.25: ("ad", 90, 91, 92),
        0.0: ("video", 90, 91, 92),
        0.5: ("video", 1, 91, 92),
        2.0: ("video", 1, 2, 92),
        8.0: ("video", 1, 2, 3),
    }
    return {
        "recommendation_group_id": "group",
        "target_domain": "video",
        "cot_ids": [100 + index, 151668],
        "cot_length": 2,
        "candidate_ids": [[200 + i, 300 + i, 400 + i] for i in range(8)],
        "candidate_texts": [f"candidate-{i}" for i in range(8)],
        "candidate_sids": [candidates[float(value)] for value in rewards],
        "gold_sids": [("video", 1, 2, 3)],
    }


def _records():
    rewards = [8.0, 2.0, 0.5, 0.0, -0.25, -1.0, 0.0, 0.5]
    return [_record(index, rewards) for index in range(4)]


def test_parent_path_and_sha_are_frozen():
    assert runner.PARENT_ADAPTER.name == "checkpoint-1106"
    assert runner.PARENT_ADAPTER_SHA256 == (
        "4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3"
    )


def test_live_parent_adapter_guard():
    audit = runner.validate_official_parent()
    assert audit == {
        "path": str(runner.PARENT_ADAPTER),
        "adapter_sha256": runner.PARENT_ADAPTER_SHA256,
    }


def test_source_dataset_and_training_topology_are_frozen():
    assert str(runner.SOURCE_DATASET) == "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"
    assert runner.SOURCE_DATASET_SHA256 == (
        "791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc"
    )
    assert runner.EXPECTED_RAW_GROUPS == 1549
    assert runner.EXPECTED_TRAIN_GROUPS == 1545
    assert runner.EXPECTED_FORMAL_STEPS == 3090
    assert runner.EXPECTED_DOMAIN_GROUPS == {
        "ad": 426, "living": 189, "prod": 381, "video": 549,
    }


def test_live_source_dataset_sha_and_plan_guards(formal_plan):
    assert runner.sha256_file(runner.SOURCE_DATASET) == runner.SOURCE_DATASET_SHA256
    assert formal_plan["raw_groups"] == 1549
    assert len(formal_plan["dataset"]) == 1545
    assert formal_plan["audit"]["optimizer_steps"] == 3090
    assert formal_plan["audit"]["domain_group_counts"] == runner.EXPECTED_DOMAIN_GROUPS
    assert formal_plan["dataset_guard"]["train_group_id_sha256"] == runner.TRAIN_GROUP_ID_SHA256
    assert formal_plan["dataset_guard"]["train_canonical_sha256"] == runner.TRAIN_CANONICAL_SHA256
    assert formal_plan["probe_train_overlap"] == []


def test_dataset_hash_guards_are_frozen():
    assert runner.TRAIN_GROUP_ID_SHA256 == (
        "c1416069c076469252668bb27f69b0904946bd2d2154cbe3024115a17175be5f"
    )
    assert runner.TRAIN_CANONICAL_SHA256 == (
        "1ff7965aab7ac0126f1898699b704e5df65b9ad270f8f50c40cf12b5f6d49055"
    )


def test_probe4_is_fixed_and_one_per_domain():
    assert len(runner.FIXED_PROBE4_IDS) == len(set(runner.FIXED_PROBE4_IDS)) == 4


def test_official_topology_and_action_mask():
    assert (trainer.COT_G, trainer.SID_G, trainer.OFFICIAL_SID_TOKENS) == (4, 8, 3)
    mask = trainer.official_action_mask([[1, 2, 3] for _ in range(8)])
    assert mask == [[1, 1, 1] for _ in range(8)]
    with pytest.raises(ValueError):
        trainer.official_action_mask([[1, 2, 3] for _ in range(7)])
    with pytest.raises(ValueError):
        trainer.official_action_mask([[1, 2] for _ in range(8)])


def test_raw_q_reward_parity():
    gold = {("video", 1, 2, 3)}
    candidates = [None, ("ad", 1, 2, 3), ("video", 9, 9, 9),
                  ("video", 1, 9, 9), ("video", 1, 2, 9), ("video", 1, 2, 3)]
    assert [q_reward(item, gold) for item in candidates] == [-1, -0.25, 0, 0.5, 2, 8]


def test_four_g8_groups_are_normalized_independently():
    groups = [[8, 0, 0, 0, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0, 0, 0],
              [0.5, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    actual = trainer.independent_official_advantages(groups)
    assert actual.shape == (4, 8)
    assert torch.allclose(actual[0], trainer.population_advantages(groups[0]))
    assert torch.all(actual[3] == 0)


def test_finalize_uses_raw_rewards_and_cot_sum():
    records = trainer.finalize_global_records(_records())
    expected = [8.0, 2.0, 0.5, 0.0, -0.25, -1.0, 0.0, 0.5]
    for record in records:
        assert record["q_rewards"] == record["sid_rewards"] == expected
        assert record["cot_reward"] == sum(expected)
        assert (record["exact"], record["ab"], record["a"]) == (1, 1, 2)


def test_scalar_sid_advantage_is_broadcast_to_abc3():
    source = inspect.getsource(trainer.ThinkOfficialSample8Trainer._branch_loss)
    assert "advantages.unsqueeze(1)" in source
    assert "per_token * action" in source


def test_loss_is_exactly_one_to_one():
    assert trainer.LAMBDA_COT == trainer.LAMBDA_SID == 1.0
    assert trainer.combine_losses(torch.tensor(2.0), torch.tensor(3.0)).item() == 5.0


def test_iteration2_reuses_fingerprint_and_sampling():
    fp = trainer.rollout_fingerprint(trainer.finalize_global_records(_records()))
    assert trainer.assert_iteration_reuse(fp, fp, 2, 4, 4)
    with pytest.raises(RuntimeError):
        trainer.assert_iteration_reuse(fp, "changed", 2, 4, 4)
    with pytest.raises(RuntimeError):
        trainer.assert_iteration_reuse(fp, fp, 2, 4, 5)


def test_official_sampling_contract_is_exact_abc3():
    source = inspect.getsource(trainer.OfficialSample8Runtime._sample_official)
    for fragment in ("do_sample=True", "temperature=1.0", "top_p=1.0", "top_k=0",
                     "num_return_sequences=SID_G", "min_new_tokens=OFFICIAL_SID_TOKENS",
                     "max_new_tokens=OFFICIAL_SID_TOKENS"):
        assert fragment in source


def test_fixed_domain_prefix_is_context_not_action():
    source = inspect.getsource(trainer.OfficialSample8Runtime.score_local_cot)
    assert "build_fixed_domain_beam_input" in source
    assert "parse_strict_abc3_ids" in source
    assert '"official_domain_prefix"' in source


def test_old_logp_is_no_grad_full_forward():
    source = inspect.getsource(trainer.ThinkOfficialSample8Trainer._build_official_rollout)
    assert "with torch.no_grad()" in source
    assert "_get_per_token_logps_and_entropies" in source
    assert "old_logps.detach()" in source


def test_no_history_or_reward_shaping_in_training_math():
    source = inspect.getsource(trainer.finalize_global_records).lower()
    forbidden = ("history", "copy", "warmup", "saturation", "duplicate", "finegrained")
    assert all(word not in source for word in forbidden)


def test_audit_exposes_all_baseline_manifest_keys():
    rows = [{"route": "think", "recommendation_group_id": f"g{i}",
             "target_domain": "video"} for i in range(3)]
    audit = trainer.audit_official_sampler(rows, object())
    required = {
        "selected_groups", "trained_groups", "dropped_groups", "think_unique_groups",
        "nothink_unique_groups", "think_rollouts", "nothink_rollouts",
        "unique_groups_per_global_rollout", "optimizer_steps", "think_optimizer_steps",
        "nothink_optimizer_steps", "route_schedule_preview",
    }
    assert required <= audit.keys()
    assert audit["optimizer_steps"] == 6


def test_config_is_frozen():
    config = runner.official_config_kwargs()
    assert config["num_iterations"] == 2
    assert config["weight_decay"] == 0.0
    assert config["lr_scheduler_type"] == "constant"
    assert config["temperature"] == 0.9 and config["top_p"] == 0.95


def test_probe_summary_is_production_official_schema():
    candidates = [{"reward": 8.0, "exact": 1, "ab": 0, "a": 0,
                   "invalid": 0, "closed": True, "completion_length": 10}]
    summary = OfficialProbeEvaluator._summary(candidates, think=True)
    assert summary["candidate_count"] == 1
    assert summary["closure_rate"] == 1.0
    assert summary["official_fixed_domain"] is True
    assert summary["beam_width"] == 32
    assert summary["reward_denominator"] == "1 sampled CoTs"


class _FakeWriter:
    rank = 0

    def __init__(self):
        self.manifest = None
        self.events = []

    def write_manifest(self, payload):
        self.manifest = payload

    def _append(self, name, payload):
        self.events.append((name, payload))
        return True


def test_manifest_and_monitor_contract():
    fake = _FakeWriter()
    writer = runner.OfficialSample8ManifestWriter(fake)
    writer.write_manifest({"fixed_probe": {}})
    manifest = fake.manifest
    assert manifest["expected_training_groups"] == 1545
    assert manifest["expected_optimizer_steps"] == 3090
    assert manifest["objective_weighting"] == "cot_loss + sid_loss (1:1)"
    assert manifest["fixed_probe"]["count"] == 4
    assert writer.write_official_sample8_v3({"step": 1})
    assert fake.events[0][0] == "official_sample8.jsonl"


def test_launcher_is_fresh_only_and_formal():
    source = (Path(__file__).with_name("launch_official_sample8_train.sh")).read_text()
    assert "--nproc_per_node=4" in source
    assert "--save-steps 50" in source
    assert "--probe-every-steps 50" in source
    assert 'RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"' in source
    assert '--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}"' in source
    assert "GRPO_MONITOR_DIR" in source and "NCCL_SOCKET_IFNAME" in source


def test_parent_hash_guard_rejects_drift(monkeypatch, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"drift")
    monkeypatch.setattr(runner, "PARENT_ADAPTER", tmp_path)
    with pytest.raises(RuntimeError, match="PARENT_SHA_MISMATCH"):
        runner.validate_official_parent()


def _complete_resume_checkpoint(root, run_id, step, *, state_step=None):
    checkpoint = root / run_id / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
        "scheduler.pt", "training_args.bin",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    ):
        (checkpoint / name).write_bytes(b"complete")
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": step if state_step is None else state_step,
    }))
    return checkpoint


def test_trusted_resume_accepts_complete_same_run_even_boundary(tmp_path):
    run_id = runner.FORMAL_RUN_ID_PREFIX + "E1-TEST"
    checkpoint = _complete_resume_checkpoint(tmp_path, run_id, 50)
    audit = runner.validate_trusted_resume_checkpoint(checkpoint, run_id, tmp_path)
    assert audit["step"] == 50
    assert audit["run_id"] == run_id
    assert audit["trusted_v3_official_checkpoint"] is True


def test_trusted_resume_rejects_other_run_and_nonofficial_run_id(tmp_path):
    run_id = runner.FORMAL_RUN_ID_PREFIX + "E1-TEST"
    other = _complete_resume_checkpoint(tmp_path, "OTHER-RUN", 50)
    with pytest.raises(RuntimeError, match="UNTRUSTED_RESUME_PATH"):
        runner.validate_trusted_resume_checkpoint(other, run_id, tmp_path)
    with pytest.raises(RuntimeError, match="UNTRUSTED_RESUME_RUN_ID"):
        runner.validate_trusted_resume_checkpoint(other, "OTHER-RUN", tmp_path)


def test_trusted_resume_rejects_odd_boundary(tmp_path):
    run_id = runner.FORMAL_RUN_ID_PREFIX + "E1-TEST"
    checkpoint = _complete_resume_checkpoint(tmp_path, run_id, 51)
    with pytest.raises(RuntimeError, match="INVALID_RESUME_BOUNDARY"):
        runner.validate_trusted_resume_checkpoint(checkpoint, run_id, tmp_path)


@pytest.mark.parametrize("missing", ["optimizer.pt", "rng_state_3.pth"])
def test_trusted_resume_rejects_incomplete_optimizer_or_rng(tmp_path, missing):
    run_id = runner.FORMAL_RUN_ID_PREFIX + "E1-TEST"
    checkpoint = _complete_resume_checkpoint(tmp_path, run_id, 50)
    (checkpoint / missing).unlink()
    with pytest.raises(RuntimeError, match="RESUME_INCOMPLETE"):
        runner.validate_trusted_resume_checkpoint(checkpoint, run_id, tmp_path)


def test_trusted_resume_rejects_trainer_state_mismatch(tmp_path):
    run_id = runner.FORMAL_RUN_ID_PREFIX + "E1-TEST"
    checkpoint = _complete_resume_checkpoint(tmp_path, run_id, 50, state_step=48)
    with pytest.raises(RuntimeError, match="RESUME_STEP_MISMATCH"):
        runner.validate_trusted_resume_checkpoint(checkpoint, run_id, tmp_path)
