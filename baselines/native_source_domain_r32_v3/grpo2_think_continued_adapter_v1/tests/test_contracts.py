from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapter_identity import canonical_tensor_hash
from continued_contracts import (
    DATASET_SHA256,
    EXPECTED_DOMAINS,
    EXPECTED_ROWS,
    GRPO1_STEP500_ADAPTER_SHA256,
    SFT_MODEL_SHA256,
    validate_config,
    validate_grpo2_resume,
)

PACKAGE = Path(__file__).resolve().parents[1]


def config(name="pilot_20.json"):
    return json.loads((PACKAGE / "config" / name).read_text(encoding="utf-8"))


def test_inherited_adapter_mode_enabled():
    assert validate_config(config())["adapter_inheritance"] == "CONTINUE_PARENT_ADAPTER"


def test_fresh_lora_disabled():
    assert config()["fresh_lora"] is False


def test_fresh_optimizer_enabled():
    assert config()["fresh_optimizer"] is True


def test_grpo1_adapter_sha_frozen():
    assert GRPO1_STEP500_ADAPTER_SHA256 == "274d4cc0a54bb9921e1576b8338d1d439ac625d00e3de0ca2aa8c4e7311057c8"


def test_sft_parent_sha_frozen():
    assert SFT_MODEL_SHA256 == "8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59"


def test_dataset_and_topology_frozen():
    assert DATASET_SHA256 == "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"
    assert EXPECTED_ROWS == 611
    assert EXPECTED_DOMAINS == {"ad": 160, "living": 73, "prod": 118, "video": 260}


def test_train_rows_are_think_only():
    value = config()["dataset"]
    assert value["rows"] == 611 and value["route"] == "think"


def test_historical_optimizer_math_unchanged():
    value = validate_config(config())["optimization"]
    assert value == {
        "learning_rate": 2e-7,
        "historical_learning_rate": 1e-6,
        "max_steps": 20,
        "weight_decay": 0.0,
        "lr_scheduler_type": "constant",
        "max_grad_norm": 1.0,
        "beta": 0.0,
        "epsilon": 0.2,
        "loss_type": "grpo",
        "scale_rewards": "group",
        "num_iterations": 2,
    }


def test_step0_parity_uses_full_canonical_tensor_hash():
    source = (PACKAGE / "scripts" / "adapter_identity.py").read_text(encoding="utf-8")
    assert "torch.equal" in source
    assert "canonical_tensor_hash(source)" in source
    assert "tensor_values_byte_exact" in source


def test_base_and_optimizer_contracts_are_runtime_gates():
    runner = (PACKAGE / "scripts" / "run_grpo2_continued.py").read_text(encoding="utf-8")
    trainer = (PACKAGE / "scripts" / "trainer.py").read_text(encoding="utf-8")
    assert "base_trainable_parameter_count" in runner
    assert "assert_optimizer_lora_only" in trainer
    assert "optimizer_lora_only" in runner


def test_no_dense_merge_or_fresh_lora_creation_reachable():
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (PACKAGE / "scripts").glob("*.py"))
    assert "merge_and_unload(" not in sources
    assert "get_peft_model(" not in sources
    assert "fresh_lora_config(" not in sources


def test_grpo1_to_grpo2_is_not_resume():
    value = config()
    assert value["training_resume"] is False
    launcher = (PACKAGE / "scripts" / "run_smokes_and_pilot.sh").read_text(encoding="utf-8")
    assert "--resume-from-checkpoint" not in launcher


def test_grpo2_checkpoint_resume_remains_supported(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "training_args.bin"):
        (checkpoint / name).write_bytes(b"x")
    for rank in range(4):
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"x")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 10}', encoding="utf-8")
    from hashlib import sha256
    adapter_sha = sha256(b"x").hexdigest()
    (checkpoint / "lineage.json").write_text(json.dumps({
        "stage": "GRPO2_REC_THINK",
        "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
        "adapter_initialization_source_sha256": GRPO1_STEP500_ADAPTER_SHA256,
        "grpo2_optimizer_step": 10,
        "resume_supported": True,
        "adapter_sha256": adapter_sha,
    }), encoding="utf-8")
    assert validate_grpo2_resume(checkpoint)["step"] == 10


def test_invalid_resume_lineage_fails_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "lineage.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="RESUME_LINEAGE_INVALID"):
        validate_grpo2_resume(checkpoint)


def test_formal_contract_is_exact_and_probe_free():
    value = validate_config(config("formal_300.json"))
    assert value["optimization"]["max_steps"] == 300
    assert value["checkpoint"]["steps"] == [100, 150, 200, 250, 300]
    assert value["checkpoint"]["save_total_limit"] == 5
    assert value["retention_probe"]["enabled"] is False


def test_formal_runner_uses_exact_schedule_and_no_merge():
    trainer = (PACKAGE / "scripts" / "trainer.py").read_text(encoding="utf-8")
    runner = (PACKAGE / "scripts" / "run_grpo2_continued.py").read_text(encoding="utf-8")
    launcher = (PACKAGE / "scripts" / "run_formal_300.sh").read_text(encoding="utf-8")
    assert "class ExactCheckpointScheduleCallback" in trainer
    assert 'save_strategy = "no"' in runner
    assert "100 150 200 250 300" in launcher
    assert "merge_and_unload" not in launcher
    assert "run_smokes_and_pilot" not in launcher
    assert "GRPO3" not in launcher


def test_formal_lineage_states_single_adapter_inference():
    sources = "\n".join(
        (PACKAGE / "scripts" / name).read_text(encoding="utf-8")
        for name in ("trainer.py", "run_grpo2_continued.py")
    )
    for phrase in (
        "contains_grpo1_and_grpo2_effect",
        "external_grpo1_best_confirmed",
        "training_resume_from_grpo1",
    ):
        assert phrase in sources


def test_checkpoint_lineage_marks_continued_single_adapter():
    source = (PACKAGE / "scripts" / "trainer.py").read_text(encoding="utf-8")
    for phrase in ("CONTINUED_SINGLE_ADAPTER", "optimizer_parent", "trainer_state_parent", "rng_parent"):
        assert phrase in source


def test_runner_loads_inherited_adapter_trainable():
    source = (PACKAGE / "scripts" / "run_grpo2_continued.py").read_text(encoding="utf-8")
    assert "PeftModel.from_pretrained" in source
    assert "is_trainable=True" in source
    assert "autocast_adapter_dtype=True" in source


def test_step0_gate_creates_no_optimizer_step():
    runner = (PACKAGE / "scripts" / "run_grpo2_continued.py").read_text(encoding="utf-8")
    launcher = (PACKAGE / "scripts" / "run_smokes_and_pilot.sh").read_text(encoding="utf-8")
    assert "--step0-only" in runner and '"optimizer_steps": 0' in runner
    assert "STEP0_INHERITED_ADAPTER_PARITY.json" in launcher


def test_runner_reuses_historical_reward_and_trainer():
    source = (PACKAGE / "scripts" / "run_grpo2_continued.py").read_text(encoding="utf-8")
    assert "sample8_fullsid_trainer" in source
    assert "q_reward_without_a_only" in source
    assert "historical_trainer.q_reward = q_reward_without_a_only" in source


def test_launcher_freezes_deterministic_environment():
    source = (PACKAGE / "scripts" / "run_smokes_and_pilot.sh").read_text(encoding="utf-8")
    for phrase in (
        "CUBLAS_WORKSPACE_CONFIG=:4096:8", "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "FLASH_ATTENTION_DETERMINISTIC=1", "NVIDIA_TF32_OVERRIDE=0",
        "NCCL_SOCKET_IFNAME=lo", "GLOO_SOCKET_IFNAME=lo", "NCCL_IB_DISABLE=1",
    ):
        assert phrase in source


def test_launcher_runs_only_smokes_then_pilot():
    source = (PACKAGE / "scripts" / "run_smokes_and_pilot.sh").read_text(encoding="utf-8")
    assert "SMOKE_A" in source and "SMOKE_B" in source and "PILOT" in source
    assert "BYTE_EXACT" in source
    assert "formal_300" not in source.lower()
    assert "grpo3" not in source.lower()


def test_canonical_hash_is_order_independent():
    torch = pytest.importorskip("torch")
    first = {"b": torch.tensor([2]), "a": torch.tensor([1])}
    second = {"a": torch.tensor([1]), "b": torch.tensor([2])}
    assert canonical_tensor_hash(first) == canonical_tensor_hash(second)
