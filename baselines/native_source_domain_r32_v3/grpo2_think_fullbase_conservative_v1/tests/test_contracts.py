import hashlib
import json
from pathlib import Path

import pytest

import contracts
from contracts import (
    DATASET_SHA256,
    EXPECTED_DOMAINS,
    EXPECTED_ROWS,
    GRPO1_ADAPTER_SHA256,
    SFT_MODEL_SHA256,
    canonical_model_identity,
    validate_config,
    validate_parent_manifest,
)
from compare_smokes import _stable

PACKAGE = Path(__file__).resolve().parents[1]


def test_contract_imports_resolve_to_grpo2_package():
    assert Path(contracts.__file__).resolve().parent == PACKAGE / "scripts"


def config(name="pilot_20.json"):
    return json.loads((PACKAGE / "config" / name).read_text(encoding="utf-8"))


def parent_fixture(tmp_path):
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    identity, files = canonical_model_identity(tmp_path)
    value = {
        "canonical": False,
        "test_parent_only": True,
        "canonical_grpo1_parent": False,
        "source_sft_model_sha256": SFT_MODEL_SHA256,
        "source_grpo1_adapter_sha256": GRPO1_ADAPTER_SHA256,
        "canonical_model_identity": identity,
        "weight_files": files,
        "auxiliary_files": [],
        "standalone_reload": "PASS",
        "functional_parity": {"status": "PASS"},
    }
    value["canonical_manifest_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "GRPO2_PARENT_MANIFEST.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


# 1. Noncanonical parents are rejected by default.
def test_test_parent_requires_explicit_flag(tmp_path):
    with pytest.raises(RuntimeError, match="ALLOW_TEST_PARENT"):
        validate_parent_manifest(parent_fixture(tmp_path), allow_test_parent=False)


# 2. The explicit test parent is accepted when every lineage field matches.
def test_test_parent_accepts_explicit_flag(tmp_path):
    assert validate_parent_manifest(parent_fixture(tmp_path), allow_test_parent=True)["test_parent_only"]


# 3. Canonical model identity includes every safetensors shard deterministically.
def test_canonical_model_identity_is_order_independent(tmp_path):
    (tmp_path / "b.safetensors").write_bytes(b"b")
    (tmp_path / "a.safetensors").write_bytes(b"a")
    first, files = canonical_model_identity(tmp_path)
    second, _ = canonical_model_identity(tmp_path)
    assert first == second and [row["name"] for row in files] == ["a.safetensors", "b.safetensors"]


# 4. Parent lineage freezes the selected GRPO-1 checkpoint-300 adapter.
def test_grpo1_parent_sha_frozen():
    assert GRPO1_ADAPTER_SHA256 == "fbae37f3892c414a7c86f2285a4568f2a5bb526c8d249616f0445bd9b90f6c19"


# 5. Parent lineage freezes the full Rec FDR V4.3 SFT model.
def test_sft_parent_sha_frozen():
    assert SFT_MODEL_SHA256 == "8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59"


# 6. Historical Positive-A0 data identity is frozen.
def test_dataset_identity_frozen():
    assert DATASET_SHA256 == "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"


# 7. Historical data topology is 611 unique Think rows.
def test_dataset_topology_frozen():
    assert EXPECTED_ROWS == 611 and config()["dataset"]["route"] == "think"


# 8. Historical domain distribution is frozen.
def test_domain_distribution_frozen():
    assert EXPECTED_DOMAINS == {"ad": 160, "living": 73, "prod": 118, "video": 260}


# 9. Pilot optimizer math is conservative and records the historical LR.
def test_learning_rate_contract():
    values = validate_config(config())["optimization"]
    assert values["learning_rate"] == 2e-7 and values["historical_learning_rate"] == 1e-6


# 10. PPO beta and two-pass rollout reuse remain historical.
def test_ppo_contract():
    values = validate_config(config())["optimization"]
    assert values["beta"] == 0.0 and values["num_iterations"] == 2


# 11. Fresh LoRA geometry is frozen.
def test_lora_contract():
    values = validate_config(config())["lora"]
    assert (values["r"], values["alpha"], values["bias"]) == (32, 64, "none")
    assert set(values["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# 12. All requested RNG domains are independently named and frozen.
def test_seed_contract():
    assert validate_config(config())["seeds"] == {
        "training": 20260816, "dataset": 20260816, "sampler": 20260816,
        "generation": 20260816, "lora_initialization": 20260905, "probe": 20260818,
    }


# 13. Pilot checkpoints are exactly 10 and 20.
def test_pilot_checkpoint_contract():
    values = config()
    assert values["optimization"]["max_steps"] == 20
    assert values["checkpoint"] == {"save_steps": 10, "save_total_limit": 2, "adapter_only": True}


# 14. Smoke is independent five-step execution, never the pilot continuation.
def test_smoke_contract():
    values = validate_config(config("smoke_5.json"))
    assert values["optimization"]["max_steps"] == 5


# 15. Launcher freezes deterministic CUDA/NCCL environment.
def test_launcher_deterministic_environment():
    source = (PACKAGE / "scripts" / "run_smokes_and_pilot.sh").read_text(encoding="utf-8")
    for phrase in (
        "CUBLAS_WORKSPACE_CONFIG=:4096:8", "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "FLASH_ATTENTION_DETERMINISTIC=1", "NVIDIA_TF32_OVERRIDE=0",
        "NCCL_SOCKET_IFNAME=lo", "GLOO_SOCKET_IFNAME=lo", "NCCL_IB_DISABLE=1",
        "GRPO_GIT_COMMIT",
    ):
        assert phrase in source


# 16. Historical trainer/reward are imported, not reimplemented.
def test_runner_reuses_historical_math():
    source = (PACKAGE / "scripts" / "run_grpo2_think.py").read_text(encoding="utf-8")
    assert "sample8_fullsid_trainer" in source
    assert "q_reward_without_a_only" in source
    assert "historical_trainer.q_reward = q_reward_without_a_only" in source
    assert "make_disabled_nothink_reward_func" in source


# 17. Execution stops after pilot and never exposes formal or GRPO-3 launch paths.
def test_no_formal_or_grpo3_launch():
    source = (PACKAGE / "scripts" / "run_smokes_and_pilot.sh").read_text(encoding="utf-8")
    assert "READY_FOR_GRPO2_PARENT_FINALIZATION" in source
    assert "formal_500" not in source.lower()
    assert "grpo3" not in source.lower()


# 18. Historical ablation imports resolve from the GRPO package root.
def test_runner_adds_historical_ablation_import_root():
    source = (PACKAGE / "scripts" / "run_grpo2_think.py").read_text(encoding="utf-8")
    assert 'GRPO_DIR = NATIVE_DIR / "grpo"' in source
    assert "(FULLBASE_SCRIPTS, GRPO_SCRIPTS, GRPO_DIR)" in source
    trainer_source = (PACKAGE / "scripts" / "trainer.py").read_text(encoding="utf-8")
    assert "from ablations.gr_rec_think_sample8_fullsid_v3.sample8_fullsid_trainer import" in trainer_source


# 19. Determinism comparison ignores timing telemetry but preserves model evidence.
def test_smoke_comparator_ignores_only_timing_telemetry():
    value = {
        "gen_wall_sec": 1.2,
        "beam_global_wall_sec": 3.4,
        "rollout_sec": 5.6,
        "timestamp": "now",
        "loss": 0.25,
        "adapter_sha256": "frozen",
    }
    assert _stable(value) == {"adapter_sha256": "frozen", "loss": 0.25}
