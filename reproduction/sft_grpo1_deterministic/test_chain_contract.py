import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GRPO = REPO / "baselines/native_source_domain_r32_v3/grpo_fullbase_conservative_v1"


def load_verifier():
    spec = importlib.util.spec_from_file_location("chain_verifier", HERE / "verify_chain.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_two_retained_states_and_frozen_hashes():
    verifier = load_verifier()
    assert verifier.SFT_MODEL_SHA256 == "8be8b4d08f2eaa295159f2333e20949aa71ed552c3646a80380485c2efa2eb59"
    assert verifier.SFT_CONFIG_SHA256 == "78451878177a5443d87440940c54177da6e833e8ab4c97dab0e263a4c97162e2"
    assert verifier.GRPO_ADAPTER_SHA256 == "274d4cc0a54bb9921e1576b8338d1d439ac625d00e3de0ca2aa8c4e7311057c8"
    script = (HERE / "run.sh").read_text(encoding="utf-8")
    assert "01_sft_final" in script
    assert "02_grpo1_step500" in script
    assert "checkpoint-100" not in script


def test_grpo_final_only_diff_is_checkpoint_cadence_only():
    original = json.loads((GRPO / "config/formal_500.json").read_text(encoding="utf-8"))
    compact = json.loads((GRPO / "config/formal_500_final_only.json").read_text(encoding="utf-8"))
    assert compact.pop("run_kind") == "formal_grpo1_500_final_only"
    assert original.pop("run_kind") == "formal_grpo1_500"
    assert compact.pop("checkpoint") == {
        "save_steps": 500,
        "save_total_limit": 1,
        "adapter_only": True,
        "full_resume_state": True,
    }
    assert original.pop("checkpoint") == {
        "save_steps": 100,
        "save_total_limit": 5,
        "adapter_only": True,
        "full_resume_state": True,
    }
    assert compact == original


def test_grpo_artifact_contract_accepts_only_step500(tmp_path, monkeypatch):
    verifier = load_verifier()
    output = tmp_path / "grpo"
    checkpoint = output / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps({"status": "PASS", "global_step": 500}), encoding="utf-8"
    )
    for name in verifier.GRPO_REQUIRED:
        (checkpoint / name).write_bytes(b"fixture")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 500, "max_steps": 500}), encoding="utf-8"
    )
    (checkpoint / "lineage.json").write_text(
        json.dumps({"parent_base_sha256": verifier.SFT_MODEL_SHA256}), encoding="utf-8"
    )
    monkeypatch.setattr(verifier, "sha256", lambda _: verifier.GRPO_ADAPTER_SHA256)
    assert verifier.verify_grpo(output)["status"] == "PASS"


def test_grpo_artifact_contract_rejects_an_intermediate_checkpoint(tmp_path, monkeypatch):
    verifier = load_verifier()
    output = tmp_path / "grpo"
    checkpoint = output / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    (output / "checkpoint-250").mkdir()
    (output / "summary.json").write_text(
        json.dumps({"status": "PASS", "global_step": 500}), encoding="utf-8"
    )
    monkeypatch.setattr(verifier, "sha256", lambda _: verifier.GRPO_ADAPTER_SHA256)
    try:
        verifier.verify_grpo(output)
    except RuntimeError as error:
        assert "only checkpoint-500" in str(error)
    else:
        raise AssertionError("unexpected intermediate checkpoint was accepted")
