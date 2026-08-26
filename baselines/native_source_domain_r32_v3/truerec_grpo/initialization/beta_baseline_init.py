"""CPU-only Beta-Baseline initialization contract and provenance audit."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "eval", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from official_aligned_contract import ACTION_LEVELS, ACTION_TOKENS, TRAIN_ROLLOUT_G, ContractError, train_prefix_contract, validate_no_bridge_contract  # noqa: E402
from rollout_runtime_v1 import GENERATION_KWARGS  # noqa: E402


CONFIG_NAME = "beta_baseline_init_v1.json"
CONTRACT_ID = "truerec_grpo_beta_baseline_init_v1"
INIT_FAMILY = "BETA_BASELINE"
BETA_CHECKPOINT = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
PRETRAINED_BASE = Path("/data/models/onereason-8b-pretrain-competition")
BETA_GAMMA_MARKERS = ("BETA-GAMMA", "beta_gamma", "Beta-Gamma")
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
PHASE07_BETAGAMMA_RAW_USED_AS_BETA_ROLLOUT = False
PHASE08_BETAGAMMA_PLAN_USED_AS_BETA_CURRENT_PLAN = False

FROZEN_CORE_SHA = {
    "credit/frontier_credit_v1.py": "639e9a24dd1885798f49a4efd9bbf6b6b9dc8daed833fe9ba8560a86a747788b",
    "credit/hpr_plan_v1.py": "fadb19056853f898bd85d6779f3d6718bf70bf4ddbd5b7ac44520de88197fe32",
    "trainer/action_alignment.py": "2ca6c2b1ca034e84be55e359100ed86cf01503200fefa789e30abb2661f505e1",
    "trainer/truerec_loss_v1.py": "4a52f40639c332824e5dc45d6e4da1defcdbf4fc9f8e0c0dc380d5c4e5287d57",
    "trainer/truerec_runtime_v1.py": "2f18378f165d523e8eceaf85f8d575c7a237faacb99a072d8f22d326cef254f7",
    "trainer/truerec_trainer_v1.py": "7b5da094f7cb000fe5d66123eb4310a208e6fd963c0443b55efe9fb2cdddbdfb",
    "trainer/rollout_runtime_v1.py": "c21f9d79ccc9f3e43b55e01278ea31b07ce92d7230b520302848beed8a896b81",
    "trainer/batch_collator_v1.py": "cafec7b6ac097cc4e95affe1681869fd7e4b58c56de48e9a9ff7b7bb62c93351",
    "trainer/evaluation_runtime_v1.py": "2a9f75eddfd8663d263a31364ba5ef1ea3cb2d52089aaa21d56e839952db12bd",
    "trainer/truerec_grpo_trainer_v1.py": "1bf73e7b9888025cb5cc5ba7fb92788bb7841881fecbf566926b6814eb12f552",
}


@dataclass(frozen=True)
class BetaBaselineInitContract:
    contract_id: str
    init_family: str
    checkpoint_path: str
    pretrained_base_path: str
    beta_gamma_used_as_init: bool
    bridge: bool
    fixed_domain_in_context: bool
    domain_generated_by_model: bool
    action_levels: tuple[str, ...]
    action_tokens: int
    rollout_g: int
    policy_rollout_source: str
    current_hpr_plan_source: str
    reuse_phase07_beta_gamma_raw: bool
    reuse_phase08_beta_gamma_plan: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BetaBaselineInitContract":
        values = dict(payload)
        values["action_levels"] = tuple(values["action_levels"])
        return cls(**values)

    def validate_static(self) -> None:
        validate_no_bridge_contract(vars(self))
        prefix = train_prefix_contract()
        if self.contract_id != CONTRACT_ID or self.init_family != INIT_FAMILY:
            raise ContractError("TrueRec V1 initialization family must be Beta-Baseline")
        if Path(self.checkpoint_path) != BETA_CHECKPOINT:
            raise ContractError("Beta checkpoint path is not the frozen checkpoint-1106")
        if any(marker.lower() in self.checkpoint_path.lower() for marker in BETA_GAMMA_MARKERS):
            raise ContractError("Beta-Gamma checkpoint cannot initialize Beta-Baseline GRPO")
        if Path(self.pretrained_base_path) != PRETRAINED_BASE:
            raise ContractError("pretrained base path mismatch")
        if self.beta_gamma_used_as_init:
            raise ContractError("Beta-Gamma cannot be used as current initialization")
        if not self.fixed_domain_in_context or self.domain_generated_by_model:
            raise ContractError("fixed target domain must be context, not action")
        if self.action_levels != ACTION_LEVELS or self.action_tokens != ACTION_TOKENS:
            raise ContractError("training action must be exactly ABC3")
        if self.rollout_g != TRAIN_ROLLOUT_G or GENERATION_KWARGS["num_return_sequences"] != TRAIN_ROLLOUT_G:
            raise ContractError("training rollout must preserve G=8")
        if self.policy_rollout_source != "live_beta_policy" or self.current_hpr_plan_source != "online_current_group":
            raise ContractError("rollout and HPR plans must be recomputed from the current Beta policy/group")
        if self.reuse_phase07_beta_gamma_raw or self.reuse_phase08_beta_gamma_plan:
            raise ContractError("policy-specific Beta-Gamma artifacts cannot be reused")
        if prefix.bridge or not prefix.fixed_domain_in_context or prefix.domain_generated:
            raise ContractError("frozen training prefix contract mismatch")


def load_contract(path: Path | None = None) -> BetaBaselineInitContract:
    config_path = path or Path(__file__).with_name(CONFIG_NAME)
    contract = BetaBaselineInitContract.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    contract.validate_static()
    return contract


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_model_provenance(contract: BetaBaselineInitContract) -> dict[str, Any]:
    checkpoint, base = Path(contract.checkpoint_path), Path(contract.pretrained_base_path)
    if not checkpoint.is_dir() or not base.is_dir():
        raise ContractError("checkpoint or pretrained base directory missing")
    files = {name: checkpoint / name for name in REQUIRED_ADAPTER_FILES}
    if not all(path.is_file() for path in files.values()):
        raise ContractError("required adapter file missing")
    adapter_config = json.loads(files["adapter_config.json"].read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != contract.pretrained_base_path:
        raise ContractError("adapter base_model_name_or_path mismatch")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ContractError("checkpoint is not a causal-LM LoRA adapter")
    return {
        "checkpoint_exists": True,
        "pretrained_base_exists": True,
        "required_files": {name: {"path": str(path), "size": path.stat().st_size, "sha256": file_sha(path)} for name, path in files.items()},
        "adapter_base_model_name_or_path": adapter_config["base_model_name_or_path"],
        "adapter_peft_type": adapter_config["peft_type"],
        "adapter_task_type": adapter_config["task_type"],
        "pass": True,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_audit(output_dir: Path, source_root: Path, runtime_root: Path, test_status: str) -> None:
    if test_status != "PASS":
        raise ContractError("CPU test gate failed")
    source_files = ("initialization/beta_baseline_init.py", f"initialization/{CONFIG_NAME}", "tests/test_beta_baseline_init.py")
    parity = {name: {"source": file_sha(source_root / name), "runtime": file_sha(runtime_root / name)} for name in source_files}
    if not all(value["source"] == value["runtime"] for value in parity.values()):
        raise ContractError("source/runtime initialization file parity failed")
    contract = load_contract(runtime_root / "initialization" / CONFIG_NAME)
    provenance = validate_model_provenance(contract)
    frozen = {name: {"actual": file_sha(runtime_root / name), "expected": expected} for name, expected in FROZEN_CORE_SHA.items()}
    if not all(value["actual"] == value["expected"] for value in frozen.values()):
        raise ContractError("frozen Phase0.8/1.0/1.1 core SHA changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "initialization_contract_audit.json", {"contract": vars(contract), "model_provenance": provenance, "source_runtime_parity": parity})
    write_json(output_dir / "training_input_contract_audit.json", {
        "train_bridge": False, "train_fixed_domain_in_context": True,
        "train_domain_generated_by_model": False, "train_action": ACTION_LEVELS,
        "train_action_tokens": ACTION_TOKENS, "train_rollout_G": TRAIN_ROLLOUT_G,
        "phase07_beta_gamma_raw_used_as_beta_rollout": False,
        "phase08_beta_gamma_plan_used_as_beta_current_plan": False,
        "frontier_math_modified": False, "hpr_math_modified": False,
        "ppo_loss_modified": False, "rollout_parser_modified": False,
        "evaluation_frontend_modified": False,
    })
    write_json(output_dir / "frozen_core_sha_audit.json", frozen)
    (output_dir / "CHATGPT_PHASE1_2A_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2A CPU initialization audit: PASS\n"
        "The frozen parent is Beta-Baseline checkpoint-1106 over the original pretrained base. Required LoRA files and adapter base provenance pass.\n"
        "The TrueRec no-bridge fixed-domain ABC3 G8 training prefix is unchanged. Beta-Gamma rollout and HPR plan artifacts are not reused.\n"
        "No GPU, model forward, generation, backward, optimizer, or training ran.\n",
        encoding="utf-8",
    )
    print("PHASE1_2A_CPU_AUDIT=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--test-status", choices=("PASS", "FAIL"), required=True)
    args = parser.parse_args()
    run_audit(args.output_dir, args.source_root, args.runtime_root, args.test_status)


if __name__ == "__main__":
    main()
