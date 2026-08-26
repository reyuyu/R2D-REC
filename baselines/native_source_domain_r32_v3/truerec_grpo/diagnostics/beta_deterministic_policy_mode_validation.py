"""Phase 1.2C1a: validate deterministic eval-mode scoring with trainable Beta LoRA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "data", TRUE_REC_ROOT / "diagnostics", TRUE_REC_ROOT / "initialization", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE, load_contract, validate_model_provenance  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from beta_old_logp_rescore_validation import REPEAT_ABS_MAX_THRESHOLD, TRAINER_SHA256, comparison_stats  # noqa: E402
from beta_single_group_loss_audit import GROUP_ID, PAD_TOKEN_ID, load_phase12b_artifact, load_pilot_record, reconstruct_business_group  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from policy_scoring_v1 import POLICY_SCORING_MODE, SCORING_MICROBATCH_SIZE, score_full_sequences  # noqa: E402


EXPECTED_LORA_DROPOUT = 0.05


class Phase12C1aError(RuntimeError):
    pass


def dropout_config_values(value: Any, prefix: str = "") -> dict[str, float]:
    """Collect numeric dropout settings from nested model/adapter configuration."""
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "dropout" in str(key).lower() and isinstance(child, (int, float)) and not isinstance(child, bool):
                output[path] = float(child)
            output.update(dropout_config_values(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            output.update(dropout_config_values(child, f"{prefix}[{index}]"))
    return output


def audit_dropout_modules(model) -> dict[str, Any]:
    import torch

    modules = [
        {"name": name, "type": module.__class__.__name__, "training": bool(module.training), "p": float(module.p)}
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.dropout._DropoutNd)
    ]
    active = [item for item in modules if item["training"] and item["p"] > 0]
    return {"module_count": len(modules), "modules": modules, "active_modules": active, "rl_dropout_active": bool(active)}


def parameter_trainability(model) -> dict[str, int]:
    lora = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name]
    base = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" not in name]
    return {
        "trainable_lora_param_count": sum(parameter.numel() for _, parameter in lora if parameter.requires_grad),
        "trainable_lora_tensor_count": sum(parameter.requires_grad for _, parameter in lora),
        "base_trainable_param_count": sum(parameter.numel() for _, parameter in base if parameter.requires_grad),
        "base_trainable_tensor_count": sum(parameter.requires_grad for _, parameter in base),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir: Path, physical_gpu_id: int) -> None:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    contract = load_contract()
    provenance = validate_model_provenance(contract)
    if contract.init_family != INIT_FAMILY or contract.beta_gamma_used_as_init:
        raise Phase12C1aError("Beta initialization contract failed")
    artifact = load_phase12b_artifact()
    record = load_pilot_record()
    renderer = BetaGammaRenderer()
    group = reconstruct_business_group(record, artifact, renderer)
    batch = collate_business_group(group, PAD_TOKEN_ID, "right")
    sampled_ids_before = batch.completion_ids.clone()
    if int(batch.completion_mask.sum()) != 24:
        raise Phase12C1aError("frozen G8 token contract failed")

    base_config = AutoConfig.from_pretrained(PRETRAINED_BASE, local_files_only=True, trust_remote_code=True)
    adapter_config = PeftConfig.from_pretrained(BETA_CHECKPOINT, local_files_only=True)
    config_dropout = {
        "base": dropout_config_values(base_config.to_dict()),
        "adapter": dropout_config_values(adapter_config.to_dict()),
    }
    lora_dropout = float(adapter_config.lora_dropout)
    if lora_dropout != EXPECTED_LORA_DROPOUT:
        raise Phase12C1aError(f"unexpected lora_dropout={lora_dropout}")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_name = torch.cuda.get_device_name(device)
    base = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_BASE,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, BETA_CHECKPOINT, is_trainable=True, local_files_only=True).to(device)
    model.eval()
    trainability = parameter_trainability(model)
    dropout_modules = audit_dropout_modules(model)
    if model.training or dropout_modules["rl_dropout_active"]:
        raise Phase12C1aError("eval-mode dropout gate failed")
    if trainability["trainable_lora_param_count"] <= 0 or trainability["base_trainable_param_count"] != 0:
        raise Phase12C1aError(f"trainability gate failed: {trainability}")

    trainable_lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    completion_ids = tuple(candidate.completion_ids for candidate in group.candidates)
    old_score = score_full_sequences(
        model, group.context_ids, completion_ids, PAD_TOKEN_ID, device,
        grad_enabled=False, trainable_parameters=trainable_lora,
    )
    if any(old_score.requires_grad_by_microbatch) or any(old_score.graph_connected_by_microbatch):
        raise Phase12C1aError("no-grad old rescore unexpectedly retained an autograd graph")

    if model.training or audit_dropout_modules(model)["rl_dropout_active"]:
        raise Phase12C1aError("policy mode changed before current forward")
    current_score = score_full_sequences(
        model, group.context_ids, completion_ids, PAD_TOKEN_ID, device,
        grad_enabled=True, trainable_parameters=trainable_lora,
    )
    current_requires_grad = all(current_score.requires_grad_by_microbatch)
    graph_connected = all(current_score.graph_connected_by_microbatch)
    old_cpu = torch.tensor(old_score.logps)
    current_cpu = torch.tensor(current_score.logps)
    sampled_ids_changed = not torch.equal(sampled_ids_before, batch.completion_ids)
    comparison = comparison_stats(old_cpu, current_cpu, batch.completion_mask)
    passed = (
        comparison["abs_max"] <= REPEAT_ABS_MAX_THRESHOLD
        and current_requires_grad
        and graph_connected
        and not sampled_ids_changed
    )
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    audit = {
        "status": "PASS" if passed else "STOPPED_EQUALITY_OR_GRAPH_GATE_FAIL",
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "group_id": GROUP_ID,
        "G": 8,
        "token_count": 24,
        "policy_scoring_mode": POLICY_SCORING_MODE,
        "adapter_config_lora_dropout": lora_dropout,
        "config_dropout": config_dropout,
        "dropout_modules": dropout_modules,
        "trainability": trainability,
        "old_current": comparison,
        "current_logps_requires_grad": current_requires_grad,
        "current_graph_connected_to_trainable_lora": graph_connected,
        "scoring_microbatch_size": SCORING_MICROBATCH_SIZE,
        "full_sequence_rows_scored": 8,
        "sampled_completion_ids_changed": sampled_ids_changed,
        "ppo_old_logp_contract_changed": False,
        "trainer_sha256": TRAINER_SHA256,
        "trainer_modified": False,
        "resource": {"physical_gpu_id": physical_gpu_id, "gpu_name": gpu_name, "peak_reserved_gb": peak_reserved},
        "execution": {"generation_started": False, "backward_started": False, "optimizer_steps": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "deterministic_policy_mode_validation.json", audit)
    write_json(output_dir / "dropout_module_audit.json", {"config_dropout": config_dropout, **dropout_modules})
    (output_dir / "CHATGPT_PHASE1_2C1A_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2C1a deterministic RL policy mode validation: " + audit["status"] + "\n"
        "The trainable Beta LoRA remained in eval mode for no-grad old rescore and autograd-enabled current scoring. No generation, backward, optimizer, Trainer, PPO, Frontier, or HPR contract was changed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"PHASE1_2C1A": audit["status"], "old_current": comparison, "trainability": trainability, "dropout_active": dropout_modules["rl_dropout_active"], "graph_connected": graph_connected, "peak_reserved_gb": peak_reserved}), flush=True)
    if not passed:
        raise Phase12C1aError(f"C1a gate failed: {audit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
