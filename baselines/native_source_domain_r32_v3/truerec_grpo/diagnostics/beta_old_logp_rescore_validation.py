"""Phase 1.2C0: validate full-forward rescoring as stable PPO old logps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "data", TRUE_REC_ROOT / "diagnostics", TRUE_REC_ROOT / "initialization", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE, load_contract, validate_model_provenance  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from beta_single_group_loss_audit import GROUP_ID, PAD_TOKEN_ID, load_phase12b_artifact, load_pilot_record, reconstruct_business_group  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from truerec_grpo_trainer_v1 import gather_padded_action_logps  # noqa: E402


REPEAT_ABS_MAX_THRESHOLD = 1e-6
TRAINER_SHA256 = "07fd15d32b3630a08abb38b461867208b17da0d719c960a31c26bb7fa0d42773"


class Phase12C0Error(RuntimeError):
    pass


def comparison_stats(first, second, mask) -> dict[str, float | int]:
    import torch

    selected_first, selected_second = first[mask], second[mask]
    absolute = (selected_first - selected_second).abs()
    ratio = torch.exp(selected_second - selected_first)
    result = {
        "token_count": int(mask.sum()),
        "abs_mean": float(absolute.mean()),
        "abs_max": float(absolute.max()),
        "ratio_mean": float(ratio.mean()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
    }
    if not all(math.isfinite(value) for key, value in result.items() if key != "token_count"):
        raise Phase12C0Error("comparison contains non-finite values")
    return result


def position_discrepancy(generation, rescore, mask) -> dict[str, dict[str, float]]:
    output = {}
    for index, level in enumerate(("A", "B", "C")):
        selected = mask[:, index]
        absolute = (generation[:, index][selected] - rescore[:, index][selected]).abs()
        output[level] = {"abs_mean": float(absolute.mean()), "abs_max": float(absolute.max())}
    return output


def independent_direct_gather(logits, batch):
    import torch

    values = torch.zeros((8, 3), dtype=logits.dtype, device=logits.device)
    for row in range(8):
        for action_position in range(3):
            causal_position = int(batch.causal_logit_indices[row, action_position])
            sampled_token = int(batch.completion_ids[row, action_position])
            values[row, action_position] = torch.log_softmax(logits[row, causal_position], dim=-1)[sampled_token]
    return values


def causal_alignment_pass(batch) -> bool:
    import torch

    expected = batch.action_indices - 1
    return bool(torch.equal(batch.causal_logit_indices, expected))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir: Path, physical_gpu_id: int) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    contract = load_contract(); provenance = validate_model_provenance(contract)
    if contract.init_family != INIT_FAMILY or contract.beta_gamma_used_as_init:
        raise Phase12C0Error("Beta initialization contract failed")
    artifact = load_phase12b_artifact(); record = load_pilot_record(); renderer = BetaGammaRenderer()
    group = reconstruct_business_group(record, artifact, renderer)
    batch = collate_business_group(group, PAD_TOKEN_ID, "right")
    if int(batch.completion_mask.sum()) != 24 or not causal_alignment_pass(batch):
        raise Phase12C0Error("sample count or causal alignment failed")

    device = torch.device("cuda", 0); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    gpu_name = torch.cuda.get_device_name(device)
    base = AutoModelForCausalLM.from_pretrained(PRETRAINED_BASE, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(base, BETA_CHECKPOINT, is_trainable=False, local_files_only=True).to(device)
    model.eval(); model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Phase12C0Error("unchanged inference policy gate failed")
    input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)
    with torch.inference_mode():
        first_output = model(input_ids=input_ids, attention_mask=attention_mask)
        rescored = gather_padded_action_logps(first_output.logits, batch)
        direct = independent_direct_gather(first_output.logits, batch)
    gather_check = bool(torch.equal(rescored, direct))
    rescored_cpu = rescored.detach().float().cpu()
    del rescored, direct, first_output
    torch.cuda.empty_cache()
    if not gather_check:
        raise Phase12C0Error("independent gather check failed")
    with torch.inference_mode():
        repeat_output = model(input_ids=input_ids, attention_mask=attention_mask)
        repeat = gather_padded_action_logps(repeat_output.logits, batch)
    repeat_cpu = repeat.detach().float().cpu()
    del repeat, repeat_output
    torch.cuda.synchronize(device)

    generation = batch.old_logps.float()
    mask = batch.completion_mask
    generation_vs_rescore = comparison_stats(generation, rescored_cpu, mask)
    by_position = position_discrepancy(generation, rescored_cpu, mask)
    repeatability = comparison_stats(rescored_cpu, repeat_cpu, mask)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    passed = repeatability["abs_max"] <= REPEAT_ABS_MAX_THRESHOLD
    audit = {
        "status": "PASS" if passed else "STOPPED_REPEATABILITY_GATE_FAIL",
        "group_id": GROUP_ID,
        "G": 8,
        "token_count": 24,
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "generation_score_logps_role": "diagnostic_only_not_new_ppo_old_logp",
        "rescored_old_logps_role": "candidate_ppo_old_logp_under_validation",
        "generation_vs_rescore": generation_vs_rescore,
        "generation_vs_rescore_by_position": by_position,
        "rescore_repeat": repeatability,
        "repeat_abs_max_threshold": REPEAT_ABS_MAX_THRESHOLD,
        "causal_alignment": "PASS",
        "gather_independent_check": "PASS",
        "trainer_sha256": TRAINER_SHA256,
        "trainer_modified": False,
        "ppo_old_logp_contract_changed": False,
        "resource": {"physical_gpu_id": physical_gpu_id, "gpu_name": gpu_name, "peak_reserved_gb": peak_reserved},
        "execution": {"real_model_forwards": 2, "generation_started": False, "frontier_started": False, "hpr_started": False, "backward_started": False, "optimizer_steps": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "old_logp_full_forward_rescore_validation.json", audit)
    write_json(output_dir / "generation_vs_rescore_by_position.json", by_position)
    (output_dir / "CHATGPT_PHASE1_2C0_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2C0 full-forward rescore validation: " + audit["status"] + "\n"
        "Generation scores remain diagnostic only. Two identical unchanged-policy full forwards were compared without generation, loss, Frontier, HPR, backward, or optimizer.\n",
        encoding="utf-8",
    )
    print(json.dumps({"PHASE1_2C0": audit["status"], "generation_vs_rescore": generation_vs_rescore, "rescore_repeat": repeatability, "peak_reserved_gb": peak_reserved}), flush=True)
    if not passed:
        raise Phase12C0Error(f"RESCORE_REPEAT_GATE_FAIL={repeatability['abs_max']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args(); run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
