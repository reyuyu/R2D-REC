"""Phase 1.2E: exactly one real LoRA-only AdamW step on the frozen Beta G8 group."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "data", TRUE_REC_ROOT / "diagnostics", TRUE_REC_ROOT / "initialization", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE, load_contract, validate_model_provenance  # noqa: E402
from beta_deterministic_policy_mode_validation import EXPECTED_LORA_DROPOUT, audit_dropout_modules, parameter_trainability  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from beta_old_logp_rescore_validation import REPEAT_ABS_MAX_THRESHOLD, TRAINER_SHA256  # noqa: E402
from beta_single_group_loss_audit import GROUP_ID, PAD_TOKEN_ID, compare_current_old, load_phase12b_artifact, load_pilot_record, lora_parameter_sha, reconstruct_business_group  # noqa: E402
from beta_streaming_backward_audit import DetachedCurrentLogpCapture, gradient_audit  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from policy_scoring_v1 import score_full_sequences  # noqa: E402
from rollout_runtime_v1 import GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE, rescore_business_group_from_completions  # noqa: E402
from truerec_grpo_trainer_v1 import TRAINER_MICROBATCH_SIZE, TrueRecGRPOTrainerV1  # noqa: E402


LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.0
EXPECTED_TRAINABLE_LORA_PARAM_COUNT = 87293952
MIN_FREE_GIB = 70.0


class Phase12EValidationError(RuntimeError):
    pass


def build_lora_optimizer(model) -> tuple[torch.optim.AdamW, dict[str, int | bool]]:
    trainable_lora = [
        parameter for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    base_ids = {
        id(parameter) for name, parameter in model.named_parameters()
        if "lora_" not in name
    }
    optimizer = torch.optim.AdamW(trainable_lora, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    lora_ids = {id(parameter) for parameter in trainable_lora}
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    audit = {
        "optimizer_lora_only": optimizer_ids == lora_ids and not (optimizer_ids & base_ids),
        "optimizer_param_count": sum(parameter.numel() for parameter in optimizer_parameters),
        "optimizer_tensor_count": len(optimizer_parameters),
        "duplicate_optimizer_tensor_count": len(optimizer_parameters) - len(optimizer_ids),
    }
    return optimizer, audit


def lora_tensor_shas(model) -> dict[str, str]:
    output = {}
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        output[name] = hashlib.sha256(raw).hexdigest()
    if not output:
        raise Phase12EValidationError("no LoRA tensors found")
    return output


def base_parameter_versions(model) -> dict[str, int]:
    return {
        name: int(parameter._version) for name, parameter in model.named_parameters()
        if "lora_" not in name
    }


def ratio_stats(current: torch.Tensor, old: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    selected_current = current[mask]
    selected_old = old.to(current.device)[mask]
    ratios = torch.exp(selected_current - selected_old)
    values = {
        "mean": float(ratios.mean()),
        "min": float(ratios.min()),
        "max": float(ratios.max()),
        "changed_token_count": int(torch.count_nonzero(ratios != 1.0)),
        "token_count": int(mask.sum()),
    }
    if not all(math.isfinite(value) for key, value in values.items() if key not in {"changed_token_count", "token_count"}):
        raise Phase12EValidationError("post-step ratio contains non-finite values")
    return values


def write_outputs(output_dir: Path, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_beta_one_optimizer_step_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output_dir / "CHATGPT_PHASE1_2E_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2E exactly-one-step audit: " + audit["status"] + "\n"
        "A fresh Beta model used formal old-logp rescore, one G8 streaming backward, and exactly one LoRA-only AdamW step. Post-step scoring reused the frozen completion IDs and old logps. No generation, scheduler, checkpoint, optimizer-state persistence, second step, or formal training ran.\n",
        encoding="utf-8",
    )


def run(output_dir: Path, physical_gpu_id: int) -> None:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_start_gib = free_bytes / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(device)
    if free_start_gib < MIN_FREE_GIB:
        raise Phase12EValidationError(f"FREE_MEMORY_AT_START_GB={free_start_gib}")

    contract = load_contract()
    provenance = validate_model_provenance(contract)
    artifact = load_phase12b_artifact()
    record = load_pilot_record()
    renderer = BetaGammaRenderer()
    saved_group = reconstruct_business_group(record, artifact, renderer)
    completion_ids = tuple(candidate.completion_ids for candidate in saved_group.candidates)
    generation_score_logps = tuple(tuple(float(value) for value in row["old_logps"]) for row in artifact["candidates"])
    if sum(len(row) for row in completion_ids) != 24:
        raise Phase12EValidationError("frozen completion token count failed")

    adapter_config = PeftConfig.from_pretrained(BETA_CHECKPOINT, local_files_only=True)
    if float(adapter_config.lora_dropout) != EXPECTED_LORA_DROPOUT:
        raise Phase12EValidationError("adapter dropout contract failed")
    torch.cuda.reset_peak_memory_stats(device)
    base = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_BASE, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, BETA_CHECKPOINT, is_trainable=True, local_files_only=True).to(device)
    model.eval()
    dropout = audit_dropout_modules(model)
    trainability = parameter_trainability(model)
    if model.training or dropout["rl_dropout_active"]:
        raise Phase12EValidationError("eval/dropout gate failed")
    if (
        trainability["trainable_lora_param_count"] != EXPECTED_TRAINABLE_LORA_PARAM_COUNT
        or trainability["base_trainable_param_count"] != 0
    ):
        raise Phase12EValidationError("trainability gate failed")

    group = rescore_business_group_from_completions(
        model, record, saved_group.context_ids, completion_ids, generation_score_logps,
        renderer.tokenizer.convert_ids_to_tokens, PAD_TOKEN_ID, device,
    )
    batch = collate_business_group(group, PAD_TOKEN_ID, "right")
    optimizer, optimizer_audit = build_lora_optimizer(model)
    if (
        not optimizer_audit["optimizer_lora_only"]
        or optimizer_audit["optimizer_param_count"] != EXPECTED_TRAINABLE_LORA_PARAM_COUNT
        or optimizer_audit["duplicate_optimizer_tensor_count"] != 0
    ):
        raise Phase12EValidationError(f"optimizer parameter gate failed: {optimizer_audit}")

    lora_sha_before, _ = lora_parameter_sha(model)
    lora_tensors_before = lora_tensor_shas(model)
    base_versions_before = base_parameter_versions(model)
    optimizer.zero_grad(set_to_none=True)
    existing_grad_count = sum(
        parameter.grad is not None for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    )
    if existing_grad_count != 0:
        raise Phase12EValidationError("LoRA gradients existed before streaming backward")

    capture = DetachedCurrentLogpCapture(model, batch).eval()
    trainer = TrueRecGRPOTrainerV1(capture, renderer.tokenizer.convert_tokens_to_ids, PAD_TOKEN_ID, device=device)
    trainer.backward_group_streaming(group)
    torch.cuda.synchronize(device)
    allocated_after_backward = torch.cuda.memory_allocated(device) / (1024 ** 3)
    if capture.current_logps is None or capture.current_logps.shape != (8, 3):
        raise Phase12EValidationError("streaming current-logp capture failed")
    initial = compare_current_old(capture.current_logps, batch.old_logps, batch.completion_mask)
    gradients = gradient_audit(model)
    pre_step_gate = (
        initial["abs_max"] <= REPEAT_ABS_MAX_THRESHOLD
        and trainer.physical_policy_forward_calls == 4
        and trainer.streaming_backward_calls == 4
        and trainer.streaming_full_g8_plan_builds == 1
        and trainer.hpr_extra_forward_calls == 0
        and gradients["lora_params_with_nonzero_grad"] > 0
        and math.isfinite(gradients["lora_grad_norm"])
        and gradients["lora_grad_norm"] > 0
        and gradients["base_params_with_grad"] == 0
        and gradients["nan_grad_count"] == 0
        and gradients["inf_grad_count"] == 0
    )
    if not pre_step_gate:
        raise Phase12EValidationError("pre-step ratio/gradient/call gate failed")

    optimizer_steps = 0
    optimizer.step()
    optimizer_steps += 1
    torch.cuda.synchronize(device)
    allocated_after_step = torch.cuda.memory_allocated(device) / (1024 ** 3)
    lora_sha_after, _ = lora_parameter_sha(model)
    lora_tensors_after = lora_tensor_shas(model)
    lora_changed = sum(lora_tensors_before[name] != lora_tensors_after[name] for name in lora_tensors_before)
    base_versions_after = base_parameter_versions(model)
    base_changed = sum(base_versions_before[name] != base_versions_after[name] for name in base_versions_before)
    if optimizer_steps != 1 or lora_sha_before == lora_sha_after or lora_changed <= 0 or base_changed != 0:
        raise Phase12EValidationError("exactly-one-step parameter mutation gate failed")

    model.eval()
    post_score = score_full_sequences(
        model, group.context_ids, completion_ids, PAD_TOKEN_ID, device,
        grad_enabled=False, trainable_parameters=(),
    )
    post_current = torch.tensor(post_score.logps, dtype=torch.float32)
    post_ratio = ratio_stats(post_current, batch.old_logps, batch.completion_mask)
    if post_ratio["changed_token_count"] <= 0:
        raise Phase12EValidationError("post-step sampled-token ratio did not change")

    audit = {
        "status": "PASS",
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "group_id": GROUP_ID,
        "G": 8,
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, **optimizer_audit},
        "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "dropout": dropout,
        "trainability": trainability,
        "lora_params_with_existing_grad": existing_grad_count,
        "forward_counts": {
            "physical_forward_calls": trainer.physical_policy_forward_calls,
            "physical_backward_calls": trainer.streaming_backward_calls,
            "full_g8_plan_builds": trainer.streaming_full_g8_plan_builds,
            "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls,
        },
        "initial_current_old": initial,
        "gradients_before_step": gradients,
        "optimizer_steps": optimizer_steps,
        "lora_sha_before": lora_sha_before,
        "lora_sha_after": lora_sha_after,
        "lora_changed_tensor_count": lora_changed,
        "base_changed_tensor_count": base_changed,
        "post_step_ratio": post_ratio,
        "trainer_sha256": TRAINER_SHA256,
        "resource": {
            "physical_gpu_id": physical_gpu_id, "gpu_name": gpu_name,
            "free_memory_at_start_gb": free_start_gib, "total_memory_gb": total_bytes / (1024 ** 3),
            "allocated_after_backward_gb": allocated_after_backward,
            "allocated_after_optimizer_step_gb": allocated_after_step,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        },
        "execution": {
            "generation_started": False, "real_model_backward_started": True,
            "optimizer_started": True, "optimizer_steps": optimizer_steps,
            "scheduler_started": False, "checkpoint_saved": False,
            "optimizer_state_saved": False, "formal_training_started": False,
            "next_experiment_started": False,
        },
    }
    write_outputs(output_dir, audit)
    print(json.dumps({"PHASE1_2E": "PASS", "initial": initial, "gradients": gradients, "post_step_ratio": post_ratio, "mutation": {"lora_changed": lora_changed, "base_changed": base_changed}, "resource": audit["resource"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
