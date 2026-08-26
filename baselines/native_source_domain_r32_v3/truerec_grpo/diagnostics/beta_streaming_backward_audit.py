"""Phase 1.2D: real Beta G8 streaming backward with zero optimizer steps."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import torch


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "data", TRUE_REC_ROOT / "diagnostics", TRUE_REC_ROOT / "initialization", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE, load_contract, validate_model_provenance  # noqa: E402
from beta_deterministic_policy_mode_validation import EXPECTED_LORA_DROPOUT, audit_dropout_modules, parameter_trainability  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from beta_old_logp_rescore_validation import REPEAT_ABS_MAX_THRESHOLD, TRAINER_SHA256  # noqa: E402
from beta_single_group_loss_audit import GROUP_ID, PAD_TOKEN_ID, compare_current_old, load_phase12b_artifact, load_pilot_record, lora_parameter_sha, reconstruct_business_group, verify_total_composition  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from policy_scoring_v1 import parameter_versions  # noqa: E402
from rollout_runtime_v1 import GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE, rescore_business_group_from_completions  # noqa: E402
from truerec_grpo_trainer_v1 import TRAINER_MICROBATCH_SIZE, TrueRecGRPOTrainerV1, gather_padded_action_logps_rows  # noqa: E402
from truerec_loss_v1 import HPR_LAMBDA  # noqa: E402
from truerec_runtime_v1 import build_group_runtime_plan  # noqa: E402


MIN_FREE_GIB = 70.0


class Phase12DValidationError(RuntimeError):
    pass


class DetachedCurrentLogpCapture(torch.nn.Module):
    """Delegate chunk forwards while retaining only detached sampled-token logps."""
    def __init__(self, policy, batch):
        super().__init__()
        self.policy = policy
        self.batch = batch
        self._chunks: list[torch.Tensor] = []
        self.forward_calls = 0

    @property
    def current_logps(self) -> torch.Tensor | None:
        return torch.cat(self._chunks) if self._chunks else None

    def forward(self, input_ids, attention_mask):
        start = sum(chunk.shape[0] for chunk in self._chunks)
        stop = start + input_ids.shape[0]
        output = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output
        self._chunks.append(
            gather_padded_action_logps_rows(logits, self.batch, start, stop).detach().float().cpu()
        )
        self.forward_calls += 1
        return output


@contextmanager
def audit_backward_returns(after_backward: Callable[[], None]):
    """Observe completed Tensor.backward calls without changing their semantics."""
    original = torch.Tensor.backward

    def wrapped(tensor, *args, **kwargs):
        result = original(tensor, *args, **kwargs)
        after_backward()
        return result

    torch.Tensor.backward = wrapped
    try:
        yield
    finally:
        torch.Tensor.backward = original


def gradient_audit(model) -> dict[str, float | int]:
    lora = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    base = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" not in name]
    with_grad = nonzero = nan_count = inf_count = 0
    norm_squared = 0.0
    for _, parameter in lora:
        gradient = parameter.grad
        if gradient is None:
            continue
        with_grad += 1
        nan_count += int(torch.isnan(gradient).sum())
        inf_count += int(torch.isinf(gradient).sum())
        if int(torch.count_nonzero(gradient)) > 0:
            nonzero += 1
        norm = float(torch.linalg.vector_norm(gradient.detach().float()))
        norm_squared += norm * norm
    return {
        "lora_params_with_grad": with_grad,
        "lora_params_with_nonzero_grad": nonzero,
        "lora_grad_norm": math.sqrt(norm_squared),
        "base_params_with_grad": sum(parameter.grad is not None for _, parameter in base),
        "nan_grad_count": nan_count,
        "inf_grad_count": inf_count,
    }


def write_outputs(output_dir: Path, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_beta_streaming_backward_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output_dir / "CHATGPT_PHASE1_2D_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2D real Beta streaming-backward audit: " + audit["status"] + "\n"
        "The frozen G8 completions were rescored through the formal full-forward old-logp runtime, then one formal streaming group executed four immediate microbatch backwards. No generation, optimizer, checkpoint, or parameter update ran.\n",
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
        raise Phase12DValidationError(f"FREE_MEMORY_AT_START_GB={free_start_gib}")

    contract = load_contract()
    provenance = validate_model_provenance(contract)
    artifact = load_phase12b_artifact()
    record = load_pilot_record()
    renderer = BetaGammaRenderer()
    saved_group = reconstruct_business_group(record, artifact, renderer)
    completion_ids = tuple(candidate.completion_ids for candidate in saved_group.candidates)
    generation_score_logps = tuple(tuple(float(value) for value in row["old_logps"]) for row in artifact["candidates"])
    if sum(len(row) for row in completion_ids) != 24:
        raise Phase12DValidationError("frozen sampled-token count failed")

    adapter_config = PeftConfig.from_pretrained(BETA_CHECKPOINT, local_files_only=True)
    if float(adapter_config.lora_dropout) != EXPECTED_LORA_DROPOUT:
        raise Phase12DValidationError("adapter dropout contract failed")
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
        raise Phase12DValidationError("eval/dropout gate failed")
    if trainability["trainable_lora_param_count"] <= 0 or trainability["base_trainable_param_count"] != 0:
        raise Phase12DValidationError("trainability gate failed")

    group = rescore_business_group_from_completions(
        model, record, saved_group.context_ids, completion_ids, generation_score_logps,
        renderer.tokenizer.convert_ids_to_tokens, PAD_TOKEN_ID, device,
    )
    batch = collate_business_group(group, PAD_TOKEN_ID, "right")
    runtime = build_group_runtime_plan(
        [candidate.metrics for candidate in group.candidates], group.all_gold_abc,
        renderer.tokenizer.convert_tokens_to_ids,
    )
    valid_count = sum(bool(candidate.metrics["format_valid"]) for candidate in group.candidates)
    a_fail_count = sum(candidate.metrics["frontier"] == "A_FAIL" for candidate in group.candidates)
    if (
        valid_count != 8 or a_fail_count != 8 or runtime.hpr.trigger != "HPR_A"
        or runtime.monitoring["frontier_A_negative"] != 8
        or runtime.monitoring["frontier_A_positive"] != 0
    ):
        raise Phase12DValidationError("live full-G8 runtime plan gate failed")

    model.zero_grad(set_to_none=True)
    trainable_lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    if not trainable_lora or any(parameter.grad is not None for parameter in trainable_lora):
        raise Phase12DValidationError("pre-backward LoRA gradient clear gate failed")
    lora_sha_before, lora_tensor_count = lora_parameter_sha(model)
    versions_before = parameter_versions(model)
    allocated_after_chunk: list[float] = []

    def after_backward() -> None:
        torch.cuda.synchronize(device)
        allocated_after_chunk.append(torch.cuda.memory_allocated(device) / (1024 ** 3))

    capture = DetachedCurrentLogpCapture(model, batch).eval()
    trainer = TrueRecGRPOTrainerV1(capture, renderer.tokenizer.convert_tokens_to_ids, PAD_TOKEN_ID, device=device)
    with audit_backward_returns(after_backward):
        streamed = trainer.backward_group_streaming(group)
    if capture.current_logps is None or capture.current_logps.shape != (8, 3):
        raise Phase12DValidationError("streaming current-logp capture failed")
    comparison = compare_current_old(capture.current_logps, batch.old_logps, batch.completion_mask)
    if comparison["abs_max"] > REPEAT_ABS_MAX_THRESHOLD:
        raise Phase12DValidationError(f"CURRENT_OLD_GATE_FAIL={comparison['abs_max']}")
    losses = {
        "frontier_value": streamed.frontier_value,
        "hpr_value_raw": streamed.hpr_value_raw,
        "hpr_value_weighted": streamed.hpr_value_weighted,
        "total_value": streamed.total_value,
    }
    composition = all(math.isfinite(value) for value in losses.values()) and verify_total_composition(
        losses["frontier_value"], losses["hpr_value_raw"], losses["hpr_value_weighted"], losses["total_value"],
    )
    gradients = gradient_audit(model)
    gradient_gate = (
        gradients["lora_params_with_grad"] > 0
        and gradients["lora_params_with_nonzero_grad"] > 0
        and math.isfinite(gradients["lora_grad_norm"])
        and gradients["lora_grad_norm"] > 0
        and gradients["base_params_with_grad"] == 0
        and gradients["nan_grad_count"] == 0
        and gradients["inf_grad_count"] == 0
    )
    lora_sha_after, after_tensor_count = lora_parameter_sha(model)
    mutation = lora_sha_before != lora_sha_after or lora_tensor_count != after_tensor_count or versions_before != parameter_versions(model)
    call_gate = (
        capture.forward_calls == 4 and trainer.physical_policy_forward_calls == 4
        and trainer.streaming_backward_calls == 4 and trainer.streaming_full_g8_plan_builds == 1
        and trainer.hpr_extra_forward_calls == 0 and len(allocated_after_chunk) == 4
    )
    passed = composition and gradient_gate and not mutation and call_gate
    audit = {
        "status": "PASS" if passed else "STOPPED_VALIDATION_GATE_FAIL",
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "group_id": GROUP_ID,
        "G": 8,
        "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "dropout": dropout,
        "trainability": trainability,
        "forward_counts": {
            "physical_forward_calls": trainer.physical_policy_forward_calls,
            "physical_backward_calls": trainer.streaming_backward_calls,
            "full_g8_plan_builds": trainer.streaming_full_g8_plan_builds,
            "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls,
        },
        "current_old": comparison,
        "runtime_plan": {
            "valid_count": valid_count, "a_fail_count": a_fail_count,
            "hpr_trigger": runtime.hpr.trigger,
            "A_negative_count": runtime.monitoring["frontier_A_negative"],
            "A_positive_count": runtime.monitoring["frontier_A_positive"],
        },
        "losses": losses,
        "loss_composition_pass": composition,
        "gradients": gradients,
        "lora_parameter_sha_before": lora_sha_before,
        "lora_parameter_sha_after": lora_sha_after,
        "parameter_mutation": mutation,
        "trainer_sha256": TRAINER_SHA256,
        "resource": {
            "physical_gpu_id": physical_gpu_id, "gpu_name": gpu_name,
            "free_memory_at_start_gb": free_start_gib, "total_memory_gb": total_bytes / (1024 ** 3),
            "allocated_after_chunk_gb": allocated_after_chunk,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        },
        "execution": {
            "generation_started": False, "real_model_backward_started": True,
            "optimizer_started": False, "optimizer_steps": 0,
            "checkpoint_saved": False, "next_experiment_started": False,
        },
    }
    write_outputs(output_dir, audit)
    print(json.dumps({"PHASE1_2D": audit["status"], "current_old": comparison, "losses": losses, "gradients": gradients, "resource": audit["resource"]}), flush=True)
    if not passed:
        raise Phase12DValidationError(f"Phase1.2D gate failed: {audit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
