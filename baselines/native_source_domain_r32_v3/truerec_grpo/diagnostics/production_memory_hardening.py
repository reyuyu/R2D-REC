"""Pilot4096 adaptive-microbatch census and one worst-context real validation."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from beta_one_optimizer_step_audit import build_lora_optimizer, lora_tensor_shas  # noqa: E402
from beta_single_group_loss_audit import compare_current_old, lora_parameter_sha, verify_total_composition  # noqa: E402
from beta_streaming_backward_audit import DetachedCurrentLogpCapture, gradient_audit  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from policy_scoring_v1 import parameter_versions  # noqa: E402
from real_three_group_resume_smoke import base_parameter_versions, common_preflight, load_pilot_order, load_runtime  # noqa: E402
from rollout_runtime_v1 import (  # noqa: E402
    FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, G, GENERATION_KWARGS,
    GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE,
    extract_generation_artifacts, rescore_business_group_from_completions,
)
from run_truerec_pilot_v1 import context_length_census, model_context_window, select_streaming_microbatch_size  # noqa: E402
from training_driver_v1 import LONG_CONTEXT_THRESHOLD_TOKENS  # noqa: E402
from truerec_grpo_trainer_v1 import (  # noqa: E402
    LONG_CONTEXT_STREAMING_MICROBATCH_SIZE, TRAINER_MICROBATCH_SIZE, TrueRecGRPOTrainerV1,
)


WORST_GROUP_ID = "67fe812d1e74a632a4b3d4cd8db79aa46b9d7a8494adb7022280d594fae3a21e"
WORST_CONTEXT_TOKEN_COUNT = 3489
WORST_DOMAIN = "video"
CURRENT_OLD_ABS_MAX_GATE = 1e-6


class MemoryHardeningError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_census() -> dict[str, Any]:
    records, order, _ = load_pilot_order()
    census = context_length_census(records, order, BetaGammaRenderer(), model_context_window())
    if (
        census["count"] != 4096
        or census["microbatch1_group_count"] + census["microbatch2_group_count"] != 4096
        or census["max_context_group_id"] != WORST_GROUP_ID
        or census["max"] != WORST_CONTEXT_TOKEN_COUNT
        or census["max_context_domain"] != WORST_DOMAIN
    ):
        raise MemoryHardeningError(f"Pilot4096 census gate failed: {census}")
    return {
        "status": "PASS",
        "pilot_record_count": census["count"],
        "default_streaming_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "long_context_streaming_microbatch_size": LONG_CONTEXT_STREAMING_MICROBATCH_SIZE,
        "microbatch1_group_count": census["microbatch1_group_count"],
        "microbatch2_group_count": census["microbatch2_group_count"],
        "context_length_census": census,
        "group_order_modified": False,
    }


def run_census(output_dir: Path) -> None:
    census = build_census()
    write_json(output_dir / "census.json", census)
    print(json.dumps({"PILOT4096_MEMORY_CENSUS": "PASS", **census}, ensure_ascii=False), flush=True)


def run_gpu(output_dir: Path, physical_gpu_id: int) -> None:
    device, resource = common_preflight(physical_gpu_id)
    torch.cuda.reset_peak_memory_stats(device)
    records, _, _ = load_pilot_order()
    record = records[WORST_GROUP_ID]
    model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
    model.eval()
    context_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
    selected_mb = select_streaming_microbatch_size(len(context_ids))
    if len(context_ids) != WORST_CONTEXT_TOKEN_COUNT or record["target_domain"] != WORST_DOMAIN or selected_mb != 1:
        raise MemoryHardeningError("worst-context identity or microbatch-selection gate failed")

    versions_before_rollout = parameter_versions(model)
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        eos_token_id=list(FORMAL_EOS_TOKEN_IDS), pad_token_id=FORMAL_PAD_TOKEN_ID,
        return_dict_in_generate=True, output_scores=True, **GENERATION_KWARGS,
    )
    if parameter_versions(model) != versions_before_rollout:
        raise MemoryHardeningError("policy changed during rollout")
    artifacts = extract_generation_artifacts(context_ids, output, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
    group = rescore_business_group_from_completions(
        model, record, context_ids, artifacts.completion_ids, artifacts.generation_score_logps,
        renderer.tokenizer.convert_ids_to_tokens, FORMAL_PAD_TOKEN_ID, device,
        expected_parameter_versions=versions_before_rollout,
    )
    batch = collate_business_group(group, FORMAL_PAD_TOKEN_ID, "right")

    lora_sha_before, _ = lora_parameter_sha(model)
    lora_tensors_before = lora_tensor_shas(model)
    base_versions_before = base_parameter_versions(model)
    optimizer.zero_grad(set_to_none=True)
    forward_before = 0
    backward_before = 0
    capture = DetachedCurrentLogpCapture(model, batch).eval()
    trainer = TrueRecGRPOTrainerV1(
        capture, renderer.tokenizer.convert_tokens_to_ids, FORMAL_PAD_TOKEN_ID,
        device=device, streaming_microbatch_size=selected_mb,
    )
    streamed = trainer.backward_group_streaming(group)
    torch.cuda.synchronize(device)
    if capture.current_logps is None or capture.current_logps.shape != (G, 3):
        raise MemoryHardeningError("sampled-token current-logp capture failed")
    comparison = compare_current_old(capture.current_logps, batch.old_logps, batch.completion_mask)
    losses = {
        "frontier": streamed.frontier_value,
        "hpr_raw": streamed.hpr_value_raw,
        "hpr_weighted": streamed.hpr_value_weighted,
        "total": streamed.total_value,
    }
    loss_finite = all(math.isfinite(value) for value in losses.values()) and verify_total_composition(
        losses["frontier"], losses["hpr_raw"], losses["hpr_weighted"], losses["total"],
    )
    gradients = gradient_audit(model)
    gradient_finite = (
        gradients["lora_params_with_nonzero_grad"] > 0
        and math.isfinite(gradients["lora_grad_norm"]) and gradients["lora_grad_norm"] > 0
        and gradients["base_params_with_grad"] == 0
        and gradients["nan_grad_count"] == gradients["inf_grad_count"] == 0
    )
    pre_step_gate = (
        comparison["abs_max"] <= CURRENT_OLD_ABS_MAX_GATE
        and capture.forward_calls == trainer.physical_policy_forward_calls == 8
        and trainer.streaming_backward_calls == 8
        and trainer.streaming_full_g8_plan_builds == 1
        and trainer.hpr_extra_forward_calls == 0
        and loss_finite and gradient_finite
    )
    if not pre_step_gate:
        raise MemoryHardeningError("pre-step ratio/loss/gradient/call gate failed")

    optimizer.step()
    torch.cuda.synchronize(device)
    lora_sha_after, _ = lora_parameter_sha(model)
    lora_tensors_after = lora_tensor_shas(model)
    lora_changed = lora_sha_after != lora_sha_before and any(
        lora_tensors_after[name] != before for name, before in lora_tensors_before.items()
    )
    base_mutation = base_parameter_versions(model) != base_versions_before
    if not lora_changed or base_mutation:
        raise MemoryHardeningError("optimizer mutation gate failed")

    optimizer.zero_grad(set_to_none=True)
    del streamed, trainer, capture, batch, group, artifacts, output, attention_mask, input_ids
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    resource.update({
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "allocated_after_group_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "reserved_after_group_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
    })
    result = {
        "status": "PASS",
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "group_id": WORST_GROUP_ID, "domain": WORST_DOMAIN, "context_token_count": WORST_CONTEXT_TOKEN_COUNT,
        "G": G, "streaming_microbatch_size": selected_mb,
        "physical_forward_calls": 8, "physical_backward_calls": 8, "hpr_extra_forward_calls": 0,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
        "current_old": comparison, "losses": losses, "loss_finite": loss_finite,
        "gradients": gradients, "gradient_finite": gradient_finite,
        "lora_changed": lora_changed, "base_parameter_mutation": base_mutation,
        "dropout": dropout, "trainability": trainability, "optimizer": optimizer_audit,
        "optimizer_steps": 1, "resource": resource, "OOM": False,
        "formal_pilot4096_started": False, "ddp_started": False, "evaluation_started": False,
        "next_experiment_started": False,
    }
    write_json(output_dir / "summary.json", result)
    (output_dir / "REVIEW.txt").write_text(
        "TrueRec-GRPO production memory hardening worst-context validation: PASS\n"
        "The unique Pilot4096 maximum-context group ran the formal live G8 rollout, unchanged-policy full-forward old-logp rescore, eight immediate microbatch=1 backwards, and exactly one LoRA-only AdamW step. No checkpoint, formal Pilot4096, DDP, or evaluation ran.\n",
        encoding="utf-8",
    )
    print(json.dumps({"PRODUCTION_MEMORY_HARDENING_GPU": "PASS", **result}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("census", "gpu"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int)
    args = parser.parse_args()
    if args.mode == "census":
        run_census(args.output_dir)
    else:
        if args.physical_gpu_id is None:
            parser.error("--physical-gpu-id is required for GPU mode")
        run_gpu(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
