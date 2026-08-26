"""Phase 1.2C: real shared-forward loss audit over the frozen Phase 1.2B rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "analysis", TRUE_REC_ROOT / "data", TRUE_REC_ROOT / "initialization", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE, load_contract, validate_model_provenance  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer, file_sha  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from rollout_metrics import assess_candidate  # noqa: E402
from rollout_runtime_v1 import BusinessGroupRollout, G, RolloutCandidate  # noqa: E402
from truerec_grpo_trainer_v1 import FORMAT_PENALTY_CONTEXT_TERMS, FORMAT_PENALTY_DOMAIN_TERMS, TrueRecGRPOTrainerV1, gather_padded_action_logps  # noqa: E402
from truerec_loss_v1 import HPR_LAMBDA  # noqa: E402
from truerec_runtime_v1 import build_group_runtime_plan  # noqa: E402


GROUP_ID = "22b28028744f58137b7a768de0d0898a890a4cf26dbc37af29658192900fb80c"
PHASE12B_ARTIFACT = Path("/data/GRPO/truerec_grpo/results/phase1_2b/single_group_beta_g8_rollout.json")
PHASE12B_ARTIFACT_SHA = "2b4339b937e31bedffa9aa1fd08cf349cc8ae07eb8bd096ad84675199d643401"
PILOT = Path("/data/GRPO/truerec_grpo/data/pilot4096/pilot4096_records.jsonl")
PILOT_SHA = "ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879"
PAD_TOKEN_ID = 151643
CURRENT_OLD_MAX_THRESHOLD = 0.05


class Phase12CError(RuntimeError):
    pass


def load_phase12b_artifact(path: Path = PHASE12B_ARTIFACT) -> dict[str, Any]:
    if file_sha(path) != PHASE12B_ARTIFACT_SHA:
        raise Phase12CError("Phase1.2B artifact SHA mismatch")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validate_phase12b_artifact(artifact)
    return artifact


def validate_phase12b_artifact(artifact: dict[str, Any]) -> None:
    candidates = artifact["candidates"]
    if artifact["group"]["recommendation_group_id"] != GROUP_ID or len(candidates) != G:
        raise Phase12CError("Phase1.2B group/G8 gate failed")
    if sum(item["actual_completion_length"] for item in candidates) != 24:
        raise Phase12CError("Phase1.2B sampled token count mismatch")
    if not all(item["format_valid"] and item["frontier"] == "A_FAIL" for item in candidates):
        raise Phase12CError("Phase1.2B expected 8 valid A_FAIL candidates")
    if any(len(item["raw_token_ids"]) != 3 or len(item["old_logps"]) != 3 for item in candidates):
        raise Phase12CError("Phase1.2B IDs/old-logp shape mismatch")
    if artifact["model"]["family"] != INIT_FAMILY:
        raise Phase12CError("old logps do not originate from Beta-Baseline rollout policy")


def load_pilot_record(path: Path = PILOT) -> dict[str, Any]:
    if file_sha(path) != PILOT_SHA:
        raise Phase12CError("Pilot4096 SHA mismatch")
    matches = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if GROUP_ID in line]
    exact = [row for row in matches if row["recommendation_group_id"] == GROUP_ID]
    if len(exact) != 1:
        raise Phase12CError("frozen group record count mismatch")
    return exact[0]


def reconstruct_business_group(record: dict[str, Any], artifact: dict[str, Any], renderer) -> BusinessGroupRollout:
    context_ids = tuple(renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"]))
    domain_ids = renderer.encode(record["fixed_domain_token"])
    if domain_ids != [176247] or context_ids[-1] != 176247:
        raise Phase12CError("fixed prod domain is not terminal token 176247")
    candidates = []
    for saved in artifact["candidates"]:
        ids = tuple(int(value) for value in saved["raw_token_ids"])
        metrics = assess_candidate(list(ids), renderer.tokenizer.convert_ids_to_tokens, record["all_gold_abc"], record["fixed_domain_token"], record["history_sids"])
        if any(metrics[key] != saved[key] for key in ("format_valid", "parsed_abc", "A_hit", "AB_hit", "exact", "frontier")):
            raise Phase12CError("saved parser diagnostics do not reconstruct")
        candidates.append(RolloutCandidate(int(saved["sample_index"]), ids, tuple(float(value) for value in saved["old_logps"]), metrics))
    return BusinessGroupRollout(GROUP_ID, context_ids, record["fixed_domain_token"], tuple(record["all_gold_abc"]), tuple(candidates))


def compare_current_old(current, old, mask) -> dict[str, float]:
    import torch

    selected_current = current[mask]
    selected_old = old.to(current.device)[mask]
    difference = (selected_current - selected_old).abs()
    ratios = torch.exp(selected_current - selected_old)
    values = {
        "abs_mean": float(difference.mean().detach().cpu()),
        "abs_max": float(difference.max().detach().cpu()),
        "ratio_mean": float(ratios.mean().detach().cpu()),
        "ratio_min": float(ratios.min().detach().cpu()),
        "ratio_max": float(ratios.max().detach().cpu()),
        "token_count": int(mask.sum()),
    }
    if not all(math.isfinite(value) for key, value in values.items() if key != "token_count"):
        raise Phase12CError("current/old comparison contains non-finite values")
    return values


def verify_total_composition(frontier: float, hpr_raw: float, hpr_weighted: float, total: float) -> bool:
    return math.isclose(hpr_weighted, HPR_LAMBDA * hpr_raw, rel_tol=1e-6, abs_tol=1e-7) and math.isclose(total, frontier + HPR_LAMBDA * hpr_raw, rel_tol=1e-6, abs_tol=1e-7)


def lora_parameter_sha(model) -> tuple[str, int]:
    digest = hashlib.sha256(); count = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        digest.update(name.encode("utf-8")); digest.update(parameter.detach().contiguous().view(__import__("torch").uint8).cpu().numpy().tobytes()); count += 1
    if not count:
        raise Phase12CError("no LoRA parameters found")
    return digest.hexdigest(), count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir: Path, physical_gpu_id: int) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    contract = load_contract(); provenance = validate_model_provenance(contract)
    artifact = load_phase12b_artifact(); record = load_pilot_record(); renderer = BetaGammaRenderer()
    group = reconstruct_business_group(record, artifact, renderer)
    batch = collate_business_group(group, PAD_TOKEN_ID, "right")
    if not bool(batch.completion_mask.all()) or batch.input_ids.shape[0] != G:
        raise Phase12CError("exact sampled-sequence reconstruction failed")
    runtime_plan = build_group_runtime_plan([candidate.metrics for candidate in group.candidates], group.all_gold_abc, renderer.tokenizer.convert_tokens_to_ids)
    if runtime_plan.hpr.trigger != "HPR_A":
        raise Phase12CError(f"live HPR trigger mismatch: {runtime_plan.hpr.trigger}")
    if runtime_plan.monitoring["frontier_A_negative"] != 8 or runtime_plan.monitoring["frontier_A_positive"] != 0:
        raise Phase12CError("A Frontier count mismatch")
    if not bool(runtime_plan.token_credit_mask[:, 0].all()) or bool(runtime_plan.token_credit_mask[:, 1:].any()):
        raise Phase12CError("Frontier direct position gate failed")
    if any(position != 0 for site in runtime_plan.hpr.sites for _, position in site.onpolicy_positions):
        raise Phase12CError("HPR_A did not use only A causal positions")

    device = torch.device("cuda", 0); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    gpu_name = torch.cuda.get_device_name(device)
    base = AutoModelForCausalLM.from_pretrained(PRETRAINED_BASE, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(base, BETA_CHECKPOINT, is_trainable=False, local_files_only=True).to(device)
    model.eval(); model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Phase12CError("inference-only model state failed")
    parameter_sha_before, lora_parameter_tensors = lora_parameter_sha(model)

    with torch.inference_mode():
        independent_output = model(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
        independent_current = gather_padded_action_logps(independent_output.logits, batch)
    comparison = compare_current_old(independent_current, batch.old_logps, batch.completion_mask.to(device))
    del independent_current, independent_output
    torch.cuda.empty_cache()
    if comparison["abs_max"] > CURRENT_OLD_MAX_THRESHOLD:
        raise Phase12CError(f"CURRENT_OLD_GATE_FAIL={comparison['abs_max']}")

    trainer = TrueRecGRPOTrainerV1(model, renderer.tokenizer.convert_tokens_to_ids, PAD_TOKEN_ID, device=device)
    with torch.inference_mode():
        result = trainer.compute_group(group)
    losses = {
        "frontier_loss": float(result.frontier_loss.detach().cpu()),
        "hpr_loss_raw": float(result.hpr_loss_raw.detach().cpu()),
        "hpr_loss_weighted": float(result.hpr_loss_weighted.detach().cpu()),
        "total_loss": float(result.total_loss.detach().cpu()),
    }
    if not all(math.isfinite(value) for value in losses.values()) or not verify_total_composition(losses["frontier_loss"], losses["hpr_loss_raw"], losses["hpr_loss_weighted"], losses["total_loss"]):
        raise Phase12CError(f"loss composition/nonfinite gate failed: {losses}")
    parameter_sha_after, after_count = lora_parameter_sha(model)
    if parameter_sha_before != parameter_sha_after or lora_parameter_tensors != after_count:
        raise Phase12CError("LoRA parameter mutation detected")
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    loss_audit = {
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "group_id": GROUP_ID, "G": G, "sampled_token_count": int(batch.completion_mask.sum()),
        "rollout_artifact_sha256": PHASE12B_ARTIFACT_SHA,
        "rollout_policy_family": artifact["model"]["family"],
        "prefix": {"train_bridge": False, "fixed_domain_terminal_token": True, "fixed_domain_token_id": 176247},
        "current_old": comparison,
        "frontier": {"A_negative_count": runtime_plan.monitoring["frontier_A_negative"], "A_positive_count": runtime_plan.monitoring["frontier_A_positive"], "B_C_gated": True},
        "hpr": {"trigger": runtime_plan.hpr.trigger, "target_count": sum(len(site.target_token_ids) for site in runtime_plan.hpr.sites), "lambda": HPR_LAMBDA, "plan_source": "live current Beta G8 rollout"},
        "losses": losses,
        "forward_counts": {"independent_consistency_audit": 1, "train_policy_forward_calls_per_group": trainer.train_policy_forward_calls, "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls},
        "position_gate": {"domain_direct_target_count": FORMAT_PENALTY_DOMAIN_TERMS, "context_direct_target_count": FORMAT_PENALTY_CONTEXT_TERMS, "pad_direct_target_count": 0, "pass": True},
        "parameter_mutation": False, "lora_parameter_tensors": lora_parameter_tensors, "lora_sha_before": parameter_sha_before, "lora_sha_after": parameter_sha_after,
        "resource": {"physical_gpu_id": physical_gpu_id, "gpu_name": gpu_name, "peak_reserved_gb": peak_reserved},
        "execution": {"real_model_forward_started": True, "generation_started": False, "training_loss_forward_started": True, "backward_started": False, "optimizer_started": False, "optimizer_steps": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "single_group_shared_forward_loss_audit.json", loss_audit)
    write_json(output_dir / "current_old_logp_consistency.json", comparison)
    write_json(output_dir / "loss_position_gate.json", loss_audit["position_gate"])
    write_json(output_dir / "parameter_mutation_audit.json", {key: loss_audit[key] for key in ("parameter_mutation", "lora_parameter_tensors", "lora_sha_before", "lora_sha_after")})
    (output_dir / "CHATGPT_PHASE1_2C_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2C real shared-forward loss audit: PASS\n"
        "The frozen Phase1.2B Beta G8 IDs/old-logps were reconstructed without generation. Independent current-logp consistency passed before one formal trainer compute_group forward.\n"
        "Live Frontier/HPR, loss composition, direct positions, and LoRA immutability passed. No backward, optimizer, or training ran.\n",
        encoding="utf-8",
    )
    print(json.dumps({"PHASE1_2C": "PASS", "group_id": GROUP_ID, "current_old_abs_max": comparison["abs_max"], "losses": losses, "peak_reserved_gb": peak_reserved}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args(); run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
