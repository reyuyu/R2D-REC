"""Phase 1.2C1b: audit the promoted full-forward PPO old-logp runtime path."""
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
from beta_deterministic_policy_mode_validation import EXPECTED_LORA_DROPOUT, audit_dropout_modules, parameter_trainability  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from beta_old_logp_rescore_validation import REPEAT_ABS_MAX_THRESHOLD, TRAINER_SHA256, comparison_stats  # noqa: E402
from beta_single_group_loss_audit import GROUP_ID, PAD_TOKEN_ID, load_phase12b_artifact, load_pilot_record, reconstruct_business_group  # noqa: E402
from policy_scoring_v1 import POLICY_SCORING_MODE, SCORING_MICROBATCH_SIZE, score_full_sequences  # noqa: E402
from rollout_runtime_v1 import GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, OLD_LOGPS_FROM_FULL_FORWARD_RESCORE, PPO_OLD_LOGP_SOURCE, rescore_business_group_from_completions  # noqa: E402


class Phase12C1bError(RuntimeError):
    pass


def rollout_metadata(group) -> dict[str, Any]:
    return {
        "recommendation_group_id": group.recommendation_group_id,
        "context_ids": list(group.context_ids),
        "fixed_domain_token": group.fixed_domain_token,
        "all_gold_abc": list(group.all_gold_abc),
        "candidates": [
            {"sample_index": candidate.sample_index, "completion_ids": list(candidate.completion_ids), "metrics": candidate.metrics}
            for candidate in group.candidates
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir: Path, physical_gpu_id: int) -> None:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    contract = load_contract()
    provenance = validate_model_provenance(contract)
    if contract.init_family != INIT_FAMILY or contract.beta_gamma_used_as_init:
        raise Phase12C1bError("Beta initialization contract failed")
    artifact = load_phase12b_artifact()
    record = load_pilot_record()
    renderer = BetaGammaRenderer()
    saved_group = reconstruct_business_group(record, artifact, renderer)
    completion_ids = tuple(candidate.completion_ids for candidate in saved_group.candidates)
    generation_score_logps = tuple(tuple(float(value) for value in row["old_logps"]) for row in artifact["candidates"])
    metadata_before = rollout_metadata(saved_group)
    if sum(len(row) for row in completion_ids) != 24:
        raise Phase12C1bError("frozen G8 token contract failed")

    adapter_config = PeftConfig.from_pretrained(BETA_CHECKPOINT, local_files_only=True)
    if float(adapter_config.lora_dropout) != EXPECTED_LORA_DROPOUT:
        raise Phase12C1bError("unexpected adapter dropout")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_name = torch.cuda.get_device_name(device)
    base = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_BASE, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, BETA_CHECKPOINT, is_trainable=True, local_files_only=True).to(device)
    model.eval()
    trainability = parameter_trainability(model)
    dropout = audit_dropout_modules(model)
    if model.training or dropout["rl_dropout_active"]:
        raise Phase12C1bError("eval/dropout gate failed")
    if trainability["trainable_lora_param_count"] <= 0 or trainability["base_trainable_param_count"] != 0:
        raise Phase12C1bError("trainability gate failed")

    rescored_group = rescore_business_group_from_completions(
        model, record, saved_group.context_ids, completion_ids, generation_score_logps,
        renderer.tokenizer.convert_ids_to_tokens, PAD_TOKEN_ID, device,
    )
    metadata_after = rollout_metadata(rescored_group)
    metadata_changed = metadata_before != metadata_after
    sampled_ids_changed = completion_ids != tuple(candidate.completion_ids for candidate in rescored_group.candidates)
    old = torch.tensor([candidate.old_logps for candidate in rescored_group.candidates], dtype=torch.float32)
    diagnostic = torch.tensor([candidate.generation_score_logps for candidate in rescored_group.candidates], dtype=torch.float32)
    diagnostic_disagreement_count = int(torch.count_nonzero(old != diagnostic))
    if diagnostic_disagreement_count == 0:
        raise Phase12C1bError("saved generation diagnostics unexpectedly equal full-forward rescore")

    trainable_lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    current_score = score_full_sequences(
        model, rescored_group.context_ids, completion_ids, PAD_TOKEN_ID, device,
        grad_enabled=True, trainable_parameters=trainable_lora,
    )
    current = torch.tensor(current_score.logps, dtype=torch.float32)
    mask = torch.ones((8, 3), dtype=torch.bool)
    comparison = comparison_stats(old, current, mask)
    current_requires_grad = all(current_score.requires_grad_by_microbatch)
    graph_connected = all(current_score.graph_connected_by_microbatch)
    passed = (
        comparison["abs_max"] <= REPEAT_ABS_MAX_THRESHOLD
        and current_requires_grad and graph_connected
        and not sampled_ids_changed and not metadata_changed
        and all(tuple(candidate.old_logps) != tuple(candidate.generation_score_logps) for candidate in rescored_group.candidates)
    )
    audit = {
        "status": "PASS" if passed else "STOPPED_CONTRACT_GATE_FAIL",
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "group_id": GROUP_ID,
        "G": 8,
        "token_count": 24,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "old_logps_from_full_forward_rescore": OLD_LOGPS_FROM_FULL_FORWARD_RESCORE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
        "policy_scoring_mode": POLICY_SCORING_MODE,
        "scoring_microbatch_size": SCORING_MICROBATCH_SIZE,
        "dropout": {"adapter_config_lora_dropout": float(adapter_config.lora_dropout), **dropout},
        "trainability": trainability,
        "current_logps_requires_grad": current_requires_grad,
        "current_graph_connected_to_trainable_lora": graph_connected,
        "sampled_completion_ids_changed": sampled_ids_changed,
        "rollout_metadata_changed": metadata_changed,
        "generation_vs_rescore_disagreement_count": diagnostic_disagreement_count,
        "old_current": comparison,
        "trainer_sha256": TRAINER_SHA256,
        "trainer_core_modified": False,
        "resource": {"physical_gpu_id": physical_gpu_id, "gpu_name": gpu_name, "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3), "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3)},
        "execution": {"generation_started": False, "frontier_started": False, "hpr_started": False, "backward_started": False, "optimizer_steps": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ppo_old_logp_runtime_contract_audit.json", audit)
    write_json(output_dir / "old_current_ratio.json", comparison)
    (output_dir / "CHATGPT_PHASE1_2C1B_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2C1b full-forward PPO old-logp runtime contract: " + audit["status"] + "\n"
        "The formal runtime helper populated candidate.old_logps from eval/no-grad full-sequence rescore; generation score logps remained diagnostic only. No generation, Frontier, HPR, backward, optimizer, or training ran.\n",
        encoding="utf-8",
    )
    print(json.dumps({"PHASE1_2C1B": audit["status"], "old_current": comparison, "generation_rescore_disagreement_count": diagnostic_disagreement_count, "peak_reserved_gb": audit["resource"]["peak_reserved_gb"]}), flush=True)
    if not passed:
        raise Phase12C1bError(f"C1b gate failed: {audit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
