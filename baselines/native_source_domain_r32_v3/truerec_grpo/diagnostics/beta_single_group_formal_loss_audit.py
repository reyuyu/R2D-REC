"""TrueRec Phase 1.2C: one real formal-runtime group through actual Trainer loss."""
from __future__ import annotations

import argparse
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
from beta_single_group_loss_audit import GROUP_ID, PAD_TOKEN_ID, compare_current_old, load_phase12b_artifact, load_pilot_record, reconstruct_business_group, verify_total_composition  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from policy_scoring_v1 import POLICY_SCORING_MODE, parameter_versions  # noqa: E402
from rollout_runtime_v1 import GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE, rescore_business_group_from_completions  # noqa: E402
from truerec_grpo_trainer_v1 import FORMAT_PENALTY_CONTEXT_TERMS, FORMAT_PENALTY_DOMAIN_TERMS, PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP, TrueRecGRPOTrainerV1, gather_padded_action_logps_rows  # noqa: E402
from truerec_loss_v1 import HPR_LAMBDA  # noqa: E402
from truerec_runtime_v1 import build_group_runtime_plan  # noqa: E402


MIN_FREE_GIB = 70.0


class Phase12CFormalLossError(RuntimeError):
    pass


class CurrentLogpCapturePolicy(torch.nn.Module):
    """Delegate one Trainer forward while retaining only its 24 action logps."""
    def __init__(self, policy, batch):
        super().__init__()
        self.policy = policy
        self.batch = batch
        self._current_logp_chunks = []
        self.forward_calls = 0

    @property
    def current_logps(self):
        return torch.cat(self._current_logp_chunks) if self._current_logp_chunks else None

    def forward(self, input_ids, attention_mask):
        start = sum(chunk.shape[0] for chunk in self._current_logp_chunks)
        stop = start + input_ids.shape[0]
        output = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        self.forward_calls += 1
        logits = output.logits if hasattr(output, "logits") else output
        self._current_logp_chunks.append(
            gather_padded_action_logps_rows(logits, self.batch, start, stop).detach().float().cpu()
        )
        return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_outputs(output_dir: Path, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "single_group_formal_loss_audit.json", audit)
    if audit.get("trainer_current_old") is not None:
        write_json(output_dir / "trainer_current_old_ratio.json", audit["trainer_current_old"])
    (output_dir / "CHATGPT_PHASE1_2C_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2C real single-group Frontier + HPR Trainer loss: " + audit["status"] + "\n"
        "The saved G8 IDs were rescored by the formal full-forward old-logp runtime before exactly one attempted Trainer compute_group call. No generation, backward, optimizer, or training ran.\n",
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
    audit: dict[str, Any] = {
        "status": "INITIALIZING",
        "group_id": GROUP_ID,
        "G": 8,
        "token_count": 24,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "policy_mode": POLICY_SCORING_MODE,
        "trainer_current_old": None,
        "losses": None,
        "loss_composition_pass": False,
        "trainer_g8_forward_oom": False,
        "trainer_core_sha256": TRAINER_SHA256,
        "trainer_core_modified": False,
        "resource": {"physical_gpu_id": physical_gpu_id, "gpu_name": gpu_name, "free_memory_at_start_gb": free_start_gib, "total_memory_gb": total_bytes / (1024 ** 3)},
        "execution": {"generation_started": False, "backward_started": False, "optimizer_steps": 0, "next_experiment_started": False},
    }
    if free_start_gib < MIN_FREE_GIB:
        audit["status"] = "STOPPED_GPU_NOT_EMPTY"
        write_outputs(output_dir, audit)
        raise Phase12CFormalLossError(f"FREE_MEMORY_AT_START_GB={free_start_gib}")

    contract = load_contract()
    provenance = validate_model_provenance(contract)
    artifact = load_phase12b_artifact()
    record = load_pilot_record()
    renderer = BetaGammaRenderer()
    saved_group = reconstruct_business_group(record, artifact, renderer)
    completion_ids = tuple(candidate.completion_ids for candidate in saved_group.candidates)
    generation_score_logps = tuple(tuple(float(value) for value in row["old_logps"]) for row in artifact["candidates"])
    if sum(len(row) for row in completion_ids) != 24:
        raise Phase12CFormalLossError("frozen token count failed")

    adapter_config = PeftConfig.from_pretrained(BETA_CHECKPOINT, local_files_only=True)
    if float(adapter_config.lora_dropout) != EXPECTED_LORA_DROPOUT:
        raise Phase12CFormalLossError("adapter dropout contract failed")
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
        raise Phase12CFormalLossError("eval/dropout gate failed")
    if trainability["trainable_lora_param_count"] <= 0 or trainability["base_trainable_param_count"] != 0:
        raise Phase12CFormalLossError("trainability gate failed")

    group = rescore_business_group_from_completions(
        model, record, saved_group.context_ids, completion_ids, generation_score_logps,
        renderer.tokenizer.convert_ids_to_tokens, PAD_TOKEN_ID, device,
    )
    batch = collate_business_group(group, PAD_TOKEN_ID, "right")
    metrics = [candidate.metrics for candidate in group.candidates]
    runtime_plan = build_group_runtime_plan(metrics, group.all_gold_abc, renderer.tokenizer.convert_tokens_to_ids)
    valid_count = sum(bool(item["format_valid"]) for item in metrics)
    a_fail_count = sum(item["frontier"] == "A_FAIL" for item in metrics)
    a_negative = int(runtime_plan.monitoring["frontier_A_negative"])
    a_positive = int(runtime_plan.monitoring["frontier_A_positive"])
    b_c_gated = bool(runtime_plan.token_credit_mask[:, 0].all() and not runtime_plan.token_credit_mask[:, 1:].any())
    if (valid_count, a_fail_count, runtime_plan.hpr.trigger, a_negative, a_positive, b_c_gated) != (8, 8, "HPR_A", 8, 0, True):
        raise Phase12CFormalLossError("mechanical runtime plan gate failed")

    versions_before = parameter_versions(model)
    capture = CurrentLogpCapturePolicy(model, batch).eval()
    trainer = TrueRecGRPOTrainerV1(capture, renderer.tokenizer.convert_tokens_to_ids, PAD_TOKEN_ID, device=device)
    try:
        result = trainer.compute_group(group)
    except torch.OutOfMemoryError:
        mutation = parameter_versions(model) != versions_before
        audit.update({
            "status": "STOPPED_TRAINER_G8_FORWARD_OOM",
            "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
            "dropout": dropout,
            "trainability": trainability,
            "runtime_plan": {"valid_count": valid_count, "a_fail_count": a_fail_count, "hpr_trigger": runtime_plan.hpr.trigger, "A_negative_count": a_negative, "A_positive_count": a_positive, "B_C_credit_gated": b_c_gated},
            "forward_counts": {"train_policy_forward_calls_per_group": trainer.train_policy_forward_calls, "capture_policy_forward_calls": capture.forward_calls, "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls},
            "position_gate": {"train_bridge": False, "domain_direct_target_count": FORMAT_PENALTY_DOMAIN_TERMS, "context_direct_target_count": FORMAT_PENALTY_CONTEXT_TERMS, "pad_direct_target_count": 0},
            "parameter_mutation": mutation,
            "trainer_g8_forward_oom": True,
        })
        audit["resource"].update({"peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3), "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3)})
        write_outputs(output_dir, audit)
        print(json.dumps({"PHASE1_2C": audit["status"], "resource": audit["resource"]}), flush=True)
        raise

    if capture.current_logps is None or capture.current_logps.shape != (8, 3):
        raise Phase12CFormalLossError("Trainer current-logp capture failed")
    comparison = compare_current_old(capture.current_logps, batch.old_logps, batch.completion_mask)
    if comparison["abs_max"] > REPEAT_ABS_MAX_THRESHOLD:
        raise Phase12CFormalLossError(f"TRAINER_CURRENT_OLD_GATE_FAIL={comparison['abs_max']}")
    losses = {
        "frontier_loss": float(result.frontier_loss.detach().cpu()),
        "hpr_loss_raw": float(result.hpr_loss_raw.detach().cpu()),
        "hpr_loss_weighted": float(result.hpr_loss_weighted.detach().cpu()),
        "total_loss": float(result.total_loss.detach().cpu()),
    }
    finite = all(math.isfinite(value) for value in losses.values())
    composition = finite and verify_total_composition(losses["frontier_loss"], losses["hpr_loss_raw"], losses["hpr_loss_weighted"], losses["total_loss"])
    mutation = parameter_versions(model) != versions_before
    passed = (
        composition and not mutation and trainer.logical_policy_scoring_passes == 1
        and trainer.physical_policy_forward_calls == PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP
        and capture.forward_calls == PHYSICAL_POLICY_FORWARD_CALLS_PER_GROUP and trainer.hpr_extra_forward_calls == 0
        and not dropout["rl_dropout_active"]
    )
    audit.update({
        "status": "PASS" if passed else "STOPPED_LOSS_GATE_FAIL",
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "dropout": dropout,
        "trainability": trainability,
        "runtime_plan": {"valid_count": valid_count, "a_fail_count": a_fail_count, "hpr_trigger": runtime_plan.hpr.trigger, "A_negative_count": a_negative, "A_positive_count": a_positive, "B_C_credit_gated": b_c_gated},
        "trainer_current_old": comparison,
        "losses": losses,
        "hpr_lambda": HPR_LAMBDA,
        "loss_composition_pass": composition,
        "forward_counts": {"train_policy_forward_calls_per_group": trainer.train_policy_forward_calls, "capture_policy_forward_calls": capture.forward_calls, "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls},
        "position_gate": {"train_bridge": False, "domain_direct_target_count": FORMAT_PENALTY_DOMAIN_TERMS, "context_direct_target_count": FORMAT_PENALTY_CONTEXT_TERMS, "pad_direct_target_count": 0},
        "parameter_mutation": mutation,
    })
    audit["resource"].update({"peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3), "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3)})
    write_outputs(output_dir, audit)
    print(json.dumps({"PHASE1_2C": audit["status"], "trainer_current_old": comparison, "losses": losses, "resource": audit["resource"]}), flush=True)
    if not passed:
        raise Phase12CFormalLossError(f"formal loss gate failed: {audit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
