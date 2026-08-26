"""D1 worst-context pre-step subgate and scoring batch-shape diagnostic."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY  # noqa: E402
from beta_one_optimizer_step_audit import lora_tensor_shas  # noqa: E402
from beta_single_group_loss_audit import compare_current_old, lora_parameter_sha, verify_total_composition  # noqa: E402
from beta_streaming_backward_audit import DetachedCurrentLogpCapture, gradient_audit  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from policy_scoring_v1 import SCORING_MICROBATCH_SIZE, parameter_versions, score_full_sequences  # noqa: E402
from production_memory_hardening import (  # noqa: E402
    CURRENT_OLD_ABS_MAX_GATE, WORST_CONTEXT_TOKEN_COUNT, WORST_DOMAIN, WORST_GROUP_ID,
)
from real_three_group_resume_smoke import base_parameter_versions, common_preflight, load_pilot_order, load_runtime  # noqa: E402
from rollout_runtime_v1 import (  # noqa: E402
    FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, G, GENERATION_KWARGS,
    GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE,
    extract_generation_artifacts, rescore_business_group_from_completions,
)
from truerec_grpo_trainer_v1 import TrueRecGRPOTrainerV1  # noqa: E402


CURRENT_STREAMING_MICROBATCH_SIZE = 1


class D1Error(RuntimeError):
    pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def token_parity_rows(group, current_logps: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for candidate_index, candidate in enumerate(group.candidates):
        for action_position, (token_id, old_logp) in enumerate(zip(candidate.completion_ids, candidate.old_logps)):
            current = float(current_logps[candidate_index, action_position])
            difference = abs(current - float(old_logp))
            rows.append({
                "candidate_index": candidate_index,
                "action_position": action_position,
                "token_id": int(token_id),
                "formal_old_mb2_logp": float(old_logp),
                "current_mb1_logp": current,
                "abs_diff": difference,
                "ratio": math.exp(current - float(old_logp)),
            })
    if len(rows) != 24:
        raise D1Error(f"expected 24 sampled-token rows, got {len(rows)}")
    return rows


def build_pre_step_components(
    comparison: dict[str, float], trainer, streamed, gradients: dict[str, Any], resource: dict[str, float],
) -> dict[str, Any]:
    losses = {
        "frontier_value": streamed.frontier_value,
        "hpr_value_raw": streamed.hpr_value_raw,
        "hpr_value_weighted": streamed.hpr_value_weighted,
        "total_value": streamed.total_value,
    }
    loss_composition = verify_total_composition(
        losses["frontier_value"], losses["hpr_value_raw"],
        losses["hpr_value_weighted"], losses["total_value"],
    )
    loss_finite = all(math.isfinite(value) for value in losses.values())
    ratio_pass = comparison["abs_max"] <= CURRENT_OLD_ABS_MAX_GATE
    call_pass = (
        trainer.physical_policy_forward_calls == 8
        and trainer.streaming_backward_calls == 8
        and trainer.streaming_full_g8_plan_builds == 1
        and trainer.hpr_extra_forward_calls == 0
    )
    loss_pass = loss_finite and loss_composition
    gradient_pass = (
        gradients["lora_params_with_nonzero_grad"] > 0
        and math.isfinite(gradients["lora_grad_norm"]) and gradients["lora_grad_norm"] > 0
        and gradients["base_params_with_grad"] == 0
        and gradients["nan_grad_count"] == gradients["inf_grad_count"] == 0
    )
    gates = {
        "GATE_RATIO_PASS": ratio_pass,
        "GATE_CALL_COUNTS_PASS": call_pass,
        "GATE_LOSS_PASS": loss_pass,
        "GATE_GRADIENT_PASS": gradient_pass,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "current_old_abs_mean": comparison["abs_mean"],
        "current_old_abs_max": comparison["abs_max"],
        "initial_ratio_mean": comparison["ratio_mean"],
        "initial_ratio_min": comparison["ratio_min"],
        "initial_ratio_max": comparison["ratio_max"],
        "physical_forward_calls": trainer.physical_policy_forward_calls,
        "physical_backward_calls": trainer.streaming_backward_calls,
        "streaming_full_g8_plan_builds": trainer.streaming_full_g8_plan_builds,
        "hpr_extra_forward_calls": trainer.hpr_extra_forward_calls,
        **losses,
        "loss_composition_pass": loss_composition,
        "loss_finite": loss_finite,
        **gradients,
        "gradient_finite": gradient_pass,
        **resource,
        **gates,
        "FAILED_SUBGATES": failed,
    }


def comparison_stats(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> dict[str, float]:
    differences = [abs(float(a) - float(b)) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b)]
    if len(differences) != 24:
        raise D1Error("diagnostic comparison must contain 24 tokens")
    return {"abs_mean": sum(differences) / len(differences), "abs_max": max(differences)}


def classify_root_cause(components: dict[str, Any], diagnostic: dict[str, Any] | None) -> str:
    if components["GATE_RATIO_PASS"]:
        return "NO_SUBGATE_FAILURE_REPRODUCED" if not components["FAILED_SUBGATES"] else "OTHER_SUBGATE_FAILURE"
    if any(not components[name] for name in ("GATE_CALL_COUNTS_PASS", "GATE_LOSS_PASS", "GATE_GRADIENT_PASS")):
        return "OTHER_SUBGATE_FAILURE"
    if diagnostic is None:
        return "NOT_EXPLAINED_BY_BATCH_SHAPE_ALONE"
    if diagnostic["current_mb1_vs_diagnostic_old_mb1"]["abs_max"] > CURRENT_OLD_ABS_MAX_GATE:
        return "NOT_EXPLAINED_BY_BATCH_SHAPE_ALONE"
    if diagnostic["formal_old_mb2_vs_diagnostic_old_mb1"]["abs_max"] > CURRENT_OLD_ABS_MAX_GATE:
        return "CONFIRMED_BF16_BATCH_SHAPE_SCORING_PARITY"
    return "OTHER_SUBGATE_FAILURE"


def run(output_dir: Path, physical_gpu_id: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device, gpu = common_preflight(physical_gpu_id)
    free_start = float(gpu["free_memory_at_start_gb"])
    known = {"free_memory_at_start_gb": free_start}
    try:
        torch.cuda.reset_peak_memory_stats(device)
        records, _, _ = load_pilot_order()
        record = records[WORST_GROUP_ID]
        model, unused_optimizer, renderer, provenance, dropout, trainability, _ = load_runtime(device)
        del unused_optimizer
        model.eval()
        context_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
        if len(context_ids) != WORST_CONTEXT_TOKEN_COUNT or record["target_domain"] != WORST_DOMAIN:
            raise D1Error("frozen worst-group identity failed")

        policy_versions = parameter_versions(model)
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        generation = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            eos_token_id=list(FORMAL_EOS_TOKEN_IDS), pad_token_id=FORMAL_PAD_TOKEN_ID,
            return_dict_in_generate=True, output_scores=True, **GENERATION_KWARGS,
        )
        artifacts = extract_generation_artifacts(context_ids, generation, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
        if parameter_versions(model) != policy_versions:
            raise D1Error("policy changed during rollout")
        group = rescore_business_group_from_completions(
            model, record, context_ids, artifacts.completion_ids, artifacts.generation_score_logps,
            renderer.tokenizer.convert_ids_to_tokens, FORMAL_PAD_TOKEN_ID, device,
            expected_parameter_versions=policy_versions,
        )
        batch = collate_business_group(group, FORMAL_PAD_TOKEN_ID, "right")
        lora_sha_before, _ = lora_parameter_sha(model)
        lora_tensors_before = lora_tensor_shas(model)
        base_versions_before = base_parameter_versions(model)
        model.zero_grad(set_to_none=True)

        capture = DetachedCurrentLogpCapture(model, batch).eval()
        trainer = TrueRecGRPOTrainerV1(
            capture, renderer.tokenizer.convert_tokens_to_ids, FORMAL_PAD_TOKEN_ID,
            device=device, streaming_microbatch_size=CURRENT_STREAMING_MICROBATCH_SIZE,
        )
        streamed = trainer.backward_group_streaming(group)
        torch.cuda.synchronize(device)
        if capture.current_logps is None or capture.current_logps.shape != (G, 3):
            raise D1Error("current logp capture failed")
        current = capture.current_logps
        comparison = compare_current_old(current, batch.old_logps, batch.completion_mask)
        gradients = gradient_audit(model)
        memory = {
            **known,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            "allocated_after_backward_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
            "reserved_after_backward_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        }
        components = build_pre_step_components(comparison, trainer, streamed, gradients, memory)
        parity_rows = token_parity_rows(group, current)
        maximum = max(parity_rows, key=lambda row: row["abs_diff"])
        token_evidence: dict[str, Any] = {
            "token_count": len(parity_rows), "rows": parity_rows,
            "MAX_DIFF_CANDIDATE": maximum["candidate_index"],
            "MAX_DIFF_ACTION_POSITION": maximum["action_position"],
            "MAX_DIFF_TOKEN_ID": maximum["token_id"],
        }
        # These two files are the pre-aggregate durable boundary.
        atomic_write_json(output_dir / "pre_step_components.json", components)
        atomic_write_json(output_dir / "token_parity.json", token_evidence)

        diagnostic = None
        if not components["GATE_RATIO_PASS"]:
            diagnostic_score = score_full_sequences(
                model, context_ids, artifacts.completion_ids, FORMAL_PAD_TOKEN_ID, device,
                grad_enabled=False, trainable_parameters=(), scoring_microbatch_size=1,
            )
            formal = tuple(candidate.old_logps for candidate in group.candidates)
            current_nested = tuple(
                tuple(float(current[row, position]) for position in range(len(candidate.completion_ids)))
                for row, candidate in enumerate(group.candidates)
            )
            diagnostic = {
                "diagnostic_old_mb1_logps": diagnostic_score.logps,
                "current_mb1_vs_formal_old_mb2": comparison_stats(current_nested, formal),
                "current_mb1_vs_diagnostic_old_mb1": comparison_stats(current_nested, diagnostic_score.logps),
                "formal_old_mb2_vs_diagnostic_old_mb1": comparison_stats(formal, diagnostic_score.logps),
                "used_for_ppo": False,
            }
            for row in token_evidence["rows"]:
                value = diagnostic_score.logps[row["candidate_index"]][row["action_position"]]
                row["diagnostic_old_mb1_logp"] = value
            token_evidence["diagnostic"] = diagnostic
            atomic_write_json(output_dir / "token_parity.json", token_evidence)

        root_cause = classify_root_cause(components, diagnostic)
        lora_sha_after, _ = lora_parameter_sha(model)
        lora_tensors_after = lora_tensor_shas(model)
        lora_mutation = lora_sha_after != lora_sha_before or lora_tensors_after != lora_tensors_before
        base_mutation = base_parameter_versions(model) != base_versions_before
        summary = {
            "status": "PASS_DIAGNOSTIC_COMPLETE",
            "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
            "group_id": WORST_GROUP_ID, "domain": WORST_DOMAIN, "context_token_count": len(context_ids), "G": G,
            "formal_old_rescore_microbatch_size": SCORING_MICROBATCH_SIZE,
            "current_streaming_microbatch_size": CURRENT_STREAMING_MICROBATCH_SIZE,
            "components": components, "diagnostic": diagnostic, "root_cause": root_cause,
            "lora_parameter_mutation": lora_mutation, "base_parameter_mutation": base_mutation,
            "dropout": dropout, "trainability": trainability,
            "optimizer_steps": 0, "OOM": False,
            "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
            "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
            "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
            "formal_pilot4096_started": False, "next_experiment_started": False,
        }
        atomic_write_json(output_dir / "summary.json", summary)
        (output_dir / "REVIEW.txt").write_text(
            f"TrueRec-GRPO Memory Hardening D1: {root_cause}\n"
            "Pre-step components and 24-token evidence were atomically persisted before classification. Formal PPO old logps remained the unchanged MB2 full-forward rescore; any MB1 rescore was diagnostic-only. No optimizer step, checkpoint, or formal Pilot was started.\n",
            encoding="utf-8",
        )
        model.zero_grad(set_to_none=True)
        del streamed, trainer, capture, batch, group, artifacts, generation, attention_mask, input_ids
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        if lora_mutation or base_mutation:
            raise D1Error("parameter mutation detected without optimizer step")
        print(json.dumps({"D1": "PASS", "root_cause": root_cause, "components": components, "diagnostic": diagnostic}, ensure_ascii=False), flush=True)
    except torch.cuda.OutOfMemoryError:
        oom = {
            "status": "STOPPED_OOM", "group_id": WORST_GROUP_ID, "context_token_count": WORST_CONTEXT_TOKEN_COUNT,
            **known, "OOM": True, "optimizer_steps": 0,
            "formal_pilot4096_started": False, "next_experiment_started": False,
        }
        atomic_write_json(output_dir / "summary.json", oom)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
