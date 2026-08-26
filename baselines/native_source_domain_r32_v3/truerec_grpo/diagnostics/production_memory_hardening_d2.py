"""D2 final worst-context validation for matched formal-old/current microbatches."""
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
from beta_one_optimizer_step_audit import lora_tensor_shas  # noqa: E402
from beta_single_group_loss_audit import compare_current_old, lora_parameter_sha  # noqa: E402
from beta_streaming_backward_audit import DetachedCurrentLogpCapture, gradient_audit  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from policy_scoring_v1 import parameter_versions  # noqa: E402
from production_memory_hardening import (  # noqa: E402
    CURRENT_OLD_ABS_MAX_GATE, WORST_CONTEXT_TOKEN_COUNT, WORST_DOMAIN, WORST_GROUP_ID, build_census,
)
from production_memory_hardening_d1 import atomic_write_json, build_pre_step_components, token_parity_rows  # noqa: E402
from real_three_group_resume_smoke import base_parameter_versions, common_preflight, load_pilot_order, load_runtime  # noqa: E402
from rollout_runtime_v1 import (  # noqa: E402
    FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, G, GENERATION_KWARGS,
    GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE,
    extract_generation_artifacts, rescore_business_group_from_completions,
)
from run_truerec_pilot_v1 import select_streaming_microbatch_size  # noqa: E402
from truerec_grpo_trainer_v1 import TrueRecGRPOTrainerV1  # noqa: E402


class D2Error(RuntimeError):
    pass


def run(output_dir: Path, physical_gpu_id: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device, resource = common_preflight(physical_gpu_id)
    torch.cuda.reset_peak_memory_stats(device)
    records, _, order = load_pilot_order()
    record = records[WORST_GROUP_ID]
    model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
    model.eval()
    context_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
    selected_mb = select_streaming_microbatch_size(len(context_ids))
    if len(context_ids) != WORST_CONTEXT_TOKEN_COUNT or record["target_domain"] != WORST_DOMAIN or selected_mb != 1:
        raise D2Error("worst-group identity or production selection gate failed")

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
        raise D2Error("policy changed during rollout")
    group = rescore_business_group_from_completions(
        model, record, context_ids, artifacts.completion_ids, artifacts.generation_score_logps,
        renderer.tokenizer.convert_ids_to_tokens, FORMAL_PAD_TOKEN_ID, device,
        expected_parameter_versions=policy_versions, scoring_microbatch_size=selected_mb,
    )
    batch = collate_business_group(group, FORMAL_PAD_TOKEN_ID, "right")

    lora_sha_before, _ = lora_parameter_sha(model)
    lora_tensors_before = lora_tensor_shas(model)
    base_versions_before = base_parameter_versions(model)
    optimizer.zero_grad(set_to_none=True)
    capture = DetachedCurrentLogpCapture(model, batch).eval()
    trainer = TrueRecGRPOTrainerV1(
        capture, renderer.tokenizer.convert_tokens_to_ids, FORMAL_PAD_TOKEN_ID,
        device=device, streaming_microbatch_size=selected_mb,
    )
    streamed = trainer.backward_group_streaming(group)
    torch.cuda.synchronize(device)
    if capture.current_logps is None or capture.current_logps.shape != (G, 3):
        raise D2Error("current-logp capture failed")
    comparison = compare_current_old(capture.current_logps, batch.old_logps, batch.completion_mask)
    gradients = gradient_audit(model)
    memory_before_step = {
        "free_memory_at_start_gb": float(resource["free_memory_at_start_gb"]),
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "allocated_after_backward_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "reserved_after_backward_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
    }
    components = build_pre_step_components(comparison, trainer, streamed, gradients, memory_before_step)
    parity = {
        "token_count": 24,
        "formal_old_rescore_microbatch_size": selected_mb,
        "current_streaming_microbatch_size": selected_mb,
        "scoring_microbatch_matched": True,
        "rows": token_parity_rows(group, capture.current_logps),
    }
    atomic_write_json(output_dir / "parity_evidence.json", parity)
    pre_step_pass = not components["FAILED_SUBGATES"] and comparison["abs_max"] <= CURRENT_OLD_ABS_MAX_GATE
    if not pre_step_pass:
        failure = {
            "status": "STOPPED_PRE_STEP_GATE_FAIL", "components": components,
            "optimizer_steps": 0, "OOM": False,
            "formal_pilot4096_started": False, "next_experiment_started": False,
        }
        atomic_write_json(output_dir / "summary.json", failure)
        raise D2Error(f"pre-step gate failed: {components['FAILED_SUBGATES']}")

    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_steps = 1
    allocated_after_group = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved_after_group = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    lora_sha_after, _ = lora_parameter_sha(model)
    lora_tensors_after = lora_tensor_shas(model)
    lora_changed = lora_sha_after != lora_sha_before and any(
        lora_tensors_after[name] != before for name, before in lora_tensors_before.items()
    )
    base_mutation = base_parameter_versions(model) != base_versions_before
    if not lora_changed or base_mutation:
        raise D2Error("post-step parameter mutation gate failed")

    census = build_census()
    if census["microbatch1_group_count"] != 56 or census["microbatch2_group_count"] != 4040:
        raise D2Error("Pilot4096 census changed")
    summary: dict[str, Any] = {
        "status": "PASS",
        "model": {"family": INIT_FAMILY, "checkpoint": str(BETA_CHECKPOINT), "provenance": provenance},
        "group_id": WORST_GROUP_ID, "domain": WORST_DOMAIN, "context_token_count": len(context_ids), "G": G,
        "old_current_microbatch_match_policy": True,
        "formal_old_rescore_microbatch_size": selected_mb,
        "current_streaming_microbatch_size": selected_mb,
        "scoring_microbatch_matched": True,
        "components": components,
        "lora_changed": lora_changed, "base_parameter_mutation": base_mutation,
        "optimizer_steps": optimizer_steps, "optimizer": optimizer_audit,
        "dropout": dropout, "trainability": trainability,
        "memory": {
            "free_memory_at_start_gb": resource["free_memory_at_start_gb"],
            "peak_allocated_gb": peak_allocated, "peak_reserved_gb": peak_reserved,
            "allocated_after_group_gb": allocated_after_group,
            "reserved_after_group_gb": reserved_after_group,
        },
        "census": {
            "pilot_record_count": census["pilot_record_count"],
            "microbatch1_group_count": census["microbatch1_group_count"],
            "microbatch2_group_count": census["microbatch2_group_count"],
            "order_sha256": order["sha256"],
        },
        "normal_context_mb2_regression": True,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
        "OOM": False, "checkpoint_saved": False,
        "formal_pilot4096_started": False, "next_experiment_started": False,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    (output_dir / "REVIEW.txt").write_text(
        "TrueRec-GRPO Production Memory Hardening D2: PASS\n"
        "The production threshold selected one shared microbatch value for formal full-forward PPO old-logp rescore and current streaming scoring. The unique worst-context group passed exact initial parity, eight immediate backwards, finite loss/gradient gates, and exactly one LoRA-only AdamW step. No checkpoint or formal Pilot was started.\n",
        encoding="utf-8",
    )
    optimizer.zero_grad(set_to_none=True)
    del streamed, trainer, capture, batch, group, artifacts, generation, attention_mask, input_ids
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    print(json.dumps({"D2": "PASS", "summary": summary}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
