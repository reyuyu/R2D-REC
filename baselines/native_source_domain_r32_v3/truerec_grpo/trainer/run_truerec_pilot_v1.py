"""Single-GPU production runner for ordered TrueRec Pilot4096 training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from beta_single_group_loss_audit import lora_parameter_sha  # noqa: E402
from beta_streaming_backward_audit import gradient_audit  # noqa: E402
from checkpoint_v1 import (  # noqa: E402
    DEFAULT_ORDER_SEED, PILOT_RECORDS_SHA256, load_checkpoint, save_checkpoint_atomic,
)
from policy_scoring_v1 import parameter_versions  # noqa: E402
from real_three_group_resume_smoke import (  # noqa: E402
    DATASET_IDENTITY, LoraCheckpointState, ORDER_SHA256, base_parameter_versions,
    common_preflight, load_pilot_order, load_runtime,
)
from rollout_runtime_v1 import (  # noqa: E402
    FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, G, GENERATION_KWARGS,
    GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE,
    extract_generation_artifacts, rescore_business_group_from_completions,
)
from training_driver_v1 import (  # noqa: E402
    HPR_LAMBDA, KL_BETA, LEARNING_RATE, TRAINER_MICROBATCH_SIZE, WEIGHT_DECAY,
    TrueRecTrainingDriverV1, frozen_contract,
)
from truerec_grpo_trainer_v1 import TrueRecGRPOTrainerV1  # noqa: E402
from truerec_runtime_v1 import build_group_runtime_plan  # noqa: E402


RUN_ID_TRAIN20 = "TRUEREC-V1-BETA-TRAIN20"
PHASE13C_CHECKPOINT = Path("/data/GRPO/truerec_grpo/checkpoints/phase1_3c_step2")
SOURCE_COMMIT_MARKER = Path("/data/GRPO/.source_commit")
ALLOCATED_DRIFT_LIMIT_GB = 2.0
MODEL_CONTEXT_KEYS = ("max_position_embeddings", "model_max_length", "max_sequence_length")


class ProductionTrainingError(RuntimeError):
    pass


def run_index_loop(
    order: Sequence[str], *, start_index: int, stop_exclusive: int,
    execute_group: Callable[[int, str], Any], checkpoint_every: int,
    save_checkpoint: Callable[[int], None], append_record: Callable[[Any], None],
) -> list[Any]:
    if not 0 <= start_index <= stop_exclusive <= len(order):
        raise ProductionTrainingError("invalid ordered training interval")
    if checkpoint_every < 1:
        raise ProductionTrainingError("checkpoint cadence must be positive")
    outputs = []
    for group_index in range(start_index, stop_exclusive):
        output = execute_group(group_index, order[group_index])
        append_record(output)
        outputs.append(output)
        completed = group_index + 1
        if completed % checkpoint_every == 0 or completed == stop_exclusive:
            save_checkpoint(completed)
    return outputs


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_frozen_manifest(path: Path, manifest: dict[str, Any], *, resume: bool) -> None:
    serialized = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if resume:
        if not path.is_file() or path.read_text(encoding="utf-8") != serialized:
            raise ProductionTrainingError("resume run manifest differs from frozen manifest")
        return
    if path.exists():
        raise ProductionTrainingError("fresh run manifest already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def read_existing_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def validate_existing_prefix(records: list[dict[str, Any]], order: Sequence[str], cursor: int) -> None:
    if len(records) != cursor:
        raise ProductionTrainingError("metrics row count does not match resume cursor")
    for index, record in enumerate(records):
        if record["group_index"] != index or record["recommendation_group_id"] != order[index]:
            raise ProductionTrainingError("metrics history is not the frozen order prefix")


def context_length_census(records_by_id, order, renderer, legal_context_window: int) -> dict[str, Any]:
    lengths = []
    max_group = None
    max_domain = None
    max_length = -1
    for group_id in order:
        record = records_by_id[group_id]
        length = len(renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"]))
        lengths.append(length)
        if length > max_length:
            max_length, max_group, max_domain = length, group_id, record["target_domain"]
    values = np.asarray(lengths, dtype=np.int64)
    result = {
        "count": len(lengths), "min": int(values.min()), "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": int(values.max()), "max_context_group_id": max_group,
        "max_context_domain": max_domain, "legal_context_window": legal_context_window,
        "truncation_applied": False,
    }
    if result["max"] > legal_context_window:
        raise ProductionTrainingError(f"context length exceeds model window: {result}")
    return result


def model_context_window() -> int:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(PRETRAINED_BASE, local_files_only=True, trust_remote_code=True)
    values = [int(getattr(config, key)) for key in MODEL_CONTEXT_KEYS if getattr(config, key, None)]
    if not values:
        raise ProductionTrainingError("model context window unavailable")
    return max(values)


def build_manifest(run_id, physical_gpu_id, resource, provenance, context_census, max_groups, checkpoint_every):
    import peft
    import transformers
    return {
        "RUN_ID": run_id,
        "git_commit": SOURCE_COMMIT_MARKER.read_text(encoding="utf-8").strip(),
        "GRPO_INIT_FAMILY": INIT_FAMILY,
        "beta_checkpoint_path": str(BETA_CHECKPOINT),
        "beta_adapter_sha256": provenance["required_files"]["adapter_model.safetensors"]["sha256"],
        "beta_adapter_config_sha256": provenance["required_files"]["adapter_config.json"]["sha256"],
        "pilot_records_sha256": PILOT_RECORDS_SHA256,
        "order_seed": DEFAULT_ORDER_SEED, "order_sha256": ORDER_SHA256, "order_count": 4096,
        "frozen_contract": frozen_contract(),
        "G": G, "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "optimizer": {"family": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        "HPR_LAMBDA": HPR_LAMBDA, "KL_BETA": KL_BETA,
        "generation": {**GENERATION_KWARGS, "eos_token_id": list(FORMAL_EOS_TOKEN_IDS), "pad_token_id": FORMAL_PAD_TOKEN_ID},
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
        "max_groups": max_groups, "checkpoint_every": checkpoint_every,
        "cuda": {"physical_gpu_id": physical_gpu_id, **resource},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "context_length_census": context_census,
        "start_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase13c_checkpoint_used": False,
        "ddp": False,
    }


def quantile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run(args) -> None:
    if args.run_id == RUN_ID_TRAIN20 and args.max_groups != 20:
        raise ProductionTrainingError("formal Train20 must stop exclusively at group 20")
    resume = args.resume_from_checkpoint is not None
    if resume and Path(args.resume_from_checkpoint).resolve() == PHASE13C_CHECKPOINT.resolve():
        raise ProductionTrainingError("Phase1.3C validation checkpoint cannot initialize formal training")
    device, resource = common_preflight(args.physical_gpu_id)
    records_by_id, order, order_info = load_pilot_order()
    if order_info != {"seed": DEFAULT_ORDER_SEED, "sha256": ORDER_SHA256, "count": 4096}:
        raise ProductionTrainingError("dataset/order identity gate failed")
    renderer_for_census = BetaGammaRenderer()
    context_census = context_length_census(records_by_id, order, renderer_for_census, model_context_window())
    model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
    output_dir = Path(args.output_dir)
    manifest_path, metrics_path = output_dir / "run_manifest.json", output_dir / "train_groups.jsonl"
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ProductionTrainingError("fresh output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        if not manifest_path.is_file():
            raise ProductionTrainingError("resume manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_expected = {
            "RUN_ID": args.run_id, "pilot_records_sha256": PILOT_RECORDS_SHA256,
            "order_seed": DEFAULT_ORDER_SEED, "order_sha256": ORDER_SHA256,
            "frozen_contract": frozen_contract(), "G": G,
            "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
            "HPR_LAMBDA": HPR_LAMBDA, "KL_BETA": KL_BETA,
            "max_groups": args.max_groups, "checkpoint_every": args.checkpoint_every,
        }
        if any(manifest.get(key) != value for key, value in frozen_expected.items()):
            raise ProductionTrainingError("resume manifest frozen fields mismatch")
    else:
        manifest = build_manifest(
            args.run_id, args.physical_gpu_id, resource, provenance, context_census,
            args.max_groups, args.checkpoint_every,
        )
    write_frozen_manifest(manifest_path, manifest, resume=resume)

    trace = {"rollout": None, "group": None, "timing": {}, "gradient": None}
    trainer = TrueRecGRPOTrainerV1(model, renderer.tokenizer.convert_tokens_to_ids, FORMAL_PAD_TOKEN_ID, device=device)

    def rollout(record):
        started = time.perf_counter(); model.eval()
        context_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
        versions = parameter_versions(model)
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            eos_token_id=list(FORMAL_EOS_TOKEN_IDS), pad_token_id=FORMAL_PAD_TOKEN_ID,
            return_dict_in_generate=True, output_scores=True, **GENERATION_KWARGS,
        )
        if parameter_versions(model) != versions:
            raise ProductionTrainingError("policy changed during rollout")
        artifacts = extract_generation_artifacts(context_ids, output, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
        torch.cuda.synchronize(device)
        trace["timing"]["rollout_seconds"] = time.perf_counter() - started
        trace["rollout"] = {"context_ids": context_ids, "artifacts": artifacts, "versions": versions}
        return trace["rollout"]

    def old_rescore(record, rollout_value):
        started = time.perf_counter(); artifacts = rollout_value["artifacts"]
        group = rescore_business_group_from_completions(
            model, record, rollout_value["context_ids"], artifacts.completion_ids,
            artifacts.generation_score_logps, renderer.tokenizer.convert_ids_to_tokens,
            FORMAL_PAD_TOKEN_ID, device, expected_parameter_versions=rollout_value["versions"],
        )
        torch.cuda.synchronize(device)
        trace["timing"]["old_rescore_seconds"] = time.perf_counter() - started
        trace["group"] = group
        return group

    def gradient_gate():
        audit = gradient_audit(model); trace["gradient"] = audit
        return (
            audit["lora_params_with_nonzero_grad"] > 0 and math.isfinite(audit["lora_grad_norm"])
            and audit["lora_grad_norm"] > 0 and audit["base_params_with_grad"] == 0
            and audit["nan_grad_count"] == 0 and audit["inf_grad_count"] == 0
        )

    driver = TrueRecTrainingDriverV1(
        rollout_fn=rollout, old_rescore_fn=old_rescore, trainer=trainer, optimizer=optimizer,
        gradient_finite_fn=gradient_gate, policy_fingerprint_fn=lambda: parameter_versions(model),
        groups_per_optimizer_step=1,
    )
    start_index = 0
    if resume:
        cursor = load_checkpoint(
            args.resume_from_checkpoint, model=LoraCheckpointState(model), optimizer=optimizer, driver=driver,
            current_dataset_identity=DATASET_IDENTITY, current_group_ids=list(records_by_id), cuda_device=device,
        )
        start_index = cursor["next_group_index"]
    existing = read_existing_metrics(metrics_path)
    validate_existing_prefix(existing, order, start_index)
    lora_sha_start, _ = lora_parameter_sha(model)
    base_versions_start = base_parameter_versions(model)
    lora_versions = tuple(int(parameter._version) for name, parameter in model.named_parameters() if "lora_" in name)
    checkpoint_records = []
    allocated_baseline = existing[0]["allocated_after_group_gb"] if existing else None
    over_limit_streak = 0
    run_started = time.perf_counter()

    def execute_group(group_index: int, group_id: str):
        nonlocal lora_versions, allocated_baseline, over_limit_streak
        record = records_by_id[group_id]
        trace["timing"] = {}; trace["gradient"] = None; trace["group"] = None; trace["rollout"] = None
        torch.cuda.reset_peak_memory_stats(device)
        group_started = time.perf_counter()
        forward_before, backward_before = trainer.physical_policy_forward_calls, trainer.streaming_backward_calls
        result = driver.run_group(record)
        torch.cuda.synchronize(device)
        total_seconds = time.perf_counter() - group_started
        backward_optimizer_seconds = total_seconds - trace["timing"]["rollout_seconds"] - trace["timing"]["old_rescore_seconds"]
        new_lora_versions = tuple(int(parameter._version) for name, parameter in model.named_parameters() if "lora_" in name)
        if len(new_lora_versions) != len(lora_versions) or not any(a != b for a, b in zip(lora_versions, new_lora_versions)):
            raise ProductionTrainingError("LoRA did not mutate after optimizer step")
        lora_versions = new_lora_versions
        if base_parameter_versions(model) != base_versions_start:
            raise ProductionTrainingError("base parameter mutation detected")
        monitoring = result.backward_result.monitoring
        if (
            trainer.physical_policy_forward_calls - forward_before != 4
            or trainer.streaming_backward_calls - backward_before != 4
            or trainer.hpr_extra_forward_calls != 0
        ):
            raise ProductionTrainingError("streaming physical call contract changed")
        losses = result.backward_result
        if not all(math.isfinite(value) for value in (losses.frontier_value, losses.hpr_value_raw, losses.hpr_value_weighted, losses.total_value)):
            raise ProductionTrainingError("non-finite detached loss")
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        if allocated_baseline is None: allocated_baseline = allocated
        over_limit_streak = over_limit_streak + 1 if allocated - allocated_baseline > ALLOCATED_DRIFT_LIMIT_GB else 0
        if over_limit_streak >= 2:
            raise ProductionTrainingError("allocated memory shows sustained graph-leak drift")
        group = trace["group"]
        candidate_metrics = [candidate.metrics for candidate in group.candidates]
        runtime_plan = build_group_runtime_plan(candidate_metrics, group.all_gold_abc, renderer.tokenizer.convert_tokens_to_ids)
        valid_abc = [item["parsed_abc"] for item in candidate_metrics if item["format_valid"]]
        parsed_tokens = [tuple(re.findall(r"<s_[abc]_\d+>", value)) for value in valid_abc]
        record_out = {
            "global_step": driver.state.global_step, "group_index": group_index,
            "recommendation_group_id": group_id, "target_domain": record["target_domain"],
            "context_token_count": len(group.context_ids), "Gold_K": len(record["all_gold_abc"]),
            "format_valid_rate": sum(item["format_valid"] for item in candidate_metrics) / G,
            "A_hit_rate": sum(item["A_hit"] for item in candidate_metrics) / G,
            "AB_hit_rate": sum(item["AB_hit"] for item in candidate_metrics) / G,
            "exact_rate": sum(item["exact"] for item in candidate_metrics) / G,
            "GROUP_ANY_A": int(any(item["A_hit"] for item in candidate_metrics)),
            "GROUP_ANY_AB": int(any(item["AB_hit"] for item in candidate_metrics)),
            "GROUP_ANY_EXACT": int(any(item["exact"] for item in candidate_metrics)),
            "HPR_trigger": runtime_plan.hpr.trigger,
            "frontier_A_positive": runtime_plan.monitoring["frontier_A_positive"],
            "frontier_A_negative": runtime_plan.monitoring["frontier_A_negative"],
            "frontier_B_positive": runtime_plan.monitoring["frontier_B_positive"],
            "frontier_B_negative": runtime_plan.monitoring["frontier_B_negative"],
            "frontier_C_positive": runtime_plan.monitoring["frontier_C_positive"],
            "frontier_C_negative": runtime_plan.monitoring["frontier_C_negative"],
            "frontier_value": losses.frontier_value, "hpr_value_raw": losses.hpr_value_raw,
            "hpr_value_weighted": losses.hpr_value_weighted, "total_value": losses.total_value,
            "wrong_history_copy_rate": sum(item["wrong_history_copy"] for item in candidate_metrics) / G,
            "unique_A_per_G8": len({tokens[:1] for tokens in parsed_tokens}),
            "unique_AB_per_G8": len({tokens[:2] for tokens in parsed_tokens}),
            "unique_ABC_per_G8": len(set(valid_abc)),
            "gradient_norm": trace["gradient"]["lora_grad_norm"],
            "allocated_after_group_gb": allocated,
            "reserved_after_group_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            "wall_time_seconds": total_seconds, "rollout_seconds": trace["timing"]["rollout_seconds"],
            "old_rescore_seconds": trace["timing"]["old_rescore_seconds"],
            "backward_optimizer_seconds": backward_optimizer_seconds,
        }
        print(json.dumps({"TRAIN_GROUP": group_index, "global_step": driver.state.global_step, "total": losses.total_value, "seconds": total_seconds}), flush=True)
        return record_out

    def save_at_step(completed: int):
        checkpoint_dir = output_dir / "checkpoints" / f"checkpoint-step-{completed}"
        save_checkpoint_atomic(
            checkpoint_dir, model=LoraCheckpointState(model), optimizer=optimizer, driver=driver,
            dataset_identity=DATASET_IDENTITY, epoch=0, next_group_index=completed,
            order=order_info, cuda_device=device,
        )
        checkpoint_sha, _ = lora_parameter_sha(model)
        checkpoint_records.append({"step": completed, "path": str(checkpoint_dir), "cursor": completed, "lora_sha256": checkpoint_sha})

    new_records = run_index_loop(
        order, start_index=start_index, stop_exclusive=args.max_groups,
        execute_group=execute_group, checkpoint_every=args.checkpoint_every,
        save_checkpoint=save_at_step, append_record=lambda value: append_jsonl(metrics_path, value),
    )
    all_records = existing + new_records
    if len(all_records) != args.max_groups or [row["recommendation_group_id"] for row in all_records] != order[:args.max_groups]:
        raise ProductionTrainingError("final group sequence is not exact frozen prefix")
    lora_sha_end, _ = lora_parameter_sha(model)
    if lora_sha_start == lora_sha_end or base_parameter_versions(model) != base_versions_start:
        raise ProductionTrainingError("final parameter mutation gate failed")
    walls = [row["wall_time_seconds"] for row in all_records]
    total_wall = time.perf_counter() - run_started
    allocated = [row["allocated_after_group_gb"] for row in all_records]
    summary = {
        "RUN_ID": args.run_id, "status": "PASS", "fresh_beta_start": not resume,
        "phase13c_checkpoint_used": False, "pilot_records_sha256": PILOT_RECORDS_SHA256,
        "order_sha256": ORDER_SHA256, "group_index_start": start_index,
        "group_index_stop_exclusive": args.max_groups, "total_groups": len(all_records),
        "total_rollouts": driver.state.rollouts_completed, "total_optimizer_steps": driver.state.optimizer_steps,
        "checkpoints": checkpoint_records, "lora_sha_start": lora_sha_start, "lora_sha_end": lora_sha_end,
        "base_parameter_mutation": False, "allocated_after_group_gb": allocated,
        "allocated_baseline_gb": allocated[0], "allocated_max_after_group_gb": max(allocated),
        "allocated_drift_gb": max(allocated) - allocated[0],
        "max_peak_allocated_gb": max(row["peak_allocated_gb"] for row in all_records),
        "max_peak_reserved_gb": max(row["peak_reserved_gb"] for row in all_records),
        "OOM": False, "total_wall_seconds": total_wall,
        "mean_seconds_per_group": statistics.fmean(walls), "median_seconds_per_group": statistics.median(walls),
        "p90_seconds_per_group": quantile(walls, 90),
        "mean_rollout_seconds": statistics.fmean(row["rollout_seconds"] for row in all_records),
        "mean_old_rescore_seconds": statistics.fmean(row["old_rescore_seconds"] for row in all_records),
        "mean_backward_optimizer_seconds": statistics.fmean(row["backward_optimizer_seconds"] for row in all_records),
        "single_gpu_pilot4096_projected_hours": (total_wall / len(new_records)) * 4096 / 3600,
        "projection_is_linear_not_guaranteed": True,
        "context_length_census": context_census,
        "train_metric_summary": {
            key: statistics.fmean(row[key] for row in all_records)
            for key in ("format_valid_rate", "A_hit_rate", "AB_hit_rate", "exact_rate", "wrong_history_copy_rate", "total_value")
        },
        "ddp_started": False, "evaluation_started": False, "formal_pilot4096_started": False,
        "next_experiment_started": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "REVIEW.txt").write_text(
        "TrueRec-GRPO Formal Train20 Production Probe: PASS\n"
        "Fresh Beta-Baseline trained exactly frozen Pilot4096 order indices [0,20), with 20 live G8 rollouts, full-forward old-logp rescoring, streaming backward, and 20 AdamW steps. Checkpoints were saved at optimizer boundaries 10 and 20. No evaluation, DDP, group 20 (the 21st group), or full Pilot4096 run started.\n",
        encoding="utf-8",
    )
    print(json.dumps({"FORMAL_TRAIN20": "PASS", "summary": summary}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
