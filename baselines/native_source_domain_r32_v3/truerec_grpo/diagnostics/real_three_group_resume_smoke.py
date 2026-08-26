"""Phase 1.3C real single-GPU two-process, three-business-group smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, INIT_FAMILY, PRETRAINED_BASE, load_contract, validate_model_provenance  # noqa: E402
from beta_deterministic_policy_mode_validation import EXPECTED_LORA_DROPOUT, audit_dropout_modules, parameter_trainability  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer, file_sha  # noqa: E402
from beta_one_optimizer_step_audit import build_lora_optimizer  # noqa: E402
from beta_single_group_loss_audit import lora_parameter_sha  # noqa: E402
from beta_streaming_backward_audit import gradient_audit  # noqa: E402
from checkpoint_v1 import (  # noqa: E402
    DEFAULT_ORDER_SEED, PILOT_RECORDS_SHA256, load_checkpoint, order_metadata, save_checkpoint_atomic,
)
from policy_scoring_v1 import parameter_versions  # noqa: E402
from rollout_runtime_v1 import (  # noqa: E402
    FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, GENERATION_KWARGS,
    GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE,
    extract_generation_artifacts, rescore_business_group_from_completions,
)
from training_driver_v1 import TrueRecTrainingDriverV1  # noqa: E402
from truerec_grpo_trainer_v1 import TRAINER_MICROBATCH_SIZE, TrueRecGRPOTrainerV1  # noqa: E402


PILOT = Path("/data/GRPO/truerec_grpo/data/pilot4096/pilot4096_records.jsonl")
MIN_FREE_GIB = 70.0
ORDER_SHA256 = "b4bf02f9b0591e1e684fa1e42e4528682848c615994e926aa33236d78337b4f6"
RUN_SEED = 1303
DATASET_IDENTITY = {
    "records_sha256": PILOT_RECORDS_SHA256,
    "record_count": 4096,
    "unique_group_count": 4096,
}


class Phase13CSmokeError(RuntimeError):
    pass


class LoraCheckpointState:
    """Expose only adapter tensors to the generic checkpoint contract."""
    def __init__(self, model) -> None:
        self.model = model

    def state_dict(self):
        from peft import get_peft_model_state_dict
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in get_peft_model_state_dict(self.model).items()
        }

    def load_state_dict(self, state, strict=True):
        from peft import set_peft_model_state_dict
        result = set_peft_model_state_dict(self.model, state)
        if strict and getattr(result, "unexpected_keys", ()):
            raise Phase13CSmokeError(f"unexpected adapter keys: {result.unexpected_keys}")
        return result


def load_pilot_order(path: Path = PILOT):
    if file_sha(path) != PILOT_RECORDS_SHA256:
        raise Phase13CSmokeError("Pilot4096 SHA mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [str(row["recommendation_group_id"]) for row in rows]
    order, metadata = order_metadata(ids, DEFAULT_ORDER_SEED)
    if not len(rows) == len(set(ids)) == 4096 or metadata["sha256"] != ORDER_SHA256:
        raise Phase13CSmokeError("frozen order identity mismatch")
    return {str(row["recommendation_group_id"]): row for row in rows}, order, metadata


def rng_probe(device: torch.device) -> dict[str, float]:
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch_cpu": float(torch.rand(())),
        "torch_cuda": float(torch.rand((), device=device).cpu()),
    }


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def base_parameter_versions(model) -> dict[str, int]:
    return {name: int(parameter._version) for name, parameter in model.named_parameters() if "lora_" not in name}


def optimizer_step_value(optimizer) -> int:
    states = optimizer.state_dict()["state"].values()
    values = {int(state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"]) for state in states}
    if len(values) != 1:
        raise Phase13CSmokeError(f"optimizer step state is not uniform: {values}")
    return values.pop()


def load_runtime(device: torch.device):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    provenance = validate_model_provenance(load_contract())
    config = PeftConfig.from_pretrained(BETA_CHECKPOINT, local_files_only=True)
    if float(config.lora_dropout) != EXPECTED_LORA_DROPOUT:
        raise Phase13CSmokeError("LoRA dropout config changed")
    renderer = BetaGammaRenderer()
    base = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_BASE, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, BETA_CHECKPOINT, is_trainable=True, local_files_only=True).to(device)
    model.eval()
    dropout, trainability = audit_dropout_modules(model), parameter_trainability(model)
    if model.training or dropout["rl_dropout_active"] or trainability["base_trainable_param_count"] != 0:
        raise Phase13CSmokeError("eval/dropout/trainability gate failed")
    optimizer, optimizer_audit = build_lora_optimizer(model)
    if not optimizer_audit["optimizer_lora_only"]:
        raise Phase13CSmokeError("optimizer includes non-LoRA parameters")
    return model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit


def build_driver(model, optimizer, renderer, device, trace):
    trainer = TrueRecGRPOTrainerV1(
        model, renderer.tokenizer.convert_tokens_to_ids, FORMAL_PAD_TOKEN_ID, device=device,
    )
    gradient_records = []

    def rollout(record):
        model.eval()
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
            raise Phase13CSmokeError("policy changed during rollout")
        artifacts = extract_generation_artifacts(context_ids, output, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
        trace.append({
            "group_id": record["recommendation_group_id"],
            "completion_ids": [list(values) for values in artifacts.completion_ids],
        })
        return {"context_ids": context_ids, "artifacts": artifacts, "versions": versions}

    def rescore(record, rollout_value):
        artifacts = rollout_value["artifacts"]
        return rescore_business_group_from_completions(
            model, record, rollout_value["context_ids"], artifacts.completion_ids,
            artifacts.generation_score_logps, renderer.tokenizer.convert_ids_to_tokens,
            FORMAL_PAD_TOKEN_ID, device, expected_parameter_versions=rollout_value["versions"],
        )

    def gradient_gate():
        audit = gradient_audit(model)
        gradient_records.append(audit)
        return (
            audit["lora_params_with_nonzero_grad"] > 0
            and math.isfinite(audit["lora_grad_norm"]) and audit["lora_grad_norm"] > 0
            and audit["base_params_with_grad"] == 0
            and audit["nan_grad_count"] == 0 and audit["inf_grad_count"] == 0
        )

    driver = TrueRecTrainingDriverV1(
        rollout_fn=rollout, old_rescore_fn=rescore, trainer=trainer, optimizer=optimizer,
        gradient_finite_fn=gradient_gate, policy_fingerprint_fn=lambda: parameter_versions(model),
        groups_per_optimizer_step=1,
    )
    return driver, trainer, gradient_records


def run_groups(model, driver, trainer, gradients, records_by_id, order, start, stop, device):
    reports = []
    base_before = base_parameter_versions(model)
    lora_before, _ = lora_parameter_sha(model)
    for index in range(start, stop):
        torch.cuda.reset_peak_memory_stats(device)
        forward_before = trainer.physical_policy_forward_calls
        backward_before = trainer.streaming_backward_calls
        result = driver.run_group(records_by_id[order[index]])
        torch.cuda.synchronize(device)
        lora_after, _ = lora_parameter_sha(model)
        backward_result = result.backward_result
        finite_losses = all(math.isfinite(value) for value in (
            backward_result.frontier_value, backward_result.hpr_value_raw,
            backward_result.hpr_value_weighted, backward_result.total_value,
        ))
        report = {
            "group_index": index,
            "group_id": order[index],
            "G": 8,
            "physical_forwards": trainer.physical_policy_forward_calls - forward_before,
            "physical_backwards": trainer.streaming_backward_calls - backward_before,
            "hpr_extra_forwards": trainer.hpr_extra_forward_calls,
            "losses": {
                "frontier": backward_result.frontier_value, "hpr_raw": backward_result.hpr_value_raw,
                "hpr_weighted": backward_result.hpr_value_weighted, "total": backward_result.total_value,
                "finite": finite_losses,
            },
            "gradient": gradients[-1],
            "lora_sha_before": lora_before,
            "lora_sha_after": lora_after,
            "lora_changed": lora_before != lora_after,
            "allocated_after_group_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
            "reserved_after_group_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        }
        if not (
            report["physical_forwards"] == report["physical_backwards"] == 4
            and report["hpr_extra_forwards"] == 0 and finite_losses and report["lora_changed"]
        ):
            raise Phase13CSmokeError(f"real group gate failed: {report}")
        reports.append(report)
        lora_before = lora_after
    if base_parameter_versions(model) != base_before:
        raise Phase13CSmokeError("base parameter mutation detected")
    return reports


def common_preflight(physical_gpu_id: int):
    device = torch.device("cuda", 0); torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / (1024 ** 3)
    if free_gib < MIN_FREE_GIB:
        raise Phase13CSmokeError(f"FREE_MEMORY_AT_START_GB={free_gib}")
    return device, {
        "physical_gpu_id": physical_gpu_id, "gpu_name": torch.cuda.get_device_name(device),
        "free_memory_at_start_gb": free_gib, "total_memory_gb": total_bytes / (1024 ** 3),
    }


def process_a(output_dir: Path, checkpoint_dir: Path, physical_gpu_id: int) -> None:
    device, resource = common_preflight(physical_gpu_id)
    records, order, order_info = load_pilot_order()
    model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
    seed_all(RUN_SEED)
    trace = []
    driver, trainer, gradients = build_driver(model, optimizer, renderer, device, trace)
    groups = run_groups(model, driver, trainer, gradients, records, order, 0, 2, device)
    if driver.state.global_step != 2 or driver.state.rollouts_completed != 2:
        raise Phase13CSmokeError("Process A counters failed")
    lora_sha_at_save, _ = lora_parameter_sha(model)
    save_checkpoint_atomic(
        checkpoint_dir, model=LoraCheckpointState(model), optimizer=optimizer, driver=driver,
        dataset_identity=DATASET_IDENTITY, epoch=0, next_group_index=2, order=order_info,
        cuda_device=device,
    )
    expected_rng_after_checkpoint = rng_probe(device)
    result = {
        "status": "PASS", "process": "A", "model_family": INIT_FAMILY,
        "provenance": provenance, "dropout": dropout, "trainability": trainability,
        "optimizer": optimizer_audit, "order": order_info, "group_trace": trace,
        "groups": groups, "driver_state_at_save": driver.export_state(),
        "next_group_index_at_save": 2, "lora_sha_at_save": lora_sha_at_save,
        "optimizer_step_state_at_save": optimizer_step_value(optimizer),
        "expected_rng_after_checkpoint": expected_rng_after_checkpoint,
        "resource": resource, "checkpoint_dir": str(checkpoint_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "process_a.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"PHASE1_3C_PROCESS_A": "PASS", "groups": [item["group_id"] for item in groups]}), flush=True)


def process_b(output_dir: Path, checkpoint_dir: Path, physical_gpu_id: int) -> None:
    device, resource = common_preflight(physical_gpu_id)
    records, order, order_info = load_pilot_order()
    process_a_result = json.loads((output_dir / "process_a.json").read_text())
    model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
    trace = []
    driver, trainer, gradients = build_driver(model, optimizer, renderer, device, trace)
    cursor = load_checkpoint(
        checkpoint_dir, model=LoraCheckpointState(model), optimizer=optimizer, driver=driver,
        current_dataset_identity=DATASET_IDENTITY, current_group_ids=list(records), cuda_device=device,
    )
    lora_sha_after_load, _ = lora_parameter_sha(model)
    rng_after_load = rng_probe(device)
    rng_equal = {name: rng_after_load[name] == process_a_result["expected_rng_after_checkpoint"][name] for name in rng_after_load}
    # Restore again so RNG validation itself cannot perturb group index 2.
    cursor = load_checkpoint(
        checkpoint_dir, model=LoraCheckpointState(model), optimizer=optimizer, driver=driver,
        current_dataset_identity=DATASET_IDENTITY, current_group_ids=list(records), cuda_device=device,
    )
    optimizer_at_load = optimizer_step_value(optimizer)
    if not (
        cursor == {"epoch": 0, "next_group_index": 2}
        and driver.state.global_step == 2 and optimizer_at_load == 2
        and lora_sha_after_load == process_a_result["lora_sha_at_save"]
        and all(rng_equal.values())
    ):
        raise Phase13CSmokeError("checkpoint restore gate failed")
    groups = run_groups(model, driver, trainer, gradients, records, order, 2, 3, device)
    sequence = [item["group_id"] for item in process_a_result["groups"]] + [groups[0]["group_id"]]
    if sequence != order[:3] or len(set(sequence)) != 3 or driver.state.global_step != 3:
        raise Phase13CSmokeError("resumed sequence/counter gate failed")
    allocated_pair = [process_a_result["groups"][-1]["allocated_after_group_gb"], groups[-1]["allocated_after_group_gb"]]
    if abs(allocated_pair[1] - allocated_pair[0]) > 2.0:
        raise Phase13CSmokeError(f"activation memory accumulation suspected: {allocated_pair}")
    result = {
        "status": "PASS", "process": "B", "provenance": provenance,
        "dropout": dropout, "trainability": trainability, "optimizer": optimizer_audit,
        "order": order_info, "group_trace": trace, "groups": groups,
        "group_sequence": sequence, "global_step_at_load": 2, "final_global_step": 3,
        "next_group_index_at_load": 2, "final_next_group_index": 3,
        "lora_sha_at_save": process_a_result["lora_sha_at_save"],
        "lora_sha_after_load": lora_sha_after_load,
        "model_restore_exact": lora_sha_after_load == process_a_result["lora_sha_at_save"],
        "optimizer_step_state_at_load": optimizer_at_load,
        "optimizer_restore_pass": optimizer_at_load == 2,
        "rng_restored": rng_equal, "driver_state_final": driver.export_state(),
        "no_duplicate_group": len(set(sequence)) == 3, "no_skipped_group": sequence == order[:3],
        "base_parameter_mutation": False,
        "lora_changed_each_step": all(item["lora_changed"] for item in process_a_result["groups"] + groups),
        "memory_accumulation_gate": "PASS", "resource": resource,
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "trainer_microbatch_size": TRAINER_MICROBATCH_SIZE,
        "ddp_started": False, "formal_training_started": False,
    }
    (output_dir / "real_three_group_resume_smoke.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "CHATGPT_PHASE1_3C_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.3C real single-GPU 3-group resume smoke: PASS\n"
        "Process A ran exactly order indices 0-1 and atomically saved cursor 2 with CUDA RNG. Fresh Process B restored LoRA/AdamW/driver/all RNG states and ran only index 2. No fourth group, DDP, scheduler, evaluation, or formal Pilot training ran.\n"
    )
    print(json.dumps({"PHASE1_3C_PROCESS_B": "PASS", "group_sequence": sequence}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", choices=("A", "B"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    if args.process == "A": process_a(args.output_dir, args.checkpoint_dir, args.physical_gpu_id)
    else: process_b(args.output_dir, args.checkpoint_dir, args.physical_gpu_id)


if __name__ == "__main__":
    main()
