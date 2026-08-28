"""Four-GPU, one-real-G8, zero-update preflight for Think suffix SID GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist

from .preflight_contract import evaluate_preflight_conditions
from .run_think_suffix_sid_train import (
    PARENT_ADAPTER,
    PARENT_ADAPTER_SHA256,
    baseline_runner,
    make_suffix_grpo_config,
    prepare_think_suffix_run_plan,
    validate_parent_adapter,
)
from .runtime_import_provenance import assert_runtime_import_provenance
from .suffix_objective import GROUP_SIZE, population_std, resample_decision
from .think_suffix_sid_trainer import (
    ThinkSuffixSIDTrainer,
    make_think_suffix_reward_func,
)
from ..gr_rec_think_composite_interest_v1.single_node_nccl import (
    configure_single_node_nccl,
)


SEED = 20260816


class PreflightTrainer(ThinkSuffixSIDTrainer):
    def _generate_and_score_completions(self, inputs):
        self.preflight_input_group_ids = [row["recommendation_group_id"] for row in inputs]
        output = super()._generate_and_score_completions(inputs)
        self.preflight_pre_shuffle = {
            key: output[key].detach().clone()
            for key in (
                "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
                "suffix_loss_mask", "advantages",
            )
        }
        return output


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[3],
    ).strip()


def checksum_lora(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if "lora" not in name.lower():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def gather_object(value):
    output = [None] * dist.get_world_size()
    dist.all_gather_object(output, value)
    return output


def core_resample_runtime(runtime: dict) -> dict:
    keys = (
        "attempt", "resample_round", "zero_std", "retry", "exhausted",
        "accepted", "generated_candidate_total", "resample_rounds_used",
        "rejected_reward_vectors", "accepted_reward_vector",
        "zero_std_rescued", "zero_std_rescue_exhausted",
    )
    return {key: runtime.get(key) for key in keys}


def expected_advantages(rewards):
    values = [float(value) for value in rewards]
    mean = sum(values) / len(values)
    std = population_std(values)
    return [0.0 if std == 0.0 else (value - mean) / (std + 1e-4) for value in values]


def vectors_close(left, right, tolerance=2e-5):
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)
    )


def real_resample_contract(runtime: dict) -> bool:
    round_index = int(runtime["resample_round"])
    rejected = runtime["rejected_reward_vectors"]
    accepted = runtime["accepted_reward_vector"]
    if len(rejected) != round_index:
        return False
    if any(population_std(vector) != 0.0 for vector in rejected):
        return False
    if int(runtime["generated_candidate_total"]) != GROUP_SIZE * (round_index + 1):
        return False
    if not 0 <= round_index <= 3:
        return False
    accepted_zero = population_std(accepted) == 0.0
    return accepted_zero == bool(runtime["zero_std_rescue_exhausted"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="GR-REC-THINK-SUFFIX-SID-G8-REROLL-V1-PREFLIGHT-20260828"
    )
    args = parser.parse_args(argv)

    parent = validate_parent_adapter()
    import_provenance = assert_runtime_import_provenance()
    nccl = configure_single_node_nccl(initialize=True)
    rank = int(nccl["local_rank"])
    world = int(nccl["world_size"])
    torch.manual_seed(SEED + rank)
    os.environ["GRPO_MONITOR"] = "0"
    os.environ["GRPO_RUN_ID"] = args.run_id

    plan_args = baseline_runner.build_arg_parser().parse_args([
        "--run-id", args.run_id,
        "--n-groups", "all",
        "--probe-groups", "0",
        "--save-steps", "250",
        "--save-total-limit", "1",
    ])
    plan = prepare_think_suffix_run_plan(plan_args)
    dataset = plan["dataset"].select([0])

    from grpo_model import ADAPTER, BASE, load_model

    model, tokenizer, _ = load_model(f"cuda:{rank}")
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())

    cfg = make_suffix_grpo_config(
        "/tmp/grpo-think-suffix-preflight", 2, 1e-6, SEED,
        save_strategy="no",
    )
    trainer = PreflightTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[make_think_suffix_reward_func(tokenizer)],
        monitor_writer=None,
    )
    trainer.current_gradient_accumulation_steps = 1

    base_versions_before = {
        name: parameter._version
        for name, parameter in model.named_parameters()
        if "lora" not in name.lower()
    }
    lora_checksum_before = checksum_lora(model)
    base_requires_grad_count = sum(
        int(parameter.requires_grad)
        for name, parameter in model.named_parameters()
        if "lora" not in name.lower()
    )

    # Distributed control proves that every rank makes the same retry decision
    # from one gathered G8 vector without changing production rewards.
    zero_local = [0.0, 0.0]
    zero_global = [value for row in gather_object(zero_local) for value in row]
    zero_decision = resample_decision(zero_global, 0)
    zero_decisions = gather_object(zero_decision)

    batch = next(iter(trainer.get_train_dataloader()))
    model.zero_grad(set_to_none=True)
    prepared = trainer._prepare_inputs(batch)
    runtime = core_resample_runtime(trainer._resample_runtime)

    captured = {}
    original_logps = trainer._get_per_token_logps_and_entropies

    def capture_logps(*call_args, **call_kwargs):
        result = original_logps(*call_args, **call_kwargs)
        logps = result[0]
        if torch.is_grad_enabled():
            logps.retain_grad()
            captured["logps"] = logps
            captured["attention_mask"] = call_args[2].detach().clone()
        return result

    trainer._get_per_token_logps_and_entropies = capture_logps
    loss = trainer._compute_loss(model, prepared)
    trainer.accelerator.backward(loss)

    logps = captured["logps"]
    action_grad = logps.grad.detach().float()
    completion_mask = prepared["completion_mask"].bool()
    suffix_mask = prepared["suffix_loss_mask"].bool()
    cot_mask = completion_mask & ~suffix_mask
    padding_mask = ~completion_mask
    attention_completion = captured["attention_mask"][:, -completion_mask.size(1):].bool()

    def masked_max(values, mask):
        selected = values[mask]
        return float(selected.abs().max()) if selected.numel() else 0.0

    suffix_grad_abs_sum = float(action_grad[suffix_mask].abs().sum())
    lora_grad_count = 0
    base_grad_count = 0
    lora_grad_sq = 0.0
    gradients_finite = bool(torch.isfinite(action_grad).all())
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if "lora" in name.lower():
            lora_grad_count += 1
            grad = parameter.grad.detach().float()
            gradients_finite = gradients_finite and bool(torch.isfinite(grad).all())
            lora_grad_sq += float((grad * grad).sum())
        else:
            base_grad_count += 1

    lora_checksum_after = checksum_lora(model)
    base_version_changed_count = sum(
        int(parameter._version != base_versions_before[name])
        for name, parameter in model.named_parameters()
        if name in base_versions_before
    )
    pre_advantages = (
        trainer.preflight_pre_shuffle["advantages"].detach().float().cpu().tolist()
    )
    rank_row = {
        "rank": rank,
        "local_group_ids": trainer.preflight_input_group_ids,
        "local_candidate_count": len(trainer.preflight_input_group_ids),
        "resample_runtime": runtime,
        "pre_shuffle_advantages": pre_advantages,
        "loss": float(loss.detach()),
        "cot_action_token_count": int(cot_mask.sum()),
        "suffix_action_token_count": int(suffix_mask.sum()),
        "padding_token_count": int(padding_mask.sum()),
        "cot_action_grad_max_abs": masked_max(action_grad, cot_mask),
        "suffix_action_grad_abs_sum": suffix_grad_abs_sum,
        "padding_action_grad_max_abs": masked_max(action_grad, padding_mask),
        "full_completion_attention": bool(torch.equal(attention_completion, completion_mask)),
        "suffix_mask_subset": not bool((suffix_mask & ~completion_mask).any()),
        "lora_grad_tensor_count": lora_grad_count,
        "base_grad_tensor_count": base_grad_count,
        "lora_grad_norm": math.sqrt(lora_grad_sq),
        "gradients_finite": gradients_finite,
        "lora_checksum_before": lora_checksum_before,
        "lora_checksum_after": lora_checksum_after,
        "base_requires_grad_count": base_requires_grad_count,
        "base_version_changed_count": base_version_changed_count,
        "optimizer_object_present": trainer.optimizer is not None,
        "scheduler_object_present": trainer.lr_scheduler is not None,
    }
    rank_rows = gather_object(rank_row)

    if rank == 0:
        group_ids = [group for row in rank_rows for group in row["local_group_ids"]]
        real_runtimes = [row["resample_runtime"] for row in rank_rows]
        actual_rewards = runtime["accepted_reward_vector"]
        actual_advantages = [
            value for row in rank_rows for value in row["pre_shuffle_advantages"]
        ]
        expected = expected_advantages(actual_rewards)
        observations = {
            "world_size_4": world == 4,
            "runtime_import_provenance": import_provenance["runtime_import_provenance"] == "PASS",
            "ddp_g8_alignment": len(group_ids) == 8 and len(set(group_ids)) == 1,
            "synthetic_zero_std_retry_sync": (
                len(zero_global) == 8
                and all(decision == zero_decisions[0] for decision in zero_decisions)
                and zero_decision["retry"]
                and not zero_decision["accepted"]
            ),
            "real_resample_sync": all(item == real_runtimes[0] for item in real_runtimes),
            "real_resample_contract": real_resample_contract(runtime),
            "online_advantage_parity": vectors_close(actual_advantages, expected),
            "full_completion_attention": all(row["full_completion_attention"] for row in rank_rows),
            "suffix_mask_subset": all(row["suffix_mask_subset"] for row in rank_rows),
            "cot_action_gradient_zero": all(row["cot_action_grad_max_abs"] == 0.0 for row in rank_rows),
            "padding_action_gradient_zero": all(row["padding_action_grad_max_abs"] == 0.0 for row in rank_rows),
            "suffix_action_gradient_nonzero": sum(row["suffix_action_grad_abs_sum"] for row in rank_rows) > 0.0,
            "finite_loss_and_gradient": all(
                math.isfinite(row["loss"])
                and math.isfinite(row["lora_grad_norm"])
                and row["gradients_finite"]
                for row in rank_rows
            ),
            "lora_has_gradient": all(row["lora_grad_tensor_count"] > 0 for row in rank_rows),
            "base_has_no_gradient": all(row["base_grad_tensor_count"] == 0 for row in rank_rows),
            "base_requires_grad_zero": all(row["base_requires_grad_count"] == 0 for row in rank_rows),
            "lora_checksum_unchanged": all(
                row["lora_checksum_before"] == row["lora_checksum_after"] for row in rank_rows
            ),
            "base_version_unchanged": all(row["base_version_changed_count"] == 0 for row in rank_rows),
            "optimizer_step_absent": True,
            "scheduler_step_absent": True,
        }
        evaluation = evaluate_preflight_conditions(observations)
        payload = {
            "experiment": "GR_REC_ThinkSuffixSID_Resample_v1",
            "gpu_validation_launch_commit": git_head(),
            "base": BASE,
            "adapter": ADAPTER,
            "parent_adapter": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_validation": parent,
            "world_size": world,
            **import_provenance,
            "nccl_bootstrap": nccl,
            "group_size": GROUP_SIZE,
            "group_id": group_ids[0] if group_ids else None,
            "global_group_ids": group_ids,
            "synthetic_zero_std_global_rewards": zero_global,
            "synthetic_zero_std_decision": zero_decision,
            "real_resample_runtime": runtime,
            "accepted_rewards": actual_rewards,
            "expected_advantages": expected,
            "actual_advantages": actual_advantages,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "parameter_update": False,
            "checkpoint_saved": False,
            "training_started": False,
            "rank_rows": rank_rows,
            **evaluation,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "preflight_pass": payload["preflight_pass"],
            "failure_reasons": payload["failure_reasons"],
            "real_resample_round": runtime["resample_round"],
            "accepted_rewards": actual_rewards,
        }), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
