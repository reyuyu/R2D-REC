"""One-batch, four-GPU, zero-update validation of the production trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from datasets import Dataset

from grpo_model import ADAPTER, BASE, load_model, render_prompt
from grpo_beam_domain import domain_prefix
from grpo_trl_trainer import make_think_reward_func
from monitor.writer import monitor_from_env
from run_grpo_trl_smoke import make_beam32_fn, make_grpo_config

from .composite_trainer import ThinkCompositeInterestRecGRPOTrainer
from .composite_trainer import slice_global
from .interest_metric import composite_reward
from .preflight_contract import (
    evaluate_preflight_conditions,
    float_vectors_close,
    post_shuffle_association_parity,
    prepare_preflight_loss_context,
    raw_decode_diagnostics,
    sid_runtime_observation,
)
from .run_gr_rec_think_composite_interest_v1 import git_head, prepare_plan
from .runtime_import_provenance import assert_runtime_import_provenance
from .single_node_nccl import configure_single_node_nccl
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_sids


class PreflightTrainer(ThinkCompositeInterestRecGRPOTrainer):
    def _calculate_rewards(self, *args, **kwargs):
        result = super()._calculate_rewards(*args, **kwargs)
        self.preflight_runtime = self._composite_runtime
        return result

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        self.preflight_pre_shuffle = {
            key: output[key].detach().clone()
            for key in (
                "prompt_ids", "prompt_mask", "completion_ids",
                "completion_mask", "advantages",
            )
        }
        return output


def checksum_trainable(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def gather(value):
    output = [None] * dist.get_world_size()
    dist.all_gather_object(output, value)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="GR-REC-THINK-COMPOSITE-INTEREST-V1-PREFLIGHT-20260822")
    args = parser.parse_args(argv)
    import_provenance = assert_runtime_import_provenance()
    nccl_bootstrap = configure_single_node_nccl(initialize=True)
    rank = nccl_bootstrap["local_rank"]
    world = nccl_bootstrap["world_size"]
    torch.manual_seed(20260818 + rank)
    os.environ["GRPO_RUN_ID"] = args.run_id
    os.environ["GRPO_MONITOR"] = "1"
    plan_args = SimpleNamespace(
        run_id=args.run_id, max_steps=1, output_dir="/tmp/grpo-composite-preflight",
        grpo_data=Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"),
        gold_data=Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl"),
    )
    plan = prepare_plan(plan_args)
    dataset = Dataset.from_list(plan["records"][:4])
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())
    base_versions_before = {name: parameter._version for name, parameter in model.named_parameters()
                            if "lora" not in name.lower()}
    checksum_before = checksum_trainable(model)
    cfg = make_grpo_config("/tmp/grpo-composite-preflight", 1, 1e-6, 20260818,
                           save_strategy="no")
    monitor = monitor_from_env(args.run_id, rank)
    beam32 = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
    trainer = PreflightTrainer(
        model=model, args=cfg, processing_class=tokenizer, train_dataset=dataset,
        reward_funcs=[make_think_reward_func(beam32_fn=beam32)], monitor_writer=monitor,
    )
    batch = next(iter(trainer.get_train_dataloader()))
    model.zero_grad(set_to_none=True)
    prepared = trainer._prepare_inputs(batch)
    beam_call = trainer._current_beam_call() or {}
    runtime = trainer.preflight_runtime
    pre_shuffle = trainer.preflight_pre_shuffle
    pre_advantages = pre_shuffle["advantages"].detach().float().cpu().tolist()
    post_advantages = prepared["advantages"].detach().float().cpu().tolist()
    expected_advantages = slice_global(
        runtime["advantages"], rank, len(pre_advantages), world,
    )
    raw_decode = raw_decode_diagnostics(
        tokenizer,
        pre_shuffle["completion_ids"],
        pre_shuffle["completion_mask"],
        runtime["candidates"],
        rank,
    )
    parity_row = {
        "rank": rank,
        "raw_decode_candidate_count": raw_decode["candidate_count"],
        "raw_decode_mismatch_count": raw_decode["mismatch_count"],
        "raw_decode_runtime_pass": raw_decode["pass"],
        "raw_decode_mismatches": raw_decode["mismatches"],
        "pre_shuffle_advantage_vector": pre_advantages,
        "post_shuffle_advantage_vector": post_advantages,
        "pre_shuffle_advantage_parity": float_vectors_close(
            pre_advantages, expected_advantages,
        ),
        "post_shuffle_association_parity": post_shuffle_association_parity(
            pre_shuffle, prepared,
        ),
        "advantage_order_changed": not float_vectors_close(
            pre_advantages, post_advantages,
        ),
    }
    parity_rows = gather(parity_row)
    gradient_accumulation_steps = prepare_preflight_loss_context(trainer)
    assert int(trainer.args.gradient_accumulation_steps) == 1
    with trainer.compute_loss_context_manager():
        loss = trainer.compute_loss(model, prepared)
    trainer.accelerator.backward(loss)
    lora_grad_count = base_grad_count = 0
    grad_sq = 0.0
    grad_finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if "lora" in name.lower():
            lora_grad_count += 1
            grad = parameter.grad.detach().float()
            grad_finite = grad_finite and bool(torch.isfinite(grad).all())
            grad_sq += float((grad * grad).sum())
        else:
            base_grad_count += 1
    grad_norm = math.sqrt(grad_sq)
    checksum_after = checksum_trainable(model)
    base_changed = any(parameter._version != base_versions_before[name]
                       for name, parameter in model.named_parameters()
                       if name in base_versions_before)
    rank_row = {
        "rank": rank, "loss": float(loss.detach()), "grad_norm": grad_norm,
        "lora_grad_tensor_count": lora_grad_count,
        "base_grad_tensor_count": base_grad_count,
        "grad_finite": grad_finite,
    }
    rank_rows = gather(rank_row)
    if rank == 0:
        groups = runtime["groups"]
        candidates = runtime["candidates"]
        composite_formula_parity = all(abs(row["composite_reward"] - composite_reward(
            row["beam_raw"], row["cot_utility"], row["interest_tiebreak_scale"]
        )) < 1e-7 for row in candidates)
        fixed_domain_tasks = beam_call.get("fixed_domain_tasks", [])
        fixed_domain_mismatches = [
            task for task in fixed_domain_tasks
            if (
                task.get("target_domain") not in {"video", "prod", "ad", "living"}
                or task.get("domain_prefix") != domain_prefix(task.get("target_domain"))
                or not task.get("context_ends_with_prefix")
            )
        ]
        beam_fixed_domain_candidate_count = len(fixed_domain_tasks)
        beam_fixed_domain_mismatch_count = len(fixed_domain_mismatches)
        beam_fixed_domain_prefix_pass = (
            beam_call.get("beam_fixed_domain_prefix") is True
            and beam_fixed_domain_candidate_count == 16
            and beam_fixed_domain_mismatch_count == 0
        )
        fixed_domain_results = beam_call.get("fixed_domain_results", [])
        beam_sequence_count = sum(
            len(item.get("generated_token_counts", [])) for item in fixed_domain_results
        )
        generated_token_count_mismatch = sum(
            int(item.get("generated_token_count_mismatch", 0))
            for item in fixed_domain_results
        )
        beam_abc3_pass = beam_sequence_count == 512 and generated_token_count_mismatch == 0
        beam_result_by_id = {
            tuple(item["task_id"]): float(item["reward"])
            for item in fixed_domain_results
        }
        fixed_domain_reward_parity = all(
            abs(float(row["beam_raw"]) - beam_result_by_id.get(
                (int(row["rank"]), int(row["local_index"])), math.inf
            )) < 1e-7
            for row in candidates
        )
        reward_parity = composite_formula_parity and fixed_domain_reward_parity
        pre_shuffle_advantage_parity = all(
            row["pre_shuffle_advantage_parity"] for row in parity_rows
        )
        post_shuffle_association_parity_pass = all(
            row["post_shuffle_association_parity"] for row in parity_rows
        )
        advantage_parity = (
            pre_shuffle_advantage_parity and post_shuffle_association_parity_pass
        )
        advantage_order_changed = any(
            row["advantage_order_changed"] for row in parity_rows
        )
        group_ids = [row["group_id"] for row in candidates]
        ddp_alignment = len(groups) == 4 and all(
            len(set(group_ids[start:start + 4])) == 1 for start in range(0, 16, 4)
        )
        generated_sid_count = sum(bool(extract_sids(row["completion"])) for row in candidates)
        raw_decode_candidate_count = sum(
            row["raw_decode_candidate_count"] for row in parity_rows
        )
        raw_decode_mismatch_count = sum(
            row["raw_decode_mismatch_count"] for row in parity_rows
        )
        raw_decode_pass = (
            raw_decode_candidate_count == len(candidates)
            and raw_decode_mismatch_count == 0
            and all(row["raw_decode_runtime_pass"] for row in parity_rows)
        )
        gold_leakage = any(row["gold_cot"] in render_prompt(tokenizer, row["prompt"])
                           for row in plan["records"][:4])
        loss_finite = all(math.isfinite(row["loss"]) for row in rank_rows)
        grad_finite = all(
            math.isfinite(row["grad_norm"]) and row["grad_finite"] for row in rank_rows
        )
        evaluation = evaluate_preflight_conditions(
            raw_decode_runtime=raw_decode_pass,
            beam_fixed_domain_prefix=beam_fixed_domain_prefix_pass,
            beam_abc3=beam_abc3_pass,
            online_reward_parity=reward_parity,
            online_advantage_parity=advantage_parity,
            ddp_g4_alignment=ddp_alignment,
            gold_leakage=gold_leakage,
            loss_finite=loss_finite,
            grad_finite=grad_finite,
            lora_has_gradient=all(row["lora_grad_tensor_count"] > 0 for row in rank_rows),
            base_has_gradient=any(row["base_grad_tensor_count"] > 0 for row in rank_rows),
            checksum_unchanged=checksum_before == checksum_after,
            base_changed=base_changed,
            nccl_error=False,
            oom=False,
            nan=False,
            inf=False,
        )
        conditions = evaluation["conditions"]
        sid_observation = sid_runtime_observation(generated_sid_count)
        payload = {
            "experiment": "GR_REC_Think_CompositeInterest_v1",
            "gpu_validation_launch_commit": git_head(),
            "base": BASE, "adapter": ADAPTER, "world_size": world,
            **import_provenance,
            "nccl_bootstrap": nccl_bootstrap,
            "optimizer_steps": 0, "zero_update": True,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "current_gradient_accumulation_steps": (
                trainer.current_gradient_accumulation_steps
            ),
            "preflight_loss_entry": "compute_loss",
            "preflight_compute_loss_context": True,
            "raw_decode_validation_mode": "MASKED_UNPADDED",
            "raw_decode_candidate_count": raw_decode_candidate_count,
            "raw_decode_mismatch_count": raw_decode_mismatch_count,
            "raw_decode_runtime_pass": raw_decode_pass,
            "beam_fixed_domain_candidate_count": beam_fixed_domain_candidate_count,
            "beam_fixed_domain_mismatch_count": beam_fixed_domain_mismatch_count,
            "beam_fixed_domain_mismatches": fixed_domain_mismatches,
            "BEAM_FIXED_DOMAIN_PREFIX_PASS": (
                "YES" if beam_fixed_domain_prefix_pass else "NO"
            ),
            "beam_sequence_count": beam_sequence_count,
            "generated_token_count_mismatch": generated_token_count_mismatch,
            "BEAM_ABC3_PASS": "YES" if beam_abc3_pass else "NO",
            "beam_min_new_tokens": 3,
            "beam_max_new_tokens": 3,
            "beam_search_space": "ABC_CONTINUATION_AFTER_FIXED_DOMAIN",
            "target_domain_source": "dataset.target_domain",
            "gold_used_to_select_domain": False,
            "raw_decode_rank_rows": [{
                "rank": row["rank"],
                "candidate_count": row["raw_decode_candidate_count"],
                "mismatch_count": row["raw_decode_mismatch_count"],
                "pass": row["raw_decode_runtime_pass"],
                "mismatches": row["raw_decode_mismatches"],
            } for row in parity_rows],
            **sid_observation,
            "online_reward_parity": reward_parity,
            "composite_formula_parity": composite_formula_parity,
            "fixed_domain_reward_parity": fixed_domain_reward_parity,
            "pre_shuffle_advantage_parity": pre_shuffle_advantage_parity,
            "post_shuffle_association_parity": post_shuffle_association_parity_pass,
            "pre_shuffle_advantage_vector_by_rank": [
                row["pre_shuffle_advantage_vector"] for row in parity_rows
            ],
            "post_shuffle_advantage_vector_by_rank": [
                row["post_shuffle_advantage_vector"] for row in parity_rows
            ],
            "advantage_order_changed": advantage_order_changed,
            "online_advantage_parity": advantage_parity,
            "ddp_g4_alignment": ddp_alignment,
            "gold_leakage": gold_leakage,
            "loss": sum(row["loss"] for row in rank_rows) / world,
            "grad_norm": sum(row["grad_norm"] for row in rank_rows) / world,
            "lora_grad_tensor_count": sum(row["lora_grad_tensor_count"] for row in rank_rows),
            "base_grad_tensor_count": sum(row["base_grad_tensor_count"] for row in rank_rows),
            "trainable_checksum_before": checksum_before,
            "trainable_checksum_after": checksum_after,
            "base_changed": base_changed,
            "rank_rows": rank_rows,
            "groups": groups,
            "conditions": conditions,
            "preflight_pass": evaluation["preflight_pass"],
            "failure_reasons": evaluation["failure_reasons"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"preflight_pass": payload["preflight_pass"], "conditions": conditions}), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
