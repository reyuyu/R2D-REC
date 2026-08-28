"""Four-GPU zero-update preflight for Think Dual Beam8 v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

# Import the runner first: it installs the current worktree scripts directory
# before any top-level grpo_* module is resolved.
from .run_dual_beam8_train import (
    FIXED_PROBE4_IDS, PARENT_ADAPTER, PARENT_ADAPTER_SHA256, DualManifestWriter,
    assert_runtime_import_provenance, baseline_runner, make_dual_grpo_config,
    prepare_dual_run_plan, validate_parent_adapter,
)
from grpo_probe import FixedProbeEvaluator
from monitor.writer import MonitorWriter
from .dual_beam8_trainer import (
    COT_G, SID_G, DualBeam8Runtime, ThinkDualBeam8Trainer,
    independent_sid_advantages, make_dual_beam8_reward_func, population_advantages,
)
from ablations.gr_rec_think_composite_interest_v1.single_node_nccl import (
    configure_single_node_nccl,
)

SEED = 20260816
PREFLIGHT_GROUP_ID = "0d3b5e5ef4df6c333ca9f556c22b3e9e9f43c4995baed7a833977eedfb537a84"


def gather(value):
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[3]).strip()


def lora_checksum(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if "lora" in name.lower():
            digest.update(name.encode())
            digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def grad_summary(model):
    lora_sq = 0.0
    lora_count = base_count = 0
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if "lora" in name.lower():
            grad = parameter.grad.detach().float()
            lora_count += 1
            lora_sq += float((grad * grad).sum())
            finite = finite and bool(torch.isfinite(grad).all())
        else:
            base_count += 1
    return {"lora_grad_norm": math.sqrt(lora_sq), "lora_grad_tensor_count": lora_count,
            "base_grad_tensor_count": base_count, "gradients_finite": finite}


def branch_backward(trainer, model, prepared, branch, diagnostic_advantages=None):
    captured = {}
    original = trainer._get_per_token_logps_and_entropies

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if torch.is_grad_enabled():
            result[0].retain_grad()
            captured["logps"] = result[0]
        return result

    trainer._get_per_token_logps_and_entropies = capture
    model.zero_grad(set_to_none=True)
    try:
        if branch == "cot":
            ids = torch.cat([prepared["prompt_ids"], prepared["completion_ids"]], 1)
            mask = torch.cat([prepared["prompt_mask"], prepared["completion_mask"]], 1)
            loss, stats = trainer._branch_loss(
                model, ids, mask, prepared["completion_mask"],
                prepared["old_per_token_logps"],
                (prepared["advantages"] if diagnostic_advantages is None else diagnostic_advantages),
                "cot")
            expected_shape = tuple(prepared["completion_mask"].shape)
        else:
            sid = trainer._sid_rollout
            action = torch.ones((SID_G, 3), dtype=torch.long, device=sid["input_ids"].device)
            loss, stats = trainer._branch_loss(
                model, sid["input_ids"], sid["attention_mask"], action,
                sid["old_per_token_logps"],
                (sid["advantages"] if diagnostic_advantages is None else diagnostic_advantages),
                "sid")
            expected_shape = (SID_G, 3)
        trainer.accelerator.backward(loss)
        logp_grad = captured["logps"].grad.detach().float()
        result = {"loss": float(loss.detach()), "action_logp_shape": list(logp_grad.shape),
                  "expected_action_logp_shape": list(expected_shape),
                  "action_logp_grad_abs_sum": float(logp_grad.abs().sum()), **stats,
                  **grad_summary(model)}
    finally:
        trainer._get_per_token_logps_and_entropies = original
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="GR-REC-THINK-DUAL-BEAM8-V2-PREFLIGHT-20260828")
    args = parser.parse_args(argv)

    parent = validate_parent_adapter()
    provenance = assert_runtime_import_provenance()
    nccl = configure_single_node_nccl(initialize=True)
    rank, world = int(nccl["local_rank"]), int(nccl["world_size"])
    if world != 4:
        raise RuntimeError(f"DUAL_BEAM8_PREFLIGHT_REQUIRES_4_GPU: {world}")
    torch.manual_seed(SEED + rank)
    random.seed(SEED + rank)
    os.environ["GRPO_MONITOR"] = "0"

    plan_args = baseline_runner.build_arg_parser().parse_args([
        "--run-id", args.run_id, "--n-groups", "all", "--probe-groups", "4",
        "--probe-every-steps", "200", "--save-steps", "250", "--save-total-limit", "8",
    ])
    plan = prepare_dual_run_plan(plan_args)
    group_ids = list(plan["dataset"]["recommendation_group_id"])
    if PREFLIGHT_GROUP_ID not in group_ids:
        raise RuntimeError("DUAL_BEAM8_PREFLIGHT_GROUP_MISSING")
    dataset = plan["dataset"].select([group_ids.index(PREFLIGHT_GROUP_ID)])

    from grpo_model import ADAPTER, BASE, load_model
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())
    base_versions = {name: parameter._version for name, parameter in model.named_parameters()
                     if "lora" not in name.lower()}
    base_requires_grad = sum(parameter.requires_grad for name, parameter in model.named_parameters()
                             if "lora" not in name.lower())
    checksum_before = lora_checksum(model)

    monitor = DualManifestWriter(MonitorWriter(False, run_id=args.run_id, rank=rank))
    runtime = DualBeam8Runtime(model, tokenizer)
    cfg = make_dual_grpo_config("/tmp/grpo-dual-beam8-preflight", 2, 1e-6, SEED,
                                save_strategy="no", save_total_limit=None)
    trainer = ThinkDualBeam8Trainer(
        model=model, args=cfg, processing_class=tokenizer, train_dataset=dataset,
        reward_funcs=[make_dual_beam8_reward_func(runtime)], monitor_writer=monitor)
    trainer.current_gradient_accumulation_steps = 1

    iterator = iter(trainer.get_train_dataloader())
    prepared = trainer._prepare_inputs(next(iterator))
    calls_after_first = runtime.beam_calls
    fingerprint_first = trainer._dual_rollout_fingerprint
    prepared_second = trainer._prepare_inputs(next(iterator))
    cache_parity = all(torch.equal(prepared[key], prepared_second[key]) for key in (
        "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
        "old_per_token_logps", "advantages"))
    iteration2_reuse = (runtime.beam_calls == calls_after_first and
                        trainer._dual_rollout_fingerprint == fingerprint_first and cache_parity)

    records = trainer._sid_rollout["global_records"]
    cot_rewards = [row["cot_reward"] for row in records]
    cot_advantages = population_advantages(cot_rewards).tolist()
    sid_rewards = [row["sid_rewards"] for row in records]
    sid_advantages = independent_sid_advantages(sid_rewards).tolist()
    wrong_g32 = population_advantages([x for group in sid_rewards for x in group]).reshape(4, 8)
    per_g8_parity = all(torch.allclose(torch.tensor(sid_advantages[index]),
                                       population_advantages(group), atol=1e-6)
                        for index, group in enumerate(sid_rewards))
    no_g32 = per_g8_parity and (
        not torch.allclose(torch.tensor(sid_advantages), wrong_g32)
        or all(len(set(group)) == 1 for group in sid_rewards)
    )

    real_signal_observed = bool(any(value != 0 for value in cot_advantages) or
                                any(value != 0 for row in sid_advantages for value in row))
    # A real all-zero rollout correctly has no gradient. For the boundary-only
    # audit, keep the immutable real actions and old logps but use an explicitly
    # diagnostic zero-mean advantage. This never enters production trainer math.
    diagnostic_cot_global = population_advantages([0.0, 0.5, 2.0, 8.0]).to(model.device)
    diagnostic_cot_local = diagnostic_cot_global[rank:rank + 1]
    diagnostic_sid = population_advantages([-1.0, -0.25, 0.0, 0.5, 2.0, 8.0, 2.0, 0.5]).to(model.device)
    cot_grad = branch_backward(trainer, model, prepared, "cot", diagnostic_cot_local)
    sid_grad = branch_backward(trainer, model, prepared, "sid", diagnostic_sid)
    model.zero_grad(set_to_none=True)
    ids = torch.cat([prepared["prompt_ids"], prepared["completion_ids"]], 1)
    mask = torch.cat([prepared["prompt_mask"], prepared["completion_mask"]], 1)
    cot_loss, _ = trainer._branch_loss(model, ids, mask, prepared["completion_mask"],
                                       prepared["old_per_token_logps"], diagnostic_cot_local, "cot")
    sid = trainer._sid_rollout
    sid_action = torch.ones((SID_G, 3), dtype=torch.long, device=sid["input_ids"].device)
    sid_loss, _ = trainer._branch_loss(model, sid["input_ids"], sid["attention_mask"], sid_action,
                                       sid["old_per_token_logps"], diagnostic_sid, "sid")
    trainer.accelerator.backward(cot_loss + sid_loss)
    combined_grad = grad_summary(model)
    model.zero_grad(set_to_none=True)

    # Existing production-shaped Beam32 Probe4, including step-0 RNG restoration.
    probe_root = args.output.parent / (args.run_id + "-monitor")
    probe_monitor = DualManifestWriter(MonitorWriter(True, probe_root, args.run_id, rank))
    probe_beam32 = baseline_runner.make_beam32_fn(model, tokenizer, monitor_writer=probe_monitor)
    evaluator = FixedProbeEvaluator(
        trainer, plan["probe_records"], list(FIXED_PROBE4_IDS), probe_beam32,
        probe_monitor, SEED, 200)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(trainer.accelerator.device).clone()
    python_rng = random.getstate()
    probe_checksum_before = lora_checksum(model)
    evaluator.evaluate(0, "baseline-preflight")
    probe_rng_restored = (torch.equal(cpu_rng, torch.get_rng_state()) and
                          torch.equal(cuda_rng, torch.cuda.get_rng_state(trainer.accelerator.device)) and
                          python_rng == random.getstate())
    probe_parameter_unchanged = probe_checksum_before == lora_checksum(model)

    checksum_after = lora_checksum(model)
    base_version_changed = sum(parameter._version != base_versions[name]
                               for name, parameter in model.named_parameters()
                               if name in base_versions)
    local = {
        "rank": rank, "group_id": records[rank]["recommendation_group_id"],
        "cot_reward": cot_rewards[rank], "cot_advantage": cot_advantages[rank],
        "sid_rewards": sid_rewards[rank], "sid_advantages": sid_advantages[rank],
        "beam_candidate_count": len(records[rank]["beam_candidate_ids"]),
        "abc3": all(len(row) == 3 for row in records[rank]["beam_candidate_ids"]),
        "beam_calls": runtime.beam_calls, "cot_gradient": cot_grad,
        "sid_gradient": sid_grad, "combined_gradient": combined_grad,
        "checksum_before": checksum_before, "checksum_after": checksum_after,
        "base_requires_grad_count": int(base_requires_grad),
        "base_version_changed_count": int(base_version_changed),
        "probe_rng_restored": probe_rng_restored,
        "probe_parameter_unchanged": probe_parameter_unchanged,
        "real_signal_observed": real_signal_observed,
    }
    rank_rows = gather(local)
    if rank == 0:
        checks = {
            "world_size_4": world == 4,
            "same_business_group": len({row["group_id"] for row in rank_rows}) == 1,
            "one_cot_per_rank": len(rank_rows) == COT_G,
            "beam8_once_per_rank": all(row["beam_calls"] == 1 for row in rank_rows),
            "exact_4x8_abc3": all(row["beam_candidate_count"] == 8 and row["abc3"] for row in rank_rows),
            "four_independent_g8_not_g32": no_g32,
            "iteration2_exact_cache_reuse": iteration2_reuse,
            "cot_old_logp_full_forward_detached": (prepared["old_per_token_logps"] is not None and
                                                     not prepared["old_per_token_logps"].requires_grad),
            "sid_old_logp_full_forward_detached": not sid["old_per_token_logps"].requires_grad,
            "cot_action_gradient_nonzero": sum(row["cot_gradient"]["action_logp_grad_abs_sum"] for row in rank_rows) > 0,
            "sid_active_action_gradient_nonzero": any(row["sid_gradient"]["action_logp_grad_abs_sum"] > 0 for row in rank_rows),
            "sid_zero_std_production_advantage_zero": all(
                (len(set(row["sid_rewards"])) != 1) or
                all(value == 0.0 for value in row["sid_advantages"])
                for row in rank_rows),
            "sid_action_exactly_abc3": all(row["sid_gradient"]["action_logp_shape"] == [8, 3] for row in rank_rows),
            "combined_lora_gradient_finite": all(row["combined_gradient"]["gradients_finite"] and
                                                  row["combined_gradient"]["lora_grad_norm"] > 0 for row in rank_rows),
            "base_no_gradient": all(row["combined_gradient"]["base_grad_tensor_count"] == 0 for row in rank_rows),
            "base_frozen": all(row["base_requires_grad_count"] == 0 for row in rank_rows),
            "parameter_checksum_unchanged": all(row["checksum_before"] == row["checksum_after"] for row in rank_rows),
            "base_version_unchanged": all(row["base_version_changed_count"] == 0 for row in rank_rows),
            "exact_probe4": tuple(plan["probe_group_ids"]) == FIXED_PROBE4_IDS,
            "probe_train_zero_overlap": not set(plan["probe_group_ids"]) & set(plan["dataset"]["recommendation_group_id"]),
            "probe_step0_beam32_completed": True,
            "probe_rng_restored": all(row["probe_rng_restored"] for row in rank_rows),
            "probe_parameter_unchanged": all(row["probe_parameter_unchanged"] for row in rank_rows),
            "runtime_import_provenance": provenance["runtime_import_provenance"] == "PASS",
        }
        failures = [name for name, passed in checks.items() if not passed]
        payload = {
            "experiment": "GR_REC_ThinkDualBeam8_v2", "preflight_pass": not failures,
            "failure_reasons": failures, "gpu_validation_launch_commit": git_head(),
            "base": BASE, "adapter": ADAPTER, "parent_adapter": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256, "parent_validation": parent,
            "probe4_ids": list(FIXED_PROBE4_IDS), "train_probe_overlap": [],
            "preflight_group_id": PREFLIGHT_GROUP_ID,
            "real_signal_observed": real_signal_observed,
            "gradient_boundary_mode": "immutable real actions/old-logps with diagnostic-only advantages",
            "gradient_diagnostic_advantages": {
                "cot_g4": diagnostic_cot_global.tolist(),
                "sid_g8": diagnostic_sid.tolist(),
                "production_reward_or_advantage_changed": False,
            },
            "cot_rewards": cot_rewards, "cot_advantages": cot_advantages,
            "sid_rewards_4x8": sid_rewards, "sid_advantages_4x8": sid_advantages,
            "old_logp": {"cot": "unchanged rollout policy no-grad full-forward rescore",
                         "sid": "unchanged rollout policy no-grad full-forward ABC3 rescore",
                         "generate_scores_used": False},
            "checks": checks, "rank_rows": rank_rows, "world_size": world,
            "nccl": nccl, **provenance,
            "optimizer_steps": 0, "scheduler_steps": 0, "parameter_update": False,
            "checkpoint_saved": False, "training_started": False,
            "formal_training_started": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"PREFLIGHT_PASS": payload["preflight_pass"],
                          "failures": failures, "output": str(args.output)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
