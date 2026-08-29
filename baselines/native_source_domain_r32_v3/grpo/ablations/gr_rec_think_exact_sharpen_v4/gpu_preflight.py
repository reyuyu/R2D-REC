"""Four-rank zero-update preflight for GR_REC_ThinkExactSharpen_v4."""
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

from .run_exact_sharpen_train import (
    FIXED_PROBE4_IDS, PARENT_ADAPTER, PARENT_ADAPTER_SHA256,
    V4MonitorWriter, assert_runtime_import_provenance, baseline_runner,
    make_v4_config, prepare_v4_run_plan, validate_dataset, validate_parent_adapter,
)
from .dual_probe import DualContractProbeEvaluator
from .exact_sharpen_trainer import (
    COT_G, FREE_SID_TOKENS, OFFICIAL_SID_TOKENS, SID_G,
    ExactSharpenRuntime, ThinkExactSharpenTrainer, branch_is_saturated,
    duplicate_penalties, independent_g8_advantages, make_exact_sharpen_reward_func,
    population_advantages, shape_g8, strict_unique_cot_reward,
)
from monitor.writer import MonitorWriter
from ablations.gr_rec_think_composite_interest_v1.single_node_nccl import configure_single_node_nccl

SEED = 20260816
EVIDENCE = Path("/data/GRPO/data/grpo_tk_positive_groups_1946_20260829/positive_group_evidence.jsonl")


def gather(value):
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[2]).strip()


def parameter_sha(model, *, lora):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ("lora" in name.lower()) != lora:
            continue
        tensor = parameter.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def grad_summary(model):
    lora_sq = 0.0
    lora_count = base_count = 0
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all())
        if "lora" in name.lower():
            lora_count += 1
            lora_sq += float((grad * grad).sum())
        else:
            base_count += 1
    return {
        "lora_grad_norm": math.sqrt(lora_sq),
        "lora_grad_tensor_count": lora_count,
        "base_grad_tensor_count": base_count,
        "gradients_finite": finite,
    }


def select_signal_group(explicit=None):
    rows = [json.loads(line) for line in EVIDENCE.read_text(encoding="utf-8").splitlines()]
    if explicit:
        return next(row for row in rows if row["recommendation_group_id"] == explicit)
    rows.sort(key=lambda row: (
        bool(row.get("has_exact")), bool(row.get("has_ab_or_better")),
        int(row.get("sid_positive_counts", {}).get("Exact", 0)),
        int(row.get("sid_positive_counts", {}).get("AB", 0)),
        float(row.get("cot_reward_max", 0.0)),
    ), reverse=True)
    return rows[0]


def branch_backward(trainer, model, prepared, branch):
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
            action = prepared["completion_mask"]
            old = prepared["old_per_token_logps"]
            advantages = prepared["advantages"]
        else:
            item = trainer._v4_rollout[branch]
            ids, mask, action = item["input_ids"], item["attention_mask"], item["action_mask"]
            old, advantages = item["old_per_token_logps"], item["advantages"]
        loss, stats = trainer._branch_loss(
            model, ids, mask, action, old, advantages, branch)
        trainer.accelerator.backward(loss)
        action_grad = captured["logps"].grad.detach().float()
        return {
            "loss": float(loss.detach()),
            "current_logp_requires_grad": bool(captured["logps"].requires_grad),
            "action_logp_grad_abs_sum": float(action_grad.abs().sum()),
            "action_logp_shape": list(action_grad.shape),
            "expected_action_logp_shape": list(action.shape),
            **stats, **grad_summary(model),
        }
    finally:
        trainer._get_per_token_logps_and_entropies = original


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group-id")
    args = parser.parse_args(argv)

    parent = validate_parent_adapter()
    dataset_guard = validate_dataset()
    provenance = assert_runtime_import_provenance()
    nccl = configure_single_node_nccl(initialize=True)
    rank, world = int(nccl["local_rank"]), int(nccl["world_size"])
    if world != 4:
        raise RuntimeError(f"V4_PREFLIGHT_REQUIRES_4_GPU: {world}")
    torch.manual_seed(SEED + rank)
    random.seed(SEED + rank)
    os.environ["GRPO_MONITOR"] = "0"

    evidence = select_signal_group(args.group_id)
    group_id = evidence["recommendation_group_id"]
    plan_args = baseline_runner.build_arg_parser().parse_args([
        "--run-id", args.run_id, "--n-groups", "all", "--probe-groups", "4",
        "--probe-every-steps", "50", "--save-steps", "50", "--save-total-limit", "64",
    ])
    plan = prepare_v4_run_plan(plan_args)
    group_ids = list(plan["dataset"]["recommendation_group_id"])
    if group_id not in group_ids:
        raise RuntimeError(f"V4_PREFLIGHT_SIGNAL_GROUP_MISSING: {group_id}")
    dataset = plan["dataset"].select([group_ids.index(group_id)])

    from grpo_model import ADAPTER, BASE, load_model
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())
    base_versions = {name: parameter._version for name, parameter in model.named_parameters()
                     if "lora" not in name.lower()}
    base_requires_grad = sum(parameter.requires_grad for name, parameter in model.named_parameters()
                             if "lora" not in name.lower())
    lora_sha_before = parameter_sha(model, lora=True)
    base_sha_before = parameter_sha(model, lora=False)

    monitor = V4MonitorWriter(MonitorWriter(False, run_id=args.run_id, rank=rank))
    runtime = ExactSharpenRuntime(model, tokenizer)
    cfg = make_v4_config("/tmp/grpo-v4-zero-update-preflight", 2, 1e-6, SEED,
                         save_strategy="no", save_total_limit=None)
    trainer = ThinkExactSharpenTrainer(
        model=model, args=cfg, processing_class=tokenizer, train_dataset=dataset,
        reward_funcs=[make_exact_sharpen_reward_func(runtime)], monitor_writer=monitor)
    trainer.current_gradient_accumulation_steps = 1

    iterator = iter(trainer.get_train_dataloader())
    prepared1 = trainer._prepare_inputs(next(iterator))
    fingerprint1 = trainer._v4_fingerprint
    calls1 = (runtime.free_sample_calls, runtime.official_sample_calls)
    prepared2 = trainer._prepare_inputs(next(iterator))
    fingerprint2 = trainer._v4_fingerprint
    cache_parity = all(torch.equal(prepared1[key], prepared2[key]) for key in (
        "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
        "old_per_token_logps", "advantages"))

    records = runtime.last_global_records
    record = runtime.last_local
    free_raw = [row["free_raw_rewards"] for row in records]
    official_raw = [row["official_raw_rewards"] for row in records]
    free_shaped = [row["free_rewards"] for row in records]
    official_shaped = [row["official_rewards"] for row in records]
    cot_rewards = [row["cot_reward"] for row in records]
    cot_advantages = population_advantages(cot_rewards)
    free_advantages = independent_g8_advantages(free_shaped)
    official_advantages = independent_g8_advantages(official_shaped)

    # Exercise both cached policy iterations without any optimizer/scheduler step.
    policy_rows = []
    for prepared in (prepared1, prepared2):
        model.zero_grad(set_to_none=True)
        loss = trainer._compute_loss(model, prepared)
        trainer.accelerator.backward(loss)
        policy_rows.append({"total_loss": float(loss.detach()),
                            "fingerprint": trainer._v4_fingerprint,
                            **grad_summary(model)})
    model.zero_grad(set_to_none=True)

    cot_gradient = branch_backward(trainer, model, prepared1, "cot")
    free_gradient = branch_backward(trainer, model, prepared1, "free")
    official_gradient = branch_backward(trainer, model, prepared1, "official")
    model.zero_grad(set_to_none=True)
    ids = torch.cat([prepared1["prompt_ids"], prepared1["completion_ids"]], 1)
    mask = torch.cat([prepared1["prompt_mask"], prepared1["completion_mask"]], 1)
    cot_loss, _ = trainer._branch_loss(model, ids, mask, prepared1["completion_mask"],
                                       prepared1["old_per_token_logps"], prepared1["advantages"], "cot")
    free = trainer._v4_rollout["free"]
    free_loss, _ = trainer._branch_loss(model, free["input_ids"], free["attention_mask"],
        free["action_mask"], free["old_per_token_logps"], free["advantages"], "free")
    official = trainer._v4_rollout["official"]
    official_loss, _ = trainer._branch_loss(model, official["input_ids"], official["attention_mask"],
        official["action_mask"], official["old_per_token_logps"], official["advantages"], "official")
    trainer.accelerator.backward(cot_loss + 0.5 * free_loss + 0.5 * official_loss)
    combined_gradient = grad_summary(model)
    model.zero_grad(set_to_none=True)

    probe_root = args.output.parent / (args.run_id + "-monitor")
    probe_monitor = V4MonitorWriter(MonitorWriter(True, probe_root, args.run_id, rank))
    probe_beam32 = baseline_runner.make_beam32_fn(model, tokenizer, monitor_writer=probe_monitor)
    evaluator = DualContractProbeEvaluator(
        trainer, plan["probe_records"], list(FIXED_PROBE4_IDS), probe_beam32,
        probe_monitor, SEED, 50)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(trainer.accelerator.device).clone()
    python_rng = random.getstate()
    beam_sentinel = {"preflight_sentinel": True}
    model._beam_stats = beam_sentinel
    probe_lora_before = parameter_sha(model, lora=True)
    evaluator.evaluate(0, "v4-zero-update-preflight")
    probe_rng_restored = (torch.equal(cpu_rng, torch.get_rng_state()) and
                          torch.equal(cuda_rng, torch.cuda.get_rng_state(trainer.accelerator.device)) and
                          python_rng == random.getstate())
    probe_beam_stats_restored = getattr(model, "_beam_stats", None) is beam_sentinel
    probe_parameter_unchanged = probe_lora_before == parameter_sha(model, lora=True)
    delattr(model, "_beam_stats")

    lora_sha_after = parameter_sha(model, lora=True)
    base_sha_after = parameter_sha(model, lora=False)
    base_version_changed = sum(parameter._version != base_versions[name]
                               for name, parameter in model.named_parameters() if name in base_versions)
    local = {
        "rank": rank, "group_id": record["recommendation_group_id"],
        "free_raw_rewards": record["free_raw_rewards"],
        "free_shaped_rewards": record["free_rewards"],
        "official_raw_rewards": record["official_raw_rewards"],
        "official_shaped_rewards": record["official_rewards"],
        "free_duplicate_penalties": record["free_duplicate_penalties"],
        "official_duplicate_penalties": record["official_duplicate_penalties"],
        "cot_reward": record["cot_reward"], "cot_coverage": record["cot_coverage"],
        "free_action_counts": [int(v) for v in free["action_mask"].sum(1)],
        "official_action_counts": [int(v) for v in official["action_mask"].sum(1)],
        "cot_gradient": cot_gradient, "free_gradient": free_gradient,
        "official_gradient": official_gradient, "combined_gradient": combined_gradient,
        "lora_sha_before": lora_sha_before, "lora_sha_after": lora_sha_after,
        "base_sha_before": base_sha_before, "base_sha_after": base_sha_after,
        "base_requires_grad_count": int(base_requires_grad),
        "base_version_changed_count": int(base_version_changed),
        "probe_rng_restored": probe_rng_restored,
        "probe_beam_stats_restored": probe_beam_stats_restored,
        "probe_parameter_unchanged": probe_parameter_unchanged,
        "sample_calls": [runtime.free_sample_calls, runtime.official_sample_calls],
    }
    rank_rows = gather(local)
    dist.barrier()

    if rank == 0:
        probe_rows = [
            json.loads(line)
            for line in (probe_monitor.run_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        probe0 = [row for row in probe_rows if row.get("step") == 0]
        duplicate_parity = all(
            row["free_duplicate_penalties"] == duplicate_penalties(row["free_sids"], row["free_raw_rewards"])
            and row["official_duplicate_penalties"] == duplicate_penalties(row["official_sids"], row["official_raw_rewards"])
            for row in records)
        shaping_parity = all(
            shape_g8(row["free_sids"], row["free_raw_rewards"], row["free_saturated"])[0] == row["free_rewards"]
            and shape_g8(row["official_sids"], row["official_raw_rewards"], row["official_saturated"])[0] == row["official_rewards"]
            for row in records)
        cot_free_only = all(math.isclose(
            strict_unique_cot_reward(row["free_sids"], row["free_raw_rewards"], row["free_saturated"])[0],
            row["cot_reward"], abs_tol=1e-9) for row in records)
        checks = {
            "world_size_4": world == 4,
            "same_business_group": len({row["group_id"] for row in rank_rows}) == 1,
            "global_g4": len(records) == COT_G,
            "topology_4_free_g8_4_official_g8": (
                len(free_raw) == len(official_raw) == 4 and
                all(len(row) == SID_G for row in free_raw + official_raw)),
            "never_g16_g64_normalization": (
                tuple(free_advantages.shape) == (4, 8) and tuple(official_advantages.shape) == (4, 8)),
            "a_plus_count_all_ranks_consistent": all(
                row["free_a_plus_count"] == records[0]["free_a_plus_count"] and
                row["official_a_plus_count"] == records[0]["official_a_plus_count"]
                for row in records),
            "strict_gt8_saturation": (
                records[0]["free_saturated"] == branch_is_saturated(free_raw) and
                records[0]["official_saturated"] == branch_is_saturated(official_raw)),
            "duplicate_penalty_parity": duplicate_parity,
            "reward_shaping_parity": shaping_parity,
            "cot_reward_free_strict_unique_only": cot_free_only,
            "official_excluded_from_cot_reward": cot_free_only,
            "free_mask_first_sid4_or_missing": all(
                count in (0, FREE_SID_TOKENS) for row in rank_rows for count in row["free_action_counts"]),
            "official_mask_exact_abc3": all(
                count == OFFICIAL_SID_TOKENS for row in rank_rows for count in row["official_action_counts"]),
            "old_logp_full_forward_detached": (
                prepared1["old_per_token_logps"] is not None and not prepared1["old_per_token_logps"].requires_grad and
                not free["old_per_token_logps"].requires_grad and not official["old_per_token_logps"].requires_grad),
            "three_current_logp_autograd": all(
                row[name]["current_logp_requires_grad"]
                for row in rank_rows for name in ("cot_gradient", "free_gradient", "official_gradient")),
            "lora_grad_finite": all(row["combined_gradient"]["gradients_finite"] and
                                     row["combined_gradient"]["lora_grad_norm"] > 0 for row in rank_rows),
            "base_no_grad": all(row["combined_gradient"]["base_grad_tensor_count"] == 0 for row in rank_rows),
            "base_frozen": all(row["base_requires_grad_count"] == 0 and
                               row["base_version_changed_count"] == 0 for row in rank_rows),
            "lora_sha_unchanged": all(row["lora_sha_before"] == row["lora_sha_after"] for row in rank_rows),
            "base_sha_unchanged": all(row["base_sha_before"] == row["base_sha_after"] for row in rank_rows),
            "iteration2_same_fingerprint_no_resample": (
                fingerprint1 == fingerprint2 and cache_parity and
                calls1 == (runtime.free_sample_calls, runtime.official_sample_calls) and
                all(row["fingerprint"] == fingerprint1 for row in policy_rows)),
            "dual_probe0_four_groups": len(probe0) == 4,
            "probe_free_schema": all(row.get("probe_free", {}).get("candidate_count") == 32 and
                                     row.get("probe_free", {}).get("reward_denominator") ==
                                     "32 Free candidates (4 CoTs x Sample8)" for row in probe0),
            "probe_official_schema": all(row.get("probe_official", {}).get("candidate_count") == 4 and
                                         row.get("probe_official", {}).get("reward_denominator") ==
                                         "4 sampled CoTs (each evaluated by Beam32)" for row in probe0),
            "probe_think_alias_official": all(row.get("think") == row.get("probe_official") for row in probe0),
            "probe_rng_restored": all(row["probe_rng_restored"] for row in rank_rows),
            "probe_beam_stats_restored": all(row["probe_beam_stats_restored"] for row in rank_rows),
            "probe_parameter_unchanged": all(row["probe_parameter_unchanged"] for row in rank_rows),
            "no_optimizer_step": trainer.optimizer is None,
            "no_scheduler_step": trainer.lr_scheduler is None,
            "no_checkpoint_save": not list(Path(cfg.output_dir).glob("checkpoint-*")),
            "runtime_import_provenance": provenance["runtime_import_provenance"] == "PASS",
        }
        failures = [name for name, passed in checks.items() if not passed]
        payload = {
            "experiment": "GR_REC_ThinkExactSharpen_v4",
            "preflight_pass": not failures, "failure_reasons": failures,
            "gpu_validation_commit": git_head(), "world_size": world,
            "base": BASE, "adapter": ADAPTER, "parent_adapter": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_validation": parent, "dataset_guard": {k: v for k, v in dataset_guard.items() if k != "records"},
            "evidence": evidence, "preflight_group_id": group_id,
            "free32_raw_rewards": [v for group in free_raw for v in group],
            "free32_shaped_rewards": [v for group in free_shaped for v in group],
            "official32_raw_rewards": [v for group in official_raw for v in group],
            "official32_shaped_rewards": [v for group in official_shaped for v in group],
            "free_a_plus_count": records[0]["free_a_plus_count"],
            "official_a_plus_count": records[0]["official_a_plus_count"],
            "free_saturated": records[0]["free_saturated"],
            "official_saturated": records[0]["official_saturated"],
            "cot_rewards": cot_rewards, "cot_advantages": cot_advantages.tolist(),
            "advantage_std": {
                "cot": float(cot_advantages.std(correction=0)),
                "free_per_g8": [float(row.std(correction=0)) for row in free_advantages],
                "official_per_g8": [float(row.std(correction=0)) for row in official_advantages],
            },
            "branch_gradients": {"cot": cot_gradient, "free": free_gradient,
                                 "official": official_gradient, "combined": combined_gradient},
            "policy_iterations": policy_rows,
            "parameter_sha": {"lora_before": lora_sha_before, "lora_after": lora_sha_after,
                              "base_before": base_sha_before, "base_after": base_sha_after},
            "probe0": probe0, "checks": checks, "rank_rows": rank_rows,
            "old_logp": {"cot": "unchanged-policy no-grad full-forward rescore",
                         "free": "unchanged-policy no-grad full-forward rescore",
                         "official": "unchanged-policy no-grad full-forward rescore"},
            "optimizer_steps": 0, "scheduler_steps": 0, "parameter_update": False,
            "checkpoint_saved": False, "formal_training_started": False,
            "nccl": nccl, **provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"PREFLIGHT_PASS": not failures, "failures": failures,
                          "output": str(args.output)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
