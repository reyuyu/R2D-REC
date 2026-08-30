"""Four-rank zero-update preflight for V6-A Fine-grained Frontier GRPO."""
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

from monitor.writer import MonitorWriter
from ablations.gr_rec_think_composite_interest_v1.single_node_nccl import (
    configure_single_node_nccl,
)
from .official_finegrained_trainer import (
    COT_G, SID_G, OfficialFineGrainedRuntime, ThinkOfficialFineGrainedTrainer,
    make_official_finegrained_reward_func, population_advantages,
    rollout_fingerprint,
)
from .official_probe import FineGrainedOfficialProbeEvaluator
from .run_official_finegrained_train import (
    FIXED_PROBE4_IDS, EXTRA_PROBE4_IDS, FIXED_PROBE8_IDS,
    HISTORY_COPY_PROBE4_IDS, PARENT_ADAPTER, PARENT_ADAPTER_SHA256,
    MixedV6MonitorWriter, assert_runtime_import_provenance, baseline_runner,
    build_v6_parser, make_v6_config, prepare_v6_run_plan,
    validate_parent_adapter, validate_source_dataset,
)

SEED = 20260816
DEFAULT_SIGNAL_GROUP = "053ed69c31f53492b3d60fd84be9f8dd24f1c9cf4bf4f72f0a1cbcdbb4740c7c"


def gather(value):
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[2],
    ).strip()


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
            attention = torch.cat([prepared["prompt_mask"], prepared["completion_mask"]], 1)
            action = prepared["completion_mask"]
            old = prepared["old_per_token_logps"]
            advantages = prepared["advantages"]
        else:
            item = trainer._v6_rollout
            ids, attention, action = (
                item["input_ids"], item["attention_mask"], item["action_mask"]
            )
            old, advantages = item["old_per_token_logps"], item["advantages"]
        loss, stats = trainer._branch_loss(
            model, ids, attention, action, old, advantages, branch
        )
        trainer.accelerator.backward(loss)
        logp_grad = captured["logps"].grad.detach().float()
        return {
            "loss": float(loss.detach()),
            "current_logp_requires_grad": bool(captured["logps"].requires_grad),
            "action_logp_grad_abs_sum": float(logp_grad.abs().sum()),
            "masked_logp_grad_abs_sum": float(logp_grad[action.bool()].abs().sum()),
            "inactive_logp_grad_abs_sum": float(logp_grad[~action.bool()].abs().sum()),
            "action_logp_shape": list(logp_grad.shape),
            **stats, **grad_summary(model),
        }
    finally:
        trainer._get_per_token_logps_and_entropies = original


def _plan_args(run_id):
    values = [
        "--run-id", run_id, "--n-groups", "all", "--max-steps", "2",
        "--probe-groups", "8", "--probe-every-steps", "50",
        "--save-steps", "50", "--save-total-limit", "64",
        "--secondary-probe-suite", "history_copy_exact",
        "--secondary-probe-every-steps", "50",
    ]
    for gid in FIXED_PROBE8_IDS:
        values += ["--probe-group-id", gid]
    for gid in HISTORY_COPY_PROBE4_IDS:
        values += ["--secondary-probe-group-id", gid]
    return build_v6_parser().parse_args(values)


def _probe_partition(rows, group_ids, suite=None):
    wanted = set(group_ids)
    return [
        row for row in rows
        if row.get("step") == 0 and row.get("group_id") in wanted
        and row.get("probe_suite") == suite
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group-id", default=DEFAULT_SIGNAL_GROUP)
    args = parser.parse_args(argv)

    parent = validate_parent_adapter()
    source = validate_source_dataset()
    provenance = assert_runtime_import_provenance()
    nccl = configure_single_node_nccl(initialize=True)
    rank, world = int(nccl["local_rank"]), int(nccl["world_size"])
    if world != 4:
        raise RuntimeError(f"V6_PREFLIGHT_REQUIRES_4_GPU: {world}")
    torch.manual_seed(SEED + rank)
    random.seed(SEED + rank)
    os.environ["GRPO_MONITOR"] = "0"

    plan = prepare_v6_run_plan(_plan_args(args.run_id))
    group_ids = list(plan["dataset"]["recommendation_group_id"])
    if args.group_id not in group_ids:
        raise RuntimeError(f"V6_PREFLIGHT_SIGNAL_GROUP_MISSING: {args.group_id}")
    dataset = plan["dataset"].select([group_ids.index(args.group_id)])

    from grpo_model import ADAPTER, BASE, load_model
    model, tokenizer, _ = load_model(f"cuda:{rank}")
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name.lower())
    base_versions = {
        name: parameter._version for name, parameter in model.named_parameters()
        if "lora" not in name.lower()
    }
    base_requires_grad = sum(
        parameter.requires_grad for name, parameter in model.named_parameters()
        if "lora" not in name.lower()
    )
    lora_sha_before = parameter_sha(model, lora=True)
    base_sha_before = parameter_sha(model, lora=False)

    monitor = MixedV6MonitorWriter(MonitorWriter(False, run_id=args.run_id, rank=rank))
    runtime = OfficialFineGrainedRuntime(model, tokenizer)
    cfg = make_v6_config(
        "/tmp/grpo-v6-zero-update-preflight", 2, 1e-6, SEED,
        save_strategy="no", save_total_limit=None,
    )
    trainer = ThinkOfficialFineGrainedTrainer(
        model=model, args=cfg, processing_class=tokenizer, train_dataset=dataset,
        reward_funcs=[make_official_finegrained_reward_func(runtime)],
        monitor_writer=monitor,
    )
    trainer.current_gradient_accumulation_steps = 1

    iterator = iter(trainer.get_train_dataloader())
    prepared1 = trainer._prepare_inputs(next(iterator))
    fingerprint1 = trainer._v6_fingerprint
    calls1 = runtime.sample_calls
    frozen1 = {
        "advantages": trainer._v6_rollout["advantages"].detach().cpu().clone(),
        "masks": trainer._v6_rollout["action_mask"].detach().cpu().clone(),
        "old_logp": trainer._v6_rollout["old_per_token_logps"].detach().cpu().clone(),
    }
    prepared2 = trainer._prepare_inputs(next(iterator))
    fingerprint2 = trainer._v6_fingerprint
    cache_parity = all(torch.equal(prepared1[key], prepared2[key]) for key in (
        "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
        "old_per_token_logps", "advantages",
    ))
    sid_cache_parity = all(torch.equal(frozen1[key], trainer._v6_rollout[
        {"advantages": "advantages", "masks": "action_mask", "old_logp": "old_per_token_logps"}[key]
    ].detach().cpu()) for key in frozen1)

    records = runtime.last_global_records
    record = runtime.last_local
    policy_rows = []
    for prepared in (prepared1, prepared2):
        model.zero_grad(set_to_none=True)
        loss = trainer._compute_loss(model, prepared)
        trainer.accelerator.backward(loss)
        policy_rows.append({
            "total_loss": float(loss.detach()),
            "fingerprint": trainer._v6_fingerprint,
            **grad_summary(model),
        })

    cot_gradient = branch_backward(trainer, model, prepared1, "cot")
    sid_gradient = branch_backward(trainer, model, prepared1, "sid")
    model.zero_grad(set_to_none=True)

    probe_root = args.output.parent / (args.run_id + "-monitor")
    probe_monitor = MixedV6MonitorWriter(
        MonitorWriter(True, probe_root, args.run_id, rank)
    )
    probe_beam32 = baseline_runner.make_beam32_fn(
        model, tokenizer, monitor_writer=probe_monitor
    )
    primary = FineGrainedOfficialProbeEvaluator(
        trainer, plan["probe_records"], list(FIXED_PROBE8_IDS), probe_beam32,
        probe_monitor, SEED, 50,
    )
    history = FineGrainedOfficialProbeEvaluator(
        trainer, plan["secondary_probe_records"], list(HISTORY_COPY_PROBE4_IDS),
        probe_beam32, probe_monitor, SEED, 50, probe_suite="history_copy_exact",
    )
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(trainer.accelerator.device).clone()
    python_rng = random.getstate()
    beam_sentinel = {"v6_preflight_sentinel": True}
    model._beam_stats = beam_sentinel
    probe_lora_before = parameter_sha(model, lora=True)
    primary.evaluate(0, "v6-zero-update-preflight")
    history.evaluate(0, "v6-zero-update-preflight-history")
    probe_rng_restored = (
        torch.equal(cpu_rng, torch.get_rng_state())
        and torch.equal(cuda_rng, torch.cuda.get_rng_state(trainer.accelerator.device))
        and python_rng == random.getstate()
    )
    probe_beam_stats_restored = getattr(model, "_beam_stats", None) is beam_sentinel
    probe_parameter_unchanged = probe_lora_before == parameter_sha(model, lora=True)
    delattr(model, "_beam_stats")

    lora_sha_after = parameter_sha(model, lora=True)
    base_sha_after = parameter_sha(model, lora=False)
    base_version_changed = sum(
        parameter._version != base_versions[name]
        for name, parameter in model.named_parameters() if name in base_versions
    )
    local = {
        "rank": rank, "group_id": record["recommendation_group_id"],
        "target_domain": record["target_domain"], "cot_length": record["cot_length"],
        "raw_q_reward": record["raw_q_rewards"],
        "hit_A": record["hit_A"], "hit_B": record["hit_B"], "hit_C": record["hit_C"],
        "token_advantages": record["token_advantages"],
        "token_masks": record["token_masks"],
        "active_tokens": [record["active_A_tokens"], record["active_B_tokens"],
                          record["active_C_tokens"]],
        "zero_std": [record["zero_std_A"], record["zero_std_B"], record["zero_std_C"]],
        "cot_reward": record["cot_reward"], "cot_advantage": record["cot_advantage"],
        "cot_gradient": cot_gradient, "sid_gradient": sid_gradient,
        "lora_sha_before": lora_sha_before, "lora_sha_after": lora_sha_after,
        "base_sha_before": base_sha_before, "base_sha_after": base_sha_after,
        "base_requires_grad_count": int(base_requires_grad),
        "base_version_changed_count": int(base_version_changed),
        "probe_rng_restored": probe_rng_restored,
        "probe_beam_stats_restored": probe_beam_stats_restored,
        "probe_parameter_unchanged": probe_parameter_unchanged,
    }
    rank_rows = gather(local)
    dist.barrier()

    if rank == 0:
        probe_rows = [
            json.loads(line) for line in
            (probe_monitor.run_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        heldout = _probe_partition(probe_rows, FIXED_PROBE4_IDS)
        diagnostic = _probe_partition(probe_rows, EXTRA_PROBE4_IDS)
        history_rows = _probe_partition(
            probe_rows, HISTORY_COPY_PROBE4_IDS, "history_copy_exact"
        )
        signal = any(
            any(row[key]) for row in records for key in ("hit_A", "hit_B", "hit_C")
        )
        cot_sum_parity = all(
            math.isclose(row["cot_reward"], sum(row["raw_q_rewards"]), abs_tol=1e-9)
            for row in records
        )
        checks = {
            "world_size_4": world == 4,
            "same_business_group": len({row["group_id"] for row in rank_rows}) == 1,
            "global_g4": len(records) == COT_G,
            "four_independent_g8_never_g32": (
                len(records) == 4 and all(len(row["raw_q_rewards"]) == SID_G for row in records)
            ),
            "cot_reward_raw_sum": cot_sum_parity,
            "token_advantage_mask_8x3": all(
                len(row["token_advantages"]) == 8 and len(row["token_advantages"][0]) == 3
                and len(row["token_masks"]) == 8 and len(row["token_masks"][0]) == 3
                for row in records
            ),
            "frontier_gate_contract": all(
                detail["mask_A"] == 1 and detail["mask_B"] == detail["hit_A"]
                and detail["mask_C"] == detail["hit_B"]
                for row in records for detail in row["candidate_details"]
            ),
            "real_positive_signal_observed": signal,
            "old_logp_detached_full_forward": (
                prepared1["old_per_token_logps"] is not None
                and not prepared1["old_per_token_logps"].requires_grad
                and not trainer._v6_rollout["old_per_token_logps"].requires_grad
            ),
            "current_logp_autograd": all(
                row[name]["current_logp_requires_grad"]
                for row in rank_rows for name in ("cot_gradient", "sid_gradient")
            ),
            "inactive_sid_tokens_no_gradient": all(
                row["sid_gradient"]["inactive_logp_grad_abs_sum"] == 0.0 for row in rank_rows
            ),
            "iteration2_exact_reuse": (
                fingerprint1 == fingerprint2 and cache_parity and sid_cache_parity
                and calls1 == runtime.sample_calls
                and all(row["fingerprint"] == fingerprint1 for row in policy_rows)
            ),
            "lora_grad_finite": all(
                row["lora_grad_tensor_count"] > 0 and row["gradients_finite"]
                for policy in policy_rows for row in [policy]
            ),
            "base_no_grad": all(
                row[name]["base_grad_tensor_count"] == 0
                for row in rank_rows for name in ("cot_gradient", "sid_gradient")
            ),
            "base_frozen": all(
                row["base_requires_grad_count"] == 0 and row["base_version_changed_count"] == 0
                for row in rank_rows
            ),
            "lora_sha_unchanged": all(
                row["lora_sha_before"] == row["lora_sha_after"] for row in rank_rows
            ),
            "base_sha_unchanged": all(
                row["base_sha_before"] == row["base_sha_after"] for row in rank_rows
            ),
            "probe0_heldout4": len(heldout) == 4,
            "probe0_diagnostic4": len(diagnostic) == 4,
            "probe0_history4": len(history_rows) == 4,
            "probe_rng_restored": all(row["probe_rng_restored"] for row in rank_rows),
            "probe_beam_stats_restored": all(
                row["probe_beam_stats_restored"] for row in rank_rows
            ),
            "probe_parameter_unchanged": all(
                row["probe_parameter_unchanged"] for row in rank_rows
            ),
            "no_optimizer_step": trainer.optimizer is None,
            "no_scheduler_step": trainer.lr_scheduler is None,
            "no_checkpoint_save": not list(Path(cfg.output_dir).glob("checkpoint-*")),
            "runtime_import_provenance": provenance["runtime_import_provenance"] == "PASS",
        }
        failures = [name for name, passed in checks.items() if not passed]
        payload = {
            "experiment": "GR_REC_OfficialFineGrained_v6A",
            "preflight_pass": not failures, "failure_reasons": failures,
            "gpu_validation_commit": git_head(), "world_size": world,
            "base": BASE, "adapter": ADAPTER, "parent_adapter": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_validation": parent, "source_validation": source,
            "dataset_guard": plan["dataset_guard"], "preflight_group_id": args.group_id,
            "records": records, "cot_rewards": [row["cot_reward"] for row in records],
            "cot_advantages": population_advantages(
                [row["cot_reward"] for row in records]
            ).tolist(),
            "policy_iterations": policy_rows,
            "branch_gradients": {"cot": cot_gradient, "sid": sid_gradient},
            "parameter_sha": {
                "lora_before": lora_sha_before, "lora_after": lora_sha_after,
                "base_before": base_sha_before, "base_after": base_sha_after,
            },
            "probe0": {
                "heldout_probe4": heldout,
                "training_overlap_diagnostic4": diagnostic,
                "history_copy_probe4": history_rows,
            },
            "checks": checks, "rank_rows": rank_rows,
            "optimizer_steps": 0, "scheduler_steps": 0,
            "parameter_update": False, "checkpoint_saved": False,
            "formal_training_started": False, "nccl": nccl, **provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({
            "PREFLIGHT_PASS": not failures, "failures": failures,
            "output": str(args.output),
        }), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
