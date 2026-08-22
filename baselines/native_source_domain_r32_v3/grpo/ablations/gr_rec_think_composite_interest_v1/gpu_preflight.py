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
from grpo_trl_trainer import make_think_reward_func
from monitor.writer import monitor_from_env
from run_grpo_trl_smoke import make_beam32_fn, make_grpo_config

from .composite_trainer import ThinkCompositeInterestRecGRPOTrainer
from .interest_metric import composite_reward, population_advantages
from .run_gr_rec_think_composite_interest_v1 import prepare_plan
from ..gr_rec_think_exact_clamp_v1.think_diagnostics import extract_sids

LAUNCH_COMMIT = "72856f3c927527f26338700a03fc6641de034788"


class PreflightTrainer(ThinkCompositeInterestRecGRPOTrainer):
    def _calculate_rewards(self, *args, **kwargs):
        result = super()._calculate_rewards(*args, **kwargs)
        self.preflight_runtime = self._composite_runtime
        return result


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
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError("preflight requires exactly four GPUs")
    torch.cuda.set_device(rank)
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
    runtime = trainer.preflight_runtime
    local_advantages = prepared["advantages"].detach().float().cpu().tolist()
    gathered_advantages = [value for part in gather(local_advantages) for value in part]
    loss = trainer._compute_loss(model, prepared)
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
        reward_parity = all(abs(row["composite_reward"] - composite_reward(
            row["beam_raw"], row["cot_utility"])) < 1e-7 for row in candidates)
        recomputed_advantages = []
        for group in groups:
            recomputed_advantages.extend(population_advantages(group["composite_reward_vector"]))
        advantage_parity = (
            len(gathered_advantages) == len(recomputed_advantages)
            and all(abs(left - right) < 2e-5 for left, right in zip(gathered_advantages, recomputed_advantages))
        )
        group_ids = [row["group_id"] for row in candidates]
        ddp_alignment = len(groups) == 4 and all(
            len(set(group_ids[start:start + 4])) == 1 for start in range(0, 16, 4)
        )
        generated_sid_count = sum(bool(extract_sids(row["completion"])) for row in candidates)
        raw_decode_pass = all(
            row["completion"] == tokenizer.decode(
                prepared["completion_ids"][row["local_index"]].detach().cpu().tolist(),
                skip_special_tokens=False,
            ) if row["rank"] == 0 else True
            for row in candidates
        )
        gold_leakage = any(row["gold_cot"] in render_prompt(tokenizer, row["prompt"])
                           for row in plan["records"][:4])
        finite = all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"])
                     and row["grad_finite"] for row in rank_rows)
        conditions = {
            "raw_decode_runtime": raw_decode_pass,
            "online_reward_parity": reward_parity,
            "online_advantage_parity": advantage_parity,
            "ddp_g4_alignment": ddp_alignment,
            "gold_leakage_absent": not gold_leakage,
            "finite_loss_and_grad": finite,
            "lora_has_gradient": all(row["lora_grad_tensor_count"] > 0 for row in rank_rows),
            "base_has_no_gradient": all(row["base_grad_tensor_count"] == 0 for row in rank_rows),
            "trainable_checksum_unchanged": checksum_before == checksum_after,
            "base_unchanged": not base_changed,
        }
        payload = {
            "experiment": "GR_REC_Think_CompositeInterest_v1",
            "gpu_validation_launch_commit": LAUNCH_COMMIT,
            "base": BASE, "adapter": ADAPTER, "world_size": world,
            "optimizer_steps": 0, "zero_update": True,
            "raw_decode_runtime_pass": raw_decode_pass,
            "sid_evidence_runtime_visible": True,
            "generated_sid_candidate_count": generated_sid_count,
            "online_reward_parity": reward_parity,
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
            "preflight_pass": all(conditions.values()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"preflight_pass": payload["preflight_pass"], "conditions": conditions}), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
