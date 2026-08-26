"""Formal four-rank TrueRec Pilot4096 loop with trajectory-neutral Probe20."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "analysis", ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from batch_collator_v1 import collate_business_group  # noqa: E402
from beta_baseline_init import BETA_CHECKPOINT  # noqa: E402
from checkpoint_ddp_v1 import (  # noqa: E402
    capture_rank_rng, load_distributed_checkpoint, restore_rank_rng, rng_fingerprints,
    save_distributed_checkpoint,
)
from distributed_trainer_v1 import (  # noqa: E402
    DDP_WORLD_SIZE, LOCAL_G, all_rank_values_equal, distributed_fail_if, runtime_plan_sha256,
)
from monitoring_v1 import (  # noqa: E402
    MonitoringWriterV1, build_train_explain, build_train_group_record,
    capture_hpr_position, runtime_monitoring_snapshot,
)
from policy_scoring_v1 import parameter_versions  # noqa: E402
from real_three_group_resume_smoke import (  # noqa: E402
    DATASET_IDENTITY, LoraCheckpointState, ORDER_SHA256, load_pilot_order, load_runtime,
)
from rollout_metrics import assess_candidate  # noqa: E402
from rollout_runtime_v1 import (  # noqa: E402
    BusinessGroupRollout, FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, GENERATION_KWARGS,
    RolloutCandidate, extract_generation_artifacts,
)
from run_truerec_pilot_ddp_v1 import (  # noqa: E402
    gather_rank_objects, initialize, run_loaded_group,
)
from run_truerec_pilot_v1 import select_streaming_microbatch_size  # noqa: E402
from truerec_grpo_trainer_v1 import format_credit_tensors  # noqa: E402
from truerec_loss_v1 import multi_positive_log_mass_loss  # noqa: E402
from truerec_runtime_v1 import build_group_runtime_plan  # noqa: E402


RUN_ID = "TRUEREC-V1-BETA-FORMAL4096-4GPU-E1"
TOTAL_STEPS = 4096
CHECKPOINT_EVERY = 256
PROBE_EVERY = 256
PROBE_SEED = 2026082601
PROBE_RECORDS = ROOT / "data/fixed_domain_abc/probe20_records.jsonl"
PROBE_RECORDS_SHA256 = "5f06976e12e60c4576ee0dc0d083d367d423251cf3e65eedbbd728f607ffa913"
PROBE_MANIFEST = ROOT / "data/splits/probe20_group_ids.json"
SOURCE_COMMIT_MARKER = Path("/data/GRPO/.source_commit")


class Formal4096Error(RuntimeError):
    pass


def checkpoint_steps(total_steps: int = TOTAL_STEPS, every: int = CHECKPOINT_EVERY) -> tuple[int, ...]:
    if total_steps < 1 or every < 1 or total_steps % every:
        raise ValueError("checkpoint schedule must exactly divide the formal run")
    return tuple(range(every, total_steps + 1, every))


def probe_steps(total_steps: int = TOTAL_STEPS, every: int = PROBE_EVERY) -> tuple[int, ...]:
    return (0, *checkpoint_steps(total_steps, every))


def formal_group_indices(cursor: int, total_steps: int = TOTAL_STEPS) -> range:
    if not 0 <= cursor <= total_steps:
        raise ValueError("formal cursor outside [0,total_steps]")
    return range(cursor, total_steps)


def driver_state_for_cursor(cursor: int) -> dict[str, int | bool]:
    if not 0 <= cursor <= TOTAL_STEPS:
        raise ValueError("invalid driver cursor")
    return {
        "business_groups_seen": cursor,
        "optimizer_steps": cursor,
        "global_step": cursor,
        "rollouts_completed": cursor,
        "old_rescores_completed": cursor,
        "streaming_backwards_completed": cursor,
        "groups_in_accumulation_window": 0,
        "failed": False,
    }


def validate_monitoring_prefix(rows: Sequence[dict[str, Any]], order: Sequence[str], cursor: int) -> None:
    if len(rows) != cursor:
        raise Formal4096Error("monitoring row count differs from checkpoint cursor")
    for index, row in enumerate(rows):
        if row.get("group_index") != index or row.get("global_step") != index + 1:
            raise Formal4096Error("monitoring step sequence is duplicated or skipped")
        if row.get("recommendation_group_id") != order[index]:
            raise Formal4096Error("monitoring group sequence differs from frozen order")


def validate_fresh_launch(run_directory: Path, resume_checkpoint: Path | None) -> None:
    if resume_checkpoint is None:
        if run_directory.exists():
            raise Formal4096Error("fresh formal run directory must not exist")
        return
    if not run_directory.is_dir() or not resume_checkpoint.is_dir():
        raise Formal4096Error("resume requires existing run and checkpoint directories")
    if resume_checkpoint.parent != run_directory / "checkpoints":
        raise Formal4096Error("resume checkpoint must belong to this formal run")


def load_probe20() -> list[dict[str, Any]]:
    raw = PROBE_RECORDS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROBE_RECORDS_SHA256:
        raise Formal4096Error("frozen Probe20 record SHA mismatch")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    by_id = {str(row["recommendation_group_id"]): row for row in records}
    manifest = json.loads(PROBE_MANIFEST.read_text(encoding="utf-8"))
    ids = [str(row["recommendation_group_id"]) for row in manifest]
    if not len(records) == len(by_id) == len(ids) == len(set(ids)) == 20 or set(ids) != set(by_id):
        raise Formal4096Error("frozen Probe20 identity mismatch")
    return [by_id[group_id] for group_id in ids]


def _rank0_checked(device: torch.device, action: Callable[[], None]) -> None:
    error = None
    if dist.get_rank() == 0:
        try:
            action()
        except Exception as exc:  # propagated to all ranks before continuing
            error = f"{type(exc).__name__}: {exc}"
    errors = gather_rank_objects(error)
    if any(value is not None for value in errors):
        raise Formal4096Error(f"rank0 monitoring action failed: {errors}")
    dist.barrier()


def _probe_metric_group(record, context_ids, completion_ids, generation_logps, id_to_token):
    candidates = []
    for index, (ids, diagnostic) in enumerate(zip(completion_ids, generation_logps)):
        ids = tuple(int(value) for value in ids)
        metrics = assess_candidate(
            list(ids), id_to_token, record["all_gold_abc"],
            record["fixed_domain_token"], record["history_sids"],
        )
        candidates.append(RolloutCandidate(index, ids, tuple(0.0 for _ in ids), metrics, tuple(diagnostic)))
    return BusinessGroupRollout(
        str(record["recommendation_group_id"]), tuple(context_ids),
        str(record["fixed_domain_token"]), tuple(record["all_gold_abc"]), tuple(candidates),
    )


def _probe_candidate_monitoring(group, runtime, format_credits, rank: int) -> list[dict[str, Any]]:
    rows = []
    for candidate_index in range(rank * LOCAL_G, (rank + 1) * LOCAL_G):
        candidate = group.candidates[candidate_index]
        tokens = []
        for action_position, token_id in enumerate(candidate.completion_ids):
            frontier_credit = float(runtime.token_credits[candidate_index, action_position])
            format_credit = float(format_credits[candidate_index, action_position])
            tokens.append({
                "action_position": action_position,
                "position_name": "ABC"[action_position],
                "token_id": int(token_id),
                "old_logp": None,
                "current_logp_pre_step": None,
                "ppo_ratio": None,
                "prefix_gate_active": bool(runtime.token_credit_mask[candidate_index, action_position]),
                "frontier_token_credit": frontier_credit,
                "format_token_credit": format_credit,
                "effective_signed_credit": frontier_credit + format_credit,
                "ppo_unclipped_surrogate": None,
                "ppo_clipped_surrogate": None,
                "ppo_selected_surrogate": None,
                "frontier_loss_contribution": None,
            })
        rows.append({"candidate_index": candidate_index, "tokens": tokens})
    return rows


def seed_probe_rank(rank: int) -> None:
    seed = PROBE_SEED + int(rank)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def execute_with_rng_restored(
    *, capture_fn: Callable[[], Any], restore_fn: Callable[[Any], Any],
    seed_fn: Callable[[], None], fingerprint_fn: Callable[[Any], Any],
    action: Callable[[], Any],
) -> tuple[Any, bool]:
    """Run a probe under isolated RNG and restore the caller trajectory even on failure."""
    saved = capture_fn()
    expected = fingerprint_fn(saved)
    seed_fn()
    result = None
    failure = None
    try:
        result = action()
    except Exception as exc:
        failure = exc
    restored = restore_fn(saved)
    exact = (not isinstance(restored, dict) or all(restored.values())) and fingerprint_fn(capture_fn()) == expected
    if not exact:
        raise Formal4096Error("Probe20 RNG restoration failed")
    if failure is not None:
        raise failure
    return result, True


def probe_one_group(record, model, ddp, renderer, device) -> tuple[dict[str, Any], dict[str, Any] | None]:
    rank = dist.get_rank()
    model.eval(); ddp.eval()
    context_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
    selected_mb = select_streaming_microbatch_size(len(context_ids))
    distributed_fail_if(
        not all_rank_values_equal(record["recommendation_group_id"]) or not all_rank_values_equal(selected_mb),
        "Probe20 rank identity/microbatch mismatch", device,
    )
    generation_kwargs = dict(GENERATION_KWARGS); generation_kwargs["num_return_sequences"] = LOCAL_G
    with torch.no_grad():
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            eos_token_id=list(FORMAL_EOS_TOKEN_IDS), pad_token_id=FORMAL_PAD_TOKEN_ID,
            return_dict_in_generate=True, output_scores=True, **generation_kwargs,
        )
        artifacts = extract_generation_artifacts(context_ids, output, FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS)
        gathered = gather_rank_objects({
            "completion_ids": artifacts.completion_ids,
            "generation_logps": artifacts.generation_score_logps,
        })
        completions = tuple(row for payload in gathered for row in payload["completion_ids"])
        diagnostic_logps = tuple(row for payload in gathered for row in payload["generation_logps"])
        group = _probe_metric_group(
            record, context_ids, completions, diagnostic_logps,
            renderer.tokenizer.convert_ids_to_tokens,
        )
        runtime = build_group_runtime_plan(
            [candidate.metrics for candidate in group.candidates], group.all_gold_abc,
            renderer.tokenizer.convert_tokens_to_ids,
        )
        plan_hash = runtime_plan_sha256(runtime)
        distributed_fail_if(not all_rank_values_equal(plan_hash), "Probe20 runtime plan mismatch", device)
        format_credits, _ = format_credit_tensors(group)
        batch = collate_business_group(group, FORMAL_PAD_TOKEN_ID, "right")
        hpr_positions = []
        local_start, local_stop = rank * LOCAL_G, (rank + 1) * LOCAL_G
        site_count = len(runtime.hpr.sites)
        if site_count:
            for start in range(local_start, local_stop, selected_mb):
                stop = min(start + selected_mb, local_stop)
                inputs = batch.input_ids[start:stop].to(device)
                attention = batch.attention_mask[start:stop].to(device)
                model_output = ddp(input_ids=inputs, attention_mask=attention)
                logits = model_output.logits if hasattr(model_output, "logits") else model_output
                for site_index, site in enumerate(runtime.hpr.sites):
                    weight = 1.0 / (site_count * len(site.onpolicy_positions))
                    for candidate_index, action_position in site.onpolicy_positions:
                        if start <= candidate_index < stop:
                            causal_index = int(batch.causal_logit_indices[candidate_index, action_position])
                            position_loss = multi_positive_log_mass_loss(
                                logits[candidate_index - start, causal_index], site.target_token_ids,
                            )
                            hpr_positions.append(capture_hpr_position(
                                site_index=site_index, site=site, candidate_index=candidate_index,
                                action_position=action_position, position_loss=position_loss,
                                position_weight=weight,
                            ))
    payloads = gather_rank_objects({
        "candidates": _probe_candidate_monitoring(group, runtime, format_credits, rank),
        "hpr_positions": hpr_positions,
    })
    metrics = [candidate.metrics for candidate in group.candidates]
    group_record = {
        "recommendation_group_id": group.recommendation_group_id,
        "target_domain": record["target_domain"],
        "context_token_count": len(context_ids),
        "selected_microbatch_size": selected_mb,
        "format_valid_rate": sum(item["format_valid"] for item in metrics) / 8,
        "A_hit_rate": sum(item["A_hit"] for item in metrics) / 8,
        "AB_hit_rate": sum(item["AB_hit"] for item in metrics) / 8,
        "exact_rate": sum(item["exact"] for item in metrics) / 8,
        "wrong_history_copy_rate": sum(item["wrong_history_copy"] for item in metrics) / 8,
        "hpr_trigger": runtime.hpr.trigger,
    }
    explain = None
    if rank == 0:
        explain = build_train_explain(
            group, runtime_monitoring_snapshot(runtime), payloads,
            renderer.tokenizer.convert_ids_to_tokens,
        )
        explain.update({"mode": "PROBE", "optimizer_update": False})
    return group_record, explain


def run_probe20(
    *, probe_step: int, records: Sequence[dict[str, Any]], model, ddp, renderer,
    device: torch.device, writer: MonitoringWriterV1,
) -> dict[str, Any]:
    rank = dist.get_rank()
    versions = parameter_versions(model)
    def probe_action():
        groups = []
        explains = []
        for record in records:
            group, explain = probe_one_group(record, model, ddp, renderer, device)
            if rank == 0:
                groups.append(group)
                explains.append(explain)
        return groups, explains

    (groups, explains), rng_exact = execute_with_rng_restored(
        capture_fn=lambda: capture_rank_rng(device),
        restore_fn=lambda value: restore_rank_rng(value, device),
        seed_fn=lambda: seed_probe_rank(rank),
        fingerprint_fn=rng_fingerprints,
        action=probe_action,
    )
    distributed_fail_if(
        not rng_exact or parameter_versions(model) != versions,
        "Probe20 changed training RNG or policy parameters", device,
    )
    rank_rng = gather_rank_objects({"rank": rank, "restored": rng_exact})
    summary = {}
    if rank == 0:
        summary = {
            "status": "PASS", "groups": len(groups), "global_G": 8,
            "optimizer_steps_before": probe_step, "optimizer_steps_after": probe_step,
            "training_rng_restored": all(item["restored"] for item in rank_rng),
            "format_valid_rate": sum(row["format_valid_rate"] for row in groups) / len(groups),
            "A_hit_rate": sum(row["A_hit_rate"] for row in groups) / len(groups),
            "AB_hit_rate": sum(row["AB_hit_rate"] for row in groups) / len(groups),
            "exact_rate": sum(row["exact_rate"] for row in groups) / len(groups),
        }
    _rank0_checked(
        device,
        lambda: writer.write_probe(probe_step, summary=summary, groups=groups, explains=explains),
    )
    return summary


def _read_train_rows(run_directory: Path) -> list[dict[str, Any]]:
    path = run_directory / "train_groups.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _write_manifest(run_directory: Path, value: dict[str, Any]) -> None:
    path = run_directory / "run_manifest.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(args) -> None:
    rank, device, resource = initialize()
    try:
        run_directory = Path(args.run_dir)
        resume_checkpoint = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        local_error = None
        if rank == 0:
            try:
                validate_fresh_launch(run_directory, resume_checkpoint)
                if resume_checkpoint is None:
                    run_directory.mkdir(parents=True)
            except Exception as exc:
                local_error = str(exc)
        distributed_fail_if(local_error is not None, f"formal launch contract failed: {local_error}", device)
        dist.barrier()

        records, order, order_info = load_pilot_order()
        probe_records = load_probe20()
        model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
        ddp = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False,
        )
        ddp.eval(); model.eval()
        cursor = 0
        if resume_checkpoint is not None:
            restored = load_distributed_checkpoint(
                resume_checkpoint, model=LoraCheckpointState(model), optimizer=optimizer,
                current_dataset_identity=DATASET_IDENTITY, current_group_ids=list(records), device=device,
            )
            cursor = int(restored["next_group_index"])
            if restored["driver_state"] != driver_state_for_cursor(cursor):
                raise Formal4096Error("checkpoint driver state differs from cursor")
        writer = MonitoringWriterV1(run_directory, total_steps=TOTAL_STEPS, rank=rank)
        _rank0_checked(
            device,
            lambda: validate_monitoring_prefix(_read_train_rows(run_directory), order, cursor),
        )

        manifest = {
            "RUN_ID": RUN_ID, "source_commit": SOURCE_COMMIT_MARKER.read_text().strip(),
            "fresh_beta_checkpoint": str(BETA_CHECKPOINT),
            "order_sha256": ORDER_SHA256, "world_size": 4, "global_G": 8, "local_G": 2,
            "total_steps": TOTAL_STEPS, "checkpoint_every": CHECKPOINT_EVERY,
            "probe_every": PROBE_EVERY, "probe_seed": PROBE_SEED,
            "probe_records_sha256": PROBE_RECORDS_SHA256,
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "resource_at_start": gather_rank_objects(resource),
        }
        if rank == 0 and resume_checkpoint is None:
            _write_manifest(run_directory, manifest)
        dist.barrier()

        initial_probe_directory = run_directory / "probe" / str(cursor)
        if cursor in probe_steps() and not initial_probe_directory.exists():
            run_probe20(
                probe_step=cursor, records=probe_records, model=model, ddp=ddp,
                renderer=renderer, device=device, writer=writer,
            )

        last_checkpoint_step = cursor if cursor in checkpoint_steps() else None
        for group_index in formal_group_indices(cursor):
            group_id = order[group_index]
            report, group, backward, rank_payloads = run_loaded_group(
                group_id, records[group_id], model, ddp, optimizer, renderer, device,
                provenance=provenance, dropout=dropout, trainability=trainability,
                optimizer_audit=optimizer_audit, strict_parameter_audit=False,
            )
            completed = group_index + 1
            if completed in checkpoint_steps():
                checkpoint = run_directory / "checkpoints" / f"checkpoint-step-{completed}"
                save_distributed_checkpoint(
                    checkpoint, model=LoraCheckpointState(model), optimizer=optimizer,
                    driver_state=driver_state_for_cursor(completed), dataset_identity=DATASET_IDENTITY,
                    epoch=0, next_group_index=completed, order=order_info, device=device,
                )
                last_checkpoint_step = completed
            if rank == 0:
                group_record = build_train_group_record(
                    record=records[group_id], group=group, backward=backward,
                    global_step=completed, group_index=group_index,
                    selected_microbatch_size=report["selected_microbatch_size"],
                    gradient_norm=report["gradient"]["lora_grad_norm"],
                    wall_time_seconds=report["wall_time_seconds"], rank_memory=report["rank_memory"],
                )
                explain = build_train_explain(
                    group, backward.runtime_monitoring_plan, rank_payloads,
                    renderer.tokenizer.convert_ids_to_tokens,
                )
            else:
                group_record = explain = None
            _rank0_checked(
                device,
                lambda: writer.record_train_group(
                    group_record, explain, last_checkpoint_step=last_checkpoint_step,
                ),
            )
            if completed in probe_steps():
                run_probe20(
                    probe_step=completed, records=probe_records, model=model, ddp=ddp,
                    renderer=renderer, device=device, writer=writer,
                )
        if rank == 0:
            (run_directory / "formal_complete.json").write_text(
                json.dumps({"status": "PASS", "global_step": TOTAL_STEPS}, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, choices=(RUN_ID,))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
