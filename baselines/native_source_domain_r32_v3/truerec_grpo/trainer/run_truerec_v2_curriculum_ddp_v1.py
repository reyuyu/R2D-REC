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
    save_distributed_checkpoint, tensor_state_sha256,
)
from distributed_trainer_v1 import (  # noqa: E402
    DDP_WORLD_SIZE, LOCAL_G, all_rank_values_equal, distributed_fail_if, runtime_plan_sha256,
)
from monitoring_v1 import (  # noqa: E402
    MonitoringWriterV1, build_train_explain, build_train_group_record,
    capture_hpr_position, runtime_monitoring_snapshot,
)
from policy_scoring_v1 import parameter_versions  # noqa: E402
from real_three_group_resume_smoke import LoraCheckpointState, load_runtime  # noqa: E402
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


RUN_ID = "TRUEREC-V2-V1STEP4096-CURRICULUM2048-2E-4GPU"
TOTAL_STEPS = 4096
EPOCH_STEPS = 2048
CHECKPOINT_EVERY = 256
PROBE_EVERY = 256
PROBE_SEED = 2026082601
PROBE_RECORDS = ROOT / "data/fixed_domain_abc/probe20_records.jsonl"
PROBE_RECORDS_SHA256 = "5f06976e12e60c4576ee0dc0d083d367d423251cf3e65eedbbd728f607ffa913"
PROBE_MANIFEST = ROOT / "data/splits/probe20_group_ids.json"
SOURCE_COMMIT_MARKER = Path("/data/GRPO/.source_commit")
CURRICULUM_DIR = Path("/data/GRPO/truerec_grpo/data/curriculum2048_v2")
RECORDS_SHA256 = "8f1a4567aa0d6a953127e74f5667ab2907b9fbbe4aa3c0b92e27d5ecf1587542"
EPOCH1_ORDER_SHA256 = "cea3fcde04f42b7ccbaa210cb2b8426a90b1bf4aee41f08824592f52e0d27c5e"
PARENT_CHECKPOINT = Path("/data/GRPO/truerec_grpo/runs/TRUEREC-V1-BETA-FORMAL4096-4GPU-E1/checkpoints/checkpoint-step-4096")
PARENT_RUN_ID = "TRUEREC-V1-BETA-FORMAL4096-4GPU-E1"
PARENT_V1_STEP = 4096
ALLOWED_OUTPUT_ROOT = Path("/root/GRPO/truerec_grpo/runs")
DOMAIN_ORDER = ("video", "prod", "ad", "living")
HPR_PRIORITY = {"HPR_C": 0, "HPR_B": 1, "HPR_NONE": 2, "NONE": 2, "HPR_A": 3}


class Formal4096Error(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_metadata(order: Sequence[str], *, name: str) -> dict[str, Any]:
    payload = json.dumps(list(order), separators=(",", ":"), ensure_ascii=True).encode()
    return {"name": name, "count": len(order), "sha256": hashlib.sha256(payload).hexdigest()}


def validate_every64(order: Sequence[str], records: dict[str, dict[str, Any]]) -> None:
    if len(order) % 64:
        raise Formal4096Error("order length is not divisible by 64")
    for start in range(0, len(order), 64):
        counts = {domain: 0 for domain in DOMAIN_ORDER}
        for group_id in order[start:start + 64]:
            counts[str(records[group_id]["target_domain"])] += 1
        if counts != {domain: 16 for domain in DOMAIN_ORDER}:
            raise Formal4096Error(f"every64 domain contract failed at {start}: {counts}")


def load_curriculum() -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    records_path = CURRICULUM_DIR / "records.jsonl"
    order_path = CURRICULUM_DIR / "epoch1_order.json"
    if file_sha256(records_path) != RECORDS_SHA256 or file_sha256(order_path) != EPOCH1_ORDER_SHA256:
        raise Formal4096Error("frozen Curriculum2048 SHA mismatch")
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records = {str(row["recommendation_group_id"]): row for row in rows}
    order = [str(value) for value in json.loads(order_path.read_text(encoding="utf-8"))]
    if not len(rows) == len(records) == len(order) == len(set(order)) == EPOCH_STEPS or set(order) != set(records):
        raise Formal4096Error("Curriculum2048 identity/count contract failed")
    domains = {domain: 0 for domain in DOMAIN_ORDER}
    for row in rows:
        domains[str(row["target_domain"])] += 1
    if domains != {domain: 512 for domain in DOMAIN_ORDER}:
        raise Formal4096Error(f"Curriculum2048 domain contract failed: {domains}")
    validate_every64(order, records)
    return records, order, {"name": "epoch1_frozen", "count": EPOCH_STEPS, "sha256": EPOCH1_ORDER_SHA256}


def build_epoch2_order(explain_rows: Sequence[dict[str, Any]], records: dict[str, dict[str, Any]]) -> list[str]:
    if len(explain_rows) != EPOCH_STEPS:
        raise Formal4096Error("epoch2 derivation requires exactly 2048 Epoch1 explain rows")
    trigger_by_id: dict[str, str] = {}
    for index, row in enumerate(explain_rows):
        if int(row.get("global_step", -1)) != index + 1:
            raise Formal4096Error("Epoch1 explain sequence is duplicated or skipped")
        group_id = str(row["recommendation_group_id"])
        trigger = str(row["hpr_trigger"])
        if group_id in trigger_by_id or trigger not in HPR_PRIORITY:
            raise Formal4096Error("invalid Epoch1 HPR inventory")
        trigger_by_id[group_id] = trigger
    if set(trigger_by_id) != set(records):
        raise Formal4096Error("Epoch1 explain does not cover Curriculum2048 exactly")
    queues: dict[str, list[str]] = {}
    for domain in DOMAIN_ORDER:
        values = [group_id for group_id, row in records.items() if row["target_domain"] == domain]
        queues[domain] = sorted(values, key=lambda gid: (
            HPR_PRIORITY[trigger_by_id[gid]],
            hashlib.sha256(f"truerec-v2-epoch2|{domain}|{gid}".encode()).hexdigest(),
        ))
    order: list[str] = []
    for block in range(32):
        for domain in DOMAIN_ORDER:
            order.extend(queues[domain][block * 16:(block + 1) * 16])
    if len(order) != EPOCH_STEPS or len(set(order)) != EPOCH_STEPS:
        raise Formal4096Error("Epoch2 order is not an exact replay")
    validate_every64(order, records)
    return order


def load_parent_weights_only(model, optimizer) -> str:
    metadata = json.loads((PARENT_CHECKPOINT / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("training_cursor", {}).get("next_group_index") != PARENT_V1_STEP:
        raise Formal4096Error("parent checkpoint is not V1 step4096")
    if optimizer.state or any(group.get("lr") != 1e-6 or group.get("weight_decay") != 0 for group in optimizer.param_groups):
        raise Formal4096Error("V2 optimizer was not freshly initialized")
    payload = torch.load(PARENT_CHECKPOINT / "state.pt", map_location="cpu", weights_only=False)
    LoraCheckpointState(model).load_state_dict(payload["model_state_dict"], strict=True)
    model_sha = tensor_state_sha256(LoraCheckpointState(model).state_dict())
    if model_sha != metadata.get("model_state_sha256"):
        raise Formal4096Error("parent model SHA exact-load gate failed")
    if optimizer.state:
        raise Formal4096Error("parent optimizer state leaked into V2")
    return model_sha


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
    resolved = run_directory.resolve()
    if ALLOWED_OUTPUT_ROOT.resolve() not in resolved.parents:
        raise Formal4096Error("V2 output must be under /root/GRPO/truerec_grpo/runs")
    if resume_checkpoint is None:
        if run_directory.exists():
            raise Formal4096Error("fresh formal run directory must not exist")
        return
    if not run_directory.is_dir() or not resume_checkpoint.is_dir():
        raise Formal4096Error("resume requires existing run and checkpoint directories")
    if resume_checkpoint.parent != run_directory / "checkpoints":
        raise Formal4096Error("resume checkpoint must belong to this formal run")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _write_epoch2_order(run_directory: Path, order: Sequence[str], explain_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    order_path = run_directory / "epoch2_order.json"
    manifest_path = run_directory / "epoch2_order_manifest.json"
    raw = json.dumps(list(order), indent=2) + "\n"
    order_path.write_text(raw, encoding="utf-8")
    trigger_counts: dict[str, int] = {}
    for row in explain_rows:
        trigger = str(row["hpr_trigger"])
        trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
    manifest = {
        "source": "actual Epoch1 train_explain.jsonl",
        "priority": ["HPR_C", "HPR_B", "HPR_NONE", "HPR_A"],
        "count": EPOCH_STEPS,
        "unique_group_count": EPOCH_STEPS,
        "every64_domain_balance": "PASS",
        "epoch2_order_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "epoch1_trigger_counts": trigger_counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _load_epoch2_order(run_directory: Path, records: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    path = run_directory / "epoch2_order.json"
    manifest_path = run_directory / "epoch2_order_manifest.json"
    order = [str(value) for value in json.loads(path.read_text(encoding="utf-8"))]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["epoch2_order_sha256"]:
        raise Formal4096Error("Epoch2 order SHA mismatch")
    if len(order) != EPOCH_STEPS or len(set(order)) != EPOCH_STEPS or set(order) != set(records):
        raise Formal4096Error("Epoch2 replay identity mismatch")
    validate_every64(order, records)
    return order, manifest


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

        records, epoch1_order, epoch1_info = load_curriculum()
        probe_records = load_probe20()
        model, optimizer, renderer, provenance, dropout, trainability, optimizer_audit = load_runtime(device)
        parent_model_sha = None
        if resume_checkpoint is None:
            parent_model_sha = load_parent_weights_only(model, optimizer)
            distributed_fail_if(
                not all_rank_values_equal(parent_model_sha) or bool(optimizer.state),
                "parent weight-only initialization differs across ranks", device,
            )
        ddp = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False,
        )
        ddp.eval(); model.eval()
        cursor = 0
        epoch2_order: list[str] | None = None
        epoch2_manifest: dict[str, Any] | None = None
        if (run_directory / "epoch2_order.json").exists():
            epoch2_order, epoch2_manifest = _load_epoch2_order(run_directory, records)
        full_order = epoch1_order + (epoch2_order or [])
        dataset_identity = {
            "records_sha256": RECORDS_SHA256,
            "epoch1_order_sha256": EPOCH1_ORDER_SHA256,
            "record_count": TOTAL_STEPS,
            "unique_group_count": EPOCH_STEPS,
        }
        if resume_checkpoint is not None:
            expected_order = order_metadata(
                full_order if epoch2_order is not None else epoch1_order,
                name="v2_two_epoch_order" if epoch2_order is not None else "epoch1_frozen",
            )
            if epoch2_order is None:
                expected_order = epoch1_info
            restored = load_distributed_checkpoint(
                resume_checkpoint, model=LoraCheckpointState(model), optimizer=optimizer,
                current_dataset_identity=dataset_identity, current_group_ids=full_order, device=device,
                allow_custom_dataset=True, expected_order=expected_order,
            )
            cursor = int(restored["next_group_index"])
            if restored["driver_state"] != driver_state_for_cursor(cursor):
                raise Formal4096Error("checkpoint driver state differs from cursor")
            parent_model_sha = json.loads((run_directory / "run_manifest.json").read_text())["parent_model_sha"]
        writer = MonitoringWriterV1(run_directory, total_steps=TOTAL_STEPS, rank=rank)
        _rank0_checked(
            device,
            lambda: validate_monitoring_prefix(_read_train_rows(run_directory), full_order, cursor),
        )

        manifest = {
            "RUN_ID": RUN_ID, "source_commit": SOURCE_COMMIT_MARKER.read_text().strip(),
            "parent_run_id": PARENT_RUN_ID, "parent_checkpoint": str(PARENT_CHECKPOINT),
            "parent_v1_step": PARENT_V1_STEP, "parent_model_sha": parent_model_sha,
            "optimizer_reset": True, "v2_global_step_at_start": 0,
            "records_sha256": RECORDS_SHA256, "epoch1_order_sha256": EPOCH1_ORDER_SHA256,
            "epoch2_order_sha256": epoch2_manifest["epoch2_order_sha256"] if epoch2_manifest else None,
            "world_size": 4, "global_G": 8, "local_G": 2,
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
            if group_index == EPOCH_STEPS and epoch2_order is None:
                if rank == 0:
                    explain_rows = _read_jsonl(run_directory / "train_explain.jsonl")
                    epoch2_order = build_epoch2_order(explain_rows, records)
                    epoch2_manifest = _write_epoch2_order(run_directory, epoch2_order, explain_rows)
                broadcast = [epoch2_order, epoch2_manifest]
                dist.broadcast_object_list(broadcast, src=0)
                epoch2_order, epoch2_manifest = broadcast
                distributed_fail_if(
                    epoch2_order is None or not all_rank_values_equal(epoch2_manifest["epoch2_order_sha256"]),
                    "Epoch2 order rank agreement failed", device,
                )
                full_order = epoch1_order + epoch2_order
            group_id = epoch1_order[group_index] if group_index < EPOCH_STEPS else epoch2_order[group_index - EPOCH_STEPS]
            report, group, backward, rank_payloads = run_loaded_group(
                group_id, records[group_id], model, ddp, optimizer, renderer, device,
                provenance=provenance, dropout=dropout, trainability=trainability,
                optimizer_audit=optimizer_audit, strict_parameter_audit=False,
            )
            completed = group_index + 1
            if completed in checkpoint_steps() and completed != EPOCH_STEPS:
                checkpoint = run_directory / "checkpoints" / f"checkpoint-step-{completed}"
                checkpoint_order = order_metadata(full_order, name="v2_two_epoch_order") if epoch2_order else epoch1_info
                save_distributed_checkpoint(
                    checkpoint, model=LoraCheckpointState(model), optimizer=optimizer,
                    driver_state=driver_state_for_cursor(completed), dataset_identity=dataset_identity,
                    epoch=completed // EPOCH_STEPS, next_group_index=completed,
                    order=checkpoint_order, device=device, allow_custom_dataset=True,
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
            if completed == EPOCH_STEPS:
                if rank == 0:
                    explain_rows = _read_jsonl(run_directory / "train_explain.jsonl")
                    epoch2_order = build_epoch2_order(explain_rows, records)
                    epoch2_manifest = _write_epoch2_order(run_directory, epoch2_order, explain_rows)
                broadcast = [epoch2_order, epoch2_manifest]
                dist.broadcast_object_list(broadcast, src=0)
                epoch2_order, epoch2_manifest = broadcast
                full_order = epoch1_order + epoch2_order
                distributed_fail_if(
                    not all_rank_values_equal(epoch2_manifest["epoch2_order_sha256"]),
                    "Epoch2 order rank agreement failed", device,
                )
                checkpoint = run_directory / "checkpoints" / f"checkpoint-step-{completed}"
                save_distributed_checkpoint(
                    checkpoint, model=LoraCheckpointState(model), optimizer=optimizer,
                    driver_state=driver_state_for_cursor(completed), dataset_identity=dataset_identity,
                    epoch=1, next_group_index=completed,
                    order=order_metadata(full_order, name="v2_two_epoch_order"), device=device,
                    allow_custom_dataset=True,
                )
                last_checkpoint_step = completed
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
