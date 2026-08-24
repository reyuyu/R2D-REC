"""Inference-only Fresh-BATA bridge-position crossover diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

RUNTIME = Path("/data/GRPO")
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]
from boundary_adapt.diagnostics import controlled_generator_decoder_crossover as cross  # noqa: E402
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import DOMAIN, MANIFEST  # noqa: E402
from boundary_adapt.diagnostics.system_prompt_crossover import (  # noqa: E402
    SOURCE,
    SOURCE_SHA256,
    true_bata_system_prompt_ids,
)

SOURCE_COMMIT = "e46cc89d221e342e4fffa50a15a731dfa941bef3"
OUTPUT = Path("/root/GRPO_audit_results/bridge_position_crossover_20260824")
PUBLIC_OUTPUT = RUNTIME / "boundary_adapt/results/bridge_position_crossover_20260824"
PREPARED = OUTPUT / "prepared_manifest.json"
PARTS = OUTPUT / "parts"
ADAPTER = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
EXPECTED_BARE = 2.2721354167
EXPECTED_AFTER = 4.0546875000
MODES = ("bare", "bridge_after", "bridge_before")
DOMAIN_ORDER = ("video", "prod", "ad", "living")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260824


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_names(tokenizer, ids: list[int]) -> list[str]:
    values = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    return [str(values)] if isinstance(values, str) else [str(value) for value in values]


def ensure_output_link() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PUBLIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if PUBLIC_OUTPUT.is_symlink():
        if PUBLIC_OUTPUT.resolve() != OUTPUT.resolve():
            raise RuntimeError(f"PUBLIC_OUTPUT points elsewhere: {PUBLIC_OUTPUT.resolve()}")
    elif PUBLIC_OUTPUT.exists():
        raise RuntimeError(f"PUBLIC_OUTPUT is not a symlink: {PUBLIC_OUTPUT}")
    else:
        PUBLIC_OUTPUT.symlink_to(OUTPUT, target_is_directory=True)


def source_group_id(row: dict[str, Any]) -> str | None:
    metadata = json.loads(row.get("aux_metadata_json") or "{}")
    return metadata.get("recommendation_group_id")


def prepare() -> None:
    ensure_output_link()
    if file_sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("TRAIN_SOURCE_SHA256_MISMATCH")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER / filename).is_file():
            raise RuntimeError(f"MISSING_D0_ADAPTER_FILE={filename}")

    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    groups = {str(item["recommendation_group_id"]) for item in frozen}
    if len(frozen) != 48 or len(groups) != 12:
        raise RuntimeError("FROZEN_PROBE_CARDINALITY_FAIL")
    systems = {group: set() for group in groups}
    bridges = {group: set() for group in groups}
    duplicate_rows = Counter()
    for line in SOURCE.open(encoding="utf-8"):
        row = json.loads(line)
        group = source_group_id(row)
        if group not in groups or row.get("data_source") != "recommend" or row.get("source_segment") != "recommendation_cot":
            continue
        systems[group].add(str(row.get("system", "")))
        target_domain = next(
            item["target_domain"] for item in frozen
            if str(item["recommendation_group_id"]) == group
        )
        output = str(row["output"])
        left = output.index("</think>") + len("</think>")
        right = output.index(DOMAIN[target_domain], left)
        bridges[group].add(output[left:right])
        duplicate_rows[group] += 1
    if any(len(systems[group]) != 1 or "" in systems[group] for group in groups):
        raise RuntimeError("SYSTEM_BYTE_CONSISTENCY_FAIL")
    if any(len(bridges[group]) != 1 for group in groups):
        raise RuntimeError("BRIDGE_BYTE_CONSISTENCY_FAIL")

    tokenizer = AutoTokenizer.from_pretrained(cross.BASE, trust_remote_code=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError(f"CLOSE_TOKEN_NOT_ATOMIC={close_ids}")
    close_id = int(close_ids[0])
    group_prompts: dict[str, set[str]] = defaultdict(set)
    for item in frozen:
        group_prompts[str(item["recommendation_group_id"])].add(str(item["think_prompt"]))
    if any(len(group_prompts[group]) != 1 for group in groups):
        raise RuntimeError("FROZEN_USER_INCONSISTENT_WITHIN_GROUP")

    rendered = {}
    for group in sorted(groups):
        system = next(iter(systems[group]))
        user = next(iter(group_prompts[group]))
        rendered[group] = true_bata_system_prompt_ids(tokenizer, user, system)

    prepared_items = []
    bridge_lengths = Counter()
    for item in frozen:
        group = str(item["recommendation_group_id"])
        cot_ids = list(map(int, tokenizer.encode(item["fixed_cot"], add_special_tokens=False)))
        if not cot_ids or cot_ids[-1] != close_id or cot_ids.count(close_id) != 1:
            raise RuntimeError(f"COT_SINGLE_CLOSE_GATE_FAIL group={group} candidate={item['candidate_id']}")
        bridge = next(iter(bridges[group]))
        if bridge != item["bridge"]:
            raise RuntimeError(f"FROZEN_BRIDGE_SOURCE_MISMATCH group={group}")
        bridge_ids = list(map(int, tokenizer.encode(bridge, add_special_tokens=False)))
        domain_ids = list(map(int, tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)))
        if len(domain_ids) != 1:
            raise RuntimeError(f"DOMAIN_TOKEN_NOT_ATOMIC domain={item['target_domain']}")
        contexts = {
            "bare": rendered[group] + cot_ids + domain_ids,
            "bridge_after": rendered[group] + cot_ids + bridge_ids + domain_ids,
            "bridge_before": rendered[group] + cot_ids[:-1] + bridge_ids + [close_id] + domain_ids,
        }
        if any(context[-1] != domain_ids[0] for context in contexts.values()):
            raise RuntimeError("DOMAIN_NOT_FINAL_CONTEXT_TOKEN")
        bridge_lengths[len(bridge_ids)] += 1
        prepared_items.append({
            **item,
            "system": next(iter(systems[group])),
            "system_prompt_ids": rendered[group],
            "fixed_cot_token_ids": cot_ids,
            "fixed_cot_token_ids_sha256": hashlib.sha256(json.dumps(cot_ids).encode()).hexdigest(),
            "bridge": bridge,
            "bridge_token_ids": bridge_ids,
            "domain_token_ids": domain_ids,
        })

    boundary_examples = []
    for domain in DOMAIN_ORDER:
        item = next(row for row in prepared_items if row["target_domain"] == domain)
        contexts = {
            "bare": item["system_prompt_ids"] + item["fixed_cot_token_ids"] + item["domain_token_ids"],
            "bridge_after": item["system_prompt_ids"] + item["fixed_cot_token_ids"] + item["bridge_token_ids"] + item["domain_token_ids"],
            "bridge_before": item["system_prompt_ids"] + item["fixed_cot_token_ids"][:-1] + item["bridge_token_ids"] + [close_id] + item["domain_token_ids"],
        }
        boundary_examples.append({
            "domain": domain,
            "group_id": item["recommendation_group_id"],
            "system_repr": repr(item["system"]),
            "user_end_repr": repr(item["think_prompt"][-300:]),
            "cot_last_200_chars_repr": repr(item["fixed_cot"][-200:]),
            "bridge_repr": repr(item["bridge"]),
            "cot_last_32_token_names": token_names(tokenizer, item["fixed_cot_token_ids"][-32:]),
            "bridge_token_ids": item["bridge_token_ids"],
            "bridge_token_names": token_names(tokenizer, item["bridge_token_ids"]),
            **{f"{mode}_last_48_context_token_names": token_names(tokenizer, context[-48:]) for mode, context in contexts.items()},
        })

    payload = {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "fixed_cot_count": 48,
        "probe_group_count": 12,
        "system_found": "12/12",
        "system_byte_consistency": True,
        "bridge_found": "12/12",
        "bridge_byte_consistency": True,
        "bridge_variant_count": len({next(iter(bridges[group])) for group in groups}),
        "bridge_token_length_distribution": dict(sorted(bridge_lengths.items())),
        "all_cot_end_with_single_close_token": True,
        "close_token_id": close_id,
        "domain_token_atomic": True,
        "boundary_examples": boundary_examples,
        "bridge_inventory": [{
            "group_id": group,
            "bridge": next(iter(bridges[group])),
            "source_duplicate_rows": duplicate_rows[group],
        } for group in sorted(groups)],
        "checkpoint": {
            "path": str(ADAPTER),
            "adapter_model_sha256": file_sha(ADAPTER / "adapter_model.safetensors"),
            "adapter_config_sha256": file_sha(ADAPTER / "adapter_config.json"),
        },
        "items": prepared_items,
        "training_started": False,
        "optimizer_steps": 0,
    }
    PREPARED.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PREPARED_MANIFEST={PREPARED}")
    print("SYSTEM_FOUND=12/12")
    print("BRIDGE_FOUND=12/12")
    print("ALL_COT_END_WITH_SINGLE_CLOSE_TOKEN=PASS")
    print("DOMAIN_TOKEN_ATOMIC=PASS")


def teacher_with_stages(model, tokenizer, context: list[int], gold_sids: list[str]) -> dict[str, Any]:
    result = cross.teacher_forced_gold(model, tokenizer, context, gold_sids)
    result["mean_nll_a"] = statistics.fmean(-row["A_logp"] for row in result["paths"])
    result["mean_nll_b"] = statistics.fmean(-row["B_logp"] for row in result["paths"])
    result["mean_nll_c"] = statistics.fmean(-row["C_logp"] for row in result["paths"])
    return result


def transition_nll(model, context: list[int], target: list[int]) -> list[float]:
    input_ids = torch.tensor([context + target[:-1]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=len(target)).logits[0].float()
        if logits.shape[0] != len(target):
            raise RuntimeError(f"TRANSITION_LOGIT_SHAPE_FAIL={tuple(logits.shape)} target={len(target)}")
        targets = torch.tensor(target, dtype=torch.long, device=logits.device)
        values = -torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return [float(value) for value in values.cpu().tolist()]


def run() -> None:
    if not PREPARED.is_file():
        raise RuntimeError("MISSING_PREPARED_MANIFEST")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("BRIDGE_POSITION_DIAGNOSTIC_REQUIRES_4_GPUS")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    model, tokenizer = cross.load_model(ADAPTER, f"cuda:{rank}")
    payload = json.loads(PREPARED.read_text(encoding="utf-8"))
    close_id = int(payload["close_token_id"])
    local = []
    for index, item in enumerate(payload["items"]):
        if index % world != rank:
            continue
        prompt = list(map(int, item["system_prompt_ids"]))
        cot = list(map(int, item["fixed_cot_token_ids"]))
        bridge = list(map(int, item["bridge_token_ids"]))
        domain_ids = list(map(int, item["domain_token_ids"]))
        contexts = {
            "bare": prompt + cot + domain_ids,
            "bridge_after": prompt + cot + bridge + domain_ids,
            "bridge_before": prompt + cot[:-1] + bridge + [close_id] + domain_ids,
        }
        history = cross.history_from_prompt(tokenizer, prompt)
        gold_set = {cross.parse_gold(value) for value in item["gold_sids"]}
        base_record = {
            "group_id": item["recommendation_group_id"],
            "domain": item["target_domain"],
            "candidate_id": item["candidate_id"],
            "probe_round": item["probe_round"],
            "fixed_cot_token_ids_sha256": item["fixed_cot_token_ids_sha256"],
            "gold_sids": item["gold_sids"],
            "history_sids": [list(value) for value in sorted(history["unique_set"])],
        }
        for mode, context in contexts.items():
            if context[-1] != domain_ids[0]:
                raise RuntimeError("RUNTIME_DOMAIN_NOT_FINAL")
            local.append({
                **base_record,
                "mode": mode,
                "context_token_count": len(context),
                "beam": cross.strict_beam(model, tokenizer, context, item["target_domain"], gold_set, history),
                "teacher_forced": teacher_with_stages(model, tokenizer, context, item["gold_sids"]),
            })

        body_context = prompt + cot[:-1]
        original_target = [close_id] + bridge + domain_ids
        proposed_target = bridge + [close_id] + domain_ids
        original = transition_nll(model, body_context, original_target)
        proposed = transition_nll(model, body_context, proposed_target)
        local[-1]["transition"] = {
            "nll_close_original": original[0],
            "nll_bridge_after": statistics.fmean(original[1:1 + len(bridge)]),
            "nll_domain_after": original[-1],
            "original_transition_nll": statistics.fmean(original),
            "nll_bridge_before": statistics.fmean(proposed[:len(bridge)]),
            "nll_close_after_bridge": proposed[len(bridge)],
            "nll_domain_before_mode": proposed[-1],
            "proposed_transition_nll": statistics.fmean(proposed),
        }
        print(f"BRIDGE_POSITION_PROGRESS rank={rank} sample={index // world + 1}/12", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"rank{rank}.json").write_text(json.dumps({
        "rank": rank,
        "records": local,
        "training_started": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_beam(records: list[dict[str, Any]]) -> dict[str, Any]:
    beams = [beam for record in records for beam in record["beam"]["beams"]]
    valid = [beam for beam in beams if beam["predicted_sid"] is not None]
    copies = Counter(beam["copy_class"] for beam in valid)
    gold_history = Counter(beam["gold_history_class"] for beam in beams)
    return {
        "beam_raw": statistics.fmean(record["beam"]["beam_raw"] for record in records),
        "exact": sum(record["beam"]["exact"] for record in records),
        "ab": sum(record["beam"]["ab"] for record in records),
        "a": sum(record["beam"]["a"] for record in records),
        "syntax_invalid": sum(record["beam"]["invalid"] for record in records),
        "total_beams": len(beams),
        "history_exact_copy": rate(copies["EXACT_COPY"], len(valid)),
        "history_ab_copy": rate(copies["AB_COPY"], len(valid)),
        "history_a_copy": rate(copies["A_COPY"], len(valid)),
        "novel": rate(copies["NOVEL"], len(valid)),
        "history_not_gold": rate(gold_history["HISTORY_NOT_GOLD"], len(beams)),
        "gold_and_history": rate(gold_history["GOLD_AND_HISTORY"], len(beams)),
        "gold_not_history": rate(gold_history["GOLD_NOT_HISTORY"], len(beams)),
    }


def aggregate_teacher(records: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record["teacher_forced"])
    if not grouped or any(len(values) != 4 for values in grouped.values()):
        raise RuntimeError("TEACHER_GROUP_AVERAGE_CONTRACT_FAIL")
    keys = ("mean_nll_a", "mean_nll_b", "mean_nll_c", "mean_gold_abc_nll")
    return {
        key: statistics.fmean(statistics.fmean(value[key] for value in values) for values in grouped.values())
        for key in keys
    }


def group_values(records: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(float(record["beam"]["beam_raw"]))
    if len(grouped) != 12 or any(len(values) != 4 for values in grouped.values()):
        raise RuntimeError("BOOTSTRAP_GROUP_PAIRING_FAIL")
    return {group: statistics.fmean(values) for group, values in grouped.items()}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def bootstrap(values: dict[str, dict[str, float]]) -> dict[str, Any]:
    groups = sorted(values["bare"])
    rng = random.Random(BOOTSTRAP_SEED)
    samples = {"after_minus_bare": [], "before_minus_bare": [], "before_minus_after": [], "position_recovery": []}
    invalid_ratio = 0
    for _ in range(BOOTSTRAP_RESAMPLES):
        chosen = [rng.choice(groups) for _ in groups]
        means = {mode: statistics.fmean(values[mode][group] for group in chosen) for mode in MODES}
        after_gain = means["bridge_after"] - means["bare"]
        before_gain = means["bridge_before"] - means["bare"]
        samples["after_minus_bare"].append(after_gain)
        samples["before_minus_bare"].append(before_gain)
        samples["before_minus_after"].append(means["bridge_before"] - means["bridge_after"])
        if abs(after_gain) <= 1e-12:
            invalid_ratio += 1
        else:
            samples["position_recovery"].append(before_gain / after_gain)
    return {
        key: {"ci95": [percentile(sample, 0.025), percentile(sample, 0.975)], "resamples": len(sample)}
        for key, sample in samples.items()
    } | {"position_recovery_invalid_denominator_resamples": invalid_ratio, "unit": "probe_group_with_4_candidates_paired"}


def top32_copy_summary(record: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(beam["copy_class"] for beam in record["beam"]["beams"]))


def transition_aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    rows = [record["transition"] for record in records if "transition" in record]
    if len(rows) != 48:
        raise RuntimeError(f"TRANSITION_RECORD_COUNT_FAIL={len(rows)}")
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    identities = [record for record in records if "transition" in record]
    for record, row in zip(identities, rows):
        grouped[record["group_id"]].append(row)
    keys = tuple(rows[0])
    return {
        key: statistics.fmean(statistics.fmean(row[key] for row in group_rows) for group_rows in grouped.values())
        for key in keys
    }


def classify_root(recovery: float, before_gain: float, ci: dict[str, Any], per_domain: dict[str, Any]) -> tuple[str, str]:
    domain_gains = [per_domain[domain]["bridge_before"]["beam_raw"] - per_domain[domain]["bare"]["beam_raw"] for domain in DOMAIN_ORDER]
    severe_heterogeneity = min(domain_gains) < 0 < max(domain_gains) and max(domain_gains) - min(domain_gains) > abs(before_gain) * 3
    recovery_ci = ci["position_recovery"]["ci95"]
    if severe_heterogeneity or recovery_ci[0] < 0.3 < recovery_ci[1] and recovery_ci[1] - recovery_ci[0] > 1.0:
        return "INCONCLUSIVE", "Group-level uncertainty or domain heterogeneity prevents a stable bridge-position mechanism conclusion."
    if recovery >= 0.70 and ci["before_minus_bare"]["ci95"][0] > 0:
        return "BRIDGE_BEFORE_CLOSE_STRONGLY_VIABLE", "Bridge-before-close preserves at least 70% of the original bridge advantage with a positive paired group trend."
    if 0.30 <= recovery < 0.70 and before_gain > 0:
        return "BRIDGE_BEFORE_CLOSE_PARTIALLY_VIABLE", "Bridge-before-close preserves a positive but incomplete portion of the original bridge advantage."
    return "BRIDGE_POSITION_DEPENDENT", "Bridge-before-close preserves less than 30% of the original bridge advantage or does not improve over bare context."


def finalize() -> None:
    ensure_output_link()
    prepared = json.loads(PREPARED.read_text(encoding="utf-8"))
    parts = [json.loads((PARTS / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(4)]
    if any(part["training_started"] or part["optimizer_created"] or part["optimizer_steps"] for part in parts):
        raise RuntimeError("NO_TRAINING_GATE_FAIL")
    records = [record for part in parts for record in part["records"]]
    if len(records) != 144:
        raise RuntimeError(f"RECORD_COUNT_FAIL={len(records)}")
    cells = {}
    for mode in MODES:
        selected = [record for record in records if record["mode"] == mode]
        if len(selected) != 48:
            raise RuntimeError(f"MODE_COUNT_FAIL mode={mode}")
        cells[mode] = {**aggregate_beam(selected), **aggregate_teacher(selected)}
    bare_pass = abs(cells["bare"]["beam_raw"] - EXPECTED_BARE) <= 1e-8
    after_pass = abs(cells["bridge_after"]["beam_raw"] - EXPECTED_AFTER) <= 1e-8
    if not bare_pass or not after_pass:
        raise RuntimeError(f"SYSTEM_REPRODUCTION_FAIL bare={cells['bare']['beam_raw']} after={cells['bridge_after']['beam_raw']}")

    per_domain = {
        domain: {
            mode: {
                **aggregate_beam([record for record in records if record["mode"] == mode and record["domain"] == domain]),
                **aggregate_teacher([record for record in records if record["mode"] == mode and record["domain"] == domain]),
            } for mode in MODES
        } for domain in DOMAIN_ORDER
    }
    values = {mode: group_values([record for record in records if record["mode"] == mode]) for mode in MODES}
    confidence = bootstrap(values)
    after_gain = cells["bridge_after"]["beam_raw"] - cells["bare"]["beam_raw"]
    if after_gain <= 0:
        raise RuntimeError("BRIDGE_ADVANTAGE_DENOMINATOR_NONPOSITIVE")
    before_gain = cells["bridge_before"]["beam_raw"] - cells["bare"]["beam_raw"]
    recovery = before_gain / after_gain

    indexed = {
        mode: {(record["group_id"], record["candidate_id"]): record for record in records if record["mode"] == mode}
        for mode in MODES
    }
    sample_diffs = []
    for identity, bare in indexed["bare"].items():
        after, before = indexed["bridge_after"][identity], indexed["bridge_before"][identity]
        sample_diffs.append({
            "group_id": identity[0], "domain": bare["domain"], "candidate_id": identity[1],
            "bare_raw": bare["beam"]["beam_raw"], "after_raw": after["beam"]["beam_raw"], "before_raw": before["beam"]["beam_raw"],
            "before_minus_bare": before["beam"]["beam_raw"] - bare["beam"]["beam_raw"],
            "before_minus_after": before["beam"]["beam_raw"] - after["beam"]["beam_raw"],
            "bare_top1": bare["beam"]["beams"][0]["predicted_sid"],
            "after_top1": after["beam"]["beams"][0]["predicted_sid"],
            "before_top1": before["beam"]["beams"][0]["predicted_sid"],
            "gold_sids": bare["gold_sids"], "history_sids": bare["history_sids"],
            "bare_top32_copy_summary": top32_copy_summary(bare),
            "after_top32_copy_summary": top32_copy_summary(after),
            "before_top32_copy_summary": top32_copy_summary(before),
        })
    helps = sorted(sample_diffs, key=lambda row: (-row["before_minus_bare"], row["group_id"], row["candidate_id"]))[:10]
    hurts = sorted(sample_diffs, key=lambda row: (row["before_minus_after"], row["group_id"], row["candidate_id"]))[:10]
    closest = sorted(sample_diffs, key=lambda row: (abs(row["before_minus_after"]), row["group_id"], row["candidate_id"]))[:10]
    transition = transition_aggregate(records)
    transition["bridge_relocation_nll_penalty"] = transition["proposed_transition_nll"] - transition["original_transition_nll"]
    root_class, root_conclusion = classify_root(recovery, before_gain, confidence, per_domain)

    summary = {
        "type": "fresh_bata_bridge_position_crossover",
        "source_commit": SOURCE_COMMIT,
        "prepared_contract": {key: prepared[key] for key in (
            "fixed_cot_count", "probe_group_count", "system_found", "system_byte_consistency",
            "bridge_found", "bridge_byte_consistency", "bridge_variant_count", "bridge_token_length_distribution",
            "all_cot_end_with_single_close_token", "domain_token_atomic", "bridge_inventory",
            "boundary_examples", "checkpoint",
        )},
        "system_bare_reproduction_pass": bare_pass,
        "system_bridge_after_reproduction_pass": after_pass,
        "cells": cells,
        "per_domain": per_domain,
        "effects": {
            "bridge_after_minus_bare": after_gain,
            "bridge_before_minus_bare": before_gain,
            "bridge_before_minus_after": cells["bridge_before"]["beam_raw"] - cells["bridge_after"]["beam_raw"],
            "position_recovery": recovery,
        },
        "bootstrap": confidence,
        "transition_nll": transition,
        "sample_diffs": sample_diffs,
        "top_before_helps_over_bare": helps,
        "top_before_hurts_vs_after": hurts,
        "top_before_closest_to_after": closest,
        "mapping_validity": {"available": False, "reason": "authoritative evaluator mapping files were not available to this diagnostic"},
        "root_class": root_class,
        "root_conclusion": root_conclusion,
        "interpretation_scope": "mechanism test on 12 training-source probe groups; not held-out or external evidence",
        "training_started": False,
        "gpu_inference_started": True,
        "optimizer_steps": 0,
    }
    result = OUTPUT / "bridge_position_summary.json"
    result.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "bridge_position_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def table(metric: str) -> str:
        lines = [f"| Mode | Overall {metric} | video | prod | ad | living |", "|---|---:|---:|---:|---:|---:|"]
        for mode in MODES:
            lines.append("| {} | {:.10f} | {} |".format(
                mode, cells[mode][metric], " | ".join(f"{per_domain[d][mode][metric]:.10f}" for d in DOMAIN_ORDER)
            ))
        return "\n".join(lines)
    matrix = "\n\n".join([
        "# Fresh BATA Bridge Position Crossover",
        "## BeamRaw\n\n" + table("beam_raw"),
        "## Gold NLL\n\n" + table("mean_gold_abc_nll"),
        "## Effects and group bootstrap\n\n```json\n" + json.dumps({"effects": summary["effects"], "bootstrap": confidence}, indent=2) + "\n```",
        "## Transition NLL\n\n```json\n" + json.dumps(transition, indent=2) + "\n```",
        f"## Root class\n\n`{root_class}`\n\n{root_conclusion}",
    ])
    (OUTPUT / "bridge_position_matrix.md").write_text(matrix + "\n", encoding="utf-8")

    def diff_section(title: str, rows: list[dict[str, Any]]) -> str:
        chunks = [f"## {title}"]
        for index, row in enumerate(rows, 1):
            chunks += [f"### {index}. {row['domain']} {row['group_id']} candidate={row['candidate_id']}", "```json", json.dumps(row, ensure_ascii=False, indent=2), "```"]
        return "\n\n".join(chunks)
    diff_text = "# Bridge Position Sample Diffs\n\n" + "\n\n".join([
        diff_section("TOP 10 BEFORE_HELPS_OVER_BARE", helps),
        diff_section("TOP 10 BEFORE_HURTS_VS_AFTER", hurts),
        diff_section("TOP 10 BEFORE_CLOSEST_TO_AFTER", closest),
    ])
    (OUTPUT / "bridge_position_sample_diffs.md").write_text(diff_text + "\n", encoding="utf-8")

    review_sections = [
        ("EXACT BRIDGE INVENTORY", prepared["bridge_inventory"]),
        ("BRIDGE TOKEN LENGTH", prepared["bridge_token_length_distribution"]),
        ("4 DOMAIN CONTEXT BOUNDARIES", prepared["boundary_examples"]),
        ("REPRODUCTION", {"bare": cells["bare"]["beam_raw"], "after": cells["bridge_after"]["beam_raw"], "bare_pass": bare_pass, "after_pass": after_pass}),
        ("BARE AFTER BEFORE MATRIX", {"cells": cells, "per_domain": per_domain}),
        ("RECOVERY AND BOOTSTRAP", {"effects": summary["effects"], "bootstrap": confidence}),
        ("TRANSITION NLL", transition),
        ("MAPPING VALIDITY", summary["mapping_validity"]),
        ("TOP BEFORE HELPS", helps), ("TOP BEFORE HURTS", hurts), ("TOP BEFORE CLOSEST", closest),
    ]
    review = ["FRESH BATA BRIDGE POSITION CROSSOVER REVIEW", "", "SCOPE=MECHANISM TEST ONLY; ALL 12 GROUPS ARE FROM TRAINING SOURCE"]
    for title, value in review_sections:
        review += ["", f"[{title}]", json.dumps(value, ensure_ascii=False, indent=2)]
    review += ["", f"ROOT_CLASS={root_class}", f"ROOT_CONCLUSION={root_conclusion}"]
    (OUTPUT / "CHATGPT_BRIDGE_POSITION_REVIEW.txt").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(f"RESULT_JSON={result}")
    print(f"ROOT_CLASS={root_class}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.run, args.finalize)) != 1:
        raise SystemExit("choose exactly one operation")
    if args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        finalize()


if __name__ == "__main__":
    main()
