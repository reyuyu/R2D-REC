"""Inference-only Boundary Adapt system-prompt crossover diagnostic."""
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
from grpo_model import BASE, encode_prompt  # noqa: E402

SOURCE_COMMIT = "6bbfb3029968781412387b6824ce8318ac9f45e9"
SOURCE = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
SOURCE_SHA256 = "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca"
OUTPUT = Path("/root/GRPO_audit_results/system_prompt_crossover_20260824")
PUBLIC_OUTPUT = RUNTIME / "boundary_adapt/results/system_prompt_crossover_20260824"
PREPARED = OUTPUT / "prepared_manifest.json"
PARTS = OUTPUT / "parts"
DOMAIN_ORDER = ("video", "prod", "ad", "living")
DECODERS = ("0", "300", "900")
MODES = (("S0", "bare"), ("S0", "bridge"), ("S1", "bare"), ("S1", "bridge"))
CHECKPOINTS = {
    "0": Path(
        "/data/outputs/baselines/native_source_domain_r32_v3/"
        "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
    ),
    "300": Path(
        "/root/data_checkpoints_backup_20260824/outputs/boundary_adapt/"
        "formal_300step/checkpoint-300"
    ),
    "900": Path(
        "/root/data_checkpoints_backup_20260824/outputs/boundary_adapt/"
        "continuation_from300_to1500/checkpoint-900"
    ),
}
OLD_MERGED = {
    label: RUNTIME / f"boundary_adapt/results/fixed_cot_sweep_20260824_parts/checkpoint_{label}_merged.json"
    for label in DECODERS
}
EXPECTED_S0_BARE = {"0": 2.5065104167, "300": 2.7447916667, "900": 3.5338541667}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260824


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_names(tokenizer, ids: list[int]) -> list[str]:
    value = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    return [str(value)] if isinstance(value, str) else [str(item) for item in value]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def ensure_output_link() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PUBLIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if PUBLIC_OUTPUT.is_symlink():
        if PUBLIC_OUTPUT.resolve() != OUTPUT.resolve():
            raise RuntimeError(f"PUBLIC_OUTPUT points elsewhere: {PUBLIC_OUTPUT.resolve()}")
    elif PUBLIC_OUTPUT.exists():
        raise RuntimeError(f"PUBLIC_OUTPUT already exists and is not a symlink: {PUBLIC_OUTPUT}")
    else:
        PUBLIC_OUTPUT.symlink_to(OUTPUT, target_is_directory=True)


def source_group_id(row: dict[str, Any]) -> str | None:
    raw = row.get("aux_metadata_json")
    if not raw:
        return None
    return json.loads(raw).get("recommendation_group_id")


def true_bata_system_prompt_ids(tokenizer, user_content: str, system: str) -> list[int]:
    from llamafactory.data.template import TEMPLATES

    template = TEMPLATES["qwen3_nothink"]
    prompt_ids, _ = template.encode_oneturn(
        tokenizer,
        [{"role": "user", "content": user_content}, {"role": "assistant", "content": ""}],
        system=system,
    )
    return list(map(int, prompt_ids))


def prepare() -> None:
    ensure_output_link()
    if file_sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("TRAIN_SOURCE_SHA256_MISMATCH")
    for label, checkpoint in CHECKPOINTS.items():
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (checkpoint / filename).is_file():
                raise RuntimeError(f"MISSING_CHECKPOINT_FILE label={label} file={filename}")

    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    if len(frozen) != 48:
        raise RuntimeError("FIXED_COT_COUNT_NOT_48")
    groups = {str(item["recommendation_group_id"]) for item in frozen}
    if len(groups) != 12:
        raise RuntimeError("PROBE_GROUP_COUNT_NOT_12")
    systems: dict[str, set[str]] = {group: set() for group in groups}
    prompts: dict[str, set[str]] = {group: set() for group in groups}
    duplicate_rows = Counter()
    all_systems = Counter()
    unique_source_groups: set[str] = set()
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("data_source") != "recommend" or row.get("source_segment") != "recommendation_cot":
                continue
            group = source_group_id(row)
            if not group:
                continue
            if group not in unique_source_groups:
                all_systems[str(row.get("system", ""))] += 1
                unique_source_groups.add(group)
            if group in groups:
                systems[group].add(str(row.get("system", "")))
                prompts[group].add(str(row.get("instruction", "")) + str(row.get("input", "")))
                duplicate_rows[group] += 1
    if len(all_systems) != 6 or any(not value for value in all_systems):
        raise RuntimeError(f"GLOBAL_SYSTEM_INVENTORY_INVALID variants={len(all_systems)}")
    if any(len(systems[group]) != 1 or "" in systems[group] for group in groups):
        raise RuntimeError("PROBE_SYSTEM_BYTE_CONSISTENCY_FAIL")
    if any(len(prompts[group]) != 1 for group in groups):
        raise RuntimeError("PROBE_PROMPT_BYTE_CONSISTENCY_FAIL")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    prepared_items = []
    render_examples = []
    rendered_groups: dict[str, dict[str, Any]] = {}
    probe_prompts: dict[str, set[str]] = {group: set() for group in groups}
    for item in frozen:
        probe_prompts[str(item["recommendation_group_id"])].add(str(item["think_prompt"]))
    if any(len(probe_prompts[group]) != 1 for group in groups):
        raise RuntimeError("FROZEN_PROBE_USER_CONTENT_INCONSISTENT_WITHIN_GROUP")
    prompt_source_comparison = []
    for group in sorted(groups):
        system = next(iter(systems[group]))
        source_user_content = next(iter(prompts[group]))
        user_content = next(iter(probe_prompts[group]))
        prompt_source_comparison.append({
            "group_id": group,
            "probe_user_matches_source_user": user_content == source_user_content,
            "probe_user_sha256": hashlib.sha256(user_content.encode()).hexdigest(),
            "source_user_sha256": hashlib.sha256(source_user_content.encode()).hexdigest(),
            "probe_user_length": len(user_content),
            "source_user_length": len(source_user_content),
        })
        no_ids = encode_prompt(tokenizer, user_content)
        system_ids = true_bata_system_prompt_ids(tokenizer, user_content, system)
        if len(system_ids) <= len(no_ids) or system_ids[-len(no_ids):] != no_ids:
            raise RuntimeError(f"USER_CONTENT_TOKEN_PARITY_FAIL group={group}")
        no_text = tokenizer.decode(no_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        system_text = tokenizer.decode(system_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        checks = {
            "no_system_role": no_text.count("<|im_start|>system") == 0,
            "no_user_role": no_text.count("<|im_start|>user") == 1,
            "no_assistant_head": no_text.count("<|im_start|>assistant") == 1,
            "system_role": system_text.count("<|im_start|>system") == 1,
            "system_user_role": system_text.count("<|im_start|>user") == 1,
            "system_assistant_head": system_text.count("<|im_start|>assistant") == 1,
            "think_suffix_parity": system_ids[-len(no_ids):] == no_ids,
        }
        if not all(checks.values()):
            raise RuntimeError(f"RENDER_CONTRACT_FAIL group={group} checks={checks}")
        rendered_groups[group] = {
            "system": system,
            "user_content": user_content,
            "source_user_content": source_user_content,
            "no_system_prompt_ids": no_ids,
            "system_prompt_ids": system_ids,
            "checks": checks,
        }
    for domain in DOMAIN_ORDER:
        group = next(
            str(item["recommendation_group_id"])
            for item in frozen if item["target_domain"] == domain
        )
        item = rendered_groups[group]
        no_ids, system_ids = item["no_system_prompt_ids"], item["system_prompt_ids"]
        render_examples.append({
            "domain": domain,
            "group_id": group,
            "system_text_repr": repr(item["system"]),
            "user_content_repr": repr(item["user_content"]),
            "no_system_render_repr": repr(tokenizer.decode(no_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)),
            "system_render_repr": repr(tokenizer.decode(system_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)),
            "no_system_token_ids": no_ids,
            "system_token_ids": system_ids,
            "no_system_first_32_token_names": token_names(tokenizer, no_ids[:32]),
            "system_first_32_token_names": token_names(tokenizer, system_ids[:32]),
            "no_system_last_32_prompt_token_names": token_names(tokenizer, no_ids[-32:]),
            "system_last_32_prompt_token_names": token_names(tokenizer, system_ids[-32:]),
        })

    for item in frozen:
        group = str(item["recommendation_group_id"])
        if item["think_prompt"] != rendered_groups[group]["user_content"]:
            raise RuntimeError(f"FROZEN_PROMPT_RENDER_MISMATCH group={group}")
        cot_ids = tokenizer.encode(item["fixed_cot"], add_special_tokens=False)
        prepared_items.append({
            **item,
            "system": rendered_groups[group]["system"],
            "no_system_prompt_ids": rendered_groups[group]["no_system_prompt_ids"],
            "system_prompt_ids": rendered_groups[group]["system_prompt_ids"],
            "fixed_cot_token_ids": cot_ids,
            "fixed_cot_token_ids_sha256": hashlib.sha256(json.dumps(cot_ids).encode()).hexdigest(),
        })

    old_reproduction = {}
    for label, path in OLD_MERGED.items():
        value = float(json.loads(path.read_text(encoding="utf-8"))["modes"]["bare"]["beam_raw_mean"])
        old_reproduction[label] = value
        if abs(value - EXPECTED_S0_BARE[label]) > 1e-8:
            raise RuntimeError(f"OLD_NO_SYSTEM_REPRO_FAIL label={label} value={value}")
    payload = {
        "source_commit": SOURCE_COMMIT,
        "source": str(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "fixed_cot_count": len(prepared_items),
        "probe_group_count": len(groups),
        "probe_system_found": f"{sum(bool(systems[group]) for group in groups)}/12",
        "probe_system_variants": len({next(iter(systems[group])) for group in groups}),
        "global_system_inventory": [{"system": key, "group_count": value} for key, value in all_systems.most_common()],
        "probe_systems": [{
            "group_id": group,
            "system": next(iter(systems[group])),
            "source_duplicate_rows": duplicate_rows[group],
        } for group in sorted(groups)],
        "system_byte_consistency": True,
        "user_content_token_parity": True,
        "system_only_crossover": True,
        "probe_vs_source_user_content": prompt_source_comparison,
        "render_examples": render_examples,
        "old_no_system_reproduction": old_reproduction,
        "checkpoint_provenance": {
            label: {
                "path": str(path),
                "adapter_config_sha256": file_sha(path / "adapter_config.json"),
                "adapter_model_sha256": file_sha(path / "adapter_model.safetensors"),
            }
            for label, path in CHECKPOINTS.items()
        },
        "items": prepared_items,
        "training_started": False,
        "optimizer_steps": 0,
    }
    PREPARED.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PREPARED_MANIFEST={PREPARED}")
    print("PROBE_SYSTEM_FOUND=12/12")
    print("PROBE_SYSTEM_BYTE_CONSISTENCY=PASS")
    print("USER_CONTENT_TOKEN_PARITY=PASS")
    print("OLD_NO_SYSTEM_REPRODUCTION=PASS")


def teacher_forced_with_stages(model, tokenizer, context: list[int], gold_sids: list[str]) -> dict[str, Any]:
    result = cross.teacher_forced_gold(model, tokenizer, context, gold_sids)
    paths = result["paths"]
    result["mean_nll_a"] = statistics.fmean(-row["A_logp"] for row in paths)
    result["mean_nll_b"] = statistics.fmean(-row["B_logp"] for row in paths)
    result["mean_nll_c"] = statistics.fmean(-row["C_logp"] for row in paths)
    return result


def run_decoder(label: str) -> None:
    if label not in DECODERS or not PREPARED.is_file():
        raise RuntimeError("UNKNOWN_DECODER_OR_MISSING_PREPARE")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("SYSTEM_CROSSOVER_REQUIRES_4_GPUS")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    model, tokenizer = cross.load_model(CHECKPOINTS[label], f"cuda:{rank}")
    payload = json.loads(PREPARED.read_text(encoding="utf-8"))
    local = []
    for index, item in enumerate(payload["items"]):
        if index % world != rank:
            continue
        cot_ids = list(map(int, item["fixed_cot_token_ids"]))
        bridge_ids = tokenizer.encode(item["bridge"], add_special_tokens=False)
        domain_ids = tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)
        if len(domain_ids) != 1:
            raise RuntimeError("DOMAIN_TOKEN_NOT_ATOMIC")
        history = cross.history_from_prompt(tokenizer, item["no_system_prompt_ids"])
        gold_set = {cross.parse_gold(value) for value in item["gold_sids"]}
        for system_mode, boundary_mode in MODES:
            prompt_ids = item["no_system_prompt_ids"] if system_mode == "S0" else item["system_prompt_ids"]
            context = list(map(int, prompt_ids)) + cot_ids
            if boundary_mode == "bridge":
                context += bridge_ids
            context += domain_ids
            beam = cross.strict_beam(model, tokenizer, context, item["target_domain"], gold_set, history)
            teacher = teacher_forced_with_stages(model, tokenizer, context, item["gold_sids"])
            local.append({
                "decoder": label,
                "system_mode": system_mode,
                "boundary_mode": boundary_mode,
                "group_id": item["recommendation_group_id"],
                "domain": item["target_domain"],
                "candidate_id": item["candidate_id"],
                "probe_round": item["probe_round"],
                "fixed_cot_token_ids_sha256": item["fixed_cot_token_ids_sha256"],
                "gold_sids": item["gold_sids"],
                "context_token_count": len(context),
                "beam": beam,
                "teacher_forced": teacher,
            })
        print(f"SYSTEM_CROSSOVER_PROGRESS decoder={label} rank={rank} sample={index // world + 1}/12", flush=True)
    target = PARTS / f"decoder_{label}"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"rank{rank}.json").write_text(json.dumps({
        "decoder": label,
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
    cross_classes = Counter(beam["gold_history_class"] for beam in beams)
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
        "history_not_gold": rate(cross_classes["HISTORY_NOT_GOLD"], len(beams)),
        "gold_not_history": rate(cross_classes["GOLD_NOT_HISTORY"], len(beams)),
    }


def aggregate_teacher(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("mean_nll_a", "mean_nll_b", "mean_nll_c", "mean_gold_abc_nll")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record["teacher_forced"])
    expected_groups = len(records) // 4
    if len(grouped) != expected_groups or any(len(values) != 4 for values in grouped.values()):
        raise RuntimeError("TEACHER_GROUP_AVERAGE_CONTRACT_FAIL")
    return {
        key: statistics.fmean(statistics.fmean(item[key] for item in values) for values in grouped.values())
        for key in keys
    }


def merge_decoder(label: str) -> None:
    parts = [json.loads((PARTS / f"decoder_{label}/rank{rank}.json").read_text(encoding="utf-8")) for rank in range(4)]
    if any(part["optimizer_steps"] != 0 or part["optimizer_created"] for part in parts):
        raise RuntimeError("NO_TRAINING_GATE_FAIL")
    records = [record for part in parts for record in part["records"]]
    if len(records) != 192:
        raise RuntimeError(f"DECODER_RECORD_COUNT_INVALID={len(records)}")
    cells = {}
    for system_mode, boundary_mode in MODES:
        selected = [
            record for record in records
            if record["system_mode"] == system_mode and record["boundary_mode"] == boundary_mode
        ]
        if len(selected) != 48:
            raise RuntimeError("CELL_RECORD_COUNT_INVALID")
        cells[f"{system_mode}_{boundary_mode}"] = {
            "overall": {**aggregate_beam(selected), **aggregate_teacher(selected)},
            "per_domain": {
                domain: {
                    **aggregate_beam([record for record in selected if record["domain"] == domain]),
                    **aggregate_teacher([record for record in selected if record["domain"] == domain]),
                }
                for domain in DOMAIN_ORDER
            },
        }
    actual = cells["S0_bare"]["overall"]["beam_raw"]
    if abs(actual - EXPECTED_S0_BARE[label]) > 1e-8:
        raise RuntimeError(f"NEW_NO_SYSTEM_REPRO_FAIL label={label} actual={actual}")
    target = OUTPUT / f"decoder_{label}_merged.json"
    target.write_text(json.dumps({
        "decoder": label,
        "adapter": str(CHECKPOINTS[label]),
        "records": records,
        "cells": cells,
        "no_system_reproduction_pass": True,
        "training_started": False,
        "optimizer_steps": 0,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MERGED_DECODER={target}")
    print(f"NO_SYSTEM_{label}_BARE_REPRODUCTION=PASS value={actual:.10f}")


def cell_records(merged: dict[str, Any], system: str, boundary: str) -> list[dict[str, Any]]:
    return [
        record for record in merged["records"]
        if record["system_mode"] == system and record["boundary_mode"] == boundary
    ]


def group_values(records: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(float(record["beam"]["beam_raw"]))
    if len(grouped) != 12 or any(len(values) != 4 for values in grouped.values()):
        raise RuntimeError("BOOTSTRAP_PAIRING_CONTRACT_FAIL")
    return {group: statistics.fmean(values) for group, values in grouped.items()}


def bootstrap_difference(left: dict[str, float], right: dict[str, float], seed_offset: int) -> dict[str, Any]:
    groups = sorted(left)
    if groups != sorted(right):
        raise RuntimeError("BOOTSTRAP_GROUP_IDS_DIFFER")
    deltas = {group: left[group] - right[group] for group in groups}
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        chosen = [rng.choice(groups) for _ in groups]
        samples.append(statistics.fmean(deltas[group] for group in chosen))
    return {
        "mean": statistics.fmean(deltas.values()),
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "unit": "probe_group_with_4_candidates_paired",
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def metric_matrix(merged: dict[str, dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    return {
        f"{system}_{boundary}": {
            label: float(merged[label]["cells"][f"{system}_{boundary}"]["overall"][key])
            for label in DECODERS
        }
        for system, boundary in MODES
    }


def markdown_matrix(title: str, matrix: dict[str, dict[str, float]]) -> str:
    lines = [f"## {title}", "", "| Cell | D0 | D300 | D900 |", "|---|---:|---:|---:|"]
    for cell in ("S0_bare", "S0_bridge", "S1_bare", "S1_bridge"):
        values = matrix[cell]
        lines.append(f"| {cell} | {values['0']:.10f} | {values['300']:.10f} | {values['900']:.10f} |")
    return "\n".join(lines)


def classify_root(no_gain: float, system_gain: float, interaction_ci: list[float], system_effects: dict[str, float]) -> tuple[str, str]:
    if (system_gain <= 0 or system_gain < 0.5 * no_gain) and interaction_ci[1] < 0:
        return (
            "SYSTEM_RENDER_MISMATCH_STRONGLY_SUPPORTED",
            "System-present substantially removes the D900-vs-D0 gain and the paired interaction CI is strictly negative.",
        )
    if system_gain >= 0.75 * no_gain:
        return (
            "SYSTEM_RENDER_MISMATCH_NOT_PRIMARY",
            "D900 retains at least 75% of its No-System adaptation gain under System-Present context.",
        )
    if system_effects["0"] > 0 and system_effects["900"] > 0 and system_gain > 0:
        return (
            "SYSTEM_AFFECTS_LEVEL_NOT_ADAPTATION_GAIN",
            "System raises both decoder levels while a positive D900-vs-D0 adaptation gain remains.",
        )
    return "INCONCLUSIVE", "System effects are mixed or the paired confidence interval does not isolate the interaction."


def finalize() -> None:
    ensure_output_link()
    prepared = json.loads(PREPARED.read_text(encoding="utf-8"))
    merged = {
        label: json.loads((OUTPUT / f"decoder_{label}_merged.json").read_text(encoding="utf-8"))
        for label in DECODERS
    }
    if not all(item["no_system_reproduction_pass"] for item in merged.values()):
        raise RuntimeError("NO_SYSTEM_REPRODUCTION_GATE_FAIL")
    beam_matrix = metric_matrix(merged, "beam_raw")
    nll_matrix = metric_matrix(merged, "mean_gold_abc_nll")
    values = {
        label: {
            f"{system}_{boundary}": group_values(cell_records(merged[label], system, boundary))
            for system, boundary in MODES
        }
        for label in DECODERS
    }
    bootstrap = {
        "system_bare_d900_minus_d0": bootstrap_difference(values["900"]["S1_bare"], values["0"]["S1_bare"], 1),
        "system_bare_d900_minus_d300": bootstrap_difference(values["900"]["S1_bare"], values["300"]["S1_bare"], 2),
        "system_effect_d0": bootstrap_difference(values["0"]["S1_bare"], values["0"]["S0_bare"], 3),
        "system_effect_d300": bootstrap_difference(values["300"]["S1_bare"], values["300"]["S0_bare"], 4),
        "system_effect_d900": bootstrap_difference(values["900"]["S1_bare"], values["900"]["S0_bare"], 5),
    }
    interaction_left = {
        group: values["900"]["S1_bare"][group] - values["0"]["S1_bare"][group]
        for group in values["0"]["S1_bare"]
    }
    interaction_right = {
        group: values["900"]["S0_bare"][group] - values["0"]["S0_bare"][group]
        for group in values["0"]["S0_bare"]
    }
    bootstrap["interaction"] = bootstrap_difference(interaction_left, interaction_right, 6)

    effects = {
        "system_bare": {label: beam_matrix["S1_bare"][label] - beam_matrix["S0_bare"][label] for label in DECODERS},
        "system_bridge": {label: beam_matrix["S1_bridge"][label] - beam_matrix["S0_bridge"][label] for label in DECODERS},
        "decoder_gain_no_system": {
            boundary: {
                "D900-D0": beam_matrix[f"S0_{boundary}"]["900"] - beam_matrix[f"S0_{boundary}"]["0"],
                "D900-D300": beam_matrix[f"S0_{boundary}"]["900"] - beam_matrix[f"S0_{boundary}"]["300"],
            } for boundary in ("bare", "bridge")
        },
        "decoder_gain_system": {
            boundary: {
                "D900-D0": beam_matrix[f"S1_{boundary}"]["900"] - beam_matrix[f"S1_{boundary}"]["0"],
                "D900-D300": beam_matrix[f"S1_{boundary}"]["900"] - beam_matrix[f"S1_{boundary}"]["300"],
            } for boundary in ("bare", "bridge")
        },
        "boundary_gap": {
            system: {
                label: beam_matrix[f"{system}_bridge"][label] - beam_matrix[f"{system}_bare"][label]
                for label in DECODERS
            } for system in ("S0", "S1")
        },
        "interaction": bootstrap["interaction"]["mean"],
    }
    root_class, root_conclusion = classify_root(
        effects["decoder_gain_no_system"]["bare"]["D900-D0"],
        effects["decoder_gain_system"]["bare"]["D900-D0"],
        bootstrap["interaction"]["ci95"],
        effects["system_bare"],
    )

    d900_s0 = {(r["group_id"], r["candidate_id"]): r for r in cell_records(merged["900"], "S0", "bare")}
    d900_s1 = {(r["group_id"], r["candidate_id"]): r for r in cell_records(merged["900"], "S1", "bare")}
    sample_diffs = []
    for identity, s0 in d900_s0.items():
        s1 = d900_s1[identity]
        sample_diffs.append({
            "group_id": identity[0],
            "domain": s0["domain"],
            "candidate_id": identity[1],
            "s0_beam_raw": s0["beam"]["beam_raw"],
            "s1_beam_raw": s1["beam"]["beam_raw"],
            "delta": s1["beam"]["beam_raw"] - s0["beam"]["beam_raw"],
            "s0_top1_abc": s0["beam"]["beams"][0]["predicted_sid"],
            "s1_top1_abc": s1["beam"]["beams"][0]["predicted_sid"],
            "s0_top32": [beam["predicted_sid"] for beam in s0["beam"]["beams"]],
            "s1_top32": [beam["predicted_sid"] for beam in s1["beam"]["beams"]],
            "gold_set": s0["gold_sids"],
            "s0_top1_copy_class": s0["beam"]["beams"][0]["copy_class"],
            "s1_top1_copy_class": s1["beam"]["beams"][0]["copy_class"],
        })
    hurts = sorted(sample_diffs, key=lambda row: (row["delta"], row["group_id"], row["candidate_id"]))[:10]
    helps = sorted(sample_diffs, key=lambda row: (-row["delta"], row["group_id"], row["candidate_id"]))[:10]

    per_domain = {
        label: {
            f"{system}_{boundary}": merged[label]["cells"][f"{system}_{boundary}"]["per_domain"]
            for system, boundary in MODES
        } for label in DECODERS
    }
    all_records = [record for label in DECODERS for record in merged[label]["records"]]
    summary = {
        "type": "boundary_adapt_system_prompt_crossover",
        "source_commit": SOURCE_COMMIT,
        "prepared_contract": {key: prepared[key] for key in (
            "fixed_cot_count", "probe_group_count", "probe_system_found", "probe_system_variants",
            "global_system_inventory", "probe_systems", "system_byte_consistency",
            "user_content_token_parity", "system_only_crossover", "probe_vs_source_user_content",
            "render_examples", "old_no_system_reproduction", "checkpoint_provenance",
        )},
        "beam_raw_matrix": beam_matrix,
        "gold_nll_matrix": nll_matrix,
        "cells": {label: merged[label]["cells"] for label in DECODERS},
        "per_domain": per_domain,
        "effects": effects,
        "bootstrap": bootstrap,
        "sample_diffs": sample_diffs,
        "top_system_hurts": hurts,
        "top_system_helps": helps,
        "mapping_validity": {"available": False, "reason": "authoritative evaluator mappings not available in this diagnostic"},
        "root_class": root_class,
        "root_conclusion": root_conclusion,
        "training_started": False,
        "gpu_inference_started": True,
        "optimizer_steps": 0,
    }
    result_path = OUTPUT / "system_crossover_summary.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "system_crossover_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    matrix_text = "\n\n".join([
        "# Boundary Adapt System-Prompt Crossover",
        markdown_matrix("BeamRaw", beam_matrix),
        markdown_matrix("Gold NLL", nll_matrix),
        "## Effects\n\n```json\n" + json.dumps(effects, ensure_ascii=False, indent=2) + "\n```",
        "## Bootstrap\n\n```json\n" + json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n```",
        "## Per-domain\n\n```json\n" + json.dumps(per_domain, ensure_ascii=False, indent=2) + "\n```",
        f"## Root class\n\n`{root_class}`\n\n{root_conclusion}",
    ])
    (OUTPUT / "system_crossover_matrix.md").write_text(matrix_text + "\n", encoding="utf-8")

    def diff_section(title: str, rows: list[dict[str, Any]]) -> str:
        chunks = [f"## {title}"]
        for index, row in enumerate(rows, 1):
            chunks.extend([
                f"### {index}. {row['domain']} {row['group_id']} candidate={row['candidate_id']} delta={row['delta']:.6f}",
                "```json",
                json.dumps(row, ensure_ascii=False, indent=2),
                "```",
            ])
        return "\n\n".join(chunks)
    diff_text = "# D900 S0 Bare vs S1 Bare sample differences\n\n" + diff_section("TOP 10 SYSTEM HURTS", hurts)
    diff_text += "\n\n" + diff_section("TOP 10 SYSTEM HELPS", helps) + "\n"
    (OUTPUT / "system_crossover_sample_diffs.md").write_text(diff_text, encoding="utf-8")

    review = [
        "BOUNDARY ADAPT SYSTEM-PROMPT CROSSOVER REVIEW",
        "",
        "[SYSTEM INVENTORY]",
        json.dumps(prepared["global_system_inventory"], ensure_ascii=False, indent=2),
        "",
        "[12 PROBE SYSTEMS]",
        json.dumps(prepared["probe_systems"], ensure_ascii=False, indent=2),
        "",
        "[PROBE VS RAW-SOURCE USER CONTENT]",
        json.dumps(prepared["probe_vs_source_user_content"], ensure_ascii=False, indent=2),
        "",
        "[4 RENDER EXAMPLES]",
        json.dumps(prepared["render_examples"], ensure_ascii=False, indent=2),
        "",
        "[NO-SYSTEM REPRODUCTION]",
        json.dumps(prepared["old_no_system_reproduction"], indent=2),
        "",
        "[BEAM RAW MATRIX]",
        json.dumps(beam_matrix, ensure_ascii=False, indent=2),
        "",
        "[GOLD NLL MATRIX]",
        json.dumps(nll_matrix, ensure_ascii=False, indent=2),
        "",
        "[PER-DOMAIN MATRIX]",
        json.dumps(per_domain, ensure_ascii=False, indent=2),
        "",
        "[EFFECTS AND CI]",
        json.dumps({"effects": effects, "bootstrap": bootstrap}, ensure_ascii=False, indent=2),
        "",
        "[TOP 10 SYSTEM HURTS]",
        json.dumps(hurts, ensure_ascii=False, indent=2),
        "",
        "[TOP 10 SYSTEM HELPS]",
        json.dumps(helps, ensure_ascii=False, indent=2),
        "",
        "[COPY AND MAPPING SUMMARY]",
        json.dumps({
            "copy": {label: {cell: value["overall"] for cell, value in merged[label]["cells"].items()} for label in DECODERS},
            "mapping": summary["mapping_validity"],
        }, ensure_ascii=False, indent=2),
        "",
        f"ROOT_CLASS={root_class}",
        f"ROOT_CONCLUSION={root_conclusion}",
    ]
    (OUTPUT / "CHATGPT_SYSTEM_CROSSOVER_REVIEW.txt").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(f"RESULT_JSON={result_path}")
    print(f"ROOT_CLASS={root_class}")
    print(f"ROOT_CONCLUSION={root_conclusion}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-decoder", choices=DECODERS)
    parser.add_argument("--merge-decoder", choices=DECODERS)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.run_decoder is not None, args.merge_decoder is not None, args.finalize)) != 1:
        raise SystemExit("choose exactly one operation")
    if args.prepare:
        prepare()
    elif args.run_decoder:
        run_decoder(args.run_decoder)
    elif args.merge_decoder:
        merge_decoder(args.merge_decoder)
    else:
        finalize()


if __name__ == "__main__":
    main()
