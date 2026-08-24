"""Controlled, persisted Generator x Decoder crossover diagnostic.

This module is inference-only. It captures the token IDs produced inside the
authoritative FixedProbeEvaluator._think path, freezes them to JSONL, and then
reuses those IDs for strict ABC3 Beam32 and teacher-forced gold diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
from types import MethodType
from typing import Any, Iterable

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RUNTIME = Path("/data/GRPO")
sys.path[:0] = [str(RUNTIME / "scripts"), str(RUNTIME)]
from grpo_model import BASE, encode_prompt, generate_batch  # noqa: E402
from grpo_probe import FixedProbeEvaluator  # noqa: E402
from grpo_sid import think_reward  # noqa: E402
from run_grpo_trl_smoke import make_beam32_fn, make_grpo_config  # noqa: E402
from ablations.gr_rec_think_composite_interest_v1.composite_trainer import (  # noqa: E402
    ThinkCompositeInterestRecGRPOTrainer,
)
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import (  # noqa: E402
    ALL_CHECKPOINTS,
    ABC,
    DOMAIN,
    MANIFEST as FROZEN_MANIFEST,
    PROBES,
    classify,
    history_from_prompt,
    parse_abc3,
    parse_gold,
)

SEED = 20260818
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 10_000
GENERATOR_LABELS = ("0", "300", "900")
DECODER_LABELS = GENERATOR_LABELS
DOMAIN_ORDER = ("video", "prod", "ad", "living")
CHECKPOINTS = {label: ALL_CHECKPOINTS[label] for label in GENERATOR_LABELS}
RAW_DIR = RUNTIME / "boundary_adapt/results/controlled_token_crossover_20260824"
PARTS_DIR = RUNTIME / "boundary_adapt/results/controlled_crossover_20260824_parts"
FULL_RESULT = RUNTIME / "boundary_adapt/results/boundary_adapt_controlled_crossover_20260824.json"
COMPACT_RESULT = (
    RUNTIME / "baselines/native_source_domain_r32_v3/boundary_adapt/results/"
    "boundary_adapt_controlled_crossover_20260824.json"
)


class CaptureMonitor:
    enabled = True

    def write_beam_detail(self, _row):
        pass


class PersistedFixedProbeEvaluator(FixedProbeEvaluator):
    """Capture the exact IDs returned inside the existing _think implementation."""

    def _generate(self, prompts):
        model = self.trainer.model_wrapped
        original_generate = model.generate
        captured = {}

        def capture_generate(_model, *args, **kwargs):
            output = original_generate(*args, **kwargs)
            captured["prompt_completion_ids"] = output.detach().cpu().tolist()
            return output

        model.generate = MethodType(capture_generate, model)
        try:
            completion_ids = super()._generate(prompts)
        finally:
            model.generate = original_generate
        self.generated_completion_ids = [list(map(int, ids)) for ids in completion_ids]
        from grpo_model import render_prompt
        rendered = [render_prompt(self.trainer.processing_class, prompt) for prompt in prompts]
        encoded = self.trainer.processing_class(
            text=rendered, return_tensors="pt", padding=True, padding_side="left",
            max_length=self.trainer.max_prompt_length, truncation=True, add_special_tokens=False,
        )
        prompt_length = int(encoded["input_ids"].shape[1])
        self.raw_generation_completion_ids = [
            list(map(int, ids[prompt_length:])) for ids in captured["prompt_completion_ids"]
        ]
        return completion_ids


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_ids_sha(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def load_probe_contract():
    rows = []
    with PROBES.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("step", -1)) == 0:
                rows.append(row)
    rows.sort(key=lambda row: (int(row["probe_round"]), DOMAIN_ORDER.index(row["target_domain"])))
    if len(rows) != 12 or Counter(row["target_domain"] for row in rows) != Counter({d: 3 for d in DOMAIN_ORDER}):
        raise RuntimeError("CONTROLLED_12_PROBE_CONTRACT_INVALID")
    records: dict[str, Any] = {}
    rounds: list[list[str]] = []
    for round_index in range(3):
        chunk = rows[round_index * 4:(round_index + 1) * 4]
        rounds.append([row["group_id"] for row in chunk])
        for row in chunk:
            records[row["group_id"]] = {"think": {
                "prompt": row["think_prompt"],
                "all_gold_sids": row["gold_sids"],
                "target_domain": row["target_domain"],
            }}
    return rows, records, rounds


def load_model(adapter: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tokenizer


def dummy_reward(completions, **_kwargs):
    return [0.0] * len(completions)


def generation_config_snapshot(trainer) -> dict[str, Any]:
    config = trainer.generation_config
    return {
        "route": "think",
        "num_generations": 4,
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "do_sample": bool(config.do_sample),
        "max_completion_length": int(trainer.args.max_completion_length),
        "cache_implementation": config.cache_implementation,
        "disable_compile": getattr(config, "disable_compile", None),
        "stop_strings": config.stop_strings,
        "stopping": "per-sample token-id </think> StoppingCriteria plus defensive token-id truncation",
    }


def artifact_name(label: str, variant: str) -> str:
    suffix = "" if variant == "main" else f"_{variant}"
    return f"G{label}{suffix}.jsonl"


def validate_rows(rows: list[dict[str, Any]], tokenizer, expected_generator: str) -> None:
    if len(rows) != 48:
        raise RuntimeError(f"CONTROLLED_COT_COUNT_INVALID={len(rows)}")
    identity = {(row["group_id"], row["candidate_id"]) for row in rows}
    if len(identity) != 48:
        raise RuntimeError("CONTROLLED_COT_IDENTITY_NOT_UNIQUE")
    if Counter(row["target_domain"] for row in rows) != Counter({d: 12 for d in DOMAIN_ORDER}):
        raise RuntimeError("CONTROLLED_COT_DOMAIN_BALANCE_INVALID")
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("THINK_CLOSE_NOT_ATOMIC")
    close_id = close_ids[0]
    vocab_size = len(tokenizer)
    for row in rows:
        ids = list(map(int, row["completion_token_ids"]))
        if row["generator_checkpoint"] != expected_generator:
            raise RuntimeError("GENERATOR_LABEL_MISMATCH")
        if tokenizer.encode(row["completion_text_display"], add_special_tokens=False) != row["text_reencode_token_ids"]:
            raise RuntimeError("TEXT_REENCODE_DIAGNOSTIC_MISMATCH")
        if (row["text_reencode_token_ids"] == ids) != row["text_roundtrip_equal"]:
            raise RuntimeError("TEXT_ROUNDTRIP_FLAG_MISMATCH")
        if text_sha(row["completion_text_display"]) != row["completion_text_sha256"]:
            raise RuntimeError("COMPLETION_TEXT_SHA_MISMATCH")
        if token_ids_sha(ids) != row["completion_token_ids_sha256"]:
            raise RuntimeError("COMPLETION_TOKEN_SHA_MISMATCH")
        if any(token_id < 0 or token_id >= vocab_size for token_id in ids):
            raise RuntimeError("COMPLETION_TOKEN_ID_OUT_OF_RANGE")
        try:
            closure_position = ids.index(close_id)
        except ValueError:
            closure_position = None
        if closure_position != row["closure_token_index"] or closure_position != row["close_index"] or (closure_position is not None) != row["closed"]:
            raise RuntimeError("COMPLETION_CLOSURE_MISMATCH")
        if len(ids) != row["completion_token_length"] or len(ids) != row["production_cropped_ids_length"]:
            raise RuntimeError("COMPLETION_LENGTH_MISMATCH")
        if row["raw_generation_ids_length"] < row["production_cropped_ids_length"]:
            raise RuntimeError("PRODUCTION_CROP_LENGTH_INVALID")


def write_jsonl_roundtrip(path: Path, rows: list[dict[str, Any]], tokenizer, label: str) -> None:
    validate_rows(rows, tokenizer, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    reopened = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate_rows(reopened, tokenizer, label)
    before = {(row["group_id"], row["candidate_id"]): row["completion_token_ids"] for row in rows}
    after = {(row["group_id"], row["candidate_id"]): row["completion_token_ids"] for row in reopened}
    if before != after:
        raise RuntimeError("TOKEN_ID_WRITE_READ_PERSISTENCE_MISMATCH")


def run_generation(label: str, variant: str) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("GENERATION_REQUIRES_FOUR_LOCAL_RANKS")
    torch.cuda.set_device(rank)
    random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed(SEED + rank)
    _, records, rounds = load_probe_contract()
    model, tokenizer = load_model(CHECKPOINTS[label], f"cuda:{rank}")
    cfg = make_grpo_config(str(PARTS_DIR / f"trainer_G{label}_{variant}"), 1, 0.0, SEED)
    seed_prompt = next(iter(records.values()))["think"]["prompt"]
    seed_rows = [{"prompt": seed_prompt, "route": "think"}] * 16
    monitor = CaptureMonitor()
    trainer = ThinkCompositeInterestRecGRPOTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(seed_rows),
        reward_funcs=[dummy_reward],
        monitor_writer=None,
    )
    beam32 = make_beam32_fn(model, tokenizer, monitor_writer=monitor)
    local = []
    for round_index, group_ids in enumerate(rounds):
        evaluator = PersistedFixedProbeEvaluator(trainer, records, group_ids, beam32, monitor, SEED, 1)
        evaluator._set_seed()
        item = evaluator._think()
        item["probe_round"] = round_index
        item["origin_rank"] = rank
        item["completion_token_ids"] = evaluator.generated_completion_ids
        item["raw_generation_completion_ids"] = evaluator.raw_generation_completion_ids
        item["generation_config"] = generation_config_snapshot(trainer)
        local.append(item)
        print(f"GENERATION_PROGRESS G={label} variant={variant} rank={rank} round={round_index + 1}/3", flush=True)
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        items = [item for rank_items in gathered for item in rank_items]
        items.sort(key=lambda item: (item["probe_round"], DOMAIN_ORDER.index(item["target_domain"])))
        output_rows = []
        for item in items:
            prompt_ids = encode_prompt(tokenizer, item["prompt"])
            for candidate_id, (candidate, ids, raw_ids) in enumerate(zip(
                item["candidates"], item["completion_token_ids"], item["raw_generation_completion_ids"]
            )):
                display = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                reencoded = tokenizer.encode(display, add_special_tokens=False)
                close_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
                close_index = ids.index(close_id) if close_id in ids else None
                if raw_ids == ids:
                    crop_reason = "NONE"
                elif close_index is not None and ids[-1] == close_id:
                    crop_reason = "THINK_CLOSE_TOKEN"
                elif trainer.eos_token_id in ids and ids[-1] == trainer.eos_token_id:
                    crop_reason = "EOS_TOKEN"
                else:
                    crop_reason = "PRODUCTION_POST_GENERATION_CROP"
                output_rows.append({
                    "generator_checkpoint": label,
                    "generator_adapter": str(CHECKPOINTS[label]),
                    "probe_round": item["probe_round"],
                    "origin_rank": item["origin_rank"],
                    "group_id": item["group_id"],
                    "target_domain": item["target_domain"],
                    "candidate_id": candidate_id,
                    "seed_base": SEED,
                    "rank_seed": SEED + item["origin_rank"],
                    "effective_seed": SEED + item["origin_rank"],
                    "seed_provenance": "FixedProbeEvaluator._set_seed resets SEED + rank before each probe round",
                    "generation_config": item["generation_config"],
                    "generation_wall_sec_for_group": float(item["generation_wall_sec"]),
                    "prompt_text": item["prompt"],
                    "prompt_sha256": text_sha(item["prompt"]),
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_ids_sha256": token_ids_sha(prompt_ids),
                    "gold_sids": item["gold_sids"],
                    "completion_token_ids": ids,
                    "completion_token_ids_sha256": token_ids_sha(ids),
                    "completion_text_display": display,
                    "completion_text_sha256": text_sha(display),
                    "text_reencode_token_ids": reencoded,
                    "text_roundtrip_equal": reencoded == ids,
                    "closed": bool(candidate["closed"]),
                    "closure_token_index": close_index,
                    "raw_generation_ids_length": len(raw_ids),
                    "production_cropped_ids_length": len(ids),
                    "close_index": close_index,
                    "crop_reason": crop_reason,
                    "completion_token_length": len(ids),
                })
        target = RAW_DIR / artifact_name(label, variant)
        write_jsonl_roundtrip(target, output_rows, tokenizer, label)
        print(f"CONTROLLED_COT_ARTIFACT={target}")
        print(f"CONTROLLED_COT_COUNT={len(output_rows)}")
    dist.barrier()
    dist.destroy_process_group()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_cot_artifacts() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    repeat_a = RAW_DIR / artifact_name("300", "A")
    repeat_b = RAW_DIR / artifact_name("300", "B")
    rows_a, rows_b = load_jsonl(repeat_a), load_jsonl(repeat_b)
    validate_rows(rows_a, tokenizer, "300")
    validate_rows(rows_b, tokenizer, "300")
    index_b = {(row["group_id"], row["candidate_id"]): row for row in rows_b}
    matches = sum(
        row["completion_token_ids_sha256"] == index_b[(row["group_id"], row["candidate_id"])]["completion_token_ids_sha256"]
        for row in rows_a
    )
    shutil.copyfile(repeat_a, RAW_DIR / artifact_name("300", "main"))
    manifests = {}
    schedules = []
    for label in GENERATOR_LABELS:
        path = RAW_DIR / artifact_name(label, "main")
        rows = load_jsonl(path)
        validate_rows(rows, tokenizer, label)
        schedules.append([
            (row["probe_round"], row["group_id"], row["candidate_id"], row["origin_rank"], row["seed_base"], row["effective_seed"])
            for row in rows
        ])
        manifests[label] = {"path": str(path), "sha256": file_sha(path), "count": len(rows)}
    schedule_pass = schedules[0] == schedules[1] == schedules[2]
    payload = {
        "controlled_cot_generation_pass": True,
        "canonical_object": "PRODUCTION_CROPPED_COMPLETION_TOKEN_IDS",
        "token_id_persistence_pass": True,
        "token_sha_pass": True,
        "text_roundtrip_match_count": sum(
            row["text_roundtrip_equal"]
            for label in GENERATOR_LABELS
            for row in load_jsonl(RAW_DIR / artifact_name(label, "main"))
        ),
        "text_roundtrip_total": 144,
        "text_roundtrip_is_hard_gate": False,
        "controlled_seed_schedule_pass": schedule_pass,
        "generation_token_repeatable": matches == 48,
        "token_repeat_match_count": matches,
        "manifests": manifests,
    }
    if not schedule_pass:
        raise RuntimeError("CONTROLLED_SEED_SCHEDULE_MISMATCH")
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    (PARTS_DIR / "controlled_cot_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def strict_beam(model, tokenizer, context: list[int], domain: str, gold_set, history):
    with torch.inference_mode():
        _, raw_ids = generate_batch(
            model,
            tokenizer,
            [context],
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
            num_beams=32,
            num_return_sequences=32,
            return_ids=True,
        )
    raw_ids = [list(map(int, ids)) for ids in raw_ids]
    sids = [parse_abc3(tokenizer, domain, ids) for ids in raw_ids]
    reward, exact, ab, a = think_reward(sids, gold_set)
    return {
        "beam_raw": float(reward),
        "exact": int(exact),
        "ab": int(ab),
        "a": int(a),
        "invalid": sum(sid is None for sid in sids),
        "beams": [
            {"beam_index": index, "raw_token_ids": ids, **classify(sid, history, gold_set)}
            for index, (ids, sid) in enumerate(zip(raw_ids, sids))
        ],
    }


def gold_token_ids(tokenizer, gold_sid: str) -> list[int]:
    parsed = parse_gold(gold_sid)
    names = [f"<s_a_{parsed[1]}>", f"<s_b_{parsed[2]}>", f"<s_c_{parsed[3]}>"]
    ids = [tokenizer.encode(name, add_special_tokens=False) for name in names]
    if any(len(value) != 1 for value in ids):
        raise RuntimeError(f"GOLD_ABC_TOKEN_NOT_ATOMIC={gold_sid}")
    return [value[0] for value in ids]


def teacher_forced_gold(model, tokenizer, context: list[int], gold_sids: list[str]) -> dict[str, Any]:
    paths = [gold_token_ids(tokenizer, sid) for sid in gold_sids]
    unique_a = list(dict.fromkeys(path[0] for path in paths))
    unique_ab = list(dict.fromkeys((path[0], path[1]) for path in paths))

    def final_logits(prefixes: list[list[int]]) -> torch.Tensor:
        input_ids = torch.tensor([context + prefix for prefix in prefixes], dtype=torch.long, device=model.device)
        return model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()

    rows = []
    with torch.inference_mode():
        a_logits = final_logits([[]])[0]
        b_logits = final_logits([[a] for a in unique_a])
        c_logits = final_logits([list(ab) for ab in unique_ab])
        b_by_a = dict(zip(unique_a, b_logits))
        c_by_ab = dict(zip(unique_ab, c_logits))
        for abc_ids in paths:
            logps, ranks = [], []
            stage_logits = (a_logits, b_by_a[abc_ids[0]], c_by_ab[(abc_ids[0], abc_ids[1])])
            for logits, target in zip(stage_logits, abc_ids):
                logp = torch.log_softmax(logits, dim=-1)[target]
                rank = int((logits > logits[target]).sum().item()) + 1
                logps.append(float(logp.item()))
                ranks.append(rank)
            rows.append({
                "gold_sid": gold_sids[len(rows)],
                "A_logp": logps[0], "A_rank": ranks[0],
                "B_logp": logps[1], "B_rank": ranks[1],
                "C_logp": logps[2], "C_rank": ranks[2],
                "gold_abc_nll": -sum(logps) / 3.0,
            })
    return {
        "paths": rows,
        "mean_gold_abc_nll": statistics.fmean(row["gold_abc_nll"] for row in rows),
        "A_rank_mean": statistics.fmean(row["A_rank"] for row in rows),
        "B_rank_mean": statistics.fmean(row["B_rank"] for row in rows),
        "C_rank_mean": statistics.fmean(row["C_rank"] for row in rows),
        "best_gold_path_nll": min(row["gold_abc_nll"] for row in rows),
        "best_A_rank": min(row["A_rank"] for row in rows),
    }


def run_decoder(label: str) -> None:
    gate = json.loads((PARTS_DIR / "controlled_cot_gate.json").read_text(encoding="utf-8"))
    if not gate["controlled_cot_generation_pass"]:
        raise RuntimeError("CONTROLLED_COT_GATE_NOT_PASSED")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("DECODER_REQUIRES_FOUR_LOCAL_RANKS")
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    model, tokenizer = load_model(CHECKPOINTS[label], f"cuda:{rank}")
    controlled = {g: load_jsonl(RAW_DIR / artifact_name(g, "main")) for g in GENERATOR_LABELS}
    local_records = []
    for generator, items in controlled.items():
        for index, item in enumerate(items):
            if index % world != rank:
                continue
            prompt_ids = list(map(int, item["prompt_token_ids"]))
            cot_ids = list(map(int, item["completion_token_ids"]))
            if encode_prompt(tokenizer, item["prompt_text"]) != prompt_ids:
                raise RuntimeError("PERSISTED_PROMPT_TOKEN_MISMATCH")
            domain_ids = tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)
            if len(domain_ids) != 1:
                raise RuntimeError("DOMAIN_TOKEN_NOT_ATOMIC")
            context = prompt_ids + cot_ids + domain_ids
            history = history_from_prompt(tokenizer, prompt_ids)
            gold_set = {parse_gold(value) for value in item["gold_sids"]}
            beam = strict_beam(model, tokenizer, context, item["target_domain"], gold_set, history)
            teacher = teacher_forced_gold(model, tokenizer, context, item["gold_sids"])
            repeat_beam = None
            if generator == label:
                repeat_beam = strict_beam(model, tokenizer, context, item["target_domain"], gold_set, history)
                if [row["raw_token_ids"] for row in beam["beams"]] != [row["raw_token_ids"] for row in repeat_beam["beams"]]:
                    raise RuntimeError(f"NEW_DIAGONAL_INTERNAL_INCONSISTENCY G={generator} D={label}")
            local_records.append({
                "generator": generator,
                "decoder": label,
                "group_id": item["group_id"],
                "candidate_id": item["candidate_id"],
                "probe_round": item["probe_round"],
                "target_domain": item["target_domain"],
                "completion_token_ids_sha256": item["completion_token_ids_sha256"],
                "beam": beam,
                "teacher_forced": teacher,
                "diagonal_repeat_pass": repeat_beam is not None,
            })
            print(f"DECODER_PROGRESS D={label} G={generator} rank={rank} item={index // world + 1}/12", flush=True)
    frozen_items = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))["items"]
    anchor_records = []
    for index, item in enumerate(frozen_items):
        if index % world != rank:
            continue
        prompt_ids = encode_prompt(tokenizer, item["think_prompt"])
        cot_ids = tokenizer.encode(item["fixed_cot"], add_special_tokens=False)
        domain_ids = tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)
        history = history_from_prompt(tokenizer, prompt_ids)
        gold_set = {parse_gold(value) for value in item["gold_sids"]}
        beam = strict_beam(model, tokenizer, prompt_ids + cot_ids + domain_ids, item["target_domain"], gold_set, history)
        anchor_records.append({
            "group_id": item["recommendation_group_id"],
            "candidate_id": item["candidate_id"],
            "target_domain": item["target_domain"],
            "beam": beam,
        })
        print(f"ANCHOR_PROGRESS D={label} rank={rank} item={index // world + 1}/12", flush=True)
    target = PARTS_DIR / f"decoder_{label}"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "decoder": label,
        "adapter": str(CHECKPOINTS[label]),
        "rank": rank,
        "records": local_records,
        "frozen_anchor_records": anchor_records,
        "training_started": False,
        "optimizer_steps": 0,
    }
    (target / f"rank{rank}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def aggregate_beam(records: list[dict[str, Any]]) -> dict[str, Any]:
    beams = [beam for record in records for beam in record["beam"]["beams"]]
    valid = [beam for beam in beams if beam["predicted_sid"] is not None]
    copies = Counter(beam["copy_class"] for beam in valid)
    cross = Counter(beam["gold_history_class"] for beam in beams)
    total = len(beams)
    return {
        "beam_raw": statistics.fmean(record["beam"]["beam_raw"] for record in records),
        "exact": sum(record["beam"]["exact"] for record in records),
        "ab": sum(record["beam"]["ab"] for record in records),
        "a": sum(record["beam"]["a"] for record in records),
        "invalid": sum(record["beam"]["invalid"] for record in records),
        "total_beams": total,
        "history_exact_copy": rate(copies["EXACT_COPY"], len(valid)),
        "history_not_gold": rate(cross["HISTORY_NOT_GOLD"], total),
        "gold_not_history": rate(cross["GOLD_NOT_HISTORY"], total),
        "novel": rate(copies["NOVEL"], len(valid)),
    }


def aggregate_teacher(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("mean_gold_abc_nll", "A_rank_mean", "B_rank_mean", "C_rank_mean", "best_gold_path_nll", "best_A_rank")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record["teacher_forced"])
    if len(groups) != 12 or any(len(values) != 4 for values in groups.values()):
        raise RuntimeError("TEACHER_GROUP_NORMALIZATION_CONTRACT_INVALID")
    return {
        key: statistics.fmean(statistics.fmean(item[key] for item in values) for values in groups.values())
        for key in keys
    }


def merge_decoder(label: str) -> None:
    parts = [json.loads((PARTS_DIR / f"decoder_{label}/rank{rank}.json").read_text(encoding="utf-8")) for rank in range(4)]
    records = [record for part in parts for record in part["records"]]
    anchors = [record for part in parts for record in part["frozen_anchor_records"]]
    if len(records) != 144 or len(anchors) != 48:
        raise RuntimeError(f"DECODER_PARTS_INCOMPLETE records={len(records)} anchors={len(anchors)}")
    cells = {}
    for generator in GENERATOR_LABELS:
        selected = [record for record in records if record["generator"] == generator]
        if len(selected) != 48:
            raise RuntimeError("CROSSOVER_CELL_INCOMPLETE")
        cells[generator] = {
            "overall": {**aggregate_beam(selected), **aggregate_teacher(selected)},
            "per_domain": {
                domain: {**aggregate_beam([r for r in selected if r["target_domain"] == domain]),
                         **aggregate_teacher_domain([r for r in selected if r["target_domain"] == domain])}
                for domain in DOMAIN_ORDER
            },
        }
    payload = {
        "decoder": label,
        "adapter": str(CHECKPOINTS[label]),
        "records": records,
        "cells": cells,
        "frozen_anchor": aggregate_beam(anchors),
        "new_diagonal_internal_consistency": all(
            record["diagonal_repeat_pass"] for record in records if record["generator"] == label
        ),
    }
    path = PARTS_DIR / f"decoder_{label}_merged.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MERGED_DECODER={path}")


def aggregate_teacher_domain(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("mean_gold_abc_nll", "A_rank_mean", "B_rank_mean", "C_rank_mean", "best_gold_path_nll", "best_A_rank")
    return {key: statistics.fmean(record["teacher_forced"][key] for record in records) for key in keys}


def contains_subsequence(ids: list[int], needle: list[int]) -> bool:
    if not needle or len(needle) > len(ids):
        return False
    return any(ids[index:index + len(needle)] == needle for index in range(len(ids) - len(needle) + 1))


def generator_diagnostics(rows: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    history_counts = Counter()
    gold_counts = Counter()
    text_history_counts = Counter()
    text_gold_counts = Counter()
    suffix_history = Counter()
    suffix_gold = Counter()
    suffix_domain = 0
    for row in rows:
        ids = row["completion_token_ids"]
        suffix = ids[-128:]
        text = row["completion_text_display"]
        prompt_history = history_from_prompt(tokenizer, row["prompt_token_ids"])["unique_set"]
        gold = {parse_gold(value) for value in row["gold_sids"]}
        domain_id = tokenizer.encode(DOMAIN[row["target_domain"]], add_special_tokens=False)
        suffix_domain += contains_subsequence(suffix, domain_id)
        for name in ("exact", "ab", "a"):
            def literal(sid):
                abc = [f"<s_a_{sid[1]}>", f"<s_b_{sid[2]}>", f"<s_c_{sid[3]}>"]
                if name == "exact":
                    return DOMAIN[sid[0]] + "".join(abc)
                if name == "ab":
                    return "".join(abc[:2])
                return abc[0]

            history_text = [literal(sid) for sid in prompt_history]
            gold_text = [literal(sid) for sid in gold]
            history_needles = [tokenizer.encode(value, add_special_tokens=False) for value in history_text]
            gold_needles = [tokenizer.encode(value, add_special_tokens=False) for value in gold_text]
            history_counts[name] += any(contains_subsequence(ids, needle) for needle in history_needles)
            gold_counts[name] += any(contains_subsequence(ids, needle) for needle in gold_needles)
            text_history_counts[name] += any(value in text for value in history_text)
            text_gold_counts[name] += any(value in text for value in gold_text)
            suffix_history[name] += any(contains_subsequence(suffix, needle) for needle in history_needles)
            suffix_gold[name] += any(contains_subsequence(suffix, needle) for needle in gold_needles)
    lengths = [row["completion_token_length"] for row in rows]
    denom = len(rows)
    return {
        "closed_rate": rate(sum(row["closed"] for row in rows), denom),
        "empty_rate": rate(sum(not row["completion_token_ids"] for row in rows), denom),
        "length": {"mean": statistics.fmean(lengths), "p50": percentile(lengths, 0.5), "p95": percentile(lengths, 0.95)},
        "token_level_history_literal": {key: rate(history_counts[key], denom) for key in ("exact", "ab", "a")},
        "token_level_gold_literal": {key: rate(gold_counts[key], denom) for key in ("exact", "ab", "a")},
        "text_view_history_literal": {key: rate(text_history_counts[key], denom) for key in ("exact", "ab", "a")},
        "text_view_gold_literal": {key: rate(text_gold_counts[key], denom) for key in ("exact", "ab", "a")},
        "last_128_tokens": {
            "history_literal": {key: rate(suffix_history[key], denom) for key in ("exact", "ab", "a")},
            "gold_literal": {key: rate(suffix_gold[key], denom) for key in ("exact", "ab", "a")},
            "target_domain_marker_rate": rate(suffix_domain, denom),
        },
    }


def group_values(records: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(float(record["beam"]["beam_raw"]))
    if len(grouped) != 12 or any(len(values) != 4 for values in grouped.values()):
        raise RuntimeError("BOOTSTRAP_GROUP_CONTRACT_INVALID")
    return {group: statistics.fmean(values) for group, values in grouped.items()}


def bootstrap_contrast(left: dict[str, float], right: dict[str, float], rng: random.Random) -> dict[str, Any]:
    groups = sorted(left)
    if groups != sorted(right):
        raise RuntimeError("BOOTSTRAP_GROUP_IDS_MISMATCH")
    observed = statistics.fmean(left[group] - right[group] for group in groups)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        selected = [rng.choice(groups) for _ in groups]
        samples.append(statistics.fmean(left[group] - right[group] for group in selected))
    return {"mean_delta": observed, "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)]}


def classify_root(effects: dict[str, dict[str, Any]], nll: dict[str, dict[str, float]]) -> tuple[str, str]:
    generator = [effects[f"generator_G900_minus_G300_at_D{d}"] for d in DECODER_LABELS]
    decoder = [effects[f"decoder_D900_minus_D300_on_G{g}"] for g in GENERATOR_LABELS]
    nll_worse = all(nll["900"][d] > nll["300"][d] for d in DECODER_LABELS)
    if all(item["ci95"][1] < 0 for item in generator) and nll_worse:
        return "GENERATOR_DRIFT", "CoT preservation replay"
    interaction = effects["interaction"]
    if interaction["ci95"][1] < 0:
        return "GENERATOR_DECODER_INTERACTION", "mixed teacher/self-CoT boundary supervision"
    if generator[0]["ci95"][0] <= 0 <= generator[0]["ci95"][1] and generator[1]["ci95"][0] <= 0 <= generator[1]["ci95"][1] and generator[2]["ci95"][1] < 0:
        return "DECODER_SELF_COT_DISTRIBUTION_MISMATCH", "offline-frozen Step900 self-CoT plus Gold ABC-only CE, mixed with teacher CoT"
    if (
        all(item["ci95"][0] <= 0 <= item["ci95"][1] for item in generator)
        and all(item["ci95"][0] > 0 for item in decoder)
        and interaction["ci95"][0] <= 0 <= interaction["ci95"][1]
    ):
        return "NO_CLEAR_REGRESSION", "keep Step900 and expand official-like held-out probe/evaluation; do not continue training"
    return "MIXED_OR_INCONCLUSIVE", "collect a larger persisted controlled probe set before selecting any training recipe"


def finalize() -> None:
    gate = json.loads((PARTS_DIR / "controlled_cot_gate.json").read_text(encoding="utf-8"))
    decoders = {label: json.loads((PARTS_DIR / f"decoder_{label}_merged.json").read_text(encoding="utf-8")) for label in DECODER_LABELS}
    if not all(item["new_diagonal_internal_consistency"] for item in decoders.values()):
        raise RuntimeError("NEW_DIAGONAL_INTERNAL_CONSISTENCY_FAILED")
    expected_anchor = {"300": 2.7447916666666665, "900": 3.5338541666666665}
    anchor_pass = all(abs(decoders[label]["frozen_anchor"]["beam_raw"] - value) < 1e-12 for label, value in expected_anchor.items())
    if not anchor_pass:
        raise RuntimeError("FROZEN_COT_ANCHOR_PARITY_FAILED")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    generator_stats = {
        label: generator_diagnostics(load_jsonl(RAW_DIR / artifact_name(label, "main")), tokenizer)
        for label in GENERATOR_LABELS
    }
    records = {d: decoders[d]["records"] for d in DECODER_LABELS}
    cell_groups = {
        (g, d): group_values([record for record in records[d] if record["generator"] == g])
        for g in GENERATOR_LABELS for d in DECODER_LABELS
    }
    rng = random.Random(BOOTSTRAP_SEED)
    effects = {}
    for decoder in DECODER_LABELS:
        effects[f"generator_G900_minus_G300_at_D{decoder}"] = bootstrap_contrast(cell_groups[("900", decoder)], cell_groups[("300", decoder)], rng)
    for generator in GENERATOR_LABELS:
        effects[f"decoder_D900_minus_D300_on_G{generator}"] = bootstrap_contrast(cell_groups[(generator, "900")], cell_groups[(generator, "300")], rng)
    interaction_left = {group: cell_groups[("900", "900")][group] - cell_groups[("900", "300")][group] for group in cell_groups[("900", "900")]}
    interaction_right = {group: cell_groups[("300", "900")][group] - cell_groups[("300", "300")][group] for group in cell_groups[("300", "900")]}
    effects["interaction"] = bootstrap_contrast(interaction_left, interaction_right, rng)
    beam_matrix = {g: {d: decoders[d]["cells"][g]["overall"]["beam_raw"] for d in DECODER_LABELS} for g in GENERATOR_LABELS}
    nll_matrix = {g: {d: decoders[d]["cells"][g]["overall"]["mean_gold_abc_nll"] for d in DECODER_LABELS} for g in GENERATOR_LABELS}
    root_cause, next_recipe = classify_root(effects, nll_matrix)
    full = {
        "type": "boundary_adaptation_controlled_generator_decoder_crossover",
        "source_commit": "836b3540549233bfeeeb1664bb85ea477b131237",
        "seed": SEED,
        "seed_claim": "same controlled RNG schedule, not the same random sample across different model distributions",
        "controlled_cot_gate": gate,
        "generator_stats": generator_stats,
        "decoders": decoders,
        "beam_raw_matrix": beam_matrix,
        "gold_nll_matrix": nll_matrix,
        "effects": effects,
        "bootstrap": {"cluster": "recommendation_group", "groups": 12, "candidates_per_group": 4, "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "frozen_cot_parity_pass": anchor_pass,
        "new_diagonal_internal_consistency": True,
        "root_cause_class": root_cause,
        "next_recommended_training_recipe": next_recipe,
        "historical_self_cot": {
            "status": "HISTORICAL_NON_REPLAYABLE",
            "step300_raw": 3.421875,
            "step900_raw": 3.2421875,
        },
        "training_started": False,
        "optimizer_steps": 0,
    }
    compact = {
        key: full[key] for key in (
            "type", "source_commit", "seed", "seed_claim", "controlled_cot_gate", "generator_stats",
            "beam_raw_matrix", "gold_nll_matrix", "effects", "bootstrap", "frozen_cot_parity_pass",
            "new_diagonal_internal_consistency", "root_cause_class", "next_recommended_training_recipe",
            "historical_self_cot", "training_started", "optimizer_steps",
        )
    }
    compact["cells"] = {d: decoders[d]["cells"] for d in DECODER_LABELS}
    compact["frozen_anchor"] = {d: decoders[d]["frozen_anchor"] for d in DECODER_LABELS}
    FULL_RESULT.parent.mkdir(parents=True, exist_ok=True)
    COMPACT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    FULL_RESULT.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    COMPACT_RESULT.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ROOT_CAUSE_CLASS={root_cause}")
    print(f"RESULT_JSON={FULL_RESULT}")
    print(f"COMPACT_JSON={COMPACT_RESULT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", choices=GENERATOR_LABELS)
    parser.add_argument("--variant", choices=("main", "A", "B"), default="main")
    parser.add_argument("--validate-cots", action="store_true")
    parser.add_argument("--decode", choices=DECODER_LABELS)
    parser.add_argument("--merge-decoder", choices=DECODER_LABELS)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    operations = (args.generate is not None, args.validate_cots, args.decode is not None, args.merge_decoder is not None, args.finalize)
    if sum(map(bool, operations)) != 1:
        raise SystemExit("choose exactly one operation")
    if args.generate is not None:
        run_generation(args.generate, args.variant)
    elif args.validate_cots:
        validate_cot_artifacts()
    elif args.decode is not None:
        run_decoder(args.decode)
    elif args.merge_decoder is not None:
        merge_decoder(args.merge_decoder)
    else:
        finalize()


if __name__ == "__main__":
    main()
