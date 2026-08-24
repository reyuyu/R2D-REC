"""Adaptation-heldout generation and strict Beam32 evaluation."""
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
import time
from typing import Any

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .common import (
    BASE,
    DOMAIN,
    DOMAIN_ORDER,
    FRESH_BATA,
    SEED,
    close_position,
    emission_metrics,
    stable_hash,
    summarize_lengths,
    token_ids_sha,
)

RUNTIME = Path("/data/GRPO")
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]
from boundary_adapt.diagnostics import controlled_generator_decoder_crossover as cross  # noqa: E402

MAX_PROMPT_LENGTH = 8192
MAX_NEW_TOKENS = 4096
TEMPERATURE = .9
TOP_P = .95


class ThinkTokenStop(StoppingCriteria):
    def __init__(self, token_id: int):
        self.token_id = token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token_id


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


def setup() -> tuple[int, int, str]:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("HELDOUT_EVAL_REQUIRES_4_GPUS")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    return rank, world, f"cuda:{rank}"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    if len(items) != 256 or len({item["group_id"] for item in items}) != 256:
        raise RuntimeError("PRIMARY_HELDOUT_MANIFEST_INVALID")
    if Counter(item["target_domain"] for item in items) != Counter({domain: 64 for domain in DOMAIN_ORDER}):
        raise RuntimeError("PRIMARY_HELDOUT_DOMAIN_BALANCE_INVALID")
    return items


def seed_for(group: str) -> int:
    # Same per-group seed at every checkpoint for paired sampling comparisons.
    digest = hashlib.sha256(f"{SEED}|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def generate_one(model, tokenizer, prompt_ids: list[int], close_id: int, seed: int) -> tuple[list[int], float]:
    if len(prompt_ids) > MAX_PROMPT_LENGTH:
        prompt_ids = prompt_ids[:MAX_PROMPT_LENGTH]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    attention = torch.ones_like(inputs)
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=attention,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=1,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([ThinkTokenStop(close_id)]),
            disable_compile=True,
        )
    torch.cuda.synchronize(model.device)
    wall = time.perf_counter() - started
    completion = list(map(int, output[0, inputs.shape[1]:].tolist()))
    close = close_position(completion, close_id)
    if close is not None:
        completion = completion[:close + 1]
    return completion, wall


def known_bridges(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    values: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for item in items:
        values[item["target_domain"]].add(tuple(map(int, item["bridge_token_ids"])))
    if any(len(values[domain]) != 1 for domain in DOMAIN_ORDER):
        raise RuntimeError(f"DOMAIN_BRIDGE_VARIANTS_INVALID={values}")
    return {domain: list(next(iter(values[domain]))) for domain in DOMAIN_ORDER}


def generate_self_cots(label: str, adapter: Path, manifest: Path, parts: Path) -> None:
    rank, world, device = setup()
    items = load_manifest(manifest)
    model, tokenizer = load_model(adapter, device)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("CLOSE_TOKEN_NOT_ATOMIC")
    close_id = int(close_ids[0])
    bridges = known_bridges(items)
    local = []
    for index, item in enumerate(items):
        if index % world != rank:
            continue
        prompt = list(map(int, item["prompt_token_ids"]))
        completion, wall = generate_one(model, tokenizer, prompt, close_id, seed_for(item["group_id"]))
        emission = emission_metrics(completion, close_id, list(map(int, item["bridge_token_ids"])), bridges, item["target_domain"])
        text = tokenizer.decode(completion, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        local.append({
            "label": label,
            "group_id": item["group_id"],
            "target_domain": item["target_domain"],
            "prompt_token_ids_sha256": token_ids_sha(prompt),
            "completion_token_ids": completion,
            "completion_token_ids_sha256": token_ids_sha(completion),
            "completion_token_length": len(completion),
            "generation_wall_sec": wall,
            "literal_sid_occurrences": text.count("<s_a_"),
            "emission": emission,
        })
        print(f"SELF_COT_PROGRESS label={label} rank={rank} sample={index // world + 1}/64", flush=True)
    target = parts / f"self_cot_{label}"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"rank{rank}.json").write_text(json.dumps({"records": local}, ensure_ascii=False) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def merge_self_cots(label: str, parts: Path, artifact: Path) -> dict[str, Any]:
    records = [record for rank in range(4) for record in json.loads((parts / f"self_cot_{label}/rank{rank}.json").read_text(encoding="utf-8"))["records"]]
    if len(records) != 256 or len({record["group_id"] for record in records}) != 256:
        raise RuntimeError("SELF_COT_SHARDS_INCOMPLETE")
    records.sort(key=lambda row: row["group_id"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    lengths = [record["completion_token_length"] for record in records]
    return {
        "groups": 256,
        "closed_rate": statistics.fmean(record["emission"]["closed"] for record in records),
        "exact_bridge_before_close_rate": statistics.fmean(record["emission"]["exact_bridge_before_close"] for record in records),
        "any_known_bridge_before_close_rate": statistics.fmean(record["emission"]["any_known_bridge_before_close"] for record in records),
        "wrong_domain_bridge_rate": statistics.fmean(record["emission"]["wrong_domain_bridge"] for record in records),
        "bridge_not_adjacent_to_close_rate": statistics.fmean(record["emission"]["bridge_not_adjacent_to_close"] for record in records),
        "multiple_bridge_rate": statistics.fmean(record["emission"]["multiple_bridge"] for record in records),
        "completion_tokens": summarize_lengths(lengths),
        "literal_sid_occurrences": sum(record["literal_sid_occurrences"] for record in records),
        "generation_wall_mean_sec": statistics.fmean(record["generation_wall_sec"] for record in records),
        "artifact": str(artifact),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def proposed_transition_nll(model, context: list[int], target: list[int]) -> float:
    inputs = torch.tensor([context + target[:-1]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=inputs, use_cache=False, logits_to_keep=len(target)).logits[0].float()
        targets = torch.tensor(target, dtype=torch.long, device=model.device)
        losses = -torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return float(losses.mean())


def teacher_with_stages(model, tokenizer, context: list[int], gold_sids: list[str]) -> dict[str, Any]:
    value = cross.teacher_forced_gold(model, tokenizer, context, gold_sids)
    value["mean_nll_a"] = statistics.fmean(-row["A_logp"] for row in value["paths"])
    value["mean_nll_b"] = statistics.fmean(-row["B_logp"] for row in value["paths"])
    value["mean_nll_c"] = statistics.fmean(-row["C_logp"] for row in value["paths"])
    return value


def evaluate_checkpoint(label: str, adapter: Path, manifest: Path, self_artifact: Path, d0_artifact: Path, parts: Path) -> None:
    rank, world, device = setup()
    items = load_manifest(manifest)
    by_group = {item["group_id"]: item for item in items}
    self_cots = {row["group_id"]: row for row in read_jsonl(self_artifact)}
    d0_cots = {row["group_id"]: row for row in read_jsonl(d0_artifact)}
    if set(by_group) != set(self_cots) or set(by_group) != set(d0_cots):
        raise RuntimeError("COT_ARTIFACT_IDENTITY_MISMATCH")
    model, tokenizer = load_model(adapter, device)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("CLOSE_TOKEN_NOT_ATOMIC")
    close_id = int(close_ids[0])
    local = []
    for index, item in enumerate(items):
        if index % world != rank:
            continue
        prompt = list(map(int, item["prompt_token_ids"]))
        domain_ids = list(map(int, item["domain_token_ids"]))
        bridge_ids = list(map(int, item["bridge_token_ids"]))
        self_ids = list(map(int, self_cots[item["group_id"]]["completion_token_ids"]))
        d0_ids = list(map(int, d0_cots[item["group_id"]]["completion_token_ids"]))
        d0_close = close_position(d0_ids, close_id)
        d0_body = d0_ids[:d0_close] if d0_close is not None else d0_ids
        contexts = {
            "self_cot_bare": prompt + self_ids + domain_ids,
            "injected_before": prompt + d0_body + bridge_ids + [close_id] + domain_ids,
        }
        if label == "0":
            contexts["original_after"] = prompt + d0_ids + bridge_ids + domain_ids
        history = cross.history_from_prompt(tokenizer, prompt)
        gold_set = {cross.parse_gold(value) for value in item["gold_sids"]}
        modes = {}
        for mode, context in contexts.items():
            if context[-1] != domain_ids[0]:
                raise RuntimeError("DOMAIN_NOT_FINAL_CONTEXT_TOKEN")
            modes[mode] = {
                "beam": cross.strict_beam(model, tokenizer, context, item["target_domain"], gold_set, history),
                "teacher_forced": teacher_with_stages(model, tokenizer, context, item["gold_sids"]),
            }
        teacher_cot = list(map(int, item["teacher_cot_token_ids"]))
        if teacher_cot[-1] != close_id:
            raise RuntimeError("TEACHER_COT_CLOSE_FAIL")
        transition = proposed_transition_nll(model, prompt + teacher_cot[:-1], bridge_ids + [close_id])
        local.append({
            "label": label,
            "group_id": item["group_id"],
            "target_domain": item["target_domain"],
            "gold_sids": item["gold_sids"],
            "modes": modes,
            "proposed_transition_nll": transition,
        })
        print(f"HELDOUT_EVAL_PROGRESS label={label} rank={rank} sample={index // world + 1}/64", flush=True)
    target = parts / f"eval_{label}"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"rank{rank}.json").write_text(json.dumps({"records": local}, ensure_ascii=False) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def aggregate_mode(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [record for record in records if mode in record["modes"]]
    beams = [beam for record in selected for beam in record["modes"][mode]["beam"]["beams"]]
    valid = [beam for beam in beams if beam["predicted_sid"] is not None]
    copies = Counter(beam["copy_class"] for beam in valid)
    history_gold = Counter(beam["gold_history_class"] for beam in beams)
    teachers = [record["modes"][mode]["teacher_forced"] for record in selected]
    return {
        "groups": len(selected),
        "beam_raw": statistics.fmean(record["modes"][mode]["beam"]["beam_raw"] for record in selected),
        "exact": sum(record["modes"][mode]["beam"]["exact"] for record in selected),
        "ab": sum(record["modes"][mode]["beam"]["ab"] for record in selected),
        "a": sum(record["modes"][mode]["beam"]["a"] for record in selected),
        "invalid": sum(record["modes"][mode]["beam"]["invalid"] for record in selected),
        "history_exact_copy": copies["EXACT_COPY"] / len(valid),
        "history_ab_copy": copies["AB_COPY"] / len(valid),
        "history_a_copy": copies["A_COPY"] / len(valid),
        "novel": copies["NOVEL"] / len(valid),
        "gold_and_history": history_gold["GOLD_AND_HISTORY"] / len(beams),
        "history_not_gold": history_gold["HISTORY_NOT_GOLD"] / len(beams),
        "gold_not_history": history_gold["GOLD_NOT_HISTORY"] / len(beams),
        "mean_gold_abc_nll": statistics.fmean(value["mean_gold_abc_nll"] for value in teachers),
        "nll_a": statistics.fmean(value["mean_nll_a"] for value in teachers),
        "nll_b": statistics.fmean(value["mean_nll_b"] for value in teachers),
        "nll_c": statistics.fmean(value["mean_nll_c"] for value in teachers),
    }


def merge_evaluation(label: str, parts: Path, self_summary: dict[str, Any], output: Path) -> dict[str, Any]:
    records = [record for rank in range(4) for record in json.loads((parts / f"eval_{label}/rank{rank}.json").read_text(encoding="utf-8"))["records"]]
    if len(records) != 256 or len({record["group_id"] for record in records}) != 256:
        raise RuntimeError("EVAL_SHARDS_INCOMPLETE")
    modes = {mode: aggregate_mode(records, mode) for mode in ("self_cot_bare", "injected_before")}
    if label == "0":
        modes["original_after"] = aggregate_mode(records, "original_after")
    per_domain = {
        domain: {mode: aggregate_mode([record for record in records if record["target_domain"] == domain], mode) for mode in modes}
        for domain in DOMAIN_ORDER
    }
    result = {
        "label": label,
        "self_cot": self_summary,
        "modes": modes,
        "per_domain": per_domain,
        "proposed_transition_nll": statistics.fmean(record["proposed_transition_nll"] for record in records),
        "mapping_validity": {"available": False, "reason": "authoritative evaluator mapping unavailable"},
    }
    if label == "0":
        denominator = modes["original_after"]["beam_raw"] - modes["self_cot_bare"]["beam_raw"]
        result["heldout_position_recovery"] = None if denominator <= 0 else (
            modes["injected_before"]["beam_raw"] - modes["self_cot_bare"]["beam_raw"]
        ) / denominator
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--merge-generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--merge-eval", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--self-artifact", required=True)
    parser.add_argument("--d0-artifact")
    parser.add_argument("--output")
    args = parser.parse_args()
    if sum((args.generate, args.merge_generate, args.evaluate, args.merge_eval)) != 1:
        raise SystemExit("choose exactly one operation")
    parts, artifact = Path(args.parts), Path(args.self_artifact)
    if args.generate:
        generate_self_cots(args.label, Path(args.adapter), Path(args.manifest), parts)
    elif args.merge_generate:
        print(json.dumps(merge_self_cots(args.label, parts, artifact), indent=2))
    elif args.evaluate:
        if not args.d0_artifact:
            raise SystemExit("evaluation requires --d0-artifact")
        evaluate_checkpoint(args.label, Path(args.adapter), Path(args.manifest), artifact, Path(args.d0_artifact), parts)
    else:
        if not args.output:
            raise SystemExit("merge-eval requires --output")
        self_summary = merge_self_cots(args.label, parts, artifact)
        print(json.dumps(merge_evaluation(args.label, parts, self_summary, Path(args.output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
