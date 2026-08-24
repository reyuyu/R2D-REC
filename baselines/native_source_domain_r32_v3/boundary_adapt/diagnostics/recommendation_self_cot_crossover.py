"""Phase 1.3: 20-group Self-CoT generator x decoder crossover.

The expensive stages are deliberately resumable.  Teacher-CoT anchors are
copied from Phase 1.2 and are never regenerated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Callable

import torch

RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]
from boundary_adapt.bridge_inside_sft.common import BASE, DOMAIN, close_position, token_ids_sha  # noqa: E402
from boundary_adapt.bridge_inside_sft.evaluate_transition import (  # noqa: E402
    GENERATION_BATCH, generate_batch as generate_self_batch, load_model, seed_for_batch,
)
from boundary_adapt.diagnostics.controlled_generator_decoder_crossover import strict_beam  # noqa: E402
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import history_from_prompt  # noqa: E402
from boundary_adapt.diagnostics.recommendation_root_cause_phase1_1 import (  # noqa: E402
    build_manifold, sid,
)

PHASE12 = RUNTIME / "boundary_adapt/results/recommendation_official_prompt_bare_crossover_40g"
OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_self_cot_crossover_20g"
PARTS = OUTPUT / "parts"
MANIFEST = OUTPUT / "manifest_20.json"
PROMPT_AUDIT = OUTPUT / "prompt_audit.json"
SELF_RECORDS = OUTPUT / "self_cot_records.jsonl"
DECODER_RECORDS = OUTPUT / "decoder_records.jsonl"
MODEL_PATHS = {
    "Beta": Path("/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"),
    "Step900": Path("/root/data_checkpoints_backup_20260824/outputs/boundary_adapt/continuation_from300_to1500/checkpoint-900"),
}
EXPECTED_SHA = {
    "Beta": "4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3",
    "Step900": "ea395df041ef7bc13d5e5e54aec7a0d376b8f6ae4ae064eb6a7d89dd3c7cdffa",
}
GENERATORS = ("Beta", "Step900")
DECODERS = ("Beta", "Step900")
DOMAINS = ("video", "prod", "ad", "living")
CUTS = (1, 5, 10, 32)
SELECTION_SALT = "recommendation_self_cot_crossover_20g_v1"
BOOTSTRAP_SEED = 20260825
SID_RE = __import__("re").compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"], text=True).strip()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(group_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}|{group_id}".encode()).hexdigest()


def item_bucket(item: dict[str, Any]) -> tuple[bool, bool]:
    return bool(item["gold_sid_in_history"]), int(item["K"]) >= 2


def select_domain(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for bucket in ((True, False), (True, True), (False, False), (False, True)):
        choices = sorted((row for row in items if item_bucket(row) == bucket), key=lambda row: stable_key(row["group_id"]))
        if choices:
            selected.append(choices[0])
    remaining = sorted((row for row in items if row not in selected), key=lambda row: stable_key(row["group_id"]))
    selected.extend(remaining[:5 - len(selected)])
    if len(selected) != 5:
        raise RuntimeError("DOMAIN_SELECTION_FAILED")
    return selected


def prepare() -> None:
    from transformers import AutoTokenizer

    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit40 = json.loads((PHASE12 / "prompt_audit.json").read_text(encoding="utf-8"))
    if len(audit40["items"]) != 40 or not audit40["prompt_audit_pass"] or not audit40["history_sid_sequence_parity"]:
        raise RuntimeError("PHASE12_PROMPT_CONTRACT_FAIL")
    actual_sha = {label: file_sha(path / "adapter_model.safetensors") for label, path in MODEL_PATHS.items()}
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError(f"ADAPTER_SHA_MISMATCH actual={actual_sha}")
    selected = []
    for domain in DOMAINS:
        selected.extend(select_domain([row for row in audit40["items"] if row["domain"] == domain]))
    if len(selected) != 20 or len({row["group_id"] for row in selected}) != 20:
        raise RuntimeError("SELECTED_GROUP_CONTRACT_FAIL")
    if Counter(row["domain"] for row in selected) != Counter({domain: 5 for domain in DOMAINS}):
        raise RuntimeError("SELECTED_DOMAIN_BALANCE_FAIL")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    gates = []
    items = []
    for order, row in enumerate(selected):
        old_sids = [match.group(0) for match in SID_RE.finditer(row["old_user"])]
        new_sids = [match.group(0) for match in SID_RE.finditer(row["new_user"])]
        old_sid_token_ids = [tokenizer.encode(value, add_special_tokens=False) for value in old_sids]
        new_sid_token_ids = [tokenizer.encode(value, add_special_tokens=False) for value in new_sids]
        gate = {
            "group_id": row["group_id"], "domain": row["domain"],
            "history_sid_sequence_parity": old_sids == new_sids,
            "history_sid_token_bytes_parity": old_sid_token_ids == new_sid_token_ids,
            "history_sid_count_old": len(old_sids), "history_sid_count_official": len(new_sids),
            "official_prompt_token_length": len(row["official_prompt_token_ids"]),
            "official_prompt_within_production_limit": len(row["official_prompt_token_ids"]) <= 8192,
            "phase12_gate_pass": all(row["gates"][key] for key in (
                "system_domain_correct", "question_domain_correct", "history_sid_sequence_parity",
            )),
        }
        gates.append(gate)
        items.append({**row, "selection_order": order, "selection_key_sha256": stable_key(row["group_id"])})
    prompt_pass = all(
        row["history_sid_sequence_parity"] and row["history_sid_token_bytes_parity"]
        and row["official_prompt_within_production_limit"] and row["phase12_gate_pass"]
        for row in gates
    )
    if not prompt_pass:
        raise RuntimeError("PROMPT_AUDIT_FAIL")
    counts = {
        "domain": dict(Counter(row["domain"] for row in items)),
        "gold_in_history": sum(bool(row["gold_sid_in_history"]) for row in items),
        "gold_not_in_history": sum(not row["gold_sid_in_history"] for row in items),
        "K1": sum(int(row["K"]) == 1 for row in items),
        "K2PLUS": sum(int(row["K"]) >= 2 for row in items),
    }
    manifest = {
        "source_commit": source_commit(), "selection_salt": SELECTION_SALT,
        "source_manifest": str(PHASE12 / "prompt_audit.json"), "groups": 20,
        "prompt_style": "OFFICIAL_DOMAIN_SPECIFIC", "counts": counts,
        "model_paths": {key: str(value) for key, value in MODEL_PATHS.items()},
        "adapter_sha256": actual_sha, "items": items,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROMPT_AUDIT.write_text(json.dumps({
        "prompt_style": "OFFICIAL_DOMAIN_SPECIFIC", "prompt_audit_pass": prompt_pass,
        "history_sid_sequence_parity": prompt_pass, "history_sid_token_bytes_parity": prompt_pass,
        "parity_count": "20/20",
        "gates": gates,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SELECTED_GROUPS=20\nDOMAIN_COUNTS={counts['domain']}\nPROMPT_AUDIT_PASS=YES")


def setup_rank() -> tuple[int, int, str]:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("PHASE13_REQUIRES_4_GPU_RANKS")
    torch.cuda.set_device(rank)
    return rank, world, f"cuda:{rank}"


def literal_sid_health(tokenizer, completion: list[int], target_domain: str, history: dict[str, Any]) -> dict[str, Any]:
    text = tokenizer.decode(completion, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    values = [(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))) for m in SID_RE.finditer(text)]
    history_set = history["unique_set"]
    return {
        "literal_any_sid_count": len(values),
        "literal_target_domain_sid_count": sum(value[0] == target_domain for value in values),
        "literal_history_sid_count": sum(value in history_set for value in values),
        "any_history_sid_mention": any(value in history_set for value in values),
        "any_target_domain_sid_mention": any(value[0] == target_domain for value in values),
    }


def generate(generator: str) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(PROMPT_AUDIT.read_text(encoding="utf-8"))
    if not audit["prompt_audit_pass"]:
        raise RuntimeError("GPU_BLOCKED_BY_PROMPT_AUDIT")
    rank, world, device = setup_rank()
    model, tokenizer = load_model(MODEL_PATHS[generator], device)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("CLOSE_TOKEN_NOT_ATOMIC")
    close_id = int(close_ids[0])
    local_items = [row for index, row in enumerate(manifest["items"]) if index % world == rank]
    records = []
    for start in range(0, len(local_items), GENERATION_BATCH):
        batch = local_items[start:start + GENERATION_BATCH]
        groups = [row["group_id"] for row in batch]
        seed = seed_for_batch(groups)
        prompts = [list(map(int, row["official_prompt_token_ids"])) for row in batch]
        completions, wall = generate_self_batch(model, tokenizer, prompts, close_id, seed)
        for row, prompt, completion in zip(batch, prompts, completions):
            close = close_position(completion, close_id)
            history = history_from_prompt(tokenizer, prompt)
            records.append({
                "generator": generator, "group_id": row["group_id"], "domain": row["domain"],
                "K": row["K"], "gold_sid_in_history": row["gold_sid_in_history"],
                "seed": seed, "seed_scope": "fixed_rank_local_batch", "seed_batch_groups": groups,
                "prompt_token_ids_sha256": token_ids_sha(prompt),
                "completion_token_ids": completion, "completion_sha256": token_ids_sha(completion),
                "closed": close is not None, "close_position": close,
                "token_length": len(completion), "last_64_token_ids": completion[-64:],
                "last_32_token_ids": completion[-32:],
                "batch_generation_wall_sec": wall, "generation_wall_sec_per_sample": wall / len(batch),
                "generation_batch_size": len(batch),
                **literal_sid_health(tokenizer, completion, row["domain"], history),
            })
        print(f"SELF_COT_PROGRESS generator={generator} rank={rank} done={min(start + len(batch), len(local_items))}/5", flush=True)
    target = PARTS / f"self_{generator}_rank{rank}.jsonl"
    write_jsonl(target, records)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] if low == high else ordered[low] * (high - pos) + ordered[high] * (pos - low)


def health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [row["token_length"] for row in rows]
    return {
        "N": len(rows), "closed_numerator": sum(row["closed"] for row in rows),
        "closed_rate": statistics.fmean(row["closed"] for row in rows),
        "token_length_mean": statistics.fmean(lengths), "token_length_p50": percentile(lengths, .5),
        "token_length_p90": percentile(lengths, .9), "token_length_max": max(lengths),
        "literal_any_sid_count": sum(row["literal_any_sid_count"] for row in rows),
        "literal_target_domain_sid_count": sum(row["literal_target_domain_sid_count"] for row in rows),
        "literal_history_sid_count": sum(row["literal_history_sid_count"] for row in rows),
        "any_history_sid_mention_numerator": sum(row["any_history_sid_mention"] for row in rows),
        "any_history_sid_mention_rate": statistics.fmean(row["any_history_sid_mention"] for row in rows),
        "any_target_domain_sid_mention_numerator": sum(row["any_target_domain_sid_mention"] for row in rows),
        "any_target_domain_sid_mention_rate": statistics.fmean(row["any_target_domain_sid_mention"] for row in rows),
        "generation_wall_per_sample_mean": statistics.fmean(row["generation_wall_sec_per_sample"] for row in rows),
    }


def merge_self() -> None:
    rows = [row for generator in GENERATORS for rank in range(4) for row in read_jsonl(PARTS / f"self_{generator}_rank{rank}.jsonl")]
    if len(rows) != 40 or len({(row["generator"], row["group_id"]) for row in rows}) != 40:
        raise RuntimeError("SELF_COT_GENERATION_CONTRACT_FAIL")
    rows.sort(key=lambda row: (GENERATORS.index(row["generator"]), row["group_id"]))
    write_jsonl(SELF_RECORDS, rows)
    payload = {generator: health([row for row in rows if row["generator"] == generator]) for generator in GENERATORS}
    (OUTPUT / "self_cot_health.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Self-CoT Health", ""]
    for generator in GENERATORS:
        value = payload[generator]
        lines += [f"## {generator}", "", f"- Closed: {value['closed_numerator']}/20 ({value['closed_rate']:.1%})",
                  f"- Length mean/p50/p90/max: {value['token_length_mean']:.1f} / {value['token_length_p50']:.1f} / {value['token_length_p90']:.1f} / {value['token_length_max']}",
                  f"- History SID mention: {value['any_history_sid_mention_numerator']}/20 ({value['any_history_sid_mention_rate']:.1%})",
                  f"- Target SID mention: {value['any_target_domain_sid_mention_numerator']}/20 ({value['any_target_domain_sid_mention_rate']:.1%})", ""]
    (OUTPUT / "self_cot_health.md").write_text("\n".join(lines), encoding="utf-8")
    print("SELF_COT_GENERATIONS=40")


def decode(decoder: str) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_group = {row["group_id"]: row for row in manifest["items"]}
    self_rows = read_jsonl(SELF_RECORDS)
    rank, world, device = setup_rank()
    model, tokenizer = load_model(MODEL_PATHS[decoder], device)
    local = []
    cases = [(generator, row) for generator in GENERATORS for row in self_rows if row["generator"] == generator]
    for index, (generator, cot_row) in enumerate(cases):
        if index % world != rank:
            continue
        item = by_group[cot_row["group_id"]]
        base = {
            "generator": generator, "decoder": decoder, "group_id": item["group_id"],
            "domain": item["domain"], "K": item["K"], "gold_sid_in_history": item["gold_sid_in_history"],
            "self_cot_closed": cot_row["closed"], "self_cot_sha256": cot_row["completion_sha256"],
        }
        if not cot_row["closed"]:
            local.append({**base, "status": "NON_CLOSED", "beams": []})
        else:
            prompt = list(map(int, item["official_prompt_token_ids"]))
            cot = list(map(int, cot_row["completion_token_ids"]))
            domain_ids = tokenizer.encode(DOMAIN[item["domain"]], add_special_tokens=False)
            if len(domain_ids) != 1:
                raise RuntimeError("DOMAIN_TOKEN_NOT_ATOMIC")
            history = history_from_prompt(tokenizer, prompt)
            gold_set = {sid(value) for value in item["all_gold_sids"]}
            result = strict_beam(model, tokenizer, prompt + cot + domain_ids, item["domain"], gold_set, history)
            local.append({**base, "status": "OK", **result})
        print(f"DECODER_PROGRESS decoder={decoder} rank={rank} case={index // world + 1}/10 status={local[-1]['status']}", flush=True)
    write_jsonl(PARTS / f"decode_{decoder}_rank{rank}.jsonl", local)


def merge_decode() -> None:
    rows = [row for decoder in DECODERS for rank in range(4) for row in read_jsonl(PARTS / f"decode_{decoder}_rank{rank}.jsonl")]
    if len(rows) != 80 or len({(row["generator"], row["decoder"], row["group_id"]) for row in rows}) != 80:
        raise RuntimeError("NEW_DECODER_CASE_CONTRACT_FAIL")
    if any(row["status"] == "OK" and len(row["beams"]) != 32 for row in rows):
        raise RuntimeError("BEAM32_CONTRACT_FAIL")
    rows.sort(key=lambda row: (GENERATORS.index(row["generator"]), DECODERS.index(row["decoder"]), row["group_id"]))
    write_jsonl(DECODER_RECORDS, rows)
    print("NEW_BEAM_CASES=80")


def group_values(row: dict[str, Any], item: dict[str, Any], manifold: dict[str, set[tuple]]) -> dict[str, float]:
    predictions = [sid(beam["predicted_sid"]) if beam["predicted_sid"] else None for beam in row.get("beams", [])]
    golds = [sid(value) for value in item["all_gold_sids"]]
    def rank(prefix: int) -> int | None:
        return next((i for i, value in enumerate(predictions, 1) if value is not None and any(value[:prefix] == gold[:prefix] for gold in golds)), None)
    exact, ab, a = rank(4), rank(3), rank(2)
    history_count = sum(beam["copy_class"] == "EXACT_COPY" for beam in row.get("beams", []))
    classes = Counter()
    for beam, value in zip(row.get("beams", []), predictions):
        if value is None:
            classes["INVALID"] += 1
        elif beam["copy_class"] == "EXACT_COPY":
            classes["HISTORY_COPY"] += 1
        elif value in manifold[value[0]]:
            classes["SEEN_NOVEL"] += 1
        else:
            classes["UNSEEN_RECOMBINATION"] += 1
    classes["INVALID"] += 32 - len(predictions)
    return {
        **{f"hit{cut}": float(exact is not None and exact <= cut) for cut in CUTS},
        "mrr": 0.0 if exact is None else 1.0 / exact,
        "rank": None if exact is None else float(exact),
        "ab32": float(ab is not None), "a32": float(a is not None),
        "history_fraction": history_count / 32, "top1_is_history": float(bool(row.get("beams")) and row["beams"][0]["copy_class"] == "EXACT_COPY"),
        "unique_a": float(len({value[:2] for value in predictions if value is not None})),
        "unique_ab": float(len({value[:3] for value in predictions if value is not None})),
        "unique_abc": float(len({value for value in predictions if value is not None})),
        **{f"class_{key}": classes[key] / 32 for key in ("HISTORY_COPY", "SEEN_NOVEL", "UNSEEN_RECOMBINATION", "INVALID")},
    }


def summarize_cell(rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]], manifold, closed_only: bool) -> dict[str, Any]:
    selected = [row for row in rows if row["status"] == "OK"] if closed_only else rows
    values = [(row, group_values(row, manifest[row["group_id"]], manifold)) for row in selected]
    if not values:
        return {"N": 0}
    metrics = {"N": len(values), "closed_numerator": sum(row["status"] == "OK" for row, _ in values)}
    for field in ("hit1", "hit5", "hit10", "hit32", "mrr", "ab32", "a32", "history_fraction", "top1_is_history", "unique_a", "unique_ab", "unique_abc", "class_HISTORY_COPY", "class_SEEN_NOVEL", "class_UNSEEN_RECOMBINATION", "class_INVALID"):
        metrics[field] = statistics.fmean(value[field] for _, value in values)
        if field in ("hit1", "hit5", "hit10", "hit32", "ab32", "a32", "top1_is_history"):
            metrics[f"{field}_numerator"] = int(sum(value[field] for _, value in values))
    hit_ranks = [value["rank"] for _, value in values if value["rank"] is not None]
    metrics["mean_best_gold_rank_hits_only"] = statistics.fmean(hit_ranks) if hit_ranks else None
    metrics["per_group"] = {row["group_id"]: value for row, value in values}
    return metrics


def add_strata(cell_rows, manifest, manifold) -> dict[str, Any]:
    result = {}
    for name, predicate in (
        ("GoldSIDInHistory", lambda item: item["gold_sid_in_history"]),
        ("GoldSIDNotInHistory", lambda item: not item["gold_sid_in_history"]),
        ("K1", lambda item: item["K"] == 1), ("K2PLUS", lambda item: item["K"] >= 2),
    ):
        subset = [row for row in cell_rows if predicate(manifest[row["group_id"]])]
        value = summarize_cell(subset, manifest, manifold, False)
        result[name] = {key: value[key] for key in ("N", "hit32", "hit32_numerator", "mrr", "history_fraction")}
    return result


def bootstrap_interaction(cells: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    group_ids = sorted(cells["GB_DB"]["per_group"])
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(1000):
        draw = [rng.choice(group_ids) for _ in group_ids]
        mean = lambda cell: statistics.fmean(cells[cell]["per_group"][group][field] for group in draw)
        samples.append((mean("G9_D9") - mean("G9_DB")) - (mean("GB_D9") - mean("GB_DB")))
    estimate = (cells["G9_D9"][field] - cells["G9_DB"][field]) - (cells["GB_D9"][field] - cells["GB_DB"][field])
    return {"estimate": estimate,
            "ci95": [percentile(samples, .025), percentile(samples, .975)], "replicates": 1000, "seed": BOOTSTRAP_SEED}


def effect(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    fields = ("mrr", "hit32", "ab32", "a32", "history_fraction")
    return {field: left[field] - right[field] for field in fields}


def label_order(beta: float, step: float) -> str:
    return "Beta > Step900" if beta > step else "Step900 > Beta" if step > beta else "Beta = Step900"


def classify_root(cells: dict[str, dict[str, Any]]) -> tuple[str, str, str, str]:
    decoder = statistics.fmean(abs(cells[key]["mrr"] - cells[other]["mrr"]) for key, other in (("GB_D9", "GB_DB"), ("G9_D9", "G9_DB")))
    generator = statistics.fmean(abs(cells[key]["mrr"] - cells[other]["mrr"]) for key, other in (("G9_DB", "GB_DB"), ("G9_D9", "GB_D9")))
    interaction = abs((cells["G9_D9"]["mrr"] - cells["G9_DB"]["mrr"]) - (cells["GB_D9"]["mrr"] - cells["GB_DB"]["mrr"]))
    if generator < .01 and decoder < .01:
        return "MODEL_SIDE_LOCAL_METRICS_INSUFFICIENT", "Neither local MRR main effect is large.", "No stable secondary local signal.", "OFFICIAL_EVALUATOR_REWARD_OR_TEST_DISTRIBUTION"
    if interaction > max(generator, decoder):
        return "GENERATOR_DECODER_SPECIALIZATION", f"MRR interaction={interaction:.6f} dominates main effects.", f"Generator magnitude={generator:.6f}; decoder magnitude={decoder:.6f}.", "CONTROLLED_LARGER_SAMPLE_CONFIRMATION"
    if generator > decoder * 1.5:
        return "SELF_COT_GENERATOR_DEGRADATION", f"Generator MRR effect magnitude={generator:.6f}.", f"Decoder magnitude={decoder:.6f}.", "SELF_COT_DISTRIBUTION_MECHANISM"
    if decoder > generator * 1.5:
        return "DECODER_RANKING_DEGRADATION", f"Decoder MRR effect magnitude={decoder:.6f}.", f"Generator magnitude={generator:.6f}.", "DECODER_RANKING_MECHANISM"
    return "MIXED_GENERATOR_DECODER_EFFECT", f"Generator magnitude={generator:.6f}; decoder magnitude={decoder:.6f}.", f"Interaction magnitude={interaction:.6f}.", "CONTROLLED_LARGER_SAMPLE_CONFIRMATION"


def finalize() -> dict[str, Any]:
    manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = {row["group_id"]: row for row in manifest_payload["items"]}
    new_rows = read_jsonl(DECODER_RECORDS)
    old_rows = [row for row in read_jsonl(PHASE12 / "records.jsonl") if row["model"] in DECODERS and row["group_id"] in manifest]
    if len(old_rows) != 40 or any(row["mode"] != "official_bare" for row in old_rows):
        raise RuntimeError("TEACHER_ANCHOR_REUSE_FAIL")
    teacher_rows = [{**row, "generator": "Teacher", "decoder": row["model"], "status": "OK"} for row in old_rows]
    manifold, manifold_stats = build_manifold()
    mapping = {"GT_DB": ("Teacher", "Beta"), "GT_D9": ("Teacher", "Step900"),
               "GB_DB": ("Beta", "Beta"), "GB_D9": ("Beta", "Step900"),
               "G9_DB": ("Step900", "Beta"), "G9_D9": ("Step900", "Step900")}
    cells, closed_cells, strata = {}, {}, {}
    all_rows = teacher_rows + new_rows
    for key, (generator, decoder) in mapping.items():
        rows = [row for row in all_rows if row["generator"] == generator and row["decoder"] == decoder]
        cells[key] = summarize_cell(rows, manifest, manifold, False)
        closed_cells[key] = summarize_cell(rows, manifest, manifold, True)
        strata[key] = add_strata(rows, manifest, manifold)
    effects = {
        "DECODER_EFFECT_ON_TEACHER": effect(cells["GT_D9"], cells["GT_DB"]),
        "DECODER_EFFECT_ON_BETA_COT": effect(cells["GB_D9"], cells["GB_DB"]),
        "DECODER_EFFECT_ON_900_COT": effect(cells["G9_D9"], cells["G9_DB"]),
        "GENERATOR_EFFECT_UNDER_BETA_DECODER": effect(cells["G9_DB"], cells["GB_DB"]),
        "GENERATOR_EFFECT_UNDER_900_DECODER": effect(cells["G9_D9"], cells["GB_D9"]),
        "BETA_SELF_VS_TEACHER_UNDER_BETA_DECODER": effect(cells["GB_DB"], cells["GT_DB"]),
        "STEP900_SELF_VS_TEACHER_UNDER_BETA_DECODER": effect(cells["G9_DB"], cells["GT_DB"]),
        "BETA_SELF_VS_TEACHER_UNDER_900_DECODER": effect(cells["GB_D9"], cells["GT_D9"]),
        "STEP900_SELF_VS_TEACHER_UNDER_900_DECODER": effect(cells["G9_D9"], cells["GT_D9"]),
    }
    interactions = {field: bootstrap_interaction(cells, field) for field in ("mrr", "hit32", "history_fraction")}
    own_orders = {field: label_order(cells["GB_DB"][field], cells["G9_D9"][field]) for field in ("hit32", "mrr", "ab32", "a32")}
    alignment = "YES" if own_orders["mrr"] == "Beta > Step900" and sum(own_orders[field] == "Beta > Step900" for field in ("ab32", "a32")) >= 1 else "PARTIAL" if own_orders["mrr"] == "Beta > Step900" or sum(own_orders[field] == "Beta > Step900" for field in ("ab32", "a32")) >= 1 else "NO"
    root, primary, secondary, next_variable = classify_root(cells)
    health_payload = json.loads((OUTPUT / "self_cot_health.json").read_text())
    result = {
        "source_commit": source_commit(), "adapter_sha256": EXPECTED_SHA, "groups": 20,
        "domain_counts": manifest_payload["counts"]["domain"], "prompt_style": "OFFICIAL_DOMAIN_SPECIFIC",
        "prompt_audit_pass": True, "history_sid_sequence_parity": True,
        "self_cot_generations": 40, "new_beam_cases": 80,
        "self_cot_health": health_payload, "cells_all20": cells, "cells_closed_only": closed_cells,
        "stratified_all20": strata, "effects_all20": effects, "interaction_bootstrap_all20": interactions,
        "train_manifold_stats": manifold_stats, "teacher_anchor_source": str(PHASE12 / "records.jsonl"),
        "teacher_anchor_gpu_regenerated": False, "own_pair_orders": own_orders,
        "external_order": "Beta > Step900", "own_pair_alignment_with_external": alignment,
        "training_started": False, "optimizer_steps": 0, "external_eval": False,
        "root_class": root, "primary_causal_signal": primary, "secondary_signal": secondary,
        "conclusion": f"N=20 descriptive crossover: {root}. ALL20 counts non-closed generations as decoder failures; CLOSED_ONLY is reported separately.",
        "next_root_variable": next_variable, "next_experiment_needed": "NO_AUTOMATIC_FOLLOWUP",
    }
    (OUTPUT / "crossover_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    (OUTPUT / "crossover_summary.md").write_text(markdown, encoding="utf-8")
    (OUTPUT / "CHATGPT_REVIEW.txt").write_text(markdown, encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Recommendation Self-CoT Crossover Phase 1.3", "", "This is a descriptive 20-group mechanism probe, not a significance experiment.", "",
             "## Self-CoT health", "", "| Generator | Closed | Mean len | P50 | P90 | History mention | Target mention |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label in GENERATORS:
        value = result["self_cot_health"][label]
        lines.append(f"| {label} | {value['closed_numerator']}/20 | {value['token_length_mean']:.1f} | {value['token_length_p50']:.1f} | {value['token_length_p90']:.1f} | {value['any_history_sid_mention_rate']:.3f} | {value['any_target_domain_sid_mention_rate']:.3f} |")
    lines += ["", "## Decoder matrix (ALL20)", "", "Non-closed Self-CoTs remain in the denominator as failures; no synthetic close was appended.", "",
              "| Cell | Closed/N | Hit@32 | MRR | AB@32 | A@32 | History fraction |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key in ("GT_DB", "GT_D9", "GB_DB", "GB_D9", "G9_DB", "G9_D9"):
        value = result["cells_all20"][key]
        lines.append(f"| {key} | {value['closed_numerator']}/{value['N']} | {value['hit32']:.3f} | {value['mrr']:.6f} | {value['ab32']:.3f} | {value['a32']:.3f} | {value['history_fraction']:.4f} |")
    lines += ["", "## Interpretation", "", f"- Root class: `{result['root_class']}`", f"- Primary: {result['primary_causal_signal']}", f"- Secondary: {result['secondary_signal']}", f"- Own-pair external alignment: {result['own_pair_alignment_with_external']}", "",
              "## Contracts", "", "- Training started: NO", "- Optimizer steps: 0", "- External evaluation: NO", "- Teacher anchors regenerated: NO", "- Bootstrap: 1,000 group-level resamples"]
    return "\n".join(lines) + "\n"


def terminal_report(result: dict[str, Any]) -> str:
    counts, healths, cells = json.loads(MANIFEST.read_text())["counts"], result["self_cot_health"], result["cells_all20"]
    lines = [f"SOURCE_COMMIT={result['source_commit']}", f"BETA_SHA={EXPECTED_SHA['Beta']}", f"STEP900_SHA={EXPECTED_SHA['Step900']}", "", "GROUPS=20",
             f"DOMAIN_COUNTS={counts['domain']}", f"GOLD_IN_HISTORY={counts['gold_in_history']}", f"GOLD_NOT_IN_HISTORY={counts['gold_not_in_history']}", f"K1={counts['K1']}", f"K2PLUS={counts['K2PLUS']}", "",
             "PROMPT_STYLE=OFFICIAL_DOMAIN_SPECIFIC", "PROMPT_AUDIT_PASS=YES", "HISTORY_SID_SEQUENCE_PARITY=PASS", "", "SELF_COT_GENERATIONS=40", "NEW_BEAM_CASES=80", "", "--- Self-CoT Health ---", ""]
    for label in GENERATORS:
        prefix, value = label.upper(), healths[label]
        lines += [f"{prefix}_CLOSED_RATE={value['closed_rate']:.6f}", f"{prefix}_COT_MEAN_LEN={value['token_length_mean']:.3f}", f"{prefix}_COT_P50={value['token_length_p50']:.3f}", f"{prefix}_COT_P90={value['token_length_p90']:.3f}", f"{prefix}_HISTORY_SID_MENTION_RATE={value['any_history_sid_mention_rate']:.6f}", f"{prefix}_TARGET_SID_MENTION_RATE={value['any_target_domain_sid_mention_rate']:.6f}", ""]
    lines += ["--- Frozen Teacher Anchor ---", "", f"GT_DB_HIT32={cells['GT_DB']['hit32']:.6f}", f"GT_D9_HIT32={cells['GT_D9']['hit32']:.6f}", f"GT_DB_MRR={cells['GT_DB']['mrr']:.6f}", f"GT_D9_MRR={cells['GT_D9']['mrr']:.6f}", "", "--- 2x2 Self-CoT Matrix ---", ""]
    for metric, suffix in (("hit32", "HIT32"), ("mrr", "MRR"), ("ab32", "AB32"), ("a32", "A32"), ("history_fraction", "HISTORY_FRACTION")):
        lines.extend(f"{cell}_{suffix}={cells[cell][metric]:.6f}" for cell in ("GB_DB", "GB_D9", "G9_DB", "G9_D9"))
        lines.append("")
    effects, interaction = result["effects_all20"], result["interaction_bootstrap_all20"]["mrr"]
    lines += ["--- Effects ---", "", f"DECODER_EFFECT_ON_TEACHER_MRR={effects['DECODER_EFFECT_ON_TEACHER']['mrr']:.6f}", f"DECODER_EFFECT_ON_BETA_COT_MRR={effects['DECODER_EFFECT_ON_BETA_COT']['mrr']:.6f}", f"DECODER_EFFECT_ON_900_COT_MRR={effects['DECODER_EFFECT_ON_900_COT']['mrr']:.6f}", f"GENERATOR_EFFECT_UNDER_BETA_DECODER_MRR={effects['GENERATOR_EFFECT_UNDER_BETA_DECODER']['mrr']:.6f}", f"GENERATOR_EFFECT_UNDER_900_DECODER_MRR={effects['GENERATOR_EFFECT_UNDER_900_DECODER']['mrr']:.6f}", f"INTERACTION_MRR={interaction['estimate']:.6f}", f"INTERACTION_MRR_CI95={interaction['ci95']}", ""]
    lines += [f"OWN_PAIR_HIT32_ORDER={result['own_pair_orders']['hit32']}", f"OWN_PAIR_MRR_ORDER={result['own_pair_orders']['mrr']}", f"OWN_PAIR_AB_ORDER={result['own_pair_orders']['ab32']}", f"OWN_PAIR_A_ORDER={result['own_pair_orders']['a32']}", "EXTERNAL_ORDER=Beta > Step900", f"OWN_PAIR_ALIGNMENT_WITH_EXTERNAL={result['own_pair_alignment_with_external']}", "", "TRAINING_STARTED=NO", "OPTIMIZER_STEPS=0", "EXTERNAL_EVAL=NO", "", f"ROOT_CLASS={result['root_class']}", f"PRIMARY_CAUSAL_SIGNAL={result['primary_causal_signal']}", f"SECONDARY_SIGNAL={result['secondary_signal']}", f"CONCLUSION={result['conclusion']}", f"NEXT_ROOT_VARIABLE={result['next_root_variable']}", f"NEXT_EXPERIMENT_NEEDED={result['next_experiment_needed']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "generate", "merge-self", "decode", "merge-decode", "finalize", "report"))
    parser.add_argument("--model", choices=("Beta", "Step900"))
    args = parser.parse_args()
    if args.action in ("generate", "decode") and not args.model:
        raise SystemExit(f"{args.action} requires --model")
    if args.action == "prepare": prepare()
    elif args.action == "generate": generate(args.model)
    elif args.action == "merge-self": merge_self()
    elif args.action == "decode": decode(args.model)
    elif args.action == "merge-decode": merge_decode()
    else:
        result = finalize() if args.action == "finalize" else json.loads((OUTPUT / "crossover_summary.json").read_text())
        print(terminal_report(result))


if __name__ == "__main__":
    main()
