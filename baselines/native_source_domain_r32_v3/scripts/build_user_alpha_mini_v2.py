#!/usr/bin/env python3
"""Build a stratified understand-user mini_v2 task pool."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "/data/models/onereason-8b-pretrain-competition"
SEED = 20260813
REQUIRED_FIELDS = {
    "system", "instruction", "input", "output", "history",
    "data_source", "source_segment", "aux_metadata_json",
}
SID_RE = re.compile(r"<\|(ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
ACTION_TARGETS = {"1-5": 270, "6-10": 360, "11-20": 540, "21-30": 360, "31-40": 180, "41+": 90}
CHAIN_COT_TARGETS = {"2": 45, "3": 165, "4": 75, "5+": 15}
CHAIN_NOCOT_TARGETS = {"2": 135, "3": 495, "4": 225, "5+": 45}


def action_bin(count: int) -> str:
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    if count <= 30:
        return "21-30"
    if count <= 40:
        return "31-40"
    return "41+"


def chain_events(row: dict) -> int | None:
    final = str(row["output"]).split("</think>", 1)[-1].strip()
    try:
        return len(json.loads(final)["logic_chain"]["events"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def chain_bin(count: int | None) -> str | None:
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    if count == 4:
        return "4"
    if count is not None and count >= 5:
        return "5+"
    return None


def load_rows(path: Path, kind: str, tokenizer) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != REQUIRED_FIELDS:
                raise ValueError(f"Unexpected schema at {path}:{line_no}")
            if kind == "action":
                bucket = action_bin(len(SID_RE.findall(str(row["output"]))))
            else:
                bucket = chain_bin(chain_events(row))
            prompt = "".join(str(row[key]) for key in ("system", "instruction", "input"))
            token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            token_count += len(tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"])
            rows.append({"row": row, "line_no": line_no, "bucket": bucket, "tokens": token_count})
    return rows


def select_length_stratified(rows: list[dict], targets: dict[str, int], rng: random.Random) -> tuple[list[dict], dict]:
    by_bucket = defaultdict(list)
    for item in rows:
        if item["bucket"] is not None:
            by_bucket[item["bucket"]].append(item)

    selected = []
    report = {}
    for bucket, target in targets.items():
        candidates = sorted(by_bucket[bucket], key=lambda item: (item["tokens"], item["line_no"]))
        if len(candidates) < target:
            raise ValueError(f"Insufficient rows in {bucket}: {len(candidates)} < {target}")
        quartiles = [[] for _ in range(4)]
        for index, item in enumerate(candidates):
            quartiles[min(3, 4 * index // len(candidates))].append(item)
        base, extra = divmod(target, 4)
        quotas = [base + (1 if index < extra else 0) for index in range(4)]
        chosen = []
        for index, quota in enumerate(quotas):
            pool = quartiles[index][:]
            rng.shuffle(pool)
            chosen.extend(pool[:quota])
        if len(chosen) != target:
            raise AssertionError("Unexpected stratified selection size")
        selected.extend(chosen)
        report[bucket] = {
            "available": len(candidates),
            "selected": target,
            "length_quartile_available": [len(pool) for pool in quartiles],
            "length_quartile_selected": quotas,
            "selected_token_min": min(item["tokens"] for item in chosen),
            "selected_token_max": max(item["tokens"] for item in chosen),
        }
    return selected, report


def write_jsonl(path: Path, items: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as output:
        for item in sorted(items, key=lambda value: value["line_no"]):
            payload = json.dumps(item["row"], ensure_ascii=False, separators=(",", ":")) + "\n"
            output.write(payload)
            digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def token_stats(items: list[dict]) -> dict:
    values = [item["tokens"] for item in items]
    return {"total": sum(values), "mean": round(sum(values) / len(values), 4), "min": min(values), "max": max(values)}


def main() -> None:
    source_dir = Path("/data/lf_data_versions/task_pools") / "\u61c2\u7528\u6237" / "chian\u5f02\u5e38\u6e05\u6d17"
    output_dir = Path("/data/lf_data_versions/task_pools") / "\u61c2\u7528\u6237" / "mini_v2"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    action = load_rows(source_dir / "user_action.jsonl", "action", tokenizer)
    cot = load_rows(source_dir / "user_chain_cot.jsonl", "chain", tokenizer)
    nocot = load_rows(source_dir / "user_chain_nocot.jsonl", "chain", tokenizer)
    rng = random.Random(SEED)
    selected_action, action_report = select_length_stratified(action, ACTION_TARGETS, rng)
    selected_cot, cot_report = select_length_stratified(cot, CHAIN_COT_TARGETS, rng)
    selected_nocot, nocot_report = select_length_stratified(nocot, CHAIN_NOCOT_TARGETS, rng)

    output_dir.mkdir(parents=True)
    files = {
        "user_action.jsonl": selected_action,
        "user_chain_cot.jsonl": selected_cot,
        "user_chain_nocot.jsonl": selected_nocot,
    }
    file_hashes = {name: write_jsonl(output_dir / name, rows) for name, rows in files.items()}
    merged = selected_action + selected_cot + selected_nocot
    file_hashes["understand_user_mini_v2.jsonl"] = write_jsonl(output_dir / "understand_user_mini_v2.jsonl", merged)

    source_counts = {"user_action": len(action), "user_chain_cot": len(cot), "user_chain_nocot": len(nocot)}
    selected_counts = {"user_action": len(selected_action), "user_chain_cot": len(selected_cot), "user_chain_nocot": len(selected_nocot)}
    manifest = {
        "kind": "unregistered_task_pool",
        "name": "mini_v2",
        "task": "understand_user",
        "source_directory": str(source_dir),
        "source_files": ["user_action.jsonl", "user_chain_cot.jsonl", "user_chain_nocot.jsonl"],
        "excluded_nontraining_files": ["removed_samples.jsonl", "understand_user_clean.jsonl"],
        "seed": SEED,
        "selection": {
            "user_action": {"stratification": "final answer complete SID count", "targets": ACTION_TARGETS, "bins": action_report},
            "user_chain_cot": {"stratification": "final logic_chain.events count", "targets": CHAIN_COT_TARGETS, "bins": cot_report},
            "user_chain_nocot": {"stratification": "final logic_chain.events count", "targets": CHAIN_NOCOT_TARGETS, "bins": nocot_report},
            "within_each_bin": "raw prompt plus output token-length quartiles; deterministic random sampling",
        },
        "counts": {"source": source_counts, "selected": selected_counts, "total_selected": len(merged)},
        "tokens": {
            "model": MODEL,
            "method": "AutoTokenizer add_special_tokens=False; prompt=system+instruction+input; total=prompt+output; excludes chat-template fixed overhead",
            "by_source_segment": {"user_action": token_stats(selected_action), "user_chain_cot": token_stats(selected_cot), "user_chain_nocot": token_stats(selected_nocot)},
            "total": token_stats(merged),
        },
        "files": file_hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
