"""Inference-only fixed-CoT Boundary Adaptation checkpoint sweep.

This deliberately reuses the 12 x 4 frozen step-0 CoTs captured by the
history-copy audit.  It never samples CoTs, constructs an optimizer, or
modifies an adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RUNTIME = Path("/data/GRPO")
sys.path.insert(0, str(RUNTIME / "scripts"))
from grpo_model import BASE, encode_prompt, generate_batch  # noqa: E402
from grpo_sid import think_reward  # noqa: E402

STEP0_ADAPTER = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
FORMAL = Path("/data/outputs/boundary_adapt/formal_300step")
PROBES = Path(
    "/data/GRPO/runs/GR-REC-THINK-COMPOSITE-INTEREST-V1-ABC3-"
    "FORMAL716-20260823/probes.jsonl"
)
BATA_SOURCE = Path(
    "/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl"
)
HISTORICAL_AUDIT = Path(
    "/data/tmp_inspect/onereason-multitask-sft/baselines/native_source_domain_r32_v3/"
    "grpo/results/gr_rec_think_composite_interest_v1_history_sid_copy_audit_20260823.json"
)
HISTORICAL_PROBE1 = Path(
    "/data/tmp_inspect/onereason-multitask-sft/baselines/native_source_domain_r32_v3/"
    "grpo/results/gr_rec_think_composite_interest_v1_probe1_bridge_abc3_20260823.json"
)
RESULTS = RUNTIME / "boundary_adapt/results"
PARTS = RESULTS / "fixed_cot_sweep_20260824_parts"
MANIFEST = PARTS / "frozen_manifest.json"
DOMAIN = {"video": "<|video_begin|>", "prod": "<|prod_begin|>", "ad": "<|ad_begin|>", "living": "<|living_begin|>"}
ABC = tuple(re.compile(rf"<s_{stage}_(\d+)>").fullmatch for stage in "abc")
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
CHECKPOINTS = {
    "0": STEP0_ADAPTER,
    "50": FORMAL / "checkpoint-50",
    "100": FORMAL / "checkpoint-100",
    "200": FORMAL / "checkpoint-200",
    "300": FORMAL / "checkpoint-300",
}


def closed_cot(text: str) -> str:
    end = text.find("</think>")
    if end < 0:
        raise RuntimeError("FROZEN_COT_NOT_CLOSED")
    return text[:end + len("</think>")]


def parse_gold(value: str) -> tuple[str, int, int, int]:
    match = SID_RE.fullmatch(value)
    if not match:
        raise RuntimeError(f"INVALID_GOLD_SID={value!r}")
    return (match.group(1), *(int(match.group(index)) for index in range(2, 5)))


def token_names(tokenizer, ids: list[int]) -> list[str]:
    value = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    return [value] if isinstance(value, str) else list(value)


def parse_abc3(tokenizer, domain: str, ids: list[int]):
    names = token_names(tokenizer, ids)
    matches = [pattern(token) for pattern, token in zip(ABC, names)]
    if len(ids) != 3 or len(names) != 3 or any(match is None for match in matches):
        return None
    return (domain, *(int(match.group(1)) for match in matches))


def history_from_prompt(tokenizer, prompt_ids: list[int]) -> dict[str, Any]:
    domain_by_id = {}
    for domain, token in DOMAIN.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"DOMAIN_TOKEN_NOT_ATOMIC={domain}")
        domain_by_id[ids[0]] = domain
    matches = []
    for start in range(max(0, len(prompt_ids) - 3)):
        domain = domain_by_id.get(prompt_ids[start])
        if domain is None:
            continue
        names = token_names(tokenizer, prompt_ids[start + 1:start + 4])
        stages = [pattern(token) for pattern, token in zip(ABC, names)]
        if len(names) == 3 and not any(stage is None for stage in stages):
            matches.append((domain, *(int(stage.group(1)) for stage in stages)) )
    if not matches:
        raise RuntimeError("MODEL_VISIBLE_PROMPT_HAS_NO_HISTORY_SID")
    ordered = list(dict.fromkeys(matches))
    return {"unique_set": set(ordered), "most_recent": matches[-1], "unique_order": [list(item) for item in ordered]}


def copy_class(sid, history_set):
    if sid is None:
        return None
    if sid in history_set:
        return "EXACT_COPY"
    if any(old[:3] == sid[:3] for old in history_set):
        return "AB_COPY"
    if any(old[:2] == sid[:2] for old in history_set):
        return "A_COPY"
    return "NOVEL"


def classify(sid, history, gold_set) -> dict[str, Any]:
    category = copy_class(sid, history["unique_set"])
    gold = sid in gold_set if sid is not None else False
    if gold and category == "EXACT_COPY":
        cross = "GOLD_AND_HISTORY"
    elif gold:
        cross = "GOLD_NOT_HISTORY"
    elif category == "EXACT_COPY":
        cross = "HISTORY_NOT_GOLD"
    else:
        cross = "NEITHER"
    return {
        "predicted_sid": list(sid) if sid else None,
        "copy_class": category,
        "gold_history_class": cross,
        "is_most_recent_exact_copy": bool(sid is not None and sid == history["most_recent"]),
    }


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_frozen_probes() -> list[dict[str, Any]]:
    rows = []
    with PROBES.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("step") == 0:
                rows.append(row)
    rows.sort(key=lambda row: (int(row["probe_round"]), tuple(DOMAIN).index(row["target_domain"])))
    if len(rows) != 12 or Counter(row["target_domain"] for row in rows) != Counter({key: 3 for key in DOMAIN}):
        raise RuntimeError("FROZEN_PROBE_CONTRACT_INVALID")
    if any(len(row["think"]["candidates"]) != 4 for row in rows):
        raise RuntimeError("FROZEN_COT_CONTRACT_INVALID")
    return rows


def prepare_manifest() -> None:
    probes = load_frozen_probes()
    needed = {row["group_id"]: row["target_domain"] for row in probes}
    bridges: dict[str, set[str]] = {group: set() for group in needed}
    source_rows: Counter[str] = Counter()
    with BATA_SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = json.loads(row.get("aux_metadata_json") or "{}")
            group = metadata.get("recommendation_group_id")
            if group not in needed or row.get("source_segment") != "recommendation_cot":
                continue
            output = row["output"]
            left = output.index("</think>") + len("</think>")
            right = output.index(DOMAIN[needed[group]], left)
            bridges[group].add(output[left:right])
            source_rows[group] += 1
    manifest: list[dict[str, Any]] = []
    for probe_index, probe in enumerate(probes):
        group = probe["group_id"]
        if source_rows[group] == 0 or len(bridges[group]) != 1:
            raise RuntimeError(f"BRIDGE_PROVENANCE_INVALID group={group} rows={source_rows[group]} variants={len(bridges[group])}")
        bridge = next(iter(bridges[group]))
        for candidate_index, candidate in enumerate(probe["think"]["candidates"]):
            cot = closed_cot(candidate["completion"])
            manifest.append({
                "probe_index": probe_index, "probe_round": probe["probe_round"],
                "recommendation_group_id": group, "target_domain": probe["target_domain"],
                "candidate_id": candidate_index, "think_prompt": probe["think_prompt"],
                "gold_sids": probe["gold_sids"], "fixed_cot": cot,
                "cot_sha256": sha(cot), "completion_sha256": candidate["completion_sha256"],
                "bridge": bridge, "bridge_sha256": sha(bridge), "bridge_source_rows": source_rows[group],
            })
    if len(manifest) != 48:
        raise RuntimeError(f"FIXED_COT_COUNT_INVALID={len(manifest)}")
    PARTS.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"fixed_cot_count": len(manifest), "items": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FIXED_COT_MANIFEST={MANIFEST}")


def load_adapter(adapter: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def run_checkpoint(label: str) -> None:
    rank, world = int(os.environ.get("LOCAL_RANK", "-1")), int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or rank not in range(4):
        raise RuntimeError("RUN_REQUIRES_FOUR_LOCAL_RANKS")
    if label not in CHECKPOINTS or not MANIFEST.is_file():
        raise RuntimeError("UNKNOWN_CHECKPOINT_OR_MISSING_MANIFEST")
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    torch.cuda.set_device(rank)
    model, tokenizer = load_adapter(CHECKPOINTS[label], f"cuda:{rank}")
    records = []
    for index, item in enumerate(items):
        if index % world != rank:
            continue
        prompt_ids = encode_prompt(tokenizer, item["think_prompt"])
        cot_ids = tokenizer.encode(item["fixed_cot"], add_special_tokens=False)
        bridge_ids = tokenizer.encode(item["bridge"], add_special_tokens=False)
        domain_ids = tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)
        if len(domain_ids) != 1:
            raise RuntimeError("DOMAIN_TOKEN_NOT_ATOMIC")
        history = history_from_prompt(tokenizer, prompt_ids)
        gold_set = {parse_gold(value) for value in item["gold_sids"]}
        modes = {"bare": prompt_ids + cot_ids + domain_ids, "bridge": prompt_ids + cot_ids + bridge_ids + domain_ids}
        per_mode = {}
        with torch.inference_mode():
            for mode, context in modes.items():
                _, raw_ids = generate_batch(model, tokenizer, [context], min_new_tokens=3, max_new_tokens=3, do_sample=False, num_beams=32, num_return_sequences=32, return_ids=True)
                sids = [parse_abc3(tokenizer, item["target_domain"], ids) for ids in raw_ids]
                reward, exact, ab, a = think_reward(sids, gold_set)
                per_mode[mode] = {
                    "beam_raw": float(reward), "exact": int(exact), "ab": int(ab), "a": int(a),
                    "invalid": sum(sid is None for sid in sids), "beams": [
                        {"beam_index": beam, "raw_token_ids": ids, **classify(sid, history, gold_set)}
                        for beam, (ids, sid) in enumerate(zip(raw_ids, sids))
                    ],
                }
        records.append({
            **{key: item[key] for key in ("probe_index", "probe_round", "recommendation_group_id", "target_domain", "candidate_id", "cot_sha256", "completion_sha256", "bridge_sha256", "bridge_source_rows")},
            "prompt_token_count": len(prompt_ids), "cot_token_count": len(cot_ids),
            "history_unique_sids": history["unique_order"], "modes": per_mode,
        })
        print(f"SWEEP_PROGRESS checkpoint={label} rank={rank} item={len(records)}/12", flush=True)
    target = PARTS / label
    target.mkdir(parents=True, exist_ok=True)
    (target / f"rank{rank}.json").write_text(json.dumps({"checkpoint": label, "rank": rank, "adapter": str(CHECKPOINTS[label]), "records": records}, ensure_ascii=False), encoding="utf-8")


def rate(value: int, denominator: int):
    return value / denominator if denominator else None


def aggregate_records(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    beams = [beam for record in records for beam in record["modes"][mode]["beams"]]
    total = len(beams)
    valid = [beam for beam in beams if beam["predicted_sid"] is not None]
    copy = Counter(beam["copy_class"] for beam in valid)
    cross = Counter(beam["gold_history_class"] for beam in beams)
    grouped = {(record["recommendation_group_id"], record["candidate_id"]): record["modes"][mode] for record in records}
    return {
        "beam_raw_mean": sum(record["modes"][mode]["beam_raw"] for record in records) / len(records),
        "exact": sum(record["modes"][mode]["exact"] for record in records),
        "ab": sum(record["modes"][mode]["ab"] for record in records),
        "a": sum(record["modes"][mode]["a"] for record in records),
        "invalid": sum(record["modes"][mode]["invalid"] for record in records),
        "total_beams": total, "valid_predictions": len(valid), "valid_rate": rate(len(valid), total),
        "exact_copy_rate_all": rate(copy["EXACT_COPY"], total), "exact_copy_rate_valid": rate(copy["EXACT_COPY"], len(valid)),
        "ab_copy_rate_valid": rate(copy["AB_COPY"], len(valid)), "a_copy_rate_valid": rate(copy["A_COPY"], len(valid)),
        "novel_rate_valid": rate(copy["NOVEL"], len(valid)),
        "most_recent_copy_rate": rate(sum(beam["is_most_recent_exact_copy"] for beam in valid), len(valid)),
        "gold_and_history": rate(cross["GOLD_AND_HISTORY"], total), "gold_not_history": rate(cross["GOLD_NOT_HISTORY"], total),
        "history_not_gold": rate(cross["HISTORY_NOT_GOLD"], total),
        "unique_predicted_sid_count": len({tuple(beam["predicted_sid"]) for beam in valid}),
        "unique_exact_copy_sid_count": len({tuple(beam["predicted_sid"]) for beam in valid if beam["copy_class"] == "EXACT_COPY"}),
        "unique_sid_exact_copy_rate": rate(len({tuple(beam["predicted_sid"]) for beam in valid if beam["copy_class"] == "EXACT_COPY"}), len({tuple(beam["predicted_sid"]) for beam in valid})),
        "any_exact_copy_within_beam32": rate(sum(any(beam["copy_class"] == "EXACT_COPY" for beam in value["beams"]) for value in grouped.values()), len(grouped)),
    }


def merge_checkpoint(label: str) -> Path:
    parts = [json.loads((PARTS / label / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(4)]
    records = [record for part in parts for record in part["records"]]
    if len(records) != 48 or len({(row["recommendation_group_id"], row["candidate_id"]) for row in records}) != 48:
        raise RuntimeError("INCOMPLETE_FIXED_COT_SWEEP")
    result = {"checkpoint": label, "adapter": str(CHECKPOINTS[label]), "fixed_cot_count": len(records), "records": records, "modes": {}, "per_domain": {}}
    for mode in ("bare", "bridge"):
        result["modes"][mode] = aggregate_records(records, mode)
    for domain in DOMAIN:
        domain_records = [record for record in records if record["target_domain"] == domain]
        result["per_domain"][domain] = {mode: aggregate_records(domain_records, mode) for mode in ("bare", "bridge")}
    target = PARTS / f"checkpoint_{label}_merged.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MERGED_CHECKPOINT={target}")
    return target


def compare_numbers(actual: dict[str, Any], expected: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if actual[key] != expected[key]:
            raise RuntimeError(f"STEP0_REPRODUCTION_MISMATCH key={key} actual={actual[key]} expected={expected[key]}")


def gate_step0() -> None:
    merged = json.loads((PARTS / "checkpoint_0_merged.json").read_text(encoding="utf-8"))
    historical = json.loads(HISTORICAL_AUDIT.read_text(encoding="utf-8"))["fixed_abc3"]
    keys = ("valid_rate", "exact_copy_rate_all", "exact_copy_rate_valid", "ab_copy_rate_valid", "a_copy_rate_valid", "novel_rate_valid", "most_recent_copy_rate", "gold_and_history", "gold_not_history", "history_not_gold", "any_copy_beam32")
    actual_history = dict(merged["modes"]["bare"])
    actual_history["any_copy_beam32"] = actual_history["any_exact_copy_within_beam32"]
    compare_numbers(actual_history, historical, keys)
    probe1 = json.loads(HISTORICAL_PROBE1.read_text(encoding="utf-8"))
    group = "662e21149595c33240d7ec0baed2282d687f71e1a0983252f28edc9c18855cbc"
    records = [record for record in merged["records"] if record["recommendation_group_id"] == group]
    for mode, key in (("bare", "bare_abc3"), ("bridge", "bridge_abc3")):
        actual = aggregate_records(records, mode)
        expected = probe1[key]
        compare_numbers(actual, expected, ("beam_raw_mean", "exact", "ab", "a", "invalid"))
    payload = {"step0_reproduction_pass": True, "probe1_bridge_reproduction_pass": True, "fixed_cot_count": 48}
    (PARTS / "step0_gate.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


def finalize() -> None:
    gate = json.loads((PARTS / "step0_gate.json").read_text())
    if not (gate["step0_reproduction_pass"] and gate["probe1_bridge_reproduction_pass"]):
        raise RuntimeError("STEP0_GATE_NOT_PASSED")
    checkpoints = {label: json.loads((PARTS / f"checkpoint_{label}_merged.json").read_text(encoding="utf-8")) for label in CHECKPOINTS}
    baseline = checkpoints["0"]["modes"]
    denominator = baseline["bridge"]["beam_raw_mean"] - baseline["bare"]["beam_raw_mean"]
    table = []
    for label, item in checkpoints.items():
        bare, bridge = item["modes"]["bare"], item["modes"]["bridge"]
        row = {"checkpoint": int(label), "bare": bare, "bridge": bridge, "bridge_dependency_gap": bridge["beam_raw_mean"] - bare["beam_raw_mean"], "interface_recovery_ratio": None if abs(denominator) < 1e-12 else (bare["beam_raw_mean"] - baseline["bare"]["beam_raw_mean"]) / denominator}
        table.append(row)
    eligible = [row for row in table if row["bare"]["history_not_gold"] <= baseline["bare"]["history_not_gold"]]
    pool = eligible or table
    best = max(pool, key=lambda row: (row["bare"]["beam_raw_mean"], row["bare"]["exact"], -row["bridge_dependency_gap"], row["bare"]["gold_not_history"]))
    result = {"type": "boundary_adaptation_fixed_cot_checkpoint_sweep", "fixed_cot_count": 48, "history_extraction_source": "MODEL_VISIBLE_PROMPT_ONLY", "step0_gate": gate, "checkpoints": checkpoints, "table": table, "best_fixed_cot_checkpoint": best["checkpoint"], "best_reason": "highest Bare Beam raw among checkpoints with History-not-Gold not worse than step0; ties use Exact, smaller gap, then Gold-not-History", "training_started": False, "optimizer_created": False, "self_cot_evaluated": False}
    target = RESULTS / "boundary_adapt_fixed_cot_sweep_20260824.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FINAL_RESULT={target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS))
    parser.add_argument("--merge", choices=tuple(CHECKPOINTS))
    parser.add_argument("--gate-step0", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.prepare, args.checkpoint, args.merge, args.gate_step0, args.finalize)) != 1:
        raise SystemExit("choose exactly one operation")
    if args.prepare: prepare_manifest()
    elif args.checkpoint: run_checkpoint(args.checkpoint)
    elif args.merge: merge_checkpoint(args.merge)
    elif args.gate_step0: gate_step0()
    else: finalize()


if __name__ == "__main__":
    main()
