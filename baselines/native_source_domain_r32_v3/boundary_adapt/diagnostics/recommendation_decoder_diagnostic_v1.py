"""Frozen-CoT Recommendation Decoder Diagnostic V1 (40 reusable groups)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch
RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]
from grpo_model import BASE, generate_batch  # noqa: E402
from boundary_adapt.bridge_inside_sft.common import DOMAIN, DOMAIN_ORDER, stable_hash  # noqa: E402
from boundary_adapt.diagnostics.controlled_generator_decoder_crossover import load_model  # noqa: E402
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import (  # noqa: E402
    ABC, classify, history_from_prompt, parse_abc3, parse_gold,
)

RESULT = RUNTIME / "boundary_adapt/results/recommendation_decoder_diagnostic_v1_40g"
MANIFEST = RESULT / "manifest_40.json"
INVENTORY = RESULT / "model_inventory.json"
PARTS = RESULT / "parts"
HELDOUT = Path(
    "/root/GRPO_audit_results/bridge_inside_transition_sft_v1_20260824/"
    "heldout_256_manifest.json"
)
HISTORICAL = Path("/root/GRPO_audit_results/bridge_position_crossover_20260824/prepared_manifest.json")
MODEL_PATHS = {
    "Beta": Path(
        "/data/outputs/baselines/native_source_domain_r32_v3/"
        "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
    ),
    "Gamma": Path(
        "/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/"
        "mini_gamma/checkpoint-136"
    ),
    "Step900": Path(
        "/root/data_checkpoints_backup_20260824/outputs/boundary_adapt/"
        "continuation_from300_to1500/checkpoint-900"
    ),
}
EXTERNAL = {"Beta": 0.6594, "Gamma": 0.5913, "Step900": 0.5945}
MODES = ("free", "bare", "old_bridge")


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_groups(rows: list[dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    """Deterministic domain-stratified selection, interleaving K/history buckets."""
    selected = []
    for domain in DOMAIN_ORDER:
        candidates = [row for row in rows if row["target_domain"] == domain and row["group_id"] not in excluded]
        buckets: dict[tuple[bool, bool], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            buckets[(bool(row["gold_sid_in_history"]), int(row["gold_count"]) >= 2)].append(row)
        for values in buckets.values():
            values.sort(key=lambda row: (stable_hash("rec_decoder_v1_40g|" + row["group_id"]), row["group_id"]))
        domain_rows = []
        keys = ((True, False), (True, True), (False, False), (False, True))
        while len(domain_rows) < 10:
            progressed = False
            for key in keys:
                if buckets[key] and len(domain_rows) < 10:
                    domain_rows.append(buckets[key].pop(0))
                    progressed = True
            if not progressed:
                raise RuntimeError(f"INSUFFICIENT_GROUPS domain={domain}")
        selected.extend(domain_rows)
    if len(selected) != 40 or len({row["group_id"] for row in selected}) != 40:
        raise RuntimeError("GROUP_CARDINALITY_OR_UNIQUENESS_FAIL")
    return selected


def prepare() -> None:
    from transformers import AutoTokenizer

    RESULT.mkdir(parents=True, exist_ok=True)
    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))["items"]
    historical = {
        str(row["recommendation_group_id"])
        for row in json.loads(HISTORICAL.read_text(encoding="utf-8"))["items"]
    }
    if len(historical) != 12:
        raise RuntimeError("HISTORICAL_12_CONTRACT_FAIL")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError("CLOSE_NOT_ATOMIC")

    enriched = []
    for row in heldout:
        prompt = list(map(int, row["prompt_token_ids"]))
        history = history_from_prompt(tokenizer, prompt)
        gold_set = {parse_gold(value) for value in row["gold_sids"]}
        teacher_cot = list(map(int, row["teacher_cot_token_ids"]))
        if teacher_cot[-1:] != close_ids:
            raise RuntimeError(f"FROZEN_COT_NOT_CLOSED group={row['group_id']}")
        enriched.append({
            **row,
            "gold_sid_in_history": bool(gold_set & history["unique_set"]),
            "history_target_domain_sids": [list(sid) for sid in history["unique_order"] if sid[0] == row["target_domain"]],
        })
    chosen = select_groups(enriched, historical)
    public = []
    for row in chosen:
        cot_ids = list(map(int, row["teacher_cot_token_ids"]))
        public.append({
            "group_id": row["group_id"],
            "domain": row["target_domain"],
            "original_bata_system": row["system"],
            "original_bata_prompt": row["user"],
            "prompt_style": "ORIGINAL_BATA",
            "prompt_token_ids": row["prompt_token_ids"],
            "frozen_cot_body": tokenizer.decode(cot_ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "frozen_cot_token_ids": cot_ids,
            "frozen_cot_sha256": hashlib.sha256(json.dumps(cot_ids, separators=(",", ":")).encode()).hexdigest(),
            "exact_original_bridge": row["bridge"],
            "exact_original_bridge_token_ids": row["bridge_token_ids"],
            "all_gold_sids": row["gold_sids"],
            "history_target_domain_sids": row["history_target_domain_sids"],
            "K": row["gold_count"],
            "gold_sid_in_history": row["gold_sid_in_history"],
            "source_row_sha256": row["source_row_sha256"],
        })
    bridge_by_domain = {}
    for row in heldout:
        domain, bridge = row["target_domain"], row["bridge"]
        if domain in bridge_by_domain and bridge_by_domain[domain] != bridge:
            raise RuntimeError(f"MULTIPLE_KNOWN_BRIDGES domain={domain}")
        bridge_by_domain[domain] = bridge
    payload = {
        "version": "recommendation_decoder_diagnostic_v1_40g",
        "selection_salt": "rec_decoder_v1_40g|",
        "source_commit": source_commit(),
        "groups": 40,
        "domain_counts": dict(Counter(row["domain"] for row in public)),
        "excluded_historical_groups": sorted(historical),
        "excluded_historical_count": len(historical),
        "known_bridges": bridge_by_domain,
        "mapping_available": False,
        "items": public,
        "training_started": False,
        "self_cot_generation": False,
        "external_eval": False,
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    inventory = {}
    for label, path in MODEL_PATHS.items():
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (path / filename).is_file():
                raise RuntimeError(f"MISSING_MODEL_FILE model={label} file={filename} path={path}")
        inventory[label] = {
            "path": str(path),
            "adapter_model_sha256": file_sha(path / "adapter_model.safetensors"),
            "adapter_config_sha256": file_sha(path / "adapter_config.json"),
        }
    INVENTORY.write_text(json.dumps({
        "source_commit": source_commit(), "base": str(BASE), "models": inventory,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"MANIFEST={MANIFEST}")
    print(f"DOMAIN_COUNTS={payload['domain_counts']}")


def scan_first_sid(tokenizer, ids: list[int], domain: str):
    domain_ids = tokenizer.encode(DOMAIN[domain], add_special_tokens=False)
    if len(domain_ids) != 1:
        raise RuntimeError("DOMAIN_NOT_ATOMIC")
    for index in range(max(0, len(ids) - 3)):
        if ids[index] == domain_ids[0]:
            sid = parse_abc3(tokenizer, domain, ids[index + 1:index + 4])
            if sid is not None:
                return sid, index
    return None, None


def beam_record(model, tokenizer, context: list[int], domain: str, gold_set, history) -> dict[str, Any]:
    with torch.inference_mode():
        _, raw_ids = generate_batch(
            model, tokenizer, [context], min_new_tokens=3, max_new_tokens=3,
            do_sample=False, num_beams=32, num_return_sequences=32, return_ids=True,
        )
    beams = []
    for index, ids in enumerate(raw_ids):
        ids = list(map(int, ids))
        sid = parse_abc3(tokenizer, domain, ids)
        beams.append({"beam_index": index, "raw_token_ids": ids, **classify(sid, history, gold_set)})
    return {"beams": beams}


def free_record(model, tokenizer, context: list[int], domain: str, gold_set, history, exact_bridge: str, known: dict[str, str]):
    with torch.inference_mode():
        _, rows = generate_batch(
            model, tokenizer, [context], min_new_tokens=1, max_new_tokens=32,
            do_sample=False, num_beams=1, num_return_sequences=1, return_ids=True,
        )
    ids = list(map(int, rows[0]))
    raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    sid, sid_position = scan_first_sid(tokenizer, ids, domain)
    exact = raw.startswith(exact_bridge)
    known_hits = [name for name, bridge in known.items() if raw.startswith(bridge)]
    return {
        "continuation_token_ids": ids,
        "continuation_raw_text": raw,
        "exact_original_bridge_emitted": exact,
        "known_bridge_emitted": bool(known_hits),
        "known_bridge_domains": known_hits,
        "first_complete_target_sid": list(sid) if sid else None,
        "first_sid_token_position": sid_position,
        "first_sid_gold_match": bool(sid in gold_set if sid else False),
        "first_sid_classification": classify(sid, history, gold_set),
    }


def run(model_label: str) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4:
        raise RuntimeError("DIAGNOSTIC_REQUIRES_4_RANKS")
    torch.cuda.set_device(rank)
    model, tokenizer = load_model(MODEL_PATHS[model_label], f"cuda:{rank}")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = payload["known_bridges"]
    local = []
    for index, item in enumerate(payload["items"]):
        if index % world != rank:
            continue
        prompt = list(map(int, item["prompt_token_ids"]))
        cot = list(map(int, item["frozen_cot_token_ids"]))
        bridge = list(map(int, item["exact_original_bridge_token_ids"]))
        domain_ids = tokenizer.encode(DOMAIN[item["domain"]], add_special_tokens=False)
        history = history_from_prompt(tokenizer, prompt)
        gold_set = {parse_gold(value) for value in item["all_gold_sids"]}
        base = {"model": model_label, "group_id": item["group_id"], "domain": item["domain"], "K": item["K"]}
        local.append({**base, "mode": "free", **free_record(
            model, tokenizer, prompt + cot, item["domain"], gold_set, history,
            item["exact_original_bridge"], known,
        )})
        for mode, context in (
            ("bare", prompt + cot + domain_ids),
            ("old_bridge", prompt + cot + bridge + domain_ids),
        ):
            local.append({**base, "mode": mode, **beam_record(
                model, tokenizer, context, item["domain"], gold_set, history,
            )})
        print(f"REC_DECODER_PROGRESS model={model_label} rank={rank} group={index // world + 1}/10", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{model_label}_rank{rank}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in local), encoding="utf-8"
    )


def sid_tuple(value):
    return tuple(value) if value is not None else None


def stage_hit(sid, golds, length: int) -> bool:
    return bool(sid is not None and any(sid[:length] == gold[:length] for gold in golds))


def aggregate_mode(rows: list[dict[str, Any]], manifest_by_group: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exact, ab, a, copy, uniques, invalid = [], [], [], [], [], []
    free_exact, free_known, free_gold = [], [], []
    for row in rows:
        golds = [parse_gold(value) for value in manifest_by_group[row["group_id"]]["all_gold_sids"]]
        if row["mode"] == "free":
            sids = [sid_tuple(row["first_complete_target_sid"])] if row["first_complete_target_sid"] else []
            free_exact.append(row["exact_original_bridge_emitted"])
            free_known.append(row["known_bridge_emitted"])
            free_gold.append(row["first_sid_gold_match"])
            classes = [row["first_sid_classification"]] if sids else []
            invalid.append(not sids)
        else:
            sids = [sid_tuple(beam["predicted_sid"]) for beam in row["beams"] if beam["predicted_sid"] is not None]
            classes = [beam for beam in row["beams"] if beam["predicted_sid"] is not None]
            invalid.append(sum(beam["predicted_sid"] is None for beam in row["beams"]))
        exact.append(any(sid in golds for sid in sids))
        ab.append(any(stage_hit(sid, golds, 3) for sid in sids))
        a.append(any(stage_hit(sid, golds, 2) for sid in sids))
        copy.append(any(item["copy_class"] == "EXACT_COPY" for item in classes))
        uniques.append(len(set(sids)))
    result = {
        "groups": len(rows),
        "GoldSIDHit@32": statistics.fmean(exact),
        "GoldABHit@32": statistics.fmean(ab),
        "GoldAHit@32": statistics.fmean(a),
        "HistoryExactCopy@32": statistics.fmean(copy),
        "MeanUniqueSID@32": statistics.fmean(uniques),
        "MeanInvalid@32": statistics.fmean(invalid),
    }
    if free_exact:
        result.update({
            "ExactBridgeEmissionRate": statistics.fmean(free_exact),
            "KnownBridgeEmissionRate": statistics.fmean(free_known),
            "FirstSIDGoldHit": statistics.fmean(free_gold),
        })
    return result


def finalize() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = {row["group_id"]: row for row in payload["items"]}
    records = []
    for model in MODEL_PATHS:
        for rank in range(4):
            path = PARTS / f"{model}_rank{rank}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"MISSING_PART={path}")
            records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    if len(records) != 360:
        raise RuntimeError(f"RECORD_COUNT={len(records)}")
    identities = {(row["model"], row["mode"], row["group_id"]) for row in records}
    if len(identities) != 360:
        raise RuntimeError("RECORD_IDENTITY_DUPLICATE")
    with (RESULT / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(records, key=lambda x: (x["model"], x["mode"], x["domain"], x["group_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {}
    for model in MODEL_PATHS:
        metrics[model] = {}
        for mode in MODES:
            subset = [row for row in records if row["model"] == model and row["mode"] == mode]
            metrics[model][mode] = aggregate_mode(subset, items)
    local_order = sorted(MODEL_PATHS, key=lambda model: metrics[model]["bare"]["GoldSIDHit@32"], reverse=True)
    external_order = sorted(MODEL_PATHS, key=EXTERNAL.get, reverse=True)
    agreement = local_order == external_order
    comparisons = {
        "Beta_FREE_minus_BARE": metrics["Beta"]["free"]["GoldSIDHit@32"] - metrics["Beta"]["bare"]["GoldSIDHit@32"],
        "Beta_OLD_BRIDGE_minus_BARE": metrics["Beta"]["old_bridge"]["GoldSIDHit@32"] - metrics["Beta"]["bare"]["GoldSIDHit@32"],
        "Gamma_BARE_minus_Beta_BARE": metrics["Gamma"]["bare"]["GoldSIDHit@32"] - metrics["Beta"]["bare"]["GoldSIDHit@32"],
        "Gamma_OLD_BRIDGE_minus_BARE": metrics["Gamma"]["old_bridge"]["GoldSIDHit@32"] - metrics["Gamma"]["bare"]["GoldSIDHit@32"],
        "Step900_BARE_minus_Beta_BARE": metrics["Step900"]["bare"]["GoldSIDHit@32"] - metrics["Beta"]["bare"]["GoldSIDHit@32"],
        "Step900_OLD_BRIDGE_minus_BARE": metrics["Step900"]["old_bridge"]["GoldSIDHit@32"] - metrics["Step900"]["bare"]["GoldSIDHit@32"],
    }
    conclusion = (
        "LOCAL_BARE_RANKING_AGREES_WITH_EXTERNAL" if agreement
        else "LOCAL_BARE_RANKING_DISAGREES_WITH_EXTERNAL; use this 40-group set as a mechanism diagnostic, not a model-selection proxy"
    )
    summary = {
        "source_commit": payload["source_commit"],
        "groups": 40,
        "domain_counts": payload["domain_counts"],
        "mapping_available": False,
        "models": json.loads(INVENTORY.read_text())["models"],
        "metrics": metrics,
        "comparisons": comparisons,
        "local_bare_order": " > ".join(local_order),
        "external_order": "Beta > Step900≈Gamma",
        "strict_external_order_for_computation": " > ".join(external_order),
        "order_agreement": agreement,
        "diagnostic_conclusion": conclusion,
        "training_started": False,
        "optimizer_steps": 0,
        "self_cot_generation": False,
        "external_eval": False,
    }
    (RESULT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Recommendation Decoder Diagnostic V1 (40 groups)", "", f"Source commit: `{summary['source_commit']}`", "", "| Model | Mode | GoldSID | GoldAB | GoldA | HistoryCopy | MeanUnique | Invalid |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODEL_PATHS:
        for mode in MODES:
            value = metrics[model][mode]
            lines.append(f"| {model} | {mode} | {value['GoldSIDHit@32']:.4f} | {value['GoldABHit@32']:.4f} | {value['GoldAHit@32']:.4f} | {value['HistoryExactCopy@32']:.4f} | {value['MeanUniqueSID@32']:.3f} | {value['MeanInvalid@32']:.3f} |")
    lines += ["", "## FREE bridge metrics", "", "| Model | Exact bridge | Known bridge | First SID gold |", "|---|---:|---:|---:|"]
    for model in MODEL_PATHS:
        value = metrics[model]["free"]
        lines.append(f"| {model} | {value['ExactBridgeEmissionRate']:.4f} | {value['KnownBridgeEmissionRate']:.4f} | {value['FirstSIDGoldHit']:.4f} |")
    lines += ["", "## Comparisons", "", *[f"- {key}: {value:+.4f}" for key, value in comparisons.items()], "", f"Local BARE order: **{summary['local_bare_order']}**", f"External order: **{summary['external_order']}**", f"Order agreement: **{'YES' if agreement else 'NO'}**", "", f"Conclusion: {conclusion}", "", "No training, Self-CoT generation, or external evaluation was run."]
    markdown = "\n".join(lines) + "\n"
    (RESULT / "summary.md").write_text(markdown, encoding="utf-8")
    (RESULT / "CHATGPT_REVIEW.txt").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--model", choices=tuple(MODEL_PATHS))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        if not args.model:
            parser.error("--model is required for run")
        run(args.model)
    else:
        finalize()


if __name__ == "__main__":
    main()
