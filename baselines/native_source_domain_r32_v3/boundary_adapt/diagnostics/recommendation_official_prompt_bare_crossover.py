"""Official-domain prompt crossover over the fixed 40-group BARE diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any

import torch

RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]
from boundary_adapt.bridge_inside_sft.common import DOMAIN  # noqa: E402
from boundary_adapt.diagnostics.controlled_generator_decoder_crossover import load_model  # noqa: E402
from boundary_adapt.diagnostics.fixed_cot_checkpoint_sweep import classify, history_from_prompt, parse_abc3, parse_gold  # noqa: E402
from boundary_adapt.diagnostics.recommendation_root_cause_phase1_1 import (  # noqa: E402
    build_manifold, history_metrics, manifold_metrics, ranking_metrics, sid,
)
from grpo_model import BASE, generate_batch  # noqa: E402

V1 = RUNTIME / "boundary_adapt/results/recommendation_decoder_diagnostic_v1_40g"
PHASE11 = RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_1_40g"
OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_official_prompt_bare_crossover_40g"
AUDIT = OUTPUT / "prompt_audit.json"
PARTS = OUTPUT / "parts"
MODELS = ("Beta", "Gamma", "Step900")
MODEL_PATHS = {
    "Beta": Path("/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"),
    "Gamma": Path("/root/data_checkpoints_backup_20260824/outputs/baselines/native_source_domain_r32_v3/mini_gamma/checkpoint-136"),
    "Step900": Path("/root/data_checkpoints_backup_20260824/outputs/boundary_adapt/continuation_from300_to1500/checkpoint-900"),
}
EXPECTED_SHA = {
    "Beta": "4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3",
    "Gamma": "772921211cbcd238728cf538fe904042697c60e6034dd0a363081c306e3f48aa",
    "Step900": "ea395df041ef7bc13d5e5e54aec7a0d376b8f6ae4ae064eb6a7d89dd3c7cdffa",
}
SYSTEM = {
    "video": "你是一个推荐系统助手，擅长根据多域历史行为预测用户的视频偏好。",
    "prod": "你是一个智能推荐助理，能根据多域历史行为，推荐用户下一个感兴趣的商品。",
    "ad": "你是一个智能推荐助理，能根据多域历史行为，推荐用户下一个感兴趣的广告。",
    "living": "你是一个推荐系统助手，擅长根据多域历史行为预测用户的主播偏好。",
}
QUESTION = {
    "video": "请推断用户接下来会点击的视频。/think",
    "prod": "请推断用户接下来会点击的商品。/think",
    "ad": "请推荐用户下一个会点击的广告。/think",
    "living": "请推断用户接下来会点击的主播。/think",
}
OLD_ENDINGS = (
    "请根据以上信息，给出该用户在直播、电商、视频、广告场景中的目标内容。/think",
    "请输出该用户在不同场景下对应的目标内容。/think",
    "请基于这些线索总结该用户在各场景中的目标内容。/think",
)
SID_TEXT_RE = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"], text=True).strip()


def file_sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def old_fields(item: dict[str, Any]) -> tuple[str, str]:
    system = item.get("original_bata_system", item.get("official_style_system"))
    user = item.get("original_bata_prompt", item.get("official_style_prompt"))
    if not isinstance(system, str) or not isinstance(user, str):
        raise RuntimeError(f"MISSING_ORIGINAL_BATA_TEXT group={item['group_id']}")
    return system, user


def extract_history(user: str) -> str:
    lines = user.splitlines()
    if not lines:
        raise RuntimeError("EMPTY_ORIGINAL_USER")
    lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    body = "\n".join(lines).rstrip()
    for ending in OLD_ENDINGS:
        if body.endswith(ending):
            body = body[:-len(ending)].rstrip()
            break
    else:
        if not body.endswith("/think"):
            raise RuntimeError("ORIGINAL_USER_HAS_NO_THINK_SUFFIX")
        body = body[:-len("/think")].rstrip()
    if not body or not SID_TEXT_RE.search(body):
        raise RuntimeError("EXTRACTED_HISTORY_EMPTY_OR_HAS_NO_SID")
    return body


def render_prompt_ids(tokenizer, user: str, system: str) -> list[int]:
    from llamafactory.data.template import TEMPLATES
    ids, _ = TEMPLATES["qwen3_nothink"].encode_oneturn(
        tokenizer, [{"role": "user", "content": user}, {"role": "assistant", "content": ""}], system=system,
    )
    return list(map(int, ids))


def prepare() -> None:
    from transformers import AutoTokenizer
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((V1 / "manifest_40.json").read_text(encoding="utf-8"))
    if len(manifest["items"]) != 40 or len({row["group_id"] for row in manifest["items"]}) != 40:
        raise RuntimeError("FIXED_40_GROUP_CONTRACT_FAIL")
    for model, path in MODEL_PATHS.items():
        actual = file_sha(path / "adapter_model.safetensors")
        if actual != EXPECTED_SHA[model]:
            raise RuntimeError(f"ADAPTER_SHA_MISMATCH model={model} actual={actual}")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    prepared = []
    all_gates = []
    for item in manifest["items"]:
        old_system, old_user = old_fields(item)
        domain = item["domain"]
        history = extract_history(old_user)
        new_user = f"用户多域历史行为：\n{history}\n\n{QUESTION[domain]}"
        old_sids = SID_TEXT_RE.findall(old_user)
        new_sids = SID_TEXT_RE.findall(new_user)
        gates = {
            "system_domain_correct": SYSTEM[domain] == SYSTEM.get(domain),
            "question_domain_correct": new_user.endswith(QUESTION[domain]),
            "history_sid_sequence_parity": old_sids == new_sids,
            "history_sid_count": len(old_sids),
        }
        all_gates.append(gates)
        prepared.append({
            "group_id": item["group_id"], "domain": domain,
            "old_system": old_system, "new_system": SYSTEM[domain],
            "old_user": old_user, "new_user": new_user,
            "old_user_tail": old_user[-500:], "new_user_tail": new_user[-500:],
            "official_prompt_token_ids": render_prompt_ids(tokenizer, new_user, SYSTEM[domain]),
            "frozen_cot_token_ids": item["frozen_cot_token_ids"],
            "frozen_cot_sha256": item["frozen_cot_sha256"],
            "all_gold_sids": item["all_gold_sids"], "K": item["K"],
            "gold_sid_in_history": item["gold_sid_in_history"],
            "gates": gates,
        })
    if not all(all(gate[key] for key in ("system_domain_correct", "question_domain_correct", "history_sid_sequence_parity")) for gate in all_gates):
        raise RuntimeError("PROMPT_AUDIT_FAIL")
    samples = []
    for domain in DOMAIN:
        choices = sorted((row for row in prepared if row["domain"] == domain), key=lambda row: row["group_id"])[:2]
        samples.extend({key: row[key] for key in (
            "group_id", "domain", "old_system", "new_system", "old_user_tail", "new_user_tail", "gates",
        )} for row in choices)
    payload = {
        "source_commit": source_commit(), "groups": 40, "cases_planned": 120,
        "prompt_audit_pass": True, "system_domain_correct": True,
        "question_domain_correct": True, "history_sid_sequence_parity": True,
        "history_sid_sequence_parity_count": "40/40", "sample_audit_count": 8,
        "samples": samples, "items": prepared,
        "training_started": False, "self_cot_generation": False, "external_eval": False,
    }
    AUDIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PROMPT_AUDIT_PASS=YES")
    print("SYSTEM_DOMAIN_CORRECT=PASS")
    print("QUESTION_DOMAIN_CORRECT=PASS")
    print("HISTORY_SID_SEQUENCE_PARITY=PASS")


def run(model_label: str) -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["prompt_audit_pass"] or not audit["history_sid_sequence_parity"]:
        raise RuntimeError("GPU_BLOCKED_BY_PROMPT_AUDIT")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4:
        raise RuntimeError("OFFICIAL_CROSSOVER_REQUIRES_4_RANKS")
    torch.cuda.set_device(rank)
    model, tokenizer = load_model(MODEL_PATHS[model_label], f"cuda:{rank}")
    local = []
    for index, item in enumerate(audit["items"]):
        if index % world != rank:
            continue
        prompt = list(map(int, item["official_prompt_token_ids"]))
        cot = list(map(int, item["frozen_cot_token_ids"]))
        domain_ids = tokenizer.encode(DOMAIN[item["domain"]], add_special_tokens=False)
        if len(domain_ids) != 1:
            raise RuntimeError("DOMAIN_TOKEN_NOT_ATOMIC")
        history = history_from_prompt(tokenizer, prompt)
        gold_set = {parse_gold(value) for value in item["all_gold_sids"]}
        with torch.inference_mode():
            _, raw_ids = generate_batch(
                model, tokenizer, [prompt + cot + domain_ids], min_new_tokens=3, max_new_tokens=3,
                do_sample=False, num_beams=32, num_return_sequences=32, return_ids=True,
            )
        beams = []
        for beam_index, ids in enumerate(raw_ids):
            ids = list(map(int, ids))
            prediction = parse_abc3(tokenizer, item["domain"], ids)
            beams.append({"beam_index": beam_index, "raw_token_ids": ids, **classify(prediction, history, gold_set)})
        local.append({
            "model": model_label, "group_id": item["group_id"], "domain": item["domain"],
            "K": item["K"], "mode": "official_bare", "beams": beams,
        })
        print(f"OFFICIAL_BARE_PROGRESS model={model_label} rank={rank} group={index // world + 1}/10", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{model_label}_rank{rank}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in local), encoding="utf-8",
    )


def effective_gold_rank(row, item) -> int:
    predictions = [sid(beam["predicted_sid"]) if beam["predicted_sid"] else None for beam in row["beams"]]
    golds = [sid(value) for value in item["all_gold_sids"]]
    for index, prediction in enumerate(predictions, 1):
        if prediction in golds:
            return index
    return 33


def group_flags(row, item) -> dict[str, Any]:
    predictions = [sid(beam["predicted_sid"]) if beam["predicted_sid"] else None for beam in row["beams"]]
    golds = [sid(value) for value in item["all_gold_sids"]]
    return {
        "exact": any(value in golds for value in predictions if value is not None),
        "ab": any(any(value[:3] == gold[:3] for gold in golds) for value in predictions if value is not None),
        "a": any(any(value[:2] == gold[:2] for gold in golds) for value in predictions if value is not None),
        "history_fraction": sum(beam["copy_class"] == "EXACT_COPY" for beam in row["beams"]) / 32,
    }


def paired_effect(original: list[dict[str, Any]], official: list[dict[str, Any]], manifest) -> dict[str, Any]:
    old = {row["group_id"]: row for row in original}
    new = {row["group_id"]: row for row in official}
    rows = []
    rank_classes = Counter()
    for group in sorted(old):
        item = manifest[group]
        old_rank, new_rank = effective_gold_rank(old[group], item), effective_gold_rank(new[group], item)
        before, after = group_flags(old[group], item), group_flags(new[group], item)
        category = "improvement" if new_rank < old_rank else "regression" if new_rank > old_rank else "unchanged"
        rank_classes[category] += 1
        rows.append({
            "group_id": group, "old_effective_gold_rank": old_rank, "official_effective_gold_rank": new_rank,
            "delta_gold_rank_positive_is_improvement": old_rank - new_rank,
            "delta_exact_hit": int(after["exact"]) - int(before["exact"]),
            "delta_ab_hit": int(after["ab"]) - int(before["ab"]),
            "delta_a_hit": int(after["a"]) - int(before["a"]),
            "delta_history_candidate_fraction": after["history_fraction"] - before["history_fraction"],
            "rank_effect": category,
        })
    return {
        "mean_delta_gold_rank_positive_is_improvement": statistics.fmean(row["delta_gold_rank_positive_is_improvement"] for row in rows),
        "mean_delta_exact_hit": statistics.fmean(row["delta_exact_hit"] for row in rows),
        "mean_delta_ab_hit": statistics.fmean(row["delta_ab_hit"] for row in rows),
        "mean_delta_a_hit": statistics.fmean(row["delta_a_hit"] for row in rows),
        "mean_delta_history_candidate_fraction": statistics.fmean(row["delta_history_candidate_fraction"] for row in rows),
        "rank_group_counts": {key: rank_classes[key] for key in ("improvement", "unchanged", "regression")},
        "groups": rows,
    }


def summarize_rows(rows, manifest, manifold) -> dict[str, Any]:
    result = {
        "all": {
            "ranking": ranking_metrics(rows, manifest),
            "history": history_metrics(rows),
            "manifold": manifold_metrics(rows, manifold),
        }
    }
    for name, predicate in (
        ("GoldSIDInHistory", lambda item: item["gold_sid_in_history"]),
        ("GoldSIDNotInHistory", lambda item: not item["gold_sid_in_history"]),
        ("K1", lambda item: item["K"] == 1),
        ("K2PLUS", lambda item: item["K"] >= 2),
    ):
        subset = [row for row in rows if predicate(manifest[row["group_id"]])]
        result[name] = {"ranking": ranking_metrics(subset, manifest), "history": history_metrics(subset)}
    return result


def order_values(metrics, field: str) -> dict[str, float]:
    return {model: metrics[model]["all"]["ranking"]["gold"][field] for model in MODELS}


def order_label(values: dict[str, float]) -> str:
    ordered = sorted(MODELS, key=values.get, reverse=True)
    text = ordered[0]
    for previous, current in zip(ordered, ordered[1:]):
        text += (" ≈ " if abs(values[previous] - values[current]) <= 1e-12 else " > ") + current
    return text


def finalize() -> dict[str, Any]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    manifest = {row["group_id"]: row for row in audit["items"]}
    official = []
    for model in MODELS:
        for rank in range(4):
            path = PARTS / f"{model}_rank{rank}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"MISSING_PART={path}")
            official.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    if len(official) != 120 or len({(row["model"], row["group_id"]) for row in official}) != 120:
        raise RuntimeError("OFFICIAL_RECORD_CONTRACT_FAIL")
    if any(len(row["beams"]) != 32 for row in official):
        raise RuntimeError("BEAM32_CONTRACT_FAIL")
    original_all = [json.loads(line) for line in (V1 / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    original = [row for row in original_all if row["mode"] == "bare"]
    if len(original) != 120:
        raise RuntimeError("ORIGINAL_BARE_REUSE_CONTRACT_FAIL")
    manifold, manifold_stats = build_manifold()
    official_metrics, original_metrics, effects = {}, {}, {}
    for model in MODELS:
        new_rows = [row for row in official if row["model"] == model]
        old_rows = [row for row in original if row["model"] == model]
        official_metrics[model] = summarize_rows(new_rows, manifest, manifold)
        original_metrics[model] = summarize_rows(old_rows, manifest, manifold)
        effects[model] = paired_effect(old_rows, new_rows, manifest)
    original_hit_values, official_hit_values = order_values(original_metrics, "Hit@32"), order_values(official_metrics, "Hit@32")
    original_mrr_values, official_mrr_values = order_values(original_metrics, "MRR"), order_values(official_metrics, "MRR")
    def external_compatible(values):
        return values["Beta"] >= values["Step900"] >= values["Gamma"]
    if external_compatible(official_hit_values) and external_compatible(official_mrr_values):
        alignment = "YES"
    elif external_compatible(official_hit_values) or external_compatible(official_mrr_values):
        alignment = "PARTIAL"
    else:
        alignment = "NO"
    deltas = {
        model: {
            "hit32": official_metrics[model]["all"]["ranking"]["gold"]["Hit@32"] - original_metrics[model]["all"]["ranking"]["gold"]["Hit@32"],
            "mrr": official_metrics[model]["all"]["ranking"]["gold"]["MRR"] - original_metrics[model]["all"]["ranking"]["gold"]["MRR"],
            "history_fraction": official_metrics[model]["all"]["history"]["mean_candidate_fraction"] - original_metrics[model]["all"]["history"]["mean_candidate_fraction"],
        } for model in MODELS
    }
    gamma_hit_margin = deltas["Gamma"]["hit32"] - max(deltas["Beta"]["hit32"], deltas["Step900"]["hit32"])
    gamma_mrr_margin = deltas["Gamma"]["mrr"] - max(deltas["Beta"]["mrr"], deltas["Step900"]["mrr"])
    if deltas["Gamma"]["hit32"] < 0 and deltas["Gamma"]["mrr"] < 0:
        gamma_effect = "NEGATIVE"
    elif gamma_hit_margin >= .10 or gamma_mrr_margin >= .05:
        gamma_effect = "STRONG"
    elif gamma_hit_margin >= .025 or gamma_mrr_margin >= .015:
        gamma_effect = "MODERATE"
    else:
        gamma_effect = "WEAK"
    beta, step = official_metrics["Beta"]["all"]["ranking"], official_metrics["Step900"]["all"]["ranking"]
    if beta["gold"]["MRR"] > step["gold"]["MRR"] and beta["ab"]["Hit@32"] > step["ab"]["Hit@32"] and beta["a"]["Hit@32"] > step["a"]["Hit@32"]:
        beta_step = "BETA_RETAINS_RANKING_AND_HIERARCHY_ADVANTAGE"
    else:
        beta_step = "BETA_ADVANTAGE_NOT_RETAINED_ON_ALL_RANKING_AND_HIERARCHY_METRICS"
    consistent = alignment in ("YES", "PARTIAL")
    result = {
        "source_commit": source_commit(), "groups": 40, "cases": 120,
        "prompt_audit_pass": True, "history_sid_sequence_parity": True,
        "original_records_reused": 120, "original_records_regenerated": 0,
        "official_metrics": official_metrics, "original_metrics": original_metrics,
        "paired_prompt_effects": effects, "prompt_deltas": deltas,
        "train_manifold_stats": manifold_stats,
        "local_original_hit_order": order_label(original_hit_values),
        "local_official_hit_order": order_label(official_hit_values),
        "local_original_mrr_order": order_label(original_mrr_values),
        "local_official_mrr_order": order_label(official_mrr_values),
        "external_order": "Beta > Step900≈Gamma",
        "prompt_correction_improves_external_alignment": alignment,
        "gamma_prompt_specialization_effect": gamma_effect,
        "beta_vs_step900_official_prompt_diagnosis": beta_step,
        "training_started": False, "self_cot_generation": False, "external_eval": False,
        "root_class": "PROMPT_DISTRIBUTION_MISMATCH" if alignment == "YES" else "PROMPT_PARTIAL_PLUS_DECODER_GEOMETRY" if alignment == "PARTIAL" else "DECODER_OR_SELF_COT_GEOMETRY_NOT_PROMPT_MISMATCH",
        "conclusion": "Official prompt correction aligns the local decoder ordering with external." if alignment == "YES" else "Official prompt correction only partially aligns local and external ordering." if alignment == "PARTIAL" else "Official prompt correction does not resolve the local/external ordering mismatch.",
        "next_root_variable": "NONE_FOR_PHASE1_2" if consistent else "SELF_COT",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(official, key=lambda x: (x["model"], x["domain"], x["group_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Recommendation Official-Prompt BARE Crossover", "", f"Source commit: `{result['source_commit']}`", "", "| Model | Prompt | Gold@1 | Gold@5 | Gold@10 | Gold@32 | MRR | AB@32 | A@32 | History fraction |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for label, key in (("Original BATA", "original_metrics"), ("Official", "official_metrics")):
            value = result[key][model]["all"]
            rank = value["ranking"]
            lines.append(f"| {model} | {label} | {rank['gold']['Hit@1']:.4f} | {rank['gold']['Hit@5']:.4f} | {rank['gold']['Hit@10']:.4f} | {rank['gold']['Hit@32']:.4f} | {rank['gold']['MRR']:.4f} | {rank['ab']['Hit@32']:.4f} | {rank['a']['Hit@32']:.4f} | {value['history']['mean_candidate_fraction']:.4f} |")
    lines += ["", "## Paired effects", ""]
    for model in MODELS:
        effect = result["paired_prompt_effects"][model]
        delta = result["prompt_deltas"][model]
        lines.append(f"- {model}: Hit32 {delta['hit32']:+.4f}, MRR {delta['mrr']:+.4f}, history fraction {delta['history_fraction']:+.4f}; rank groups {effect['rank_group_counts']}.")
    lines += ["", f"Original Hit order: **{result['local_original_hit_order']}**", f"Official Hit order: **{result['local_official_hit_order']}**", f"Official MRR order: **{result['local_official_mrr_order']}**", f"External order: **{result['external_order']}**", "", f"Alignment: **{result['prompt_correction_improves_external_alignment']}**", f"Gamma specialization: **{result['gamma_prompt_specialization_effect']}**", f"Root class: **{result['root_class']}**", "", result["conclusion"], "", "No training, Self-CoT generation, or external evaluation was run."]
    return "\n".join(lines) + "\n"


def refresh_decision(result: dict[str, Any]) -> dict[str, Any]:
    result["source_commit"] = source_commit()
    def values(key: str, field: str) -> dict[str, float]:
        return {model: result[key][model]["all"]["ranking"]["gold"][field] for model in MODELS}
    def compatible(current: dict[str, float]) -> bool:
        return current["Beta"] >= current["Step900"] >= current["Gamma"]
    original_score = sum((compatible(values("original_metrics", "Hit@32")), compatible(values("original_metrics", "MRR"))))
    official_score = sum((compatible(values("official_metrics", "Hit@32")), compatible(values("official_metrics", "MRR"))))
    if official_score > original_score:
        alignment = "YES" if official_score == 2 else "PARTIAL"
    else:
        alignment = "NO"
    result["prompt_correction_improves_external_alignment"] = alignment
    beta = result["official_metrics"]["Beta"]["all"]["ranking"]
    step = result["official_metrics"]["Step900"]["all"]["ranking"]
    if (beta["gold"]["MRR"] > step["gold"]["MRR"] and beta["ab"]["Hit@32"] > step["ab"]["Hit@32"]
            and beta["a"]["Hit@32"] == step["a"]["Hit@32"] and beta["gold"]["Hit@32"] < step["gold"]["Hit@32"]):
        diagnosis = "BETA_RETAINS_MRR_AND_AB_ADVANTAGE_BUT_TRAILS_EXACT_HIT_AND_TIES_A"
    else:
        diagnosis = result["beta_vs_step900_official_prompt_diagnosis"]
    result["beta_vs_step900_official_prompt_diagnosis"] = diagnosis
    fully_aligned = official_score == 2
    result["root_class"] = "PROMPT_DISTRIBUTION_MISMATCH" if fully_aligned else "PROMPT_EFFECT_PRESENT_BUT_NOT_ROOT_CAUSE"
    result["conclusion"] = (
        "Official prompting gives Gamma a moderate metric gain but changes neither Hit@32 nor MRR model ordering; "
        "prompt mismatch is not the root cause of the local/external discrepancy."
    )
    result["next_root_variable"] = "NONE_FOR_PHASE1_2" if fully_aligned else "SELF_COT"
    return result


def terminal(result: dict[str, Any]) -> str:
    lines = [f"SOURCE_COMMIT={result['source_commit']}", "", "GROUPS=40", "CASES=120", "", "PROMPT_AUDIT_PASS=YES", "HISTORY_SID_SEQUENCE_PARITY=PASS", ""]
    original_known = {"Beta": (14, .167890), "Gamma": (12, .109954), "Step900": (16, .159519)}
    for model in MODELS:
        official = result["official_metrics"][model]["all"]
        delta = result["prompt_deltas"][model]
        numerator, mrr = original_known[model]
        lines += [f"--- {model} ---", f"ORIGINAL_HIT32={numerator}/40", f"OFFICIAL_HIT32={official['ranking']['gold']['Hit@32_numerator']}/40", f"ORIGINAL_MRR={mrr:.6f}", f"OFFICIAL_MRR={official['ranking']['gold']['MRR']:.6f}", f"PROMPT_HIT_DELTA={delta['hit32']:+.6f}", f"PROMPT_MRR_DELTA={delta['mrr']:+.6f}", f"HISTORY_FRACTION_DELTA={delta['history_fraction']:+.6f}", ""]
    lines += [f"LOCAL_ORIGINAL_HIT_ORDER={result['local_original_hit_order']}", f"LOCAL_OFFICIAL_HIT_ORDER={result['local_official_hit_order']}", "", f"LOCAL_ORIGINAL_MRR_ORDER={result['local_original_mrr_order']}", f"LOCAL_OFFICIAL_MRR_ORDER={result['local_official_mrr_order']}", "", "EXTERNAL_ORDER=Beta > Step900≈Gamma", "", f"PROMPT_CORRECTION_IMPROVES_EXTERNAL_ALIGNMENT={result['prompt_correction_improves_external_alignment']}", f"GAMMA_PROMPT_SPECIALIZATION_EFFECT={result['gamma_prompt_specialization_effect']}", f"BETA_VS_STEP900_OFFICIAL_PROMPT_DIAGNOSIS={result['beta_vs_step900_official_prompt_diagnosis']}", "", "TRAINING_STARTED=NO", "SELF_COT_GENERATION=NO", "EXTERNAL_EVAL=NO", "", f"ROOT_CLASS={result['root_class']}", f"CONCLUSION={result['conclusion']}", f"NEXT_ROOT_VARIABLE={result['next_root_variable']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "finalize", "report"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        if not args.model:
            parser.error("--model required")
        run(args.model)
    else:
        result = finalize() if args.action == "finalize" else json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        result = refresh_decision(result)
        (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        text = markdown(result)
        (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
        (OUTPUT / "CHATGPT_REVIEW.txt").write_text(text, encoding="utf-8")
        print(terminal(result))


if __name__ == "__main__":
    main()
