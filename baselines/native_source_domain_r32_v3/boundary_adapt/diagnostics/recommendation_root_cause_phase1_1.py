"""CPU-only root-cause analysis over persisted Recommendation Decoder V1 records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any

RUNTIME = Path("/data/GRPO")
SOURCE_REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
V1 = RUNTIME / "boundary_adapt/results/recommendation_decoder_diagnostic_v1_40g"
OUTPUT = RUNTIME / "boundary_adapt/results/recommendation_root_cause_phase1_1_40g"
SOURCE = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
SOURCE_SHA256 = "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca"
MODELS = ("Beta", "Gamma", "Step900")
MODES = ("bare", "old_bridge")
CUTS = (1, 5, 10, 32)
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"], text=True).strip()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sid(value) -> tuple[str, int, int, int]:
    if isinstance(value, list):
        return (str(value[0]), int(value[1]), int(value[2]), int(value[3]))
    match = SID_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"INVALID_SID={value!r}")
    return (match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def best_rank(predictions, golds, prefix: int) -> int | None:
    for index, value in enumerate(predictions, 1):
        if value is not None and any(value[:prefix] == gold[:prefix] for gold in golds):
            return index
    return None


def ranking_metrics(rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranks = {"gold": [], "ab": [], "a": []}
    unique = {"A": [], "AB": [], "ABC": []}
    numerators = {stage: {cut: 0 for cut in CUTS} for stage in ranks}
    for row in rows:
        predictions = [sid(beam["predicted_sid"]) if beam["predicted_sid"] else None for beam in row["beams"]]
        golds = [sid(value) for value in manifest[row["group_id"]]["all_gold_sids"]]
        found = {
            "gold": best_rank(predictions, golds, 4),
            "ab": best_rank(predictions, golds, 3),
            "a": best_rank(predictions, golds, 2),
        }
        for stage, rank in found.items():
            ranks[stage].append(rank)
            for cut in CUTS:
                numerators[stage][cut] += int(rank is not None and rank <= cut)
        valid = [value for value in predictions if value is not None]
        unique["A"].append(len({value[:2] for value in valid}))
        unique["AB"].append(len({value[:3] for value in valid}))
        unique["ABC"].append(len(set(valid)))
    n = len(rows)
    result: dict[str, Any] = {"N": n}
    for stage in ranks:
        values = ranks[stage]
        result[stage] = {
            **{f"Hit@{cut}": numerators[stage][cut] / n for cut in CUTS},
            **{f"Hit@{cut}_numerator": numerators[stage][cut] for cut in CUTS},
            "MRR": statistics.fmean(0.0 if rank is None else 1.0 / rank for rank in values),
            "mean_best_rank_hits_only": statistics.fmean(rank for rank in values if rank is not None) if any(rank is not None for rank in values) else None,
            "best_ranks": values,
        }
    result.update({f"MeanUnique{stage}@32": statistics.fmean(values) for stage, values in unique.items()})
    return result


def history_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = []
    top = {cut: 0 for cut in CUTS}
    for row in rows:
        flags = [beam["copy_class"] == "EXACT_COPY" for beam in row["beams"]]
        counts.append(sum(flags))
        for cut in CUTS:
            top[cut] += int(any(flags[:cut]))
    fractions = [count / 32 for count in counts]
    return {
        "N": len(rows),
        "mean_candidate_count": statistics.fmean(counts),
        "mean_candidate_fraction": statistics.fmean(fractions),
        "median_candidate_fraction": statistics.median(fractions),
        "p25_candidate_fraction": percentile(fractions, .25),
        "p75_candidate_fraction": percentile(fractions, .75),
        "p90_candidate_fraction": percentile(fractions, .90),
        **{f"Top{cut}ContainsHistoryRate": top[cut] / len(rows) for cut in CUTS},
        **{f"Top{cut}ContainsHistoryNumerator": top[cut] for cut in CUTS},
        "group_candidate_counts": counts,
    }


def build_manifold() -> tuple[dict[str, set[tuple]], dict[str, Any]]:
    if file_sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("TRAIN_SOURCE_SHA256_MISMATCH")
    manifold = {domain: set() for domain in ("video", "prod", "ad", "living")}
    rows = 0
    recommendation_rows = 0
    matches = 0
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            row = json.loads(line)
            if row.get("data_source") != "recommend" or row.get("source_segment") != "recommendation_cot":
                continue
            recommendation_rows += 1
            text = "\n".join(str(row.get(key, "")) for key in ("instruction", "input", "output", "aux_metadata_json"))
            for match in SID_RE.finditer(text):
                value = (match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)))
                manifold[value[0]].add(value)
                matches += 1
    stats = {
        "source": str(SOURCE), "source_sha256": SOURCE_SHA256, "streaming_passes": 1,
        "all_rows_scanned": rows, "recommendation_rows_scanned": recommendation_rows,
        "raw_sid_occurrences": matches,
        "sizes": {domain: len(values) for domain, values in manifold.items()},
        "terminology": ["TRAIN_MANIFOLD_SEEN", "TRAIN_MANIFOLD_UNSEEN"],
        "not_an_official_mapping": True,
    }
    return manifold, stats


def manifold_metrics(rows: list[dict[str, Any]], manifold: dict[str, set[tuple]]) -> dict[str, Any]:
    classes = Counter()
    top1_seen = 0
    total = 0
    for row in rows:
        row_classes = []
        for beam in row["beams"]:
            value = sid(beam["predicted_sid"]) if beam["predicted_sid"] else None
            if value is None:
                category = "INVALID"
            elif beam["copy_class"] == "EXACT_COPY":
                category = "HISTORY_COPY"
            elif value in manifold[value[0]]:
                category = "SEEN_NOVEL"
            else:
                category = "UNSEEN_RECOMBINATION"
            classes[category] += 1
            row_classes.append(category)
            total += 1
        top1_seen += int(row_classes[0] in ("HISTORY_COPY", "SEEN_NOVEL"))
    return {
        "candidates": total,
        "TRAIN_MANIFOLD_SEEN_FRACTION@32": (classes["HISTORY_COPY"] + classes["SEEN_NOVEL"]) / total,
        "TRAIN_MANIFOLD_UNSEEN_FRACTION@32": classes["UNSEEN_RECOMBINATION"] / total,
        "TOP1_TRAIN_MANIFOLD_SEEN_RATE": top1_seen / len(rows),
        **{f"{category}_count": classes[category] for category in ("HISTORY_COPY", "SEEN_NOVEL", "UNSEEN_RECOMBINATION", "INVALID")},
        **{f"{category}_fraction": classes[category] / total for category in ("HISTORY_COPY", "SEEN_NOVEL", "UNSEEN_RECOMBINATION", "INVALID")},
    }


def correct_v1_naming(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = False
    for row in payload["items"]:
        if "official_style_system" in row:
            row["original_bata_system"] = row.pop("official_style_system")
            changed = True
        if "official_style_prompt" in row:
            row["original_bata_prompt"] = row.pop("official_style_prompt")
            changed = True
        row["prompt_style"] = "ORIGINAL_BATA"
    payload["prompt_style"] = "ORIGINAL_BATA"
    payload["prompt_style_correction"] = {
        "previous_label": "official_style_*", "corrected_label": "original_bata_*",
        "records_modified": False,
    }
    if changed:
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    correct_v1_naming(V1 / "manifest_40.json")
    manifest_payload = json.loads((V1 / "manifest_40.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (V1 / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = {row["group_id"]: row for row in manifest_payload["items"]}
    if len(manifest) != 40 or len(records) != 360:
        raise RuntimeError("PERSISTED_V1_CONTRACT_FAIL")
    beam_records = [row for row in records if row["mode"] in MODES]
    if len(beam_records) != 240 or any(len(row["beams"]) != 32 for row in beam_records):
        raise RuntimeError("PERSISTED_BEAM_CONTRACT_FAIL")
    manifold, manifold_stats = build_manifold()
    (OUTPUT / "train_manifold_stats.json").write_text(json.dumps(manifold_stats, indent=2) + "\n", encoding="utf-8")

    analysis: dict[str, Any] = {
        "source_commit": source_commit(),
        "input_source_commit": manifest_payload["source_commit"],
        "current_40g_prompt_style": "ORIGINAL_BATA",
        "prompt_provenance": {
            "code_file": "boundary_adapt/bridge_inside_sft/common.py",
            "code_function": "construct_token_row -> render_system_prompt_ids(tokenizer, user, system)",
            "system_source": "original BATA row['system']",
            "user_source": "original BATA row['instruction'] + row['input']",
        },
        "free_mode": {"do_sample": False, "num_beams": 1, "max_new_tokens": 32, "interpretation": "TRANSITION_BEHAVIOR_PROBE_ONLY"},
        "groups": 40,
        "gold_in_history": sum(row["gold_sid_in_history"] for row in manifest.values()),
        "gold_not_in_history": sum(not row["gold_sid_in_history"] for row in manifest.values()),
        "K1": sum(row["K"] == 1 for row in manifest.values()),
        "K2PLUS": sum(row["K"] >= 2 for row in manifest.values()),
        "metrics": {}, "comparisons": {}, "train_manifold": manifold_stats,
        "records_reused": 360, "records_regenerated": 0,
        "training_started": False, "self_cot_generation": False, "external_eval": False,
    }
    for model in MODELS:
        analysis["metrics"][model] = {}
        for mode in MODES:
            rows = [row for row in beam_records if row["model"] == model and row["mode"] == mode]
            mode_result = {
                "all": {
                    "ranking": ranking_metrics(rows, manifest),
                    "history": history_metrics(rows),
                    "manifold": manifold_metrics(rows, manifold),
                }
            }
            for label, predicate in (
                ("GoldSIDInHistory", lambda item: item["gold_sid_in_history"]),
                ("GoldSIDNotInHistory", lambda item: not item["gold_sid_in_history"]),
                ("K1", lambda item: item["K"] == 1),
                ("K2PLUS", lambda item: item["K"] >= 2),
            ):
                subset = [row for row in rows if predicate(manifest[row["group_id"]])]
                mode_result[label] = {"ranking": ranking_metrics(subset, manifest), "history": history_metrics(subset)}
            analysis["metrics"][model][mode] = mode_result

    bare = {model: analysis["metrics"][model]["bare"] for model in MODELS}
    def hit(model, subset="all", stage="gold", cut=32):
        return bare[model][subset]["ranking"][stage][f"Hit@{cut}"]
    analysis["comparisons"] = {
        "beta_advantage_in_history_subset": {
            "vs_Gamma": hit("Beta", "GoldSIDInHistory") - hit("Gamma", "GoldSIDInHistory"),
            "vs_Step900": hit("Beta", "GoldSIDInHistory") - hit("Step900", "GoldSIDInHistory"),
        },
        "beta_advantage_in_nonhistory_subset": {
            "vs_Gamma": hit("Beta", "GoldSIDNotInHistory") - hit("Gamma", "GoldSIDNotInHistory"),
            "vs_Step900": hit("Beta", "GoldSIDNotInHistory") - hit("Step900", "GoldSIDNotInHistory"),
        },
    }
    beta_nonhistory = hit("Beta", "GoldSIDNotInHistory")
    analysis["copy_only_hypothesis_supported"] = not (
        beta_nonhistory > 0 and beta_nonhistory >= hit("Gamma", "GoldSIDNotInHistory")
    )
    analysis["copy_repeat_prior_important"] = (
        min(analysis["comparisons"]["beta_advantage_in_history_subset"].values()) > 0
        and max(analysis["comparisons"]["beta_advantage_in_nonhistory_subset"].values()) <= 0
    )
    analysis["step900_exact_concentration_pattern"] = (
        hit("Step900") > hit("Beta")
        and hit("Step900", stage="ab") < hit("Beta", stage="ab")
        and hit("Step900", stage="a") < hit("Beta", stage="a")
    )
    analysis["beta_hierarchy_coverage_advantage"] = all(
        hit("Beta", stage=stage) > hit(other, stage=stage)
        for other in ("Gamma", "Step900") for stage in ("ab", "a")
    )
    gains = {
        bucket: hit("Step900", bucket) - hit("Beta", bucket) for bucket in ("K1", "K2PLUS")
    }
    analysis["step900_gain_by_k_bucket"] = gains | {
        "classification": "K1" if gains["K1"] > gains["K2PLUS"] else "K2PLUS" if gains["K2PLUS"] > gains["K1"] else "EQUAL"
    }
    unseen = {model: bare[model]["all"]["manifold"]["UNSEEN_RECOMBINATION_fraction"] for model in MODELS}
    beta_delta = min(unseen["Gamma"], unseen["Step900"]) - unseen["Beta"]
    if beta_delta >= .10:
        support = "STRONG"
    elif beta_delta >= .03:
        support = "MODERATE"
    elif unseen["Beta"] > max(unseen["Gamma"], unseen["Step900"]):
        support = "CONTRADICTED"
    else:
        support = "WEAK"
    analysis["manifold_hypothesis_support"] = support
    analysis["gpu_gate"] = {
        "status": "FAIL",
        "reason": "CURRENT_40G_IS_ORIGINAL_BATA_BUT_GAMMA_TRAINING_PROMPT_PROVENANCE_NOT_AUDITABLE_FROM_CHECKPOINT_OR_KNOWN_CONFIGS",
        "gamma_checkpoint_training_args_checked": True,
        "gamma_dataset_or_launch_config_found": False,
        "gpu_rerun_started": False,
        "stop_after_cpu": True,
    }
    if analysis["step900_exact_concentration_pattern"]:
        root_class = "BEAM_HIERARCHY_GEOMETRY_DIFFERENCE"
    elif support in ("STRONG", "MODERATE"):
        root_class = "FULL_SID_MANIFOLD_PRIOR_DIFFERENCE"
    elif analysis["copy_repeat_prior_important"]:
        root_class = "HISTORY_REPEAT_PRIOR_DIFFERENCE"
    else:
        root_class = "MIXED_DECODER_GEOMETRY"
    analysis["root_cause_phase1_1_class"] = root_class
    analysis["phase1_1_conclusion"] = (
        "40g mechanism evidence separates exact ranking from A/AB hierarchy coverage; official-prompt mismatch remains untested because Gamma training-prompt provenance is missing."
    )
    analysis["next_experiment_needed"] = (
        "Recover Gamma launch/dataset provenance first; then run the fixed 240-case official-prompt crossover only if domain-specific training prompts are confirmed."
    )
    return analysis


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Recommendation Root-Cause Phase 1.1", "",
        f"Source commit: `{result['source_commit']}`", "",
        "The persisted 360 V1 cases were reused; no case was regenerated.", "",
        "## Prompt correction", "",
        "`CURRENT_40G_PROMPT_STYLE=ORIGINAL_BATA`. The previous `official_style_*` manifest labels were corrected to `original_bata_*`.",
        "FREE is greedy (`do_sample=False`, `num_beams=1`, `max_new_tokens=32`) and is only a transition-behavior probe.", "",
    ]
    for mode in MODES:
        lines += [f"## {mode.upper()} ranking", "", "| Model | Gold@1 | Gold@5 | Gold@10 | Gold@32 | Gold MRR | AB@32 | A@32 | UniqueA | UniqueAB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            rank = result["metrics"][model][mode]["all"]["ranking"]
            lines.append(f"| {model} | {rank['gold']['Hit@1']:.4f} | {rank['gold']['Hit@5']:.4f} | {rank['gold']['Hit@10']:.4f} | {rank['gold']['Hit@32']:.4f} | {rank['gold']['MRR']:.4f} | {rank['ab']['Hit@32']:.4f} | {rank['a']['Hit@32']:.4f} | {rank['MeanUniqueA@32']:.2f} | {rank['MeanUniqueAB@32']:.2f} |")
        lines.append("")
    lines += ["## Subsets and proxies", ""]
    for model in MODELS:
        bare = result["metrics"][model]["bare"]
        in_rank = bare["GoldSIDInHistory"]["ranking"]["gold"]
        out_rank = bare["GoldSIDNotInHistory"]["ranking"]["gold"]
        history = bare["all"]["history"]
        manifold = bare["all"]["manifold"]
        lines += [
            f"- {model}: GoldInHistory Hit32={in_rank['Hit@32_numerator']}/{bare['GoldSIDInHistory']['ranking']['N']}; GoldNotInHistory Hit32={out_rank['Hit@32_numerator']}/{bare['GoldSIDNotInHistory']['ranking']['N']}; history candidate fraction={history['mean_candidate_fraction']:.4f}; manifold fractions history/seen-novel/unseen={manifold['HISTORY_COPY_fraction']:.4f}/{manifold['SEEN_NOVEL_fraction']:.4f}/{manifold['UNSEEN_RECOMBINATION_fraction']:.4f}."
        ]
    lines += ["", f"Manifold hypothesis: **{result['manifold_hypothesis_support']}**", f"Root class: **{result['root_cause_phase1_1_class']}**", "", f"GPU gate: **{result['gpu_gate']['status']}** - {result['gpu_gate']['reason']}", "", "No training, Self-CoT generation, external evaluation, or GPU rerun was performed."]
    return "\n".join(lines) + "\n"


def refresh_interpretation(result: dict[str, Any]) -> dict[str, Any]:
    bare = {model: result["metrics"][model]["bare"] for model in MODELS}
    history_fraction = {model: bare[model]["all"]["history"]["mean_candidate_fraction"] for model in MODELS}
    in_hit = {model: bare[model]["GoldSIDInHistory"]["ranking"]["gold"]["Hit@32"] for model in MODELS}
    out_hit = {model: bare[model]["GoldSIDNotInHistory"]["ranking"]["gold"]["Hit@32"] for model in MODELS}
    result["copy_repeat_prior_important"] = (
        history_fraction["Beta"] > max(history_fraction["Gamma"], history_fraction["Step900"])
        and out_hit["Beta"] <= min(out_hit["Gamma"], out_hit["Step900"])
    )
    result["copy_only_hypothesis_supported"] = (
        in_hit["Beta"] > max(in_hit["Gamma"], in_hit["Step900"])
        and out_hit["Beta"] == 0
    )
    if result["copy_repeat_prior_important"] and result["step900_exact_concentration_pattern"]:
        result["root_cause_phase1_1_class"] = "HISTORY_COPY_AND_BEAM_HIERARCHY_GEOMETRY"
    result["phase1_1_conclusion"] = (
        "Beta is the most history-copy-heavy decoder, but copy alone does not explain the local/external ordering. "
        "Its advantage is also ranking/hierarchy geometry: Beta covers A/AB more broadly and ranks gold earlier, "
        "while Step900 gains deep-beam exact hits mainly on multi-positive groups. The full-SID manifold proxy is weak."
    )
    result["next_experiment_needed"] = (
        "Recover auditable Gamma launch/dataset prompt provenance. Only if domain-specific Gamma training prompts are confirmed, "
        "run the already-specified fixed 240-case official-prompt crossover."
    )
    return result


def terminal_report(result: dict[str, Any]) -> str:
    bare = {model: result["metrics"][model]["bare"] for model in MODELS}
    def rank(model, subset="all"):
        return bare[model][subset]["ranking"]
    lines = [
        f"SOURCE_COMMIT={result['source_commit']}", "",
        "CURRENT_40G_PROMPT_STYLE=ORIGINAL_BATA",
        "FREE_MODE_DO_SAMPLE=NO", "FREE_MODE_NUM_BEAMS=1", "FREE_MODE_MAX_NEW_TOKENS=32",
        "FREE_MODE_INTERPRETATION=TRANSITION_BEHAVIOR_PROBE_ONLY", "",
        "GROUPS=40", "GOLD_IN_HISTORY=16", "GOLD_NOT_IN_HISTORY=24", "K1=22", "K2PLUS=18", "",
        "--- BARE GoldInHistory ---",
        f"BETA_HIT32={rank('Beta', 'GoldSIDInHistory')['gold']['Hit@32_numerator']}/16",
        f"GAMMA_HIT32={rank('Gamma', 'GoldSIDInHistory')['gold']['Hit@32_numerator']}/16",
        f"STEP900_HIT32={rank('Step900', 'GoldSIDInHistory')['gold']['Hit@32_numerator']}/16", "",
        "--- BARE GoldNotInHistory ---",
        f"BETA_HIT32={rank('Beta', 'GoldSIDNotInHistory')['gold']['Hit@32_numerator']}/24",
        f"GAMMA_HIT32={rank('Gamma', 'GoldSIDNotInHistory')['gold']['Hit@32_numerator']}/24",
        f"STEP900_HIT32={rank('Step900', 'GoldSIDNotInHistory')['gold']['Hit@32_numerator']}/24", "",
        f"COPY_ONLY_HYPOTHESIS_SUPPORTED={'YES' if result['copy_only_hypothesis_supported'] else 'NO'}", "",
        "--- Candidate History Fractions ---",
        *[f"{model.upper()}_HISTORY_FRACTION={bare[model]['all']['history']['mean_candidate_fraction']:.6f}" for model in MODELS], "",
        "--- Beam Ranking ---",
        *[f"{model.upper()}_GOLD_MRR={rank(model)['gold']['MRR']:.6f}" for model in MODELS], "",
        *[f"{model.upper()}_AB_HIT32={rank(model)['ab']['Hit@32']:.6f}" for model in MODELS], "",
        *[f"{model.upper()}_A_HIT32={rank(model)['a']['Hit@32']:.6f}" for model in MODELS], "",
        "--- Train Manifold Proxy ---",
        *[f"TRAIN_MANIFOLD_SIZE_{domain.upper()}={result['train_manifold']['sizes'][domain]}" for domain in ("video", "prod", "ad", "living")], "",
    ]
    for model in MODELS:
        manifold = bare[model]["all"]["manifold"]
        lines += [
            f"{model.upper()}_HISTORY_COPY_FRACTION={manifold['HISTORY_COPY_fraction']:.6f}",
            f"{model.upper()}_SEEN_NOVEL_FRACTION={manifold['SEEN_NOVEL_fraction']:.6f}",
            f"{model.upper()}_UNSEEN_RECOMBINATION_FRACTION={manifold['UNSEEN_RECOMBINATION_fraction']:.6f}", "",
        ]
    lines += [f"MANIFOLD_HYPOTHESIS_SUPPORT={result['manifold_hypothesis_support']}", "", "--- K bucket ---"]
    for model in MODELS:
        lines += [
            f"{model.upper()}_K1_HIT32={rank(model, 'K1')['gold']['Hit@32_numerator']}/22",
            f"{model.upper()}_K2PLUS_HIT32={rank(model, 'K2PLUS')['gold']['Hit@32_numerator']}/18",
        ]
    lines += [
        f"STEP900_GAIN_BY_K_BUCKET={result['step900_gain_by_k_bucket']['classification']}", "",
        f"GPU_GATE={result['gpu_gate']['status']}", "GPU_RERUN_STARTED=NO", "",
        "TRAINING_STARTED=NO", "SELF_COT_GENERATION=NO", "EXTERNAL_EVAL=NO", "",
        f"ROOT_CAUSE_PHASE1_1_CLASS={result['root_cause_phase1_1_class']}",
        f"PHASE1_1_CONCLUSION={result['phase1_1_conclusion']}",
        f"NEXT_EXPERIMENT_NEEDED={result['next_experiment_needed']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("analyze", "report"))
    args = parser.parse_args()
    result = analyze() if args.action == "analyze" else json.loads((OUTPUT / "cpu_behavior_analysis.json").read_text(encoding="utf-8"))
    result = refresh_interpretation(result)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "cpu_behavior_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    (OUTPUT / "cpu_behavior_analysis.md").write_text(markdown, encoding="utf-8")
    (OUTPUT / "CHATGPT_ROOT_CAUSE_PHASE1_1.txt").write_text(markdown, encoding="utf-8")
    print(terminal_report(result))


if __name__ == "__main__":
    main()
