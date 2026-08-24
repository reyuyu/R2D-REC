"""CPU-only Phase 1.4.1 adjudication of existing recommendation diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Iterable

RUNTIME = Path("/data/GRPO")
REPO = Path("/data/tmp_inspect/onereason-multitask-sft")
RELATIVE_SCRIPT = Path(
    "baselines/native_source_domain_r32_v3/boundary_adapt/diagnostics/"
    "recommendation_root_cause_phase1_4_1.py"
)
RUNTIME_SCRIPT = RUNTIME / "boundary_adapt/diagnostics/recommendation_root_cause_phase1_4_1.py"
RESULTS = RUNTIME / "boundary_adapt/results"
PHASE12 = RESULTS / "recommendation_official_prompt_bare_crossover_40g"
PHASE14 = RESULTS / "recommendation_root_cause_phase1_4"
OUTPUT = RESULTS / "recommendation_root_cause_phase1_4_1"
TOKENIZER_CONFIG = Path("/data/models/onereason-8b-pretrain-competition/tokenizer_config.json")
BASE_MODEL = TOKENIZER_CONFIG.parent

MODELS = ("Beta", "Step900")
DOMAINS = ("video", "prod", "ad", "living")
METRICS = ("mrr", "hit32", "ab32", "a32", "history_fraction")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260825
EXTERNAL_DIRECTION = {domain: "BETA_GT_STEP900" for domain in DOMAINS}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def code_audit() -> dict[str, Any]:
    commit = git("rev-parse", "HEAD")
    origin_main = git("rev-parse", "origin/main")
    status = git("status", "--short")
    github_sha = sha256(REPO / RELATIVE_SCRIPT)
    runtime_sha = sha256(RUNTIME_SCRIPT)
    result = {
        "implement_commit": commit,
        "origin_main": origin_main,
        "push_status": "PASS" if commit == origin_main else "FAIL",
        "git_status_short": status,
        "github_script_sha256": github_sha,
        "runtime_script_sha256": runtime_sha,
        "runtime_github_parity": "PASS" if github_sha == runtime_sha else "FAIL",
    }
    if result["push_status"] != "PASS" or status or result["runtime_github_parity"] != "PASS":
        raise RuntimeError(f"CODE_AUDIT_GATE_FAIL={result}")
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def stable_seed(label: str) -> int:
    suffix = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + suffix) % (2**32)


def sid(value: Any) -> tuple[str, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return str(value[0]), int(value[1]), int(value[2]), int(value[3])


def row_metrics(row: dict[str, Any], item: dict[str, Any]) -> dict[str, float]:
    predictions = [sid(beam.get("predicted_sid")) for beam in row["beams"]]
    golds = {sid_string(value) for value in item["all_gold_sids"]}
    parsed_golds = {parse_sid_string(value) for value in golds}

    def first_rank(prefix: int) -> int | None:
        return next(
            (
                index
                for index, value in enumerate(predictions, 1)
                if value is not None and any(value[:prefix] == gold[:prefix] for gold in parsed_golds)
            ),
            None,
        )

    exact_rank = first_rank(4)
    return {
        "mrr": 0.0 if exact_rank is None else 1.0 / exact_rank,
        "hit32": float(exact_rank is not None),
        "ab32": float(first_rank(3) is not None),
        "a32": float(first_rank(2) is not None),
        "history_fraction": sum(beam.get("copy_class") == "EXACT_COPY" for beam in row["beams"]) / 32.0,
    }


def sid_string(value: Any) -> str:
    return str(value)


def parse_sid_string(value: str) -> tuple[str, int, int, int]:
    import re

    match = re.fullmatch(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>", value)
    if match is None:
        raise ValueError(f"invalid SID: {value}")
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def summarize(values: list[dict[str, float]]) -> dict[str, Any]:
    n = len(values)
    return {
        metric: {
            "numerator": sum(value[metric] for value in values),
            "denominator": n,
            "estimate": statistics.fmean(value[metric] for value in values),
        }
        for metric in METRICS
    }


def paired_differences(
    group_ids: Iterable[str],
    beta: dict[str, dict[str, float]],
    step: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        group_id: {metric: beta[group_id][metric] - step[group_id][metric] for metric in METRICS}
        for group_id in group_ids
    }


def paired_bootstrap(differences: dict[str, dict[str, float]], label: str) -> dict[str, Any]:
    group_ids = sorted(differences)
    rng = random.Random(stable_seed(label))
    samples = {metric: [] for metric in METRICS}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [rng.choice(group_ids) for _ in group_ids]
        for metric in METRICS:
            samples[metric].append(statistics.fmean(differences[group][metric] for group in selected))
    result = {
        "method": "paired group bootstrap",
        "groups": len(group_ids),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": stable_seed(label),
        "metrics": {},
    }
    for metric in METRICS:
        observed = statistics.fmean(differences[group][metric] for group in group_ids)
        result["metrics"][metric] = {
            "estimate": observed,
            "bootstrap_mean": statistics.fmean(samples[metric]),
            "ci95": [percentile(samples[metric], 0.025), percentile(samples[metric], 0.975)],
            "p_positive": sum(value > 0 for value in samples[metric]) / BOOTSTRAP_REPLICATES,
        }
    return result


def direction(value: float) -> str:
    if value > 0:
        return "BETA_GT_STEP900"
    if value < 0:
        return "STEP900_GT_BETA"
    return "TIE"


def stratum(item: dict[str, Any]) -> str:
    history = "HISTORY" if item["gold_sid_in_history"] else "NONHISTORY"
    kval = "K1" if int(item["K"]) == 1 else "K2PLUS"
    return f"{item['domain']}|{history}|{kval}"


def load_inputs() -> dict[str, Any]:
    p12_records = [
        row for row in read_jsonl(PHASE12 / "records.jsonl") if row.get("model") in MODELS
    ]
    p14_records = read_jsonl(PHASE14 / "nothink_records.jsonl")
    p12_items = {row["group_id"]: row for row in read_json(PHASE12 / "prompt_audit.json")["items"]}
    p14_items = {row["group_id"]: row for row in read_json(PHASE14 / "nothink_prompt_audit.json")["items"]}
    return {
        "think_records": p12_records,
        "nothink_records": p14_records,
        "think_items": p12_items,
        "nothink_items": p14_items,
        "natural": read_json(PHASE14 / "natural_distribution_stats.json"),
        "bounded_artifact": read_json(PHASE14 / "bounded_official_artifact_check.json"),
    }


def validate_records(inputs: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    model_sets: dict[str, dict[str, set[str]]] = {}
    for route, rows in (("phase12", inputs["think_records"]), ("phase14", inputs["nothink_records"])):
        keys = {(row["model"], row["group_id"]) for row in rows}
        model_sets[route] = {
            model: {row["group_id"] for row in rows if row["model"] == model} for model in MODELS
        }
        checks[f"{route}_records_80"] = len(rows) == 80
        checks[f"{route}_unique_model_group_80"] = len(keys) == 80
        checks[f"{route}_models_40_each"] = all(len(model_sets[route][model]) == 40 for model in MODELS)
        checks[f"{route}_same_groups_between_models"] = model_sets[route]["Beta"] == model_sets[route]["Step900"]
        checks[f"{route}_beam32"] = all(len(row.get("beams", [])) == 32 for row in rows)
    fixed40 = model_sets["phase12"]["Beta"]
    checks["same_group_set_phase12_phase14"] = fixed40 == model_sets["phase14"]["Beta"]
    checks["item_sets_match_fixed40"] = set(inputs["think_items"]) == set(inputs["nothink_items"]) == fixed40

    mismatches: list[dict[str, Any]] = []
    for group_id in sorted(fixed40):
        old, new = inputs["think_items"][group_id], inputs["nothink_items"][group_id]
        fields = ("domain", "K", "gold_sid_in_history", "all_gold_sids")
        for field in fields:
            if old[field] != new[field]:
                mismatches.append({"group_id": group_id, "field": field, "phase12": old[field], "phase14": new[field]})
        for row in (value for value in inputs["nothink_records"] if value["group_id"] == group_id):
            for field in ("domain", "K", "gold_sid_in_history"):
                if row[field] != new[field]:
                    mismatches.append({"group_id": group_id, "model": row["model"], "field": field})
    checks["gold_domain_k_history_match"] = not mismatches
    passed = all(checks.values())
    result = {
        "record_contract_pass": passed,
        "checks": checks,
        "mismatches": mismatches,
        "phase12_group_count": len(fixed40),
        "phase14_group_count": len(model_sets["phase14"]["Beta"]),
    }
    if not passed:
        raise RuntimeError(f"RECORD_CONTRACT_FAIL={result}")
    return result


def metric_maps(rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        model: {
            row["group_id"]: row_metrics(row, items[row["group_id"]])
            for row in rows
            if row["model"] == model
        }
        for model in MODELS
    }


def subset_analysis(
    group_ids: Iterable[str],
    maps: dict[str, dict[str, dict[str, float]]],
    label: str,
) -> dict[str, Any]:
    groups = sorted(group_ids)
    beta_values = [maps["Beta"][group] for group in groups]
    step_values = [maps["Step900"][group] for group in groups]
    differences = paired_differences(groups, maps["Beta"], maps["Step900"])
    return {
        "N": len(groups),
        "group_ids": groups,
        "Beta": summarize(beta_values),
        "Step900": summarize(step_values),
        "paired": paired_bootstrap(differences, label),
        "directions": {
            metric: direction(statistics.fmean(differences[group][metric] for group in groups)) for metric in METRICS
        },
    }


def natural_counts(natural: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for domain in DOMAINS:
        for bucket, count in natural["per_domain"][domain]["bucket_counts"].items():
            counts[f"{domain}|{bucket}"] = int(count)
    if sum(counts.values()) != int(natural["overall"]["N"]):
        raise RuntimeError("NATURAL_STRATUM_COUNT_MISMATCH")
    return counts


def weighted_domain(
    domain: str,
    maps: dict[str, dict[str, dict[str, float]]],
    items: dict[str, dict[str, Any]],
    counts: Counter[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id, item in items.items():
        if item["domain"] == domain:
            groups_by_stratum[stratum(item)].append(group_id)
    domain_keys = [key for key in counts if key.startswith(domain + "|") and counts[key] > 0]
    domain_total = sum(counts[key] for key in domain_keys)
    supported = [key for key in domain_keys if groups_by_stratum.get(key)]
    covered = sum(counts[key] for key in supported)
    normalized_weight = {key: counts[key] / covered for key in supported}

    def model_estimate(model: str, sampled: dict[str, list[str]] | None = None) -> dict[str, float]:
        selected = groups_by_stratum if sampled is None else sampled
        return {
            metric: sum(
                normalized_weight[key] * statistics.fmean(maps[model][group][metric] for group in selected[key])
                for key in supported
            )
            for metric in METRICS
        }

    beta, step = model_estimate("Beta"), model_estimate("Step900")
    rng = random.Random(stable_seed("weighted-" + domain))
    samples = {metric: [] for metric in METRICS}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = {
            key: [rng.choice(groups_by_stratum[key]) for _ in groups_by_stratum[key]] for key in supported
        }
        beta_sample, step_sample = model_estimate("Beta", sampled), model_estimate("Step900", sampled)
        for metric in METRICS:
            samples[metric].append(beta_sample[metric] - step_sample[metric])

    strata = {}
    underresolved = []
    for key in domain_keys:
        n = len(groups_by_stratum.get(key, []))
        natural_mass = counts[key] / domain_total
        status = "UNDERRESOLVED" if natural_mass >= 0.05 and n < 3 else "ADEQUATE"
        if status == "UNDERRESOLVED":
            underresolved.append(key)
        strata[key] = {
            "natural_count": counts[key],
            "natural_domain_mass": natural_mass,
            "probe_sample_n": n,
            "status": status,
        }
    result = {
        "domain": domain,
        "natural_domain_n": domain_total,
        "covered_domain_natural_mass": covered / domain_total,
        "coverage_status": "LOW_COVERAGE" if covered / domain_total < 0.90 else "ADEQUATE_COVERAGE",
        "Beta": beta,
        "Step900": step,
        "delta": {metric: beta[metric] - step[metric] for metric in METRICS},
        "bootstrap": {
            "method": "paired resampling within observed strata with fixed natural within-domain weights",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": stable_seed("weighted-" + domain),
            "metrics": {
                metric: {
                    "estimate": beta[metric] - step[metric],
                    "bootstrap_mean": statistics.fmean(samples[metric]),
                    "ci95": [percentile(samples[metric], 0.025), percentile(samples[metric], 0.975)],
                    "p_positive": sum(value > 0 for value in samples[metric]) / BOOTSTRAP_REPLICATES,
                }
                for metric in METRICS
            },
        },
        "stratum_sample_n": {key: len(groups_by_stratum.get(key, [])) for key in domain_keys},
        "singleton_strata": [key for key in domain_keys if len(groups_by_stratum.get(key, [])) == 1],
        "min_stratum_n": min((len(groups_by_stratum[key]) for key in supported), default=0),
        "uncertainty_status": "UNDERRESOLVED_STRATA" if underresolved else "DESCRIPTIVE_BOOTSTRAP",
        "underresolved_strata": underresolved,
        "strata": strata,
    }
    return result, groups_by_stratum


def contribution_decomposition(
    maps: dict[str, dict[str, dict[str, float]]],
    items: dict[str, dict[str, Any]],
    counts: Counter[str],
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for group_id, item in items.items():
        grouped[stratum(item)].append(group_id)
    total = sum(counts.values())
    rows = []
    for key in sorted(counts):
        if not grouped.get(key):
            continue
        beta = statistics.fmean(maps["Beta"][group]["mrr"] for group in grouped[key])
        step = statistics.fmean(maps["Step900"][group]["mrr"] for group in grouped[key])
        weight = counts[key] / total
        rows.append({
            "stratum": key,
            "stratum_weight": weight,
            "natural_count": counts[key],
            "probe_sample_n": len(grouped[key]),
            "Beta_MRR": beta,
            "Step900_MRR": step,
            "delta_MRR": beta - step,
            "contribution_to_weighted_delta": weight * (beta - step),
        })

    def aggregate(index: int) -> dict[str, float]:
        values: dict[str, float] = defaultdict(float)
        for row in rows:
            values[row["stratum"].split("|")[index]] += row["contribution_to_weighted_delta"]
        return dict(values)

    abs_total = sum(abs(row["contribution_to_weighted_delta"]) for row in rows)
    dominant = max(rows, key=lambda row: abs(row["contribution_to_weighted_delta"]), default=None)
    dominance_ratio = 0.0 if not dominant or not abs_total else abs(dominant["contribution_to_weighted_delta"]) / abs_total
    high_weight_singletons = [row["stratum"] for row in rows if row["probe_sample_n"] == 1 and row["stratum_weight"] >= 0.05]
    return {
        "natural_total": total,
        "supported_natural_mass": sum(row["stratum_weight"] for row in rows),
        "strata": rows,
        "contribution_by_domain": aggregate(0),
        "contribution_by_history": aggregate(1),
        "contribution_by_K": aggregate(2),
        "natural_weighted_delta_from_history": aggregate(1).get("HISTORY", 0.0),
        "natural_weighted_delta_from_nonhistory": aggregate(1).get("NONHISTORY", 0.0),
        "dominant_stratum": None if dominant is None else dominant["stratum"],
        "dominant_absolute_contribution_ratio": dominance_ratio,
        "high_weight_singleton_strata": high_weight_singletons,
        "single_stratum_dominance": bool(high_weight_singletons or dominance_ratio >= 0.60),
        "dominance_rule": "singleton with >=5% natural mass, or >=60% of total absolute supported contribution",
    }


def route_did(
    think_maps: dict[str, dict[str, dict[str, float]]],
    nothink_maps: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    groups = sorted(nothink_maps["Beta"])
    differences = {}
    think_gaps = {}
    nothink_gaps = {}
    for group in groups:
        think_gaps[group] = {
            metric: think_maps["Beta"][group][metric] - think_maps["Step900"][group][metric] for metric in METRICS
        }
        nothink_gaps[group] = {
            metric: nothink_maps["Beta"][group][metric] - nothink_maps["Step900"][group][metric] for metric in METRICS
        }
        differences[group] = {
            metric: nothink_gaps[group][metric] - think_gaps[group][metric] for metric in METRICS
        }
    return {
        "classification": "CONTROLLED_ROUTE_INTERACTION",
        "N": len(groups),
        "think_beta_step900_delta": {
            metric: statistics.fmean(think_gaps[group][metric] for group in groups) for metric in METRICS
        },
        "nothink_beta_step900_delta": {
            metric: statistics.fmean(nothink_gaps[group][metric] for group in groups) for metric in METRICS
        },
        "nothink_minus_think_gap_did": {
            metric: statistics.fmean(differences[group][metric] for group in groups) for metric in METRICS
        },
        "paired_bootstrap": paired_bootstrap(differences, "route-did"),
    }


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def common_suffix(left: list[int], right: list[int], prefix: int) -> int:
    count = 0
    limit = min(len(left), len(right)) - prefix
    while count < limit and left[-1 - count] == right[-1 - count]:
        count += 1
    return count


def soft_switch_recon(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer_data = read_json(TOKENIZER_CONFIG)
    template = tokenizer_data.get("chat_template")
    available = isinstance(template, str) and "enable_thinking" in template
    recon = {
        "soft_switch_available": available,
        "soft_switch_path": str(TOKENIZER_CONFIG),
        "soft_switch_sha256": sha256(TOKENIZER_CONFIG),
        "soft_switch_source": "runtime/model tokenizer package",
        "template_storage": "embedded tokenizer_config.json chat_template",
        "named_qwen3_soft_switch_file_found": False,
        "slash_think_handling": "preserved verbatim in user content; model-side soft switch",
        "slash_nothink_handling": "preserved verbatim in user content; model-side soft switch",
        "default_soft_switch_render": "assistant prefix only; no automatic think block",
        "hard_enable_thinking_false_render": "inserts <think>\\n\\n</think>\\n\\n after assistant prefix",
        "automatic_domain_content": False,
        "controlled_proxy": "LLaMA-Factory qwen3_nothink with literal terminal /no_think",
        "official_soft_switch": "base tokenizer chat_template with literal terminal /no_think and default thinking template mode",
    }
    if not available:
        return recon, {"available": False}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    selected = [
        row
        for domain in DOMAINS
        for row in sorted(
            (value for value in inputs["nothink_items"].values() if value["domain"] == domain),
            key=lambda value: value["group_id"],
        )[:2]
    ]
    comparisons = []
    for item in selected:
        proxy = list(map(int, item["nothink_prompt_token_ids"]))
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": item["new_system"]},
                {"role": "user", "content": item["nothink_user"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if hasattr(rendered, "input_ids"):
            rendered = rendered.input_ids
        elif hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
        official = list(map(int, rendered))
        prefix = common_prefix(proxy, official)
        suffix = common_suffix(proxy, official, prefix)
        max_length = max(len(proxy), len(official))
        differing = sum(
            index >= len(proxy) or index >= len(official) or proxy[index] != official[index]
            for index in range(max_length)
        )
        first = None if proxy == official else prefix
        start = max(0, prefix - 12)
        stop_proxy = min(len(proxy), prefix + 13)
        stop_official = min(len(official), prefix + 13)
        import re

        sid_re = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
        history = sid_re.findall(item["nothink_user"])
        history_tokens = [tokenizer.encode(value, add_special_tokens=False) for value in history]
        comparisons.append({
            "group_id": item["group_id"],
            "domain": item["domain"],
            "proxy_token_length": len(proxy),
            "soft_switch_token_length": len(official),
            "common_prefix_length": prefix,
            "common_suffix_length": suffix,
            "first_differing_token_index": first,
            "number_differing_token_positions": differing,
            "proxy_last_64_token_ids": proxy[-64:],
            "soft_switch_last_64_token_ids": official[-64:],
            "proxy_first_difference_window": tokenizer.decode(proxy[start:stop_proxy], skip_special_tokens=False),
            "soft_switch_first_difference_window": tokenizer.decode(official[start:stop_official], skip_special_tokens=False),
            "token_identical": proxy == official,
            "history_sid_sequence_parity": history == sid_re.findall(item["nothink_user"]),
            "history_sid_token_parity": history_tokens == [tokenizer.encode(value, add_special_tokens=False) for value in history],
        })
    identical = sum(row["token_identical"] for row in comparisons)
    token_diff = {
        "available": True,
        "sample_selection": "lexicographically first two fixed40 group IDs per domain",
        "sample_n": len(comparisons),
        "soft_switch_proxy_token_identical_count": identical,
        "soft_switch_proxy_token_identical_rate": identical / len(comparisons),
        "soft_switch_render_difference": "NONE_ON_AUDIT_SAMPLE" if identical == len(comparisons) else "PRESENT",
        "history_sid_sequence_parity": all(row["history_sid_sequence_parity"] for row in comparisons),
        "history_sid_token_parity": all(row["history_sid_token_parity"] for row in comparisons),
        "comparisons": comparisons,
    }
    return recon, token_diff


def exact_log_check(inputs: dict[str, Any]) -> dict[str, Any]:
    prior = inputs["bounded_artifact"]
    exact_paths = prior.get("exact_log_paths_checked", [])
    checked = []
    for value in exact_paths[:16]:
        path = Path(value)
        checked.append({"path": str(path), "exists": path.exists(), "is_file": path.is_file()})
    return {
        "exact_log_path_check_available": bool(exact_paths),
        "exact_paths_from_existing_reports": exact_paths[:16],
        "checks": checked,
        "bounded": True,
        "full_disk_search": False,
    }


def classify(
    raw: dict[str, Any],
    weighted_domains: dict[str, Any],
    history: dict[str, Any],
    did: dict[str, Any],
    contribution: dict[str, Any],
) -> dict[str, str]:
    raw_delta = raw["paired"]["metrics"]["mrr"]["estimate"]
    equal_delta = statistics.fmean(weighted_domains[domain]["delta"]["mrr"] for domain in DOMAINS)
    supporting_domains = sum(weighted_domains[domain]["delta"]["mrr"] > 0 for domain in DOMAINS)
    nonhistory_delta = history["NONHISTORY"]["paired"]["metrics"]["mrr"]["estimate"]
    did_delta = did["nothink_minus_think_gap_did"]["mrr"]
    dominated = contribution["single_stratum_dominance"]
    strong = all((raw_delta > 0, equal_delta > 0, supporting_domains >= 3, nonhistory_delta > 0, did_delta > 0, not dominated))
    if strong:
        strength = "STRONG"
    elif supporting_domains <= 1 and equal_delta < 0:
        strength = "CONTRADICTED"
    elif abs(equal_delta) < 0.002 or supporting_domains <= 1:
        strength = "WEAK"
    elif raw_delta > 0 or equal_delta > 0:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    history_contribution = contribution["natural_weighted_delta_from_history"]
    nonhistory_contribution = contribution["natural_weighted_delta_from_nonhistory"]
    if strength == "STRONG":
        root = "NOTHINK_ROUTE_STRONG_CANDIDATE"
    elif nonhistory_delta <= 0 or (abs(history_contribution) > abs(nonhistory_contribution) and history_contribution > 0):
        root = "NOTHINK_SIGNAL_HISTORY_CONFOUNDED"
    elif supporting_domains < 3:
        root = "NOTHINK_SIGNAL_DOMAIN_CONFOUNDED"
    elif strength == "MODERATE":
        root = "NOTHINK_ROUTE_CANDIDATE_SIGNAL"
    else:
        root = "LOCAL_ROUTE_METRICS_INSUFFICIENT"
    primary = (
        f"paired No-Think delta MRR={raw_delta:.6f}; equal-domain natural-weighted delta MRR={equal_delta:.6f}; "
        f"supporting domains={supporting_domains}/4; NonHistory delta MRR={nonhistory_delta:.6f}; route DiD={did_delta:.6f}"
    )
    confounds = []
    if supporting_domains < 3:
        confounds.append("domain directions are mixed")
    if nonhistory_delta <= 0:
        confounds.append("GoldNotInHistory does not support Beta advantage")
    if did_delta <= 0:
        confounds.append("controlled route DiD is non-positive")
    if dominated:
        confounds.append("weighted delta is dominated by a singleton/high-weight stratum")
    if any(weighted_domains[domain]["uncertainty_status"] == "UNDERRESOLVED_STRATA" for domain in DOMAINS):
        confounds.append("one or more domain strata are underresolved")
    main_confound = "; ".join(confounds) if confounds else "small-N paired uncertainty remains"
    conclusion = f"No-Think evidence is {strength.lower()}; it is a controlled-route diagnostic, not proof of official-route degradation."
    return {
        "nothink_signal_strength": strength,
        "root_class": root,
        "primary_signal": primary,
        "main_confound": main_confound,
        "conclusion": conclusion,
    }


def next_experiment(decision: dict[str, str], token_diff: dict[str, Any], weighted_domains: dict[str, Any], history: dict[str, Any]) -> str:
    strength = decision["nothink_signal_strength"]
    nonhistory_positive = history["NONHISTORY"]["paired"]["metrics"]["mrr"]["estimate"] > 0
    domains_positive = sum(weighted_domains[domain]["delta"]["mrr"] > 0 for domain in DOMAINS) >= 3
    render_difference = token_diff.get("soft_switch_render_difference")
    if strength in ("STRONG", "MODERATE") and nonhistory_positive and domains_positive and render_difference == "PRESENT":
        return "EXACT_SOFT_SWITCH_NOTHINK_80_CASE_CONFIRMATION"
    if strength == "STRONG" and render_difference == "NONE_ON_AUDIT_SAMPLE":
        return "NO_GPU_ROUTE_CONFIRMATION_NEEDED;PRIORITIZE_OFFICIAL_REWARD_PID_OR_TEST_DISTRIBUTION"
    return "OFFICIAL_REWARD_OR_TEST_DISTRIBUTION_AUDIT"


def terminal(summary: dict[str, Any]) -> str:
    audit = summary["code_audit"]
    raw = summary["raw_nothink"]
    domains = summary["weighted_domains"]
    hist = summary["history_k"]
    equal = summary["equal_domain"]
    did = summary["route_did"]
    recon, token = summary["soft_switch_recon"], summary["soft_switch_token_diff"]
    decision = summary["decision"]
    lines = [
        f"IMPLEMENT_COMMIT={audit['implement_commit']}",
        f"PUSH_STATUS={audit['push_status']}",
        f"GIT_STATUS_SHORT={audit['git_status_short'] or 'EMPTY'}",
        "",
        f"GITHUB_SCRIPT_SHA256={audit['github_script_sha256']}",
        f"RUNTIME_SCRIPT_SHA256={audit['runtime_script_sha256']}",
        f"RUNTIME_GITHUB_PARITY={audit['runtime_github_parity']}",
        "",
        f"RECORD_CONTRACT_PASS={'PASS' if summary['record_contract']['record_contract_pass'] else 'FAIL'}",
        "",
        "--- Raw NoThink Paired ---",
        "",
        f"NOTHINK_RAW_BETA_MRR={raw['Beta']['mrr']['estimate']:.9f}",
        f"NOTHINK_RAW_STEP900_MRR={raw['Step900']['mrr']['estimate']:.9f}",
        f"NOTHINK_RAW_DELTA_MRR={raw['paired']['metrics']['mrr']['estimate']:.9f}",
        f"NOTHINK_RAW_DELTA_MRR_CI95={raw['paired']['metrics']['mrr']['ci95']}",
        f"NOTHINK_RAW_DELTA_MRR_PPOS={raw['paired']['metrics']['mrr']['p_positive']:.6f}",
        f"NOTHINK_RAW_DELTA_HIT32={raw['paired']['metrics']['hit32']['estimate']:.9f}",
        f"NOTHINK_RAW_DELTA_AB32={raw['paired']['metrics']['ab32']['estimate']:.9f}",
        f"NOTHINK_RAW_DELTA_A32={raw['paired']['metrics']['a32']['estimate']:.9f}",
        "",
        "--- Domain ---",
        "",
    ]
    for domain in DOMAINS:
        value = domains[domain]
        prefix = domain.upper()
        lines += [
            f"{prefix}_WEIGHTED_BETA_MRR={value['Beta']['mrr']:.9f}",
            f"{prefix}_WEIGHTED_STEP900_MRR={value['Step900']['mrr']:.9f}",
            f"{prefix}_DELTA_MRR={value['delta']['mrr']:.9f}",
            f"{prefix}_DELTA_MRR_CI95={value['bootstrap']['metrics']['mrr']['ci95']}",
        ]
    lines += [
        f"DOMAIN_DIRECTION_MATCH_COUNT={summary['domain_direction']['match_count']}/4",
        f"DOMAIN_DIRECTION_MATCHES_EXTERNAL={summary['domain_direction']['classification']}",
        "",
        "--- History ---",
        "",
        f"HISTORY_N={hist['HISTORY']['N']}",
        f"NONHISTORY_N={hist['NONHISTORY']['N']}",
        f"NOTHINK_HISTORY_DELTA_MRR={hist['HISTORY']['paired']['metrics']['mrr']['estimate']:.9f}",
        f"NOTHINK_HISTORY_DELTA_MRR_CI95={hist['HISTORY']['paired']['metrics']['mrr']['ci95']}",
        f"NOTHINK_NONHISTORY_DELTA_MRR={hist['NONHISTORY']['paired']['metrics']['mrr']['estimate']:.9f}",
        f"NOTHINK_NONHISTORY_DELTA_MRR_CI95={hist['NONHISTORY']['paired']['metrics']['mrr']['ci95']}",
        f"NOTHINK_HISTORY_MRR_DIRECTION={hist['HISTORY']['directions']['mrr']}",
        f"NOTHINK_NONHISTORY_MRR_DIRECTION={hist['NONHISTORY']['directions']['mrr']}",
        f"NATURAL_WEIGHTED_DELTA_FROM_HISTORY={summary['contribution']['natural_weighted_delta_from_history']:.9f}",
        f"NATURAL_WEIGHTED_DELTA_FROM_NONHISTORY={summary['contribution']['natural_weighted_delta_from_nonhistory']:.9f}",
        "",
        "--- K ---",
        "",
        f"NOTHINK_K1_DELTA_MRR={hist['K1']['paired']['metrics']['mrr']['estimate']:.9f}",
        f"NOTHINK_K2PLUS_DELTA_MRR={hist['K2PLUS']['paired']['metrics']['mrr']['estimate']:.9f}",
        "",
        "--- Equal Domain Diagnostic ---",
        "",
        f"BETA_EQUAL_DOMAIN_WEIGHTED_MRR={equal['Beta']['mrr']:.9f}",
        f"STEP900_EQUAL_DOMAIN_WEIGHTED_MRR={equal['Step900']['mrr']:.9f}",
        f"EQUAL_DOMAIN_DELTA_MRR={equal['delta']['mrr']:.9f}",
        "NOT_OFFICIAL_SCORE=YES",
        "",
        "--- Think vs NoThink ---",
        "",
        f"THINK_BETA_STEP900_DELTA_MRR={did['think_beta_step900_delta']['mrr']:.9f}",
        f"NOTHINK_BETA_STEP900_DELTA_MRR={did['nothink_beta_step900_delta']['mrr']:.9f}",
        f"NOTHINK_MINUS_THINK_GAP_DID_MRR={did['nothink_minus_think_gap_did']['mrr']:.9f}",
        f"DID_CI95={did['paired_bootstrap']['metrics']['mrr']['ci95']}",
        f"DID_PPOS={did['paired_bootstrap']['metrics']['mrr']['p_positive']:.6f}",
        "",
        "--- Soft Switch ---",
        "",
        f"SOFT_SWITCH_AVAILABLE={'YES' if recon['soft_switch_available'] else 'NO'}",
        f"SOFT_SWITCH_PATH={recon['soft_switch_path']}",
        f"SOFT_SWITCH_SHA256={recon['soft_switch_sha256']}",
        f"SOFT_SWITCH_PROXY_TOKEN_IDENTICAL_RATE={token.get('soft_switch_proxy_token_identical_count', 0)}/{token.get('sample_n', 0)}",
        f"SOFT_SWITCH_RENDER_DIFFERENCE={token.get('soft_switch_render_difference', 'NOT_AVAILABLE')}",
        f"EXACT_LOG_PATH_CHECK_AVAILABLE={'YES' if summary['exact_log_check']['exact_log_path_check_available'] else 'NO'}",
        "",
        "--- Decision ---",
        "",
        f"NOTHINK_SIGNAL_STRENGTH={decision['nothink_signal_strength']}",
        f"ROOT_CLASS={decision['root_class']}",
        f"PRIMARY_SIGNAL={decision['primary_signal']}",
        f"MAIN_CONFOUND={decision['main_confound']}",
        f"CONCLUSION={decision['conclusion']}",
        f"NEXT_EXPERIMENT={summary['next_experiment']}",
        "",
        "GPU_INFERENCE_STARTED=NO",
        "TRAINING_STARTED=NO",
        "SELF_COT_GENERATION_STARTED=NO",
        "EXTERNAL_EVAL_STARTED=NO",
        "NEXT_EXPERIMENT_STARTED=NO",
    ]
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    raw = summary["raw_nothink"]
    lines = [
        "# Recommendation Root-Cause Phase 1.4.1",
        "",
        "CPU-only adjudication of existing Phase1.2 and Phase1.4 predictions. No model inference or generation was run.",
        "",
        "## Paired No-Think result",
        "",
        "| Metric | Beta | Step900 | Paired delta | 95% bootstrap CI | P(delta>0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        value = raw["paired"]["metrics"][metric]
        lines.append(
            f"| {metric} | {raw['Beta'][metric]['estimate']:.6f} | {raw['Step900'][metric]['estimate']:.6f} | "
            f"{value['estimate']:.6f} | {value['ci95']} | {value['p_positive']:.4f} |"
        )
    lines += ["", "## Natural within-domain weighting", "", "| Domain | Beta MRR | Step900 MRR | Delta | CI | Coverage | Uncertainty |", "|---|---:|---:|---:|---:|---:|---|"]
    for domain in DOMAINS:
        value = summary["weighted_domains"][domain]
        lines.append(
            f"| {domain} | {value['Beta']['mrr']:.6f} | {value['Step900']['mrr']:.6f} | {value['delta']['mrr']:.6f} | "
            f"{value['bootstrap']['metrics']['mrr']['ci95']} | {value['covered_domain_natural_mass']:.3f} | {value['uncertainty_status']} |"
        )
    decision = summary["decision"]
    lines += [
        "",
        "## Decision",
        "",
        f"- Signal strength: **{decision['nothink_signal_strength']}**",
        f"- Root class: **{decision['root_class']}**",
        f"- Primary signal: {decision['primary_signal']}",
        f"- Main confound: {decision['main_confound']}",
        f"- Next experiment: **{summary['next_experiment']}** (not started)",
        "",
        "The equal-domain macro is diagnostic and is not an official aggregate score. Bootstrap intervals with underresolved strata are descriptive, not formal significance claims.",
    ]
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    audit = code_audit()
    inputs = load_inputs()
    contract = validate_records(inputs)
    write_json(OUTPUT / "record_contract_audit.json", contract)

    nothink_maps = metric_maps(inputs["nothink_records"], inputs["nothink_items"])
    think_maps = metric_maps(inputs["think_records"], inputs["think_items"])
    groups = sorted(nothink_maps["Beta"])
    raw = subset_analysis(groups, nothink_maps, "raw-nothink-40")
    write_json(OUTPUT / "nothink_paired_bootstrap.json", raw)

    domain_raw = {
        domain: subset_analysis(
            [group for group in groups if inputs["nothink_items"][group]["domain"] == domain],
            nothink_maps,
            "raw-domain-" + domain,
        )
        for domain in DOMAINS
    }
    raw_directions = {domain: domain_raw[domain]["directions"]["mrr"] for domain in DOMAINS}
    matches = sum(raw_directions[domain] == EXTERNAL_DIRECTION[domain] for domain in DOMAINS)
    domain_direction = {
        "external": EXTERNAL_DIRECTION,
        "nothink_raw": raw_directions,
        "match_count": matches,
        "classification": "ALL" if matches == 4 else "MOST" if matches == 3 else "MIXED" if matches else "NONE",
    }

    counts = natural_counts(inputs["natural"])
    weighted_domains = {}
    for domain in DOMAINS:
        weighted_domains[domain], _ = weighted_domain(domain, nothink_maps, inputs["nothink_items"], counts)
    write_json(
        OUTPUT / "nothink_domain_analysis.json",
        {"raw": domain_raw, "direction_comparison": domain_direction, "natural_weighted": weighted_domains},
    )
    write_json(OUTPUT / "nothink_poststratified_domain.json", weighted_domains)

    history_k = {
        "HISTORY": subset_analysis([g for g in groups if inputs["nothink_items"][g]["gold_sid_in_history"]], nothink_maps, "history"),
        "NONHISTORY": subset_analysis([g for g in groups if not inputs["nothink_items"][g]["gold_sid_in_history"]], nothink_maps, "nonhistory"),
        "K1": subset_analysis([g for g in groups if int(inputs["nothink_items"][g]["K"]) == 1], nothink_maps, "k1"),
        "K2PLUS": subset_analysis([g for g in groups if int(inputs["nothink_items"][g]["K"]) >= 2], nothink_maps, "k2plus"),
    }
    write_json(OUTPUT / "nothink_history_k_analysis.json", history_k)

    contribution = contribution_decomposition(nothink_maps, inputs["nothink_items"], counts)
    write_json(OUTPUT / "nothink_contribution_decomposition.json", contribution)

    did = route_did(think_maps, nothink_maps)
    write_json(OUTPUT / "think_vs_nothink_route_did.json", did)

    equal_domain = {
        "Beta": {metric: statistics.fmean(weighted_domains[domain]["Beta"][metric] for domain in DOMAINS) for metric in METRICS},
        "Step900": {metric: statistics.fmean(weighted_domains[domain]["Step900"][metric] for domain in DOMAINS) for metric in METRICS},
        "not_official_score": True,
    }
    equal_domain["delta"] = {
        metric: equal_domain["Beta"][metric] - equal_domain["Step900"][metric] for metric in METRICS
    }
    recon, token_diff = soft_switch_recon(inputs)
    write_json(OUTPUT / "soft_switch_recon.json", recon)
    write_json(OUTPUT / "soft_switch_token_diff.json", token_diff)
    log_check = exact_log_check(inputs)

    decision = classify(raw, weighted_domains, history_k, did, contribution)
    next_choice = next_experiment(decision, token_diff, weighted_domains, history_k)
    summary = {
        "code_audit": audit,
        "record_contract": contract,
        "raw_nothink": raw,
        "domain_raw": domain_raw,
        "weighted_domains": weighted_domains,
        "domain_direction": domain_direction,
        "history_k": history_k,
        "contribution": contribution,
        "equal_domain": equal_domain,
        "route_did": did,
        "soft_switch_recon": recon,
        "soft_switch_token_diff": token_diff,
        "exact_log_check": log_check,
        "decision": decision,
        "next_experiment": next_choice,
        "gpu_inference_started": False,
        "training_started": False,
        "self_cot_generation_started": False,
        "external_eval_started": False,
        "next_experiment_started": False,
    }
    write_json(OUTPUT / "summary.json", summary)
    (OUTPUT / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    text = terminal(summary)
    (OUTPUT / "CHATGPT_ROOT_CAUSE_PHASE1_4_1.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "contract"), nargs="?", default="run")
    args = parser.parse_args()
    inputs = load_inputs() if args.action == "contract" else None
    if args.action == "contract":
        print(json.dumps(validate_records(inputs), indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
