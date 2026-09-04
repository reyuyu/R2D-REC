#!/usr/bin/env python3
"""Build frozen or leak-clean V4.3-HCR views and train-fitted metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_SOURCE = Path("/data/LLm-8B/code/train/data/rec_fdr_v42_interestonly")
DEFAULT_OUTPUT = Path("/mnt/wanqing-training-runs/datasets/rec_fdr_v43_hcr_frozen_v42a")
FULL_TRAIN = [f"rec_full_{bucket}_train.jsonl" for bucket in ("k1", "k2", "k3")]
FULL_FILES = FULL_TRAIN + ["rec_full_val.jsonl"]
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SID_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
DOMAINS = ("video", "prod", "ad", "living")
LEVELS = ("a", "b", "c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-artifacts", type=Path)
    parser.add_argument("--leak-policy", choices=("preserve", "drop"), default="preserve")
    return parser.parse_args()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def sid_tuple(value: str) -> tuple[int, int, int]:
    match = SID_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid canonical SID: {value!r}")
    return int(match["a"]), int(match["b"]), int(match["c"])


def sid_text(match: re.Match[str]) -> str:
    return match.group(0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_gold_in_cot(row: Mapping[str, Any]) -> bool:
    match = THINK_RE.search(str(row["response"]))
    if match is None:
        raise ValueError(f"row {row.get('prompt_group_id')} has no complete think block")
    return str(row["final_target_sid"]) in match.group(1)


def filter_full_file(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_rows = kept = deleted = 0
    deleted_by_domain: Counter[str] = Counter()
    deleted_examples: list[dict[str, str]] = []
    with source.open(encoding="utf-8") as source_handle, target.open("w", encoding="utf-8") as target_handle:
        for line_number, line in enumerate(source_handle, 1):
            row = json.loads(line)
            source_rows += 1
            if exact_gold_in_cot(row):
                deleted += 1
                deleted_by_domain[str(row["target_domain"])] += 1
                if len(deleted_examples) < 20:
                    deleted_examples.append(
                        {
                            "file": source.name,
                            "line": str(line_number),
                            "prompt_group_id": str(row["prompt_group_id"]),
                            "target_domain": str(row["target_domain"]),
                            "final_target_sid": str(row["final_target_sid"]),
                        }
                    )
                continue
            target_handle.write(line)
            kept += 1
    return {
        "source_rows": source_rows,
        "kept_rows": kept,
        "deleted_rows": deleted,
        "deleted_by_domain": dict(sorted(deleted_by_domain.items())),
        "deleted_examples": deleted_examples,
    }


def preserve_full_file(source: Path, target: Path) -> dict[str, Any]:
    source_rows = observed_leaks = 0
    observed_by_domain: Counter[str] = Counter()
    observed_examples: list[dict[str, str]] = []
    for line_number, row in enumerate(rows(source), 1):
        source_rows += 1
        if exact_gold_in_cot(row):
            observed_leaks += 1
            observed_by_domain[str(row["target_domain"])] += 1
            if len(observed_examples) < 20:
                observed_examples.append(
                    {
                        "file": source.name,
                        "line": str(line_number),
                        "prompt_group_id": str(row["prompt_group_id"]),
                        "target_domain": str(row["target_domain"]),
                        "final_target_sid": str(row["final_target_sid"]),
                    }
                )
    shutil.copy2(source, target)
    return {
        "source_rows": source_rows,
        "kept_rows": source_rows,
        "deleted_rows": 0,
        "observed_gold_in_cot_rows": observed_leaks,
        "observed_by_domain": dict(sorted(observed_by_domain.items())),
        "observed_examples": observed_examples,
    }


def _domain_line(prompt: str, domain: str) -> str:
    marker = f"<|{domain}_begin|>"
    return next((line for line in prompt.splitlines() if marker in line), "")


def _sids_in(text: str, domain: str | None = None) -> list[tuple[int, int, int]]:
    return [
        (int(match["a"]), int(match["b"]), int(match["c"]))
        for match in SID_RE.finditer(text)
        if domain is None or match["domain"] == domain
    ]


def _capture(pattern: str, text: str, domain: str) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for match in re.finditer(pattern, text):
        result.extend(_sids_in(match.group(1), domain))
    return result


def behavior_occurrences(line: str, domain: str) -> dict[str, list[tuple[int, int, int]]]:
    """Parse the frozen prompt grammar once during dataset preparation."""
    found: defaultdict[str, list[tuple[int, int, int]]] = defaultdict(list)
    if domain == "video":
        patterns = {
            "深度观看": r"深度观看了(.+?)(?=，(?:普通观看|观看|看过|对视频|长播|点赞|收藏|关注)|。|$)",
            "普通观看": r"(?<!深度)(?:普通观看了|观看了|看过)(.+?)(?=，(?:深度观看|对视频|长播|点赞|收藏|关注)|。|$)",
            "长播": r"长播(?:了)?(.+?)(?=，(?:深度观看|普通观看|观看|看过|对视频|点赞|收藏|关注)|。|$)",
            "点赞": r"点赞(?:了)?(.+?)(?=，(?:深度观看|普通观看|观看|看过|对视频|长播|收藏|关注)|。|$)",
            "收藏": r"收藏(?:了)?(.+?)(?=，(?:深度观看|普通观看|观看|看过|对视频|长播|点赞|关注)|。|$)",
            "关注": r"关注(?:了)?(.+?)(?=，(?:深度观看|普通观看|观看|看过|对视频|长播|点赞|收藏)|。|$)",
        }
        for behavior, pattern in patterns.items():
            found[behavior].extend(_capture(pattern, line, domain))
        for match in re.finditer(r"对视频(.+?)有过(.+?)行为", line):
            values = _sids_in(match.group(1), domain)
            for behavior in ("长播", "点赞", "收藏", "关注", "普通观看", "深度观看"):
                if behavior in match.group(2):
                    found[behavior].extend(values)
    elif domain == "prod":
        patterns = {
            "浏览": r"浏览了商品(.+?)(?=，(?:购买|加购)|。|$)",
            "购买": r"购买了(?:商品)?(.+?)(?=，(?:浏览|加购)|。|$)",
            "加购": r"加购了(?:商品)?(.+?)(?=，(?:浏览|购买)|。|$)",
        }
        for behavior, pattern in patterns.items():
            found[behavior].extend(_capture(pattern, line, domain))
    elif domain == "ad":
        found["深度转化"].extend(_capture(r"对广告(.+?)完成过深度转化", line, domain))
        found["点击"].extend(_capture(r"点击过广告(.+?)(?=。|$)", line, domain))
    elif domain == "living":
        patterns = {
            "关注": r"关注了主播(.+?)(?=，(?:首次打赏|打赏|对主播)|。|$)",
            "首次打赏": r"首次打赏了(.+?)(?=，(?:关注|打赏|对主播)|。|$)",
            "打赏": r"(?<!首次)打赏了(.+?)(?=，(?:关注|首次打赏|对主播)|。|$)",
        }
        for behavior, pattern in patterns.items():
            found[behavior].extend(_capture(pattern, line, domain))
        for match in re.finditer(r"对主播(.+?)有过(.+?)行为", line):
            values = _sids_in(match.group(1), domain)
            for behavior in ("关注", "首次打赏", "打赏"):
                if behavior in match.group(2):
                    found[behavior].extend(values)
    else:
        raise ValueError(domain)

    assigned = Counter(sid for values in found.values() for sid in values)
    all_values = _sids_in(line, domain)
    missing = list((Counter(all_values) - assigned).elements())
    if missing:
        found["unknown"].extend(missing)
    return {behavior: values for behavior, values in found.items() if values}


def _level_counts(values: Iterable[tuple[int, int, int]]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {level: Counter() for level in LEVELS}
    for a, b, c in values:
        counters["a"][str(a)] += 1
        counters["b"][f"{a},{b}"] += 1
        counters["c"][f"{a},{b},{c}"] += 1
    return {level: dict(sorted(counter.items())) for level, counter in counters.items()}


def _behavior_level_counts(
    behavior_values: Mapping[str, Iterable[tuple[int, int, int]]]
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {level: {} for level in LEVELS}
    for behavior, values in sorted(behavior_values.items()):
        counts = _level_counts(values)
        for level in LEVELS:
            result[level][behavior] = counts[level]
    return result


def _novelty(anchor: tuple[int, int, int], history: Iterable[tuple[int, int, int]]) -> str:
    values = set(history)
    if anchor[0] not in {item[0] for item in values}:
        return "T0"
    if anchor[:2] not in {item[:2] for item in values}:
        return "T1"
    if anchor not in values:
        return "T2"
    return "T3"


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0.0, "mean": 0.0, "p80": 0.0, "p95": 0.0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p80": percentile(values, 0.80),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def load_prompt_histories(source: Path) -> dict[str, dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for name in FULL_FILES:
        for row in rows(source / name):
            prompt_group_id = str(row["prompt_group_id"])
            prompt = str(row["prompt"])
            current = histories.get(prompt_group_id)
            if current is not None:
                if current["prompt"] != prompt:
                    raise ValueError(f"prompt changed within group {prompt_group_id}")
                continue
            per_domain: dict[str, Any] = {}
            for domain in DOMAINS:
                line = _domain_line(prompt, domain)
                values = _sids_in(line, domain)
                per_domain[domain] = {
                    "sids": values,
                    "behavior": behavior_occurrences(line, domain) if line else {},
                }
            histories[prompt_group_id] = {
                "prompt": prompt,
                "all_history_len": len(_sids_in(prompt)),
                "domains": per_domain,
            }
    return histories


def fit_behavior_reliability(
    cleaned_root: Path,
    histories: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    support: defaultdict[tuple[str, str], int] = defaultdict(int)
    hits: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for name in FULL_TRAIN:
        for row in rows(cleaned_root / name):
            domain = str(row["target_domain"])
            anchor = sid_tuple(str(row["final_target_sid"]))
            behavior = histories[str(row["prompt_group_id"])]["domains"][domain]["behavior"]
            for behavior_name, values in behavior.items():
                support[(domain, behavior_name)] += 1
                value_set = set(values)
                behavior_hit = {
                    "a": anchor[0] in {item[0] for item in value_set},
                    "b": anchor[:2] in {item[:2] for item in value_set},
                    "c": anchor in value_set,
                }
                for level, value in behavior_hit.items():
                    hits[(domain, behavior_name, level)] += int(value)

    weights: dict[str, dict[str, dict[str, float]]] = {domain: {} for domain in DOMAINS}
    audit: dict[str, Any] = {domain: {} for domain in DOMAINS}
    for domain in DOMAINS:
        behaviors = sorted(behavior for local_domain, behavior in support if local_domain == domain)
        effective_rates: dict[str, dict[str, float]] = {behavior: {} for behavior in behaviors}
        for behavior in behaviors:
            n = support[(domain, behavior)]
            audit[domain][behavior] = {"support_rows": n, "levels": {}}
            for level in LEVELS:
                behavior_hits = hits[(domain, behavior, level)]
                raw_rate = behavior_hits / n if n else 0.0
                smoothed_rate = (behavior_hits + 1.0) / (n + 2.0)
                shrinkage = n / (n + 20.0)
                effective_rates[behavior][level] = smoothed_rate * shrinkage
                audit[domain][behavior]["levels"][level] = {
                    "hit_rows": behavior_hits,
                    "raw_hit_rate": raw_rate,
                    "smoothed_hit_rate": smoothed_rate,
                    "shrinkage": shrinkage,
                }
        for level in LEVELS:
            maximum = max(
                (effective_rates[behavior][level] for behavior in behaviors if behavior != "unknown"),
                default=0.0,
            )
            for behavior in behaviors:
                weights[domain].setdefault(behavior, {})
                final = 0.0 if behavior == "unknown" or maximum <= 0.0 else effective_rates[behavior][level] / maximum
                weights[domain][behavior][level] = final
                audit[domain][behavior]["levels"][level]["domain_level_max_effective_rate"] = maximum
                audit[domain][behavior]["levels"][level]["final_nonnegative_weight"] = final
    return weights, audit


def build_metadata(
    source: Path,
    cleaned_root: Path,
    histories: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = json.loads((source / "rec_group_catalog.json").read_text(encoding="utf-8"))
    reliability, reliability_audit = fit_behavior_reliability(cleaned_root, histories)
    group_metadata: dict[str, str] = {}
    history_records: dict[str, Any] = {}
    split_counts: Counter[str] = Counter()
    for split_key, split in (("train_groups", "train"), ("validation_groups", "validation")):
        for item in catalog[split_key]:
            group_index = int(item["group_index"])
            prompt_group_id = str(item["prompt_group_id"])
            domain = str(item["domain"])
            history = histories[prompt_group_id]
            target_values = history["domains"][domain]["sids"]
            behavior = history["domains"][domain]["behavior"]
            record_key = f"{prompt_group_id}|{domain}"
            record = {
                "prompt_group_id": prompt_group_id,
                "target_domain": domain,
                "all_history_len": int(history["all_history_len"]),
                "target_history_len": len(target_values),
                "target_history_n_sa": len({value[0] for value in target_values}),
                "target_history_n_sab": len({value[:2] for value in target_values}),
                "target_history_n_sabc": len(set(target_values)),
                "target_history_level_counts": _level_counts(target_values),
                "behavior_level_counts": _behavior_level_counts(behavior),
            }
            if record_key in history_records and history_records[record_key] != record:
                raise ValueError(f"history metadata changed for {record_key}")
            history_records[record_key] = record
            group_metadata[str(group_index)] = record_key
            split_counts[split] += 1
    metadata = {
        "schema_version": 1,
        "description": "Train-fitted V4.3 HCR metadata; catalog is auxiliary-only and is never a generation mask.",
        "source_dataset": str(source),
        "behavior_reliability_split": "train_only",
        "groups": group_metadata,
        "history_records": history_records,
        "behavior_reliability": reliability,
        "behavior_reliability_audit": reliability_audit,
    }
    report = {
        "schema_version": 1,
        "group_counts": dict(sorted(split_counts.items())),
        "metadata_group_count": len(group_metadata),
        "unique_history_record_count": len(history_records),
        "unknown_behavior_sid_occurrences": sum(
            sum(item["behavior_level_counts"]["c"].get("unknown", {}).values())
            for item in history_records.values()
        ),
        "train_only_behavior_reliability": True,
        "generation_mask": False,
        "extra_transformer_forwards": 0,
    }
    return metadata, report


def preflight_report(
    source: Path,
    dataset_root: Path,
    histories: Mapping[str, Mapping[str, Any]],
    leak_report: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_display_path: Path | None = None,
) -> dict[str, Any]:
    domain_counts: Counter[str] = Counter()
    novelty_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    overlap: defaultdict[str, Counter[str]] = defaultdict(Counter)
    h_values: defaultdict[str, list[int]] = defaultdict(list)
    nsa_values: defaultdict[str, list[int]] = defaultdict(list)
    prompt_groups: set[str] = set()
    k_counts: Counter[str] = Counter()
    cleaned_rows = 0
    behavior_coverage: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for name in FULL_TRAIN:
        for row in rows(dataset_root / name):
            cleaned_rows += 1
            domain = str(row["target_domain"])
            prompt_group_id = str(row["prompt_group_id"])
            prompt_groups.add(prompt_group_id)
            domain_counts[domain] += 1
            k_counts[str(row["future_entropy_bucket"])] += 1
            anchor = sid_tuple(str(row["final_target_sid"]))
            history = histories[prompt_group_id]
            values = history["domains"][domain]["sids"]
            value_set = set(values)
            current_novelty = _novelty(anchor, values)
            novelty_counts[domain][current_novelty] += 1
            overlap[domain]["sa"] += int(anchor[0] in {item[0] for item in value_set})
            overlap[domain]["sab"] += int(anchor[:2] in {item[:2] for item in value_set})
            overlap[domain]["sabc"] += int(anchor in value_set)
            h_values[domain].append(int(history["all_history_len"]))
            nsa_values[domain].append(len({item[0] for item in values}))
            for behavior in history["domains"][domain]["behavior"]:
                behavior_coverage[domain][behavior] += 1
    domain_report: dict[str, Any] = {}
    for domain in DOMAINS:
        count = domain_counts[domain]
        domain_report[domain] = {
            "rows": count,
            "row_fraction": count / cleaned_rows if cleaned_rows else 0.0,
            "overlap": {
                level: {"count": overlap[domain][level], "rate": overlap[domain][level] / count if count else 0.0}
                for level in ("sa", "sab", "sabc")
            },
            "novelty_counts": dict(novelty_counts[domain]),
            "all_history_h": distribution(h_values[domain]),
            "target_history_n_sa": distribution(nsa_values[domain]),
            "behavior_coverage": {
                behavior: {"count": value, "rate": value / count if count else 0.0}
                for behavior, value in sorted(behavior_coverage[domain].items())
            },
        }
    catalog = json.loads((source / "rec_group_catalog.json").read_text(encoding="utf-8"))
    train_full_groups = [item for item in catalog["train_groups"] if str(item["group_key"]).startswith("full:")]
    multi_positive = Counter("k1" if int(item["k"]) == 1 else "k2" if int(item["k"]) <= 4 else "k3" for item in train_full_groups)
    file_rows = {}
    for path in sorted(dataset_root.glob("*.jsonl")):
        file_rows[path.name] = sum(1 for _ in path.open(encoding="utf-8"))
    return {
        "schema_version": 1,
        "source_dataset": str(source),
        "output_dataset": str(output_display_path or dataset_root),
        "source_full_train_rows": sum(int(leak_report["per_file"][name]["source_rows"]) for name in FULL_TRAIN),
        "output_full_train_rows": cleaned_rows,
        "output_full_train_prompt_groups": len(prompt_groups),
        "full_train_domain_counts": dict(sorted(domain_counts.items())),
        "full_train_k_bucket_counts": dict(sorted(k_counts.items())),
        "full_train_group_k_distribution": dict(sorted(multi_positive.items())),
        "domain": domain_report,
        "behavior_reliability_audit": metadata["behavior_reliability_audit"],
        "gold_sid_leak_cleaning": leak_report,
        "all_jsonl_file_rows": file_rows,
        "candidate_catalog_split": "train_only for training auxiliaries; validation inventory is never merged",
    }


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V4.3-HCR Preflight Distribution Report",
        "",
        f"- Source Full train rows: `{report['source_full_train_rows']}`",
        f"- Output Full train rows: `{report['output_full_train_rows']}`",
        f"- Output prompt groups: `{report['output_full_train_prompt_groups']}`",
        f"- Gold-in-CoT policy: `{report['gold_sid_leak_cleaning']['policy']}`",
        f"- Observed exact gold-in-CoT train rows: `{report['gold_sid_leak_cleaning']['train_observed_gold_in_cot_rows']}`",
        f"- Deleted exact gold-in-CoT train rows: `{report['gold_sid_leak_cleaning']['train_deleted_rows']}`",
        "- Candidate catalog: train-only for training auxiliaries; no generation-time mask.",
        "",
        "| Domain | Rows | SA in H | SAB in H | SABC in H | H median | N_SA median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        item = report["domain"][domain]
        lines.append(
            f"| {domain} | {item['rows']} | {item['overlap']['sa']['rate']:.2%} | "
            f"{item['overlap']['sab']['rate']:.2%} | {item['overlap']['sabc']['rate']:.2%} | "
            f"{item['all_history_h']['median']:.1f} | {item['target_history_n_sa']['median']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Behavior reliability is fitted only on the output train split with Beta(1,1) smoothing,",
            "support shrinkage and nonnegative domain-level normalization; unknown behavior is fixed to zero.",
            "",
        ]
    )
    return "\n".join(lines)


def build(source: Path, output: Path, review_artifacts: Path | None, leak_policy: str) -> dict[str, Any]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"stale temporary directory exists: {temporary}")
    temporary.mkdir(parents=True)

    per_file: dict[str, Any] = {}
    try:
        for path in sorted(source.iterdir()):
            target = temporary / path.name
            if path.name in FULL_FILES:
                per_file[path.name] = (
                    preserve_full_file(path, target)
                    if leak_policy == "preserve"
                    else filter_full_file(path, target)
                )
            elif path.is_file():
                shutil.copy2(path, target)
        train_deleted = sum(per_file[name]["deleted_rows"] for name in FULL_TRAIN)
        val_deleted = per_file["rec_full_val.jsonl"]["deleted_rows"]
        leak_report = {
            "schema_version": 1,
            "policy": leak_policy,
            "rule": (
                "audit only; preserve every V4.2-A row byte-for-byte"
                if leak_policy == "preserve"
                else "delete Full row iff final_target_sid is an exact substring of the response <think>...</think> block"
            ),
            "train_deleted_rows": train_deleted,
            "validation_deleted_rows": val_deleted,
            "train_observed_gold_in_cot_rows": sum(
                int(per_file[name].get("observed_gold_in_cot_rows", per_file[name]["deleted_rows"]))
                for name in FULL_TRAIN
            ),
            "validation_observed_gold_in_cot_rows": int(
                per_file["rec_full_val.jsonl"].get(
                    "observed_gold_in_cot_rows", per_file["rec_full_val.jsonl"]["deleted_rows"]
                )
            ),
            "per_file": per_file,
        }
        histories = load_prompt_histories(source)
        metadata, metadata_report = build_metadata(source, temporary, histories)
        (temporary / "hcr_group_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        (temporary / "gold_sid_leak_report.json").write_text(
            json.dumps(leak_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "hcr_metadata_report.json").write_text(
            json.dumps(metadata_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        preflight = preflight_report(
            source, temporary, histories, leak_report, metadata, output_display_path=output
        )
        (temporary / "preflight_distribution_report.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "preflight_distribution_report.md").write_text(report_markdown(preflight), encoding="utf-8")

        remaining_leaks = []
        for name in FULL_FILES:
            for index, row in enumerate(rows(temporary / name), 1):
                if exact_gold_in_cot(row):
                    remaining_leaks.append({"file": name, "line": index})
        if leak_policy == "drop" and remaining_leaks:
            raise AssertionError(f"remaining exact gold-in-CoT leaks: {remaining_leaks[:5]}")
        expected_remaining = (
            leak_report["train_observed_gold_in_cot_rows"]
            + leak_report["validation_observed_gold_in_cot_rows"]
            if leak_policy == "preserve"
            else 0
        )
        if len(remaining_leaks) != expected_remaining:
            raise AssertionError(
                f"gold-in-CoT audit mismatch: expected {expected_remaining}, found {len(remaining_leaks)}"
            )
        manifest_files = {}
        for path in sorted(temporary.iterdir()):
            if path.is_file() and path.name != "dataset_manifest.json":
                manifest_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "schema_version": 1,
            "dataset_id": (
                "rec_fdr_v43_hcr_frozen_v42a"
                if leak_policy == "preserve"
                else "rec_fdr_v43_hcr_goldsid_leakclean"
            ),
            "parent_dataset": str(source),
            "changes": [
                (
                    "Preserved every V4.2-A Full/R1/R2 row byte-for-byte; gold-in-CoT is audit-only."
                    if leak_policy == "preserve"
                    else "Deleted Full train/validation rows whose own final target SID occurs exactly inside CoT."
                ),
                "Added group-indexed target-domain history and train-fitted behavior metadata for HCR.",
                "R1 and R2 payload files are byte-identical to V4.2-A.",
            ],
            "gold_sid_leak_rule": leak_report["rule"],
            "gold_sid_leak_policy": leak_policy,
            "training_candidate_catalog": "train split only; auxiliary ranking only; never a generation mask",
            "files": manifest_files,
        }
        (temporary / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    if review_artifacts is not None:
        review_artifacts.mkdir(parents=True, exist_ok=True)
        for name in (
            "dataset_manifest.json",
            "gold_sid_leak_report.json",
            "hcr_metadata_report.json",
            "preflight_distribution_report.json",
            "preflight_distribution_report.md",
        ):
            shutil.copy2(output / name, review_artifacts / name)
        sample_path = review_artifacts / "review_samples.jsonl"
        samples = []
        for name in FULL_TRAIN:
            for row in rows(output / name):
                if row["target_domain"] not in {item["target_domain"] for item in samples}:
                    samples.append(row)
                if len(samples) == 4:
                    break
            if len(samples) == 4:
                break
        with sample_path.open("w", encoding="utf-8") as handle:
            for row in samples:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "status": "built",
        "output": str(output),
        "train_deleted_rows": sum(per_file[name]["deleted_rows"] for name in FULL_TRAIN),
        "validation_deleted_rows": per_file["rec_full_val.jsonl"]["deleted_rows"],
        "leak_policy": leak_policy,
    }


def main() -> None:
    args = parse_args()
    result = build(
        args.source.resolve(),
        args.output.resolve(),
        args.review_artifacts.resolve() if args.review_artifacts else None,
        args.leak_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
