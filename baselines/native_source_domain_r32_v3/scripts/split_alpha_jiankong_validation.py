#!/usr/bin/env python3
"""Create the deterministic, leakage-safe alpha-jiankong 98/2 split.

This tool is deliberately data-only: it never imports a model, trainer, loss,
or sampler.  Rows are emitted verbatim from the source JSONL, preserving the
training/monitor metadata byte-for-byte inside each JSON record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPLIT_SEED = "20260812"
ALGORITHM_VERSION = "alpha-leak-safe-v1"
DEV_RATE = 0.02
REQUIRED_FIELDS = (
    "system", "instruction", "input", "output", "history", "data_source", "source_segment", "aux_metadata_json"
)
REC_METADATA_FIELDS = (
    "recommendation_group_id", "recommendation_group_size", "recommendation_current_gold_sid", "recommendation_all_gold_sids"
)
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
THINK_SUFFIX_RE = re.compile(r"\s*/(?:no_)?think\s*$", re.IGNORECASE)


class SplitError(RuntimeError):
    pass


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            if left < right:
                self.parent[right] = left
            else:
                self.parent[left] = right


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text(value: Any, *, strip_think_suffix: bool = False) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    if strip_think_suffix:
        text = THINK_SUFFIX_RE.sub("", text)
    return "\n".join(part.strip() for part in text.split("\n") if part.strip())


def prompt_key(row: dict[str, Any]) -> str:
    """Canonical complete supervised prompt; think/no-think suffixes coalesce."""

    payload = {
        field: canonical_text(row.get(field, ""), strip_think_suffix=field in {"instruction", "input"})
        for field in ("system", "instruction", "input", "history")
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def user_family_key(row: dict[str, Any]) -> str:
    """History/input identity, intentionally independent of CoT/no-think wording."""

    payload = {
        "input": canonical_text(row.get("input", ""), strip_think_suffix=True),
        "history": canonical_text(row.get("history", [])),
    }
    if not payload["input"] and payload["history"] in {"", "[]"}:
        payload["fallback_prompt"] = prompt_key(row)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def rec_history_domain_key(row: dict[str, Any], metadata: dict[str, Any], domain: str) -> str:
    """Independent recommendation audit identity: ordered prompt SID history + domain."""

    history_text = "\n".join(
        (canonical_text(row.get("system", "")), canonical_text(row.get("instruction", ""), strip_think_suffix=True),
         canonical_text(row.get("input", ""), strip_think_suffix=True), canonical_text(row.get("history", [])))
    )
    sid_history = SID_RE.findall(history_text)
    # Keep full SID strings, not only domains; `finditer` avoids lossy cross-SID joins.
    full_sid_history = [match.group(0) for match in SID_RE.finditer(history_text)]
    if not full_sid_history:
        raise SplitError("Recommendation row has no SID history for canonical history/domain audit.")
    payload = {"sid_history": full_sid_history, "target_domain": domain}
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def domain_from_current_gold(value: Any) -> str:
    if not isinstance(value, str):
        raise SplitError("recommendation_current_gold_sid must be a SID string.")
    match = SID_RE.fullmatch(value)
    if not match:
        raise SplitError(f"Invalid recommendation_current_gold_sid: {value!r}")
    return match.group(1)


def size_bin(value: int) -> str:
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


@dataclass(frozen=True)
class Row:
    index: int
    task: str
    source: str
    source_segment: str
    prompt: str
    row_hash: str
    family_node: str
    rec_group_id: str | None = None
    rec_history_domain: str | None = None
    rec_domain: str | None = None
    rec_size: int | None = None


def task_for(row: dict[str, Any]) -> str:
    source = str(row["data_source"])
    segment = str(row["source_segment"])
    if source == "recommend":
        return "recommendation"
    if source in {"material_sample", "sid_bucket_canonical_no_think", "sid_bucket_reverse"}:
        return "material"
    if source == "understand_user" and segment == "user_action":
        return "user_action"
    if source == "understand_user" and segment in {"user_chain_cot", "user_chain_nocot"}:
        return "user_chain"
    raise SplitError(f"Unknown task route: data_source={source!r}, source_segment={segment!r}")


def load_rows(path: Path) -> tuple[list[Row], DSU]:
    rows: list[Row] = []
    dsu = DSU()
    required_set = set(REQUIRED_FIELDS)
    with path.open("r", encoding="utf-8") as source:
        for index, raw_line in enumerate(source):
            raw = raw_line.rstrip("\n")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise SplitError(f"Invalid JSON at source row {index + 1}.") from error
            missing = required_set - set(value)
            if missing:
                raise SplitError(f"Source row {index + 1} missing required fields: {sorted(missing)}")
            task = task_for(value)
            prompt = prompt_key(value)
            source_name, source_segment = str(value["data_source"]), str(value["source_segment"])
            rec_group_id = rec_hdom = rec_domain = None
            rec_size: int | None = None
            if task == "recommendation":
                try:
                    metadata = json.loads(value["aux_metadata_json"])
                except json.JSONDecodeError as error:
                    raise SplitError(f"Recommendation row {index + 1} has invalid aux_metadata_json.") from error
                if not isinstance(metadata, dict) or set(REC_METADATA_FIELDS) - set(metadata):
                    raise SplitError(f"Recommendation row {index + 1} lacks required multi-positive metadata.")
                rec_group_id = str(metadata["recommendation_group_id"])
                if not rec_group_id:
                    raise SplitError(f"Recommendation row {index + 1} has empty recommendation_group_id.")
                rec_size = int(metadata["recommendation_group_size"])
                if rec_size < 1 or not isinstance(metadata["recommendation_all_gold_sids"], list):
                    raise SplitError(f"Recommendation row {index + 1} has invalid group metadata.")
                rec_domain = domain_from_current_gold(metadata["recommendation_current_gold_sid"])
                rec_hdom = rec_history_domain_key(value, metadata, rec_domain)
                family_node = f"rec_gid:{rec_group_id}"
                dsu.union(family_node, f"rec_hdom:{rec_hdom}")
            elif task in {"user_action", "user_chain"}:
                family_node = f"user:{user_family_key(value)}"
            else:
                family_node = f"material:{prompt}"
            # Exact prompt duplication across any task must never leak either.
            dsu.union(family_node, f"prompt:{prompt}")
            rows.append(Row(index, task, source_name, source_segment, prompt, sha256_text(raw), family_node,
                            rec_group_id, rec_hdom, rec_domain, rec_size))
    return rows, dsu


def component_stratum(rows: list[Row]) -> str:
    tasks = {row.task for row in rows}
    if tasks == {"recommendation"}:
        domains = {row.rec_domain for row in rows}
        sizes = {size_bin(int(row.rec_size)) for row in rows if row.rec_size is not None}
        group_types = {"singleton" if int(row.rec_size) == 1 else "multi" for row in rows if row.rec_size is not None}
        return "rec|{}|{}|{}".format("+".join(sorted(domains)), "+".join(sorted(group_types)), "+".join(sorted(sizes)))
    if tasks <= {"user_action", "user_chain"}:
        return "user|" + "+".join(sorted(tasks))
    if tasks == {"material"}:
        return "material|" + "+".join(sorted({row.source for row in rows}))
    # Should be extremely rare; only a canonical prompt shared across task types can create this.
    return "mixed|" + "+".join(sorted(tasks))


def score(stratum: str, component: str) -> str:
    return sha256_text(f"{SPLIT_SEED}|{ALGORITHM_VERSION}|{stratum}|{component}")


def select_dev_components(components: dict[str, list[Row]]) -> set[str]:
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for component, values in components.items():
        buckets[component_stratum(values)].append((component, len(values)))
    selected: set[str] = set()
    for stratum, candidates in buckets.items():
        target = sum(size for _, size in candidates) * DEV_RATE
        chosen = 0
        # SHA ordering makes selection stable; closest-prefix avoids a row-level weighted sample.
        for component, size in sorted(candidates, key=lambda value: score(stratum, value[0])):
            if abs((chosen + size) - target) <= abs(chosen - target):
                selected.add(component)
                chosen += size
        # Every populated recommendation domain must be represented in dev. The group-level
        # selection above normally guarantees this, but fail closed if it cannot.
        if stratum.startswith("rec|") and not any(component in selected for component, _ in candidates):
            component, _ = min(candidates, key=lambda value: (abs(value[1] - target), score(stratum, value[0])))
            selected.add(component)

    # A group-integrity split cannot always hit 2% inside every tiny stratum.
    # Do a deterministic global nearest-target correction without removing the
    # mandatory recommendation-domain representation selected above.
    total = sum(len(values) for values in components.values())
    target_total = total * DEV_RATE
    selected_rows = sum(len(components[component]) for component in selected)
    while True:
        candidates = [component for component in components if component not in selected]
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda component: (abs((selected_rows + len(components[component])) - target_total), score("global-add", component)),
        )
        if abs((selected_rows + len(components[best])) - target_total) >= abs(selected_rows - target_total):
            break
        selected.add(best)
        selected_rows += len(components[best])

    # The per-stratum nearest choices can lie slightly above the global 2%
    # target. Remove only a component that improves the global distance and
    # never remove the last dev component for any recommendation domain.
    def component_domains(component: str) -> set[str]:
        return {str(row.rec_domain) for row in components[component] if row.task == "recommendation"}

    selected_domain_counts: Counter[str] = Counter()
    for component in selected:
        selected_domain_counts.update(component_domains(component))
    while True:
        removable = [
            component for component in selected
            if all(selected_domain_counts[domain] > 1 for domain in component_domains(component))
        ]
        if not removable:
            break
        best = min(
            removable,
            key=lambda component: (abs((selected_rows - len(components[component])) - target_total), score("global-remove", component)),
        )
        if abs((selected_rows - len(components[best])) - target_total) >= abs(selected_rows - target_total):
            break
        selected.remove(best)
        selected_rows -= len(components[best])
        selected_domain_counts.subtract(component_domains(best))

    # Restore row-level domain representativeness after the global correction.
    # group type/bin remains in the initial stratification, while this final
    # pass keeps each target-domain share close to its real row distribution.
    for domain in ("video", "prod", "ad", "living"):
        domain_components = [
            component for component in components if component_domains(component) == {domain}
        ]
        domain_target = sum(len(components[component]) for component in domain_components) * DEV_RATE
        while True:
            domain_current = sum(len(components[component]) for component in domain_components if component in selected)
            if domain_current < domain_target:
                candidates = [component for component in domain_components if component not in selected]
                if not candidates:
                    break
                best = min(candidates, key=lambda component: (abs(domain_current + len(components[component]) - domain_target), score(f"domain-add:{domain}", component)))
                if abs(domain_current + len(components[best]) - domain_target) >= abs(domain_current - domain_target):
                    break
                selected.add(best)
                continue
            if domain_current > domain_target:
                candidates = [component for component in domain_components if component in selected]
                if len(candidates) <= 1:
                    break
                best = min(candidates, key=lambda component: (abs(domain_current - len(components[component]) - domain_target), score(f"domain-remove:{domain}", component)))
                if abs(domain_current - len(components[best]) - domain_target) >= abs(domain_current - domain_target):
                    break
                selected.remove(best)
                continue
            break
    return selected


def pct(part: int, total: int) -> float:
    return 0.0 if total == 0 else 100.0 * part / total


def count_field(rows: Iterable[Row], field: str) -> Counter[str]:
    return Counter(str(getattr(row, field)) for row in rows)


def rec_group_records(rows: Iterable[Row]) -> dict[str, list[Row]]:
    result: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.task == "recommendation":
            assert row.rec_group_id is not None
            result[row.rec_group_id].append(row)
    return result


def grouped_counts(rows: Iterable[Row]) -> dict[str, int]:
    return dict(sorted(Counter(row.source_segment for row in rows).items()))


def audit_rows(original: list[Row], train: list[Row], dev: list[Row]) -> dict[str, Any]:
    original_exact = {row.row_hash for row in original}
    train_exact, dev_exact = {row.row_hash for row in train}, {row.row_hash for row in dev}
    train_prompt, dev_prompt = {row.prompt for row in train}, {row.prompt for row in dev}
    train_user, dev_user = {row.family_node for row in train if row.task.startswith("user_")}, {row.family_node for row in dev if row.task.startswith("user_")}
    train_groups = rec_group_records(train)
    dev_groups = rec_group_records(dev)
    train_group_ids, dev_group_ids = set(train_groups), set(dev_groups)
    train_hdom = {row.rec_history_domain for row in train if row.task == "recommendation"}
    dev_hdom = {row.rec_history_domain for row in dev if row.task == "recommendation"}
    if len(train) + len(dev) != len(original):
        raise SplitError("Row conservation failed.")
    if train_exact & dev_exact or train_prompt & dev_prompt or train_group_ids & dev_group_ids or train_hdom & dev_hdom:
        raise SplitError("Leakage audit found a required train/dev overlap.")
    if original_exact != train_exact | dev_exact:
        raise SplitError("Output rows do not exactly cover source rows.")
    original_rec_groups = rec_group_records(original)
    if len(original_rec_groups) != len(train_groups) + len(dev_groups):
        raise SplitError("Recommendation group conservation failed.")
    domain_rows = lambda values: Counter(row.rec_domain for row in values if row.task == "recommendation")
    domain_groups = lambda groups: Counter(
        next(iter({row.rec_domain for row in values})) for values in groups.values()
    )
    type_groups = lambda groups: Counter(
        "singleton" if int(next(iter(values)).rec_size) == 1 else "multi" for values in groups.values()
    )
    bin_groups = lambda groups: Counter(size_bin(int(next(iter(values)).rec_size)) for values in groups.values())
    report = {
        "overall": {"original": len(original), "train": len(train), "dev": len(dev), "dev_percent": pct(len(dev), len(original))},
        "by_task": {task: {"original": sum(row.task == task for row in original), "train": sum(row.task == task for row in train), "dev": sum(row.task == task for row in dev), "dev_percent": pct(sum(row.task == task for row in dev), sum(row.task == task for row in original))} for task in ("material", "recommendation", "user_action", "user_chain")},
        "by_data_source": {name: {"original": sum(row.source == name for row in original), "train": sum(row.source == name for row in train), "dev": sum(row.source == name for row in dev), "dev_percent": pct(sum(row.source == name for row in dev), sum(row.source == name for row in original))} for name in sorted({row.source for row in original})},
        "by_source_segment": {name: {"original": sum(row.source_segment == name for row in original), "train": sum(row.source_segment == name for row in train), "dev": sum(row.source_segment == name for row in dev), "dev_percent": pct(sum(row.source_segment == name for row in dev), sum(row.source_segment == name for row in original))} for name in sorted({row.source_segment for row in original})},
        "recommendation": {
            "groups": {"original": len(original_rec_groups), "train": len(train_groups), "dev": len(dev_groups)},
            "routes": {name: {"original": sum(row.source_segment == name and row.task == "recommendation" for row in original), "train": sum(row.source_segment == name and row.task == "recommendation" for row in train), "dev": sum(row.source_segment == name and row.task == "recommendation" for row in dev)} for name in ("recommendation_cot", "recommendation_nocot")},
            "domain_rows": {name: {"original": domain_rows(original)[name], "train": domain_rows(train)[name], "dev": domain_rows(dev)[name]} for name in ("video", "prod", "ad", "living")},
            "domain_groups": {name: {"original": domain_groups(original_rec_groups)[name], "train": domain_groups(train_groups)[name], "dev": domain_groups(dev_groups)[name]} for name in ("video", "prod", "ad", "living")},
            "group_type": {name: {"original": type_groups(original_rec_groups)[name], "train": type_groups(train_groups)[name], "dev": type_groups(dev_groups)[name]} for name in ("singleton", "multi")},
            "group_size_bins": {name: {"original": bin_groups(original_rec_groups)[name], "train": bin_groups(train_groups)[name], "dev": bin_groups(dev_groups)[name]} for name in ("1", "2", "3-4", "5-8", "9+")},
        },
        "leakage": {
            "recommendation_group_id_overlap": len(train_group_ids & dev_group_ids),
            "recommendation_history_domain_overlap": len(train_hdom & dev_hdom),
            "user_family_overlap": len(train_user & dev_user),
            "exact_full_row_overlap": len(train_exact & dev_exact),
            "canonical_prompt_overlap": len(train_prompt & dev_prompt),
            "row_conservation": len(train) + len(dev) == len(original),
            "source_exact_duplicate_extra_instances": len(original) - len(original_exact),
            "output_exact_duplicate_extra_instances": len(train) + len(dev) - len(train_exact | dev_exact),
            "duplicate_delta_from_source": (len(train) + len(dev) - len(train_exact | dev_exact)) - (len(original) - len(original_exact)),
        },
    }
    if any(report["recommendation"]["domain_rows"][name]["dev"] == 0 for name in ("video", "prod", "ad", "living")):
        raise SplitError("Recommendation domain coverage failed: a dev domain is empty.")
    return report


def add_distribution_deltas(report: dict[str, Any]) -> None:
    for category in ("routes", "domain_rows", "group_type", "group_size_bins"):
        values = report["recommendation"][category]
        original_total = sum(item["original"] for item in values.values())
        dev_total = sum(item["dev"] for item in values.values())
        for item in values.values():
            item["original_share_percent"] = pct(item["original"], original_total)
            item["dev_share_percent"] = pct(item["dev"], dev_total)
            item["delta_percentage_points"] = item["dev_share_percent"] - item["original_share_percent"]


def markdown_audit(audit: dict[str, Any]) -> str:
    lines = ["# alpha-jiankong 98/2 Leak-Safe Split Audit", "", "## 总体", "", "|集合|行数|占比|", "|---|---:|---:|"]
    overall = audit["overall"]
    lines.extend([f"|原始|{overall['original']}|100.000%|", f"|train98|{overall['train']}|{100 - overall['dev_percent']:.3f}%|", f"|dev2|{overall['dev']}|{overall['dev_percent']:.3f}%|"])
    for title, key in (("任务", "by_task"), ("data_source", "by_data_source"), ("source_segment", "by_source_segment")):
        lines += ["", f"## {title}", "", "|类别|原始|train|dev|dev %|", "|---|---:|---:|---:|---:|"]
        for name, item in audit[key].items():
            lines.append(f"|{name}|{item['original']}|{item['train']}|{item['dev']}|{item['dev_percent']:.3f}%|")
    rec = audit["recommendation"]
    lines += ["", "## 推荐 group", "", f"总 group：原始 {rec['groups']['original']}，train {rec['groups']['train']}，dev {rec['groups']['dev']}。"]
    for title, category in (("推荐路由", "routes"), ("推荐域（row）", "domain_rows"), ("推荐 group 类型", "group_type"), ("推荐 group_size", "group_size_bins")):
        lines += ["", f"### {title}", "", "|类别|原始|train|dev|原始占比|dev占比|delta pp|", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, item in rec[category].items():
            lines.append(f"|{name}|{item['original']}|{item['train']}|{item['dev']}|{item.get('original_share_percent', 0):.3f}%|{item.get('dev_share_percent', 0):.3f}%|{item.get('delta_percentage_points', 0):+.3f}|")
    lines += ["", "## 泄漏审计", "", "|检查|结果|", "|---|---:|"]
    lines += [f"|{name}|{value}|" for name, value in audit["leakage"].items()]
    return "\n".join(lines) + "\n"


def write_dataset_info(path: Path) -> None:
    payload = {
        "onereason_alpha_jiankong_train98": {"file_name": "train.jsonl", "formatting": "alpaca", "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"}},
        "onereason_alpha_jiankong_dev2": {"file_name": "dev.jsonl", "formatting": "alpaca", "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"}},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.allow_existing:
        raise SplitError(f"Refusing to overwrite existing output: {args.output}")
    rows, dsu = load_rows(args.source)
    components: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        components[dsu.find(row.family_node)].append(row)
    selected = select_dev_components(components)
    train = [row for row in rows if dsu.find(row.family_node) not in selected]
    dev = [row for row in rows if dsu.find(row.family_node) in selected]
    audit = audit_rows(rows, train, dev)
    add_distribution_deltas(audit)
    if not (1.95 <= audit["overall"]["dev_percent"] <= 2.05):
        raise SplitError(f"Dev share {audit['overall']['dev_percent']:.5f}% is outside [1.95, 2.05].")

    temp = args.output.with_name(args.output.name + ".tmp-split")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    dev_indexes = {row.index for row in dev}
    with args.source.open("r", encoding="utf-8") as source, \
         (temp / "train.jsonl").open("w", encoding="utf-8", newline="\n") as train_file, \
         (temp / "dev.jsonl").open("w", encoding="utf-8", newline="\n") as dev_file:
        for index, raw in enumerate(source):
            (dev_file if index in dev_indexes else train_file).write(raw)
    write_dataset_info(temp / "dataset_info.json")
    manifest = {
        "name": "alpha-jiankong-split-v1",
        "source_dataset": "alpha-jiankong",
        "source_jsonl": str(args.source),
        "source_sha256": sha256_file(args.source),
        "split_seed": int(SPLIT_SEED),
        "split_algorithm_version": ALGORITHM_VERSION,
        "split_rate_target_percent": DEV_RATE * 100,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "leakage_key_definitions": {
            "recommendation": "connected components over recommendation_group_id and canonical ordered prompt SID-history + target-domain",
            "user": "canonical input + structured history; independent of CoT/NoThink wording",
            "material": "canonical complete prompt, with think/no_think suffix normalized",
            "global": "canonical complete prompt is unioned across all task families",
        },
        "files": {name: {"rows": len(values), "sha256": sha256_file(temp / name)} for name, values in (("train.jsonl", train), ("dev.jsonl", dev))},
        "audit": audit,
        "schema": {"required_fields": list(REQUIRED_FIELDS), "recommendation_metadata_fields": list(REC_METADATA_FIELDS), "rows_preserved_verbatim": True},
        "builder_script": {"path": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
    }
    (temp / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temp / "split_audit.md").write_text(markdown_audit(audit), encoding="utf-8")
    (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output.exists():
        shutil.rmtree(args.output)
    temp.rename(args.output)
    print(json.dumps({"output": str(args.output), "overall": audit["overall"], "leakage": audit["leakage"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except SplitError as error:
        print(f"ALPHA_SPLIT_FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
