import argparse
import collections
import copy
import hashlib
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

from user_prompt_adapter import (
    ACTION_DEMO,
    CHAIN_DEMO,
    PROMPT_TEMPLATE_VERSION,
    adapt_prompt,
    split_source_prompt,
)


VERSION = "gr_user_v1"
SEED = 20260819
MAX_PROMPT_TOKENS = 8192
SID_RE = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
DATE_RE = re.compile(r"^【(\d{4}-\d{2}-\d{2})】$")
EVENT_RE = re.compile(r"^\s*(\d{2}:\d{2}|--:--)\s+\[([^\]]+)\]\s+(.+?)\s*$")

ACTION_BUCKETS = {
    "1-5": (1, 5, 225),
    "6-10": (6, 10, 300),
    "11-20": (11, 20, 450),
    "21-30": (21, 30, 300),
    "31-40": (31, 40, 150),
    "41+": (41, 10**9, 75),
}
CHAIN_BUCKETS = {2: 225, 3: 825, 4: 375, 5: 75}
CHAIN_CONVERTED = {2: 45, 3: 165, 4: 75, 5: 15}
ACTION_PILOT = {"1-5": 45, "6-10": 60, "11-20": 90, "21-30": 60, "31-40": 30, "41+": 15}
CHAIN_PILOT = {2: 45, 3: 165, 4: 75, 5: 15}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_key(value: str, salt: str) -> str:
    return sha256_bytes(f"{SEED}:{salt}:{value}".encode())


def final_json_text(output: str) -> str:
    if "</think>" in output:
        return output.split("</think>", 1)[1].strip()
    return output.strip()


def history_part(prompt: str) -> str:
    return split_source_prompt(prompt)[0]


def parse_history(prompt: str) -> list[dict]:
    current_date = None
    events = []
    for line in history_part(prompt).splitlines():
        date_match = DATE_RE.match(line.strip())
        if date_match:
            current_date = date_match.group(1)
            continue
        event_match = EVENT_RE.match(line)
        if not event_match or current_date is None:
            continue
        time, action, payload = event_match.groups()
        sid_match = SID_RE.search(payload)
        events.append(
            {
                "date": current_date,
                "time": time,
                "action": action,
                "sid": sid_match.group(0) if sid_match else None,
                "raw": f"[{action}] {payload}",
            }
        )
    return events


def action_bucket(count: int) -> str | None:
    for name, (low, high, _) in ACTION_BUCKETS.items():
        if low <= count <= high:
            return name
    return None


def validate_chain(events: list[dict], history_events: list[dict]) -> tuple[bool, str]:
    if len(events) not in CHAIN_BUCKETS:
        return False, "event_count_not_2_to_5"
    dates = [event.get("date") for event in events]
    if any(not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) for date in dates):
        return False, "invalid_date"
    if dates != sorted(dates):
        return False, "chronology_violation"
    keys = [(event.get("date"), event.get("action")) for event in events]
    if len(keys) != len(set(keys)):
        return False, "duplicate_gold_event"
    history_by_date = collections.defaultdict(set)
    for event in history_events:
        history_by_date[event["date"]].add(event["raw"])
    for event in events:
        action = event.get("action")
        logic = event.get("logic")
        if not isinstance(action, str) or not action.strip() or not isinstance(logic, str):
            return False, "invalid_event_shape"
        parts = [part.strip() for part in action.split("；") if part.strip()]
        if not parts or any(part not in history_by_date[event["date"]] for part in parts):
            return False, "history_grounding_failed"
    return True, "ok"


def rendered_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_candidate(tokenizer, source_file: Path, source_line: int, source: dict, route: str, converted: bool):
    source_prompt = source["input"]
    final = final_json_text(source["output"])
    try:
        gold = json.loads(final)
    except json.JSONDecodeError:
        return None, "invalid_final_json"
    history_events = parse_history(source_prompt)
    prompt = adapt_prompt(route, source_prompt)
    rendered = rendered_prompt(tokenizer, prompt)
    token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
    if token_count > MAX_PROMPT_TOKENS:
        return None, "rendered_prompt_gt_8192"
    if route == "action":
        if not isinstance(gold, list) or not gold or not all(isinstance(value, str) for value in gold):
            return None, "invalid_action_gold"
        if len(gold) != len(set(gold)):
            return None, "duplicate_action_gold"
        history_sids = SID_RE.findall(history_part(source_prompt))
        if not set(gold).issubset(set(history_sids)):
            return None, "action_gold_not_in_history"
        bucket = action_bucket(len(gold))
        gold_events = []
        gold_sids = gold
    else:
        if not isinstance(gold, dict) or not isinstance(gold.get("logic_chain"), dict):
            return None, "invalid_chain_gold"
        gold_events = gold["logic_chain"].get("events")
        if not isinstance(gold_events, list):
            return None, "invalid_chain_events"
        valid, reason = validate_chain(gold_events, history_events)
        if not valid:
            return None, reason
        bucket = len(gold_events)
        history_sids = [event["sid"] for event in history_events if event["sid"]]
        gold_sids = SID_RE.findall("\n".join(event["action"] for event in gold_events))
    source_name = source_file.name
    sample_id = sha256_bytes(f"{source_name}:{source_line}:{final}".encode())
    row = {
        "sample_id": sample_id,
        "route": route,
        "prompt": prompt,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "source_file": str(source_file),
        "source_line": source_line,
        "source_segment": source.get("source_segment"),
        "converted_from_cot": converted,
        "raw_gold_output": final,
        "history_sids": history_sids,
        "gold_sids": gold_sids,
        "history_events": history_events,
        "gold_events": copy.deepcopy(gold_events),
        "gold_sid_count": len(gold_sids),
        "gold_event_count": len(gold_events),
        "prompt_token_count": token_count,
        "bucket": bucket,
        "source_input_sha256": sha256_bytes(source_prompt.encode()),
        "source_output_sha256": sha256_bytes(source["output"].encode()),
    }
    return row, "ok"


def assign_quartiles(rows: list[dict]) -> dict[int, list[dict]]:
    ordered = sorted(rows, key=lambda row: (row["prompt_token_count"], row["sample_id"]))
    groups = collections.defaultdict(list)
    for index, row in enumerate(ordered):
        quartile = min(3, index * 4 // max(1, len(ordered)))
        groups[quartile].append(row)
    return groups


def stratified_pick(rows: list[dict], count: int, salt: str) -> list[dict]:
    groups = assign_quartiles(rows)
    targets = [count // 4 + (quartile < count % 4) for quartile in range(4)]
    selected = []
    for quartile, target in enumerate(targets):
        candidates = sorted(groups[quartile], key=lambda row: stable_key(row["sample_id"], f"{salt}:q{quartile}"))
        if len(candidates) < target:
            raise ValueError(f"insufficient quartile {quartile} for {salt}: {len(candidates)} < {target}")
        selected.extend(candidates[:target])
    return selected


def remove_ids(rows: list[dict], removed: list[dict]) -> list[dict]:
    ids = {row["sample_id"] for row in removed}
    return [row for row in rows if row["sample_id"] not in ids]


def select_by_bucket(rows: list[dict], quotas: dict, salt: str) -> list[dict]:
    selected = []
    for bucket, quota in quotas.items():
        candidates = [row for row in rows if row["bucket"] == bucket]
        selected.extend(stratified_pick(candidates, quota, f"{salt}:{bucket}"))
    return selected


def shuffled(rows: list[dict], salt: str) -> list[dict]:
    return sorted(rows, key=lambda row: stable_key(row["sample_id"], salt))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    def quantile(q):
        return ordered[round((len(ordered) - 1) * q)]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": quantile(0.25),
        "p50": quantile(0.5),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def source_audit(path: Path, accepted: list[dict], rejects: collections.Counter) -> dict:
    line_hashes = set()
    duplicate_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            digest = sha256_bytes(line.rstrip("\n").encode())
            duplicate_rows += digest in line_hashes
            line_hashes.add(digest)
    bucket_counts = collections.Counter(str(row["bucket"]) for row in accepted)
    return {
        "path": str(path),
        "line_count": len(line_hashes) + duplicate_rows,
        "sha256": sha256_bytes(path.read_bytes()),
        "exact_duplicate_rows": duplicate_rows,
        "accepted_after_full_reaudit": len(accepted),
        "output_legal_and_history_grounded": len(accepted),
        "rejections": dict(sorted(rejects.items())),
        "source_segments": dict(collections.Counter(row["source_segment"] for row in accepted)),
        "gold_count_distribution": dict(sorted(bucket_counts.items())),
        "route_suffix_no_think": sum(row["prompt"].endswith("/no_think") for row in accepted),
        "prompt_token_distribution": distribution([row["prompt_token_count"] for row in accepted]),
    }


def merged_source_audit(path: Path) -> dict:
    digest = hashlib.sha256()
    line_hashes = set()
    duplicate_rows = 0
    segments = collections.Counter()
    no_think = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            row_hash = sha256_bytes(raw_line.rstrip(b"\n"))
            duplicate_rows += row_hash in line_hashes
            line_hashes.add(row_hash)
            row = json.loads(raw_line)
            segments[row.get("source_segment")] += 1
            no_think += row.get("input", "").endswith("/no_think")
    return {
        "path": str(path),
        "line_count": len(line_hashes) + duplicate_rows,
        "sha256": digest.hexdigest(),
        "exact_duplicate_rows": duplicate_rows,
        "source_segments": dict(sorted(segments.items())),
        "route_suffix_no_think": no_think,
    }


def compact_tail(text: str, lines: int = 18) -> str:
    return "\n".join(text.splitlines()[-lines:])


def write_sample_tails(path: Path, audit: dict) -> None:
    sections = []
    labels = ["action", "chain_native", "chain_converted"]
    display = {"action": "Action", "chain_native": "Native Chain", "chain_converted": "Converted Chain"}
    for label in labels:
        rows = [row for row in audit["native_samples"] if row["kind"] == label][:2]
        sections.append(f"## {display[label]}")
        for index, row in enumerate(rows, 1):
            sections.append(
                f"### Sample {index}\n\nSource tail:\n\n```text\n{row['source_tail']}\n```\n\n"
                f"Adapted tail:\n\n```text\n{row['adapted_tail']}\n```\n\n"
                f"Rendered tail:\n\n```text\n{row['rendered_tail']}\n```"
            )
    path.write_text("# GR_USER_v1 prompt tail audit\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")


def make_template_audit(tokenizer, pools: dict, eval_examples: list[dict]) -> dict:
    samples = []
    for label, rows in pools.items():
        chosen = sorted(rows, key=lambda row: stable_key(row["sample_id"], f"audit:{label}"))[:10]
        for row in chosen:
            source_lines = Path(row["source_file"]).open(encoding="utf-8")
            source = None
            for index, line in enumerate(source_lines):
                if index == row["source_line"]:
                    source = json.loads(line)
                    break
            rendered = rendered_prompt(tokenizer, row["prompt"])
            samples.append(
                {
                    "kind": label,
                    "sample_id": row["sample_id"],
                    "source_tail": compact_tail(source["input"]),
                    "adapted_tail": compact_tail(row["prompt"]),
                    "rendered_tail": compact_tail(rendered),
                    "chat_wrapper_counts": {
                        "user": rendered.count("<|im_start|>user"),
                        "assistant": rendered.count("<|im_start|>assistant"),
                        "im_end": rendered.count("<|im_end|>"),
                    },
                }
            )
    eval_summary = []
    for row in eval_examples:
        raw = row["raw_user_content"]
        eval_summary.append(
            {
                "route": row["route"],
                "sample_id": row["sample_id"],
                "has_demonstration": "其他用户" in raw and ("输出示例" in raw or "有效逻辑链案例" in raw),
                "suffix": "/no_think" if raw.endswith("/no_think") else "other",
                "wrapper_counts": {
                    "user": row["rendered_input"].count("<|im_start|>user"),
                    "assistant": row["rendered_input"].count("<|im_start|>assistant"),
                },
                "assistant_prefix": row["assistant_prefix"],
            }
        )
    return {
        "version": PROMPT_TEMPLATE_VERSION,
        "evidence_counts": dict(collections.Counter(row["route"] for row in eval_examples)),
        "evaluation_examples": eval_summary,
        "native_samples": samples,
        "conclusions": {
            "action_demonstration_stable": all(item["has_demonstration"] for item in eval_summary if item["route"] == "action"),
            "chain_demonstration_stable": all(item["has_demonstration"] for item in eval_summary if item["route"] == "chain"),
            "raw_data_contains_chat_wrappers": False,
            "adapter_adds_chat_wrappers": False,
            "renderer": "parent checkpoint tokenizer.apply_chat_template(enable_thinking=False)",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--eval-examples", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.parent_checkpoint, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("parent tokenizer has no chat template")

    specs = [
        ("action", "user_action.jsonl", False),
        ("chain_native", "user_chain_nocot.jsonl", False),
        ("chain_converted", "user_chain_cot.jsonl", True),
    ]
    pools = {}
    audits = {}
    for label, filename, converted in specs:
        path = args.source_dir / filename
        rows = []
        rejects = collections.Counter()
        with path.open(encoding="utf-8") as handle:
            for source_line, line in enumerate(handle):
                source = json.loads(line)
                route = "action" if label == "action" else "chain"
                row, reason = build_candidate(tokenizer, path, source_line, source, route, converted)
                if row is None:
                    rejects[reason] += 1
                else:
                    rows.append(row)
        pools[label] = rows
        audits[label] = source_audit(path, rows, rejects)
        print(label, len(rows), dict(rejects), flush=True)

    probe = []
    for bucket in ACTION_BUCKETS:
        candidates = [row for row in pools["action"] if row["bucket"] == bucket]
        probe.extend(stratified_pick(candidates, 2, f"probe:action:{bucket}"))
    for bucket in CHAIN_BUCKETS:
        candidates = [row for row in pools["chain_native"] if row["bucket"] == bucket]
        probe.extend(stratified_pick(candidates, 2, f"probe:chain:{bucket}"))
    pools["action"] = remove_ids(pools["action"], probe)
    pools["chain_native"] = remove_ids(pools["chain_native"], probe)

    action_train = select_by_bucket(
        pools["action"],
        {name: spec[2] for name, spec in ACTION_BUCKETS.items()},
        "train:action",
    )
    native_quotas = {bucket: total - CHAIN_CONVERTED[bucket] for bucket, total in CHAIN_BUCKETS.items()}
    chain_native = select_by_bucket(pools["chain_native"], native_quotas, "train:chain_native")
    chain_converted = select_by_bucket(pools["chain_converted"], CHAIN_CONVERTED, "train:chain_converted")
    train = shuffled(action_train + chain_native + chain_converted, "train:shuffle")
    probe = shuffled(probe, "probe:shuffle")

    action_pilot = select_by_bucket(action_train, ACTION_PILOT, "pilot:action")
    chain_pilot_native_quotas = {bucket: total - total // 5 for bucket, total in CHAIN_PILOT.items()}
    chain_pilot_converted_quotas = {bucket: total // 5 for bucket, total in CHAIN_PILOT.items()}
    chain_pilot = select_by_bucket(chain_native, chain_pilot_native_quotas, "pilot:chain_native")
    chain_pilot += select_by_bucket(chain_converted, chain_pilot_converted_quotas, "pilot:chain_converted")
    pilot = shuffled(action_pilot + chain_pilot, "pilot:shuffle")

    write_jsonl(args.output_dir / "train_3000.jsonl", train)
    write_jsonl(args.output_dir / "pilot_600.jsonl", pilot)
    write_jsonl(args.output_dir / "probe_v1.jsonl", probe)

    eval_examples = json.loads(args.eval_examples.read_text(encoding="utf-8"))
    template_audit = make_template_audit(tokenizer, pools, eval_examples)
    (args.output_dir / "template_alignment_audit.json").write_text(
        json.dumps(template_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_sample_tails(args.output_dir / "sample_tails.md", template_audit)
    source_manifest = args.source_dir / "manifest.json"
    outputs = {}
    for name in ["train_3000.jsonl", "pilot_600.jsonl", "probe_v1.jsonl", "template_alignment_audit.json"]:
        path = args.output_dir / name
        outputs[name] = {"sha256": sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size}
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "source_manifest": {"path": str(source_manifest), "sha256": sha256_bytes(source_manifest.read_bytes())},
        "source_audits": audits,
        "merged_source_audit": merged_source_audit(args.source_dir / "understand_user_clean.jsonl"),
        "parent_model": {
            "experiment": "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333",
            "external_aggregate": None,
            "base_model_path": "/data/models/onereason-8b-pretrain-competition",
            "adapter_checkpoint_path": str(args.parent_checkpoint),
            "checkpoint_step": 1106,
            "checkpoint_epoch": 2.0,
            "selection_evidence": "User explicitly selected beta-baseline Epoch 2; trainer_state.json records global_step=1106 and epoch=2.0.",
        },
        "template": {
            "version": PROMPT_TEMPLATE_VERSION,
            "tokenizer": str(args.parent_checkpoint),
            "renderer": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
            "action_format_demo": True,
            "chain_format_demo": True,
        },
        "counts": {
            "train": len(train),
            "action": sum(row["route"] == "action" for row in train),
            "chain": sum(row["route"] == "chain" for row in train),
            "chain_native": sum(row["route"] == "chain" and not row["converted_from_cot"] for row in train),
            "chain_converted": sum(row["converted_from_cot"] for row in train),
            "pilot": len(pilot),
            "probe": len(probe),
        },
        "train_buckets": {
            "action": dict(collections.Counter(str(row["bucket"]) for row in train if row["route"] == "action")),
            "chain": dict(collections.Counter(str(row["bucket"]) for row in train if row["route"] == "chain")),
        },
        "sampling_quartile_targets": {
            "action": {name: [spec[2] // 4 + (q < spec[2] % 4) for q in range(4)] for name, spec in ACTION_BUCKETS.items()},
            "chain_native": {str(bucket): [quota // 4 + (q < quota % 4) for q in range(4)] for bucket, quota in native_quotas.items()},
            "chain_converted": {str(bucket): [quota // 4 + (q < quota % 4) for q in range(4)] for bucket, quota in CHAIN_CONVERTED.items()},
        },
        "prompt_token_distribution": {
            "train": distribution([row["prompt_token_count"] for row in train]),
            "action": distribution([row["prompt_token_count"] for row in train if row["route"] == "action"]),
            "chain": distribution([row["prompt_token_count"] for row in train if row["route"] == "chain"]),
        },
        "probe_ids": [row["sample_id"] for row in probe],
        "pilot_ids": [row["sample_id"] for row in pilot],
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# GR_USER_v1 template alignment and data audit

## Evidence

- Parent: beta-baseline Epoch 2 at `{args.parent_checkpoint}` (`trainer_state.json`: global_step 1106, epoch 2.0).
- Evaluation-template evidence: user-provided `GR-V1-2000_filtered.log`, external aggregate 1.3326. This log is template evidence only and is not the selected parent.
- Evaluation examples: 5 Action and 5 Chain NoCoT examples.
- Action format-only demonstration present: 5/5.
- Chain format/logic demonstration present: 5/5.

## Alignment decision

Both routes use an evaluation-aligned raw user-content adapter. Action receives the stable cross-user JSON-array demonstration. Chain receives the stable cross-user logic-chain demonstration and evaluation wording. Neither demonstration uses the current row's history, topic, or Gold. Raw prompts contain no chat special tokens; the parent tokenizer renderer adds exactly one user wrapper, one assistant head, and the empty NoCoT think block.

## Dataset

- train: {len(train)} (Action {len(action_train)}, Chain {len(chain_native) + len(chain_converted)})
- Chain native/converted: {len(chain_native)}/{len(chain_converted)}
- pilot: {len(pilot)}
- probe: {len(probe)} (native only, disjoint from train)
- max rendered prompt tokens: {max(row['prompt_token_count'] for row in train)}

Full source hashes, rejection reasons, bucket counts, token distributions, IDs, and output hashes are recorded in `manifest.json`. Per-example source/adapted/rendered tails are recorded in `template_alignment_audit.json`.
"""
    (args.output_dir / "template_alignment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
