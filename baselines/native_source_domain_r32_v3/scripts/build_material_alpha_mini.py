#!/usr/bin/env python3
"""Build alpha_mini material task pool with coverage-first deterministic sampling."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


SEED = 20260813
MODEL = "/data/models/onereason-8b-pretrain-competition"
SID_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
DOMAINS = ("video", "prod", "ad", "living")
MATERIAL_QUOTAS = {"video": 1505, "prod": 1459, "ad": 1138, "living": 898}
MATERIAL_MODE_QUOTAS = {
    "video": {"cot": 753, "nocot": 752},
    "prod": {"cot": 729, "nocot": 730},
    "ad": {"cot": 569, "nocot": 569},
    "living": {"cot": 449, "nocot": 449},
}
SCHEMA = {
    "system", "instruction", "input", "output", "history",
    "data_source", "source_segment", "aux_metadata_json",
}


def thought_mode(row: dict) -> str:
    suffix = str(row["instruction"]).rstrip()
    if suffix.endswith("/think"):
        return "cot"
    if suffix.endswith("/no_think"):
        return "nocot"
    raise ValueError("Row has no think mode suffix")


def normalized_caption(row: dict) -> str:
    instruction = re.sub(r"/(?:think|no_think)\s*$", "", str(row["instruction"])).rstrip()
    return "\x1f".join((str(row["system"]), instruction, str(row["input"])))


def sid_for(row: dict, direction: str) -> tuple[str, str]:
    text = "".join(str(row[key]) for key in (("system", "instruction", "input") if direction == "sid_to_text" else ("output",)))
    match = SID_RE.search(text)
    if match is None:
        raise ValueError(f"No SID for direction={direction}")
    return match.group(0), match.group(1)


def read_source(path: Path) -> dict[str, list[dict]]:
    result = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != SCHEMA:
                raise ValueError(f"Unexpected schema at line {line_no}")
            source = row["data_source"]
            if source not in {"material_sample", "sid_bucket_canonical_no_think", "sid_bucket_reverse"}:
                raise ValueError(f"Unexpected material route: {source}")
            direction = "sid_to_text" if source == "sid_bucket_canonical_no_think" else "text_to_sid" if source == "sid_bucket_reverse" else None
            if source == "material_sample":
                prompt_sid = SID_RE.search("".join(str(row[key]) for key in ("system", "instruction", "input")))
                output_sid = SID_RE.search(str(row["output"]))
                direction = "sid_to_text" if prompt_sid else "text_to_sid" if output_sid else None
                if direction is None:
                    raise ValueError(f"material_sample row {line_no} has no SID")
            sid, domain = sid_for(row, direction)
            result[source].append({
                "row": row, "line_no": line_no, "direction": direction,
                "sid": sid, "domain": domain, "mode": thought_mode(row),
                "caption": normalized_caption(row),
            })
    return result


def choose_row(candidates: list[dict], mode: str, rng: random.Random) -> dict:
    choices = [item for item in candidates if item["mode"] == mode]
    if not choices:
        raise ValueError(f"No candidate for requested mode={mode}")
    return rng.choice(sorted(choices, key=lambda item: item["line_no"]))


def reverse_sample(rows: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    by_sid = defaultdict(list)
    for item in rows:
        by_sid[item["sid"]].append(item)
    if len(rows) != 29586 or len(by_sid) != 11298:
        raise ValueError(f"Unexpected reverse source size: rows={len(rows)} SID={len(by_sid)}")

    only_cot, only_nocot, both = [], [], []
    for sid, candidates in by_sid.items():
        modes = {item["mode"] for item in candidates}
        if modes == {"cot"}:
            only_cot.append(sid)
        elif modes == {"nocot"}:
            only_nocot.append(sid)
        else:
            both.append(sid)
    target_first_cot = len(by_sid) // 2 + len(by_sid) % 2  # 5,649
    if len(only_cot) > target_first_cot or len(only_nocot) > len(by_sid) - target_first_cot:
        raise ValueError("Cannot build balanced first reverse coverage pass")
    rng.shuffle(both)
    first_cot_sids = set(only_cot + both[:target_first_cot - len(only_cot)])
    primary = {}
    for sid in sorted(by_sid):
        primary[sid] = choose_row(by_sid[sid], "cot" if sid in first_cot_sids else "nocot", rng)
    first_modes = Counter(item["mode"] for item in primary.values())
    second_targets = {"cot": 7500 - first_modes["cot"], "nocot": 7500 - first_modes["nocot"]}
    if sum(second_targets.values()) != 3702:
        raise AssertionError("Wrong reverse second-pass size")

    eligible = {"cot": [], "nocot": [], "both": []}
    alternatives = {}
    for sid, original in by_sid.items():
        initial_caption = primary[sid]["caption"]
        valid = [item for item in original if item["caption"] != initial_caption]
        modes = {item["mode"] for item in valid}
        alternatives[sid] = valid
        if modes == {"cot"}:
            eligible["cot"].append(sid)
        elif modes == {"nocot"}:
            eligible["nocot"].append(sid)
        elif modes:
            eligible["both"].append(sid)
    second = {}
    forced = {mode: eligible[mode][:] for mode in ("cot", "nocot")}
    for values in forced.values():
        rng.shuffle(values)
    selected_by_mode = {"cot": [], "nocot": []}
    for mode in ("cot", "nocot"):
        selected_by_mode[mode].extend(forced[mode][:min(second_targets[mode], len(forced[mode]))])
    remaining = {mode: second_targets[mode] - len(selected_by_mode[mode]) for mode in ("cot", "nocot")}
    both_sids = eligible["both"][:]
    rng.shuffle(both_sids)
    if sum(remaining.values()) > len(both_sids):
        raise ValueError("Insufficient distinct-caption reverse alternatives")
    cursor = 0
    for mode in ("cot", "nocot"):
        selected_by_mode[mode].extend(both_sids[cursor:cursor + remaining[mode]])
        cursor += remaining[mode]
    for mode, sids in selected_by_mode.items():
        if len(sids) != second_targets[mode]:
            raise AssertionError("Unexpected reverse mode selection count")
        for sid in sids:
            second[sid] = choose_row(alternatives[sid], mode, rng)
    selected = list(primary.values()) + list(second.values())
    if len(selected) != 15000 or len({item["sid"] for item in selected}) != 11298:
        raise AssertionError("Reverse coverage requirement failed")
    if any(primary[sid]["caption"] == second[sid]["caption"] for sid in second):
        raise AssertionError("Second reverse caption is not different")
    report = {
        "source_rows": len(rows), "source_unique_sids": len(by_sid), "selected_rows": len(selected),
        "selected_unique_sids": len({item["sid"] for item in selected}),
        "second_caption_sids": len(second), "max_rows_per_sid": max(Counter(item["sid"] for item in selected).values()),
        "mode_counts": dict(Counter(item["mode"] for item in selected)),
        "first_pass_mode_counts": dict(first_modes), "second_pass_mode_counts": dict(Counter(item["mode"] for item in second.values())),
    }
    return selected, report


def material_sample(rows: list[dict], direction: str, rng: random.Random) -> tuple[list[dict], dict]:
    candidates = [item for item in rows if item["direction"] == direction]
    selected, used_sids = [], set()
    strata_report = {}
    for domain in DOMAINS:
        for mode in ("cot", "nocot"):
            target = MATERIAL_MODE_QUOTAS[domain][mode]
            stratum = [item for item in candidates if item["domain"] == domain and item["mode"] == mode]
            by_sid = defaultdict(list)
            for item in stratum:
                by_sid[item["sid"]].append(item)
            sid_order = list(by_sid)
            rng.shuffle(sid_order)
            chosen = []
            for sid in sid_order:
                if sid not in used_sids:
                    chosen.append(rng.choice(sorted(by_sid[sid], key=lambda item: item["line_no"])))
                    used_sids.add(sid)
                    if len(chosen) == target:
                        break
            if len(chosen) < target:
                fallback = [item for item in stratum if item not in chosen]
                rng.shuffle(fallback)
                chosen.extend(fallback[:target - len(chosen)])
            if len(chosen) != target:
                raise ValueError(f"Insufficient material stratum {direction}/{domain}/{mode}")
            selected.extend(chosen)
            strata_report[f"{domain}|{mode}"] = {"available_rows": len(stratum), "available_unique_sids": len(by_sid), "selected": len(chosen)}
    domains = Counter(item["domain"] for item in selected)
    modes = Counter(item["mode"] for item in selected)
    if len(selected) != 5000 or dict(domains) != MATERIAL_QUOTAS or dict(modes) != {"cot": 2500, "nocot": 2500}:
        raise AssertionError("Material quota requirement failed")
    return selected, {
        "source_rows": len(candidates), "selected_rows": len(selected),
        "selected_unique_sids": len({item["sid"] for item in selected}),
        "domain_counts": dict(domains), "mode_counts": dict(modes), "strata": strata_report,
    }


def write_jsonl(path: Path, items: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as stream:
        for item in sorted(items, key=lambda value: value["line_no"]):
            payload = json.dumps(item["row"], ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(payload)
            digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def tokens(items: list[dict], tokenizer) -> dict:
    prompt_total = output_total = 0
    values = []
    for item in items:
        row = item["row"]
        prompt = "".join(str(row[key]) for key in ("system", "instruction", "input"))
        prompt_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        output_count = len(tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"])
        prompt_total += prompt_count; output_total += output_count; values.append(prompt_count + output_count)
    return {"prompt_tokens": prompt_total, "output_tokens": output_total, "total_tokens": sum(values), "mean_tokens": round(sum(values) / len(values), 4), "max_tokens": max(values)}


def readme(manifest: dict) -> str:
    counts = manifest["counts"]
    return f'''# 懂物料 alpha_mini

## 来源与原则

- 上游：`{manifest["source_dataset"]}`
- 上游版本：懂物料 beta版（原样清洗任务池）
- 固定随机种子：`{SEED}`
- 原始行不被改写；本版本仅抽样并保留统一训练 schema。

## 路由构成

| 路由 | 规则 | 保留行数 |
|---|---|---:|
| `sid_bucket_canonical_no_think` | 全部保留 | 11,298 |
| `sid_bucket_reverse` | 11,298 个 SID 各至少一条；3,702 个不同 SID 再各补一条不同 caption | 15,000 |
| `material_sample` | SID→文本 5,000；文本→SID 5,000 | 10,000 |
| 合计 |  | **36,298** |

## Reverse 约束

- unique SID coverage：11,298 / 11,298（100%）。
- 每个 SID 最多两条，避免高 caption SID 获得高采样权重。
- 两条时保证第二条 prompt caption 与第一条不同。
- COT / NoThink：{counts["reverse"]["mode_counts"]["cot"]:,} / {counts["reverse"]["mode_counts"]["nocot"]:,}。

## material_sample 分层

每个方向均为 5,000 条，四域精确配额：video 1,505、prod 1,459、ad 1,138、living 898；COT / NoThink 各 2,500。每个方向内优先抽取未使用 SID，最大化 SID 覆盖。

## Token 统计

总 token：{manifest["tokens"]["all"]["total_tokens"]:,}。计算为当前模型 tokenizer 对 `system + instruction + input + output` 的原始 token 数，`add_special_tokens=false`，不含 chat-template 固定开销。

完整机器可读记录见 `manifest.json`。
'''


def main() -> None:
    source = Path("/data/lf_data_versions/task_pools") / "\u61c2\u7269\u6599" / "beta\u7248" / "material_beta.jsonl"
    out_dir = Path("/data/lf_data_versions/task_pools") / "\u61c2\u7269\u6599" / "alpha_mini"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    source_rows = read_source(source)
    if len(source_rows["sid_bucket_canonical_no_think"]) != 11298 or len(source_rows["material_sample"]) != 100000:
        raise ValueError("Unexpected source route counts")
    rng = random.Random(SEED)
    canonical = source_rows["sid_bucket_canonical_no_think"]
    reverse, reverse_report = reverse_sample(source_rows["sid_bucket_reverse"], rng)
    sample_forward, forward_report = material_sample(source_rows["material_sample"], "sid_to_text", rng)
    sample_reverse, sample_reverse_report = material_sample(source_rows["material_sample"], "text_to_sid", rng)
    merged = canonical + reverse + sample_forward + sample_reverse
    if len(merged) != 36298:
        raise AssertionError("Unexpected total")

    out_dir.mkdir(parents=True)
    files = {
        "sid_bucket_canonical_no_think.jsonl": canonical,
        "sid_bucket_reverse.jsonl": reverse,
        "material_sample.jsonl": sample_forward + sample_reverse,
        "material_alpha_mini.jsonl": merged,
    }
    hashes = {name: write_jsonl(out_dir / name, rows) for name, rows in files.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    manifest = {
        "kind": "unregistered_task_pool", "name": "alpha_mini", "task": "material",
        "source_dataset": str(source), "seed": SEED,
        "selection_rule": "canonical all; reverse coverage-first one plus at most one different caption per SID; material_sample domain/mode balanced coverage-first",
        "counts": {
            "canonical": {"rows": len(canonical), "unique_sids": len({item["sid"] for item in canonical}), "mode_counts": dict(Counter(item["mode"] for item in canonical))},
            "reverse": reverse_report,
            "material_sample_sid_to_text": forward_report,
            "material_sample_text_to_sid": sample_reverse_report,
            "total_rows": len(merged),
        },
        "tokens": {"model": MODEL, "method": "AutoTokenizer add_special_tokens=False; prompt=system+instruction+input; total=prompt+output; excludes chat-template fixed overhead", "canonical": tokens(canonical, tokenizer), "reverse": tokens(reverse, tokenizer), "material_sample_sid_to_text": tokens(sample_forward, tokenizer), "material_sample_text_to_sid": tokens(sample_reverse, tokenizer), "all": tokens(merged, tokenizer)},
        "files": hashes, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(readme(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
