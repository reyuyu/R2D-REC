#!/usr/bin/env python3
"""Build Mini-Fix-U3K with fixed total rows and unchanged recommendation data.

The user pool is replaced by the validated mini_v2 3,000-row pool. The
recommendation rows come byte-for-byte from Mini-Fix. To keep the total dataset
size and optimizer-step budget unchanged, 1,000 deterministic supplemental
material reverse rows are removed; the first reverse-coverage row for every SID
is always retained.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


MODEL = "/data/models/onereason-8b-pretrain-competition"
BASE = Path("/data/lf_data_versions/alltrain/mini_fix/onereason_mini_fix.jsonl")
USER_V2 = Path(
    "/data/lf_data_versions/task_pools/"
    + "\u61c2\u7528\u6237/mini_v2/understand_user_mini_v2.jsonl"
)
OUT_DIR = Path("/data/lf_data_versions/alltrain/mini_fix_u3k")
OUT_FILE = OUT_DIR / "onereason_mini_fix_u3k.jsonl"
SCHEMA = {
    "system", "instruction", "input", "output", "history",
    "data_source", "source_segment", "aux_metadata_json",
}
SID_RE = re.compile(
    r"<\|(ad|video|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
REMOVE_REVERSE_ROWS = 1000


def read_rows(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != SCHEMA:
                raise ValueError(f"Unexpected schema at {path}:{line_no}")
            result.append(row)
    return result


def row_bytes(row: dict) -> bytes:
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_tokens(row: dict, tokenizer) -> int:
    prompt = "".join(str(row[key]) for key in ("system", "instruction", "input"))
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + len(
        tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"]
    )


def token_stats(rows: list[dict], tokenizer) -> dict:
    values = [row_tokens(row, tokenizer) for row in rows]
    return {
        "rows": len(values),
        "total": sum(values),
        "mean": round(sum(values) / len(values), 4),
        "min": min(values),
        "max": max(values),
    }


def sid_key(row: dict) -> str:
    match = SID_RE.search(str(row["output"]))
    if not match:
        raise ValueError("reverse material row has no canonical SID")
    return match.group(0)


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT_DIR}")

    base_rows = read_rows(BASE)
    user_rows = read_rows(USER_V2)
    if len(base_rows) != 49490:
        raise ValueError(f"Mini-Fix base row count is {len(base_rows)}, expected 49490")
    if len(user_rows) != 3000:
        raise ValueError(f"mini_v2 user row count is {len(user_rows)}, expected 3000")

    base_user = [row for row in base_rows if row["data_source"] == "understand_user"]
    recommendation = [row for row in base_rows if row["data_source"] == "recommend"]
    material = [row for row in base_rows if row["data_source"] not in {"understand_user", "recommend"}]
    if Counter(row["source_segment"] for row in base_user) != Counter(
        {"user_action": 1200, "user_chain_cot": 200, "user_chain_nocot": 600}
    ):
        raise ValueError("Unexpected Mini-Fix user composition")
    if len(recommendation) != 11192 or len(material) != 36298:
        raise ValueError("Unexpected Mini-Fix recommendation/material composition")
    if Counter(row["source_segment"] for row in user_rows) != Counter(
        {"user_action": 1800, "user_chain_cot": 300, "user_chain_nocot": 900}
    ):
        raise ValueError("Unexpected mini_v2 U3K user composition")

    reverse = [row for row in material if row["source_segment"] == "sid_bucket_reverse"]
    reverse_by_sid: dict[str, list[dict]] = defaultdict(list)
    for row in reverse:
        reverse_by_sid[sid_key(row)].append(row)
    if len(reverse) != 15000 or len(reverse_by_sid) != 11298:
        raise ValueError("Unexpected Mini-Fix reverse material pool")
    primary = [rows[0] for rows in reverse_by_sid.values()]
    extras = [row for rows in reverse_by_sid.values() for row in rows[1:]]
    if len(primary) != 11298 or len(extras) != 3702:
        raise ValueError("Reverse pool is not 11,298 primary + 3,702 supplemental rows")

    remove_hashes = {
        hashlib.sha256(row_bytes(row)).hexdigest()
        for row in sorted(extras, key=row_bytes)[:REMOVE_REVERSE_ROWS]
    }
    kept_material = [
        row
        for row in material
        if not (
            row["source_segment"] == "sid_bucket_reverse"
            and hashlib.sha256(row_bytes(row)).hexdigest() in remove_hashes
        )
    ]
    if len(kept_material) != 35298:
        raise AssertionError("Material removal did not preserve the fixed 49,490-row budget")
    kept_reverse = [row for row in kept_material if row["source_segment"] == "sid_bucket_reverse"]
    if len(kept_reverse) != 14000 or len({sid_key(row) for row in kept_reverse}) != 11298:
        raise AssertionError("Reverse SID coverage was damaged")

    merged = user_rows + recommendation + kept_material
    if len(merged) != 49490:
        raise AssertionError("Unexpected Mini-Fix-U3K row count")

    OUT_DIR.mkdir(parents=True)
    digest = hashlib.sha256()
    with OUT_FILE.open("x", encoding="utf-8") as stream:
        for row in merged:
            payload = row_bytes(row)
            stream.write(payload.decode("utf-8"))
            digest.update(payload)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    by_segment = Counter(row["source_segment"] for row in merged)
    by_source = Counter(row["data_source"] for row in merged)
    manifest = {
        "kind": "train_ready_composed_dataset",
        "name": "mini_fix_u3k",
        "dataset_key": "onereason_mini_fix_u3k",
        "input_files": {
            "base_mini_fix": str(BASE),
            "user_mini_v2": str(USER_V2),
        },
        "input_sha256": {
            "base_mini_fix": sha256_file(BASE),
            "user_mini_v2": sha256_file(USER_V2),
        },
        "composition_order": ["understand_user", "recommendation", "material"],
        "composition_rule": (
            "Mini-Fix recommendation rows preserved byte-for-byte; user pool replaced "
            "by validated mini_v2 U3K pool; 1,000 supplemental reverse material rows "
            "removed deterministically by row hash while retaining one primary row per SID"
        ),
        "counts": {
            "total_rows": len(merged),
            "by_data_source": dict(sorted(by_source.items())),
            "by_source_segment": dict(sorted(by_segment.items())),
            "reverse_rows_removed": REMOVE_REVERSE_ROWS,
            "reverse_rows_kept": len(kept_reverse),
            "reverse_unique_sids_kept": len({sid_key(row) for row in kept_reverse}),
        },
        "recommendation_contract": {
            "rows": len(recommendation),
            "sha256": hashlib.sha256(b"".join(row_bytes(row) for row in recommendation)).hexdigest(),
            "short_think_high_purity_fix_preserved": True,
        },
        "material_contract": {
            "canonical_rows": by_segment["sid_bucket_canonical_no_think"],
            "reverse_rows": by_segment["sid_bucket_reverse"],
            "material_sample_rows": by_segment["material_sample"],
            "primary_sid_coverage_preserved": True,
        },
        "user_contract": {
            "rows": len(user_rows),
            "source": str(USER_V2),
            "composition": dict(Counter(row["source_segment"] for row in user_rows)),
        },
        "tokens": {
            "model": MODEL,
            "method": "AutoTokenizer add_special_tokens=False; prompt=system+instruction+input; total=prompt+output",
            "all": token_stats(merged, tokenizer),
            "user": token_stats(user_rows, tokenizer),
            "recommendation": token_stats(recommendation, tokenizer),
            "material": token_stats(kept_material, tokenizer),
        },
        "file": OUT_FILE.name,
        "sha256": digest.hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dataset_info = {
        "onereason_mini_fix_u3k": {
            "file_name": OUT_FILE.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction", "query": "input", "response": "output",
                "history": "history", "system": "system",
            },
        }
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# mini_fix_u3k\n\n"
        "Mini-Fix recommendation contract + validated 3,000-row user pool. "
        "The fixed 49,490-row budget is preserved by removing 1,000 supplemental "
        "reverse material rows without reducing primary SID coverage.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
