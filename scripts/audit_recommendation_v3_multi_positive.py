#!/usr/bin/env python3
"""Audit Recommendation V3 metadata, text identity, and group integrity."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SID_RE = re.compile(r"<\|(?P<domain>prod|video|living|ad)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
DOMAIN_NAMES = {"prod": "商品", "video": "视频", "living": "直播", "ad": "广告"}


def canonical_prompt(row: dict) -> str:
    input_text = re.sub(r"\s*/(?:think|no_think)\s*$", "", row.get("input", "")).rstrip()
    return json.dumps([row.get("instruction", ""), input_text, row.get("history", [])], ensure_ascii=False, separators=(",", ":"))


def without_metadata(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("recommendation_")}


def audit(args: argparse.Namespace) -> None:
    source_rows = []
    for path in (args.cot_source, args.nocot_source):
        with path.open(encoding="utf-8") as file:
            source_rows.extend(without_metadata(json.loads(line)) for line in file)
    output_rows = []
    groups: dict[str, list[dict]] = {}
    for path in (args.cot_output, args.nocot_output):
        with path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                output_rows.append(row)
                match = SID_RE.fullmatch(row["recommendation_current_gold_sid"])
                if match is None:
                    raise AssertionError("current_gold_sid is not a complete SID")
                domain = DOMAIN_NAMES[match.group("domain")]
                expected_id = hashlib.sha256((canonical_prompt(row) + "\0" + domain).encode("utf-8")).hexdigest()
                assert expected_id == row["recommendation_group_id"]
                assert row["recommendation_group_size"] == len(row["recommendation_all_gold_sids"])
                assert row["recommendation_current_gold_sid"] in row["recommendation_all_gold_sids"]
                groups.setdefault(row["recommendation_group_id"], []).append(row)
    for rows in groups.values():
        current = [row["recommendation_current_gold_sid"] for row in rows]
        assert len(current) == len(set(current))
        assert len(rows) == rows[0]["recommendation_group_size"]
        assert all(row["recommendation_all_gold_sids"] == rows[0]["recommendation_all_gold_sids"] for row in rows)
    actual = Counter(json.dumps(without_metadata(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in output_rows)
    expected = Counter(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in source_rows)
    assert actual == expected, "V3 text fields differ from V2 source"
    print(json.dumps({
        "status": "PASS",
        "output_rows": len(output_rows),
        "groups": len(groups),
        "cot_samples": sum("/think" in row.get("input", "") for row in output_rows),
        "nocot_samples": sum("/no_think" in row.get("input", "") for row in output_rows),
        "cot_sha256": hashlib.sha256(args.cot_output.read_bytes()).hexdigest(),
        "nocot_sha256": hashlib.sha256(args.nocot_output.read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    source = Path("/data/lf_data_versions/alltrain/v2_recommendation_dual")
    output = Path("/data/lf_data_versions/alltrain/v3_recommendation_multi_positive")
    p.add_argument("--cot-source", type=Path, default=source / "onereason_recommendation_cot_v2_dual.jsonl")
    p.add_argument("--nocot-source", type=Path, default=source / "onereason_recommendation_nocot_v2_dual.jsonl")
    p.add_argument("--cot-output", type=Path, default=output / "onereason_recommendation_cot_v3_multi_positive.jsonl")
    p.add_argument("--nocot-output", type=Path, default=output / "onereason_recommendation_nocot_v3_multi_positive.jsonl")
    return p


if __name__ == "__main__":
    audit(build_parser().parse_args())
