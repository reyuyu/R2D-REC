#!/usr/bin/env python3
"""Build Mini-Short-CoT by retaining only recommendation CoT interest summaries."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PARENT_ROOT = Path("/data/lf_data_versions/alltrain/mini_fix")
PARENT_DATA = PARENT_ROOT / "onereason_mini_fix.jsonl"
PARENT_MANIFEST = PARENT_ROOT / "manifest.json"
OUTPUT_ROOT = Path("/data/lf_data_versions/alltrain/mini_short_cot")
OUTPUT_DATA = OUTPUT_ROOT / "onereason_mini_short_cot.jsonl"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest.json"
OUTPUT_REGISTRY = OUTPUT_ROOT / "dataset_info.json"
INTEREST = "【兴趣归纳】"
LATER_SECTIONS = ("【行为模式】", "【预测总结】")
NUMBERED_INTEREST_RE = re.compile(
    r"(?m)^[ \t]*#{1,6}[ \t]*(?:\*\*)?(?:1[.、][ \t]*)?兴趣归纳(?:\*\*)?[ \t]*[:：]?[ \t]*$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def short_cot_output(output: str) -> tuple[str, dict]:
    open_index = output.find("<think>")
    close_index = output.find("</think>")
    if open_index < 0 or close_index < 0 or close_index <= open_index:
        raise ValueError("recommendation_cot output must contain one ordered think span")
    if output.find("<think>", open_index + 1) >= 0 or output.find("</think>", close_index + 1) >= 0:
        raise ValueError("recommendation_cot output contains multiple think spans")
    think_start = open_index + len("<think>")
    think = output[think_start:close_index]
    interest_index = think.find(INTEREST)
    marker_variant = "bracketed"
    if interest_index >= 0:
        content_start = interest_index + len(INTEREST)
    else:
        match = NUMBERED_INTEREST_RE.search(think)
        if match is None:
            raise ValueError("recommendation_cot think span has no interest-summary marker")
        interest_index = match.start()
        content_start = match.end()
        marker_variant = "numbered_heading"
    later_positions = [think.find(marker, content_start) for marker in LATER_SECTIONS]
    later_positions = [position for position in later_positions if position >= 0]
    content_end = min(later_positions) if later_positions else len(think)
    body = think[content_start:content_end]
    # Remove only heading-adjacent markdown/punctuation, not section content.
    body = re.sub(r"^[\s*#:：_-]+", "", body).strip()
    body = re.sub(r"(?:^|\n)\s*#{1,6}\s*\**\s*$", "", body).strip()
    if not body:
        raise ValueError("interest-summary section is empty")
    new_think = f"\n{INTEREST}\n{body}\n"
    new_output = output[:think_start] + new_think + output[close_index:]
    new_inner = new_output[think_start:new_output.index("</think>")]
    if any(marker in new_inner for marker in LATER_SECTIONS):
        raise AssertionError("later recommendation CoT section survived shortening")
    if output[close_index:] != new_output[new_output.index("</think>"):]:
        raise AssertionError("final recommendation answer changed")
    return new_output, {
        "old_think_chars": len(think),
        "new_think_chars": len(new_inner),
        "had_behavior": LATER_SECTIONS[0] in think,
        "had_prediction": LATER_SECTIONS[1] in think,
        "marker_variant": marker_variant,
    }


def main() -> None:
    if not PARENT_DATA.is_file() or not PARENT_MANIFEST.is_file():
        raise FileNotFoundError("Mini-Fix parent dataset is incomplete")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing data version: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)

    parent_manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    counts = Counter()
    transformed = 0
    unchanged_rows = 0
    final_answer_parity = 0
    metadata_parity = 0
    old_chars = 0
    new_chars = 0
    had_behavior = 0
    had_prediction = 0
    marker_variants = Counter()

    with PARENT_DATA.open("r", encoding="utf-8", newline="") as reader, OUTPUT_DATA.open(
        "w", encoding="utf-8", newline=""
    ) as writer:
        for row_index, raw_line in enumerate(reader, 1):
            row = json.loads(raw_line)
            segment = str(row.get("source_segment", ""))
            counts[segment] += 1
            if segment != "recommendation_cot":
                writer.write(raw_line)
                unchanged_rows += 1
                continue

            before = dict(row)
            old_output = str(row.get("output", ""))
            new_output, stats = short_cot_output(old_output)
            row["output"] = new_output
            if any(row[key] != before[key] for key in before if key != "output"):
                raise AssertionError(f"Non-output field changed at row {row_index}")
            old_suffix = old_output[old_output.index("</think>"):]
            new_suffix = new_output[new_output.index("</think>"):]
            final_answer_parity += old_suffix == new_suffix
            metadata_parity += row.get("aux_metadata_json") == before.get("aux_metadata_json")
            transformed += 1
            old_chars += stats["old_think_chars"]
            new_chars += stats["new_think_chars"]
            had_behavior += stats["had_behavior"]
            had_prediction += stats["had_prediction"]
            marker_variants[stats["marker_variant"]] += 1
            writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    expected_counts = parent_manifest["counts"]["by_source_segment"]
    if dict(counts) != expected_counts:
        raise RuntimeError(f"Parent row-count contract changed: {dict(counts)} != {expected_counts}")
    if transformed != 6235 or unchanged_rows != 43255:
        raise RuntimeError(f"Unexpected transformation counts: transformed={transformed}, unchanged={unchanged_rows}")
    if final_answer_parity != transformed or metadata_parity != transformed:
        raise RuntimeError("Final-answer or recommendation metadata parity failed")

    output_sha = sha256(OUTPUT_DATA)
    manifest = {
        "kind": "train_ready_composed_dataset",
        "name": "mini_short_cot",
        "dataset_key": "onereason_mini_short_cot",
        "parent": {
            "name": "mini_fix",
            "path": str(PARENT_DATA),
            "manifest": str(PARENT_MANIFEST),
            "sha256": sha256(PARENT_DATA),
        },
        "single_variable": (
            "For every recommendation_cot row, retain only the content under 【兴趣归纳】 inside "
            "<think>...</think>; preserve the final answer and every other field."
        ),
        "counts": {
            "total_rows": sum(counts.values()),
            "by_source_segment": dict(counts),
            "recommendation_cot_transformed": transformed,
            "unchanged_rows_byte_preserved": unchanged_rows,
        },
        "parity": {
            "row_order_preserved": True,
            "non_cot_rows_byte_preserved": unchanged_rows == 43255,
            "cot_non_output_fields_preserved": True,
            "cot_final_answer_preserved": final_answer_parity == transformed,
            "cot_aux_metadata_preserved": metadata_parity == transformed,
            "recommendation_nocot_preserved": True,
            "user_preserved": True,
            "material_preserved": True,
        },
        "cot_audit": {
            "interest_marker_found": transformed,
            "had_behavior_section": had_behavior,
            "had_prediction_section": had_prediction,
            "old_think_chars": old_chars,
            "new_think_chars": new_chars,
            "think_char_reduction_ratio": 1.0 - new_chars / old_chars,
            "later_section_markers_remaining": 0,
            "marker_variants": dict(marker_variants),
        },
        "file": OUTPUT_DATA.name,
        "sha256": output_sha,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry = {
        "onereason_mini_short_cot": {
            "file_name": str(OUTPUT_DATA),
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }
    }
    OUTPUT_REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
