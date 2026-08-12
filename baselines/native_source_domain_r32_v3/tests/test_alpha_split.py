"""Synthetic regression tests for the alpha leak-safe split primitives."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "split_alpha_jiankong_validation.py"
SPEC = importlib.util.spec_from_file_location("alpha_split", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(source: str, segment: str, instruction: str, output: str, *, metadata: dict | None = None) -> dict:
    return {
        "system": "s",
        "instruction": instruction,
        "input": "",
        "output": output,
        "history": [],
        "data_source": source,
        "source_segment": segment,
        "aux_metadata_json": "" if metadata is None else json.dumps(metadata, ensure_ascii=False),
    }


def rec(group: str, gold: str, all_golds: list[str], segment: str) -> dict:
    domain = gold.split("_begin", 1)[0].replace("<|", "")
    return row(
        "recommend",
        segment,
        f"history <|video_begin|><s_a_{group.rsplit('-', 1)[-1]}><s_b_2><s_c_3> {domain}/no_think",
        gold,
        metadata={
            "recommendation_group_id": group,
            "recommendation_group_size": len(all_golds),
            "recommendation_current_gold_sid": gold,
            "recommendation_all_gold_sids": all_golds,
        },
    )


def test_deterministic_and_preserved() -> None:
    values = []
    domains = ("video", "prod", "ad", "living")
    for domain in domains:
        for index in range(100):
            g1 = f"<|{domain}_begin|><s_a_{index}><s_b_2><s_c_3>"
            g2 = f"<|{domain}_begin|><s_a_{index + 100}><s_b_2><s_c_3>"
            values += [rec(f"{domain}-{index}", g1, [g1, g2], "recommendation_cot"), rec(f"{domain}-{index}", g2, [g1, g2], "recommendation_nocot")]
    for index in range(200):
        values.append(row("understand_user", "user_action", f"history-{index}/no_think", "answer"))
        values.append(row("material_sample", "material_sample", f"item-{index}/think", "caption"))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jsonl"
        source.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in values), encoding="utf-8")
        first, second = root / "first", root / "second"
        import sys
        old_argv = sys.argv
        try:
            sys.argv = [str(SCRIPT), "--source", str(source), "--output", str(first)]
            MODULE.main()
            sys.argv = [str(SCRIPT), "--source", str(source), "--output", str(second)]
            MODULE.main()
        finally:
            sys.argv = old_argv
        assert MODULE.sha256_file(first / "train.jsonl") == MODULE.sha256_file(second / "train.jsonl")
        assert MODULE.sha256_file(first / "dev.jsonl") == MODULE.sha256_file(second / "dev.jsonl")
        source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        output_rows = [
            json.loads(line)
            for path in (first / "train.jsonl", first / "dev.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        required = {"system", "instruction", "input", "output", "history", "data_source", "source_segment", "aux_metadata_json"}
        assert len(output_rows) == len(source_rows)
        assert all(required <= set(item) for item in output_rows)
        # Splitting must preserve every row and its metadata byte-for-byte at JSON-field level.
        assert Counter(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in output_rows) == Counter(
            json.dumps(item, ensure_ascii=False, sort_keys=True) for item in source_rows
        )
        for item in output_rows:
            if item["data_source"] == "recommend":
                metadata = json.loads(item["aux_metadata_json"])
                assert {"recommendation_group_id", "recommendation_group_size", "recommendation_current_gold_sid", "recommendation_all_gold_sids"} <= set(metadata)
        audit = json.loads((first / "split_audit.json").read_text(encoding="utf-8"))
        assert all(value == 0 for key, value in audit["leakage"].items() if key.endswith("overlap"))
        assert audit["leakage"]["row_conservation"]
        assert audit["leakage"]["duplicate_delta_from_source"] == 0
        assert all(audit["recommendation"]["domain_rows"][domain]["dev"] > 0 for domain in domains)


if __name__ == "__main__":
    test_deterministic_and_preserved()
    print("alpha split tests: PASS")
