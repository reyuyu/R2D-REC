import json
import os
import re
from collections import Counter
from pathlib import Path


root = Path(os.getenv("NATIVE_BASELINE_DATASET_DIR", "/data/baselines/native_source_domain_r32_v3/dataset"))
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
dataset = root / "onereason_native_source_domain_r32_v3.jsonl"

sources = Counter()
domains = Counter()
last_source_rank = -1
order = manifest["source_order"]
source_rank = {name: index for index, name in enumerate(order)}
domain_pattern = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
expected_schema = manifest["schema"]
rows = 0
with dataset.open(encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        row = json.loads(line)
        if list(row) != expected_schema:
            raise AssertionError(f"line {line_no}: unexpected schema={list(row)}")
        source = row.get("data_source")
        if source not in source_rank:
            raise AssertionError(f"line {line_no}: unknown data_source={source!r}")
        if source_rank[source] < last_source_rank:
            raise AssertionError(f"line {line_no}: source buckets are not deterministic")
        last_source_rank = source_rank[source]
        if "world" in source.lower() or "world" in str(row.get("dataset", "")).lower():
            raise AssertionError(f"line {line_no}: world row leaked into baseline")
        if not isinstance(row.get("instruction"), str) or not isinstance(row.get("output"), str):
            raise AssertionError(f"line {line_no}: malformed Alpaca sample")
        metadata = row.get("aux_metadata_json")
        if not isinstance(metadata, str):
            raise AssertionError(f"line {line_no}: aux_metadata_json is not a string")
        if source == "recommend":
            parsed = json.loads(metadata)
            required = {"version", "group_id", "group_size", "all_gold_sids", "current_gold_sid"}
            if set(parsed) != required or parsed["version"] != "recommendation_v3_multi_positive":
                raise AssertionError(f"line {line_no}: malformed recommendation metadata")
        elif metadata:
            raise AssertionError(f"line {line_no}: non-recommend row has auxiliary metadata")
        sources[source] += 1
        # Domain weights are deliberately applied only to material_sample.
        # The reverse bucket follows the reference's uniform SID-token loss.
        if source == "material_sample":
            text = "\n".join(str(row.get(key, "")) for key in ("instruction", "input", "output"))
            match = domain_pattern.search(text)
            if match is None:
                raise AssertionError(f"line {line_no}: material sample has no semantic domain token")
            domains[match.group(1)] += 1
        rows += 1

assert rows == manifest["records"]
assert dict(sources) == manifest["source_counts"]
assert sum(sources.values()) == rows
print(f"PASS records={rows}")
print("PASS sources=" + json.dumps(dict(sources), sort_keys=True))
print("PASS material_domains=" + json.dumps(dict(domains), sort_keys=True))
