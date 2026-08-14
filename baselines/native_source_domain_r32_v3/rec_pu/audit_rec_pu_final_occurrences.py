from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/data/lf_data_versions/alltrain/BETA")
for filename in ("onereason_recommendation_cot.jsonl", "onereason_recommendation_nocot.jsonl"):
    counts = Counter()
    examples = []
    with (ROOT / filename).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            current = row["recommendation_current_gold_sid"]
            found = row["output"].count(current)
            counts[found] += 1
            if found != 1 and len(examples) < 3:
                examples.append({"line": number, "occurrences": found, "tail": row["output"][-500:]})
    print(json.dumps({"file": filename, "occurrences": dict(sorted(counts.items())), "examples": examples}, ensure_ascii=False))
