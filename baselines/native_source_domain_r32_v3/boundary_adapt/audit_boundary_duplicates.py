"""Read-only duplicate-group audit for Boundary Adapt source selection."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from build_boundary_adapt_dataset import is_think_recommendation, expand_row

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    groups=defaultdict(set); rows=defaultdict(int)
    with Path(a.source).open(encoding="utf-8") as fh:
        for line in fh:
            raw=json.loads(line)
            if not is_think_recommendation(raw): continue
            paths=expand_row(raw); first=paths[0]
            key=(hashlib.sha256(first["prompt"].encode()).hexdigest(), first["cot_sha256"], tuple(x["boundary_gold_sid"] for x in paths), first["target_domain"])
            groups[first["boundary_group_id"]].add(key); rows[first["boundary_group_id"]]+=1
    duplicated={g for g,n in rows.items() if n>1}; equivalent={g for g in duplicated if len(groups[g])==1}; conflicting=duplicated-equivalent
    result={"unique_groups":len(groups),"source_rows":sum(rows.values()),"duplicate_groups":len(duplicated),"byte_equivalent_duplicate_groups":len(equivalent),"conflicting_duplicate_groups":len(conflicting),"safe_dedup_possible":not conflicting}
    Path(a.out).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
