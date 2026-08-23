"""Build immutable-CoT, target-domain-boundary adaptation rows from real REC data."""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from pathlib import Path

DOMAIN = {"video":"<|video_begin|>", "prod":"<|prod_begin|>", "ad":"<|ad_begin|>", "living":"<|living_begin|>"}
SID_RE = re.compile(r"^(<s_a_\d+>)(<s_b_\d+>)(<s_c_\d+>)$")

def _get(row, *names):
    for name in names:
        if name in row and row[name] is not None: return row[name]
    raise ValueError(f"missing required field: {names}")

def response_of(row): return _get(row, "output", "response", "completion")
def is_think_recommendation(row):
    if row.get("route") == "think": return True
    segment = str(row.get("source_segment", "")).lower()
    return row.get("data_source") == "recommend" and "cot" in segment and "nocot" not in segment
def golds_of(row):
    raw = row.get("recommendation_all_gold_sids")
    if raw is None and "aux_metadata_json" in row:
        raw = json.loads(row["aux_metadata_json"]).get("recommendation_all_gold_sids")
    if raw is None: raise ValueError("missing recommendation_all_gold_sids")
    if isinstance(raw, str): raw = json.loads(raw)
    if not isinstance(raw, list) or not raw: raise ValueError("recommendation_all_gold_sids is empty")
    return raw

def adapt_response(response: str, domain: str, sid: str):
    if domain not in DOMAIN: raise ValueError(f"unknown target_domain {domain!r}")
    if not isinstance(response, str) or "</think>" not in response: raise ValueError("missing </think>")
    cot, tail = response.split("</think>", 1)
    if "</think>" in tail or not cot.startswith("<think>") or not cot[len("<think>"):].strip(): raise ValueError("malformed or empty think response")
    if not SID_RE.fullmatch(sid): raise ValueError(f"malformed full SID {sid!r}")
    return cot + "</think>" + DOMAIN[domain] + sid, cot, tail

def expand_row(row):
    is_grpo = row.get("route") == "think"
    is_bata = is_think_recommendation(row) and not is_grpo
    if not (is_grpo or is_bata): return []
    prompt = row.get("prompt") if is_grpo else _get(row, "instruction") + str(row.get("input", ""))
    metadata = json.loads(row["aux_metadata_json"]) if "aux_metadata_json" in row else row
    group_id = _get(metadata, "recommendation_group_id")
    domain = row.get("target_domain")
    response, golds = response_of(row), golds_of(row)
    # BATA golds include their source-domain token; validate it then store ABC only.
    inferred = []
    for gold in golds:
        matched = [key for key, token in DOMAIN.items() if str(gold).startswith(token)]
        if matched:
            if len(matched) != 1: raise ValueError(f"gold has no unique domain token {gold!r}")
            if domain is None: domain = matched[0]
            if domain != matched[0]: raise ValueError("target_domain conflicts with full gold SID")
            inferred.append(str(gold)[len(DOMAIN[domain]):])
        else:
            if domain is None: raise ValueError("ABC-only gold requires target_domain")
            inferred.append(str(gold))
    unique = list(dict.fromkeys(inferred))
    if len(unique) != len(golds): raise ValueError("duplicate all-gold SID")
    out=[]
    for sid in unique:
        adapted, cot, bridge = adapt_response(response, domain, sid)
        out.append({"boundary_group_id":group_id,"boundary_group_size":len(unique),"boundary_sample_weight":1.0/len(unique),"boundary_gold_sid":sid,"prompt":prompt,"target_domain":domain,"original_response":response,"adapted_response":adapted,"cot_sha256":hashlib.sha256(cot.encode()).hexdigest(),"bridge_removed":bridge})
    return out

def build(source: Path, output: Path):
    stats=collections.Counter(); groups=set(); domains=collections.Counter(); sizes=collections.Counter(); rows=[]; seen={}
    with source.open(encoding="utf-8") as fh:
        for line_no,line in enumerate(fh,1):
            raw=json.loads(line)
            if not is_think_recommendation(raw): continue
            stats["ORIGINAL_THINK_REC_SAMPLES"]+=1
            try: expanded=expand_row(raw)
            except Exception as exc: raise ValueError(f"fail-closed source line {line_no}: {exc}") from exc
            group=expanded[0]["boundary_group_id"]
            signature=(expanded[0]["prompt"],expanded[0]["cot_sha256"],tuple(x["boundary_gold_sid"] for x in expanded),expanded[0]["target_domain"])
            if group in seen:
                if seen[group] != signature: raise ValueError(f"conflicting duplicate group {group}")
                stats["BYTE_EQUIVALENT_DUPLICATES_DROPPED"] += 1
                continue
            seen[group]=signature; groups.add(group); k=len(expanded); sizes[k]+=1; domains[expanded[0]["target_domain"]]+=1
            for item in expanded:
                if hashlib.sha256(item["adapted_response"].split("</think>",1)[0].encode()).hexdigest()!=item["cot_sha256"]: raise AssertionError("CoT changed")
                rows.append(item)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8") as fh:
        for row in rows: fh.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {"dataset_source":str(source),"source_think_rows":stats["ORIGINAL_THINK_REC_SAMPLES"],"original_groups":len(groups),"byte_equivalent_duplicates_dropped":stats["BYTE_EQUIVALENT_DUPLICATES_DROPPED"],"expanded_paths":len(rows),"group_size_distribution":dict(sizes),"per_domain":dict(domains),"cot_preserve_pass":True,"bridge_removal_pass":True,"fixed_domain_pass":True}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--output",required=True); p.add_argument("--stats",required=True); a=p.parse_args()
    stats=build(Path(a.source),Path(a.output)); Path(a.stats).write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
if __name__=="__main__": main()
