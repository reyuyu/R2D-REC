#!/usr/bin/env python3
"""CPU-only static eligibility audit for mini_topK on alpha_mini_v1 cache."""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CACHE = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/tokenized_alpha_mini_v1_train_8k_sid8w8")
REPORT = Path("/data/lf_data_versions/alltrain/alpha_mini_v1/mini_topk_static_cache_audit.json")
CONFIG = ROOT / "config/train_mini_topK_v1_4gpu_gc04_2epoch.yaml"
IGNORE = -100

def main() -> None:
    # Tokenizer only; never load a model or initialize CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from rec_pu.sid8_rec_pu_integration import build_sid_component_vocab
    cfg = __import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    vocab = build_sid_component_vocab(tok)
    level_vocab = {key: set(value) for key, value in (("a",vocab.a),("b",vocab.b),("c",vocab.c))}
    data = load_from_disk(str(CACHE))["train"]
    source = Counter(); target_source = Counter(); failures = Counter(); segments = 0; packs = 0
    positive_sizes = {level: Counter() for level in "abc"}
    for row in data:
        packs += 1
        labels, weights = row["labels"], row["loss_weights"]
        targets = json.loads(row.get("rec_pu_targets_json", "[]"))
        for target in targets:
            segments += 1
            segment = target.get("source_segment")
            target_source[segment] += 1
            if segment not in {"recommendation_cot", "recommendation_nocot"}: failures["source_segment"] += 1
            for level in "abc":
                label_pos = int(target[f"{level}_label_position"]); logit_pos = int(target[f"{level}_logit_position"])
                positives = tuple(int(x) for x in target["positives"][level])
                positive_sizes[level][len(positives)] += 1
                if not positives: failures[f"{level}_empty"] += 1
                if label_pos - 1 != logit_pos: failures["causal_offset"] += 1
                if label_pos < 0 or label_pos >= len(labels) or labels[label_pos] == IGNORE: failures["ignored_or_oob"] += 1
                elif int(labels[label_pos]) not in positives: failures["gold_not_positive"] += 1
                if not set(positives).issubset(level_vocab[level]): failures[f"{level}_vocab"] += 1
                if label_pos < len(weights) and float(weights[label_pos]) != 8.0: failures["sid8"] += 1
    report = {
        "kind": "mini_topk_alpha_mini_v1_static_cache_audit_v1", "cache": str(CACHE), "cache_packs": packs,
        "recommendation_segments": segments, "target_source": dict(target_source),
        "positive_size_histograms": {key: dict(value) for key,value in positive_sizes.items()},
        "component_vocab_sizes": {"a":len(vocab.a),"b":len(vocab.b),"c":len(vocab.c)},
        "failures": dict(failures), "pass": segments == 11192 and not failures and target_source == {"recommendation_cot":6235,"recommendation_nocot":4957},
        "cuda_used": False, "model_loaded": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["pass"]: raise SystemExit("mini_topK cache audit failed")

if __name__ == "__main__": main()
