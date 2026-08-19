"""Tokenizer-only bridge token contract audit; this never loads model weights."""

from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from grpo_model import BASE


TOKEN_RE = re.compile(r"<s_(a|b)_(\d+)>")


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    vocabulary = tokenizer.get_vocab()
    hierarchy_tokens = sorted(
        (token for token in vocabulary if TOKEN_RE.fullmatch(token)),
        key=lambda token: (TOKEN_RE.fullmatch(token).group(1), int(TOKEN_RE.fullmatch(token).group(2))),
    )
    domains = ["<|video_begin|>", "<|prod_begin|>", "<|ad_begin|>", "<|living_begin|>"]
    failures = []
    for token in domains + hierarchy_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != vocabulary[token]:
            failures.append({"token": token, "encoded": encoded, "vocab_id": vocabulary.get(token)})
    by_level = {}
    for level in ("a", "b"):
        values = [int(TOKEN_RE.fullmatch(token).group(2)) for token in hierarchy_tokens if f"<s_{level}_" in token]
        by_level[level] = {
            "count": len(values),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    result = {
        "audit": "NoThink bridge tokenizer-only single-token contract",
        "model_weights_loaded": False,
        "tokenizer_path": BASE,
        "domains_checked": domains,
        "hierarchy": by_level,
        "failure_count": len(failures),
        "failures": failures[:20],
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "runtime_contract": "Every concrete domain/A/B teacher token is revalidated before use.",
    }
    output = Path(__file__).with_name("bridge_tokenizer_audit.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
