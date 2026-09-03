#!/usr/bin/env python3
"""Build rec_mp_grpo_v2 by domain-specializing rec_mp_grpo_v1 prompts.

This is the standalone form of the historical one-off builder recovered from
the 2026-08-16 construction session. It changes only the ``prompt`` field.
Group IDs, routes, target domains, gold SID sets, and history SID lines are
preserved exactly.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


SID_DOMAIN_RE = re.compile(r"<\|(?:video|prod|ad|living)_begin\|>")
OPENING = {
    "video": "请阅读该用户的对话行为记录，推断该用户下一个可能感兴趣的视频。",
    "prod": "请阅读该用户的对话行为记录，推断该用户下一个可能感兴趣的商品。",
    "ad": "请阅读该用户的对话行为记录，推断该用户下一个可能感兴趣的广告。",
    "living": "请阅读该用户的对话行为记录，推断该用户下一个可能观看的直播。",
}
CLOSING = {
    "video": [
        "请根据以上信息，推断该用户下一个可能感兴趣的视频。",
        "请结合以上用户行为，推荐该用户下一个可能观看的视频。",
        "请综合以上信息，推测该用户接下来最可能点击观看的视频。",
    ],
    "prod": [
        "请根据以上信息，推断该用户下一个可能感兴趣的商品。",
        "请结合以上用户行为，推荐该用户下一个可能购买的商品。",
        "请综合以上信息，推测该用户接下来最可能关注购买的商品。",
    ],
    "ad": [
        "请根据以上信息，推断该用户下一个可能感兴趣的广告。",
        "请结合以上用户行为，推荐该用户下一个可能点击的广告。",
        "请综合以上信息，推测该用户接下来最可能点击的广告。",
    ],
    "living": [
        "请根据以上信息，推断该用户下一个可能观看的直播。",
        "请结合以上用户行为，推荐该用户下一个可能关注的直播。",
        "请综合以上信息，推测该用户接下来最可能观看的直播。",
    ],
}


def variant_for(group_id: str, domain: str) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % len(CLOSING[domain])


def sid_history_lines(prompt: str) -> list[str]:
    return [line for line in prompt.split("\n") if SID_DOMAIN_RE.search(line)]


def rewrite_record(row: dict[str, Any]) -> dict[str, Any]:
    group_id = row["recommendation_group_id"]
    domain = row["target_domain"]
    route = row["route"]
    if domain not in OPENING:
        raise ValueError(f"unsupported target_domain: {domain!r}")
    if route not in {"think", "no_think"}:
        raise ValueError(f"unsupported route: {route!r}")

    marker = "/think" if route == "think" else "/no_think"
    body = row["prompt"].rstrip()
    if not body.endswith(marker):
        raise ValueError(f"prompt marker does not match route for group {group_id}")

    lines = body[: -len(marker)].split("\n")
    text_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip() and not SID_DOMAIN_RE.search(line)
    ]
    if not text_indices:
        raise ValueError(f"prompt has no instruction line for group {group_id}")

    first, last = text_indices[0], text_indices[-1]
    lines[first] = OPENING[domain]
    if first != last:
        lines[last] = CLOSING[domain][variant_for(group_id, domain)]

    rewritten = dict(row)
    rewritten["prompt"] = "\n".join(lines) + marker
    return rewritten


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        rewritten = rewrite_record(row)
        changed_fields = {
            key
            for key in set(row) | set(rewritten)
            if row.get(key) != rewritten.get(key)
        }
        if changed_fields != {"prompt"}:
            raise AssertionError(f"unexpected changed fields: {sorted(changed_fields)}")
        if sid_history_lines(row["prompt"]) != sid_history_lines(rewritten["prompt"]):
            raise AssertionError("history SID lines changed")
        output.append(rewritten)
    return output


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            # Keep the historical writer format for byte-level reproducibility.
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    gold_counts = sorted(int(row["gold_count"]) for row in rows)
    return {
        "kind": "grpo_train_dataset",
        "name": "rec_mp_grpo_v2",
        "experiment": "REC-MP-GRPO-v1",
        "note": (
            "v2: prompts rewritten with target-domain-specific instructions "
            "(video/prod/ad/living), 3 template variants per domain, stably "
            "assigned per group; history SID lines and /think|/no_think markers unchanged"
        ),
        "source": {"version": "rec_mp_grpo_v1", "path": str(source)},
        "grpo_train_records": len(rows),
        "route_split": dict(collections.Counter(row["route"] for row in rows)),
        "target_domain_records": dict(
            collections.Counter(row["target_domain"] for row in rows)
        ),
        "gold_count": {
            "mean": round(statistics.mean(gold_counts), 3),
            "p50": gold_counts[len(gold_counts) // 2],
        },
        "issues": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    rows = build_rows(load_jsonl(args.source))
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "train.jsonl", rows)
    manifest = build_manifest(rows, args.source)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
