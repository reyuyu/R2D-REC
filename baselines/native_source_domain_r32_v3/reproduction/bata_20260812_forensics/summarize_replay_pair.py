#!/usr/bin/env python3
"""Summarize a forensic replay pair without exposing training examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def one_event(events: list[dict[str, Any]], event: str, label: str | None = None) -> dict[str, Any]:
    matches = [
        row for row in events if row.get("event") == event and (label is None or row.get("label") == label)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {event}/{label} event, found {len(matches)}")
    return matches[0]


def rank_files(run_dir: Path) -> dict[int, Path]:
    files = {}
    for path in (run_dir / "evidence").glob("rank*.jsonl"):
        files[int(path.stem.removeprefix("rank"))] = path
    if sorted(files) != [0, 1, 2, 3]:
        raise RuntimeError(f"Expected ranks 0..3, found {sorted(files)}")
    return files


def summarize_pair(run_a: Path, run_b: Path) -> dict[str, Any]:
    files_a, files_b = rank_files(run_a), rank_files(run_b)
    ranks = []
    optimizer_a: dict[int, dict[str, Any]] = {}
    optimizer_b: dict[int, dict[str, Any]] = {}
    for rank in range(4):
        optimizer_a[rank] = one_event(load_events(files_a[rank]), "optimizer_step")
        optimizer_b[rank] = one_event(load_events(files_b[rank]), "optimizer_step")
        a, b = optimizer_a[rank], optimizer_b[rank]
        ranks.append(
            {
                "rank": rank,
                "ordered_batch_fingerprint": a["ordered_batch_fingerprint"],
                "ordered_batch_equal": a["ordered_batch_fingerprint"] == b["ordered_batch_fingerprint"],
                "ordered_microbatches_equal": a["ordered_microbatch_sha256"]
                == b["ordered_microbatch_sha256"],
                "rng_fingerprint_a": canonical_hash(a["rng"]),
                "rng_fingerprint_b": canonical_hash(b["rng"]),
                "rng_equal": a["rng"] == b["rng"],
                "microbatch_count": a["microbatch_count"],
                "micro_losses_equal": a["rank_local_micro_losses"] == b["rank_local_micro_losses"],
                "local_loss_a": a["rank_local_loss_mean"],
                "local_loss_b": b["rank_local_loss_mean"],
                "local_loss_equal": a["rank_local_loss_mean"] == b["rank_local_loss_mean"],
                "grad_norm_a": a["grad_norm"],
                "grad_norm_b": b["grad_norm"],
                "grad_norm_equal": a["grad_norm"] == b["grad_norm"],
            }
        )

    events_a0, events_b0 = load_events(files_a[0]), load_events(files_b[0])
    initial_a = one_event(events_a0, "heavy_fingerprint", "initial553")
    initial_b = one_event(events_b0, "heavy_fingerprint", "initial553")
    final_a = one_event(events_a0, "heavy_fingerprint", "step554")
    final_b = one_event(events_b0, "heavy_fingerprint", "step554")
    final = {
        "initial_lora_equal": initial_a["canonical_lora_sha256"] == initial_b["canonical_lora_sha256"],
        "initial_optimizer_equal": initial_a["optimizer"]["sha256"] == initial_b["optimizer"]["sha256"],
        "lora_a": final_a["canonical_lora_sha256"],
        "lora_b": final_b["canonical_lora_sha256"],
        "lora_equal": final_a["canonical_lora_sha256"] == final_b["canonical_lora_sha256"],
        "effective_ba_a": final_a["effective_ba"]["sha256"],
        "effective_ba_b": final_b["effective_ba"]["sha256"],
        "effective_ba_equal": final_a["effective_ba"]["sha256"] == final_b["effective_ba"]["sha256"],
        "optimizer_a": final_a["optimizer"]["sha256"],
        "optimizer_b": final_b["optimizer"]["sha256"],
        "optimizer_equal": final_a["optimizer"]["sha256"] == final_b["optimizer"]["sha256"],
    }
    local_equal = all(
        row["ordered_batch_equal"]
        and row["ordered_microbatches_equal"]
        and row["rng_equal"]
        and row["micro_losses_equal"]
        and row["local_loss_equal"]
        for row in ranks
    )
    post_backward_equal = all(row["grad_norm_equal"] for row in ranks) and all(
        final[key] for key in ("lora_equal", "effective_ba_equal", "optimizer_equal")
    )
    return {
        "classification": "D2" if local_equal and post_backward_equal else "D3",
        "pre_backward_local_loss_repeatable": local_equal,
        "post_backward_state_repeatable": post_backward_equal,
        "ranks": ranks,
        "step554": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_pair(args.run_a, args.run_b), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
