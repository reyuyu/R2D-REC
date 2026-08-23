"""Four-GPU, inference-only Probe1 comparison of bare and SFT-bridge ABC3."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from .probe1_beam_anatomy import (
    ADAPTER,
    BASE,
    FIXED_PROBE,
    GROUP_ID,
    HISTORICAL_EXACT,
    OLD_PROBE,
    closed_cot,
    completion_sha,
    encode_prompt,
    gold_ids,
    hf_beam,
    load_model,
    load_probe,
    parse_abc3,
    parse_gold,
    prefill,
    score_sequences,
    token_rank,
)

GRPO_ROOT = Path(__file__).resolve().parents[3]
SFT_SOURCE = Path(
    "/data/lf_data_versions/alltrain/bata_baseline_v1/"
    "onereason_bata_baseline.jsonl"
)
DEFAULT_PARTS = Path("/data/GRPO/results/probe1_bridge_abc3_20260823_parts")
DEFAULT_OUTPUT = GRPO_ROOT / (
    "results/gr_rec_think_composite_interest_v1_probe1_bridge_abc3_20260823.json"
)
VIDEO_PREFIX = "<|video_begin|>"


def extract_sft_bridge():
    bridges = set()
    matching_rows = 0
    with SFT_SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = json.loads(row.get("aux_metadata_json") or "{}")
            if metadata.get("recommendation_group_id") != GROUP_ID:
                continue
            if row.get("source_segment") != "recommendation_cot":
                continue
            matching_rows += 1
            target = row["output"]
            left = target.index("</think>") + len("</think>")
            right = target.index(VIDEO_PREFIX, left)
            bridges.add(target[left:right])
    if matching_rows != 12 or len(bridges) != 1:
        raise RuntimeError(
            f"SFT_BRIDGE_PROVENANCE_INVALID rows={matching_rows} bridges={bridges!r}"
        )
    return bridges.pop(), matching_rows


def paired_probe():
    old, fixed = load_probe(OLD_PROBE), load_probe(FIXED_PROBE)
    old_candidates = old["think"]["candidates"]
    fixed_candidates = fixed["think"]["candidates"]
    pairs = [
        {
            "candidate_id": index,
            "old": left["completion_sha256"],
            "fixed": right["completion_sha256"],
            "match": left["completion_sha256"] == right["completion_sha256"],
        }
        for index, (left, right) in enumerate(zip(old_candidates, fixed_candidates))
    ]
    if len(pairs) != 4 or sum(row["match"] for row in pairs) != 4:
        raise RuntimeError("COT_SHA_MISMATCH")
    return fixed, pairs


def run_rank(rank, phase, parts_dir):
    fixed, sha_pairs = paired_probe()
    bridge, bridge_source_rows = extract_sft_bridge()
    torch.cuda.set_device(rank)
    model, tokenizer, _ = load_model(device=f"cuda:{rank}")
    candidate = fixed["think"]["candidates"][rank]
    cot_ids = tokenizer.encode(
        closed_cot(candidate["completion"]), add_special_tokens=False
    )
    prompt_ids = encode_prompt(tokenizer, fixed["think_prompt"])
    domain_ids = tokenizer.encode(VIDEO_PREFIX, add_special_tokens=False)
    bridge_ids = tokenizer.encode(bridge, add_special_tokens=False)
    if len(domain_ids) != 1:
        raise RuntimeError(f"VIDEO_PREFIX_NOT_SINGLE_TOKEN: {domain_ids}")
    bare_context = prompt_ids + cot_ids + domain_ids
    context = (
        bare_context
        if phase == "bare"
        else prompt_ids + cot_ids + bridge_ids + domain_ids
    )
    historical_ids = [gold_ids(tokenizer, sid) for sid in HISTORICAL_EXACT]
    full_gold = {parse_gold(value) for value in fixed["gold_sids"]}
    started = time.perf_counter()
    with torch.inference_mode():
        beam, beam_ids = hf_beam(
            model,
            tokenizer,
            context,
            32,
            3,
            parse_abc3,
            full_gold,
        )
        scores = score_sequences(model, tokenizer, context, historical_ids + beam_ids)
        gold_scores = scores[: len(historical_ids)]
        beam_scores = scores[len(historical_ids) :]
        cutoff = sorted(
            row["abc_conditional_logprob"] for row in beam_scores
        )[0]
        a_logits, _ = prefill(model, context)
    logit_path = parts_dir / f"rank{rank}_bare_a_logits.pt"
    comparison = None
    if phase == "bare":
        parts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(a_logits.cpu(), logit_path)
    else:
        bare_logits = torch.load(logit_path, map_location="cpu", weights_only=True)
        bridge_logits = a_logits.cpu()
        diff = (bridge_logits - bare_logits).abs()
        bare_top32 = torch.topk(bare_logits, 32).indices.tolist()
        bridge_top32 = torch.topk(bridge_logits, 32).indices.tolist()
        comparison = {
            "max_abs_logit_diff": float(diff.max()),
            "mean_abs_logit_diff": float(diff.mean()),
            "top32_a_token_overlap": len(set(bare_top32) & set(bridge_top32)),
            "bare_top32_a_token_ids": bare_top32,
            "bridge_top32_a_token_ids": bridge_top32,
        }
    gold_rows = []
    beam_set = {tuple(ids) for ids in beam_ids}
    for sid, ids, score in zip(HISTORICAL_EXACT, historical_ids, gold_scores):
        gold_rows.append(
            {
                "gold_sid": list(sid),
                "found": tuple(ids) in beam_set,
                **score,
            }
        )
    part = {
        "phase": phase,
        "rank": rank,
        "candidate_id": rank,
        "device": f"cuda:{rank}",
        "model_parent": {
            "base": BASE,
            "adapter": ADAPTER,
            "fresh_original_bata": True,
        },
        "cot_sha_pairs": sha_pairs,
        "completion_sha256": candidate["completion_sha256"],
        "production_closed_cot_sha256": completion_sha(cot_ids),
        "bridge_source": str(SFT_SOURCE),
        "bridge_source_matching_rows": bridge_source_rows,
        "bridge_text_repr": repr(bridge),
        "bridge_token_ids": bridge_ids,
        "bridge_token_count": len(bridge_ids),
        "beam": beam,
        "beam32_cutoff_logprob": cutoff,
        "historical_gold": gold_rows,
        "a_logit_comparison": comparison,
        "elapsed_sec": time.perf_counter() - started,
        "training_started": False,
        "optimizer_created": False,
    }
    parts_dir.mkdir(parents=True, exist_ok=True)
    path = parts_dir / f"rank{rank}_{phase}.json"
    path.write_text(json.dumps(part, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"BRIDGE_ABC3_RANK_DONE phase={phase} rank={rank} "
        f"summary={beam['summary']} elapsed={part['elapsed_sec']:.1f}s",
        flush=True,
    )


def load_parts(parts_dir, phase):
    return [
        json.loads((parts_dir / f"rank{rank}_{phase}.json").read_text(encoding="utf-8"))
        for rank in range(4)
    ]


def aggregate(parts):
    rows = [part["beam"]["summary"] for part in parts]
    return {
        "beam_raw_mean": sum(row["beam_raw"] for row in rows) / len(rows),
        **{
            field: sum(row[field] for row in rows)
            for field in ("exact", "ab", "a", "invalid")
        },
    }


def validate_bare(parts_dir):
    parts = load_parts(parts_dir, "bare")
    actual = aggregate(parts)
    expected = {
        "beam_raw_mean": 1.125,
        "exact": 0,
        "ab": 1,
        "a": 8,
        "invalid": 0,
    }
    passed = all(
        math.isclose(actual[key], value, abs_tol=1e-9)
        if isinstance(value, float)
        else actual[key] == value
        for key, value in expected.items()
    )
    if not passed:
        raise RuntimeError(f"BARE_ABC3_REPRODUCTION_FAILED actual={actual}")
    print(json.dumps({"BARE_ABC3_REPRODUCED": True, "summary": actual}, indent=2))


def matched_gold(parts, sid):
    return [
        next(row for row in part["historical_gold"] if tuple(row["gold_sid"]) == sid)
        for part in parts
    ]


def classify_effect(bare, bridge, historical):
    exact_delta = bridge["exact"] - bare["exact"]
    raw_delta = bridge["beam_raw_mean"] - bare["beam_raw_mean"]
    logp_deltas = [row["logprob_delta"] for item in historical for row in item["per_candidate"]]
    mean_logp_delta = sum(logp_deltas) / len(logp_deltas)
    if exact_delta > 0 and raw_delta >= 0.5:
        return "STRONG_POSITIVE"
    if exact_delta > 0 or raw_delta >= 0.25 or mean_logp_delta >= 1.0:
        return "MODERATE_POSITIVE"
    if abs(raw_delta) < 0.1 and abs(mean_logp_delta) < 0.5:
        return "NEGLIGIBLE"
    if raw_delta < 0 and mean_logp_delta <= 0:
        return "NEGATIVE"
    return "MIXED"


def merge(parts_dir, output):
    bare_parts = load_parts(parts_dir, "bare")
    bridge_parts = load_parts(parts_dir, "bridge")
    validate_bare(parts_dir)
    bare, bridge = aggregate(bare_parts), aggregate(bridge_parts)
    sha_match = sum(row["match"] for row in bare_parts[0]["cot_sha_pairs"])
    if sha_match != 4:
        raise RuntimeError("COT_SHA_MATCH is not 4/4")
    if any(
        part["bridge_text_repr"] != bare_parts[0]["bridge_text_repr"]
        or part["bridge_token_ids"] != bare_parts[0]["bridge_token_ids"]
        for part in bare_parts + bridge_parts
    ):
        raise RuntimeError("SFT_BRIDGE_DRIFT")
    historical = []
    for sid in HISTORICAL_EXACT:
        bare_rows, bridge_rows = matched_gold(bare_parts, sid), matched_gold(bridge_parts, sid)
        per_candidate = []
        for rank, (left, right) in enumerate(zip(bare_rows, bridge_rows)):
            per_candidate.append(
                {
                    "rank": rank,
                    "bare_found": left["found"],
                    "bridge_found": right["found"],
                    "bare_abc_logprob": left["abc_conditional_logprob"],
                    "bridge_abc_logprob": right["abc_conditional_logprob"],
                    "logprob_delta": (
                        right["abc_conditional_logprob"] - left["abc_conditional_logprob"]
                    ),
                    "bare_ranks": [left["a_rank"], left["b_rank"], left["c_rank"]],
                    "bridge_ranks": [right["a_rank"], right["b_rank"], right["c_rank"]],
                }
            )
        historical.append(
            {
                "sid": list(sid),
                "bare_found": any(row["found"] for row in bare_rows),
                "bridge_found": any(row["found"] for row in bridge_rows),
                "bare_abc_logprob_mean": sum(row["abc_conditional_logprob"] for row in bare_rows) / 4,
                "bridge_abc_logprob_mean": sum(row["abc_conditional_logprob"] for row in bridge_rows) / 4,
                "logprob_delta_mean": sum(row["logprob_delta"] for row in per_candidate) / 4,
                "per_candidate": per_candidate,
            }
        )
    logit_rows = [part["a_logit_comparison"] for part in bridge_parts]
    effect = classify_effect(bare, bridge, historical)
    exact_recovered = bridge["exact"] > 0
    root_conclusion = (
        "SFT bridge restores at least one strict ABC3 Exact; inspect per-stage ranks for the primary gain."
        if exact_recovered
        else "SFT bridge does not restore strict ABC3 Exact; historical old Exact remains primarily attributable to the long-continuation parse artifact."
    )
    result = {
        "type": "probe1_bare_vs_sft_bridge_abc3",
        "diagnostic_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=GRPO_ROOT, text=True
        ).strip(),
        "recommendation_group_id": GROUP_ID,
        "model_parent": bare_parts[0]["model_parent"],
        "cot_sha_match": f"{sha_match}/4",
        "sft_bridge": {
            "source": bare_parts[0]["bridge_source"],
            "matching_rows": bare_parts[0]["bridge_source_matching_rows"],
            "text_repr": bare_parts[0]["bridge_text_repr"],
            "token_ids": bare_parts[0]["bridge_token_ids"],
            "token_count": bare_parts[0]["bridge_token_count"],
        },
        "bare_abc3": bare,
        "bridge_abc3": bridge,
        "per_candidate": [
            {
                "rank": rank,
                "bare": bare_parts[rank]["beam"]["summary"],
                "bridge": bridge_parts[rank]["beam"]["summary"],
                "bare_beam32_cutoff_logprob": bare_parts[rank]["beam32_cutoff_logprob"],
                "bridge_beam32_cutoff_logprob": bridge_parts[rank]["beam32_cutoff_logprob"],
                "a_logit_comparison": bridge_parts[rank]["a_logit_comparison"],
            }
            for rank in range(4)
        ],
        "historical_gold": historical,
        "a_logit_summary": {
            "max_abs_logit_diff_max": max(row["max_abs_logit_diff"] for row in logit_rows),
            "mean_abs_logit_diff_mean": sum(row["mean_abs_logit_diff"] for row in logit_rows) / 4,
            "top32_a_token_overlap": [row["top32_a_token_overlap"] for row in logit_rows],
        },
        "bridge_effect": effect,
        "exact_recovered": exact_recovered,
        "root_conclusion": root_conclusion,
        "gpu_used": 4,
        "training_started": False,
        "production_code_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"BRIDGE_ABC3_RESULT={output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("bare", "bridge"))
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-bare", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.validate_bare:
        validate_bare(args.parts_dir)
        return
    if args.merge:
        merge(args.parts_dir, args.output)
        return
    if args.phase is None:
        parser.error("--phase is required unless --validate-bare or --merge is used")
    rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    if world != 4 or rank not in range(4):
        raise RuntimeError("Run with torchrun --nproc_per_node=4")
    run_rank(rank, args.phase, args.parts_dir)


if __name__ == "__main__":
    main()
