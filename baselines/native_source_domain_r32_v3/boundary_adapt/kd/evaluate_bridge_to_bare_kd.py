"""Strict Historical Fixed-CoT and persisted G300/G900 evaluation for KD decoders."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent; BOUNDARY = HERE.parent; DIAGNOSTICS = BOUNDARY / "diagnostics"
sys.path.insert(0, str(DIAGNOSTICS)); sys.path.insert(0, "/data/GRPO/scripts")
import controlled_generator_decoder_crossover as crossover
import fixed_cot_checkpoint_sweep as fixed

DOMAIN = fixed.DOMAIN
PARTS = Path("/data/GRPO/boundary_adapt/results/bridge_to_bare_kd_eval_parts")
RESULT = Path("/data/GRPO/boundary_adapt/results/boundary_adapt_step900_bridge_to_bare_kd_v1_20260824.json")
REPO_RESULT = BOUNDARY / "results/boundary_adapt_step900_bridge_to_bare_kd_v1_20260824.json"
ADAPTERS = {
    "step900": Path("/data/outputs/boundary_adapt/continuation_from300_to1500/checkpoint-900"),
    **{f"kd{step}": Path(f"/data/outputs/boundary_adapt/step900_bridge_to_bare_kd_v1/checkpoint-{step}") for step in (50, 100, 200, 300)},
}


def evaluate(label: str) -> None:
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 4 or label not in ADAPTERS:
        raise RuntimeError("evaluation requires four ranks and a known decoder")
    torch.cuda.set_device(rank); model, tokenizer = crossover.load_model(ADAPTERS[label], f"cuda:{rank}")
    historical = []
    items = json.loads(fixed.MANIFEST.read_text(encoding="utf-8"))["items"]
    for index, item in enumerate(items):
        if index % world != rank: continue
        prompt = crossover.encode_prompt(tokenizer, item["think_prompt"]); cot = tokenizer.encode(item["fixed_cot"], add_special_tokens=False)
        bridge = tokenizer.encode(item["bridge"], add_special_tokens=False); domain = tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)
        history = crossover.history_from_prompt(tokenizer, prompt); gold = {crossover.parse_gold(value) for value in item["gold_sids"]}
        modes = {}
        for mode, context in (("bare", prompt + cot + domain), ("bridge", prompt + cot + bridge + domain)):
            modes[mode] = {
                "beam": crossover.strict_beam(model, tokenizer, context, item["target_domain"], gold, history),
                "teacher_forced": crossover.teacher_forced_gold(model, tokenizer, context, item["gold_sids"]),
            }
        historical.append({"group_id": item["recommendation_group_id"], "candidate_id": item["candidate_id"], "target_domain": item["target_domain"], "modes": modes})
    controlled = []
    for generator in ("300", "900"):
        for index, item in enumerate(crossover.load_jsonl(crossover.RAW_DIR / crossover.artifact_name(generator, "main"))):
            if index % world != rank: continue
            prompt = list(map(int, item["prompt_token_ids"])); cot = list(map(int, item["completion_token_ids"])); domain = tokenizer.encode(DOMAIN[item["target_domain"]], add_special_tokens=False)
            history = crossover.history_from_prompt(tokenizer, prompt); gold = {crossover.parse_gold(value) for value in item["gold_sids"]}; context = prompt + cot + domain
            controlled.append({"generator": generator, "group_id": item["group_id"], "candidate_id": item["candidate_id"], "target_domain": item["target_domain"],
                               "beam": crossover.strict_beam(model, tokenizer, context, item["target_domain"], gold, history),
                               "teacher_forced": crossover.teacher_forced_gold(model, tokenizer, context, item["gold_sids"])})
    target = PARTS / label; target.mkdir(parents=True, exist_ok=True)
    (target / f"rank{rank}.json").write_text(json.dumps({"historical": historical, "controlled": controlled}, ensure_ascii=False), encoding="utf-8")


def summarize(records, mode=None):
    selected = [{"group_id": row["group_id"], **(row["modes"][mode] if mode else row)} for row in records]
    return {**crossover.aggregate_beam(selected), **crossover.aggregate_teacher(selected)}


def merge(label: str) -> dict:
    shards = [json.loads((PARTS / label / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(4)]
    historical = [row for shard in shards for row in shard["historical"]]; controlled = [row for shard in shards for row in shard["controlled"]]
    if len(historical) != 48 or len(controlled) != 96: raise RuntimeError("incomplete evaluation shards")
    bare, bridge = summarize(historical, "bare"), summarize(historical, "bridge")
    result = {"label": label, "adapter": str(ADAPTERS[label]), "historical": {"bare": bare, "bridge": bridge, "gap": bridge["beam_raw"] - bare["beam_raw"]}, "controlled": {}}
    for generator in ("300", "900"):
        result["controlled"][f"G{generator}"] = summarize([row for row in controlled if row["generator"] == generator])
    (PARTS / f"{label}_merged.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def finalize(implementation_commit: str, training_result: str) -> None:
    evaluations = {label: json.loads((PARTS / f"{label}_merged.json").read_text(encoding="utf-8")) for label in ADAPTERS}
    baseline = evaluations["step900"]; candidates = [evaluations[f"kd{step}"] for step in (50, 100, 200, 300)]
    def key(item):
        hist, control = item["historical"], item["controlled"]["G900"]
        return (hist["bare"]["beam_raw"] > baseline["historical"]["bare"]["beam_raw"], -hist["gap"],
                -hist["bare"]["history_not_gold"], control["beam_raw"], -hist["bare"]["mean_gold_abc_nll"])
    best = max(candidates, key=key)
    training = json.loads(Path(training_result).read_text(encoding="utf-8"))
    payload = {"type": "step900_bridge_to_bare_sid_family_kd_v1", "source_commit": "94ae31ff87c99ba71c9f0142d97d92b8e6f5efc3", "implementation_commit": implementation_commit,
               "teacher_adapter": str(ADAPTERS["step900"]), "student_init_adapter": str(ADAPTERS["step900"]), "lambda": .3, "temperature": 1.0, "lr": 1e-7, "steps": 300,
               "teacher_usefulness_preflight": training["preflight"]["teacher_usefulness"], "lambda0_parity": {"forward": True, "gradient": True, "multi_positive": True},
               "training": training, "fixed_cot_sweep": {key: value["historical"] for key, value in evaluations.items()},
               "controlled_generator_sweep": {key: value["controlled"] for key, value in evaluations.items()}, "best_checkpoint": best["adapter"]}
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPO_RESULT.parent.mkdir(parents=True, exist_ok=True); REPO_RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--evaluate", choices=ADAPTERS); parser.add_argument("--merge", choices=ADAPTERS)
    parser.add_argument("--finalize", action="store_true"); parser.add_argument("--implementation-commit"); parser.add_argument("--training-result")
    args = parser.parse_args()
    if sum((args.evaluate is not None, args.merge is not None, args.finalize)) != 1: raise SystemExit("choose one operation")
    if args.evaluate: evaluate(args.evaluate)
    elif args.merge: print(json.dumps(merge(args.merge), ensure_ascii=False, indent=2))
    else:
        if not args.implementation_commit or not args.training_result: raise SystemExit("finalize needs commit and training result")
        finalize(args.implementation_commit, args.training_result)
