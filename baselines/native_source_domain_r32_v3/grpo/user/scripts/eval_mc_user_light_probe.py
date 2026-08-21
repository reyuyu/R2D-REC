#!/usr/bin/env python3
"""Gated, paired checkpoint evaluator for the six-sample User light probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from build_user_probe_light import select_light_probe
from run_mc_user_real_smoke import gpu_preflight
from user_action_reward import score_action
from user_chain_reward import score_chain


BASE_MODEL = "/data/models/onereason-8b-pretrain-competition"
BETA_ADAPTER = (
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
PROBE_DATA = "/data/GRPO_USER/data/gr_user_v1/probe_v1.jsonl"
PROBE_SHA256 = "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61"
PROBE_SEED = 20260820
EVAL_CANDIDATE_COUNT = 4
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 512
CHECKPOINT_STEPS = (0, 8, 16, 32)
MEMORY_THRESHOLD_MIB = 1024


class UserLightProbeError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_light_probe(
    path: Path, expected_sha: str = PROBE_SHA256
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise UserLightProbeError("frozen probe_v1 SHA mismatch")
    selected, audit = select_light_probe(read_jsonl(path))
    action_count = sum(row.get("route") == "action" for row in selected)
    chain_count = sum(row.get("route") == "chain" for row in selected)
    if (len(selected), action_count, chain_count) != (6, 3, 3):
        raise UserLightProbeError("light probe selection must be 3 Action + 3 Chain")
    if len({row["sample_id"] for row in selected}) != 6:
        raise UserLightProbeError("light probe sample IDs are not unique")
    return selected, {"sha256": actual_sha, "selection_audit": audit}


def checkpoint_specs(pilot_run_dir: Path, beta_adapter: Path) -> list[dict[str, Any]]:
    return [
        {"step": 0, "name": "BETA", "path": beta_adapter},
        *[
            {
                "step": step,
                "name": f"prompt-step-{step:04d}",
                "path": pilot_run_dir / "checkpoints" / f"prompt-step-{step:04d}",
            }
            for step in CHECKPOINT_STEPS[1:]
        ],
    ]


def validate_adapter_only(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)
    if not (path / "adapter_config.json").is_file():
        raise UserLightProbeError(f"adapter_config.json missing from {path}")
    if not any(
        (path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise UserLightProbeError(f"adapter weights missing from {path}")
    forbidden = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    for item in path.iterdir():
        if item.name in forbidden or item.name.startswith(("model-", "pytorch_model-")):
            raise UserLightProbeError(f"base-model weight found in adapter checkpoint: {item.name}")


def load_pilot_manifest(pilot_run_dir: Path) -> dict[str, Any]:
    path = pilot_run_dir / "manifest.json"
    if not path.is_file():
        raise UserLightProbeError("pilot manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prompts = manifest.get("prompts", [])
    ids = [item.get("sample_id") for item in prompts]
    if len(ids) != 32 or None in ids or len(set(ids)) != 32:
        raise UserLightProbeError("pilot manifest must contain 32 unique sample IDs")
    return manifest


def validate_probe_disjoint(
    probe_rows: Sequence[Mapping[str, Any]], pilot_manifest: Mapping[str, Any]
) -> None:
    probe_ids = {str(row["sample_id"]) for row in probe_rows}
    pilot_ids = {str(item["sample_id"]) for item in pilot_manifest["prompts"]}
    overlap = probe_ids.intersection(pilot_ids)
    if overlap:
        raise UserLightProbeError(f"light probe overlaps pilot32 manifest: {sorted(overlap)}")


def sample_seed_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(row["sample_id"]): PROBE_SEED + index
        for index, row in enumerate(rows)
    }


def generation_config() -> dict[str, Any]:
    return {
        "candidate_count": EVAL_CANDIDATE_COUNT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": PROBE_SEED,
    }


def output_paths(pilot_run_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = pilot_run_dir / "evaluations" / "user_light_probe"
    return output_dir, output_dir / "results.json", output_dir / "status.json"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_preflight(
    args: argparse.Namespace,
    *,
    gpu_checker: Callable[..., Mapping[str, Any]] = gpu_preflight,
) -> dict[str, Any]:
    if not args.base_model.is_dir():
        raise UserLightProbeError("base model directory is missing")
    validate_adapter_only(args.beta_adapter)
    probe_rows, probe_contract = load_light_probe(args.probe)
    pilot_manifest = load_pilot_manifest(args.pilot_run_dir)
    validate_probe_disjoint(probe_rows, pilot_manifest)
    specs = checkpoint_specs(args.pilot_run_dir, args.beta_adapter)
    missing = []
    for spec in specs[1:]:
        if not spec["path"].is_dir():
            missing.append(spec["step"])
        else:
            validate_adapter_only(spec["path"])
    gpu = gpu_checker(args.gpu_id, args.memory_threshold_mib)
    status = "WAITING_FOR_PILOT_CHECKPOINTS" if missing else "READY_TO_EXECUTE"
    preflight = {
        "status": status,
        "gpu": gpu,
        "probe_rows": probe_rows,
        "probe_sha256": probe_contract["sha256"],
        "selection_audit": probe_contract["selection_audit"],
        "sample_ids": [row["sample_id"] for row in probe_rows],
        "sample_seeds": sample_seed_map(probe_rows),
        "checkpoints": specs,
        "missing_pilot_checkpoint_steps": missing,
        "generation_config": generation_config(),
    }
    _, _, status_path = output_paths(args.pilot_run_dir)
    atomic_json(status_path, {
        "status": status,
        "missing_pilot_checkpoint_steps": missing,
        "probe_sha256": probe_contract["sha256"],
    })
    return preflight


def score_candidate(
    completion: str,
    completion_token_count: int,
    sample: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    if sample["route"] == "action":
        score = score_action(completion, dict(sample), tokenizer)
        return {
            "completion": completion,
            "completion_token_count": completion_token_count,
            "format_valid": bool(score.format_valid),
            "reward": float(score.reward),
            "f1": float(score.f1),
            "precision": float(score.precision),
            "recall": float(score.recall),
            "exact_match": bool(score.exact_set_match),
            "pred_sid_count": len(score.pred_sids_unique),
        }
    score = score_chain(completion, dict(sample), tokenizer)
    return {
        "completion": completion,
        "completion_token_count": completion_token_count,
        "format_valid": bool(score.format_valid),
        "reward": float(score.total_reward),
        "total_reward": float(score.total_reward),
        "action_alignment": float(score.action_f1),
        "logic_alignment": float(score.logic_f1),
        "predicted_event_count": len(score.predicted_events),
    }


def summarize_checkpoint(
    step: int, samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    action = [candidate for sample in samples if sample["route"] == "action" for candidate in sample["candidates"]]
    chain = [candidate for sample in samples if sample["route"] == "chain" for candidate in sample["candidates"]]
    if len(action) != 3 * EVAL_CANDIDATE_COUNT or len(chain) != 3 * EVAL_CANDIDATE_COUNT:
        raise UserLightProbeError("checkpoint result does not preserve 3+3 samples with four candidates")
    action_summary = {
        "f1": statistics.fmean(float(item["f1"]) for item in action),
        "precision": statistics.fmean(float(item["precision"]) for item in action),
        "recall": statistics.fmean(float(item["recall"]) for item in action),
        "exact_match_rate": statistics.fmean(float(item["exact_match"]) for item in action),
    }
    chain_summary = {
        "total_reward": statistics.fmean(float(item["total_reward"]) for item in chain),
        "action_alignment": statistics.fmean(float(item["action_alignment"]) for item in chain),
        "logic_alignment": statistics.fmean(float(item["logic_alignment"]) for item in chain),
    }
    return {
        "step": step,
        "action": action_summary,
        "chain": chain_summary,
        "overall_user_proxy": action_summary["f1"] + chain_summary["total_reward"],
    }


def paired_deltas(
    checkpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if [int(item["step"]) for item in checkpoints] != list(CHECKPOINT_STEPS):
        raise UserLightProbeError("paired checkpoints must be ordered 0/8/16/32")
    baseline = checkpoints[0]
    baseline_samples = {
        str(item["sample_id"]): statistics.fmean(
            float(candidate["reward"]) for candidate in item["candidates"]
        )
        for item in baseline["samples"]
    }
    output = []
    for checkpoint in checkpoints[1:]:
        summary = checkpoint["summary"]
        base_summary = baseline["summary"]
        sample_results = []
        for sample in checkpoint["samples"]:
            sample_id = str(sample["sample_id"])
            checkpoint_mean = statistics.fmean(
                float(candidate["reward"]) for candidate in sample["candidates"]
            )
            baseline_mean = baseline_samples[sample_id]
            sample_results.append({
                "sample_id": sample_id,
                "route": sample["route"],
                "baseline_mean_reward": baseline_mean,
                "checkpoint_mean_reward": checkpoint_mean,
                "paired_delta": checkpoint_mean - baseline_mean,
            })
        output.append({
            "step": int(checkpoint["step"]),
            "delta_action_f1": summary["action"]["f1"] - base_summary["action"]["f1"],
            "delta_chain_total": summary["chain"]["total_reward"] - base_summary["chain"]["total_reward"],
            "delta_chain_action": summary["chain"]["action_alignment"] - base_summary["chain"]["action_alignment"],
            "delta_chain_logic": summary["chain"]["logic_alignment"] - base_summary["chain"]["logic_alignment"],
            "delta_overall_user_proxy": summary["overall_user_proxy"] - base_summary["overall_user_proxy"],
            "samples": sample_results,
        })
    return output


def evaluate_adapter(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seeds: Mapping[str, int],
    device: torch.device,
    generate_fn: Callable[..., Sequence[Sequence[int]]],
) -> list[dict[str, Any]]:
    model.eval()
    samples = []
    for sample in rows:
        seed = int(seeds[str(sample["sample_id"])])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        completion_ids = list(generate_fn(model, tokenizer, [sample], device))
        if len(completion_ids) != EVAL_CANDIDATE_COUNT:
            raise UserLightProbeError("User light probe requires exactly four candidates")
        candidates = []
        for candidate_id, ids in enumerate(completion_ids):
            token_ids = [int(token_id) for token_id in ids]
            completion = tokenizer.decode(token_ids, skip_special_tokens=False)
            candidate = score_candidate(completion, len(token_ids), sample, tokenizer)
            candidate["candidate_id"] = candidate_id
            candidates.append(candidate)
        samples.append({
            "sample_id": sample["sample_id"],
            "route": sample["route"],
            "seed": seed,
            "candidates": candidates,
        })
    return samples


def evaluate_checkpoint_sequence(
    base: Any,
    specs: Sequence[Mapping[str, Any]],
    load_adapter_fn: Callable[[Any, Path], Any],
    evaluate_fn: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
    empty_cache_fn: Callable[[], Any],
) -> tuple[Any, list[dict[str, Any]]]:
    outputs = []
    for spec in specs:
        model = load_adapter_fn(base, Path(spec["path"]))
        output = dict(evaluate_fn(model, spec))
        outputs.append(output)
        base = model.unload()
        del model
        empty_cache_fn()
    return base, outputs


def execute_probe(
    preflight: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if preflight["status"] != "READY_TO_EXECUTE":
        return {"status": preflight["status"]}
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        raise UserLightProbeError("CUDA_VISIBLE_DEVICES must equal --gpu-id")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_user_grpo_smoke import generate_route, render_prompt

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    for row in preflight["probe_rows"]:
        render_prompt(tokenizer, row)
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    started = time.perf_counter()
    _, results_path, status_path = output_paths(args.pilot_run_dir)
    atomic_json(status_path, {"status": "RUNNING", "completed_checkpoints": 0})

    def load_adapter(current_base: Any, path: Path) -> Any:
        model = PeftModel.from_pretrained(
            current_base, str(path), is_trainable=False, local_files_only=True
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise UserLightProbeError("probe model is not fully frozen")
        return model

    def evaluate(model: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
        checkpoint_started = time.perf_counter()
        samples = evaluate_adapter(
            model,
            tokenizer,
            preflight["probe_rows"],
            preflight["sample_seeds"],
            device,
            generate_route,
        )
        summary = summarize_checkpoint(int(spec["step"]), samples)
        result = {
            "step": int(spec["step"]),
            "name": spec["name"],
            "path": str(spec["path"]),
            "summary": summary,
            "samples": samples,
            "wall_seconds": time.perf_counter() - checkpoint_started,
        }
        atomic_json(status_path, {
            "status": "RUNNING",
            "completed_checkpoints": CHECKPOINT_STEPS.index(int(spec["step"])) + 1,
            "current_step": int(spec["step"]),
        })
        return result

    _base, checkpoints = evaluate_checkpoint_sequence(
        base,
        preflight["checkpoints"],
        load_adapter,
        evaluate,
        torch.cuda.empty_cache,
    )
    output = {
        "status": "PASS",
        "probe_sha256": preflight["probe_sha256"],
        "sample_ids": preflight["sample_ids"],
        "sample_seeds": preflight["sample_seeds"],
        "generation_config": preflight["generation_config"],
        "checkpoints": checkpoints,
        "paired_deltas": paired_deltas(checkpoints),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(results_path, output)
    atomic_json(status_path, {
        "status": "PASS",
        "completed_checkpoints": len(checkpoints),
        "wall_seconds": output["wall_seconds"],
    })
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-model", type=Path, default=Path(BASE_MODEL))
    parser.add_argument("--beta-adapter", type=Path, default=Path(BETA_ADAPTER))
    parser.add_argument("--probe", type=Path, default=Path(PROBE_DATA))
    parser.add_argument("--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB)
    return parser


def public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": preflight["status"],
        "gpu": preflight["gpu"],
        "probe_sha256": preflight["probe_sha256"],
        "sample_ids": preflight["sample_ids"],
        "counts": {"action": 3, "chain": 3},
        "checkpoint_steps": [int(spec["step"]) for spec in preflight["checkpoints"]],
        "missing_pilot_checkpoint_steps": preflight["missing_pilot_checkpoint_steps"],
        "generation_config": preflight["generation_config"],
        "execute_required": True,
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    preflight_fn: Callable[..., Mapping[str, Any]] = run_preflight,
    execute_fn: Callable[..., Mapping[str, Any]] = execute_probe,
) -> Mapping[str, Any]:
    args = build_parser().parse_args(argv)
    preflight = preflight_fn(args)
    if not args.execute or preflight["status"] != "READY_TO_EXECUTE":
        output = public_preflight(preflight)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if output["status"] == "READY_TO_EXECUTE":
            print("READY_TO_EXECUTE")
        return output
    output = execute_fn(preflight, args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main() -> int:
    try:
        output = run_cli()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False))
        return 1
    return 0 if output["status"] in {
        "PASS", "READY_TO_EXECUTE", "WAITING_FOR_PILOT_CHECKPOINTS"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
