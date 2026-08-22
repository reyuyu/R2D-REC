#!/usr/bin/env python3
"""Inference-only live fixed-probe sidecar for MC_USER formal training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from build_user_probe_light import select_light_probe
from user_action_reward import score_action
from user_chain_reward import score_chain


BASE_MODEL = Path("/data/models/onereason-8b-pretrain-competition")
BETA_ADAPTER = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
PROBE_DATA = Path("/data/GRPO_USER/data/gr_user_v1/probe_v1.jsonl")
PROBE_SHA256 = "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61"
CHECKPOINT_STEPS = (0, 128, 256, 384, 512)
PROBE_SEED = 20260820
EVAL_CANDIDATE_COUNT = 4
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 512
MEMORY_THRESHOLD_MIB = 1024


class FormalProbeError(RuntimeError):
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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def output_paths(formal_run_dir: Path) -> tuple[Path, Path]:
    output_dir = formal_run_dir / "evaluations" / "user_light_probe"
    return output_dir / "results.json", output_dir / "status.json"


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


def load_probe(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha = file_sha256(path)
    if actual_sha != PROBE_SHA256:
        raise FormalProbeError("frozen probe_v1 SHA mismatch")
    rows, audit = select_light_probe(read_jsonl(path))
    counts = {route: sum(row["route"] == route for row in rows) for route in ("action", "chain")}
    if len(rows) != 6 or counts != {"action": 3, "chain": 3}:
        raise FormalProbeError("formal light probe must be 3 Action + 3 Chain")
    return rows, {"sha256": actual_sha, "selection_audit": audit}


def load_formal_manifest(formal_run_dir: Path) -> dict[str, Any]:
    path = formal_run_dir / "manifest.json"
    if not path.is_file():
        raise FormalProbeError("formal manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prompts = manifest.get("prompts", [])
    ids = [row.get("sample_id") for row in prompts]
    routes = [str(row.get("route", "")).lower() for row in prompts]
    if len(ids) != 512 or None in ids or len(set(ids)) != 512:
        raise FormalProbeError("formal manifest must contain 512 unique samples")
    if routes != [route for _ in range(256) for route in ("action", "chain")]:
        raise FormalProbeError("formal manifest route schedule is not strict alternating")
    return manifest


def validate_probe_disjoint(
    probe_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    probe_ids = {str(row["sample_id"]) for row in probe_rows}
    formal_ids = {str(row["sample_id"]) for row in manifest["prompts"]}
    overlap = sorted(probe_ids.intersection(formal_ids))
    if overlap:
        raise FormalProbeError(f"formal probe overlaps training manifest: {overlap}")


def gpu_preflight(gpu_id: int, memory_threshold_mib: int) -> dict[str, Any]:
    if gpu_id != 1:
        raise FormalProbeError("formal probe sidecar is pinned to physical GPU 1")
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip().splitlines()
    states = {}
    for row in rows:
        index, uuid, used, free, utilization = [part.strip() for part in row.split(",")]
        states[int(index)] = {
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
        }
    if gpu_id not in states:
        raise FormalProbeError(f"GPU {gpu_id} does not exist")
    process_text = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    processes = []
    for row in process_text.splitlines() if process_text else []:
        uuid, pid, name, used = [part.strip() for part in row.split(",", 3)]
        processes.append(
            {"uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(used)}
        )
    state = states[gpu_id]
    selected = [process for process in processes if process["uuid"] == state["uuid"]]
    if selected or state["memory_used_mib"] >= memory_threshold_mib:
        raise FormalProbeError("GPU 1 is not idle")
    return {**state, "compute_processes": selected}


def validate_adapter_only(path: Path, step: int) -> None:
    if not path.is_dir():
        raise FormalProbeError(f"adapter directory missing: {path}")
    required = {"adapter_config.json", "adapter_model.safetensors"}
    names = {item.name for item in path.iterdir()}
    if not required.issubset(names):
        raise FormalProbeError(f"adapter checkpoint incomplete: {path}")
    if any(
        name == "model.safetensors"
        or name.startswith("model-")
        or name.startswith("pytorch_model")
        for name in names
    ):
        raise FormalProbeError(f"base weights found in adapter checkpoint: {path}")
    if step:
        state_path = path / "formal_state.json"
        if not state_path.is_file():
            raise FormalProbeError(f"formal_state.json missing: {path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state.get("prompt_step", -1)) != step:
            raise FormalProbeError(f"formal checkpoint step mismatch: {path}")


def checkpoint_spec(formal_run_dir: Path, beta_adapter: Path, step: int) -> dict[str, Any]:
    if step == 0:
        return {"step": 0, "name": "BETA", "path": beta_adapter}
    name = f"prompt-step-{step:04d}"
    manifest_path = formal_run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_root = manifest.get("checkpoint_root")
    else:
        checkpoint_root = None
    root = Path(checkpoint_root) / formal_run_dir.name if checkpoint_root else formal_run_dir
    return {"step": step, "name": name, "path": root / "checkpoints" / name}


def checkpoint_complete(spec: Mapping[str, Any]) -> bool:
    path = Path(spec["path"])
    step = int(spec["step"])
    if not path.is_dir():
        return False
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if step:
        required.add("formal_state.json")
    if not required.issubset({item.name for item in path.iterdir()}):
        return False
    validate_adapter_only(path, step)
    return True


def score_candidate(
    completion: str,
    token_count: int,
    sample: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    if sample["route"] == "action":
        score = score_action(completion, dict(sample), tokenizer)
        return {
            "completion": completion,
            "completion_length": token_count,
            "format_valid": bool(score.format_valid),
            "reward": float(score.reward),
            "f1": float(score.f1),
            "precision": float(score.precision),
            "recall": float(score.recall),
            "exact_match": bool(score.exact_set_match),
            "gold_sids": list(score.gold_sids),
            "pred_sids": list(score.pred_sids_unique),
        }
    score = score_chain(completion, dict(sample), tokenizer)
    return {
        "completion": completion,
        "completion_length": token_count,
        "format_valid": bool(score.format_valid),
        "reward": float(score.total_reward),
        "total_reward": float(score.total_reward),
        "action_alignment": float(score.action_f1),
        "logic_alignment": float(score.logic_f1),
        "predicted_event_count": len(score.predicted_events),
    }


def summarize_checkpoint(step: int, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action = [candidate for sample in samples if sample["route"] == "action" for candidate in sample["candidates"]]
    chain = [candidate for sample in samples if sample["route"] == "chain" for candidate in sample["candidates"]]
    if len(action) != 12 or len(chain) != 12:
        raise FormalProbeError("probe checkpoint must contain 3+3 samples with four candidates")
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


def incremental_result(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    probe_sha256: str,
    sample_ids: Sequence[str],
    sample_seeds: Mapping[str, int],
) -> dict[str, Any]:
    ordered = [dict(checkpoint) for checkpoint in sorted(checkpoints, key=lambda row: int(row["step"]))]
    evaluated_steps = [int(row["step"]) for row in ordered]
    if evaluated_steps and evaluated_steps[0] != 0:
        raise FormalProbeError("BETA must be evaluated first")
    baseline_means = {}
    if ordered:
        baseline_means = {
            str(sample["sample_id"]): statistics.fmean(float(candidate["reward"]) for candidate in sample["candidates"])
            for sample in ordered[0]["samples"]
        }
    deltas = []
    for checkpoint in ordered:
        for sample in checkpoint["samples"]:
            mean_reward = statistics.fmean(float(candidate["reward"]) for candidate in sample["candidates"])
            sample["mean_reward"] = mean_reward
            sample["relative_to_beta_delta"] = mean_reward - baseline_means.get(str(sample["sample_id"]), mean_reward)
        if int(checkpoint["step"]) == 0:
            continue
        base = ordered[0]["summary"]
        summary = checkpoint["summary"]
        deltas.append(
            {
                "step": int(checkpoint["step"]),
                "delta_action_f1": summary["action"]["f1"] - base["action"]["f1"],
                "delta_chain_total": summary["chain"]["total_reward"] - base["chain"]["total_reward"],
                "delta_chain_action": summary["chain"]["action_alignment"] - base["chain"]["action_alignment"],
                "delta_chain_logic": summary["chain"]["logic_alignment"] - base["chain"]["logic_alignment"],
                "delta_overall_user_proxy": summary["overall_user_proxy"] - base["overall_user_proxy"],
            }
        )
    waiting = [step for step in CHECKPOINT_STEPS if step not in evaluated_steps]
    return {
        "status": "PASS" if not waiting else "WAITING_FOR_CHECKPOINTS",
        "mode": "FIXED_SAMPLES_INFERENCE_ONLY",
        "score_semantics": "LOCAL_TREND_PROXY_NOT_OFFICIAL_SCORE",
        "probe_sha256": probe_sha256,
        "sample_ids": list(sample_ids),
        "sample_seeds": dict(sample_seeds),
        "generation_config": generation_config(),
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "evaluated_steps": evaluated_steps,
        "waiting_steps": waiting,
        "checkpoints": ordered,
        "paired_deltas": deltas,
    }


def publish_incremental(
    formal_run_dir: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    result = incremental_result(
        checkpoints,
        probe_sha256=str(preflight["probe_sha256"]),
        sample_ids=preflight["sample_ids"],
        sample_seeds=preflight["sample_seeds"],
    )
    results_path, status_path = output_paths(formal_run_dir)
    atomic_json(results_path, result)
    atomic_json(
        status_path,
        {
            "status": result["status"],
            "evaluated_steps": result["evaluated_steps"],
            "waiting_steps": result["waiting_steps"],
        },
    )
    return result


def watch_schedule(
    specs: Sequence[Mapping[str, Any]],
    *,
    is_ready: Callable[[Mapping[str, Any]], bool],
    evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    publish: Callable[[Sequence[Mapping[str, Any]]], Any],
    sleep: Callable[[float], Any],
    poll_seconds: float,
) -> list[dict[str, Any]]:
    outputs = []
    evaluated = set()
    for spec in specs:
        step = int(spec["step"])
        while not is_ready(spec):
            sleep(poll_seconds)
        if step in evaluated:
            continue
        outputs.append(dict(evaluate(spec)))
        evaluated.add(step)
        publish(outputs)
    return outputs


def render_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(tokenizer.encode(rendered, add_special_tokens=False)) != int(row["prompt_token_count"]):
        raise FormalProbeError("formal probe prompt renderer drift")
    return rendered


def stop_token_ids(tokenizer: Any) -> set[int]:
    output = {int(tokenizer.eos_token_id)}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        output.add(im_end)
    return output


def trim_at_stop(token_ids: Sequence[int], stop_ids: set[int]) -> list[int]:
    for index, token_id in enumerate(token_ids):
        if int(token_id) in stop_ids:
            return [int(value) for value in token_ids[:index]]
    return [int(value) for value in token_ids]


@torch.inference_mode()
def evaluate_adapter(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seeds: Mapping[str, int],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    samples = []
    for row in rows:
        seed = int(seeds[str(row["sample_id"])])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rendered = render_prompt(tokenizer, row)
        encoded = tokenizer([rendered], add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt")
        prompt_width = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        stops = stop_token_ids(tokenizer)
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=EVAL_CANDIDATE_COUNT,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=sorted(stops),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        generated = output[:, prompt_width:].cpu().tolist()
        candidates = []
        for candidate_id, raw_ids in enumerate(generated):
            token_ids = trim_at_stop(raw_ids, stops)
            completion = tokenizer.decode(token_ids, skip_special_tokens=False)
            candidate = score_candidate(completion, len(token_ids), row, tokenizer)
            candidate["candidate_id"] = candidate_id
            candidates.append(candidate)
        samples.append(
            {
                "sample_id": row["sample_id"],
                "route": row["route"],
                "seed": seed,
                "prompt": row["prompt"],
                "gold_sids": list(row.get("gold_sids", [])),
                "gold_events": list(row.get("gold_events", [])),
                "candidates": candidates,
            }
        )
    return samples


def run_preflight(
    args: argparse.Namespace,
    *,
    gpu_checker: Callable[[int, int], Mapping[str, Any]] = gpu_preflight,
) -> dict[str, Any]:
    if not args.base_model.is_dir():
        raise FormalProbeError("base model directory is missing")
    validate_adapter_only(args.beta_adapter, 0)
    rows, probe_contract = load_probe(args.probe)
    manifest = load_formal_manifest(args.formal_run_dir)
    validate_probe_disjoint(rows, manifest)
    gpu = dict(gpu_checker(args.gpu_id, args.memory_threshold_mib))
    specs = [checkpoint_spec(args.formal_run_dir, args.beta_adapter, step) for step in CHECKPOINT_STEPS]
    return {
        "status": "READY_TO_EXECUTE",
        "gpu": gpu,
        "formal_run_dir": str(args.formal_run_dir),
        "probe_rows": rows,
        "probe_sha256": probe_contract["sha256"],
        "selection_audit": probe_contract["selection_audit"],
        "sample_ids": [row["sample_id"] for row in rows],
        "sample_seeds": sample_seed_map(rows),
        "checkpoints": specs,
        "generation_config": generation_config(),
        "formal_manifest_sha256": file_sha256(args.formal_run_dir / "manifest.json"),
    }


def execute_sidecar(preflight: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or args.gpu_id != 1:
        raise FormalProbeError("sidecar requires CUDA_VISIBLE_DEVICES=1 and --gpu-id 1")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )

    def evaluate(spec: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal base
        validate_adapter_only(Path(spec["path"]), int(spec["step"]))
        model = PeftModel.from_pretrained(base, str(spec["path"]), is_trainable=False, local_files_only=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise FormalProbeError("probe model is not fully frozen")
        started = time.perf_counter()
        samples = evaluate_adapter(model, tokenizer, preflight["probe_rows"], preflight["sample_seeds"], device)
        checkpoint = {
            "step": int(spec["step"]),
            "name": spec["name"],
            "path": str(spec["path"]),
            "summary": summarize_checkpoint(int(spec["step"]), samples),
            "samples": samples,
            "wall_seconds": time.perf_counter() - started,
        }
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        return checkpoint

    outputs = watch_schedule(
        preflight["checkpoints"],
        is_ready=checkpoint_complete,
        evaluate=evaluate,
        publish=lambda rows: publish_incremental(args.formal_run_dir, rows, preflight),
        sleep=time.sleep,
        poll_seconds=args.poll_seconds,
    )
    return publish_incremental(args.formal_run_dir, outputs, preflight)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--beta-adapter", type=Path, default=BETA_ADAPTER)
    parser.add_argument("--probe", type=Path, default=PROBE_DATA)
    parser.add_argument("--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": preflight["status"],
        "gpu": preflight["gpu"],
        "formal_run_dir": preflight["formal_run_dir"],
        "probe_sha256": preflight["probe_sha256"],
        "sample_ids": preflight["sample_ids"],
        "counts": {"action": 3, "chain": 3},
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "generation_config": preflight["generation_config"],
        "execute_required": True,
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    preflight_fn: Callable[..., Mapping[str, Any]] = run_preflight,
    execute_fn: Callable[..., Mapping[str, Any]] = execute_sidecar,
) -> Mapping[str, Any]:
    args = build_parser().parse_args(argv)
    preflight = preflight_fn(args)
    if not args.execute:
        output = public_preflight(preflight)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        print("READY_TO_EXECUTE")
        return output
    output = execute_fn(preflight, args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main() -> int:
    try:
        output = run_cli()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    return 0 if output["status"] in {"PASS", "READY_TO_EXECUTE", "WAITING_FOR_CHECKPOINTS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
