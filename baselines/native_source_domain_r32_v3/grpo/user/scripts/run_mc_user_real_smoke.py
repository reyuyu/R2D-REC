#!/usr/bin/env python3
"""Fixed-completion real-model correctness smoke for MC_USER_v1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from user_mc_batch import make_mc_policy_batch
from user_mc_rollout import prepare_mc_scored_rollout
from user_mc_train_step import mc_optimizer_step
from user_common import SID_RE


BASE_MODEL = "/data/models/onereason-8b-pretrain-competition"
ADAPTER = (
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
TRAIN_DATA = "/data/GRPO_USER/data/gr_user_v1/train_3000.jsonl"
RESULT_OUTPUT = "/data/GRPO_USER/results/mc_user_v1_real_smoke.json"
TRAIN_SHA256 = "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801"
SEED = 20260822
MEMORY_THRESHOLD_MIB = 1024
SMOKE_CONFIG = {
    "K": 2,
    "temperature": 0.9,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "forward_batch_size": 1,
    "gradient_checkpointing": True,
    "gradient_checkpointing_use_reentrant": False,
    "dropout": 0.0,
}
UNRELATED_EVENT = {
    "date": "2099-12-31",
    "action": "[搜索] mc_user_v1_deterministic_unrelated_event",
    "logic": "无关控制事件：与已有行为链不存在需求或因果联系。",
}


class MCRealSmokeError(RuntimeError):
    pass


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_fixed_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    action = next(
        (
            row
            for row in ordered
            if row.get("route") == "action"
            and len(row.get("gold_sids", [])) >= 2
            and all(SID_RE.fullmatch(sid) for sid in row["gold_sids"])
        ),
        None,
    )
    chain = next(
        (
            row
            for row in ordered
            if row.get("route") == "chain" and row.get("gold_events")
        ),
        None,
    )
    if action is None or chain is None:
        raise MCRealSmokeError("train data lacks eligible Action or Chain smoke row")
    sid_vocabulary = sorted(
        {
            sid
            for row in rows
            for sid in list(row.get("history_sids", []))
            + list(row.get("gold_sids", []))
            if SID_RE.fullmatch(sid)
        }
    )
    action_gold = set(action["gold_sids"])
    false_positive = next(
        (sid for sid in sid_vocabulary if sid not in action_gold), None
    )
    if false_positive is None:
        raise MCRealSmokeError("no deterministic non-Gold SID is available")
    return {"action": action, "chain": chain, "action_false_positive": false_positive}


def fixed_completion_texts(selection: Mapping[str, Any]) -> dict[str, list[str]]:
    action = selection["action"]
    chain = selection["chain"]
    action_gold = list(action["gold_sids"])
    action_completions = [
        str(action["raw_gold_output"]),
        json.dumps(
            [action_gold[0], selection["action_false_positive"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    ]
    chain_gold = json.loads(chain["raw_gold_output"])
    chain_variant = copy.deepcopy(chain_gold)
    chain_variant["logic_chain"]["events"].append(copy.deepcopy(UNRELATED_EVENT))
    chain_completions = [
        str(chain["raw_gold_output"]),
        json.dumps(chain_variant, ensure_ascii=False, separators=(",", ":")),
    ]
    return {"action": action_completions, "chain": chain_completions}


def summarize_units(units_per_candidate: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    units = [unit for candidate in units_per_candidate for unit in candidate]
    return {
        "active_unit_count": sum(float(unit["delta"]) != 0.0 for unit in units),
        "positive_unit_count": sum(float(unit["delta"]) > 0.0 for unit in units),
        "negative_unit_count": sum(float(unit["delta"]) < 0.0 for unit in units),
        "zero_unit_count": sum(float(unit["delta"]) == 0.0 for unit in units),
        "positive_credit_mass": sum(
            float(unit["delta"]) for unit in units if float(unit["delta"]) > 0.0
        ),
        "negative_credit_mass": sum(
            abs(float(unit["delta"])) for unit in units if float(unit["delta"]) < 0.0
        ),
    }


def validate_fixed_rollout(route: str, rollout: Mapping[str, Any]) -> dict[str, Any]:
    if rollout["route"] != route or rollout["K"] != 2:
        raise MCRealSmokeError(f"{route} rollout route/K contract failed")
    units = rollout["credit_units_per_candidate"]
    if len(units) != 2:
        raise MCRealSmokeError(f"{route} rollout does not contain K=2 candidates")
    first_positive = sum(float(unit["delta"]) > 0.0 for unit in units[0])
    second_positive = sum(float(unit["delta"]) > 0.0 for unit in units[1])
    second_negative = sum(float(unit["delta"]) < 0.0 for unit in units[1])
    if route == "action" and not (
        first_positive > 0 and second_positive > 0 and second_negative > 0
    ):
        raise MCRealSmokeError("Action fixed K2 marginal-credit contract failed")
    if route == "chain" and not (first_positive > 0 and second_negative > 0):
        raise MCRealSmokeError("Chain fixed K2 marginal-credit contract failed")
    return summarize_units(units)


def prepare_fixed_data(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selection = select_fixed_rows(rows)
    completion_texts = fixed_completion_texts(selection)
    rollouts = {}
    batches = {}
    summaries = {}
    for route in ("action", "chain"):
        completion_ids = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in completion_texts[route]
        ]
        rollouts[route] = prepare_mc_scored_rollout(
            [selection[route]], completion_ids, tokenizer
        )
        summaries[route] = validate_fixed_rollout(route, rollouts[route])
        batches[route] = make_mc_policy_batch(tokenizer, rollouts[route], device="cpu")
    return {
        "selection": selection,
        "completion_texts": completion_texts,
        "rollouts": rollouts,
        "batches": batches,
        "summaries": summaries,
    }


def parse_gpu_state(
    gpu_text: str,
    compute_text: str,
    gpu_id: int,
    memory_threshold_mib: int = MEMORY_THRESHOLD_MIB,
) -> dict[str, Any]:
    states = {}
    for row in gpu_text.strip().splitlines() if gpu_text.strip() else []:
        index, uuid, used, free, utilization = [item.strip() for item in row.split(",")]
        states[int(index)] = {
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
        }
    if gpu_id not in states:
        raise MCRealSmokeError(f"GPU {gpu_id} does not exist")
    processes = []
    for row in compute_text.strip().splitlines() if compute_text.strip() else []:
        uuid, pid, process_name, used = [item.strip() for item in row.split(",", 3)]
        processes.append(
            {
                "uuid": uuid,
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": int(used),
            }
        )
    state = states[gpu_id]
    selected = [item for item in processes if item["uuid"] == state["uuid"]]
    if selected or state["memory_used_mib"] >= memory_threshold_mib:
        raise MCRealSmokeError(f"GPU {gpu_id} is busy")
    return {**state, "compute_processes": selected}


def gpu_preflight(
    gpu_id: int,
    memory_threshold_mib: int = MEMORY_THRESHOLD_MIB,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    gpu_text = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    compute_text = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return parse_gpu_state(
        gpu_text, compute_text, gpu_id, memory_threshold_mib=memory_threshold_mib
    )


def validate_paths(base_model: Path, adapter: Path, train_data: Path) -> dict[str, Any]:
    if not base_model.is_dir():
        raise MCRealSmokeError(f"base model directory is missing: {base_model}")
    if not adapter.is_dir():
        raise MCRealSmokeError(f"adapter directory is missing: {adapter}")
    if not train_data.is_file():
        raise MCRealSmokeError(f"train data is missing: {train_data}")
    adapter_config_path = adapter / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise MCRealSmokeError("BETA checkpoint lacks adapter_config.json")
    if not any(
        (adapter / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise MCRealSmokeError("BETA checkpoint lacks adapter weights")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise MCRealSmokeError("BETA adapter is not LoRA")
    actual_sha = file_sha256(train_data)
    if actual_sha != TRAIN_SHA256:
        raise MCRealSmokeError("frozen train_3000 SHA mismatch")
    return {"train_sha256": actual_sha, "adapter_config": adapter_config}


def load_tokenizer(path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def run_preflight(
    args: argparse.Namespace,
    *,
    tokenizer_loader: Any = load_tokenizer,
    gpu_checker: Any = gpu_preflight,
) -> dict[str, Any]:
    paths = validate_paths(args.base_model, args.adapter, args.train_data)
    tokenizer = tokenizer_loader(args.base_model)
    fixed = prepare_fixed_data(tokenizer, read_jsonl(args.train_data))
    gpu_state = gpu_checker(args.gpu_id, args.memory_threshold_mib)
    return {
        "status": "READY_TO_EXECUTE",
        "gpu": gpu_state,
        "paths": paths,
        "tokenizer": tokenizer,
        "fixed": fixed,
        "selected_sample_ids": {
            route: fixed["selection"][route]["sample_id"]
            for route in ("action", "chain")
        },
        "fixed_k2": fixed["summaries"],
    }


def parameter_sha256(model: torch.nn.Module, *, lora: bool) -> tuple[str, int]:
    digest = hashlib.sha256()
    matched = 0
    for name, parameter in model.named_parameters():
        if ("lora_" in name.lower()) != lora:
            continue
        value = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
        matched += 1
    if matched == 0:
        raise MCRealSmokeError("parameter hash matched no tensors")
    return digest.hexdigest(), matched


def lora_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name.lower()
    }


def restore_lora(model: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in snapshot:
                parameter.copy_(snapshot[name].to(parameter.device, dtype=parameter.dtype))


def lora_delta(
    model: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> tuple[float, float]:
    squared = 0.0
    maximum = 0.0
    for name, parameter in model.named_parameters():
        if name not in snapshot:
            continue
        delta = parameter.detach().float().cpu() - snapshot[name]
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return math.sqrt(squared), maximum


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def load_beta_model(args: argparse.Namespace) -> torch.nn.Module:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(
        base, str(args.adapter), is_trainable=True, local_files_only=True
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def validate_trainable(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name.lower() for name, _ in trainable):
        raise MCRealSmokeError("only LoRA parameters may be trainable")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name.lower()
    ):
        raise MCRealSmokeError("base model has trainable parameters")
    return trainable


def optimizer_state_is_lora_only(
    optimizer: torch.optim.Optimizer,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> bool:
    trainable_ids = {id(parameter) for _, parameter in trainable}
    return bool(optimizer.state) and all(
        id(parameter) in trainable_ids for parameter in optimizer.state
    )


def run_active_route(
    route: str,
    model: torch.nn.Module,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    batch: Mapping[str, Any],
    initial_base_hash: str,
    initial_lora_hash: str,
    initial_lora: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    restore_lora(model, initial_lora)
    restored_hash, _ = parameter_sha256(model, lora=True)
    if restored_hash != initial_lora_hash:
        raise MCRealSmokeError(f"{route} did not restore the identical LoRA start")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=SMOKE_CONFIG["learning_rate"],
        weight_decay=SMOKE_CONFIG["weight_decay"],
    )
    before = lora_snapshot(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = mc_optimizer_step(
        model,
        optimizer,
        batch_to_device(batch, device),
        batch["credit_units_per_candidate"],
        forward_batch_size=SMOKE_CONFIG["forward_batch_size"],
    )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    base_hash_after, _ = parameter_sha256(model, lora=False)
    lora_hash_after, _ = parameter_sha256(model, lora=True)
    delta_l2, delta_max = lora_delta(model, before)
    metadata = result["objective_metadata"]
    if not result["finite"] or result["grad_norm"] <= 0.0:
        raise MCRealSmokeError(f"{route} loss/gradient is invalid")
    if result["skipped_update"] or not result["optimizer_step_performed"]:
        raise MCRealSmokeError(f"{route} active update was skipped")
    if base_hash_after != initial_base_hash:
        raise MCRealSmokeError(f"{route} changed frozen base parameters")
    if lora_hash_after == initial_lora_hash or delta_l2 <= 0.0 or delta_max <= 0.0:
        raise MCRealSmokeError(f"{route} did not update LoRA parameters")
    if not optimizer_state_is_lora_only(optimizer, trainable):
        raise MCRealSmokeError(f"{route} optimizer state is not LoRA-only")
    return {
        "loss": result["loss"],
        "grad_norm": result["grad_norm"],
        "active_unit_count": result["active_unit_count"],
        "active_token_count": result["active_token_count"],
        "positive_unit_count": int(metadata["positive_unit_count"]),
        "negative_unit_count": int(metadata["negative_unit_count"]),
        "positive_credit_mass": float(metadata["positive_credit_mass"]),
        "negative_credit_mass": float(metadata["negative_credit_mass"]),
        "base_hash_before": initial_base_hash,
        "base_hash_after": base_hash_after,
        "base_delta_l2": 0.0,
        "lora_hash_before": initial_lora_hash,
        "lora_hash_after": lora_hash_after,
        "lora_delta_l2": delta_l2,
        "lora_delta_max_abs": delta_max,
        "optimizer_state_lora_only": True,
        "optimizer_state_parameter_count": len(optimizer.state),
        "peak_vram_mib": peak_vram_mib,
        "wall_seconds": wall_seconds,
    }


def execute_real_smoke(
    preflight: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        raise MCRealSmokeError("CUDA_VISIBLE_DEVICES must equal the explicit --gpu-id")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    execution_started = time.perf_counter()
    device = torch.device("cuda:0")
    model = load_beta_model(args)
    trainable = validate_trainable(model)
    model.train()
    initial_base_hash, base_tensor_count = parameter_sha256(model, lora=False)
    initial_lora_hash, lora_tensor_count = parameter_sha256(model, lora=True)
    initial_lora = lora_snapshot(model)

    control_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=SMOKE_CONFIG["learning_rate"],
        weight_decay=SMOKE_CONFIG["weight_decay"],
    )
    control_batch = batch_to_device(preflight["fixed"]["batches"]["action"], device)
    control_units = [[], []]
    control_result = mc_optimizer_step(
        model,
        control_optimizer,
        control_batch,
        control_units,
        forward_batch_size=SMOKE_CONFIG["forward_batch_size"],
    )
    control_base_hash, _ = parameter_sha256(model, lora=False)
    control_lora_hash, _ = parameter_sha256(model, lora=True)
    if not control_result["skipped_update"] or control_result["optimizer_step_performed"]:
        raise MCRealSmokeError("no-credit control did not skip optimizer step")
    if control_optimizer.state:
        raise MCRealSmokeError("no-credit control advanced optimizer state")
    if control_base_hash != initial_base_hash or control_lora_hash != initial_lora_hash:
        raise MCRealSmokeError("no-credit control changed model parameters")

    action = run_active_route(
        "action",
        model,
        trainable,
        preflight["fixed"]["batches"]["action"],
        initial_base_hash,
        initial_lora_hash,
        initial_lora,
        device,
    )
    chain = run_active_route(
        "chain",
        model,
        trainable,
        preflight["fixed"]["batches"]["chain"],
        initial_base_hash,
        initial_lora_hash,
        initial_lora,
        device,
    )
    output = {
        "status": "PASS",
        "config": SMOKE_CONFIG,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "selected_sample_ids": preflight["selected_sample_ids"],
        "trainable_tensor_count": len(trainable),
        "base_tensor_count": base_tensor_count,
        "lora_tensor_count": lora_tensor_count,
        "no_credit_control": {
            "loss": control_result["loss"],
            "grad_norm": control_result["grad_norm"],
            "skipped_update": True,
            "optimizer_step_performed": False,
            "optimizer_state_unchanged": True,
            "base_hash_unchanged": True,
            "lora_hash_unchanged": True,
        },
        "action": action,
        "chain": chain,
        "peak_vram_mib": max(action["peak_vram_mib"], chain["peak_vram_mib"]),
        "wall_seconds": time.perf_counter() - execution_started,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-model", type=Path, default=Path(BASE_MODEL))
    parser.add_argument("--adapter", type=Path, default=Path(ADAPTER))
    parser.add_argument("--train-data", type=Path, default=Path(TRAIN_DATA))
    parser.add_argument("--result-output", type=Path, default=Path(RESULT_OUTPUT))
    parser.add_argument(
        "--memory-threshold-mib", type=int, default=MEMORY_THRESHOLD_MIB
    )
    return parser


def public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": preflight["status"],
        "gpu": preflight["gpu"],
        "selected_sample_ids": preflight["selected_sample_ids"],
        "fixed_k2": preflight["fixed_k2"],
        "train_sha256": preflight["paths"]["train_sha256"],
        "execute_required": True,
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    preflight_fn: Any = run_preflight,
    execute_fn: Any = execute_real_smoke,
) -> dict[str, Any]:
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
        run_cli()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
