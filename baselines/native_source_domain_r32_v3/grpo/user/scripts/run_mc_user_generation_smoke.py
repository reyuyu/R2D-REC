#!/usr/bin/env python3
"""Generation-only real-model smoke for the MC_USER_v1 K=2 contract."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from run_mc_user_real_smoke import (
    ADAPTER,
    BASE_MODEL,
    MEMORY_THRESHOLD_MIB,
    TRAIN_DATA,
    gpu_preflight,
    load_tokenizer,
    read_jsonl,
    select_fixed_rows,
    validate_paths,
)
from run_user_grpo_smoke import render_prompt, trim_at_stop
from user_mc_batch import make_mc_policy_batch
from user_mc_objective import mc_unit_credit_loss
from user_mc_projection import MCCreditProjectionError
from user_mc_rollout import K, prepare_mc_scored_rollout


TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 512
SEED = 20260819
RESULT_OUTPUT = "/data/GRPO_USER/results/mc_user_v1_generation_smoke.json"


class MCGenerationSmokeError(RuntimeError):
    pass


class MCGenerationProjectionFailure(MCGenerationSmokeError):
    def __init__(self, record: Mapping[str, Any]):
        super().__init__(str(record["exception"]))
        self.record = dict(record)


def stop_token_ids(tokenizer: Any) -> set[int]:
    stop_ids = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int) and eos_token_id >= 0:
        stop_ids.add(eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.add(im_end)
    if not stop_ids:
        raise MCGenerationSmokeError("tokenizer provides no EOS or <|im_end|> stop ID")
    return stop_ids


def extract_generated_ids(
    output_ids: Any,
    *,
    prompt_width: int,
    stop_ids: set[int],
) -> list[list[int]]:
    values = output_ids.tolist() if hasattr(output_ids, "tolist") else output_ids
    rows = [[int(token_id) for token_id in row] for row in values]
    if len(rows) != K:
        raise MCGenerationSmokeError(f"generation returned {len(rows)} candidates, expected K=2")
    if any(len(row) < prompt_width for row in rows):
        raise MCGenerationSmokeError("generation output is shorter than the padded prompt")
    return [trim_at_stop(row[prompt_width:], stop_ids) for row in rows]


@torch.inference_mode()
def generate_k2_route(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[list[int]], dict[str, Any]]:
    rendered = render_prompt(tokenizer, row)
    encoded = tokenizer(
        [rendered],
        add_special_tokens=False,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    prompt_width = int(encoded["input_ids"].shape[1])
    if int(encoded["attention_mask"].sum()) != int(row["prompt_token_count"]):
        raise MCGenerationSmokeError("prompt true-token count drifted before generation")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    stop_ids = stop_token_ids(tokenizer)
    output = model.generate(
        **encoded,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_return_sequences=K,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=sorted(stop_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    generated_ids = extract_generated_ids(
        output, prompt_width=prompt_width, stop_ids=stop_ids
    )
    return generated_ids, {
        "prompt_token_count": int(row["prompt_token_count"]),
        "prompt_padded_width": prompt_width,
        "stop_ids": sorted(stop_ids),
    }


def _projection_failure_record(
    route: str,
    candidate_index: int,
    generated_ids: Sequence[int],
    tokenizer: Any,
    exc: Exception,
) -> dict[str, Any]:
    ids = [int(token_id) for token_id in generated_ids]
    try:
        completion = tokenizer.decode(ids, skip_special_tokens=False)
    except Exception as decode_exc:
        completion = f"<decode failed: {decode_exc}>"
    return {
        "route": route,
        "candidate_index": candidate_index,
        "exception": f"{type(exc).__name__}: {exc}",
        "decoded_completion": completion,
        "generated_token_count": len(ids),
        "generated_ids_head": ids[:32],
        "generated_ids_tail": ids[-32:],
    }


def _identify_projection_failure(
    route: str,
    row: Mapping[str, Any],
    generated_ids: Sequence[Sequence[int]],
    tokenizer: Any,
    original_exc: Exception,
) -> dict[str, Any]:
    for candidate_index, candidate_ids in enumerate(generated_ids):
        try:
            prepare_mc_scored_rollout(
                [row], [list(candidate_ids), list(candidate_ids)], tokenizer
            )
        except MCCreditProjectionError as exc:
            return _projection_failure_record(
                route, candidate_index, candidate_ids, tokenizer, exc
            )
    return _projection_failure_record(
        route, -1, [token for ids in generated_ids for token in ids], tokenizer, original_exc
    )


def validate_exact_policy_ids(
    batch: Mapping[str, Any], generated_ids: Sequence[Sequence[int]]
) -> None:
    if int(batch["candidate_count"]) != K or len(generated_ids) != K:
        raise MCGenerationSmokeError("policy batch does not preserve K=2")
    for candidate_index, expected in enumerate(generated_ids):
        expected_ids = [int(token_id) for token_id in expected]
        real_length = int(batch["completion_lengths"][candidate_index])
        if real_length != len(expected_ids):
            raise MCGenerationSmokeError("policy completion length differs from generated IDs")
        actual = batch["completion_ids"][candidate_index, :real_length].tolist()
        if actual != expected_ids:
            raise MCGenerationSmokeError("policy batch changed exact generated token IDs")
        if not bool(torch.all(batch["completion_mask"][candidate_index, :real_length] == 1)):
            raise MCGenerationSmokeError("generated tokens are not fully enabled in policy mask")
        if not bool(torch.all(batch["completion_mask"][candidate_index, real_length:] == 0)):
            raise MCGenerationSmokeError("policy completion padding mask is invalid")
        for unit in batch["credit_units_per_candidate"][candidate_index]:
            indices = [int(index) for index in unit["generated_token_indices"]]
            if any(index >= real_length for index in indices):
                raise MCGenerationSmokeError("credit unit touches policy padding")


def _candidate_overlap_metadata(
    units: Sequence[Mapping[str, Any]], generated_token_count: int
) -> dict[str, Any]:
    width = max(1, generated_token_count)
    logps = torch.zeros((1, width), dtype=torch.float64)
    mask = torch.zeros_like(logps)
    mask[:, :generated_token_count] = 1
    _loss, metadata = mc_unit_credit_loss(logps, [units], mask)
    return metadata


def score_generated_route(
    route: str,
    row: Mapping[str, Any],
    generated_ids: Sequence[Sequence[int]],
    tokenizer: Any,
) -> dict[str, Any]:
    if row.get("route") != route:
        raise MCGenerationSmokeError("route batch is not homogeneous")
    if len(generated_ids) != K:
        raise MCGenerationSmokeError("MC generation scoring requires exactly K=2")
    exact_ids = [[int(token_id) for token_id in ids] for ids in generated_ids]
    try:
        rollout = prepare_mc_scored_rollout([row], exact_ids, tokenizer)
    except MCCreditProjectionError as exc:
        raise MCGenerationProjectionFailure(
            _identify_projection_failure(route, row, exact_ids, tokenizer, exc)
        ) from exc
    batch = make_mc_policy_batch(tokenizer, rollout, device="cpu")
    validate_exact_policy_ids(batch, exact_ids)

    candidates = []
    for candidate_index in range(K):
        marginal = rollout["marginal_results"][candidate_index]
        units = rollout["credit_units_per_candidate"][candidate_index]
        overlap = _candidate_overlap_metadata(units, len(exact_ids[candidate_index]))
        projection = rollout["projection_per_candidate"][candidate_index]
        candidate_record = {
            "sample_id": row["sample_id"],
            "candidate_index": candidate_index,
            "generated_token_count": len(exact_ids[candidate_index]),
            "completion": rollout["completions"][candidate_index],
            "format_valid": bool(marginal["valid"]),
            "full_reward": float(marginal["full_reward"]),
            "positive_unit_count": sum(float(unit["delta"]) > 0 for unit in units),
            "negative_unit_count": sum(float(unit["delta"]) < 0 for unit in units),
            "zero_unit_count": sum(float(unit["delta"]) == 0 for unit in units),
            "active_marginal_unit_count": sum(
                float(unit["delta"]) != 0 for unit in units
            ),
            "projection_required": bool(projection["projection_required"]),
            "canonical_token_count": int(projection["canonical_token_count"]),
            "overlap_token_count": int(overlap["overlap_token_count"]),
            "same_sign_overlap_token_count": int(
                overlap["same_sign_overlap_token_count"]
            ),
            "mixed_sign_overlap_token_count": int(
                overlap["mixed_sign_overlap_token_count"]
            ),
            "max_active_units_per_token": int(
                overlap["max_active_units_per_token"]
            ),
        }
        if route == "action":
            candidate_record["f1"] = float(marginal["full_reward"])
        else:
            candidate_record.update(
                {
                    "full_action_alignment": float(
                        marginal.get("full_action_alignment", 0.0)
                    ),
                    "full_logic_alignment": float(
                        marginal.get("full_logic_alignment", 0.0)
                    ),
                }
            )
        candidates.append(candidate_record)
    return {
        "route": route,
        "candidate_count": K,
        "candidates": candidates,
        "rollout": rollout,
        "policy_batch": batch,
    }


def load_beta_for_generation(args: argparse.Namespace) -> torch.nn.Module:
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
        base, str(args.adapter), is_trainable=False, local_files_only=True
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.config.use_cache = True
    model.eval()
    return model


def run_preflight(
    args: argparse.Namespace,
    *,
    tokenizer_loader: Any = load_tokenizer,
    gpu_checker: Any = gpu_preflight,
) -> dict[str, Any]:
    paths = validate_paths(args.base_model, args.adapter, args.train_data)
    tokenizer = tokenizer_loader(args.base_model)
    selection = select_fixed_rows(read_jsonl(args.train_data))
    prompt_counts = {}
    for route in ("action", "chain"):
        render_prompt(tokenizer, selection[route])
        prompt_counts[route] = int(selection[route]["prompt_token_count"])
    gpu_state = gpu_checker(args.gpu_id, args.memory_threshold_mib)
    return {
        "status": "READY_TO_EXECUTE",
        "gpu": gpu_state,
        "paths": paths,
        "tokenizer": tokenizer,
        "selection": selection,
        "selected_sample_ids": {
            route: selection[route]["sample_id"] for route in ("action", "chain")
        },
        "prompt_token_counts": prompt_counts,
    }


def _write_result(path: Path, output: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def execute_generation_smoke(
    preflight: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    model_loader: Any = load_beta_for_generation,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        raise MCGenerationSmokeError(
            "CUDA_VISIBLE_DEVICES must equal the explicit --gpu-id"
        )
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    from transformers import set_seed

    set_seed(SEED)
    device = torch.device("cuda:0")
    tokenizer = preflight["tokenizer"]
    model = model_loader(args)
    started = time.perf_counter()
    route_outputs = {}
    projection_counts = {"identity_projection_count": 0, "noncanonical_projection_count": 0}
    try:
        for route in ("action", "chain"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            route_started = time.perf_counter()
            generated_ids, generation_contract = generate_k2_route(
                model, tokenizer, preflight["selection"][route], device
            )
            torch.cuda.synchronize(device)
            generation_wall = time.perf_counter() - route_started
            generation_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            scored = score_generated_route(
                route, preflight["selection"][route], generated_ids, tokenizer
            )
            for candidate in scored["candidates"]:
                key = (
                    "noncanonical_projection_count"
                    if candidate["projection_required"]
                    else "identity_projection_count"
                )
                projection_counts[key] += 1
            route_outputs[route] = {
                **generation_contract,
                "candidate_count": scored["candidate_count"],
                "candidates": scored["candidates"],
                "generation_peak_vram_mib": generation_peak,
                "generation_wall_seconds": generation_wall,
                "policy_batch_exact_generated_ids": True,
            }
    except MCGenerationProjectionFailure as exc:
        output = {
            "status": "FAIL",
            "failure_kind": "PROJECTION_FAIL_CLOSED",
            "projection_failures": [exc.record],
        }
        _write_result(args.result_output, output)
        return output

    output = {
        "status": "PASS",
        "config": {
            "K": K,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "use_cache": True,
            "seed": SEED,
        },
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "selected_sample_ids": preflight["selected_sample_ids"],
        "routes": route_outputs,
        "projection": projection_counts,
        "overall_peak_vram_mib": max(
            output["generation_peak_vram_mib"] for output in route_outputs.values()
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_result(args.result_output, output)
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
        "prompt_token_counts": preflight["prompt_token_counts"],
        "train_sha256": preflight["paths"]["train_sha256"],
        "execute_required": True,
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    preflight_fn: Any = run_preflight,
    execute_fn: Any = execute_generation_smoke,
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
        output = run_cli()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False))
        return 1
    return 0 if output["status"] in {"PASS", "READY_TO_EXECUTE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
