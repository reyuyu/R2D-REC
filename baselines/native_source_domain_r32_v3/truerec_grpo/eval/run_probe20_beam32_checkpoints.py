"""Evaluate every saved TrueRec checkpoint on frozen Probe20 with Beam32 ABC3."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "analysis", ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from beta_baseline_init import BETA_CHECKPOINT, PRETRAINED_BASE  # noqa: E402
from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from real_three_group_resume_smoke import LoraCheckpointState  # noqa: E402
from rollout_metrics import assess_candidate, group_summary  # noqa: E402
from rollout_runtime_v1 import FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID  # noqa: E402


PROBE_RECORDS = ROOT / "data/fixed_domain_abc/probe20_records.jsonl"
RESULT_DIRNAME = "beam32_probe"
STATUS_FILENAME = "status.json"
CONTRACT = {
    "name": "TRUEREC_PROBE20_NOTHINK_EMPTY_THINK_FIXED_DOMAIN_BEAM32_ABC3_V1",
    "route": "NoThink",
    "empty_think": True,
    "fixed_domain_in_context": True,
    "bridge": False,
    "num_beams": 32,
    "num_return_sequences": 32,
    "do_sample": False,
    "min_new_tokens": 3,
    "max_new_tokens": 3,
    "action": ["A", "B", "C"],
    "score_scope": "fixed Probe20 checkpoint diagnostic; not official external evaluation",
}
GENERATION_KWARGS = {
    "do_sample": False,
    "num_beams": 32,
    "num_return_sequences": 32,
    "min_new_tokens": 3,
    "max_new_tokens": 3,
    "early_stopping": True,
    "use_cache": True,
    "return_dict_in_generate": True,
    "output_scores": True,
}


class Beam32ProbeError(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_probe_records(path: Path = PROBE_RECORDS) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    group_ids = [str(row["recommendation_group_id"]) for row in rows]
    if len(rows) != 20 or len(set(group_ids)) != 20:
        raise Beam32ProbeError("Probe20 must contain exactly 20 unique business groups")
    return rows


def checkpoint_inventory(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for checkpoint in (run_dir / "checkpoints").glob("checkpoint-step-*"):
        try:
            step = int(checkpoint.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if not (checkpoint / "state.pt").is_file():
            continue
        metadata = {}
        try:
            metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        rows.append({
            "step": step,
            "checkpoint": checkpoint.name,
            "path": str(checkpoint.resolve()),
            "model_state_sha256": metadata.get("model_state_sha256"),
        })
    return sorted(rows, key=lambda row: row["step"])


def generation_kwargs() -> dict[str, Any]:
    return {
        **GENERATION_KWARGS,
        "eos_token_id": list(FORMAL_EOS_TOKEN_IDS),
        "pad_token_id": FORMAL_PAD_TOKEN_ID,
    }


def summarize_groups(groups: Sequence[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    if not groups:
        raise Beam32ProbeError("cannot summarize an empty Beam32 result")
    candidates = [candidate for group in groups for candidate in group["candidates"]]
    count = len(groups)
    candidate_count = len(candidates)
    def rate(values: Iterable[Any]) -> float:
        materialized = list(values)
        return sum(bool(value) for value in materialized) / len(materialized)
    return {
        "contract": CONTRACT,
        "group_count": count,
        "candidate_count": candidate_count,
        "group_any_A_rate": rate(group["ANY_A_HIT"] for group in groups),
        "group_any_AB_rate": rate(group["ANY_AB_HIT"] for group in groups),
        "group_any_exact_rate": rate(group["ANY_EXACT"] for group in groups),
        "candidate_format_valid_rate": rate(row["format_valid"] for row in candidates),
        "candidate_A_hit_rate": rate(row["A_hit"] for row in candidates),
        "candidate_AB_hit_rate": rate(row["AB_hit"] for row in candidates),
        "candidate_exact_rate": rate(row["exact"] for row in candidates),
        "mean_unique_sid_at_32": sum(group["unique_valid_ABC_count"] for group in groups) / count,
        "wall_time_seconds": float(wall_seconds),
    }


def evaluate_record(
    model: Any, renderer: Any, record: dict[str, Any], device: Any,
    generate_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    import torch

    context = renderer.rl_context_ids(
        record["system"], record["user_content_nothink"], record["fixed_domain_token"],
    )
    empty_think_domain = renderer.encode("<think>\n\n</think>\n" + record["fixed_domain_token"])
    if context[-len(empty_think_domain):] != empty_think_domain:
        raise Beam32ProbeError("renderer context is not empty-think plus fixed-domain")
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generate = generate_fn or model.generate
    output = generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs())
    sequences = output.sequences[:, input_ids.shape[1]:]
    tails = sequences.detach().cpu().tolist()
    if len(tails) != 32 or any(len(row) != 3 for row in tails):
        raise Beam32ProbeError("Beam32 output must be exactly 32 x 3 action tokens")
    scores = getattr(output, "sequences_scores", None)
    score_values = scores.detach().float().cpu().tolist() if scores is not None else [None] * 32
    candidates = []
    for beam_rank, (token_ids, beam_score) in enumerate(zip(tails, score_values)):
        metrics = assess_candidate(
            token_ids, renderer.tokenizer.convert_ids_to_tokens, record["all_gold_abc"],
            record["fixed_domain_token"], record.get("history_sids", ()),
        )
        candidates.append({
            "beam_rank": beam_rank,
            "completion_ids": token_ids,
            "completion_tokens": renderer.tokenizer.convert_ids_to_tokens(token_ids),
            "raw_text_with_special_tokens": renderer.tokenizer.decode(token_ids, skip_special_tokens=False),
            "beam_score": None if beam_score is None else float(beam_score),
            **metrics,
        })
    aggregate = group_summary([{"raw_token_ids": row["completion_ids"], **row} for row in candidates])
    return {
        "recommendation_group_id": str(record["recommendation_group_id"]),
        "target_domain": str(record["target_domain"]),
        "context_token_count": len(context),
        "all_gold_abc": list(record["all_gold_abc"]),
        "all_gold_sids": list(record.get("all_gold_sids", ())),
        "K": int(record.get("K", len(record["all_gold_abc"]))),
        "candidates": candidates,
        **aggregate,
    }


def load_model(device: Any) -> tuple[Any, BetaGammaRenderer]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    renderer = BetaGammaRenderer()
    base = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_BASE, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        base, BETA_CHECKPOINT, is_trainable=False, local_files_only=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, renderer


def load_checkpoint_weights(model: Any, checkpoint: Path) -> str | None:
    import torch

    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise Beam32ProbeError(f"checkpoint has no LoRA state: {checkpoint}")
    LoraCheckpointState(model).load_state_dict(state, strict=True)
    model.eval()
    return metadata.get("model_state_sha256")


def available_gpu(min_free_gib: float) -> dict[str, Any] | None:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    choices = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        index, name, free_mib, utilization = values
        row = {"index": int(index), "name": name, "free_gib": float(free_mib) / 1024, "utilization": int(utilization)}
        if row["free_gib"] >= min_free_gib and row["utilization"] <= 5:
            choices.append(row)
    return max(choices, key=lambda row: row["free_gib"], default=None)


def completed_steps(result_root: Path) -> set[int]:
    result = set()
    for path in result_root.glob("checkpoint-step-*/summary.json"):
        try:
            result.add(int(path.parent.name.rsplit("-", 1)[-1]))
        except ValueError:
            pass
    return result


def rebuild_curve(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in result_root.glob("checkpoint-step-*/summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "step": int(value["step"]),
            "checkpoint": value["checkpoint"],
            "group_any_A_rate": value["group_any_A_rate"],
            "group_any_AB_rate": value["group_any_AB_rate"],
            "group_any_exact_rate": value["group_any_exact_rate"],
            "candidate_exact_rate": value["candidate_exact_rate"],
            "mean_unique_sid_at_32": value["mean_unique_sid_at_32"],
        })
    rows.sort(key=lambda row: row["step"])
    atomic_json(result_root / "curve.json", rows)
    return rows


def run_all(run_dir: Path, *, min_free_gib: float = 70.0, poll_seconds: int = 60) -> None:
    result_root = run_dir / RESULT_DIRNAME
    status_path = result_root / STATUS_FILENAME
    result_root.mkdir(parents=True, exist_ok=True)
    inventory = checkpoint_inventory(run_dir)
    if not inventory:
        raise Beam32ProbeError("run has no checkpoints")
    done = completed_steps(result_root)
    pending = [row for row in inventory if row["step"] not in done]
    atomic_json(status_path, {
        "state": "WAITING_FOR_GPU", "run_id": run_dir.name, "total": len(inventory),
        "completed": len(done), "pending": len(pending), "min_free_gib": min_free_gib,
        "contract": CONTRACT, "pid": os.getpid(), "updated_at": time.time(),
    })
    gpu = available_gpu(min_free_gib)
    while gpu is None:
        time.sleep(max(5, poll_seconds))
        gpu = available_gpu(min_free_gib)
    # Training may have produced more checkpoints while this job waited for a GPU.
    inventory = checkpoint_inventory(run_dir)
    done = completed_steps(result_root)
    pending = [row for row in inventory if row["step"] not in done]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    import torch

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, renderer = load_model(device)
    records = load_probe_records()
    for position, item in enumerate(pending, start=len(done) + 1):
        checkpoint = Path(item["path"])
        atomic_json(status_path, {
            "state": "RUNNING", "run_id": run_dir.name, "total": len(inventory),
            "completed": position - 1, "pending": len(inventory) - position + 1,
            "current_checkpoint": item["checkpoint"], "current_step": item["step"],
            "gpu": gpu, "contract": CONTRACT, "pid": os.getpid(), "updated_at": time.time(),
        })
        model_sha = load_checkpoint_weights(model, checkpoint)
        started = time.perf_counter()
        groups = []
        with torch.inference_mode():
            for record in records:
                groups.append(evaluate_record(model, renderer, record, device))
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        output_dir = result_root / item["checkpoint"]
        summary = {
            "step": item["step"], "checkpoint": item["checkpoint"],
            "checkpoint_path": str(checkpoint), "model_state_sha256": model_sha,
            **summarize_groups(groups, wall),
        }
        write_jsonl(output_dir / "groups.jsonl", [
            {key: value for key, value in group.items() if key != "candidates"} for group in groups
        ])
        write_jsonl(output_dir / "explain.jsonl", groups)
        atomic_json(output_dir / "summary.json", summary)
        rebuild_curve(result_root)
    curve = rebuild_curve(result_root)
    atomic_json(status_path, {
        "state": "COMPLETED", "run_id": run_dir.name, "total": len(inventory),
        "completed": len(curve), "pending": 0, "gpu": gpu, "contract": CONTRACT,
        "pid": os.getpid(), "updated_at": time.time(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=70.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    try:
        run_all(args.run_dir.resolve(), min_free_gib=args.min_free_gib, poll_seconds=args.poll_seconds)
    except Exception as exc:
        atomic_json(args.run_dir / RESULT_DIRNAME / STATUS_FILENAME, {
            "state": "FAILED", "error": f"{type(exc).__name__}: {exc}", "updated_at": time.time(),
            "contract": CONTRACT, "pid": os.getpid(),
        })
        raise


if __name__ == "__main__":
    main()
