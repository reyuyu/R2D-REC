#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_DATA = {
    "01_sft_beta/onereason_bata_baseline.jsonl": {
        "sha256": "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca",
        "rows": 222001,
    },
    "02_recommendation_grpo/train.jsonl": {
        "sha256": "791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc",
        "rows": 3098,
    },
    "03_user_grpo/train_3000.jsonl": {
        "sha256": "5fc4f2ede241ca8049185d8a1e9303b92747d399806d4d939e4040793e9ed801",
        "rows": 3000,
    },
}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def assert_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    parent = root.expanduser().resolve(strict=True)
    if resolved != parent and parent not in resolved.parents:
        raise RuntimeError(f"{label} must resolve under {parent}: {resolved}")
    return resolved


def render_configs(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    data = assert_under(args.data_root, Path("/root/reproduce_datasets"), "data root")
    config_dir = root / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    sft_output = root / "outputs" / "01_sft_beta"
    sft = f"""### Final-chain byte-exact BETA SFT reproduction.
model_name_or_path: {args.base_model.resolve()}
trust_remote_code: true
flash_attn: fa2

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all
enable_liger_kernel: true

dataset: onereason_bata_baseline
dataset_dir: {data / '01_sft_beta'}
template: qwen3_nothink
cutoff_len: 8192
packing: true
neat_packing: true
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 8
remove_unused_columns: false

output_dir: {sft_output}
logging_steps: 5
save_strategy: epoch
save_total_limit: 3
save_only_model: false
plot_loss: true
overwrite_output_dir: true
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
bf16: true
pure_bf16: true
gradient_checkpointing: true
use_reentrant_gc: false
ddp_timeout: 180000000
seed: 20260806
resume_from_checkpoint: null

rec_pu_enabled: false
multitask_pack_ratio_enabled: false
rec_candidate_metrics_enabled: true
rec_candidate_metrics_interval: 50
"""
    (config_dir / "01_sft_beta.yaml").write_text(sft, encoding="utf-8", newline="\n")

    mc = {
        "experiment_type": "formal",
        "stage": "strongparent_lr3e7_step100_repro",
        "base_model": str(args.base_model.resolve()),
        "adapter": str(root / "outputs" / "03_grpo_tk" / "GRPO-TK-REPRO-TO250" / "checkpoint-250"),
        "parent_experiment": "GR_REC_ThinkSample8_FullSID_v3",
        "parent_checkpoint_step": 250,
        "parent_recorded_external_score": 1.3579,
        "train_data": str(data / "03_user_grpo" / "train_3000.jsonl"),
        "train_sha256": EXPECTED_DATA["03_user_grpo/train_3000.jsonl"]["sha256"],
        "prompt_count": 100,
        "action_count": 50,
        "chain_count": 50,
        "selection_seed": 20260823,
        "route_schedule": "strict_alternating",
        "K": 4,
        "world_size": 4,
        "parallelism": "candidate_parallel",
        "temperature": 0.9,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "learning_rate": 3e-7,
        "weight_decay": 0.0,
        "forward_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_weight": 1.0,
        "local_weight": 0.3,
        "checkpoint_steps": [25, 50, 75, 100],
        "resume_supported": False,
        "resume_policy": "continuous_run_only",
    }
    json_write(config_dir / "04_mc_user.json", mc)
    json_write(
        config_dir / "data_registry.json",
        {
            "registry_root": str(data),
            "datasets": {
                "sft_beta": str(data / "01_sft_beta" / "onereason_bata_baseline.jsonl"),
                "recommendation_grpo": str(data / "02_recommendation_grpo" / "train.jsonl"),
                "user_grpo": str(data / "03_user_grpo" / "train_3000.jsonl"),
                "user_selection": str(data / "03_user_grpo" / "selected_200_order.jsonl"),
            },
        },
    )
    json_write(
        config_dir / "seed_contract.json",
        {
            "sft_seed": 20260806,
            "recommendation_grpo_seed": 20260816,
            "recommendation_probe_seed": 20260818,
            "grpo_tk_seed": 20260816,
            "mc_user_selection_seed": 20260823,
            "mc_candidate_seed": "sha256(selection_seed:prompt_step:sample_id:candidate_index)",
        },
    )


def git_state(code_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(code_root), "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("reproduction code worktree is dirty")
    return {"commit": commit, "clean": True}


def python_environment(python: Path) -> dict[str, Any]:
    script = """
import json, platform
versions = {}
for name in ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'safetensors', 'datasets'):
    module = __import__(name)
    versions[name] = getattr(module, '__version__', 'unknown')
print(json.dumps({'python': platform.python_version(), 'packages': versions}, sort_keys=True))
"""
    output = subprocess.check_output([str(python), "-c", script], text=True)
    return json.loads(output)


def verify_environment(args: argparse.Namespace) -> dict[str, Any]:
    lock = json.loads(args.environment_lock.read_text(encoding="utf-8"))
    observed_system = python_environment(args.system_python)
    observed_sft = python_environment(args.sft_python)
    if observed_system != lock["system_python"]:
        raise RuntimeError(f"system Python environment mismatch: {observed_system}")
    if observed_sft != lock["sft_python"]:
        raise RuntimeError(f"SFT Python environment mismatch: {observed_sft}")

    factory = git_state(args.llamafactory_root.resolve())
    expected_factory = lock["llamafactory"]
    if factory["commit"] != expected_factory["reproduction_commit"]:
        raise RuntimeError(f"LLaMAFactory commit mismatch: {factory['commit']}")
    args_file = args.llamafactory_root / "src/llamafactory/hparams/finetuning_args.py"
    if sha256_file(args_file) != expected_factory["finetuning_args_sha256"]:
        raise RuntimeError("LLaMAFactory native argument patch hash mismatch")

    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    expected_gpu = lock["gpu"]
    expected_line = f"{expected_gpu['name']}, {expected_gpu['compute_capability']}, {expected_gpu['driver_version']}"
    if len(gpu_rows) != expected_gpu["count"] or any(row.strip() != expected_line for row in gpu_rows):
        raise RuntimeError(f"GPU environment mismatch: {gpu_rows}")
    return {
        "system_python": observed_system,
        "sft_python": observed_sft,
        "llamafactory": factory,
        "gpu": expected_gpu,
    }


def preflight(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    data = assert_under(args.data_root, Path("/root/reproduce_datasets"), "data root")
    if not args.base_model.resolve().is_dir():
        raise RuntimeError(f"base model missing: {args.base_model}")
    if not (args.base_model / "config.json").is_file():
        raise RuntimeError("base model config.json missing")

    if not args.base_manifest.is_file():
        raise RuntimeError("base model SHA256 manifest missing")
    base_checksum_result = subprocess.run(
        ["sha256sum", "-c", str(args.base_manifest.resolve())],
        cwd=args.base_model.resolve(),
        check=False,
        capture_output=True,
        text=True,
    )
    if base_checksum_result.returncode != 0:
        raise RuntimeError(f"base model checksum failed: {base_checksum_result.stderr.strip()}")
    base_file_count = len([line for line in args.base_manifest.read_text(encoding="utf-8").splitlines() if line.strip()])

    environment = verify_environment(args)
    data_report = {}
    for relative, contract in EXPECTED_DATA.items():
        path = data / relative
        actual_hash = sha256_file(path)
        actual_rows = line_count(path)
        if actual_hash != contract["sha256"] or actual_rows != contract["rows"]:
            raise RuntimeError(f"frozen dataset mismatch: {relative}")
        data_report[relative] = {"sha256": actual_hash, "rows": actual_rows}

    checksum_file = data / "SHA256SUMS"
    if not checksum_file.is_file():
        raise RuntimeError("frozen data package SHA256SUMS missing")
    checksum_result = subprocess.run(
        ["sha256sum", "-c", checksum_file.name],
        cwd=data,
        check=False,
        capture_output=True,
        text=True,
    )
    if checksum_result.returncode != 0:
        raise RuntimeError(f"frozen data package checksum failed: {checksum_result.stderr.strip()}")

    selection = [json.loads(line) for line in (data / "03_user_grpo/selected_200_order.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(selection) != 200 or len({x["sample_id"] for x in selection}) != 200:
        raise RuntimeError("MC selection must contain 200 unique sample IDs")
    if any(x["route"] != ("action" if i % 2 == 0 else "chain") for i, x in enumerate(selection)):
        raise RuntimeError("MC selection must be strict Action/Chain alternating")

    teacher_root = args.teacher_root.resolve()
    teacher_weights = [p for p in teacher_root.rglob("*") if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES]
    if teacher_weights:
        raise RuntimeError(f"teacher declaration says none, but weights exist: {teacher_weights}")
    if not (teacher_root / "README.md").is_file():
        raise RuntimeError("teacher no-model declaration missing")

    registry = json.loads((root / "runtime_configs/data_registry.json").read_text(encoding="utf-8"))
    for name, path_text in registry["datasets"].items():
        assert_under(Path(path_text), Path("/root/reproduce_datasets"), f"registered dataset {name}")

    code = git_state(args.code_root.resolve())
    required_markers = {
        "GRPO_DATA_PATH": args.code_root / "baselines/native_source_domain_r32_v3/grpo/scripts/run_grpo_trl_smoke.py",
        "GRPO_TK_PARENT_ADAPTER": args.code_root / "baselines/native_source_domain_r32_v3/grpo/ablations/gr_rec_think_sample8_fullsid_v3/run_sample8_fullsid_train.py",
        "strongparent_lr3e7_step100_repro": args.code_root / "baselines/native_source_domain_r32_v3/grpo/user/scripts/run_mc_user_formal_hybrid_k4_ddp_v1.py",
    }
    for marker, path in required_markers.items():
        if marker not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"reproduction code marker missing: {marker}")

    report = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "data_root": str(data),
        "teacher_model_used": False,
        "datasets": data_report,
        "package_sha256sums_verified": True,
        "selection_count": len(selection),
        "code": code,
        "base_model": str(args.base_model.resolve()),
        "base_model_sha256s_verified": True,
        "base_model_file_count": base_file_count,
        "environment": environment,
    }
    json_write(root / "state/preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def gpu_check(args: argparse.Namespace) -> None:
    gpu_csv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    rows = []
    for line in gpu_csv.splitlines():
        index, memory = [part.strip() for part in line.split(",", 1)]
        rows.append({"index": int(index), "memory_used_mib": int(memory)})
    if len(rows) != args.expected_gpus or [x["index"] for x in rows] != list(range(args.expected_gpus)):
        raise RuntimeError(f"expected physical GPUs 0..{args.expected_gpus - 1}, found {rows}")
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if compute:
        raise RuntimeError("GPU compute process exists; reproduction will not preempt it")
    busy = [x for x in rows if x["memory_used_mib"] >= args.memory_threshold_mib]
    if busy:
        raise RuntimeError(f"GPU memory preflight failed: {busy}")
    print(json.dumps({"status": "PASS", "gpus": rows, "compute_process_count": 0}, indent=2))


def adapter_contract(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    weights = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise RuntimeError(f"adapter files missing: {path}")
    forbidden = [name for name in ("model.safetensors", "pytorch_model.bin") if (path / name).exists()]
    if forbidden:
        raise RuntimeError(f"base/full-model weights found in adapter checkpoint: {forbidden}")
    with safe_open(weights, framework="pt", device="cpu") as handle:
        names = list(handle.keys())
    lora = [name for name in names if "lora_" in name.lower()]
    if len(names) != 504 or len(lora) != 504:
        raise RuntimeError(f"expected 504/504 LoRA tensors, got {len(names)}/{len(lora)}")
    return {
        "path": str(path.resolve()),
        "adapter_sha256": sha256_file(weights),
        "adapter_tensor_count": len(names),
        "lora_tensor_count": len(lora),
    }


def verify_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    path = args.path.resolve(strict=True)
    result = adapter_contract(path)
    result.update({"kind": args.kind, "step": args.step})
    if args.kind == "trainer":
        required = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", *(f"rng_state_{i}.pth" for i in range(4))]
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise RuntimeError(f"trainer checkpoint incomplete: {missing}")
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        if int(state.get("global_step", -1)) != args.step:
            raise RuntimeError(f"trainer global_step mismatch: {state.get('global_step')} != {args.step}")
    else:
        formal = path / "formal_state.json"
        if not formal.is_file():
            raise RuntimeError("MC formal_state.json missing")
        state = json.loads(formal.read_text(encoding="utf-8"))
        observed = state.get("prompt_step", state.get("step"))
        if observed is not None and int(observed) != args.step:
            raise RuntimeError(f"MC prompt step mismatch: {observed} != {args.step}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def verify_mc_selection(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    generated = manifest.get("prompts", [])
    expected = [json.loads(line) for line in args.expected.read_text(encoding="utf-8").splitlines()][: args.count]
    expected_compact = [
        {"prompt_step": index, "route": row["route"], "sample_id": row["sample_id"]}
        for index, row in enumerate(expected, 1)
    ]
    if generated != expected_compact:
        raise RuntimeError("MC generated selection differs from packaged historical order")
    print(json.dumps({"status": "PASS", "selection_count": len(generated), "exact_order_match": True}, indent=2))


def verify_mc_summary(args: argparse.Namespace) -> None:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    checks = {
        "status": summary.get("status") == "PASS",
        "prompt_step": int(summary.get("prompt_step", -1)) == args.prompt_step,
        "optimizer_step": 0 <= int(summary.get("optimizer_step", -1)) <= args.optimizer_step_max,
        "base_hash_unchanged": summary.get("base_hash_unchanged") is True,
        "trainable_tensor_count": int(summary.get("trainable_tensor_count", -1)) == 504,
        "lora_tensor_count": int(summary.get("lora_tensor_count", -1)) == 504,
    }
    if not all(checks.values()):
        raise RuntimeError(f"MC summary contract failed: {checks}")
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2))


def mark_pass(args: argparse.Namespace) -> None:
    checkpoint = verify_checkpoint(args)
    json_write(
        args.state_dir / f"{args.stage}.PASS.json",
        {"status": "PASS", "stage": args.stage, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": checkpoint},
    )


def mark_failure(args: argparse.Namespace) -> None:
    json_write(
        args.state_dir / f"{args.stage}.STOPPED.json",
        {"status": "STOPPED", "stage": args.stage, "exit_code": args.exit_code, "time_utc": datetime.now(timezone.utc).isoformat()},
    )


def final_report(args: argparse.Namespace) -> None:
    checkpoints = {}
    groups = (
        ("sft_beta", args.sft.parent, "trainer", (553, 1106)),
        ("gr_rec_v1", args.grrec.parent, "trainer", (500, 1000, 1500)),
        ("grpo_tk", args.tk.parent, "trainer", (50, 100, 150, 200, 250)),
        ("mc_user", args.mc.parent, "mc", (25, 50, 75, 100)),
    )
    for name, parent, kind, steps in groups:
        checkpoints[name] = []
        for step in steps:
            child = parent / (f"checkpoint-{step}" if kind == "trainer" else f"prompt-step-{step:04d}")
            ns = argparse.Namespace(path=child, kind=kind, step=step)
            checkpoints[name].append(verify_checkpoint(ns))
    report = {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": checkpoints}
    json_write(args.root / "FINAL_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("render-configs")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.set_defaults(func=render_configs)

    p = sub.add_parser("preflight")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument("--system-python", type=Path, required=True)
    p.add_argument("--sft-python", type=Path, required=True)
    p.add_argument("--llamafactory-root", type=Path, required=True)
    p.add_argument("--environment-lock", type=Path, required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.set_defaults(func=preflight)

    p = sub.add_parser("gpu-check")
    p.add_argument("--expected-gpus", type=int, required=True)
    p.add_argument("--memory-threshold-mib", type=int, required=True)
    p.set_defaults(func=gpu_check)

    for command, func in (("verify-checkpoint", verify_checkpoint), ("mark-pass", mark_pass)):
        p = sub.add_parser(command)
        option = "--path" if command == "verify-checkpoint" else "--checkpoint"
        p.add_argument(option, type=Path, required=True, dest="path" if command == "verify-checkpoint" else "checkpoint")
        if command == "mark-pass":
            p.add_argument("--state-dir", type=Path, required=True)
            p.add_argument("--stage", required=True)
            p.set_defaults(path=None)
        p.add_argument("--kind", choices=("trainer", "mc"), required=True)
        p.add_argument("--step", type=int, required=True)
        p.set_defaults(func=func)

    p = sub.add_parser("mark-failure")
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.set_defaults(func=mark_failure)

    p = sub.add_parser("verify-mc-selection")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.set_defaults(func=verify_mc_selection)

    p = sub.add_parser("verify-mc-summary")
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--prompt-step", type=int, required=True)
    p.add_argument("--optimizer-step-max", type=int, required=True)
    p.set_defaults(func=verify_mc_summary)

    p = sub.add_parser("final-report")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--grrec", type=Path, required=True)
    p.add_argument("--tk", type=Path, required=True)
    p.add_argument("--mc", type=Path, required=True)
    p.set_defaults(func=final_report)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "mark-pass":
        args.path = args.checkpoint
    args.func(args)


if __name__ == "__main__":
    main()
