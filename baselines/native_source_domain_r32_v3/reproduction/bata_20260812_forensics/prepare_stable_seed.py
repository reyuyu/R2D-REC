#!/usr/bin/env python3
"""Prepare fresh-base, two-epoch BATA SFT seed experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


RUN_ID_RE = re.compile(r"^SEED-S([0-9]{1,10})-([0-9]{8}-[0-9]{6})$")
MAX_SEED = 2**31 - 1
MIDPOINT_STEP = 553
FINAL_STEP = 1106
MIN_FREE_BYTES = 20 * 1024**3
REQUIRED_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
    return seed


def validate_run_id(run_id: str, seed: int) -> str:
    match = RUN_ID_RE.fullmatch(run_id)
    if match is None or int(match.group(1)) != seed:
        raise ValueError("run_id does not match the fixed seed-labelled format")
    return run_id


def ensure_capacity(root: Path) -> dict[str, int]:
    usage = shutil.disk_usage(root)
    if usage.free < MIN_FREE_BYTES:
        raise RuntimeError("seed experiment root has less than 20 GiB available")
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def checkpoint_contract(checkpoint: Path, expected_step: int) -> dict[str, str]:
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"checkpoint is incomplete: {missing}")
    trainer = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if (trainer.get("global_step"), trainer.get("max_steps")) != (expected_step, FINAL_STEP):
        raise RuntimeError(f"checkpoint is not the expected step-{expected_step} state")
    return {name: sha256_file(checkpoint / name) for name in REQUIRED_CHECKPOINT_FILES}


def prepare_seed_runtime(stable_root: Path, support_dir: Path) -> Path:
    source = stable_root / "source_runtime"
    runtime = stable_root / "source_runtime_seed"
    contract_path = stable_root / "seed_runtime_contract.json"
    files = (
        "scripts/train_native_source_domain_r32_v3.py",
        "scripts/stable_seed_runtime.py",
    )
    if runtime.exists():
        if not contract_path.is_file():
            raise RuntimeError("seed runtime exists without its contract")
        expected = json.loads(contract_path.read_text(encoding="utf-8"))
        actual = {name: sha256_file(runtime / name) for name in files}
        if actual != expected:
            raise RuntimeError("seed runtime changed after preparation")
        return runtime

    temporary = stable_root / ".source_runtime_seed.tmp"
    if temporary.exists():
        raise RuntimeError("incomplete seed runtime preparation exists")
    shutil.copytree(source, temporary, copy_function=shutil.copy2)
    shutil.copy2(support_dir / "stable_seed_runtime.py", temporary / "scripts/stable_seed_runtime.py")
    trainer_path = temporary / "scripts/train_native_source_domain_r32_v3.py"
    trainer = trainer_path.read_text(encoding="utf-8")
    import_anchor = "from stable553_runtime import stable553_callbacks\n"
    callback_anchor = "    callbacks.extend(stable553_callbacks())\n"
    if trainer.count(import_anchor) != 1 or trainer.count(callback_anchor) != 1:
        raise RuntimeError("trainer integration anchors are not unique")
    trainer = trainer.replace(
        import_anchor,
        import_anchor + "from stable_seed_runtime import stable_seed_callbacks\n",
    ).replace(
        callback_anchor,
        callback_anchor + "    callbacks.extend(stable_seed_callbacks())\n",
    )
    trainer_path.write_text(trainer, encoding="utf-8")
    temporary.replace(runtime)
    contract_path.write_text(
        json.dumps({name: sha256_file(runtime / name) for name in files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime


def build_configs(source: Path, output_dir: Path, checkpoint: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    fresh = yaml.safe_load(source.read_text(encoding="utf-8"))
    fresh.update(
        {
            "output_dir": str(output_dir),
            "resume_from_checkpoint": None,
            "seed": seed,
            "data_seed": seed,
        }
    )
    if fresh.get("num_train_epochs") != 2 or fresh.get("save_strategy") != "epoch":
        raise RuntimeError("source two-epoch training contract changed")
    resume = dict(fresh)
    resume["resume_from_checkpoint"] = str(checkpoint)
    return fresh, resume


def prepare_run(stable_root: Path, run_id: str, seed: int, early_stop: bool, support_dir: Path) -> dict[str, Any]:
    seed = validate_seed(seed)
    validate_run_id(run_id, seed)
    if not isinstance(early_stop, bool):
        raise ValueError("early_stop must be boolean")
    storage = ensure_capacity(stable_root)
    runtime = prepare_seed_runtime(stable_root, support_dir)
    run_dir = stable_root / "seed_runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse seed run: {run_id}")
    (run_dir / "evidence_initial").mkdir(parents=True)
    (run_dir / "evidence_resume").mkdir()
    output_dir = run_dir / "output"
    checkpoint = output_dir / f"checkpoint-{MIDPOINT_STEP}"
    source_config = stable_root / "runs/STABLE553-A/training_config.yaml"
    fresh, resume = build_configs(source_config, output_dir, checkpoint, seed)
    fresh_path = run_dir / "training_config_fresh.yaml"
    resume_path = run_dir / "training_config_resume.yaml"
    fresh_path.write_text(yaml.safe_dump(fresh, sort_keys=False, allow_unicode=False), encoding="utf-8")
    resume_path.write_text(yaml.safe_dump(resume, sort_keys=False, allow_unicode=False), encoding="utf-8")
    manifest = {
        "status": "READY",
        "experiment_kind": "fresh_seed",
        "run_id": run_id,
        "seed": seed,
        "start_mode": "base",
        "requested_epochs": 1 if early_stop else 2,
        "early_stop_epoch1": early_stop,
        "start_step": 0,
        "initial_stop_step": MIDPOINT_STEP if early_stop else FINAL_STEP,
        "final_stop_step": FINAL_STEP,
        "scheduler_horizon": FINAL_STEP,
        "world_size": 4,
        "runtime": str(runtime),
        "base_model": json.loads((stable_root / "manifest.json").read_text(encoding="utf-8"))["base_model"],
        "fresh_config": str(fresh_path),
        "fresh_config_sha256": sha256_file(fresh_path),
        "resume_config": str(resume_path),
        "resume_config_sha256": sha256_file(resume_path),
        "output_dir": str(output_dir),
        "resume_checkpoint": str(checkpoint),
        "storage": storage,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_and_verify(stable_root: Path, run_id: str, support_dir: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = stable_root / "seed_runs" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    seed = validate_seed(manifest["seed"])
    validate_run_id(run_id, seed)
    if manifest.get("experiment_kind") != "fresh_seed" or manifest.get("scheduler_horizon") != FINAL_STEP:
        raise RuntimeError("seed manifest contract mismatch")
    if Path(manifest["runtime"]) != prepare_seed_runtime(stable_root, support_dir):
        raise RuntimeError("seed runtime path contract mismatch")
    for key in ("fresh", "resume"):
        path = Path(manifest[f"{key}_config"])
        if sha256_file(path) != manifest[f"{key}_config_sha256"]:
            raise RuntimeError(f"{key} config changed after preparation")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected_resume = None if key == "fresh" else manifest["resume_checkpoint"]
        expected = {
            "output_dir": manifest["output_dir"],
            "resume_from_checkpoint": expected_resume,
            "seed": seed,
            "data_seed": seed,
            "num_train_epochs": 2,
        }
        mismatch = {name: (config.get(name), value) for name, value in expected.items() if config.get(name) != value}
        if mismatch:
            raise RuntimeError(f"{key} config contract mismatch: {mismatch}")
    ensure_capacity(stable_root)
    return run_dir, manifest


def verify_initial(stable_root: Path, run_id: str, support_dir: Path) -> dict[str, Any]:
    _run_dir, manifest = load_and_verify(stable_root, run_id, support_dir)
    if Path(manifest["output_dir"]).exists():
        raise RuntimeError("fresh seed run has already started")
    return {"status": "READY", "run_id": run_id, "phase": "initial"}


def prepare_resume(stable_root: Path, run_id: str, support_dir: Path) -> dict[str, Any]:
    run_dir, manifest = load_and_verify(stable_root, run_id, support_dir)
    if not manifest["early_stop_epoch1"]:
        raise RuntimeError("only an epoch-1 early-stop run can use the resume action")
    if any((run_dir / "evidence_resume").glob("runtime_rank*.json")):
        raise RuntimeError("seed resume evidence already exists")
    if (run_dir / "output" / f"checkpoint-{FINAL_STEP}").exists():
        raise RuntimeError("seed run already has a final checkpoint")
    checkpoint = Path(manifest["resume_checkpoint"])
    contract = checkpoint_contract(checkpoint, MIDPOINT_STEP)
    expected_path = run_dir / "checkpoint_553_sha256.json"
    if expected_path.exists():
        if json.loads(expected_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("checkpoint-553 changed after resume preparation")
    else:
        expected_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "READY", "run_id": run_id, "phase": "resume", "checkpoint_sha256": contract}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--seed", type=int, required=True)
    prepare_parser.add_argument("--early-stop", action="store_true")
    prepare_parser.add_argument("--support-dir", type=Path, default=Path(__file__).resolve().parent)
    for name in ("verify-initial", "prepare-resume"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--run-id", required=True)
        child.add_argument("--support-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_run(args.root, args.run_id, args.seed, args.early_stop, args.support_dir)
    elif args.command == "verify-initial":
        result = verify_initial(args.root, args.run_id, args.support_dir)
    else:
        result = prepare_resume(args.root, args.run_id, args.support_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
