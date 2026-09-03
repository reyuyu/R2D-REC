#!/usr/bin/env python3
"""Prepare isolated, seed-labelled checkpoint-553 to 1106 continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


RUN_ID_RE = re.compile(r"^EPOCH2-S([0-9]{1,10})-([0-9]{8}-[0-9]{6})$")
MAX_SEED = 2**31 - 1
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


def checkpoint_contract(checkpoint: Path) -> dict[str, str]:
    trainer = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if (trainer.get("global_step"), trainer.get("max_steps")) != (553, 1106):
        raise RuntimeError("source checkpoint is not the expected step-553 midpoint")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"source checkpoint is incomplete: {missing}")
    return {name: sha256_file(checkpoint / name) for name in REQUIRED_CHECKPOINT_FILES}


def prepare_epoch2_runtime(stable_root: Path, support_dir: Path) -> Path:
    source = stable_root / "source_runtime"
    runtime = stable_root / "source_runtime_epoch2"
    contract_path = stable_root / "epoch2_runtime_contract.json"
    files = (
        "scripts/train_native_source_domain_r32_v3.py",
        "scripts/stable_epoch2_runtime.py",
    )
    if runtime.exists():
        if not contract_path.is_file():
            raise RuntimeError("epoch-2 runtime exists without its contract")
        expected = json.loads(contract_path.read_text(encoding="utf-8"))
        actual = {name: sha256_file(runtime / name) for name in files}
        if actual != expected:
            raise RuntimeError("epoch-2 runtime changed after preparation")
        return runtime

    temporary = stable_root / ".source_runtime_epoch2.tmp"
    if temporary.exists():
        raise RuntimeError("incomplete epoch-2 runtime preparation exists")
    shutil.copytree(source, temporary, copy_function=shutil.copy2)
    shutil.copy2(support_dir / "stable_epoch2_runtime.py", temporary / "scripts/stable_epoch2_runtime.py")
    trainer_path = temporary / "scripts/train_native_source_domain_r32_v3.py"
    trainer = trainer_path.read_text(encoding="utf-8")
    import_anchor = "from stable553_runtime import stable553_callbacks\n"
    callback_anchor = "    callbacks.extend(stable553_callbacks())\n"
    if trainer.count(import_anchor) != 1 or trainer.count(callback_anchor) != 1:
        raise RuntimeError("trainer integration anchors are not unique")
    trainer = trainer.replace(
        import_anchor,
        import_anchor + "from stable_epoch2_runtime import stable_epoch2_callbacks\n",
    ).replace(
        callback_anchor,
        callback_anchor + "    callbacks.extend(stable_epoch2_callbacks())\n",
    )
    trainer_path.write_text(trainer, encoding="utf-8")
    temporary.replace(runtime)
    contract = {name: sha256_file(runtime / name) for name in files}
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return runtime


def build_config(source: Path, output_dir: Path, checkpoint: Path, seed: int) -> dict[str, Any]:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["output_dir"] = str(output_dir)
    config["resume_from_checkpoint"] = str(checkpoint)
    config["seed"] = seed
    config["data_seed"] = seed
    if config.get("num_train_epochs") != 2 or config.get("save_strategy") != "epoch":
        raise RuntimeError("source two-epoch training contract changed")
    return config


def prepare_run(stable_root: Path, run_id: str, seed: int, support_dir: Path) -> dict[str, Any]:
    seed = validate_seed(seed)
    validate_run_id(run_id, seed)
    source_checkpoint = stable_root / "runs/STABLE553-A/output/checkpoint-553"
    expected_checkpoint = checkpoint_contract(source_checkpoint)
    runtime = prepare_epoch2_runtime(stable_root, support_dir)
    run_dir = stable_root / "epoch2_runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse epoch-2 run: {run_id}")
    (run_dir / "evidence").mkdir(parents=True)
    output_dir = run_dir / "output"
    source_config = stable_root / "runs/STABLE553-A/training_config.yaml"
    config = build_config(source_config, output_dir, source_checkpoint, seed)
    config_path = run_dir / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    expected_path = run_dir / "expected_checkpoint_sha256.json"
    expected_path.write_text(json.dumps(expected_checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY",
        "run_id": run_id,
        "seed": seed,
        "source_label": "STABLE553-A",
        "start_step": 553,
        "stop_step": 1106,
        "scheduler_horizon": 1106,
        "world_size": 4,
        "runtime": str(runtime),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": expected_checkpoint,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output_dir": str(output_dir),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_run(stable_root: Path, run_id: str, support_dir: Path) -> dict[str, Any]:
    run_dir = stable_root / "epoch2_runs" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    seed = validate_seed(manifest["seed"])
    validate_run_id(run_id, seed)
    runtime = prepare_epoch2_runtime(stable_root, support_dir)
    if Path(manifest["runtime"]) != runtime:
        raise RuntimeError("epoch-2 runtime path contract mismatch")
    checkpoint = Path(manifest["source_checkpoint"])
    if checkpoint_contract(checkpoint) != manifest["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint SHA contract mismatch")
    config_path = Path(manifest["config"])
    if sha256_file(config_path) != manifest["config_sha256"]:
        raise RuntimeError("epoch-2 config changed after preparation")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "output_dir": manifest["output_dir"],
        "resume_from_checkpoint": manifest["source_checkpoint"],
        "seed": seed,
        "data_seed": seed,
        "num_train_epochs": 2,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise RuntimeError(f"epoch-2 config contract mismatch: {mismatch}")
    return {"status": "READY", "run_id": run_id, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--run-id", required=True)
        if name == "prepare":
            child.add_argument("--seed", type=int, required=True)
        child.add_argument("--support-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_run(args.root, args.run_id, args.seed, args.support_dir)
    else:
        result = verify_run(args.root, args.run_id, args.support_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
