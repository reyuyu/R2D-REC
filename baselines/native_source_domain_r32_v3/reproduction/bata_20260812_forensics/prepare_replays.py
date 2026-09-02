#!/usr/bin/env python3
"""Prepare two isolated recovered-source BATA checkpoint-553 micro replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml


CHECKPOINT_SHA256 = {
    "adapter_model.safetensors": "9c5426a5c329e9ec19a5dddc351c51c498dab4b4ed3a81ce80cfa07dc84cab9e",
    "optimizer.pt": "4806181f0754d954c4c9f30fe38d5b33587582dbb2a710e9571e7d7493213d98",
    "scheduler.pt": "b951d51e69752e0946c62e098ed98dc80e281b252d1eca4f0c996ae08dc7dccb",
    "trainer_state.json": "bacd549ddeaadde8d065d9d6184c68aa1fda88083a1577005fae8a79839e29d8",
    "training_args.bin": "2007c7967597a348c3f8ff68991fff0ab399ce019501a55dbbe3c741da6bb7d2",
    "rng_state_0.pth": "5d7293a18c2d02ee3cd516babc094c680cf849faba4e4352053bccf44a760996",
    "rng_state_1.pth": "b1d4c3eebf0e573130bb904de7c972893cb4fadc5f6a5c4d33a4548714f67196",
    "rng_state_2.pth": "e148a4b7c58775628765cd1617cad9edee5fcddab223723c80dc47ce79f9fe9a",
    "rng_state_3.pth": "e31d3245179273a4f709cfc7b6529dc141494bab2b36d977ac14370970dde64d",
}

RUNTIME_FILES = (
    "scripts/train_native_source_domain_r32_v3.py",
    "rec_pu/sid8_rec_pu_integration.py",
    "scripts/run_native_source_domain_r32_v3.sh",
    "scripts/launch_bata_baseline_4gpu_gc04_2epoch.sh",
    "config/train_bata_baseline_4gpu_gc04_2epoch.yaml",
    "scripts/validate_bata_baseline.py",
    "pack_ratio_sampler.py",
    "scripts/replay_forensics.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE
    ).stdout


def verify_checkpoint(checkpoint: Path) -> None:
    actual = {name: sha256_file(checkpoint / name) for name in CHECKPOINT_SHA256}
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(f"Historical checkpoint SHA mismatch: {actual}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != 553 or state.get("epoch") != 1.0 or state.get("max_steps") != 1106:
        raise RuntimeError(f"Unexpected historical trainer state: {state}")


def prepare_one(
    root: Path,
    label: str,
    checkpoint: Path,
    source_config: Path,
    tokenized_path: Path | None,
) -> dict[str, str]:
    run_dir = root / label
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse replay directory: {run_dir}")
    output_dir = run_dir / "output"
    evidence_dir = run_dir / "evidence"
    copied_checkpoint = output_dir / "checkpoint-553"
    evidence_dir.mkdir(parents=True)
    shutil.copytree(checkpoint, copied_checkpoint, copy_function=shutil.copy2)
    verify_checkpoint(copied_checkpoint)

    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["output_dir"] = str(output_dir)
    config["resume_from_checkpoint"] = str(copied_checkpoint)
    if tokenized_path is not None:
        config["tokenized_path"] = str(tokenized_path)
    config_path = run_dir / "replay_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    expected_path = run_dir / "expected_checkpoint_sha256.json"
    expected_path.write_text(json.dumps(CHECKPOINT_SHA256, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(copied_checkpoint),
        "evidence_dir": str(evidence_dir),
        "config": str(config_path),
        "expected_sha": str(expected_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--llamafactory-source", type=Path, required=True)
    parser.add_argument("--tokenized-path", type=Path)
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Replay label to prepare; repeat for multiple labels.",
    )
    args = parser.parse_args()

    verify_checkpoint(args.checkpoint)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "manifest.json"
    previous_replays = {}
    if manifest_path.is_file():
        previous_replays = json.loads(manifest_path.read_text(encoding="utf-8")).get("replays", {})
    manifest = {
        "historical_checkpoint": str(args.checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_source": str(args.runtime_source),
        "runtime_source_tree_sha256": sha256_tree(args.runtime_source),
        "runtime_file_sha256": {
            name: sha256_file(args.runtime_source / name)
            for name in RUNTIME_FILES
            if (args.runtime_source / name).is_file()
        },
        "llamafactory_source": str(args.llamafactory_source),
        "llamafactory_git_head": git_output(args.llamafactory_source, "rev-parse", "HEAD").decode().strip(),
        "llamafactory_git_diff_sha256": hashlib.sha256(
            git_output(args.llamafactory_source, "diff", "--binary", "HEAD", "--", "src")
        ).hexdigest(),
        "llamafactory_python_tree_sha256": sha256_tree(args.llamafactory_source / "src" / "llamafactory"),
        "replays": previous_replays,
    }
    labels = args.labels or ["RECOVERED553-REPLAY-A", "RECOVERED553-REPLAY-B"]
    for label in labels:
        manifest["replays"][label] = prepare_one(
            args.root, label, args.checkpoint, args.source_config, args.tokenized_path
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
