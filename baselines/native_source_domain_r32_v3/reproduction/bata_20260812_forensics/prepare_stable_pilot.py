#!/usr/bin/env python3
"""Prepare or verify three isolated BATA-STABLE-V0 checkpoint-553 continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
PRISTINE_SOURCE_SHA256 = {
    "scripts/train_native_source_domain_r32_v3.py": "91b9bfb360cfe2b0bfef8d333dcace433a99098c1facc0c898f1c7db117f8a10",
    "rec_pu/sid8_rec_pu_integration.py": "b272923dbbc6e744ade2d5e5f1e3007520f27ceaf1b563a32e5fb355a32944f4",
}
LLAMAFACTORY_HEAD = "01398eb18dd475a6e27c36f15b970aeacf0d4a60"
LLAMAFACTORY_DIFF_SHA256 = "1eabb66650106e6e56586544e5b330e8ff6a1e4863da889304e52e72c81f5ec3"
LLAMAFACTORY_PYTHON_TREE_SHA256 = "6f7bce5478bc164279cc3aff25e4f3c1e4e665bb0bb3b9ebcdd0279c74766e8e"
LABELS = ("STABLE560-A", "STABLE560-B", "STABLE560-C")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(checkpoint: Path) -> None:
    actual = {name: sha256_file(checkpoint / name) for name in CHECKPOINT_SHA256}
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(f"historical checkpoint contract mismatch: {actual}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if (state.get("global_step"), state.get("max_steps")) != (553, 1106):
        raise RuntimeError("historical checkpoint trainer-state contract mismatch")


def verify_pristine_source(source: Path) -> None:
    actual = {name: sha256_file(source / name) for name in PRISTINE_SOURCE_SHA256}
    if actual != PRISTINE_SOURCE_SHA256:
        raise RuntimeError(f"recovered source contract mismatch: {actual}")


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*.py") if "__pycache__" not in item.parts):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def llamafactory_contract(source: Path) -> dict[str, str]:
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(source), "diff", "--binary", "HEAD", "--", "src"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    result = {
        "git_head": head,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "python_tree_sha256": sha256_tree(source / "src/llamafactory"),
    }
    expected = {
        "git_head": LLAMAFACTORY_HEAD,
        "git_diff_sha256": LLAMAFACTORY_DIFF_SHA256,
        "python_tree_sha256": LLAMAFACTORY_PYTHON_TREE_SHA256,
    }
    if result != expected:
        raise RuntimeError(f"LLaMAFactory source contract mismatch: {result}")
    return result


def runtime_hashes(runtime: Path) -> dict[str, str]:
    names = (
        "scripts/train_native_source_domain_r32_v3.py",
        "scripts/stable_pilot_runtime.py",
        "rec_pu/sid8_rec_pu_integration.py",
        "scripts/run_native_source_domain_r32_v3.sh",
        "config/train_bata_baseline_4gpu_gc04_2epoch.yaml",
    )
    return {name: sha256_file(runtime / name) for name in names}


def apply_integration(runtime: Path, support_dir: Path) -> None:
    shutil.copy2(support_dir / "stable_pilot_runtime.py", runtime / "scripts/stable_pilot_runtime.py")
    subprocess.run(
        ["patch", "-p1", "-i", str(support_dir / "stable_trainer_integration.patch")],
        cwd=runtime,
        check=True,
    )
    trainer = (runtime / "scripts/train_native_source_domain_r32_v3.py").read_text(encoding="utf-8")
    if "replay_forensics" in trainer or "install_replay_instrumentation" in trainer:
        raise RuntimeError("production-like stable runtime contains forensic instrumentation")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.root.exists():
        raise FileExistsError(f"refusing to reuse stable pilot root: {args.root}")
    verify_checkpoint(args.checkpoint)
    verify_pristine_source(args.pristine_source)
    framework = llamafactory_contract(args.llamafactory_source)
    args.root.mkdir(parents=True)
    runtime = args.root / "source_runtime"
    shutil.copytree(args.pristine_source, runtime, copy_function=shutil.copy2)
    apply_integration(runtime, args.support_dir)

    source_config = runtime / "config/train_bata_baseline_4gpu_gc04_2epoch.yaml"
    expected_path = args.root / "expected_checkpoint_sha256.json"
    expected_path.write_text(json.dumps(CHECKPOINT_SHA256, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(args.support_dir / "stable560_contract.json", args.root / "stable560_contract.json")
    runs: dict[str, Any] = {}
    for label in LABELS:
        run_dir = args.root / "runs" / label
        checkpoint = run_dir / "output/checkpoint-553"
        (run_dir / "evidence").mkdir(parents=True)
        shutil.copytree(args.checkpoint, checkpoint, copy_function=shutil.copy2)
        verify_checkpoint(checkpoint)
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        config["output_dir"] = str(run_dir / "output")
        config["resume_from_checkpoint"] = str(checkpoint)
        config["tokenized_path"] = str(args.tokenized_path)
        config_path = run_dir / "training_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
        )
        runs[label] = {
            "run_dir": str(run_dir),
            "config": str(config_path),
            "checkpoint_553": str(checkpoint),
        }

    manifest = {
        "recipe": "BATA-STABLE-V0",
        "historical_checkpoint": str(args.checkpoint),
        "historical_checkpoint_sha256": CHECKPOINT_SHA256,
        "pristine_source": str(args.pristine_source),
        "pristine_source_sha256": PRISTINE_SOURCE_SHA256,
        "runtime_source": str(runtime),
        "runtime_source_sha256": runtime_hashes(runtime),
        "llamafactory_source": str(args.llamafactory_source),
        "llamafactory_contract": framework,
        "tokenized_path": str(args.tokenized_path),
        "labels": list(LABELS),
        "runs": runs,
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_root(args.root)
    return manifest


def verify_root(root: Path, label: str | None = None) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    runtime = Path(manifest["runtime_source"])
    if runtime_hashes(runtime) != manifest["runtime_source_sha256"]:
        raise RuntimeError("runtime source changed after preparation")
    if llamafactory_contract(Path(manifest["llamafactory_source"])) != manifest["llamafactory_contract"]:
        raise RuntimeError("LLaMAFactory source changed after preparation")
    labels = (label,) if label else LABELS
    for item in labels:
        if item not in LABELS:
            raise RuntimeError(f"unknown stable pilot label: {item}")
        run = manifest["runs"][item]
        verify_checkpoint(Path(run["checkpoint_553"]))
        config = yaml.safe_load(Path(run["config"]).read_text(encoding="utf-8"))
        if config.get("num_train_epochs") != 2 or config.get("gradient_accumulation_steps") != 16:
            raise RuntimeError("stable training math/config contract mismatch")
        if config.get("flash_attn") != "fa2" or config.get("enable_liger_kernel") is not True:
            raise RuntimeError("stable FA2/Liger contract mismatch")
        if config.get("resume_from_checkpoint") != run["checkpoint_553"]:
            raise RuntimeError("stable resume path mismatch")
    return {"status": "READY", "root": str(root), "labels": list(labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--pristine-source", type=Path, required=True)
    prepare_parser.add_argument("--tokenized-path", type=Path, required=True)
    prepare_parser.add_argument(
        "--llamafactory-source", type=Path, default=Path("/data/reference/llamafactory-01398eb")
    )
    prepare_parser.add_argument("--support-dir", type=Path, default=Path(__file__).resolve().parent)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--label", choices=LABELS)
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else verify_root(args.root, args.label)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
