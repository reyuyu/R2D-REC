#!/usr/bin/env python3
"""Prepare and fail-closed verify two fresh-base BATA-STABLE-V0 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


LABELS = ("STABLE553-A", "STABLE553-B")
SOURCE_SHA256 = {
    "scripts/train_native_source_domain_r32_v3.py": "91b9bfb360cfe2b0bfef8d333dcace433a99098c1facc0c898f1c7db117f8a10",
    "rec_pu/sid8_rec_pu_integration.py": "b272923dbbc6e744ade2d5e5f1e3007520f27ceaf1b563a32e5fb355a32944f4",
    "scripts/run_native_source_domain_r32_v3.sh": "28534965b0b3e96fb77401306bc73d440702512bd0edf152f428d4cad66b7f2b",
    "config/train_bata_baseline_4gpu_gc04_2epoch.yaml": "ff6b589b16c075b354569033159ddfa1c1cf0978dc55a5fe909a2d3e068d8445",
}
BASE_MODEL_SHA256 = {
    ".gitattributes": "a5e36d5e68449b77c4b6693aab13c4b264907536203af2205f37b52fa00346c4",
    "Modelfile": "fa70a865283a98fd578564269d038dbc087080f8e2900879db262c210d1f2d6a",
    "README.md": "133ab3c62a70544db75490a98f1c61a319b1f6856faaede0acb7361089a7f015",
    "added_tokens.json": "ef84f6a200be4d051b58efd0112c6ae967befb2a7b07f88a38d61decd3ceab36",
    "config.json": "1788f05b25f00d9b02610c2f5a922c50d2611ca918e036acdfed94cd0d51392a",
    "generation_config.json": "427565fe9c01ac08f4a0baaeb536322779193d8d28ba723831b49ed5ae778587",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00004.safetensors": "6d041c9244bd1f2221a3d89c9b75274d197dbe61f9333844032d129a9e0ec6c3",
    "model-00002-of-00004.safetensors": "f463b87721fdd488dc9d0e2d1027d426c4b00eb47bc9dedb53b6e763832e94d8",
    "model-00003-of-00004.safetensors": "380878c8c7269ec922fd24e55fc92b2ea62a0a6536a2748d20bdbcc2a8272571",
    "model-00004-of-00004.safetensors": "8e1a537c8e965de9a31334113c8254e6a8d4e3a60096f4de3d8fe6803d97c52f",
    "model.safetensors.index.json": "3a554d5309dc8dac8bf18255dc1017075354a52e99a9fc059b77d49e43661ce0",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer.json": "e68ed2927d899c8592126ec1e8e463983ff878e9763dac94109e0b7cba7e3859",
    "tokenizer_config.json": "5059344eb9660ec6fc8d8bc704f326b9c033318d6bd5131c26c2c77d5057863a",
    "vocab.json": "a5cdac1b456c85d1937cdcfc07b94469f390f1e6f4849d986ce47774d84d7db0",
}
BASE_MANIFEST_SHA256 = "bd726e3bc8507870441cdb2152dd1bf2bfca892cbdf4221dc659c266bf271031"
DATASET_SHA256 = {
    "manifest.json": "83f6e1d7e48212e765591b5f12f87ff0485bc0f78dbc3ccf8371a097f04ed1ca",
    "dataset_info.json": "84dab075839e2dda30280182cad1071f1770ee149ff03df5f0ac53cbaa3106af",
    "onereason_bata_baseline.jsonl": "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca",
}
CACHE_MANIFEST_SHA256 = "a0e2ab69b546bcef3d67d1e9fa3128ea077213160aad5165b351f69a447d679f"
LLAMAFACTORY_HEAD = "01398eb18dd475a6e27c36f15b970aeacf0d4a60"
LLAMAFACTORY_DIFF_SHA256 = "1eabb66650106e6e56586544e5b330e8ff6a1e4863da889304e52e72c81f5ec3"
LLAMAFACTORY_PYTHON_TREE_SHA256 = "6f7bce5478bc164279cc3aff25e4f3c1e4e665bb0bb3b9ebcdd0279c74766e8e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_sha_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name.strip().lstrip("*")] = digest
    return result


def verify_base_model(base_model: Path, sha_manifest: Path) -> dict[str, str]:
    if sha256_file(sha_manifest) != BASE_MANIFEST_SHA256:
        raise RuntimeError("base SHA manifest contract mismatch")
    if parse_sha_manifest(sha_manifest) != BASE_MODEL_SHA256:
        raise RuntimeError("base SHA manifest entries mismatch")
    actual = {name: sha256_file(base_model / name) for name in BASE_MODEL_SHA256}
    if actual != BASE_MODEL_SHA256:
        raise RuntimeError("base model file contract mismatch")
    return actual


def verify_source(source: Path) -> dict[str, str]:
    actual = {name: sha256_file(source / name) for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        raise RuntimeError("recovered source contract mismatch")
    return actual


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    actual = {name: sha256_file(dataset_dir / name) for name in DATASET_SHA256}
    if actual != DATASET_SHA256:
        raise RuntimeError("BATA dataset contract mismatch")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("records") != 222001 or manifest.get("sha256") != DATASET_SHA256[
        "onereason_bata_baseline.jsonl"
    ]:
        raise RuntimeError("BATA dataset row-count contract mismatch")
    return {"sha256": actual, "segment_count": 222001}


def verify_cache(tokenized_path: Path, *, hash_shards: bool) -> dict[str, Any]:
    metadata_path = tokenized_path / "HISTORICAL_CACHE_MOUNT.json"
    if sha256_file(metadata_path) != CACHE_MANIFEST_SHA256:
        raise RuntimeError("historical packed-cache manifest mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS" or metadata.get("row_count") != 35380:
        raise RuntimeError("historical packed-cache row contract mismatch")
    if metadata.get("shard_count") != 16 or len(metadata.get("shards", [])) != 16:
        raise RuntimeError("historical packed-cache shard contract mismatch")
    for shard in metadata["shards"]:
        path = tokenized_path / "train" / Path(shard["mount"]).name
        if not path.is_file() or path.stat().st_size != shard["source_size"]:
            raise RuntimeError(f"historical packed-cache shard size mismatch: {path.name}")
        if hash_shards and sha256_file(path) != shard["source_sha256"]:
            raise RuntimeError(f"historical packed-cache shard SHA mismatch: {path.name}")
    return {"manifest_sha256": CACHE_MANIFEST_SHA256, "packed_row_count": 35380}


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
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(source), "diff", "--binary", "HEAD", "--", "src"],
        check=True, stdout=subprocess.PIPE,
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
        raise RuntimeError("LLaMAFactory source contract mismatch")
    return result


def apply_integration(runtime: Path, support_dir: Path) -> None:
    for name in ("stable_pilot_runtime.py", "stable553_runtime.py"):
        shutil.copy2(support_dir / name, runtime / "scripts" / name)
    subprocess.run(
        ["patch", "-p1", "-i", str(support_dir / "stable553_trainer_integration.patch")],
        cwd=runtime, check=True,
    )
    trainer = (runtime / "scripts/train_native_source_domain_r32_v3.py").read_text(encoding="utf-8")
    forbidden = ("replay_forensics", "install_replay_instrumentation", "register_comm_hook")
    if any(item in trainer for item in forbidden):
        raise RuntimeError("production-like runtime contains forensic instrumentation")


def runtime_hashes(runtime: Path) -> dict[str, str]:
    names = (
        "scripts/train_native_source_domain_r32_v3.py",
        "scripts/stable_pilot_runtime.py",
        "scripts/stable553_runtime.py",
        "rec_pu/sid8_rec_pu_integration.py",
        "scripts/run_native_source_domain_r32_v3.sh",
        "config/train_bata_baseline_4gpu_gc04_2epoch.yaml",
    )
    return {name: sha256_file(runtime / name) for name in names}


def verify_training_config(
    config: dict[str, Any], *, base_model: Path, tokenized_path: Path, output_dir: Path
) -> None:
    exact = {
        "model_name_or_path": str(base_model),
        "flash_attn": "fa2",
        "enable_liger_kernel": True,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "cutoff_len": 8192,
        "packing": True,
        "neat_packing": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.0002,
        "num_train_epochs": 2,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "bf16": True,
        "pure_bf16": True,
        "seed": 20260806,
        "rec_pu_enabled": False,
        "multitask_pack_ratio_enabled": False,
        "tokenized_path": str(tokenized_path),
        "output_dir": str(output_dir),
        "resume_from_checkpoint": None,
    }
    mismatch = {key: (config.get(key), expected) for key, expected in exact.items() if config.get(key) != expected}
    if mismatch:
        raise RuntimeError(f"stable553 training config contract mismatch: {mismatch}")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.root.exists():
        raise FileExistsError(f"refusing to reuse stable553 root: {args.root}")
    source_contract = verify_source(args.pristine_source)
    base_contract = verify_base_model(args.base_model, args.base_sha_manifest)
    dataset_contract = verify_dataset(args.dataset_dir)
    cache_contract = verify_cache(args.tokenized_path, hash_shards=True)
    framework = llamafactory_contract(args.llamafactory_source)

    args.root.mkdir(parents=True)
    runtime = args.root / "source_runtime"
    shutil.copytree(args.pristine_source, runtime, copy_function=shutil.copy2)
    apply_integration(runtime, args.support_dir)
    source_config = runtime / "config/train_bata_baseline_4gpu_gc04_2epoch.yaml"

    runs: dict[str, Any] = {}
    for label in LABELS:
        run_dir = args.root / "runs" / label
        (run_dir / "evidence").mkdir(parents=True)
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        config["model_name_or_path"] = str(args.base_model)
        config["dataset_dir"] = str(args.dataset_dir)
        config["tokenized_path"] = str(args.tokenized_path)
        config["output_dir"] = str(run_dir / "output")
        config["resume_from_checkpoint"] = None
        config_path = run_dir / "training_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
        verify_training_config(
            config, base_model=args.base_model, tokenized_path=args.tokenized_path,
            output_dir=run_dir / "output",
        )
        runs[label] = {
            "run_dir": str(run_dir),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
        }

    manifest = {
        "recipe": "BATA-STABLE-V0",
        "scope": {"start_step": 0, "stop_step": 553, "scheduler_horizon": 1106},
        "pristine_source": str(args.pristine_source),
        "pristine_source_sha256": source_contract,
        "runtime_source": str(runtime),
        "runtime_source_sha256": runtime_hashes(runtime),
        "base_model": str(args.base_model),
        "base_model_sha256": base_contract,
        "base_sha_manifest_sha256": BASE_MANIFEST_SHA256,
        "dataset_dir": str(args.dataset_dir),
        "dataset_contract": dataset_contract,
        "tokenized_path": str(args.tokenized_path),
        "cache_contract": cache_contract,
        "llamafactory_source": str(args.llamafactory_source),
        "llamafactory_contract": framework,
        "labels": list(LABELS),
        "runs": runs,
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_sha = canonical_json_sha(source_contract)
    lines = [
        f"{BASE_MANIFEST_SHA256} base_contract_sha256",
        f"{source_sha} source_contract_sha256",
        *(f"{runs[label]['config_sha256']} {label}_config_sha256" for label in LABELS),
    ]
    (args.root / "contract.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(args.support_dir / "stable553_contract.json", args.root / "stable553_contract.json")
    verify_root(args.root, hash_base=False)
    return manifest


def verify_root(root: Path, label: str | None = None, *, hash_base: bool = True) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("scope") != {"start_step": 0, "stop_step": 553, "scheduler_horizon": 1106}:
        raise RuntimeError("stable553 scope contract mismatch")
    runtime = Path(manifest["runtime_source"])
    if runtime_hashes(runtime) != manifest["runtime_source_sha256"]:
        raise RuntimeError("runtime source changed after preparation")
    if verify_source(Path(manifest["pristine_source"])) != manifest["pristine_source_sha256"]:
        raise RuntimeError("pristine source changed after preparation")
    if hash_base:
        actual = {name: sha256_file(Path(manifest["base_model"]) / name) for name in BASE_MODEL_SHA256}
        if actual != manifest["base_model_sha256"] or actual != BASE_MODEL_SHA256:
            raise RuntimeError("base model changed after preparation")
    verify_dataset(Path(manifest["dataset_dir"]))
    verify_cache(Path(manifest["tokenized_path"]), hash_shards=False)
    if llamafactory_contract(Path(manifest["llamafactory_source"])) != manifest["llamafactory_contract"]:
        raise RuntimeError("LLaMAFactory changed after preparation")
    labels = (label,) if label else LABELS
    for item in labels:
        if item not in LABELS:
            raise RuntimeError(f"unknown stable553 label: {item}")
        run = manifest["runs"][item]
        config_path = Path(run["config"])
        if sha256_file(config_path) != run["config_sha256"]:
            raise RuntimeError("stable553 generated config changed after preparation")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        verify_training_config(
            config, base_model=Path(manifest["base_model"]),
            tokenized_path=Path(manifest["tokenized_path"]),
            output_dir=Path(run["run_dir"]) / "output",
        )
    return {"status": "READY", "root": str(root), "labels": list(labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--pristine-source", type=Path, required=True)
    prepare_parser.add_argument("--base-model", type=Path, required=True)
    prepare_parser.add_argument("--base-sha-manifest", type=Path, required=True)
    prepare_parser.add_argument("--dataset-dir", type=Path, required=True)
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
