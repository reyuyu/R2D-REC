"""Merge a validated GRPO adapter into its full-SFT parent and publish it."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_status(path: Path, *, state: str, phase: str, progress: int, message: str, **extra: Any) -> None:
    write_json_atomic(path, {
        "state": state,
        "phase": phase,
        "progress": int(progress),
        "message": message,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    })


def read_token(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if os.name != "nt" and mode & 0o077:
        raise RuntimeError("ModelScope token file must not be accessible by group or others")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("ModelScope token file is empty")
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--visibility", choices=("private", "public"), default="private")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--status-file", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base_model).resolve()
    adapter = Path(args.adapter).resolve()
    work_dir = Path(args.work_dir).resolve()
    status_file = Path(args.status_file).resolve()
    merged_dir = work_dir / "merged-model"
    token = ""
    try:
        update_status(status_file, state="running", phase="validating", progress=5, message="正在核验父模型与 adapter")
        base_weight = base / "model.safetensors"
        adapter_weight = adapter / "adapter_model.safetensors"
        required = [
            base_weight, base / "config.json", adapter_weight,
            adapter / "adapter_config.json", adapter / "lineage.json",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"required merge inputs are missing: {missing}")
        if file_sha256(base_weight) != args.expected_base_sha256:
            raise RuntimeError("full-SFT parent SHA256 mismatch")
        if file_sha256(adapter_weight) != args.expected_adapter_sha256:
            raise RuntimeError("adapter SHA256 mismatch")
        lineage = json.loads((adapter / "lineage.json").read_text(encoding="utf-8"))
        if lineage.get("parent_mode") != "full_model" or lineage.get("parent_base_sha256") != args.expected_base_sha256:
            raise RuntimeError("adapter lineage does not match the full-SFT parent")

        token = read_token(Path(args.token_file).resolve())
        usage = shutil.disk_usage(work_dir.parent)
        required_free = max(base_weight.stat().st_size * 2, 32 * 1024**3)
        if usage.free < required_free:
            raise RuntimeError("insufficient free disk for a verified merged model")

        update_status(status_file, state="running", phase="loading", progress=20, message="正在 CPU 加载完整 SFT 父模型")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        update_status(status_file, state="running", phase="merging", progress=45, message="正在将 GRPO adapter 融合到 SFT 父模型")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model = model.merge_and_unload(safe_merge=True, progressbar=True)

        update_status(status_file, state="running", phase="saving", progress=65, message="正在保存并校验融合权重")
        merged_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=True)
        tokenizer.save_pretrained(merged_dir)
        del tokenizer, model
        gc.collect()

        weight_files = sorted(merged_dir.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError("merged model did not produce safetensors weights")
        weight_hashes = {path.name: file_sha256(path) for path in weight_files}
        public_manifest = {
            "schema": "grpo_merged_model_v1",
            "run_id": args.run_id,
            "checkpoint": args.checkpoint,
            "parent_model_sha256": args.expected_base_sha256,
            "adapter_sha256": args.expected_adapter_sha256,
            "merged_weight_sha256": weight_hashes,
        }
        write_json_atomic(merged_dir / "MERGE_MANIFEST.json", public_manifest)
        (merged_dir / "configuration.json").write_text(
            json.dumps({"framework": "pytorch", "task": "text-generation"}, indent=2),
            encoding="utf-8",
        )

        update_status(status_file, state="running", phase="uploading", progress=82, message="正在上传完整融合模型到 ModelScope")
        from modelscope.hub.api import HubApi

        api = HubApi()
        visibility = 1 if args.visibility == "private" else 5
        if not api.repo_exists(args.model_id, repo_type="model", token=token):
            api.create_model(
                model_id=args.model_id,
                visibility=visibility,
                license="Apache License 2.0",
                token=token,
            )
        api.upload_folder(
            repo_id=args.model_id,
            folder_path=merged_dir,
            repo_type="model",
            token=token,
            commit_message=f"Merge {args.run_id} {args.checkpoint}",
            max_workers=4,
        )
        shutil.rmtree(merged_dir)
        update_status(
            status_file,
            state="completed",
            phase="completed",
            progress=100,
            message="融合模型已上传，服务器临时权重已清理",
            model_url=f"https://modelscope.cn/models/{args.model_id}",
            merged_weight_sha256=weight_hashes,
        )
        token = ""
        return 0
    except Exception as exc:
        message = str(exc)
        if token:
            message = message.replace(token, "[redacted]")
        update_status(
            status_file,
            state="failed",
            phase="failed",
            progress=0,
            message=message[:500],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
