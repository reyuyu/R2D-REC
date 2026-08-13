#!/usr/bin/env python3
"""Build the isolated Alpha CoT-repeat train cache without model loading."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_alpha_cot_repeat05n_4gpu_gc04_2epoch.yaml"
SPLIT = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1")
# `/data` has insufficient headroom for a second ~18 GiB Arrow cache.  This is
# an immutable training artifact, not source data, so keep it on the local
# large-volume cache store while leaving all data/validation artifacts on /data.
TARGET = Path("/app/lf_tokenized/alpha-jiankong-split-v1/tokenized_train98_8k_sid8w8_cot05n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if os.environ.get("GLOBAL_ITEM_WEIGHT") != "8":
        raise SystemExit("GLOBAL_ITEM_WEIGHT=8 is required before importing the production preprocessor.")
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite existing experimental cache: {TARGET}")
    raw_manifest = yaml.safe_load((SPLIT / "alpha_cot_repeat_count_manifest_v1.json").read_text(encoding="utf-8"))
    if raw_manifest.get("source_train_sha256") != _sha256(SPLIT / "train.jsonl"):
        raise SystemExit("train98 SHA differs from the repeat-count manifest; refusing to build a mismatched cache.")
    os.chdir(ROOT)
    sys.argv = ["build_alpha_cot_repeat_cache.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("native_alpha_cot_repeat", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert spec and spec.loader
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    if not native.ALPHA_COT_REPEAT_CONFIG.enabled:
        raise SystemExit("Alpha CoT repeat weighting unexpectedly OFF after loading the dedicated YAML.")
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw.update(
        {
            "dataset": "onereason_alpha_jiankong_train98",
            "dataset_dir": str(SPLIT),
            "tokenized_path": str(TARGET),
            "overwrite_cache": False,
            "do_train": False,
            "do_eval": False,
            "do_predict": False,
            "report_to": "none",
        }
    )
    # The launcher consumed these options at module import.  Keep the generic
    # LLaMA-Factory parser independent of experiment-only keys.
    for key in (
        "rec_pu_enabled", "multitask_pack_ratio_enabled", "rec_candidate_metrics_enabled",
        "alpha_monitor_enabled", "alpha_train_tf_enabled", "alpha_train_tf_interval",
        "alpha_validation_enabled", "alpha_validation_dev_dataset", "alpha_validation_split_dir",
        "alpha_validation_dev_cache", "alpha_validation_probe_cache", "alpha_validation_metrics_path",
        "alpha_dev_probe_enabled", "alpha_dev_probe_interval", "alpha_full_dev_enabled",
        "alpha_full_dev_at_epoch_end", "alpha_cot_repeat_weighting", "alpha_cot_repeat_manifest",
    ):
        raw.pop(key, None)
    native.install_native_patches()
    model_args, data_args, training_args, _, _ = _parse_train_args(raw)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    print(json.dumps({"cache": str(TARGET), "packs": len(module["train_dataset"]), "manifest": native.ALPHA_COT_REPEAT_CONFIG.manifest_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
