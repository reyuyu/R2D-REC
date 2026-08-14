#!/usr/bin/env python3
"""Build an isolated mini-cot cache from the Alpha-mini baseline dataset."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import yaml


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_mini_cot_v1_4gpu_gc04_2epoch.yaml"
SOURCE_DIR = Path("/data/lf_data_versions/alltrain/alpha_mini_v1")
TARGET = SOURCE_DIR / "tokenized_alpha_mini_v1_train_8k_sid8w8_cot05n"


def main() -> None:
    if os.environ.get("GLOBAL_ITEM_WEIGHT") != "8":
        raise SystemExit("GLOBAL_ITEM_WEIGHT=8 must be exported before preprocessing import")
    if TARGET.exists():
        raise SystemExit(f"refusing to overwrite existing cache: {TARGET}")
    manifest = SOURCE_DIR / "mini_cot_repeat_count_manifest_v1.json"
    if not manifest.is_file():
        raise SystemExit(f"missing mini manifest: {manifest}")
    os.chdir(ROOT)
    sys.argv = ["build_mini_cot_repeat_cache.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("native_mini_cot_cache", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    if not spec or not spec.loader:
        raise SystemExit("failed to load native training module")
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    if not native.ALPHA_COT_REPEAT_CONFIG.enabled:
        raise SystemExit("mini-cot weighting is unexpectedly disabled")
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw.update({
        "dataset": "onereason_alpha_mini_v1",
        "dataset_dir": str(SOURCE_DIR),
        "tokenized_path": str(TARGET),
        "overwrite_cache": False,
        "do_train": False,
        "do_eval": False,
        "do_predict": False,
        "report_to": "none",
    })
    for key in tuple(raw):
        if key.startswith("alpha_") or key in {"rec_pu_enabled", "multitask_pack_ratio_enabled", "rec_candidate_metrics_enabled"}:
            raw.pop(key, None)
    native.install_native_patches()
    model_args, data_args, training_args, _, _ = _parse_train_args(raw)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    print({"cache": str(TARGET), "packs": len(module["train_dataset"]), "manifest": str(manifest)})


if __name__ == "__main__":
    main()
