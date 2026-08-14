#!/usr/bin/env python3
"""Pack the fixed alpha dev probe with the exact native 8K route, no model."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_alpha_jiankong_monitor_validation_4gpu_gc04_2epoch.yaml"
PROBE_DIR = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/probe_v1")
CACHE = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1/tokenized_dev_probe_v1_8k")


def main() -> None:
    if CACHE.exists():
        raise RuntimeError(f"Refusing to overwrite existing probe cache: {CACHE}")
    os.chdir(ROOT)
    sys.argv = ["audit_alpha_dev_probe_packing.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("native_alpha_probe", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert spec and spec.loader
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw.update({
        "dataset": "onereason_alpha_jiankong_dev_probe_v1",
        "dataset_dir": str(PROBE_DIR), "tokenized_path": str(CACHE),
        "overwrite_cache": False, "do_train": False, "do_eval": False, "do_predict": False, "report_to": "none",
    })
    for key in tuple(raw):
        if key.startswith("alpha_") or key in {
            "rec_pu_enabled", "multitask_pack_ratio_enabled", "rec_candidate_metrics_enabled",
        }:
            raw.pop(key)
    native.install_native_patches()
    model_args, data_args, training_args, _, _ = _parse_train_args(raw)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    rows = len(module["train_dataset"])
    (PROBE_DIR / "packing_audit.json").write_text(json.dumps({"packed_sequences": rows, "cache": str(CACHE), "model_loaded": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"packed_sequences": rows, "cache": str(CACHE)}, sort_keys=True))


if __name__ == "__main__":
    main()
