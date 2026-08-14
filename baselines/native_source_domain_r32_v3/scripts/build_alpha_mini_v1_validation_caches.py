#!/usr/bin/env python3
"""Build independent native SID8 caches for Alpha's immutable dev and probe."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import yaml


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_alpha_mini_v1_4gpu_gc04_2epoch.yaml"
SPLIT = Path("/data/lf_data_versions/alltrain/alpha_mini_v1_validation_filtered_v1")
TARGETS = (
    ("onereason_alpha_mini_v1_dev_filtered", SPLIT, SPLIT / "tokenized_alpha_mini_v1_dev_8k_sid8w8"),
    ("onereason_alpha_mini_v1_probe_filtered", SPLIT / "probe_v1", SPLIT / "tokenized_alpha_mini_v1_probe_8k_sid8w8"),
)


def _native():
    os.chdir(ROOT)
    sys.argv = ["build_alpha_mini_v1_validation_caches.py", str(CONFIG)]
    spec = importlib.util.spec_from_file_location("native_alpha_mini_validation_cache", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert spec and spec.loader
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    native.install_native_patches()


def main() -> None:
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    _native()
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for dataset, dataset_dir, cache in TARGETS:
        if cache.exists():
            raise RuntimeError(f"Refusing to overwrite independent Mini validation cache: {cache}")
        raw = dict(base)
        raw.update({"dataset": dataset, "dataset_dir": str(dataset_dir), "tokenized_path": str(cache), "do_train": False, "do_eval": False, "do_predict": False})
        for key in tuple(raw):
            if key.startswith("alpha_") or key in {"rec_pu_enabled", "multitask_pack_ratio_enabled", "rec_candidate_metrics_enabled"}:
                raw.pop(key)
        model_args, data_args, training_args, _, _ = _parse_train_args(raw)
        tokenizer_module = load_tokenizer(model_args)
        template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
        module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
        print(f"CACHE_BUILT dataset={dataset} packed_rows={len(module['train_dataset'])} cache={cache}")


if __name__ == "__main__":
    main()
