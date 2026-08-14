#!/usr/bin/env python3
"""Tokenize/pack the alpha split without loading a model or running a forward."""

from __future__ import annotations

import json
import os
from pathlib import Path
import importlib.util

import yaml


ROOT = Path("/data/baselines/native_source_domain_r32_v3")
CONFIG = ROOT / "config/train_alpha_jiankong_monitor_only_4gpu_gc04_2epoch.yaml"
SPLIT_DIR = Path("/data/lf_data_versions/alltrain/alpha-jiankong-split-v1")


def build(dataset_name: str, target: Path) -> int:
    # The native patch supplies the exact production converter and 8K neat
    # packing semantics, but this script deliberately stops before model load.
    os.chdir(ROOT)
    import sys

    sys.argv = ["audit_alpha_split_packing.py", str(CONFIG)]
    native_spec = importlib.util.spec_from_file_location("native_alpha_audit", ROOT / "scripts/train_native_source_domain_r32_v3.py")
    assert native_spec and native_spec.loader
    native = importlib.util.module_from_spec(native_spec)
    sys.modules[native_spec.name] = native
    native_spec.loader.exec_module(native)
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw.update(
        {
            "dataset": dataset_name,
            "dataset_dir": str(SPLIT_DIR),
            "tokenized_path": str(target),
            "overwrite_cache": False,
            "do_train": False,
            "do_eval": False,
            "do_predict": False,
            "report_to": "none",
        }
    )
    for key in (
        "rec_pu_enabled",
        "multitask_pack_ratio_enabled",
        "rec_candidate_metrics_enabled",
        "alpha_monitor_enabled",
        "alpha_train_tf_enabled",
        "alpha_train_tf_interval",
    ):
        raw.pop(key, None)
    native.install_native_patches()
    model_args, data_args, training_args, _, _ = _parse_train_args(raw)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    rows = len(module["train_dataset"])
    print(json.dumps({"dataset": dataset_name, "tokenized_path": str(target), "packed_rows": rows}, sort_keys=True), flush=True)
    return rows


def main() -> None:
    targets = {
        "onereason_alpha_jiankong_train98": SPLIT_DIR / "tokenized_train98_8k",
        "onereason_alpha_jiankong_dev2": SPLIT_DIR / "tokenized_dev2_8k",
    }
    result = {name: build(name, target) for name, target in targets.items()}
    (SPLIT_DIR / "packing_audit.json").write_text(
        json.dumps(
            {
                "kind": "data_only_8k_neat_packing_audit",
                "model_loaded": False,
                "forward_runs": 0,
                "packed_rows": result,
                "optimizer_steps_per_train_epoch_global_batch64": (result["onereason_alpha_jiankong_train98"] + 63) // 64,
                "dev_packed_batches_four_gpu": (result["onereason_alpha_jiankong_dev2"] + 3) // 4,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
