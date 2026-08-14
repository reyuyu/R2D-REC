#!/usr/bin/env python3
"""CPU-only build and static audit for the PackRatio packed Arrow dataset."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys

import yaml


BASELINE_ROOT = Path("/data/baselines/native_source_domain_r32_v3")
REFERENCE_SRC = Path("/data/reference/llamafactory-01398eb/src")
CONFIG_PATH = BASELINE_ROOT / "config" / "train_rec_pu_beta_material_aligned_r32_b005_2epoch_packratio_20452015.yaml"

sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(REFERENCE_SRC))
sys.argv = ["pack_ratio_dataset_audit", str(CONFIG_PATH)]

native_path = BASELINE_ROOT / "scripts" / "train_native_source_domain_r32_v3.py"
spec = importlib.util.spec_from_file_location("native_pack_ratio_audit", native_path)
native = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(native)

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.data.converter import AlpacaDatasetConverter  # noqa: E402
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_tokenizer  # noqa: E402


def main() -> None:
    os.environ.setdefault(
        "MATERIAL_DOMAIN_MANIFEST", "/data/lf_data_versions/alltrain/BETA_material_aligned_v1/manifest.json"
    )
    os.environ.setdefault("GLOBAL_ITEM_WEIGHT", "8")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    # The first audit invocation intentionally builds the new Arrow schema.
    # Follow-up reads must not rerun the expensive tokenizer/packing pass.
    config["overwrite_cache"] = False
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)

    AlpacaDatasetConverter.__call__ = native._convert_with_source
    PackedSupervisedDatasetProcessor.preprocess_dataset = native._preprocess_packed_with_weights

    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    dataset_module = get_dataset(
        template, model_args, data_args, training_args, stage="sft", **tokenizer_module
    )
    dataset = dataset_module["train_dataset"]
    if "pack_task_id" not in dataset.column_names:
        raise RuntimeError("PackRatio cache build completed without `pack_task_id`.")
    if len(dataset) != 33616:
        raise RuntimeError("Unexpected packed dataset length: {}".format(len(dataset)))

    pack_task_ids = dataset["pack_task_id"]
    pools = Counter(pack_task_ids)
    if sum(pools.values()) != len(dataset):
        raise RuntimeError("pack_task_id pool counts do not sum to packed dataset length.")

    sampler = native.PackRatioSampler(pack_task_ids, native.PACK_RATIO_CONFIG.target_ratios, training_args.seed)
    audits = []
    seen = [set() for _ in native.TASK_NAMES]
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        plan, plan_tasks = sampler.build_plan()
        current = [set() for _ in native.TASK_NAMES]
        for index, task_id in zip(plan, plan_tasks):
            current[task_id].add(index)
            seen[task_id].add(index)
        audits.append(
            {
                "epoch": epoch,
                "fingerprint": sampler.fingerprint(),
                "planned": {
                    native.TASK_NAMES[task_id]: int(plan_tasks.count(task_id)) for task_id in range(4)
                },
                "unique": {native.TASK_NAMES[task_id]: len(current[task_id]) for task_id in range(4)},
                "repeat": {
                    native.TASK_NAMES[task_id]: int(plan_tasks.count(task_id) - len(current[task_id]))
                    for task_id in range(4)
                },
                "cumulative_coverage": {
                    native.TASK_NAMES[task_id]: len(seen[task_id]) / len(sampler.pools[task_id])
                    for task_id in range(4)
                },
            }
        )

    print(
        "PACK_RATIO_REAL_DATASET_AUDIT="
        + json.dumps(
            {
                "columns": dataset.column_names,
                "packed_rows": len(dataset),
                "pools": {native.TASK_NAMES[task_id]: pools[task_id] for task_id in range(4)},
                "target_counts": dict(zip(native.TASK_NAMES, sampler.quotas)),
                "epochs": audits,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
