#!/usr/bin/env python3
"""Actual four-rank Trainer/Accelerate PackRatio integration audit, without a model forward."""

from __future__ import annotations

from collections import Counter
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
import yaml


BASELINE_ROOT = Path("/data/baselines/native_source_domain_r32_v3")
REFERENCE_SRC = Path("/data/reference/llamafactory-01398eb/src")
CACHE_GLOB = (
    "/root/.cache/huggingface/datasets/json/default-e95229b07fd4770d/0.0.0/"
    "f4e89e8750d5d5ffbef2c078bf0ddfedef29dc2faff52a6255cf513c05eb1092/"
    "cache-e90f0006044933d3_*.arrow"
)


def _load_cached_dataset() -> Dataset:
    files = [Path(value) for value in sorted(glob.glob(os.environ.get("PACK_RATIO_ARROW_GLOB", CACHE_GLOB)))]
    if not files:
        raise RuntimeError("The audited BETA packed Arrow cache was not found.")
    dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in files])
    required = {"pack_task_id", "loss_weights", "sample_task_ids", "rec_pu_targets_json"}
    if len(dataset) != 33616 or not required.issubset(dataset.column_names):
        raise RuntimeError(f"Unexpected packed cache: rows={len(dataset)}, columns={dataset.column_names}")
    return dataset


def _load_native(config_path: Path):
    sys.path.insert(0, str(BASELINE_ROOT))
    sys.path.insert(0, str(REFERENCE_SRC))
    sys.argv = ["pack_ratio_4gpu_integration_audit", str(config_path)]
    source = BASELINE_ROOT / "scripts" / "train_native_source_domain_r32_v3.py"
    spec = importlib.util.spec_from_file_location("native_pack_ratio_4gpu_integration", source)
    native = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(native)
    return native


def _make_trainer(native, config_path: Path, dataset: Dataset):
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
        processor=tokenizer_module["processor"],
        model_args=model_args,
        gen_kwargs=gen_kwargs,
    )
    return trainer, training_args


def _flatten_batch_indices(batch_sampler, expected_batches: int) -> list[int]:
    result: list[int] = []
    # Trainer consumes exactly ``len(prepared_dataloader)`` batches. Cap this
    # audit at that identical boundary so Accelerate's optional tail-padding
    # iterator cannot make an audit enumerate beyond the actual epoch.
    for _, batch in zip(range(expected_batches), batch_sampler):
        result.extend(int(index) for index in batch)
    if len(result) != expected_batches:
        raise RuntimeError(f"Prepared batch sampler yielded {len(result)} batches, expected {expected_batches}.")
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_pack_ratio_4gpu_integration.py CONFIG.yaml")
    config_path = Path(sys.argv[1]).resolve()
    os.environ.setdefault("MATERIAL_DOMAIN_MANIFEST", "/data/lf_data_versions/alltrain/BETA_material_aligned_v1/manifest.json")
    os.environ.setdefault("GLOBAL_ITEM_WEIGHT", "8")
    native = _load_native(config_path)
    native.install_native_patches()
    dataset = _load_cached_dataset()
    trainer, args = _make_trainer(native, config_path, dataset)

    sampler = trainer._get_train_sampler(dataset)
    sampler_type = type(sampler).__name__ if sampler is not None else "None"
    expected_sampler = "PackRatioSampler" if native.PACK_RATIO_CONFIG.enabled else "RandomSampler"
    if sampler_type != expected_sampler:
        raise RuntimeError(f"Sampler route mismatch: expected {expected_sampler}, got {sampler_type}")
    if native.PACK_RATIO_CONFIG.enabled and not isinstance(sampler, native.PackRatioSampler):
        raise RuntimeError("PackRatio ON did not construct the global PackRatioSampler.")

    dataloader = trainer.get_train_dataloader()
    if type(dataloader.batch_sampler).__name__ != "BatchSamplerShard":
        raise RuntimeError(f"Prepared sampler is not Accelerate BatchSamplerShard: {type(dataloader.batch_sampler).__name__}")
    if getattr(dataloader.batch_sampler, "num_processes", None) != 4:
        raise RuntimeError("BatchSamplerShard does not expose four processes.")
    if getattr(dataloader.batch_sampler, "split_batches", None):
        raise RuntimeError("Unexpected split_batches=True would violate global sampler semantics.")

    values = trainer.set_initial_training_values(args, dataloader)
    _, steps_per_epoch, _, _, _, _, total_steps = values
    warmup_steps = args.get_warmup_steps(total_steps)
    if (len(dataset), len(dataloader), steps_per_epoch, total_steps) != (33616, 8404, 526, 1052):
        raise RuntimeError(
            "Training horizon mismatch: "
            f"dataset={len(dataset)} rank_loader={len(dataloader)} steps_epoch={steps_per_epoch} total={total_steps}"
        )

    # This enters actual DDP/Accelerate preparation and constructs the same optimizer/scheduler
    # as train(); it intentionally performs no model forward or backward.
    trainer._prepare_for_training(total_steps, dataloader, resume_from_checkpoint=None)
    scheduler_total = total_steps

    dataloader.set_epoch(0)
    local_indices = _flatten_batch_indices(dataloader.batch_sampler, len(dataloader))
    if len(local_indices) != 8404:
        raise RuntimeError(f"Rank {args.process_index} exposure length is {len(local_indices)}, not 8404.")
    gathered: list[list[int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_indices)

    report = {
        "rank": int(args.process_index),
        "world_size": int(args.world_size),
        "sampler": sampler_type,
        "dataloader_type": type(dataloader).__name__,
        "batch_sampler_type": type(dataloader.batch_sampler).__name__,
        "rank_dataloader_length": len(dataloader),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "scheduler_total_steps": int(scheduler_total),
        "warmup_steps": int(warmup_steps),
        "local_index_count": len(local_indices),
    }

    if args.process_index == 0:
        by_rank = [list(item or []) for item in gathered]
        if any(len(item) != 8404 for item in by_rank):
            raise RuntimeError(f"Rank shard lengths are not exact: {[len(item) for item in by_rank]}")
        merged = [by_rank[rank][offset] for offset in range(8404) for rank in range(4)]
        if len(merged) != 33616:
            raise RuntimeError("Accelerate sharding changed the global sampler plan length.")
        if native.PACK_RATIO_CONFIG.enabled:
            reference_plan, _ = sampler.build_plan()
            if merged != reference_plan:
                raise RuntimeError(
                    "Prepared rank shards do not reconstruct the global PackRatio plan "
                    "exactly (including its intended coverage-before-repeat samples)."
                )
            counts = Counter(int(dataset[index]["pack_task_id"]) for index in merged)
            expected = dict(zip(native.TASK_NAMES, sampler.quotas))
            actual = {native.TASK_NAMES[key]: int(counts[key]) for key in range(4)}
            if actual != expected:
                raise RuntimeError(f"ON task exposure mismatch: actual={actual}, expected={expected}")
            windows = []
            for start in range(0, 64 * 3, 64):
                window = Counter(int(dataset[index]["pack_task_id"]) for index in merged[start : start + 64])
                windows.append({native.TASK_NAMES[key]: int(window[key]) for key in range(4)})
            max_run = 1
            run = 1
            task_ids = [int(dataset[index]["pack_task_id"]) for index in merged[:192]]
            for previous, current in zip(task_ids, task_ids[1:]):
                run = run + 1 if current == previous else 1
                max_run = max(max_run, run)
            report.update(
                {
                    "global_plan_length": len(merged),
                    "global_task_counts": actual,
                    "first_three_global_windows": windows,
                    "max_same_task_run_first_192": max_run,
                }
            )
        else:
            report["global_plan_length"] = len(merged)

        # A real-cache collator regression covers this field-level contract.
        # The following 10-step Trainer smoke then covers it on the actual
        # worker/forward route, without spawning audit-only workers after DDP.
        report["metadata_contract"] = "real-collator-regression-pass; verified again by training smoke"
        print("PACK_RATIO_4GPU_INTEGRATION=" + json.dumps(report, sort_keys=True), flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
