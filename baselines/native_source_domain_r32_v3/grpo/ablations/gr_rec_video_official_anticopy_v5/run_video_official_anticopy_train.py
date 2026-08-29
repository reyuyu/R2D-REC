#!/usr/bin/env python3
"""Formal runner contract for Video-only Official Anti-Copy GRPO V5."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

GRPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = GRPO_ROOT / "scripts"
SOURCE_DATASET = Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl")
FORBIDDEN_POSITIVE_DATASET = Path(
    "/data/GRPO/data/grpo_tk_positive_groups_1946_20260829/train.jsonl"
)
SOURCE_DATASET_SHA256 = "791d5696b87d18720fccfff5b4dbbb3c72d996095b8e2bc9d7c7f7c57fdcf3dc"
FINAL_TRAIN_CANONICAL_SHA256 = "c444193bf410882b18b29131b0fb7b2b993d31a3ee58fd1481ec5adf1a6ac8d0"
FINAL_GROUP_ID_LIST_SHA256 = "553df18f586ae8f79a473f67ffd5eb03b4d33af26a341f48f091fbf542418b37"
PARENT_ADAPTER = Path(
    "/root/GRPO-checkpoints/"
    "GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-1500"
)
PARENT_ADAPTER_SHA256 = "3bc818109c8895225133c5cc77cfd1916ad8777292e7865969730708d0b48740"
EXPECTED_RAW_GROUPS = 1549
EXPECTED_AFTER_PROBE4 = 1545
EXPECTED_VIDEO_GROUPS = 549
EXPECTED_TRAIN_GROUPS = 537
EXPECTED_STEPS = EXPECTED_TRAIN_GROUPS * 2
FIXED_PROBE4_IDS = (
    "fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e",
    "6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f",
    "281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700",
    "2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8",
)
HELDOUT_VIDEO_PROBE_IDS = (
    "b95d81c7269ac6b70c3113d88199fecc6d6aba4a05c7cb28dc75c298490fdd70",
    "373b94e4ffdb36f3be1f103fc0b37250522637287b9b09cb3649d14f9fb6039d",
    "ad1d41af6bc5c57925840757beb58e25771500828798a282548c714130d9c8ba",
    "a9a163c6d74e32855af83cf8ade719ffdfc8ae48d4f8ee4ab3eb68f7bf7e7e18",
    "3cb9d10a3a03bd417eac94a3d4c85b4159ddf4f685953558ac4ec4cbb1a02679",
    "0876f7c1d1e7d603d4369dffa5b2fdcc2f424dbdf6455176c085b09375708a48",
    "a815192197963ad0595ad1eb34c08cb8e206cb7993ff0fbaafd53f2e3d7aea39",
    "788283924d2c968a7dace5c706d0153f4f44d3bdeff6574b8b9c978a3354a793",
    "47d237f51c4a0c6fe0a33807f49846fd27c1ce808af5a5bf908d1715b72a9840",
    "eb72fd8af5a16c06ca87d28e99a67baa3e9132bbe349e3c4f00640d4f309db4e",
    "10d200346d800ad2a5e0bb65225a2f93294442908cad109ab150e5f9ea12cf08",
    "5d07654194fb1028a813459cbac32deb969c8fe8149f210d3581ed5b20a4ad7e",
)

if SOURCE_DATASET.resolve() == FORBIDDEN_POSITIVE_DATASET.resolve():
    raise RuntimeError("V5 must never use the 611 positive-filtered dataset")

os.environ["GRPO_PARENT_ADAPTER"] = str(PARENT_ADAPTER)
sys.path.insert(0, str(SCRIPTS_DIR))

import run_grpo_trl_train as baseline_runner
from datasets import Dataset
from grpo_probe import load_probe_records
from monitor.writer import monitor_from_env as base_monitor_from_env
from trl import GRPOConfig

from .official_probe import VideoOfficialProbeEvaluator
from .video_official_anticopy_trainer import (
    COT_G,
    SID_G,
    VideoOfficialAntiCopyRuntime,
    ThinkG4SingleGroupSampler,
    ThinkVideoOfficialAntiCopyTrainer,
    audit_v5_sampler,
    extract_history_sids,
    make_video_official_anticopy_reward_func,
)
from ablations.gr_rec_think_suffix_sid_v1.runtime_import_provenance import (
    assert_runtime_import_provenance,
)
from grpo_sid import parse_sid

_BASE_PREPARE = baseline_runner.prepare_run_plan
_BASE_LOAD_MODEL = baseline_runner.load_model
_BASE_PARSER = baseline_runner.build_arg_parser
_RUNTIME_MODEL = None
_RUNTIME_TOKENIZER = None
_RUNTIME_PROVENANCE = None
_DATASET_GUARD = None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_id_list_sha256(group_ids):
    payload = "\n".join(sorted(group_ids)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_dataset_sha256(records):
    digest = hashlib.sha256()
    keys = (
        "prompt", "route", "recommendation_group_id",
        "target_domain", "all_gold_sids",
    )
    for record in records:
        payload = {key: record[key] for key in keys}
        digest.update(
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        )
    return digest.hexdigest()


def validate_parent_adapter():
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (PARENT_ADAPTER / name).is_file():
            raise FileNotFoundError(f"V5_PARENT_INCOMPLETE: {name}")
    actual = sha256_file(PARENT_ADAPTER / "adapter_model.safetensors")
    if actual != PARENT_ADAPTER_SHA256:
        raise RuntimeError(
            f"V5_PARENT_SHA_MISMATCH expected={PARENT_ADAPTER_SHA256} actual={actual}"
        )
    return {"path": str(PARENT_ADAPTER), "adapter_sha256": actual}


def validate_source_dataset():
    baseline_data = Path(baseline_runner.DATA)
    if baseline_data.resolve() != SOURCE_DATASET.resolve():
        raise RuntimeError(
            f"V5 baseline DATA drift: baseline={baseline_data} expected={SOURCE_DATASET}"
        )
    actual = sha256_file(SOURCE_DATASET)
    if actual != SOURCE_DATASET_SHA256:
        raise RuntimeError(
            f"V5_SOURCE_SHA_MISMATCH expected={SOURCE_DATASET_SHA256} actual={actual}"
        )
    rows = [
        json.loads(line)
        for line in SOURCE_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = {row["recommendation_group_id"] for row in rows}
    if len(rows) != EXPECTED_RAW_GROUPS * 2 or len(groups) != EXPECTED_RAW_GROUPS:
        raise RuntimeError("V5 source is not the original V3 1549-group route-paired dataset")
    return {
        "path": str(SOURCE_DATASET),
        "baseline_data_path": str(baseline_data),
        "sha256": actual,
        "rows": len(rows),
        "business_groups": len(groups),
    }


def build_video_dataset(base_plan):
    if base_plan["raw_groups"] != EXPECTED_RAW_GROUPS:
        raise RuntimeError("V5 raw group count drift")
    if tuple(base_plan["probe_group_ids"]) != FIXED_PROBE4_IDS:
        raise RuntimeError("V5 Probe4 selection drift")
    base_groups = set(base_plan["dataset"]["recommendation_group_id"])
    if len(base_groups) != EXPECTED_AFTER_PROBE4:
        raise RuntimeError("V5 Probe4 exclusion count drift")
    video = [
        dict(row) for row in base_plan["dataset"]
        if row["route"] == "think" and row["target_domain"] == "video"
    ]
    if len(video) != EXPECTED_VIDEO_GROUPS:
        raise RuntimeError(f"V5 Video group count drift: {len(video)}")
    heldout = set(HELDOUT_VIDEO_PROBE_IDS)
    train = [row for row in video if row["recommendation_group_id"] not in heldout]
    group_ids = [row["recommendation_group_id"] for row in train]
    if len(train) != EXPECTED_TRAIN_GROUPS or len(set(group_ids)) != EXPECTED_TRAIN_GROUPS:
        raise RuntimeError("V5 final train topology drift")
    if set(group_ids) & set(FIXED_PROBE4_IDS):
        raise RuntimeError("V5 final train overlaps Probe4")
    if set(group_ids) & heldout:
        raise RuntimeError("V5 final train overlaps held-out Video Probe")
    actual_dataset_sha = canonical_dataset_sha256(train)
    actual_ids_sha = group_id_list_sha256(group_ids)
    if actual_dataset_sha != FINAL_TRAIN_CANONICAL_SHA256:
        raise RuntimeError("V5 final canonical dataset SHA drift")
    if actual_ids_sha != FINAL_GROUP_ID_LIST_SHA256:
        raise RuntimeError("V5 final group ID list SHA drift")
    gold_hist = 0
    gold_total = 0
    overlap_groups = 0
    history_counts = collections.Counter()
    gold_counts = collections.Counter()
    for row in train:
        history = extract_history_sids(row["prompt"])
        gold = {
            sid for raw in row["all_gold_sids"]
            if (sid := parse_sid(raw)) is not None
        }
        overlap = gold & history
        history_counts[len(history)] += 1
        gold_counts[len(gold)] += 1
        overlap_groups += bool(overlap)
        gold_hist += len(overlap)
        gold_total += len(gold)
    return Dataset.from_list(train), {
        "source_business_groups": EXPECTED_RAW_GROUPS,
        "groups_after_probe4": EXPECTED_AFTER_PROBE4,
        "video_think_groups_before_v5_heldout": EXPECTED_VIDEO_GROUPS,
        "heldout_video_groups": len(HELDOUT_VIDEO_PROBE_IDS),
        "train_rows": len(train),
        "unique_groups": len(set(group_ids)),
        "think_only": True,
        "video_only": True,
        "probe4_overlap": 0,
        "heldout_video_overlap": 0,
        "canonical_train_sha256": actual_dataset_sha,
        "group_id_list_sha256": actual_ids_sha,
        "gold_count_distribution": dict(sorted(gold_counts.items())),
        "history_sid_count_distribution": dict(sorted(history_counts.items())),
        "groups_with_gold_history_exact_overlap": overlap_groups,
        "gold_history_exact_overlap_count": gold_hist,
        "gold_sid_count": gold_total,
    }


def load_model_and_capture(*args, **kwargs):
    global _RUNTIME_MODEL, _RUNTIME_TOKENIZER
    result = _BASE_LOAD_MODEL(*args, **kwargs)
    _RUNTIME_MODEL, _RUNTIME_TOKENIZER = result[0], result[1]
    return result


def make_runtime_reward(beam32_fn=None):
    del beam32_fn
    if _RUNTIME_MODEL is None or _RUNTIME_TOKENIZER is None:
        raise RuntimeError("V5_RUNTIME_MODEL_NOT_CAPTURED")
    runtime = VideoOfficialAntiCopyRuntime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER)
    return make_video_official_anticopy_reward_func(runtime)


def prepare_v5_run_plan(args):
    global _DATASET_GUARD
    validate_source_dataset()
    base = _BASE_PREPARE(args)
    dataset, guard = build_video_dataset(base)
    sampler = ThinkG4SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
    audit = audit_v5_sampler(dataset, sampler)
    max_steps = args.max_steps if args.max_steps is not None else EXPECTED_STEPS
    if not 1 <= max_steps <= EXPECTED_STEPS or max_steps % 2:
        raise ValueError(f"--max-steps must be even and in [2, {EXPECTED_STEPS}]")
    _DATASET_GUARD = guard
    base.update({
        "dataset": dataset,
        "audit": audit,
        "max_steps": max_steps,
        "probe_train_overlap": [],
        "dataset_guard": guard,
    })
    return base


def v5_config_kwargs(**overrides):
    values = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=COT_G,
        generation_batch_size=COT_G,
        max_prompt_length=8192,
        max_completion_length=2048,
        num_iterations=2,
        beta=0.0,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        disable_dropout=True,
        importance_sampling_level="token",
        top_entropy_quantile=1.0,
        mask_truncated_completions=False,
        use_vllm=False,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        report_to="none",
        temperature=0.9,
        top_p=0.95,
        generation_kwargs=None,
        shuffle_dataset=False,
    )
    values.update(overrides)
    return values


def make_v5_config(output_dir, max_steps, lr, seed, *, save_strategy="no",
                   save_steps=50, save_total_limit=None, use_cpu=False):
    if float(lr) != 1e-6:
        raise ValueError("V5 learning rate is frozen at 1e-6")
    return GRPOConfig(**v5_config_kwargs(
        output_dir=output_dir,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        learning_rate=lr,
        use_cpu=use_cpu,
    ))


def build_v5_parser():
    parser = _BASE_PARSER()
    parser.set_defaults(
        output_dir="/root/GRPO-checkpoints",
        save_steps=50,
        save_total_limit=64,
        probe_every_steps=50,
    )
    return parser


class V5MonitorWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_VideoOfficialAntiCopy_v5",
            "experiment_type": "Video-only Think G4 + Official Sample8 anti-copy GRPO",
            "source_dataset_path": str(SOURCE_DATASET),
            "source_dataset_sha256": SOURCE_DATASET_SHA256,
            "forbidden_positive_dataset": str(FORBIDDEN_POSITIVE_DATASET),
            "positive_reward_filtering": False,
            "dataset_guard": _DATASET_GUARD,
            "parent_adapter_path": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "optimizer_initialization": "fresh AdamW",
            "optimizer": {
                "learning_rate": 1e-6,
                "weight_decay": 0.0,
                "scheduler": "constant",
            },
            "rollout_topology": "1xG4 CoT + 4xG8 Official",
            "normalization": "four independent G8; never G32",
            "cot_sampling": {
                "temperature": 0.9,
                "top_p": 0.95,
                "closure_retry": True,
                "length_reward": False,
            },
            "official_sampling": {
                "fixed_domain_prefix": "<|video_begin|>",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 3,
            },
            "history_copy": "strict complete Video domain+A+B+C SID equality in user history",
            "sid_reward": {
                "copied_non_exact": 0,
                "copied_exact": 8,
                "noncopy": "unchanged q_reward",
            },
            "cot_reward": "sum raw q_reward for non-copy candidates; every complete copy contributes 0",
            "loss": "L_cot + L_sid (1:1)",
            "num_iterations": 2,
            "iteration2_reuse": [
                "CoT", "Official Sample8", "reward", "advantage", "old_logp",
            ],
            "fixed_probe": {
                "enabled": True,
                "group_ids": list(FIXED_PROBE4_IDS),
                "contract": "all-domain production Official Beam32 ABC3",
            },
            "heldout_video_official_probe": {
                "group_ids": list(HELDOUT_VIDEO_PROBE_IDS),
                "count": len(HELDOUT_VIDEO_PROBE_IDS),
                "excluded_from_training": True,
                "probe4_overlap": 0,
                "contract": "fixed Video Think CoT -> fixed Video begin -> Beam32 ABC3",
            },
            "expected_optimizer_steps": EXPECTED_STEPS,
        })
        if _RUNTIME_PROVENANCE:
            payload.update(_RUNTIME_PROVENANCE)
        self._writer.write_manifest(payload)

    def write_video_official_anticopy_v5(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append(
            "video_official_anticopy_v5.jsonl",
            {"type": "video_official_anticopy_v5", **event},
        )


def v5_monitor_from_env(run_id, rank):
    return V5MonitorWriter(base_monitor_from_env(run_id, rank))


class BoundVideoOfficialProbeEvaluator(VideoOfficialProbeEvaluator):
    def __init__(self, *args, **kwargs):
        records = load_probe_records(SOURCE_DATASET, HELDOUT_VIDEO_PROBE_IDS)
        super().__init__(
            *args,
            heldout_records=records,
            heldout_group_ids=HELDOUT_VIDEO_PROBE_IDS,
            **kwargs,
        )


def install_bindings():
    baseline_runner.build_arg_parser = build_v5_parser
    baseline_runner.prepare_run_plan = prepare_v5_run_plan
    baseline_runner.load_model = load_model_and_capture
    baseline_runner.RecGRPOTrainer = ThinkVideoOfficialAntiCopyTrainer
    baseline_runner.make_think_reward_func = make_runtime_reward
    baseline_runner.make_grpo_config = make_v5_config
    baseline_runner.monitor_from_env = v5_monitor_from_env
    baseline_runner.FixedProbeEvaluator = BoundVideoOfficialProbeEvaluator


def main(argv=None):
    global _RUNTIME_PROVENANCE
    _RUNTIME_PROVENANCE = assert_runtime_import_provenance()
    validate_parent_adapter()
    validate_source_dataset()
    install_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
