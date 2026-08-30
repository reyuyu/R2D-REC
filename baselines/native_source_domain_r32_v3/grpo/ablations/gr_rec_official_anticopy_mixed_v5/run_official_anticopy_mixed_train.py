#!/usr/bin/env python3
"""Formal runner contract for all-domain Official Anti-Copy GRPO V5-mixed."""
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
FINAL_TRAIN_CANONICAL_SHA256 = "e43cef3572e095e064e10d1fd40d88e92dea48120e99384e8d2df2be7b413efb"
FINAL_GROUP_ID_LIST_SHA256 = "1900ce3dd322e0237468210c7f4dee3f738bd59f8f8c2b1eeafbe91d2defa038"
PARENT_ADAPTER = Path(
    "/root/GRPO-checkpoints/"
    "GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-1500"
)
PARENT_ADAPTER_SHA256 = "3bc818109c8895225133c5cc77cfd1916ad8777292e7865969730708d0b48740"
EXPECTED_RAW_GROUPS = 1549
EXPECTED_AFTER_PROBE4 = 1545
EXPECTED_AFTER_PROBE16 = 1533
EXPECTED_DOMAIN_GROUPS = {"ad": 423, "living": 186, "prod": 378, "video": 546}
EXPECTED_TRAIN_GROUPS = 1533
EXPECTED_STEPS = EXPECTED_TRAIN_GROUPS * 2
FIXED_PROBE4_IDS = (
    "fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e",
    "6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f",
    "281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700",
    "2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8",
)
EXTRA_PROBE12_IDS = (
    "7ee11d79f71960090e4ed33719367b4a06d2bd2373e17bc4692c7b474756ad26",
    "73bbd41bd5449a4d28a7a2bb73f6f66ce96c20f01b34398c6942a71612b8de73",
    "a26b9927310bb09d606ae8a2632137326222f983d8bde7d29183940d068d4460",
    "6081a4847c2ccea83ad57faeb370d987721b8b82a0387eacf5c2d33d6ce1ddd8",
    "02cdfed2e24fc36a7dd2939c4ca0d34aa189f187908dc23e9fce986b07bbb09e",
    "87ac33d73d5afa9d5d180e87ebd0f8e78d30d4d1ba92564bc53a38b38d1885f6",
    "fd487b0a9bf3cd0069dea7a3070a1cabe3ba90567a228bb2f286257c1bd15e87",
    "74e68ff55efc91c30341116349157915fe0d94fc7bd4fdf9446970d88d18b89b",
    "c399e01d9eaa30e416df6c29ee118db4b9a6361bf558433439ccec27500e0d46",
    "88a0d179ef5072e8172ebb56bc74c27900a596f9a7df5241355d5d141f3c5a55",
    "9cec7983b7387dd0b399bde5a31fa49e142618d56754459afa36f88030e39fee",
    "9dabef351f0971928e9bfc0e3799f0222b497edb9a01deb2ee93b6c3c9e9b316",
)
FIXED_PROBE16_IDS = FIXED_PROBE4_IDS + EXTRA_PROBE12_IDS

if SOURCE_DATASET.resolve() == FORBIDDEN_POSITIVE_DATASET.resolve():
    raise RuntimeError("V5 must never use the 611 positive-filtered dataset")

os.environ["GRPO_PARENT_ADAPTER"] = str(PARENT_ADAPTER)
sys.path.insert(0, str(SCRIPTS_DIR))

import run_grpo_trl_train as baseline_runner
from datasets import Dataset
from monitor.writer import monitor_from_env as base_monitor_from_env
from trl import GRPOConfig

from .official_probe import MixedOfficialProbeEvaluator
from .official_anticopy_mixed_trainer import (
    COT_G,
    SID_G,
    OfficialAntiCopyMixedRuntime,
    ThinkG4SingleGroupSampler,
    ThinkOfficialAntiCopyMixedTrainer,
    audit_v5_sampler,
    extract_history_sids,
    make_official_anticopy_mixed_reward_func,
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


def build_mixed_dataset(base_plan):
    if base_plan["raw_groups"] != EXPECTED_RAW_GROUPS:
        raise RuntimeError("V5 raw group count drift")
    if tuple(base_plan["probe_group_ids"]) != FIXED_PROBE16_IDS:
        raise RuntimeError("V5 Probe16 selection drift")
    base_groups = set(base_plan["dataset"]["recommendation_group_id"])
    if len(base_groups) != EXPECTED_AFTER_PROBE16:
        raise RuntimeError("V5 Probe16 exclusion count drift")
    train = [
        dict(row) for row in base_plan["dataset"]
        if row["route"] == "think"
    ]
    group_ids = [row["recommendation_group_id"] for row in train]
    if len(train) != EXPECTED_TRAIN_GROUPS or len(set(group_ids)) != EXPECTED_TRAIN_GROUPS:
        raise RuntimeError("V5 final train topology drift")
    if set(group_ids) & set(FIXED_PROBE16_IDS):
        raise RuntimeError("V5 final train overlaps Probe16")
    domain_counts = dict(sorted(collections.Counter(
        row["target_domain"] for row in train
    ).items()))
    if domain_counts != EXPECTED_DOMAIN_GROUPS:
        raise RuntimeError(f"V5-mixed domain counts drift: {domain_counts}")
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
        history = extract_history_sids(row["prompt"], row["target_domain"])
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
        "groups_after_probe16": EXPECTED_AFTER_PROBE16,
        "train_rows": len(train),
        "unique_groups": len(set(group_ids)),
        "think_only": True,
        "all_domains": sorted(EXPECTED_DOMAIN_GROUPS),
        "domain_group_counts": domain_counts,
        "probe4_overlap": 0,
        "probe16_overlap": 0,
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
    runtime = OfficialAntiCopyMixedRuntime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER)
    return make_official_anticopy_mixed_reward_func(runtime)


def prepare_v5_run_plan(args):
    global _DATASET_GUARD
    validate_source_dataset()
    base = _BASE_PREPARE(args)
    dataset, guard = build_mixed_dataset(base)
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
    for action in parser._actions:
        if action.dest == "probe_groups":
            action.choices = (0, 4, 16)
            break
    parser.set_defaults(
        output_dir="/root/GRPO-checkpoints",
        save_steps=50,
        save_total_limit=64,
        probe_every_steps=50,
    )
    return parser


class MixedV5MonitorWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_OfficialAntiCopy_Mixed_v5",
            "experiment_type": "All-domain Think G4 + Official Sample8 anti-copy GRPO",
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
                "fixed_domain_prefix": "<|target_domain_begin|>",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 3,
            },
            "history_copy": "strict complete same-target-domain domain+A+B+C SID equality in user history",
            "sid_reward": {
                "video": {
                    "copied_non_exact": 0,
                    "copied_exact": 8,
                    "noncopy": "unchanged q_reward",
                },
                "ad_prod_living": {
                    "noncopy": [0, 0.5, 2, 8],
                    "fresh_rollout_0_49_copy": [0, 0.25, 1, 6],
                    "fresh_rollout_50_plus_copy": [0, 0, 1, 6],
                },
            },
            "cot_reward": {
                "video": "sum raw q_reward for non-copy candidates; every copy contributes 0",
                "ad_prod_living": "sum final domain-shaped SID reward",
            },
            "loss": "L_cot + L_sid (1:1)",
            "num_iterations": 2,
            "iteration2_reuse": [
                "CoT", "Official Sample8", "reward", "advantage", "old_logp",
            ],
            "fixed_probe": {
                "enabled": True,
                "group_ids": list(FIXED_PROBE16_IDS),
                "groups_per_domain": 4,
                "contract": "all-domain production Official Beam32 ABC3",
                "training_copy_discount_applied": False,
                "extra_diagnostics": [
                    "history_copy_rate", "copy/noncopy A/AB/Exact", "CoT length",
                ],
            },
            "expected_optimizer_steps": EXPECTED_STEPS,
        })
        if _RUNTIME_PROVENANCE:
            payload.update(_RUNTIME_PROVENANCE)
        self._writer.write_manifest(payload)

    def write_official_anticopy_mixed_v5(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append(
            "official_anticopy_mixed_v5.jsonl",
            {"type": "official_anticopy_mixed_v5", **event},
        )


def v5_monitor_from_env(run_id, rank):
    return MixedV5MonitorWriter(base_monitor_from_env(run_id, rank))


def install_bindings():
    baseline_runner.build_arg_parser = build_v5_parser
    baseline_runner.prepare_run_plan = prepare_v5_run_plan
    baseline_runner.load_model = load_model_and_capture
    baseline_runner.RecGRPOTrainer = ThinkOfficialAntiCopyMixedTrainer
    baseline_runner.make_think_reward_func = make_runtime_reward
    baseline_runner.make_grpo_config = make_v5_config
    baseline_runner.monitor_from_env = v5_monitor_from_env
    baseline_runner.FixedProbeEvaluator = MixedOfficialProbeEvaluator


def main(argv=None):
    global _RUNTIME_PROVENANCE
    _RUNTIME_PROVENANCE = assert_runtime_import_provenance()
    validate_parent_adapter()
    validate_source_dataset()
    install_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
