#!/usr/bin/env python3
"""Formal runner contract for V6-A Fine-grained Frontier GRPO."""
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
FINAL_TRAIN_CANONICAL_SHA256 = "4a4e6ed3abf2deafef5a931c8f3b97419f40dea2ca01661d9b399347a3998909"
FINAL_GROUP_ID_LIST_SHA256 = "c1416069c076469252668bb27f69b0904946bd2d2154cbe3024115a17175be5f"
PARENT_ADAPTER = Path(
    "/root/GRPO-checkpoints/"
    "GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-1500"
)
PARENT_ADAPTER_SHA256 = "3bc818109c8895225133c5cc77cfd1916ad8777292e7865969730708d0b48740"
EXPECTED_RAW_GROUPS = 1549
EXPECTED_AFTER_PROBE4 = 1545
EXPECTED_AFTER_PROBE8 = 1541
EXPECTED_DOMAIN_GROUPS = {"ad": 426, "living": 189, "prod": 381, "video": 549}
EXPECTED_TRAIN_GROUPS = 1545
EXPECTED_STEPS = EXPECTED_TRAIN_GROUPS * 2
FIXED_PROBE4_IDS = (
    "fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e",
    "6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f",
    "281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700",
    "2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8",
)
EXTRA_PROBE4_IDS = (
    "7ee11d79f71960090e4ed33719367b4a06d2bd2373e17bc4692c7b474756ad26",
    "73bbd41bd5449a4d28a7a2bb73f6f66ce96c20f01b34398c6942a71612b8de73",
    "a26b9927310bb09d606ae8a2632137326222f983d8bde7d29183940d068d4460",
    "6081a4847c2ccea83ad57faeb370d987721b8b82a0387eacf5c2d33d6ce1ddd8",
)
FIXED_PROBE8_IDS = FIXED_PROBE4_IDS + EXTRA_PROBE4_IDS
HISTORY_COPY_PROBE4_IDS = (
    "38a1cd737cddfae7bddb24c0429d643047d203e7870078216bb75725e4f11f8f",
    "8018a2ceba46df4065db69256444553385b8d5dcd4114208054af17adecb3b08",
    "383b75f18ffa0656534ba0ddf2139f3912008e118da72b285b7f28ec10df8152",
    "15cdfbc23e15631e68e6fbf0da62b34a90ef661fa5e738424720b57f064fa7ed",
)
HISTORY_COPY_PROBE4_OVERLAP = {"video": 3, "ad": 9, "prod": 3, "living": 2}

if SOURCE_DATASET.resolve() == FORBIDDEN_POSITIVE_DATASET.resolve():
    raise RuntimeError("V6 must never use the 611 positive-filtered dataset")

os.environ["GRPO_PARENT_ADAPTER"] = str(PARENT_ADAPTER)
sys.path.insert(0, str(SCRIPTS_DIR))

import run_grpo_trl_train as baseline_runner
from datasets import Dataset
from monitor.writer import monitor_from_env as base_monitor_from_env
from trl import GRPOConfig

from .official_probe import FineGrainedOfficialProbeEvaluator
from .official_finegrained_trainer import (
    COT_G,
    SID_G,
    OfficialFineGrainedRuntime,
    ThinkG4SingleGroupSampler,
    ThinkOfficialFineGrainedTrainer,
    audit_v6_sampler,
    extract_history_sids,
    make_official_finegrained_reward_func,
)
from ablations.gr_rec_think_suffix_sid_v1.runtime_import_provenance import (
    assert_runtime_import_provenance,
)
from ablations.gr_rec_think_sample8_fullsid_v3.run_sample8_fullsid_train import (
    enable_trusted_torch_load_for_resume,
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
            raise FileNotFoundError(f"V6_PARENT_INCOMPLETE: {name}")
    actual = sha256_file(PARENT_ADAPTER / "adapter_model.safetensors")
    if actual != PARENT_ADAPTER_SHA256:
        raise RuntimeError(
            f"V6_PARENT_SHA_MISMATCH expected={PARENT_ADAPTER_SHA256} actual={actual}"
        )
    return {"path": str(PARENT_ADAPTER), "adapter_sha256": actual}


def validate_source_dataset():
    baseline_data = Path(baseline_runner.DATA)
    if baseline_data.resolve() != SOURCE_DATASET.resolve():
        raise RuntimeError(
            f"V6 baseline DATA drift: baseline={baseline_data} expected={SOURCE_DATASET}"
        )
    actual = sha256_file(SOURCE_DATASET)
    if actual != SOURCE_DATASET_SHA256:
        raise RuntimeError(
            f"V6_SOURCE_SHA_MISMATCH expected={SOURCE_DATASET_SHA256} actual={actual}"
        )
    rows = [
        json.loads(line)
        for line in SOURCE_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = {row["recommendation_group_id"] for row in rows}
    if len(rows) != EXPECTED_RAW_GROUPS * 2 or len(groups) != EXPECTED_RAW_GROUPS:
        raise RuntimeError("V6 source is not the original V3 1549-group route-paired dataset")
    return {
        "path": str(SOURCE_DATASET),
        "baseline_data_path": str(baseline_data),
        "sha256": actual,
        "rows": len(rows),
        "business_groups": len(groups),
    }


def validate_history_copy_probe(records):
    if tuple(records) != HISTORY_COPY_PROBE4_IDS:
        raise RuntimeError("V6 history-copy Probe4 group selection drift")
    observed = {}
    for gid in HISTORY_COPY_PROBE4_IDS:
        row = records[gid]["think"]
        domain = row["target_domain"]
        history = extract_history_sids(row["prompt"], domain)
        gold = {sid for raw in row["all_gold_sids"] if (sid := parse_sid(raw)) is not None}
        observed[domain] = len(history & gold)
    if observed != HISTORY_COPY_PROBE4_OVERLAP:
        raise RuntimeError(f"V6 history-copy Probe4 overlap drift: {observed}")
    return observed


def build_finegrained_dataset(base_plan):
    if base_plan["raw_groups"] != EXPECTED_RAW_GROUPS:
        raise RuntimeError("V6 raw group count drift")
    if tuple(base_plan["probe_group_ids"]) != FIXED_PROBE8_IDS:
        raise RuntimeError("V6 Probe8 selection drift")
    base_groups = set(base_plan["dataset"]["recommendation_group_id"])
    if len(base_groups) != EXPECTED_AFTER_PROBE8:
        raise RuntimeError("V6 Probe8 baseline exclusion count drift")
    source_rows = [
        json.loads(line) for line in SOURCE_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [
        dict(row) for row in source_rows
        if row["route"] == "think"
        and row["recommendation_group_id"] not in FIXED_PROBE4_IDS
    ]
    group_ids = [row["recommendation_group_id"] for row in train]
    if len(train) != EXPECTED_TRAIN_GROUPS or len(set(group_ids)) != EXPECTED_TRAIN_GROUPS:
        raise RuntimeError("V6 final train topology drift")
    if set(group_ids) & set(FIXED_PROBE4_IDS):
        raise RuntimeError("V6 final train overlaps held-out Probe4")
    domain_counts = dict(sorted(collections.Counter(
        row["target_domain"] for row in train
    ).items()))
    if domain_counts != EXPECTED_DOMAIN_GROUPS:
        raise RuntimeError(f"V6-A domain counts drift: {domain_counts}")
    actual_dataset_sha = canonical_dataset_sha256(train)
    actual_ids_sha = group_id_list_sha256(group_ids)
    if actual_dataset_sha != FINAL_TRAIN_CANONICAL_SHA256:
        raise RuntimeError("V6 final canonical dataset SHA drift")
    if actual_ids_sha != FINAL_GROUP_ID_LIST_SHA256:
        raise RuntimeError("V6 final group ID list SHA drift")
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
        "groups_after_probe8": EXPECTED_AFTER_PROBE8,
        "train_rows": len(train),
        "unique_groups": len(set(group_ids)),
        "think_only": True,
        "all_domains": sorted(EXPECTED_DOMAIN_GROUPS),
        "domain_group_counts": domain_counts,
        "probe4_overlap": 0,
        "heldout_probe4_overlap": 0,
        "probe8_training_overlap": len(set(group_ids) & set(EXTRA_PROBE4_IDS)),
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
        raise RuntimeError("V6_RUNTIME_MODEL_NOT_CAPTURED")
    runtime = OfficialFineGrainedRuntime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER)
    return make_official_finegrained_reward_func(runtime)


def prepare_v6_run_plan(args):
    global _DATASET_GUARD
    validate_source_dataset()
    base = _BASE_PREPARE(args)
    dataset, guard = build_finegrained_dataset(base)
    history_overlap = None
    if base["secondary_probe_group_ids"]:
        history_overlap = validate_history_copy_probe(base["secondary_probe_records"])
    sampler = ThinkG4SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
    audit = audit_v6_sampler(dataset, sampler)
    max_steps = args.max_steps if args.max_steps is not None else EXPECTED_STEPS
    if not 1 <= max_steps <= EXPECTED_STEPS or max_steps % 2:
        raise ValueError(f"--max-steps must be even and in [2, {EXPECTED_STEPS}]")
    _DATASET_GUARD = guard
    base.update({
        "dataset": dataset,
        "audit": audit,
        "max_steps": max_steps,
        "probe_train_overlap": sorted(set(EXTRA_PROBE4_IDS)),
        "dataset_guard": guard,
        "history_copy_probe_overlap": history_overlap,
    })
    return base


def v6_config_kwargs(**overrides):
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


def make_v6_config(output_dir, max_steps, lr, seed, *, save_strategy="no",
                   save_steps=50, save_total_limit=None, use_cpu=False):
    if float(lr) != 1e-6:
        raise ValueError("V6 learning rate is frozen at 1e-6")
    return GRPOConfig(**v6_config_kwargs(
        output_dir=output_dir,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        learning_rate=lr,
        use_cpu=use_cpu,
    ))


def build_v6_parser():
    parser = _BASE_PARSER()
    for action in parser._actions:
        if action.dest == "probe_groups":
            action.choices = (0, 4, 8)
            break
    parser.set_defaults(
        output_dir="/root/GRPO-checkpoints",
        save_steps=50,
        save_total_limit=64,
        probe_every_steps=50,
    )
    return parser


class MixedV6MonitorWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_OfficialFineGrained_v6A",
            "experiment_type": "All-domain Think G4 + Official Sample8 fine-grained frontier GRPO",
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
            "history_copy": "passive diagnostic only; excluded from all training math",
            "sid_signal": {
                "advantages": "independent Norm_G8(hit_A), Norm_G8(hit_B), Norm_G8(hit_C)",
                "token_gate": {"A": 1, "B": "hit_A", "C": "hit_B"},
                "normalization": "four independent G8 at every hierarchy; never G32",
            },
            "cot_reward": "sum of 8 untouched raw q_reward values, V3 parity",
            "loss": "L_cot + L_sid (1:1)",
            "num_iterations": 2,
            "iteration2_reuse": [
                "CoT", "Official Sample8", "reward", "advantage", "old_logp",
            ],
            "fixed_probe": {
                "enabled": True,
                "group_ids": list(FIXED_PROBE8_IDS),
                "groups_per_domain": 2,
                "contract": "all-domain production Official Beam32 ABC3",
                "heldout_groups_per_domain": 1,
                "training_overlap_groups_per_domain": 1,
                "extra_diagnostics": [
                    "history_copy_rate", "copy/noncopy A/AB/Exact", "CoT length",
                ],
            },
            "history_copy_probe": {
                "enabled": True,
                "suite": "history_copy_exact",
                "group_ids": list(HISTORY_COPY_PROBE4_IDS),
                "groups_per_domain": 1,
                "gold_history_exact_overlap": HISTORY_COPY_PROBE4_OVERLAP,
                "training_overlap_allowed": True,
                "contract": "Official Beam32 ABC3; production reward; exact Gold appears in same-domain history",
            },
            "expected_optimizer_steps": EXPECTED_STEPS,
        })
        if _RUNTIME_PROVENANCE:
            payload.update(_RUNTIME_PROVENANCE)
        self._writer.write_manifest(payload)

    def write_official_finegrained_v6(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append(
            "official_finegrained_v6.jsonl",
            {"type": "official_finegrained_v6", **event},
        )


def v6_monitor_from_env(run_id, rank):
    return MixedV6MonitorWriter(base_monitor_from_env(run_id, rank))


def install_bindings():
    baseline_runner.build_arg_parser = build_v6_parser
    baseline_runner.prepare_run_plan = prepare_v6_run_plan
    baseline_runner.load_model = load_model_and_capture
    baseline_runner.RecGRPOTrainer = ThinkOfficialFineGrainedTrainer
    baseline_runner.make_think_reward_func = make_runtime_reward
    baseline_runner.make_grpo_config = make_v6_config
    baseline_runner.monitor_from_env = v6_monitor_from_env
    baseline_runner.FixedProbeEvaluator = FineGrainedOfficialProbeEvaluator


def main(argv=None):
    global _RUNTIME_PROVENANCE
    _RUNTIME_PROVENANCE = assert_runtime_import_provenance()
    values = list(argv or ())
    resume_path = None
    for index, value in enumerate(values):
        if value == "--resume-from-checkpoint" and index + 1 < len(values):
            resume_path = values[index + 1]
            break
        if value.startswith("--resume-from-checkpoint="):
            resume_path = value.split("=", 1)[1]
            break
    if resume_path:
        _RUNTIME_PROVENANCE["trusted_resume"] = (
            enable_trusted_torch_load_for_resume(resume_path)
        )
    validate_parent_adapter()
    validate_source_dataset()
    install_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
