#!/usr/bin/env python3
"""V3 second stage on positive groups with only A-only reward changed to zero."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

GRPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = GRPO_ROOT / "scripts"
DATASET = Path("/data/GRPO/data/grpo_tk_positive_groups_1946_20260829/train.jsonl")
DATASET_SHA256 = "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"
PARENT_ADAPTER = Path(
    "/root/GRPO-checkpoints/"
    "GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-250"
)
PARENT_ADAPTER_SHA256 = "64e1a85500b68f48d1996e6a215ea7a0bfe0dfdb925d6d53a86579ea23650be5"
PARENT_RECORDED_EXTERNAL_SCORE = 1.3579
EXPECTED_ROWS = 611
EXPECTED_STEPS = EXPECTED_ROWS * 2
EXPECTED_DOMAIN_COUNTS = {"ad": 160, "living": 73, "prod": 118, "video": 260}
FIXED_PROBE4_IDS = (
    "fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e",
    "6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f",
    "281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700",
    "2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8",
)

# Set this before importing the shared loader. The imported V3 runner resets the
# environment to its own parent, so grpo_model.ADAPTER is reasserted below too.
os.environ["GRPO_PARENT_ADAPTER"] = str(PARENT_ADAPTER)
sys.path.insert(0, str(SCRIPTS_DIR))

import grpo_model
import run_grpo_trl_train as baseline_runner
from datasets import Dataset
from monitor.writer import monitor_from_env as base_monitor_from_env
from trl import GRPOConfig

from .a0_reward import q_reward_without_a_only
from ablations.gr_rec_think_sample8_fullsid_v3 import run_sample8_fullsid_train as v3_runner
from ablations.gr_rec_think_sample8_fullsid_v3 import sample8_fullsid_trainer as v3_trainer
from ablations.gr_rec_think_suffix_sid_v1.runtime_import_provenance import (
    assert_runtime_import_provenance,
)

_BASE_PREPARE = baseline_runner.prepare_run_plan
_BASE_PARSER = baseline_runner.build_arg_parser
_RUNTIME_PROVENANCE = None
_DATASET_GUARD = None

# Fail closed against the parent accidentally selected while importing V3.
os.environ["GRPO_PARENT_ADAPTER"] = str(PARENT_ADAPTER)
grpo_model.ADAPTER = str(PARENT_ADAPTER)
baseline_runner.ADAPTER = str(PARENT_ADAPTER)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_adapter(path=PARENT_ADAPTER, expected_sha=PARENT_ADAPTER_SHA256):
    path = Path(path)
    missing = [
        name for name in ("adapter_config.json", "adapter_model.safetensors")
        if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"POSITIVE_A0_PARENT_INCOMPLETE: {missing}")
    actual = sha256_file(path / "adapter_model.safetensors")
    if actual != expected_sha:
        raise RuntimeError(
            f"POSITIVE_A0_PARENT_SHA_MISMATCH expected={expected_sha} actual={actual}"
        )
    return {"path": str(path), "adapter_sha256": actual}


def validate_dataset(path=DATASET, expected_sha=DATASET_SHA256):
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(
            f"POSITIVE_A0_DATASET_SHA_MISMATCH expected={expected_sha} actual={actual}"
        )
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gids = [row.get("recommendation_group_id") for row in rows]
    domain_counts = dict(sorted(collections.Counter(
        row.get("target_domain") for row in rows
    ).items()))
    if len(rows) != EXPECTED_ROWS or len(set(gids)) != EXPECTED_ROWS:
        raise RuntimeError("POSITIVE_A0_DATASET_TOPOLOGY_MISMATCH")
    if any(row.get("route") != "think" for row in rows):
        raise RuntimeError("POSITIVE_A0_DATASET_NOT_THINK_ONLY")
    if domain_counts != EXPECTED_DOMAIN_COUNTS:
        raise RuntimeError(
            f"POSITIVE_A0_DOMAIN_DISTRIBUTION_MISMATCH: {domain_counts}"
        )
    overlap = sorted(set(gids) & set(FIXED_PROBE4_IDS))
    if overlap:
        raise RuntimeError(f"POSITIVE_A0_PROBE4_OVERLAP: {overlap}")
    return {
        "path": str(path),
        "sha256": actual,
        "rows": len(rows),
        "unique_groups": len(set(gids)),
        "think_only": True,
        "domain_counts": domain_counts,
        "probe4_overlap": overlap,
        "records": rows,
    }


def prepare_positive_a0_run_plan(args):
    global _DATASET_GUARD
    if args.resume_from_checkpoint is not None:
        raise ValueError(
            "Positive A0 second stage must fresh-start from V3 checkpoint-250; "
            "checkpoint resume is forbidden"
        )
    if str(args.n_groups) != "all":
        raise ValueError("Positive A0 second stage requires --n-groups all")
    base = _BASE_PREPARE(args)
    if base["raw_groups"] != 1549:
        raise RuntimeError(f"POSITIVE_A0_V3_SOURCE_GROUP_DRIFT: {base['raw_groups']}")
    if tuple(base["probe_group_ids"]) != FIXED_PROBE4_IDS:
        raise RuntimeError(
            f"POSITIVE_A0_PROBE4_ID_DRIFT: {tuple(base['probe_group_ids'])}"
        )
    guard = validate_dataset()
    records = guard.pop("records")
    dataset = Dataset.from_list(records)
    sampler = v3_trainer.ThinkG4SingleGroupSampler(
        dataset, repeat_count=2, shuffle=False
    )
    audit = v3_trainer.audit_sample8_sampler(dataset, sampler)
    if audit["trained_groups"] != EXPECTED_ROWS:
        raise RuntimeError("POSITIVE_A0_TRAIN_TOPOLOGY_MISMATCH")
    max_steps = args.max_steps if args.max_steps is not None else EXPECTED_STEPS
    if not 1 <= max_steps <= EXPECTED_STEPS or max_steps % 2:
        raise ValueError(f"--max-steps must be even and in [2, {EXPECTED_STEPS}]")
    _DATASET_GUARD = guard
    base.update({
        "raw_groups": EXPECTED_ROWS,
        "dataset": dataset,
        "audit": audit,
        "max_steps": max_steps,
        "probe_train_overlap": [],
        "dataset_guard": guard,
    })
    return base


def positive_a0_config_kwargs(**overrides):
    values = v3_runner.sample8_config_kwargs()
    values.update(overrides)
    return values


def make_positive_a0_config(
    output_dir, max_steps, lr, seed, *, save_strategy="no",
    save_steps=50, save_total_limit=None, use_cpu=False,
):
    if float(lr) != 1e-6:
        raise ValueError("Positive A0 second-stage learning rate is frozen at 1e-6")
    return GRPOConfig(**positive_a0_config_kwargs(
        output_dir=output_dir,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        learning_rate=lr,
        use_cpu=use_cpu,
    ))


def build_positive_a0_parser():
    parser = _BASE_PARSER()
    parser.set_defaults(
        output_dir="/root/GRPO-checkpoints",
        save_steps=50,
        save_total_limit=64,
        probe_every_steps=50,
    )
    return parser


class PositiveA0MonitorWriter:
    def __init__(self, writer):
        object.__setattr__(self, "_writer", writer)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_ThinkSample8_FullSID_PositiveA0_v3",
            "experiment_type": "V3 second-stage positive-group GRPO; A-only reward removed",
            "dataset_path": str(DATASET),
            "dataset_sha256": DATASET_SHA256,
            "dataset_guard": _DATASET_GUARD,
            "positive_group_filter": "archived V3 signal: cot_reward>0 or any sid_reward>0",
            "parent_adapter_path": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_recorded_external_score": PARENT_RECORDED_EXTERNAL_SCORE,
            "optimizer_initialization": "fresh AdamW",
            "optimizer": {
                "learning_rate": 1e-6,
                "weight_decay": 0.0,
                "scheduler": "constant",
            },
            "rollout_topology": "unchanged V3: 1x global G4 CoT + 4 independent G8 Free FullSID",
            "normalization": "unchanged V3: G4 CoT and four independent G8 SID groups; never G32",
            "sid_reward": {
                "invalid": -1.0,
                "wrong_domain": -0.25,
                "no_hit": 0.0,
                "A_only": 0.0,
                "AB": 2.0,
                "Exact": 8.0,
            },
            "reward_delta_from_v3": "A-only 0.5 -> 0; all other levels unchanged",
            "cot_reward": "unchanged V3 sum of eight shaped SID rewards; therefore A-only contributes 0",
            "loss": "unchanged V3 L_cot + L_sid (1:1)",
            "num_iterations": 2,
            "iteration2_reuse": [
                "CoT", "Sample8 FullSID", "reward", "advantage", "old_logp",
            ],
            "fixed_probe": {
                "enabled": True,
                "group_ids": list(FIXED_PROBE4_IDS),
                "excluded_from_training": True,
                "every_steps": 50,
                "contract": "unchanged production-shaped Probe4/Beam32",
            },
            "expected_training_groups": EXPECTED_ROWS,
            "expected_fresh_rollouts": EXPECTED_ROWS,
            "expected_optimizer_steps": EXPECTED_STEPS,
            "explicit_final_checkpoint_step": EXPECTED_STEPS,
        })
        if _RUNTIME_PROVENANCE:
            payload.update(_RUNTIME_PROVENANCE)
        self._writer.write_manifest(payload)

    def write_sample8_fullsid(self, event):
        if self._writer.rank != 0:
            return False
        return self._writer._append(
            "sample8_fullsid.jsonl", {"type": "sample8_fullsid", **event}
        )


def positive_a0_monitor_from_env(run_id, rank):
    return PositiveA0MonitorWriter(base_monitor_from_env(run_id, rank))


def install_bindings():
    # This is the sole training-math override. It is process-local and leaves
    # the original V3 source module and every other experiment unchanged.
    v3_trainer.q_reward = q_reward_without_a_only
    baseline_runner.build_arg_parser = build_positive_a0_parser
    baseline_runner.prepare_run_plan = prepare_positive_a0_run_plan
    baseline_runner.load_model = v3_runner.load_model_and_capture_runtime
    baseline_runner.RecGRPOTrainer = v3_trainer.ThinkSample8FullSIDTrainer
    baseline_runner.make_think_reward_func = v3_runner.make_runtime_sample8_reward
    baseline_runner.make_grpo_config = make_positive_a0_config
    baseline_runner.monitor_from_env = positive_a0_monitor_from_env


def main(argv=None):
    global _RUNTIME_PROVENANCE
    _RUNTIME_PROVENANCE = assert_runtime_import_provenance()
    validate_parent_adapter()
    validate_dataset()
    if any(
        value == "--resume-from-checkpoint"
        or value.startswith("--resume-from-checkpoint=")
        for value in (argv or [])
    ):
        raise ValueError(
            "Positive A0 second stage must fresh-start from V3 checkpoint-250"
        )
    if getattr(baseline_runner, "ADAPTER", None) != str(PARENT_ADAPTER):
        raise RuntimeError("POSITIVE_A0_PARENT_BINDING_DRIFT")
    install_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
