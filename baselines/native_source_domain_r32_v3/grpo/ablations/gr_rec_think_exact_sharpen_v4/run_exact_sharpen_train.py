#!/usr/bin/env python3
"""Formal runner contract for GR_REC_ThinkExactSharpen_v4."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

GRPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = GRPO_ROOT / "scripts"
DATASET = Path("/data/GRPO/data/grpo_tk_positive_groups_1946_20260829/train.jsonl")
DATASET_SHA256 = "e5a78e4e051dde3f8ec95d8564c6af094dd657e8608f84a31a310c3b749ff693"
PARENT_ADAPTER = Path("/root/GRPO-checkpoints/GR-REC-THINK-SAMPLE8-FULLSID-V3-FORMAL-E1-20260828/checkpoint-1500")
PARENT_ADAPTER_SHA256 = "3bc818109c8895225133c5cc77cfd1916ad8777292e7865969730708d0b48740"
EXPECTED_ROWS = 611
EXPECTED_STEPS = 1222
FIXED_PROBE4_IDS = (
    "fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e",
    "6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f",
    "281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700",
    "2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8",
)

os.environ["GRPO_PARENT_ADAPTER"] = str(PARENT_ADAPTER)
sys.path.insert(0, str(SCRIPTS_DIR))

import run_grpo_trl_train as baseline_runner
from datasets import Dataset
from grpo_probe import FixedProbeCallback
from monitor.writer import monitor_from_env as base_monitor_from_env
from trl import GRPOConfig

from .dual_probe import DualContractProbeEvaluator
from .exact_sharpen_trainer import (
    COT_G, SID_G, ExactSharpenRuntime, ThinkExactSharpenTrainer,
    ThinkG4SingleGroupSampler, audit_v4_sampler, make_exact_sharpen_reward_func,
)
from ablations.gr_rec_think_suffix_sid_v1.runtime_import_provenance import assert_runtime_import_provenance

_BASE_PREPARE = baseline_runner.prepare_run_plan
_BASE_LOAD_MODEL = baseline_runner.load_model
_BASE_PARSER = baseline_runner.build_arg_parser
_RUNTIME_MODEL = None
_RUNTIME_TOKENIZER = None
_RUNTIME_PROVENANCE = None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_adapter():
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (PARENT_ADAPTER / name).is_file():
            raise FileNotFoundError(f"V4_PARENT_INCOMPLETE: {name}")
    actual = sha256_file(PARENT_ADAPTER / "adapter_model.safetensors")
    if actual != PARENT_ADAPTER_SHA256:
        raise RuntimeError(f"V4_PARENT_SHA_MISMATCH expected={PARENT_ADAPTER_SHA256} actual={actual}")
    return {"path": str(PARENT_ADAPTER), "adapter_sha256": actual}


def validate_dataset(path=DATASET):
    actual = sha256_file(path)
    if actual != DATASET_SHA256:
        raise RuntimeError(f"V4_DATASET_SHA_MISMATCH expected={DATASET_SHA256} actual={actual}")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    gids = [row.get("recommendation_group_id") for row in rows]
    if len(rows) != EXPECTED_ROWS or len(set(gids)) != EXPECTED_ROWS:
        raise RuntimeError("V4_DATASET_TOPOLOGY_MISMATCH")
    if any(row.get("route") != "think" for row in rows):
        raise RuntimeError("V4_DATASET_NOT_THINK_ONLY")
    overlap = sorted(set(gids) & set(FIXED_PROBE4_IDS))
    if overlap:
        raise RuntimeError(f"V4_PROBE4_OVERLAP: {overlap}")
    return {"rows": len(rows), "unique_groups": len(set(gids)), "think_only": True,
            "probe4_overlap": overlap, "sha256": actual, "records": rows}


def load_model_and_capture(*args, **kwargs):
    global _RUNTIME_MODEL, _RUNTIME_TOKENIZER
    result = _BASE_LOAD_MODEL(*args, **kwargs)
    _RUNTIME_MODEL, _RUNTIME_TOKENIZER = result[0], result[1]
    return result


def make_runtime_reward(beam32_fn=None):
    del beam32_fn
    if _RUNTIME_MODEL is None or _RUNTIME_TOKENIZER is None:
        raise RuntimeError("V4_RUNTIME_MODEL_NOT_CAPTURED")
    return make_exact_sharpen_reward_func(ExactSharpenRuntime(_RUNTIME_MODEL, _RUNTIME_TOKENIZER))


def prepare_v4_run_plan(args):
    base = _BASE_PREPARE(args)
    audit = validate_dataset()
    dataset = Dataset.from_list(audit.pop("records"))
    sampler = ThinkG4SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
    sampler_audit = audit_v4_sampler(dataset, sampler)
    max_steps = args.max_steps if args.max_steps is not None else EXPECTED_STEPS
    if not 1 <= max_steps <= EXPECTED_STEPS or max_steps % 2:
        raise ValueError(f"--max-steps must be even and in [2, {EXPECTED_STEPS}]")
    base.update({"raw_groups": EXPECTED_ROWS, "dataset": dataset, "audit": sampler_audit,
                 "max_steps": max_steps, "probe_train_overlap": [],
                 "dataset_guard": audit})
    return base


def v4_config_kwargs(**overrides):
    values = dict(per_device_train_batch_size=1, gradient_accumulation_steps=1,
                  num_generations=COT_G, generation_batch_size=COT_G,
                  max_prompt_length=8192, max_completion_length=2048,
                  num_iterations=2, beta=0.0, epsilon=0.2, loss_type="grpo",
                  scale_rewards="group", disable_dropout=True,
                  importance_sampling_level="token", top_entropy_quantile=1.0,
                  mask_truncated_completions=False, use_vllm=False,
                  weight_decay=0.0, max_grad_norm=1.0, lr_scheduler_type="constant",
                  logging_steps=1, report_to="none", temperature=0.9, top_p=0.95,
                  generation_kwargs=None, shuffle_dataset=False)
    values.update(overrides)
    return values


def make_v4_config(output_dir, max_steps, lr, seed, *, save_strategy="no",
                   save_steps=50, save_total_limit=None, use_cpu=False):
    if float(lr) != 1e-6:
        raise ValueError("V4 learning rate is frozen at 1e-6")
    return GRPOConfig(**v4_config_kwargs(output_dir=output_dir, max_steps=max_steps,
        save_strategy=save_strategy, save_steps=save_steps, save_total_limit=save_total_limit,
        seed=seed, learning_rate=lr, use_cpu=use_cpu))


def build_v4_parser():
    parser = _BASE_PARSER()
    parser.set_defaults(output_dir="/root/GRPO-checkpoints", save_steps=50,
                        save_total_limit=64, probe_every_steps=50)
    return parser


class V4MonitorWriter:
    def __init__(self, writer): object.__setattr__(self, "_writer", writer)
    def __getattr__(self, name): return getattr(self._writer, name)
    def write_manifest(self, manifest):
        payload = dict(manifest)
        payload.update({
            "experiment": "GR_REC_ThinkExactSharpen_v4",
            "dataset_path": str(DATASET), "dataset_sha256": DATASET_SHA256,
            "dataset_guard": {"rows": 611, "unique_groups": 611, "think_only": True,
                              "probe4_overlap": 0},
            "parent_adapter_path": str(PARENT_ADAPTER),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "optimizer_initialization": "fresh AdamW",
            "optimizer": {"learning_rate": 1e-6, "weight_decay": 0.0, "scheduler": "constant"},
            "rollout_topology": "1xG4 CoT + 4xG8 Free + 4xG8 Official",
            "normalization": "eight independent G8; never G16/G64",
            "saturation": "Free32 and Official32 independently saturated iff A_plus_count > 8",
            "loss": "L_cot + 0.5*L_free_sid + 0.5*L_official_sid",
            "num_iterations": 2,
            "iteration2_reuse": ["CoT", "Free Sample8", "Official Sample8", "reward",
                                 "advantage", "saturation", "old_logp"],
            "fixed_probe": {"enabled": True, "group_ids": list(FIXED_PROBE4_IDS),
                            "free": "stochastic Sample8 full-SID scan without fixed domain",
                            "official": "production Beam32 ABC3 after fixed target domain"},
            "expected_optimizer_steps": EXPECTED_STEPS,
        })
        if _RUNTIME_PROVENANCE: payload.update(_RUNTIME_PROVENANCE)
        self._writer.write_manifest(payload)
    def write_exact_sharpen_v4(self, event):
        if self._writer.rank != 0: return False
        return self._writer._append("exact_sharpen_v4.jsonl", {"type": "exact_sharpen_v4", **event})


def v4_monitor_from_env(run_id, rank):
    return V4MonitorWriter(base_monitor_from_env(run_id, rank))


def install_bindings():
    baseline_runner.build_arg_parser = build_v4_parser
    baseline_runner.prepare_run_plan = prepare_v4_run_plan
    baseline_runner.load_model = load_model_and_capture
    baseline_runner.RecGRPOTrainer = ThinkExactSharpenTrainer
    baseline_runner.make_think_reward_func = make_runtime_reward
    baseline_runner.make_grpo_config = make_v4_config
    baseline_runner.monitor_from_env = v4_monitor_from_env
    baseline_runner.FixedProbeEvaluator = DualContractProbeEvaluator


def main(argv=None):
    global _RUNTIME_PROVENANCE
    _RUNTIME_PROVENANCE = assert_runtime_import_provenance()
    validate_parent_adapter()
    validate_dataset()
    install_bindings()
    baseline_runner.main(argv)


if __name__ == "__main__": main(sys.argv[1:])
