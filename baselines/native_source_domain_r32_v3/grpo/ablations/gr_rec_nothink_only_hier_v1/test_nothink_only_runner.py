"""CPU-only route, sampler, checkpoint, parent and manifest contract tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

import run_grpo_trl_train as baseline_runner
from grpo_trl_trainer import ROUTE_ID, ROUTE_LOSS_W, RouteAwareRepeatSampler
from run_nothink_only_hier_train import (
    AttributionProbeEvaluator,
    EXPECTED_ADAPTER,
    EXPECTED_BASE,
    FORMAL_RUN_ID,
    REQUESTED_CHECKPOINT_STEPS,
    RUN_ID_PREFIX,
    FormalCheckpointCallback,
    NoThinkOnlyHierTrainer,
    _ManifestWriter,
    audit_nothink_only_sampler,
    build_nothink_only_dataset,
    formal_checkpoint_steps,
    validate_experiment_args,
)
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer


with tempfile.TemporaryDirectory() as tmp:
    data_path = Path(tmp) / "routes.jsonl"
    rows = []
    for index in range(6):
        group_id = f"g{index}"
        for route in ("think", "no_think"):
            rows.append({
                "prompt": f"{route}-{group_id}",
                "route": route,
                "recommendation_group_id": group_id,
                "target_domain": "video",
                "all_gold_sids": [
                    "<|video_begin|><s_a_1><s_b_2><s_c_3>"
                ],
            })
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    mixed = baseline_runner.build_route_dataset(
        data_path, n_groups=6, seed=20260816, chunk=8,
        exclude_group_ids=["g0", "g1"],
    )
    dataset = build_nothink_only_dataset(mixed)
    assert len(dataset) == 4
    assert {row["route"] for row in dataset} == {"no_think"}
    sampler = RouteAwareRepeatSampler(
        dataset, generation_batch_size=16, repeat_count=2, shuffle=False
    )
    audit = audit_nothink_only_sampler(dataset, sampler)
    assert audit["nothink_unique_groups"] == 4
    assert audit["fresh_rollout_count"] == 2
    assert audit["unique_groups_per_global_rollout"] == 2
    assert audit["optimizer_steps"] == 4
    assert audit["think_rollouts"] == 0
    assert audit["think_optimizer_steps"] == 0
    parity_dataset = build_nothink_only_dataset(mixed, exclude_group_ids=["g2", "g3"])
    parity_sampler = RouteAwareRepeatSampler(
        parity_dataset, generation_batch_size=16, repeat_count=2, shuffle=False
    )
    parity_audit = audit_nothink_only_sampler(parity_dataset, parity_sampler)
    assert parity_audit["nothink_unique_groups"] == 2
    assert parity_audit["optimizer_steps"] == 2


assert issubclass(NoThinkOnlyHierTrainer, ThinkExactClampRecGRPOTrainer)
assert AttributionProbeEvaluator.__name__ == "AttributionProbeEvaluator"
assert ROUTE_LOSS_W["no_think"] == 0.5
assert REQUESTED_CHECKPOINT_STEPS == (250, 500, 666, 750, 1000, 1250)
assert formal_checkpoint_steps(1544) == (250, 500, 666, 750, 1000, 1250, 1544)
try:
    formal_checkpoint_steps(1543)
except ValueError:
    pass
else:
    raise AssertionError("odd final checkpoint must be rejected")


class Control:
    should_save = False


callback = FormalCheckpointCallback(1544)
for step in (248, 252, 664, 668, 1542):
    control = Control()
    callback.on_step_end(None, type("State", (), {"global_step": step})(), control)
    assert control.should_save is False
for step in formal_checkpoint_steps(1544):
    control = Control()
    callback.on_step_end(None, type("State", (), {"global_step": step})(), control)
    assert control.should_save is True


class Sink:
    def write_manifest(self, payload):
        return payload


manifest = _ManifestWriter(Sink()).write_manifest({})
assert manifest["training_routes"] == ["no_think"]
assert manifest["think_optimizer_updates"] == 0
assert manifest["training_beam32"] is False
assert manifest["nothink_route_multiplier"] == 0.5
assert manifest["dead_zero_bridge"] == {"branch": "gold_A_only", "lambda": 0.02}

probe = object.__new__(AttributionProbeEvaluator)
parent_think = AttributionProbeEvaluator.__mro__[1]._think
try:
    AttributionProbeEvaluator.__mro__[1]._think = lambda self: {
        "prompt": "history <|prod_begin|><s_a_1><s_b_2><s_c_3>",
        "candidates": [{
            "completion": (
                "<think>\n【兴趣归纳】\n1. X："
                "<|prod_begin|><s_a_1><s_b_2><s_c_3>\n</think>"
            )
        }],
    }
    probed = probe._think()
finally:
    AttributionProbeEvaluator.__mro__[1]._think = parent_think
assert probed["candidates"][0]["raw_n"] == 1
assert probed["candidates"][0]["grounded_n"] == 1
assert probed["candidates"][0]["coverage"] == 1.0

args = validate_experiment_args(["--run-id", FORMAL_RUN_ID])
assert args.resume_from_checkpoint is None
assert args.lr == 1e-6 and args.seed == 20260816
assert EXPECTED_BASE.endswith("onereason-8b-pretrain-competition")
assert "BATA-BASELINE-R32-2E-GC04-4GPU" in EXPECTED_ADAPTER
for bad in (
    ["--run-id", "wrong"],
    ["--run-id", FORMAL_RUN_ID, "--resume-from-checkpoint", "/tmp/other/checkpoint-2"],
):
    try:
        validate_experiment_args(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid runner arguments accepted: {bad}")

trainer = object.__new__(NoThinkOnlyHierTrainer)
try:
    trainer._calculate_rewards([{"route": "think"}])
except RuntimeError:
    pass
else:
    raise AssertionError("Think reward batch was not rejected")
try:
    trainer._compute_loss(None, {"route_id": torch.tensor([ROUTE_ID["think"]])})
except RuntimeError:
    pass
else:
    raise AssertionError("Think loss batch was not rejected")

print("NOTHINK-ONLY RUNNER CPU TESTS PASSED")
