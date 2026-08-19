"""CPU-only formula, trainer isolation, monitor and runner tests."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

from audit_think_counterfactual import summarize
from grpo_trl_trainer import M_NO, M_THINK, RecGRPOTrainer, group_advantages_population
from run_think_exact_clamp_train import (
    MAX_EXPERIMENT_STEPS,
    RUN_ID_PREFIX,
    _ManifestWriter,
    validate_experiment_args,
)
from think_diagnostics import interest_diagnostics
from think_exact_clamp import think_exact_clamp_advantages
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer


ARCHETYPES = [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 0, 0.5], [-0.015625, -0.015625, -0.015625, 0.046875]),
    ([0, 0, 0, 2], [-0.0625, -0.0625, -0.0625, 0.1875]),
    ([0, 0, 0, 8], [-0.25, -0.25, -0.25, 0.75]),
    ([8, 8, 8, 9], [0, 0, 0, 0.09375]),
    ([8, 8, 8, 12], [0, 0, 0, 0.375]),
    ([8, 2, 3, 9], [0.3125, -0.4375, -0.3125, 0.4375]),
    ([8, 8, 8, 8], [0, 0, 0, 0]),
    ([12, 12, 12, 12], [0, 0, 0, 0]),
    ([12, 12, 12, 14], [0, 0, 0, 0.1875]),
]
for rewards, expected in ARCHETYPES:
    actual = think_exact_clamp_advantages(torch.tensor(rewards, dtype=torch.float64))
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    high = [value for reward, value in zip(rewards, actual.tolist()) if reward >= 8]
    assert not any(value < 0 for value in high)

# Two G4 groups do not share a mean; dtype and device are preserved.
combined = torch.tensor(ARCHETYPES[3][0] + ARCHETYPES[8][0], dtype=torch.float32)
actual = think_exact_clamp_advantages(combined)
torch.testing.assert_close(actual[:4], torch.tensor(ARCHETYPES[3][1], dtype=torch.float32))
torch.testing.assert_close(actual[4:], torch.tensor(ARCHETYPES[8][1], dtype=torch.float32))
assert actual.dtype == combined.dtype and actual.device == combined.device

for bad, group_size, error in (
    (torch.tensor([[0.0]]), 4, ValueError),
    (torch.tensor([0, 1, 2, 3]), 4, TypeError),
    (torch.tensor([0.0] * 8), 8, ValueError),
    (torch.tensor([0.0] * 5), 4, ValueError),
):
    try:
        think_exact_clamp_advantages(bad, group_size)
    except error:
        pass
    else:
        raise AssertionError("invalid Think formula input must be rejected")

# NoThink parity: the child returns the exact same parent output object/tensor.
trainer = object.__new__(ThinkExactClampRecGRPOTrainer)
trainer._think_exact_clamp_route = "no_think"
nothink_rewards = torch.tensor([0, 0, 0, 0.5, 0, 0, 0, 0], dtype=torch.float32)
baseline_advantages = group_advantages_population(nothink_rewards, M_NO)
parent_output = {"advantages": baseline_advantages, "sentinel": object()}
original_generate = RecGRPOTrainer._generate_and_score_completions
try:
    RecGRPOTrainer._generate_and_score_completions = lambda self, inputs: parent_output
    child_output = trainer._generate_and_score_completions([])
finally:
    RecGRPOTrainer._generate_and_score_completions = original_generate
assert child_output is parent_output
assert child_output["advantages"] is baseline_advantages
assert torch.equal(child_output["advantages"], group_advantages_population(nothink_rewards, M_NO))

for rewards in (
    torch.tensor([-1, -0.25, 0, 0.5, 2, 8, 0, 0], dtype=torch.float64),
    torch.tensor([8, 8, 8, 8, 0, 0, 0, 0], dtype=torch.float64),
):
    parent = group_advantages_population(rewards, M_NO)
    trainer._think_exact_clamp_route = "no_think"
    output = {"advantages": parent}
    try:
        RecGRPOTrainer._generate_and_score_completions = lambda self, inputs, value=output: value
        returned = trainer._generate_and_score_completions([])
    finally:
        RecGRPOTrainer._generate_and_score_completions = original_generate
    assert returned is output and returned["advantages"] is parent

# Reward output parity for NoThink: planning observes but does not replace the parent output.
rewards_per_func = torch.tensor([[1.0, float("nan")], [2.0, float("nan")]])
original_calculate = RecGRPOTrainer._calculate_rewards
planned = []
trainer._prepare_nothink_bridge = lambda inputs, completions, rewards: planned.append(
    (inputs, completions, rewards)
)
try:
    RecGRPOTrainer._calculate_rewards = lambda self, inputs, *args, **kwargs: rewards_per_func
    returned_rewards = trainer._calculate_rewards(
        [{"route": "no_think"}], None, ["completion"], None
    )
finally:
    RecGRPOTrainer._calculate_rewards = original_calculate
assert returned_rewards is rewards_per_func
assert not hasattr(trainer, "_think_exact_clamp_global_rewards")
assert planned[0][2] is rewards_per_func

original_write_monitor = RecGRPOTrainer._write_rollout_monitor
monitor_sentinel = object()
trainer._monitor = None
trainer._nothink_bridge_runtime = None
try:
    RecGRPOTrainer._write_rollout_monitor = lambda self, *args, **kwargs: monitor_sentinel
    trainer._think_exact_clamp_route = "no_think"
    returned_monitor = trainer._write_rollout_monitor(
        {}, [], [], [], [], None, [], None, []
    )
finally:
    RecGRPOTrainer._write_rollout_monitor = original_write_monitor
assert returned_monitor is monitor_sentinel

# Think public path replaces only advantages.
trainer._think_exact_clamp_route = "think"
trainer._think_exact_clamp_global_rewards = torch.tensor([8, 8, 8, 12], dtype=torch.float32)
trainer._think_exact_clamp_advantages_list = [0, 0, 0, 0.375]
trainer.num_generations = 4
trainer.accelerator = type("Accelerator", (), {"process_index": 0})()
trainer._detailed_monitor = False
trainer._parity_audit = False
trainer._parity_log = []
try:
    RecGRPOTrainer._generate_and_score_completions = lambda self, inputs: {
        "advantages": torch.full((4,), 99.0), "sentinel": "unchanged"
    }
    think_output = trainer._generate_and_score_completions([])
finally:
    RecGRPOTrainer._generate_and_score_completions = original_generate
torch.testing.assert_close(think_output["advantages"], torch.tensor([0, 0, 0, 0.375]))
assert think_output["sentinel"] == "unchanged"

# Diagnostic parser is monitor-only and preserves raw-vs-grounded semantics.
sid = "<|prod_begin|><s_a_1><s_b_2><s_c_3>"
fake = "<think>\n【兴趣归纳】\n1. A：" + sid + "\n2. B：无证据\n</think>"
diagnostic = interest_diagnostics(fake, "history " + sid)
assert diagnostic == {"raw_n": 2, "grounded_n": 1, "coverage": 0.5, "parser_success": True}

class FakeMonitor:
    enabled = True
    def __init__(self): self.events = []
    def _append(self, path, event): self.events.append((path, event)); return True

trainer._monitor = FakeMonitor()
trainer.state = type("State", (), {"global_step": 12})()
trainer.accelerator = type("Accelerator", (), {"process_index": 0})()
inputs = [{
    "prompt": "history " + sid,
    "all_gold_sids": [("prod", 1, 2, 3)],
    "recommendation_group_id": "g1",
}] * 4
beam_call = {"local_results": [
    {"exact": 1, "ab": 0, "a": 0, "beam_sids": [("prod", 1, 2, 3)]},
    {"exact": 0, "ab": 1, "a": 0, "beam_sids": []},
    {"exact": 0, "ab": 0, "a": 1, "beam_sids": []},
    {"exact": 0, "ab": 0, "a": 0, "beam_sids": []},
]}
trainer._write_think_exact_clamp_diagnostics(
    {"rollout_id": 7}, inputs, [[1, 2]] * 4, [fake] * 4, [8, 8, 8, 12], beam_call
)
event = trainer._monitor.events[0][1]["groups"][0]
assert event["rewards"] == [8, 8, 8, 12]
assert event["exact_candidate_count"] == 4 and event["multi_positive_candidate_count"] == 1
assert event["high_quality_negative_rate"] == 0
assert event["distinct_exact_gold_sids_covered_across_g4"] == 1
assert event["candidates"][0]["coverage"] == 0.5

# Runner enforces fresh BATA, prefix, and the bounded future plan.
valid = validate_experiment_args([
    "--run-id", RUN_ID_PREFIX + "TEST", "--max-steps", "1500"
])
assert valid.max_steps == 1500 and valid.resume_from_checkpoint is None
assert valid.seed == 20260816 and valid.probe_seed == 20260818
assert valid.lr == 1e-6 and valid.probe_groups == 0
assert MAX_EXPERIMENT_STEPS == 1500


class ManifestSink:
    def write_manifest(self, payload):
        return payload


manifest = _ManifestWriter(ManifestSink()).write_manifest({})
assert manifest["parent"] == "GR_REC_v1 / original BATA adapter"
assert manifest["initialization"] == "fresh original BATA adapter"
assert manifest["nothink_bridge"]["lambda"] == 0.02
assert manifest["future_checkpoints"] == [600, 800, 1000, 1200, 1400, 1500]
same_run = validate_experiment_args([
    "--run-id", RUN_ID_PREFIX + "TEST", "--max-steps", "1500",
    "--resume-from-checkpoint",
    "/data/GRPO/outputs/formal/" + RUN_ID_PREFIX + "TEST/checkpoint-200",
])
assert same_run.resume_from_checkpoint.endswith("checkpoint-200")
for argv in (
    ["--run-id", "wrong", "--max-steps", "1500"],
    ["--run-id", RUN_ID_PREFIX + "TEST"],
    ["--run-id", RUN_ID_PREFIX + "TEST", "--max-steps", "1501"],
    ["--run-id", RUN_ID_PREFIX + "TEST", "--max-steps", "100",
     "--resume-from-checkpoint", "/data/GRPO/outputs/formal/OTHER/checkpoint-100"],
):
    try:
        validate_experiment_args(argv)
    except ValueError:
        pass
    else:
        raise AssertionError(f"runner arguments should be rejected: {argv}")

# Synthetic audit confirms the two protected high-quality conditions.
audit, _ = summarize([
    {"source": "test", "route": "think", "domain": "video", "gold_count": 5,
     "rewards": [8, 8, 8, 12]},
    {"source": "test", "route": "think", "domain": "prod", "gold_count": 2,
     "rewards": [0, 0, 0, 0.5]},
])
assert audit["reward_levels"]["8"]["think_exact_clamp"]["negative_rate"] == 0
assert audit["reward_levels"][">8"]["think_exact_clamp"]["negative_rate"] == 0

assert M_THINK == 4 and M_NO == 8
source = inspect.getsource(ThinkExactClampRecGRPOTrainer)
for forbidden in ("def _generate(", "def forward(", "all_gather", "synchronize"):
    assert forbidden not in source

print("THINK EXACTCLAMP CPU TESTS PASSED")
