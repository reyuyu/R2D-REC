"""CPU-only planner, CE, loss isolation, DDP and runtime contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

from grpo_trl_trainer import ROUTE_ID, RecGRPOTrainer
import trl.trainer.grpo_trainer as trl_grpo
from nothink_bridge import (
    BRIDGE_COLLAPSE_AB,
    BRIDGE_DEAD_A,
    BRIDGE_OFF,
    BridgePlan,
    ddp_group_weight,
    plan_nothink_bridge,
    require_single_token,
    uniform_multi_positive_ce,
)
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer
import think_exact_clamp_trainer as trainer_module


def gold(unique_a=3):
    values = []
    for a in range(1, unique_a + 1):
        values.extend([("prod", a, 10 + a, 1), ("prod", a, 20 + a, 2)])
    return values


def predicted(a=1, domain="prod"):
    return [(domain, a, 99, index) for index in range(8)]


# Planner cases A-J and branch priority.
plan = plan_nothink_bridge([0] * 8, predicted(9), gold(), "prod")
assert plan.branch == BRIDGE_DEAD_A and plan.gold_a_targets == (1, 2, 3)
assert not plan.missing_a_targets and not plan.current_b_targets

plan = plan_nothink_bridge([0.5] * 8, predicted(1), gold(), "prod")
assert plan.branch == BRIDGE_COLLAPSE_AB
assert plan.missing_a_targets == (2, 3) and plan.current_b_targets == (11, 21)
assert plan.current_a == 1

assert plan_nothink_bridge([0.5] * 8, predicted(1), gold(2), "prod").branch == BRIDGE_OFF
assert plan_nothink_bridge([0] * 8, predicted(9), gold(), "prod").branch == BRIDGE_DEAD_A
assert plan_nothink_bridge([0.5] * 7 + [2], predicted(1), gold(), "prod").branch == BRIDGE_OFF
assert plan_nothink_bridge([0.5] * 7 + [8], predicted(1), gold(), "prod").branch == BRIDGE_OFF
assert plan_nothink_bridge([8] * 8, predicted(1), gold(), "prod").branch == BRIDGE_OFF
assert plan_nothink_bridge([2] * 8, predicted(1), gold(), "prod").branch == BRIDGE_OFF
mixed_a = predicted(1)[:4] + predicted(2)[4:]
assert plan_nothink_bridge([0.5] * 8, mixed_a, gold(), "prod").branch == BRIDGE_OFF
invalid = predicted(1)[:-1] + [None]
wrong_domain = predicted(1)[:-1] + [("video", 1, 11, 1)]
assert plan_nothink_bridge([0.5] * 8, invalid, gold(), "prod").branch == BRIDGE_OFF
assert plan_nothink_bridge([0.5] * 8, wrong_domain, gold(), "prod").branch == BRIDGE_OFF

try:
    plan_nothink_bridge([0] * 7, predicted(1), gold(), "prod")
except ValueError:
    pass
else:
    raise AssertionError("planner must reject non-G8 input")


# Uniform multi-positive CE: single/multiple A, dedup, B, mean-not-sum.
def assert_all_targets_rise(targets):
    logits = torch.zeros(32, dtype=torch.float64, requires_grad=True)
    before = torch.softmax(logits.detach(), -1)
    loss = uniform_multi_positive_ce(logits, targets)
    loss.backward()
    assert all(logits.grad[target] < 0 for target in set(targets))
    after = torch.softmax((logits - 0.1 * logits.grad).detach(), -1)
    assert all(after[target] > before[target] for target in set(targets))
    return loss.detach()


assert_all_targets_rise([3])
multiple_a = assert_all_targets_rise([3, 5, 7])
duplicate_a = assert_all_targets_rise([3, 5, 7, 3, 5])
torch.testing.assert_close(multiple_a, duplicate_a, rtol=0, atol=0)
assert_all_targets_rise([11, 13])
logits = torch.linspace(-1, 1, 32, dtype=torch.float64)
targets = [2, 8, 17]
manual_mean = -torch.log_softmax(logits, -1)[targets].sum() / len(targets)
torch.testing.assert_close(uniform_multi_positive_ce(logits, targets), manual_mean)
assert not torch.equal(manual_mean, -torch.log_softmax(logits, -1)[targets].sum())


# Token contract fails closed.
class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [7] if text != "bad" else [7, 8]


assert require_single_token(TinyTokenizer(), "good") == 7
try:
    require_single_token(TinyTokenizer(), "bad")
except RuntimeError:
    pass
else:
    raise AssertionError("multi-token bridge target must fail closed")


# DDP group-level weighting: two ranks see each group, but DDP mean is one mean/group.
weight = ddp_group_weight(world_size=4, global_group_count=2, ranks_for_group=2)
assert weight == 1.0
rank_losses = [3.0 * weight, 3.0 * weight, 7.0 * weight, 7.0 * weight]
assert sum(rank_losses) / 4 == (3.0 + 7.0) / 2


# Runtime planning gathers compact metadata and plans each global group once.
class PlanningTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        special = {
            "<|prod_begin|>": 3,
            "<s_a_1>": 5,
            "<s_a_2>": 6,
            "<s_a_3>": 7,
        }
        return [special[text]] if text in special else [1, 2]


planning_trainer = object.__new__(ThinkExactClampRecGRPOTrainer)
planning_trainer.processing_class = PlanningTokenizer()
planning_trainer.reward_weights = torch.tensor([1.0])
planning_trainer.accelerator = SimpleNamespace(process_index=0, num_processes=4, device="cpu")
gold_text = [
    "<|prod_begin|><s_a_1><s_b_11><s_c_1>",
    "<|prod_begin|><s_a_2><s_b_12><s_c_1>",
    "<|prod_begin|><s_a_3><s_b_13><s_c_1>",
]
local_inputs = [{
    "recommendation_group_id": "g0",
    "target_domain": "prod",
    "all_gold_sids": gold_text,
    "prompt": "prompt",
}] * 4
local_completions = ["<|prod_begin|><s_a_9><s_b_9><s_c_9>"] * 4
global_records = []
for group_id, ranks in (("g0", (0, 1)), ("g1", (2, 3))):
    for rank in ranks:
        global_records.extend([{
            "group_id": group_id,
            "rank": rank,
            "predicted_sid": ("prod", 9, 9, 9),
            "gold_sids": [("prod", 1, 11, 1), ("prod", 2, 12, 1), ("prod", 3, 13, 1)],
            "target_domain": "prod",
            "prompt": "prompt",
        }] * 4)
plan_count = 0
original_gather = trl_grpo.gather_object
original_plan = trainer_module.plan_nothink_bridge


def counted_plan(*args, **kwargs):
    global plan_count
    plan_count += 1
    return original_plan(*args, **kwargs)


try:
    trl_grpo.gather_object = lambda records: global_records
    trainer_module.plan_nothink_bridge = counted_plan
    planning_trainer._prepare_nothink_bridge(
        local_inputs, local_completions, torch.zeros((16, 1))
    )
finally:
    trl_grpo.gather_object = original_gather
    trainer_module.plan_nothink_bridge = original_plan
assert plan_count == 2
assert planning_trainer._nothink_bridge_runtime["plan"].branch == BRIDGE_DEAD_A
assert planning_trainer._nothink_bridge_runtime["ddp_group_weight"] == 1.0


def runtime(branch=BRIDGE_DEAD_A, global_active=True):
    plan = BridgePlan(
        branch=branch,
        gold_a_targets=(1, 2, 3),
        current_a=1 if branch == BRIDGE_COLLAPSE_AB else None,
        missing_a_targets=(2, 3) if branch == BRIDGE_COLLAPSE_AB else (),
        current_b_targets=(11, 21) if branch == BRIDGE_COLLAPSE_AB else (),
    )
    return {
        "plan": plan,
        "global_active": global_active,
        "ddp_group_weight": 1.0,
        "prompt_ids": (1, 2),
        "domain_id": 3,
        "current_a_id": 4 if plan.current_a is not None else None,
        "a_target_ids": (5, 6),
        "b_target_ids": (7, 8) if plan.current_a is not None else (),
        "rewards": (0.0,) * 8,
        "gold_unique_a_count": 3,
        "pred_unique_a_count": 1,
        "gold_a_candidate_hit_rate": 0.0,
        "gold_ab_candidate_hit_rate": 0.0,
        "exact_candidate_hit_rate": 0.0,
        "wrong_domain_rate": 0.0,
        "valid_sid_rate": 1.0,
    }


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(16))
        self.forward_count = 0

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
        self.forward_count += 1
        assert logits_to_keep == 1
        values = self.logits.view(1, 1, -1).expand(input_ids.shape[0], 1, -1)
        return SimpleNamespace(logits=values)


def bare_trainer(bridge_runtime, bridge_lambda=0.02):
    trainer = object.__new__(ThinkExactClampRecGRPOTrainer)
    trainer.bridge_lambda = bridge_lambda
    trainer._nothink_bridge_runtime = bridge_runtime
    trainer.processing_class = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    trainer.current_gradient_accumulation_steps = 1
    trainer._smoke_rollout_id = 4
    trainer._smoke_log = [{}]
    trainer.state = SimpleNamespace(global_step=10)
    trainer.accelerator = SimpleNamespace(process_index=0)
    trainer._monitor_enabled = lambda: False
    return trainer


original_compute = RecGRPOTrainer._compute_loss
primary_parameter = torch.nn.Parameter(torch.tensor(2.0))
primary = primary_parameter.square()
try:
    RecGRPOTrainer._compute_loss = lambda self, model, inputs, **kwargs: primary
    model = FakeModel()

    # Think isolation and lambda=0 NoThink parity return the exact primary tensor.
    trainer = bare_trainer(runtime(), bridge_lambda=0.02)
    result = trainer._compute_loss(model, {"route_id": torch.tensor([ROUTE_ID["think"]])})
    assert result is primary and model.forward_count == 0
    trainer = bare_trainer(runtime(), bridge_lambda=0.0)
    result = trainer._compute_loss(model, {"route_id": torch.tensor([ROUTE_ID["no_think"]])})
    assert result is primary and model.forward_count == 0
    result.backward(retain_graph=True)
    assert primary_parameter.grad.item() == 4.0
    primary_parameter.grad.zero_()

    # Global-off rollouts add no teacher forward.
    trainer = bare_trainer(runtime(branch=BRIDGE_OFF, global_active=False))
    result = trainer._compute_loss(model, {"route_id": torch.tensor([ROUTE_ID["no_think"]])})
    assert result is primary and model.forward_count == 0

    # One immutable rollout plan is reused; current-policy teacher logits run twice.
    shared_runtime = runtime(branch=BRIDGE_COLLAPSE_AB)
    trainer = bare_trainer(shared_runtime)
    first = trainer._compute_loss(model, {"route_id": torch.tensor([ROUTE_ID["no_think"]])})
    second = trainer._compute_loss(model, {"route_id": torch.tensor([ROUTE_ID["no_think"]])})
    assert trainer._nothink_bridge_runtime is shared_runtime
    assert model.forward_count == 2
    assert first > primary and second > primary
finally:
    RecGRPOTrainer._compute_loss = original_compute

print("NOTHINK BRIDGE CPU TESTS PASSED")
