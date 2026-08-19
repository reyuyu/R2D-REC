"""CPU-only conditional hierarchical token-credit and dead-zero bridge tests."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

from grpo_trl_trainer import ROUTE_ID
import trl.trainer.grpo_trainer as trl_grpo
from nothink_bridge import (
    BRIDGE_DEAD_A,
    BRIDGE_OFF,
    plan_dead_zero_bridge,
    uniform_multi_positive_ce,
)
from nothink_hierarchical_credit import (
    HierarchyState,
    conditional_hierarchical_credits,
    find_final_sid_token_positions,
    hierarchy_state,
)
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer


STATE_BY_REWARD = {
    0: HierarchyState(True, True, False, False, False),
    0.5: HierarchyState(True, True, True, False, False),
    2: HierarchyState(True, True, True, True, False),
    8: HierarchyState(True, True, True, True, True),
}


def credits(rewards):
    return conditional_hierarchical_credits([STATE_BY_REWARD[value] for value in rewards])


# Required mixed archetype exact values.
mixed = credits([0, 0.5, 2, 8, 0, 0.5, 2, 8])
expected_cycle = [
    (-0.046875, 0.0, 0.0),
    (0.015625, -0.125, 0.0),
    (0.015625, 0.0625, -0.375),
    (0.015625, 0.0625, 0.375),
]
assert mixed == expected_cycle * 2

# Each partial hierarchy isolates the first stage with variance.
a_only = credits([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5])
assert [row[0] for row in a_only] == [-0.03125, 0.03125] * 4
assert all(row[1:] == (0.0, 0.0) for row in a_only)

b_only = credits([0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2])
assert [row[0] for row in b_only] == [0.0] * 8
assert [row[1] for row in b_only] == [-0.09375, 0.09375] * 4
assert all(row[2] == 0.0 for row in b_only)

c_only = credits([2, 8, 2, 8, 2, 8, 2, 8])
assert all(row[:2] == (0.0, 0.0) for row in c_only)
assert [row[2] for row in c_only] == [-0.375, 0.375] * 4

for reward in (0.5, 2, 8, 0):
    assert credits([reward] * 8) == [(0.0, 0.0, 0.0)] * 8

# Correct prefix invariants: suffix failure never makes the prefix credit negative.
for reward, row in zip([0, 0.5, 2, 8] * 2, mixed):
    if reward >= 0.5:
        assert row[0] >= 0
    if reward >= 2:
        assert row[1] >= 0

try:
    conditional_hierarchical_credits([STATE_BY_REWARD[0]] * 7)
except ValueError:
    pass
else:
    raise AssertionError("hierarchical credit must reject non-G8 input")

# State derivation is monotonic and independent of q_reward scalar values.
gold = [("prod", 1, 11, 111), ("prod", 2, 22, 222)]
assert hierarchy_state(None, gold, "prod") == HierarchyState(False, False, False, False, False)
assert hierarchy_state(("video", 1, 11, 111), gold, "prod") == HierarchyState(True, False, False, False, False)
assert hierarchy_state(("prod", 9, 99, 999), gold, "prod") == STATE_BY_REWARD[0]
assert hierarchy_state(("prod", 1, 99, 999), gold, "prod") == STATE_BY_REWARD[0.5]
assert hierarchy_state(("prod", 1, 11, 999), gold, "prod") == STATE_BY_REWARD[2]
assert hierarchy_state(("prod", 1, 11, 111), gold, "prod") == STATE_BY_REWARD[8]


# Final SID token positions use the last contiguous exact four-token block.
class PositionTokenizer:
    token_map = {
        "<|prod_begin|>": 10,
        "<s_a_1>": 11,
        "<s_b_11>": 12,
        "<s_c_111>": 13,
    }

    def encode(self, text, add_special_tokens=False):
        return [self.token_map[text]]


position_tokenizer = PositionTokenizer()
candidate_ids = [10, 11, 12, 13, 99, 10, 11, 12, 13, 100]
assert find_final_sid_token_positions(
    candidate_ids, ("prod", 1, 11, 111), position_tokenizer
) == (6, 7, 8)
try:
    find_final_sid_token_positions([10, 11, 99, 12, 13], ("prod", 1, 11, 111), position_tokenizer)
except RuntimeError:
    pass
else:
    raise AssertionError("non-contiguous final SID must fail closed")


# Global G8 credits map back to the correct rank/local candidate rows.
class RuntimeTokenizer(PositionTokenizer):
    pad_token_id = 0
    eos_token_id = 0
    token_map = {
        **PositionTokenizer.token_map,
        "<s_a_9>": 19,
        "<s_b_9>": 29,
        "<s_c_9>": 39,
    }

    def encode(self, text, add_special_tokens=False):
        return [self.token_map[text]] if text in self.token_map else [1, 2]


runtime_tokenizer = RuntimeTokenizer()
runtime_trainer = object.__new__(ThinkExactClampRecGRPOTrainer)
runtime_trainer.processing_class = runtime_tokenizer
runtime_trainer.reward_weights = torch.tensor([1.0])
runtime_trainer.accelerator = SimpleNamespace(process_index=0, num_processes=4, device="cpu")
gold_text = ["<|prod_begin|><s_a_1><s_b_11><s_c_111>"]
sid_cycle = [
    ("prod", 9, 9, 9),
    ("prod", 1, 9, 9),
    ("prod", 1, 11, 9),
    ("prod", 1, 11, 111),
]
text_cycle = [
    "<|prod_begin|><s_a_9><s_b_9><s_c_9>",
    "<|prod_begin|><s_a_1><s_b_9><s_c_9>",
    "<|prod_begin|><s_a_1><s_b_11><s_c_9>",
    "<|prod_begin|><s_a_1><s_b_11><s_c_111>",
]
id_cycle = [
    [10, 19, 29, 39],
    [10, 11, 29, 39],
    [10, 11, 12, 39],
    [10, 11, 12, 13],
]
runtime_inputs = [{
    "recommendation_group_id": "g0",
    "target_domain": "prod",
    "all_gold_sids": gold_text,
    "prompt": "prompt",
}] * 4
global_records = []
for group_id, ranks in (("g0", (0, 1)), ("g1", (2, 3))):
    for rank in ranks:
        for local_index, sid in enumerate(sid_cycle):
            global_records.append({
                "group_id": group_id,
                "rank": rank,
                "local_index": local_index,
                "predicted_sid": sid,
                "token_positions": (1, 2, 3),
                "gold_sids": [("prod", 1, 11, 111)],
                "target_domain": "prod",
                "prompt": "prompt",
            })
original_gather = trl_grpo.gather_object
try:
    trl_grpo.gather_object = lambda records: global_records
    runtime_trainer._prepare_nothink_credit_and_bridge(
        runtime_inputs,
        text_cycle,
        id_cycle,
        torch.tensor(([0, 0.5, 2, 8] * 4), dtype=torch.float32).unsqueeze(1),
    )
finally:
    trl_grpo.gather_object = original_gather
assert runtime_trainer._nothink_bridge_runtime["token_credits"] == tuple(expected_cycle)
assert runtime_trainer._nothink_bridge_runtime["token_positions"] == ((1, 2, 3),) * 4


# Token tensor writes only A/B/C positions and leaves every other token at zero.
trainer = object.__new__(ThinkExactClampRecGRPOTrainer)
trainer._nothink_bridge_runtime = {
    "token_credits": (mixed[0], mixed[1]),
    "token_positions": ((1, 2, 3), (2, 3, 4)),
}
output = {
    "completion_ids": torch.ones((2, 6), dtype=torch.long),
    "advantages": torch.tensor([999.0, -999.0]),
}
trainer._attach_nothink_token_advantages(output)
expected = torch.zeros((2, 6))
expected[0, 1:4] = torch.tensor(mixed[0])
expected[1, 2:5] = torch.tensor(mixed[1])
torch.testing.assert_close(output["token_advantages"], expected)


# NoThink PPO consumes token_advantages and ignores the debug sequence advantage.
trainer.args = SimpleNamespace(delta=None)
trainer.epsilon_low = 0.2
trainer.epsilon_high = 0.2
trainer.loss_type = "grpo"
trainer.current_gradient_accumulation_steps = 1
trainer._smoke_rollout_id = 1
trainer._smoke_policy_epoch = {}
trainer._smoke_log = [{}]
trainer._detailed_monitor = False
trainer.state = SimpleNamespace(global_step=1)
logps = torch.nn.Parameter(torch.zeros((2, 6)))
trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: (logps, None)
loss_inputs = {
    "prompt_ids": torch.ones((2, 2), dtype=torch.long),
    "prompt_mask": torch.ones((2, 2), dtype=torch.long),
    "completion_ids": torch.ones((2, 6), dtype=torch.long),
    "completion_mask": torch.ones((2, 6), dtype=torch.long),
    "route_id": torch.full((2,), ROUTE_ID["no_think"]),
    "advantages": torch.tensor([1e6, -1e6]),
    "token_advantages": expected,
}
first_loss = trainer._compute_nothink_token_loss(None, loss_inputs)
loss_inputs["advantages"] = -loss_inputs["advantages"]
second_loss = trainer._compute_nothink_token_loss(None, loss_inputs)
torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
first_loss.backward()
assert logps.grad is not None
assert torch.equal(logps.grad == 0, expected == 0)


# Branch B is gone: only exact all-zero can activate the Gold-A bridge.
assert plan_dead_zero_bridge([0] * 8, gold, "prod") == plan_dead_zero_bridge([0] * 8, gold, "prod")
dead_plan = plan_dead_zero_bridge([0] * 8, gold, "prod")
assert dead_plan.branch == BRIDGE_DEAD_A and dead_plan.gold_a_targets == (1, 2)
for rewards in ([0.5] * 8, [2] * 8, [8] * 8, [0, 0.5, 2, 8] * 2):
    assert plan_dead_zero_bridge(rewards, gold, "prod").branch == BRIDGE_OFF

# Dead-zero teacher remains uniform mean CE over deduplicated Gold A.
teacher_logits = torch.zeros(32, dtype=torch.float64, requires_grad=True)
single = uniform_multi_positive_ce(teacher_logits, [3])
multiple = uniform_multi_positive_ce(teacher_logits, [3, 5, 3])
assert single.ndim == 0 and multiple.ndim == 0
multiple.backward()
assert teacher_logits.grad[3] < 0 and teacher_logits.grad[5] < 0


# The retained teacher path is one A-query row; global-off performs no forward.
class TeacherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher_logits = torch.nn.Parameter(torch.zeros(64))
        self.forward_count = 0

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
        self.forward_count += 1
        assert input_ids.shape[0] == 1 and logits_to_keep == 1
        return SimpleNamespace(logits=self.teacher_logits.view(1, 1, -1))


def bridge_runtime(plan, global_active):
    return {
        "plan": plan,
        "global_active": global_active,
        "ddp_group_weight": 1.0,
        "prompt_ids": (1, 2),
        "domain_id": 3,
        "a_target_ids": (5, 6) if plan.active else (),
        "rewards": (0.0,) * 8,
        "gold_unique_a_count": 2,
        "pred_unique_a_count": 1,
        "gold_a_candidate_hit_rate": 0.0,
        "gold_ab_candidate_hit_rate": 0.0,
        "exact_candidate_hit_rate": 0.0,
        "wrong_domain_rate": 0.0,
        "valid_sid_rate": 1.0,
    }


bridge_trainer = object.__new__(ThinkExactClampRecGRPOTrainer)
bridge_trainer.bridge_lambda = 0.02
bridge_trainer.current_gradient_accumulation_steps = 1
bridge_trainer._smoke_rollout_id = 2
bridge_trainer._smoke_log = [{}]
bridge_trainer.state = SimpleNamespace(global_step=2)
bridge_trainer.accelerator = SimpleNamespace(process_index=0)
bridge_trainer._monitor_enabled = lambda: False
primary = torch.tensor(1.0, requires_grad=True)
bridge_trainer._compute_nothink_token_loss = lambda model, inputs: primary
teacher_model = TeacherModel()
bridge_trainer._nothink_bridge_runtime = bridge_runtime(dead_plan, True)
total = bridge_trainer._compute_loss(
    teacher_model, {"route_id": torch.tensor([ROUTE_ID["no_think"]])}
)
assert total > primary and teacher_model.forward_count == 1
bridge_trainer._nothink_bridge_runtime = bridge_runtime(
    plan_dead_zero_bridge([0.5] * 8, gold, "prod"), False
)
off_total = bridge_trainer._compute_loss(
    teacher_model, {"route_id": torch.tensor([ROUTE_ID["no_think"]])}
)
assert off_total is primary and teacher_model.forward_count == 1

source = inspect.getsource(ThinkExactClampRecGRPOTrainer._compute_nothink_token_loss)
assert "token_advantages" in source
assert "advantages.unsqueeze(1)" not in source
trainer_source = inspect.getsource(ThinkExactClampRecGRPOTrainer)
for removed in ("a_collapse_ab_bridge", "missing_a_targets", "current_b_targets", "b_target_ids"):
    assert removed not in trainer_source

print("NOTHINK HIERARCHICAL TOKEN CREDIT CPU TESTS PASSED")
