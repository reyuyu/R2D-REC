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
    apply_domain_text_alignment_gate,
    conditional_hierarchical_credits,
    find_final_sid_token_positions,
    hierarchy_state,
    locate_domain_commitment_token,
    locate_text_domain_token,
)
from think_exact_clamp_trainer import ThinkExactClampRecGRPOTrainer


STATE_BY_REWARD = {
    -0.25: HierarchyState(True, False, False, False, False),
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
    (0.0, -0.046875, 0.0, 0.0),
    (0.0, 0.015625, -0.09375, 0.0),
    (0.0, 0.015625, 0.09375, -0.1875),
    (0.0, 0.015625, 0.09375, 0.5625),
]
assert mixed == expected_cycle * 2

# G8-anchored milestone regression A-J.
test_a = credits([0] * 7 + [0.5])
assert [row[1] for row in test_a] == [-0.0078125] * 7 + [0.0546875]
test_b = credits([0] * 7 + [2])
assert test_b[-1] == (0.0, 0.0546875, 0.1640625, 0.0)
test_c = credits([0] * 7 + [8])
assert test_c[-1] == (0.0, 0.0546875, 0.1640625, 0.65625)
test_d = credits([0] * 6 + [2, 2])
assert test_d[-2:] == [(0.0, 0.046875, 0.140625, 0.0)] * 2
test_e = credits([0.5] * 7 + [2])
assert [row[1] for row in test_e] == [0.0] * 8
assert [row[2] for row in test_e] == [-0.0234375] * 7 + [0.1640625]
test_f = credits([2] * 7 + [8])
assert all(row[1:3] == (0.0, 0.0) for row in test_f)
assert [row[3] for row in test_f] == [-0.09375] * 7 + [0.65625]
assert credits([8] * 8) == [(0.0, 0.0, 0.0, 0.0)] * 8
assert credits([0] * 8) == [(0.0, 0.0, 0.0, 0.0)] * 8
assert credits([-0.25] * 8) == [(0.0, 0.0, 0.0, 0.0)] * 8
assert mixed == expected_cycle * 2

# GPU-observed wrong-domain/correct-domain topology receives Domain-only credit.
domain_only = credits([-0.25, 0, -0.25, -0.25, -0.25, 0, -0.25, -0.25])
assert [row[0] for row in domain_only] == [
    -0.0078125, 0.0234375, -0.0078125, -0.0078125,
    -0.0078125, 0.0234375, -0.0078125, -0.0078125,
]
assert all(row[1:] == (0.0, 0.0, 0.0) for row in domain_only)
gated_domain_only = apply_domain_text_alignment_gate(domain_only, False)
assert all(row[0] == 0 for row in gated_domain_only)
assert [row[1:] for row in gated_domain_only] == [row[1:] for row in domain_only]

# Each partial hierarchy isolates the first stage with variance.
a_only = credits([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5])
assert [row[1] for row in a_only] == [-0.03125, 0.03125] * 4
assert all((row[0], *row[2:]) == (0.0, 0.0, 0.0) for row in a_only)

b_only = credits([0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2])
assert all(row[:2] == (0.0, 0.0) for row in b_only)
assert [row[2] for row in b_only] == [-0.09375, 0.09375] * 4
assert all(row[3] == 0.0 for row in b_only)

c_only = credits([2, 8, 2, 8, 2, 8, 2, 8])
assert all(row[:3] == (0.0, 0.0, 0.0) for row in c_only)
assert [row[3] for row in c_only] == [-0.375, 0.375] * 4

for reward in (0.5, 2, 8, 0):
    assert credits([reward] * 8) == [(0.0, 0.0, 0.0, 0.0)] * 8
assert credits([-0.25] * 8) == [(0.0, 0.0, 0.0, 0.0)] * 8

# Correct prefix invariants: suffix failure never makes the prefix credit negative.
for reward, row in zip([0, 0.5, 2, 8] * 2, mixed):
    assert row[0] == 0
    if reward >= 0.5:
        assert row[1] >= 0
    if reward >= 2:
        assert row[2] >= 0
for reward, row in zip([-0.25, 0] * 4, credits([-0.25, 0] * 4)):
    if reward == 0:
        assert row[0] >= 0

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
        "</think>": [2],
        "商品": [3],
        "视频": [4],
        "广告": [5],
        "主播": [6],
        "该用户最近喜欢的视频有: ": [100, 101, 102, 104, 105, 4, 106],
        "该用户最近点击了商品: ": [100, 101, 102, 103, 107, 3, 106],
        "该用户最近感兴趣的广告有: ": [100, 101, 102, 108, 5, 106],
        "该用户最近首次打赏了主播: ": [100, 101, 102, 109, 110, 111, 107, 6, 106],
        "<|prod_begin|>": [10],
        "<|video_begin|>": [20],
        "<s_a_1>": [11],
        "<s_b_11>": [12],
        "<s_c_111>": [13],
    }

    def encode(self, text, add_special_tokens=False):
        return self.token_map[text]


position_tokenizer = PositionTokenizer()
candidate_ids = [10, 11, 12, 13, 99, 10, 11, 12, 13, 100]
assert find_final_sid_token_positions(
    candidate_ids, ("prod", 1, 11, 111), position_tokenizer
) == (5, 6, 7, 8)
try:
    find_final_sid_token_positions([10, 11, 99, 12, 13], ("prod", 1, 11, 111), position_tokenizer)
except RuntimeError:
    pass
else:
    raise AssertionError("non-contiguous final SID must fail closed")

# Text Domain is the final known span token after </think> and before final SID.
alignment = locate_text_domain_token(
    [2, 4, 3, 10, 11, 12, 13], ("prod", 1, 11, 111), position_tokenizer
)
assert alignment.valid is True
assert alignment.text_domain == "prod" and alignment.text_domain_token_position == 2
assert alignment.sid_domain_token_position == 3
assert alignment.hierarchy_token_positions == (2, 4, 5, 6)
mismatch = locate_text_domain_token(
    [2, 4, 10, 11, 12, 13], ("prod", 1, 11, 111), position_tokenizer
)
assert mismatch.valid is False and mismatch.failure == "text_sid_domain_mismatch"
missing = locate_text_domain_token(
    [2, 10, 11, 12, 13], ("prod", 1, 11, 111), position_tokenizer
)
assert missing.valid is False and missing.failure == "missing_text_domain_before_final_sid"

# Formal Domain commitment uses the first declaration branch and candidate-level SID fallback.
prod_ids = [2, 100, 101, 102, 103, 107, 3, 106, 10, 11, 12, 13]
prod_commitment = locate_domain_commitment_token(
    prod_ids, ("prod", 1, 11, 111), position_tokenizer
)
assert prod_commitment.mode == "branch"
assert prod_commitment.commitment_token_position == 4
assert prod_commitment.hierarchy_token_positions == (4, 9, 10, 11)
video_ids = [2, 100, 101, 102, 104, 105, 4, 106, 20, 11, 12, 13]
video_commitment = locate_domain_commitment_token(
    video_ids, ("video", 1, 11, 111), position_tokenizer
)
assert video_commitment.mode == "branch"
assert video_commitment.commitment_token_position == 4
direct_ids = [2, 10, 11, 12, 13]
direct_commitment = locate_domain_commitment_token(
    direct_ids, ("prod", 1, 11, 111), position_tokenizer
)
assert direct_commitment.mode == "direct_sid_fallback"
assert direct_commitment.commitment_token_position == 1
mixed_commitments = [
    locate_domain_commitment_token(
        prod_ids if index < 7 else direct_ids,
        ("prod", 1, 11, 111), position_tokenizer,
    )
    for index in range(8)
]
assert sum(item.mode == "branch" for item in mixed_commitments) == 7
assert sum(item.mode == "direct_sid_fallback" for item in mixed_commitments) == 1
assert all(item.eligible for item in mixed_commitments)

# One unresolved candidate is zeroed only for Domain; eligible rows re-center on E.
subset_states = [
    STATE_BY_REWARD[value]
    for value in [-0.25, 0, -0.25, -0.25, -0.25, 0, -0.25, -0.25]
]
subset = conditional_hierarchical_credits(
    subset_states, domain_eligible=[False] + [True] * 7
)
assert subset[0][0] == 0
assert [row[0] for row in subset[1:]] == [
    0.0234375,
    -0.0078125,
    -0.0078125,
    -0.0078125,
    0.0234375,
    -0.0078125,
    -0.0078125,
]


# Global G8 credits map back to the correct rank/local candidate rows.
class RuntimeTokenizer(PositionTokenizer):
    pad_token_id = 0
    eos_token_id = 0
    token_map = {
        **PositionTokenizer.token_map,
        "<s_a_9>": [19],
        "<s_b_9>": [29],
        "<s_c_9>": [39],
    }

    def encode(self, text, add_special_tokens=False):
        return self.token_map[text] if text in self.token_map else [1, 2]


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
    "<think></think>该用户最近点击了商品: <|prod_begin|><s_a_9><s_b_9><s_c_9>",
    "<think></think>该用户最近点击了商品: <|prod_begin|><s_a_1><s_b_9><s_c_9>",
    "<think></think>该用户最近点击了商品: <|prod_begin|><s_a_1><s_b_11><s_c_9>",
    "<think></think>该用户最近点击了商品: <|prod_begin|><s_a_1><s_b_11><s_c_111>",
]
id_cycle = [
    [2, 100, 101, 102, 103, 107, 3, 106, 10, 19, 29, 39],
    [2, 100, 101, 102, 103, 107, 3, 106, 10, 11, 29, 39],
    [2, 100, 101, 102, 103, 107, 3, 106, 10, 11, 12, 39],
    [2, 100, 101, 102, 103, 107, 3, 106, 10, 11, 12, 13],
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
                "token_positions": (4, 9, 10, 11),
                "domain_commitment_eligible": True,
                "domain_commitment_mode": "branch",
                "domain_commitment_token_position": 4,
                "sid_domain_token_position": 8,
                "domain_commitment_failure": None,
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
assert runtime_trainer._nothink_bridge_runtime["token_positions"] == (
    (4, 9, 10, 11),
) * 4
assert runtime_trainer._nothink_bridge_runtime["domain_text_alignment_valid"] is True
assert runtime_trainer._nothink_bridge_runtime["stage_active"] == (
    False, True, True, True,
)
assert runtime_trainer._nothink_bridge_runtime["b_singleton_success_group"] is False
assert runtime_trainer._nothink_bridge_runtime["c_singleton_success_group"] is False


def prepare_with_global_records(records, rewards):
    original = trl_grpo.gather_object
    try:
        trl_grpo.gather_object = lambda local: records
        runtime_trainer._prepare_nothink_credit_and_bridge(
            runtime_inputs,
            text_cycle,
            id_cycle,
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
        )
    finally:
        trl_grpo.gather_object = original
    return runtime_trainer._nothink_bridge_runtime


# A single Exact candidate activates both B/C and is counted as singleton success.
singleton_records = []
singleton_rewards = []
for group_id, ranks in (("g0", (0, 1)), ("g1", (2, 3))):
    for rank in ranks:
        for local_index in range(4):
            exact = rank in (0, 2) and local_index == 3
            singleton_records.append({
                "group_id": group_id,
                "rank": rank,
                "local_index": local_index,
                "predicted_sid": (
                    ("prod", 1, 11, 111) if exact else ("prod", 9, 9, 9)
                ),
                "token_positions": (4, 9, 10, 11),
                "domain_commitment_eligible": True,
                "domain_commitment_mode": "branch",
                "domain_commitment_token_position": 4,
                "sid_domain_token_position": 8,
                "domain_commitment_failure": None,
                "gold_sids": [("prod", 1, 11, 111)],
                "target_domain": "prod",
                "prompt": "prompt",
            })
            singleton_rewards.append(8.0 if exact else 0.0)
singleton_runtime = prepare_with_global_records(singleton_records, singleton_rewards)
assert singleton_runtime["stage_active"] == (False, True, True, True)
assert singleton_runtime["b_singleton_success_group"] is True
assert singleton_runtime["c_singleton_success_group"] is True


# Formal trainer path: eligible commitment receives Domain +/- credit.
domain_records = []
domain_pattern = [False, True, False, False]
for group_id, ranks in (("g0", (0, 1)), ("g1", (2, 3))):
    for rank in ranks:
        for local_index, correct in enumerate(domain_pattern):
            domain_records.append({
                "group_id": group_id,
                "rank": rank,
                "local_index": local_index,
                "predicted_sid": ("prod" if correct else "video", 9, 9, 9),
                "token_positions": (4, 9, 10, 11),
                "domain_commitment_eligible": True,
                "domain_commitment_mode": "branch",
                "domain_commitment_token_position": 4,
                "sid_domain_token_position": 8,
                "domain_commitment_failure": None,
                "gold_sids": [("prod", 1, 11, 111)],
                "target_domain": "prod",
                "prompt": "prompt",
            })
domain_rewards = [-0.25, 0.0, -0.25, -0.25] * 4
aligned_runtime = prepare_with_global_records(domain_records, domain_rewards)
assert [row[0] for row in aligned_runtime["token_credits"]] == [
    -0.0078125, 0.0234375, -0.0078125, -0.0078125,
]
assert aligned_runtime["domain_text_alignment_valid"] is True

# One unresolved candidate is zeroed without suppressing eligible peers.
mismatch_records = [dict(record) for record in domain_records]
mismatch_records[1]["domain_commitment_eligible"] = False
mismatch_records[1]["domain_commitment_mode"] = None
mismatch_records[1]["token_positions"] = None
mismatch_records[1]["domain_commitment_failure"] = "unresolved"
mismatch_runtime = prepare_with_global_records(mismatch_records, domain_rewards)
assert [row[0] for row in mismatch_runtime["token_credits"]] == [
    -0.0078125,
    0.0,
    -0.0078125,
    -0.0078125,
]
assert mismatch_runtime["domain_text_alignment_valid"] is False

# Candidate-level Domain eligibility leaves every A/B/C value exactly unchanged.
mixed_mismatch_records = [dict(record) for record in global_records]
mixed_mismatch_records[0]["domain_commitment_eligible"] = False
mixed_mismatch_records[0]["domain_commitment_mode"] = None
mixed_mismatch_records[0]["token_positions"] = None
mixed_mismatch_records[0]["domain_commitment_failure"] = "unresolved"
mixed_runtime = prepare_with_global_records(
    mixed_mismatch_records, [0, 0.5, 2, 8] * 4
)
assert mixed_runtime["token_credits"] == tuple(expected_cycle)
assert mixed_runtime["domain_text_alignment_valid"] is False


# Token tensor writes only text-Domain/SID-A/B/C positions.
trainer = object.__new__(ThinkExactClampRecGRPOTrainer)
trainer._nothink_bridge_runtime = {
    "token_credits": (mixed[0], mixed[1]),
    "token_positions": ((1, 3, 4, 5), (1, 3, 4, 5)),
}
output = {
    "completion_ids": torch.ones((2, 6), dtype=torch.long),
    "advantages": torch.tensor([999.0, -999.0]),
}
trainer._attach_nothink_token_advantages(output)
expected = torch.zeros((2, 6))
expected[0, [1, 3, 4, 5]] = torch.tensor(mixed[0])
expected[1, [1, 3, 4, 5]] = torch.tensor(mixed[1])
torch.testing.assert_close(output["token_advantages"], expected)

# Domain-only +/- credit lands on the sampled natural-language token, never SID Domain.
trainer._nothink_bridge_runtime = {
    "token_credits": (domain_only[0], domain_only[1]),
    "token_positions": ((1, 3, 4, 5), (1, 3, 4, 5)),
}
domain_output = {
    "completion_ids": torch.ones((2, 6), dtype=torch.long),
    "advantages": torch.zeros(2),
}
trainer._attach_nothink_token_advantages(domain_output)
assert domain_output["token_advantages"][0, 1] == domain_only[0][0]
assert domain_output["token_advantages"][1, 1] == domain_only[1][0]
assert torch.count_nonzero(domain_output["token_advantages"][:, 2]) == 0


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


# GRPO reduction is invariant to unrelated completion length.
def token_policy_loss(completion_mask, token_advantages):
    batch, width = completion_mask.shape
    policy_logps = torch.nn.Parameter(torch.zeros((batch, width)))
    trainer._get_per_token_logps_and_entropies = (
        lambda *args, **kwargs: (policy_logps, None)
    )
    trainer._smoke_policy_epoch = {}
    trainer._smoke_log = [{}]
    inputs = {
        "prompt_ids": torch.ones((batch, 2), dtype=torch.long),
        "prompt_mask": torch.ones((batch, 2), dtype=torch.long),
        "completion_ids": torch.ones((batch, width), dtype=torch.long),
        "completion_mask": completion_mask,
        "route_id": torch.full((batch,), ROUTE_ID["no_think"]),
        "advantages": torch.full((batch,), 1e6),
        "token_advantages": token_advantages,
    }
    loss = trainer._compute_nothink_token_loss(None, inputs)
    loss.backward()
    return loss.detach(), policy_logps.grad


length_mask = torch.tensor([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])
length_credit = torch.zeros((2, 10))
length_credit[0, 2] = 0.1
length_credit[1, 7] = 0.1
length_loss, length_grad = token_policy_loss(length_mask, length_credit)
torch.testing.assert_close(length_grad[0, 2], length_grad[1, 7], rtol=0, atol=0)
assert length_grad[0, 2] == -0.025  # 0.1 * route .5 / batch 2
assert torch.equal(length_grad == 0, length_credit == 0)
torch.testing.assert_close(length_loss, torch.tensor(-0.05), rtol=0, atol=1e-8)

# A single credited token enters per-sample loss at its exact value, then the
# unchanged NoThink route multiplier (.5) is applied.
for value, width in ((0.015625, 4), (-0.09375, 7), (0.5625, 10)):
    mask = torch.ones((1, width))
    token_credit = torch.zeros((1, width))
    token_credit[0, width // 2] = value
    exact_loss, exact_grad = token_policy_loss(mask, token_credit)
    torch.testing.assert_close(exact_loss, torch.tensor(-0.5 * value), rtol=0, atol=0)
    torch.testing.assert_close(
        exact_loss.abs() / 0.5, torch.tensor(abs(value)), rtol=0, atol=0
    )
    assert torch.equal(exact_grad == 0, token_credit == 0)


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
        "domain_text_alignment_valid": True,
        "stage_active": (False, False, False, False),
        "b_singleton_success_group": False,
        "c_singleton_success_group": False,
        "domain_commitment_eligible_count": 8,
        "domain_commitment_branch_count": 8,
        "domain_commitment_direct_sid_fallback_count": 0,
        "domain_commitment_unresolved_count": 0,
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
