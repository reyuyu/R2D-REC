"""CPU-only ExactFloor math, integration and frozen-contract tests."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

from audit_historical_rewards import current_advantages, reward_pattern, summarize
from exact_floor import exact_floor_advantages
from exact_floor_trainer import ExactFloorRecGRPOTrainer
from grpo_trl_trainer import (
    M_NO,
    M_THINK,
    ROUTE_LOSS_W,
    ROUTE_TEMP,
    ROUTE_TOP_P,
    RecGRPOTrainer,
)


PATTERNS = [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 0, 0.5], [-0.015625, -0.015625, -0.015625, 0.046875]),
    ([0, 0, 0, 2], [-0.0625, -0.0625, -0.0625, 0.1875]),
    ([0, 0, 0, 8], [-0.25, -0.25, -0.25, 0.75]),
    ([8, 8, 8, 9], [0, 0, 0, 0.125]),
    ([8, 8, 8, 12], [0, 0, 0, 0.5]),
    ([8, 2, 3, 9], [0.3125, -0.4375, -0.3125, 0.4375]),
    ([8, 8, 8, 8], [0, 0, 0, 0]),
    ([12, 12, 12, 12], [0.5, 0.5, 0.5, 0.5]),
]
for rewards, expected in PATTERNS:
    actual = exact_floor_advantages(torch.tensor(rewards, dtype=torch.float64), 4)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert torch.isfinite(actual).all()

# G8 and independent group baselines.
g4_a = [0, 0, 0, 8]
g4_b = [12, 12, 12, 12]
combined = exact_floor_advantages(torch.tensor(g4_a + g4_b, dtype=torch.float32), 4)
torch.testing.assert_close(combined[:4], exact_floor_advantages(torch.tensor(g4_a, dtype=torch.float32), 4))
torch.testing.assert_close(combined[4:], exact_floor_advantages(torch.tensor(g4_b, dtype=torch.float32), 4))
g8 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 8], dtype=torch.float32)
torch.testing.assert_close(
    exact_floor_advantages(g8, 8),
    torch.tensor([-0.125] * 7 + [0.875], dtype=torch.float32),
)
assert combined.dtype == torch.float32 and combined.device.type == "cpu"

for bad, error in (
    (torch.tensor([[0.0]]), ValueError),
    (torch.tensor([0, 1]), TypeError),
    (torch.tensor([0.0, 1.0, 2.0]), ValueError),
):
    try:
        exact_floor_advantages(bad, 2)
    except error:
        pass
    else:
        raise AssertionError("invalid rewards must be rejected")

# The subclass adds no generation/forward/reward implementation and keeps every
# parent contract except the returned advantage tensor.
assert issubclass(ExactFloorRecGRPOTrainer, RecGRPOTrainer)
source = inspect.getsource(ExactFloorRecGRPOTrainer)
assert "def _generate(" not in source and "def forward(" not in source
assert "super()._calculate_rewards" in source
for forbidden in ("all_gather", "gather_object", "torch.cuda.synchronize", ".item("):
    assert forbidden not in source
assert M_THINK == 4 and M_NO == 8
assert ROUTE_TEMP == {"think": 0.9, "no_think": 1.0}
assert ROUTE_TOP_P == {"think": 0.95, "no_think": 1.0}
assert ROUTE_LOSS_W == {"think": 1.0, "no_think": 0.5}

# Replacement helper uses global group boundaries then takes only this rank.
trainer = object.__new__(ExactFloorRecGRPOTrainer)
trainer.num_generations = 4
trainer._exact_floor_global_rewards = torch.tensor(g4_a + g4_b, dtype=torch.float32)
trainer.accelerator = type("Accelerator", (), {"process_index": 1})()
local, global_values = trainer._exact_floor_local_advantages(4)
torch.testing.assert_close(local, torch.tensor([0.5] * 4))
assert global_values.numel() == 8

# The public parent override path actually replaces, rather than merely computes,
# the advantages returned to TRL's loss.
original_generate_and_score = RecGRPOTrainer._generate_and_score_completions
try:
    RecGRPOTrainer._generate_and_score_completions = lambda self, inputs: {
        "advantages": torch.full((4,), 99.0)
    }
    trainer.accelerator = type("Accelerator", (), {"process_index": 0})()
    trainer._exact_floor_global_rewards = torch.tensor(g4_a, dtype=torch.float32)
    trainer._exact_floor_advantages_list = [-0.25, -0.25, -0.25, 0.75]
    trainer._detailed_monitor = False
    trainer._parity_audit = False
    trainer._parity_log = []
    trainer._smoke_log = []
    replaced = trainer._generate_and_score_completions([])
    torch.testing.assert_close(replaced["advantages"], torch.tensor([-0.25, -0.25, -0.25, 0.75]))
finally:
    RecGRPOTrainer._generate_and_score_completions = original_generate_and_score

# Counterfactual formula and requested pattern labels.
standardized = current_advantages(torch.tensor([0.0, 0.0, 0.0, 0.5]), 4)
assert math.isclose(float(standardized[-1]), 1.7312508, rel_tol=1e-4)
assert reward_pattern([0, 0, 0, 0]) == "ALL_ZERO"
assert reward_pattern([0, 0, 0, 0.5]) == "A_ONLY"
assert reward_pattern([0, 0, 0, 2]) == "AB_SIGNAL"
assert reward_pattern([0, 0, 0, 8]) == "SINGLE_EXACT"
assert reward_pattern([8, 8, 8, 8]) == "EXACT_SATURATED"
assert reward_pattern([8, 8, 8, 12]) == "MULTI_EXACT"
assert reward_pattern([8, 2, 3, 9]) == "MIXED_HIGH"

audit = summarize([
    {"source": "test", "route": "think", "domain": "prod", "gold_count": 2,
     "rewards": [8, 8, 8, 9]},
    {"source": "test", "route": "think", "domain": "video", "gold_count": 5,
     "rewards": [12, 12, 12, 12]},
])
assert audit["reward_levels"]["test"]["8"]["current"]["negative_rate"] == 1.0
assert audit["reward_levels"]["test"]["8"]["exact_floor"]["negative_rate"] == 0.0
assert audit["prevalence"]["test"]["all_gt_8_rate"] == 0.5

# No historical implementation is edited by this ablation.
ablation_root = Path(__file__).resolve().parent
assert all(path.name != "grpo_trl_trainer.py" for path in ablation_root.iterdir())

print("EXACTFLOOR CPU TESTS PASSED")
