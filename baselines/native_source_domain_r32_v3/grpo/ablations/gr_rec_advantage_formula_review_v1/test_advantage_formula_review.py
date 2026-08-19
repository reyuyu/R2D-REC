"""CPU-only tests for Centered Exact-Clamp and the comparison audit."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
EXACT_FLOOR_DIR = GRPO_ROOT / "ablations" / "gr_rec_exact_floor_v1"
sys.path.insert(0, str(EXACT_FLOOR_DIR))

from audit_advantage_formulas import archetype_results, summarize
from exact_floor import exact_floor_advantages
from formulas import centered_exact_clamp_advantages


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
    tensor = torch.tensor(rewards, dtype=torch.float64)
    actual = centered_exact_clamp_advantages(tensor, 4)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert torch.isfinite(actual).all()

FLOOR_EXPECTED = [
    [0, 0, 0, 0],
    [-0.015625, -0.015625, -0.015625, 0.046875],
    [-0.0625, -0.0625, -0.0625, 0.1875],
    [-0.25, -0.25, -0.25, 0.75],
    [0, 0, 0, 0.125],
    [0, 0, 0, 0.5],
    [0.3125, -0.4375, -0.3125, 0.4375],
    [0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.75],
]
formula_archetypes = archetype_results()
for index, ((rewards, _), expected) in enumerate(zip(ARCHETYPES, FLOOR_EXPECTED)):
    tensor = torch.tensor(rewards, dtype=torch.float64)
    torch.testing.assert_close(
        exact_floor_advantages(tensor, 4),
        torch.tensor(expected, dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    mean = sum(rewards) / 4
    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / 4)
    expected_current = torch.tensor(
        [(reward - mean) / (std + 1e-4) for reward in rewards], dtype=torch.float64
    )
    current = torch.tensor(formula_archetypes[index]["current"], dtype=torch.float64)
    torch.testing.assert_close(current, expected_current, rtol=1e-12, atol=1e-12)

# G8 and multiple groups use only their own group means.
g8_a = [0, 0, 0, 0, 0, 0, 0, 8]
g8_b = [12, 12, 12, 12, 12, 12, 12, 14]
combined = torch.tensor(g8_a + g8_b, dtype=torch.float32)
result = centered_exact_clamp_advantages(combined, 8)
torch.testing.assert_close(
    result[:8], centered_exact_clamp_advantages(torch.tensor(g8_a, dtype=torch.float32), 8)
)
torch.testing.assert_close(
    result[8:], centered_exact_clamp_advantages(torch.tensor(g8_b, dtype=torch.float32), 8)
)
assert result.dtype == combined.dtype and result.device == combined.device

for bad, error in (
    (torch.tensor([[0.0]]), ValueError),
    (torch.tensor([0, 1]), TypeError),
    (torch.tensor([0.0, 1.0, 2.0]), ValueError),
):
    try:
        centered_exact_clamp_advantages(bad, 2)
    except error:
        pass
    else:
        raise AssertionError("invalid formula input must be rejected")

# No model, trainer, collective or explicit/implicit CUDA synchronization path.
source = inspect.getsource(centered_exact_clamp_advantages)
for forbidden in ("cuda", "generate", "forward", "all_gather", "gather_object", ".item("):
    assert forbidden not in source

printed = archetype_results()
assert len(printed) == 10
assert printed[-1]["rewards"] == [12, 12, 12, 14]
assert printed[-1]["exact_clamp"] == [0.0, 0.0, 0.0, 0.1875]

groups = [
    {"source": "test", "route": "think", "domain": "video", "gold_count": 5,
     "rewards": [8, 8, 8, 9]},
    {"source": "test", "route": "think", "domain": "video", "gold_count": 5,
     "rewards": [12, 12, 12, 12]},
    {"source": "test", "route": "think", "domain": "prod", "gold_count": 2,
     "rewards": [0, 0, 0, 0.5]},
]
audit, rows = summarize(groups)
assert len(rows) == 3
for level in ("8", ">8"):
    assert audit["overall"]["reward_levels"][level]["exact_clamp"]["negative_rate"] == 0
assert audit["overall"]["equal_high"]["equal_high"]["exact_clamp"]["total_abs_sum"] == 0
assert audit["overall"]["equal_high"]["equal_high"]["exact_floor"]["total_abs_sum"] == 2.0
assert math.isfinite(audit["overall"]["key_ratios"]["exact_clamp"]["video_mass_share"])

print("ADVANTAGE FORMULA REVIEW CPU TESTS PASSED")
