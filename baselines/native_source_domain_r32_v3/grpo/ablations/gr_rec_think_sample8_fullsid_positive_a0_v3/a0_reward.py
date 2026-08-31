"""The only mathematical change in the positive-group V3 second stage."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from grpo_sid import q_reward as v3_q_reward


def remove_a_only_reward(raw_reward: float) -> float:
    """Map A-only 0.5 to zero while preserving every other V3 reward level."""
    value = float(raw_reward)
    return 0.0 if value == 0.5 else value


def q_reward_without_a_only(candidate, gold_sids) -> float:
    return remove_a_only_reward(v3_q_reward(candidate, gold_sids))
