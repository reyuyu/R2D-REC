"""CPU tests for training-signal versus diagnostic-only monitor semantics."""
from .simple_monitor import summarize_simple_think_records


def _record(reward, raw_n, unique_a, grounded_n=0):
    return {
        "group_id": "g",
        "primary_reward": reward,
        "raw_interest_n": raw_n,
        "s_n": 1.0 if 2 <= raw_n <= 4 else 0.0,
        "unique_valid_target_a": unique_a,
        "d_a": min(unique_a / 8.0, 1.0),
        "beam_invalid": 1,
        "diagnostic_only": {
            "grounded_n": grounded_n,
            "grounding_coverage": grounded_n / raw_n if raw_n else None,
            "fake_or_ungrounded_sid_count": 2,
            "d_cot": 0.9,
            "s_prefix": 0.8,
            "exact": 0,
            "ab": 1,
            "a": 2,
        },
    }


def test_monitor_separates_training_and_diagnostic_fields():
    records = [
        _record(0, 4, 1, 2),
        _record(0, 1, 8, 1),
        _record(0, 3, 4, 0),
        _record(0, 2, 2, 2),
    ]
    payload, scores, advantages = summarize_simple_think_records(records)
    assert scores == [1.125, 1.0, 1.5, 1.25]
    assert payload["training_signal"]["diagnostic_only"] is False
    assert payload["diagnostic_only"]["diagnostic_only"] is True
    assert payload["diagnostic_only"]["grounding_coverage_mean"] == 0.625
    assert payload["simple_branches"]["dead_zero_count_diversity_rescue"] == 1
    assert len(advantages) == 4
    assert all(record["simple_branch"] == "dead_zero_count_diversity_rescue" for record in records)


def test_primary_signal_monitor_advantages_are_exact_zero():
    records = [_record(reward, 4, 8, 4) for reward in (8, 2, 0.5, 0)]
    payload, scores, advantages = summarize_simple_think_records(records)
    assert scores == advantages == [0.0] * 4
    assert payload["simple_rescue_active_rate"] == 0.0
    assert payload["simple_branches"]["primary_only"] == 1
