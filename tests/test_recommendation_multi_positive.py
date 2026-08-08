#!/usr/bin/env python3
"""CPU tests for the REC_G2 prefix-trie replacement controller."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "src")

from llamafactory.train.sft.recommendation_multi_positive import (  # noqa: E402
    REC_MP_STAT_INDEX,
    RecommendationMultiPositiveTrieController,
    recommendation_multi_positive_statistics_to_metrics,
)

IGNORE_INDEX = -100
DOMAIN = "<|video_begin|>"


class FakeTokenizer:
    def __init__(self):
        self.ids = {DOMAIN: 10}
        self.ids.update({f"<s_a_{i}>": 20 + i for i in range(4)})
        self.ids.update({f"<s_b_{i}>": 30 + i for i in range(4)})
        self.ids.update({f"<s_c_{i}>": 40 + i for i in range(4)})

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, -1)


def sid(a, b, c):
    return f"{DOMAIN}<s_a_{a}><s_b_{b}><s_c_{c}>"


def metadata(group_id, all_gold, current):
    return {
        "recommendation_multi_positive": {
            "group_id": group_id,
            "group_size": len(all_gold),
            "all_gold_sids": all_gold,
            "current_gold_sid": current,
        }
    }


def args(enabled=True):
    return SimpleNamespace(
        recommendation_multi_positive_trie_enabled=enabled,
        recommendation_final_sid_weight=8.0,
        sid_token_weight=8.0,
    )


def make_labels(path):
    # The full SID begins at sequence position 1; causal logits for a/b/c are
    # therefore positions 1/2/3, respectively.
    return torch.tensor([[IGNORE_INDEX, 10, 20 + path[0], 30 + path[1], 40 + path[2]]], dtype=torch.long)


def run_controller(all_gold, current_path, logits=None, enabled=True):
    controller = RecommendationMultiPositiveTrieController(FakeTokenizer(), args(enabled))
    labels = make_labels(current_path)
    if logits is None:
        logits = torch.zeros(1, labels.shape[1], 64, dtype=torch.float32, requires_grad=True)
    result = controller.compute(
        logits,
        labels,
        [metadata("g", [sid(*path) for path in all_gold], sid(*current_path))],
        torch.tensor([[0, labels.shape[1]]]),
        "recommendation",
        "nocot",
        torch.tensor(32.0),
    )
    return result, labels


def test_singleton_exact_equivalence():
    result, _ = run_controller([(0, 0, 0)], (0, 0, 0))
    assert abs(float(result.loss_delta.item())) < 1e-6
    metrics = recommendation_multi_positive_statistics_to_metrics(result.statistics)
    assert metrics["rec_mp_allowed_a"] == 1
    assert metrics["rec_mp_allowed_b"] == 1
    assert metrics["rec_mp_allowed_c"] == 1
    assert metrics["rec_mp_delta"] <= 1e-6


def test_a_level_multi_positive_and_no_cartesian_branch():
    result, _ = run_controller([(0, 0, 0), (1, 1, 1)], (0, 0, 0))
    metrics = recommendation_multi_positive_statistics_to_metrics(result.statistics)
    assert metrics["rec_mp_allowed_a"] == 2
    assert metrics["rec_mp_allowed_b"] == 1
    assert metrics["rec_mp_allowed_c"] == 1
    assert metrics["rec_mp_multi_a_ratio"] == 1
    assert metrics["rec_mp_multi_b_ratio"] == 0
    assert metrics["rec_mp_delta"] <= 1e-6


def test_b_and_c_branch_filtering():
    b_result, _ = run_controller([(0, 0, 0), (0, 1, 2)], (0, 0, 0))
    b_metrics = recommendation_multi_positive_statistics_to_metrics(b_result.statistics)
    assert b_metrics["rec_mp_allowed_a"] == 1
    assert b_metrics["rec_mp_allowed_b"] == 2
    assert b_metrics["rec_mp_allowed_c"] == 1
    c_result, _ = run_controller([(0, 0, 0), (0, 0, 1)], (0, 0, 0))
    c_metrics = recommendation_multi_positive_statistics_to_metrics(c_result.statistics)
    assert c_metrics["rec_mp_allowed_a"] == 1
    assert c_metrics["rec_mp_allowed_b"] == 1
    assert c_metrics["rec_mp_allowed_c"] == 2


def test_nested_raw_field_names_normalize_like_runtime_v3_cache():
    controller = RecommendationMultiPositiveTrieController(FakeTokenizer(), args())
    labels = make_labels((0, 0, 0))
    all_gold = [sid(0, 0, 0), sid(0, 1, 2)]
    raw_nested = {
        "recommendation_multi_positive": {
            "recommendation_group_id": "runtime-g",
            "recommendation_group_size": 2,
            "recommendation_all_gold_sids": all_gold,
            "recommendation_current_gold_sid": all_gold[0],
        }
    }
    result = controller.compute(
        torch.zeros(1, labels.shape[1], 64), labels, [raw_nested],
        torch.tensor([[0, labels.shape[1]]]), "recommendation", "nocot", torch.tensor(32.0),
    )
    metrics = recommendation_multi_positive_statistics_to_metrics(result.statistics)
    assert metrics["rec_mp_segments"] == 1
    assert metrics["rec_mp_group_size"] == 2
    assert metrics["rec_mp_allowed_b"] == 2


def test_invalid_current_gold_fails_fast():
    controller = RecommendationMultiPositiveTrieController(FakeTokenizer(), args())
    labels = make_labels((0, 0, 0))
    try:
        controller.compute(
            torch.zeros(1, labels.shape[1], 64),
            labels,
            [metadata("g", [sid(0, 0, 0)], sid(1, 1, 1))],
            torch.tensor([[0, labels.shape[1]]]),
            "recommendation", "nocot", torch.tensor(8.0),
        )
    except ValueError as exc:
        assert "current_gold_sid" in str(exc)
    else:
        raise AssertionError("corrupt V3 metadata did not fail fast")


def test_packed_segments_are_independent():
    labels = torch.tensor([[IGNORE_INDEX, 10, 20, 30, 40, 10, 20, 31, 41]], dtype=torch.long)
    controller = RecommendationMultiPositiveTrieController(FakeTokenizer(), args())
    result = controller.compute(
        torch.zeros(1, labels.shape[1], 64, requires_grad=True), labels,
        [metadata("a", [sid(0, 0, 0), sid(1, 1, 1)], sid(0, 0, 0)), metadata("b", [sid(0, 1, 1)], sid(0, 1, 1))],
        torch.tensor([[0, 5], [5, 9]]), "recommendation", "nocot", torch.tensor(64.0),
    )
    metrics = recommendation_multi_positive_statistics_to_metrics(result.statistics)
    assert metrics["rec_mp_segments"] == 2
    assert metrics["rec_mp_allowed_a"] == 1.5


def test_causal_shift_and_gradient_direction():
    result, labels = run_controller([(0, 0, 0), (1, 1, 1)], (0, 0, 0))
    assert result.loss_delta.requires_grad
    result.loss_delta.backward()
    # At the A label position 2, logits position 1 controls the gradient.
    # This test uses a separate tensor so the exact shifted row is observable.
    logits = torch.zeros(1, labels.shape[1], 64, requires_grad=True)
    logits.data[0, 1, 20] = 2.0
    logits.data[0, 1, 21] = 2.0
    shifted = RecommendationMultiPositiveTrieController(FakeTokenizer(), args()).compute(
        logits, labels, [metadata("g", [sid(0, 0, 0), sid(1, 1, 1)], sid(0, 0, 0))],
        torch.tensor([[0, labels.shape[1]]]), "recommendation", "nocot", torch.tensor(32.0),
    )
    shifted.loss_delta.backward()
    assert logits.grad[0, 1, 21].item() < 0
    assert logits.grad[0, 0].abs().sum().item() == 0


def test_disabled_is_exact_noop_and_invariant():
    controller = RecommendationMultiPositiveTrieController(FakeTokenizer(), args(False))
    labels = make_labels((0, 0, 0))
    result = controller.compute(torch.zeros(1, labels.shape[1], 64), labels, [], torch.tensor([[0, labels.shape[1]]]), "recommendation", "nocot", torch.tensor(8.0))
    assert result.loss_delta.item() == 0
    enabled, _ = run_controller([(0, 0, 0), (0, 0, 1)], (0, 0, 0))
    metrics = recommendation_multi_positive_statistics_to_metrics(enabled.statistics)
    assert metrics["rec_mp_invariant_violation_max"] <= 1e-6


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
