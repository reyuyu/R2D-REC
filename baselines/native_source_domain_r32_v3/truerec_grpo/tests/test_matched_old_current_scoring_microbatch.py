"""CPU regression for the formal old-rescore microbatch extension."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "trainer", ROOT / "diagnostics"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from rollout_runtime_v1 import PPO_OLD_LOGP_SOURCE, rescore_business_group_from_completions  # noqa: E402


class BatchPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12032)
        self.weight = torch.nn.Parameter(torch.randn(16, 32))
        self.batch_sizes = []

    def forward(self, input_ids, attention_mask):
        self.batch_sizes.append(int(input_ids.shape[0]))
        return SimpleNamespace(logits=self.weight[input_ids])


def rescore(policy, scoring_microbatch_size="OMITTED"):
    record = {
        "recommendation_group_id": "g", "fixed_domain_token": "<|video_begin|>",
        "all_gold_abc": ["<s_a_1><s_b_2><s_c_3>"], "history_sids": [],
    }
    completion_ids = ((3, 4, 5),) * 8
    diagnostic = ((99.0, 99.0, 99.0),) * 8
    kwargs = {} if scoring_microbatch_size == "OMITTED" else {"scoring_microbatch_size": scoring_microbatch_size}
    return rescore_business_group_from_completions(
        policy.eval(), record, (1, 2), completion_ids, diagnostic,
        lambda value: {3: "<s_a_1>", 4: "<s_b_2>", 5: "<s_c_3>"}.get(value, "x"),
        0, "cpu", **kwargs,
    )


class MatchedScoringMicrobatchTests(unittest.TestCase):
    def test_default_formal_rescore_remains_mb2_and_explicit_mb2_is_exact(self):
        default_policy, explicit_policy = BatchPolicy(), BatchPolicy()
        default_group = rescore(default_policy)
        explicit_group = rescore(explicit_policy, 2)
        self.assertEqual(default_policy.batch_sizes, [2, 2, 2, 2])
        self.assertEqual(explicit_policy.batch_sizes, [2, 2, 2, 2])
        self.assertEqual(
            tuple(candidate.old_logps for candidate in default_group.candidates),
            tuple(candidate.old_logps for candidate in explicit_group.candidates),
        )

    def test_explicit_mb1_is_formal_full_forward_and_generation_scores_remain_diagnostic(self):
        policy = BatchPolicy()
        group = rescore(policy, 1)
        self.assertEqual(policy.batch_sizes, [1] * 8)
        self.assertEqual(PPO_OLD_LOGP_SOURCE, "FULL_FORWARD_RESCORE")
        self.assertTrue(all(candidate.generation_score_logps == (99.0, 99.0, 99.0) for candidate in group.candidates))
        self.assertTrue(all(candidate.old_logps != candidate.generation_score_logps for candidate in group.candidates))


if __name__ == "__main__":
    unittest.main()
