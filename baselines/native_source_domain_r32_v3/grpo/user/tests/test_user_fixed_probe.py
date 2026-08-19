import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from user_action_reward import score_action
from user_chain_reward import score_chain
from user_fixed_probe import (
    aggregate_probe_steps,
    build_probe_event,
    correct_match_spans,
    partition_probe_rows,
    probe_due,
    validate_probe_rows,
)


S1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
S2 = "<|prod_begin|><s_a_4><s_b_5><s_c_6>"


def probe_rows():
    return [
        {
            "sample_id": f"action-{index:02d}",
            "route": "action",
            "bucket": index % 6,
        }
        for index in range(12)
    ] + [
        {
            "sample_id": f"chain-{index:02d}",
            "route": "chain",
            "bucket": 2 + index % 4,
        }
        for index in range(8)
    ]


class UserFixedProbeTests(unittest.TestCase):
    def test_schedule_includes_baseline_periodic_and_final(self):
        self.assertTrue(probe_due(0, 20, 10))
        self.assertTrue(probe_due(10, 20, 10))
        self.assertTrue(probe_due(20, 20, 10))
        self.assertFalse(probe_due(7, 20, 10))

    def test_contract_and_four_rank_partition(self):
        rows = probe_rows()
        validate_probe_rows(rows)
        partitions = [partition_probe_rows(rows, rank) for rank in range(4)]
        self.assertEqual([len(item["action"]) for item in partitions], [3, 3, 3, 3])
        self.assertEqual([len(item["chain"]) for item in partitions], [2, 2, 2, 2])
        assigned = [row["sample_id"] for part in partitions for route in part.values() for row in route]
        self.assertEqual(sorted(assigned), sorted(row["sample_id"] for row in rows))

    def test_action_only_true_positive_sid_is_green(self):
        completion = json.dumps([S1, S2])
        sample = {"gold_sids": [S1], "history_sids": [S1, S2]}
        score = score_action(completion, sample)
        spans = correct_match_spans(score, sample, "action")
        self.assertEqual(len(spans), 1)
        self.assertEqual(completion[spans[0]["start"]:spans[0]["end"]], S1)
        self.assertEqual(spans[0]["kind"], "correct_sid")

    def test_chain_marks_exact_event_and_gold_sid(self):
        event = {
            "date": "2026-01-01",
            "action": f"[video] {S1}",
            "logic": "watch",
        }
        completion = json.dumps({"logic_chain": {"name": "x", "events": [event]}})
        sample = {
            "gold_events": [event],
            "gold_sids": [S1],
            "history_events": [{"date": event["date"], "raw": event["action"]}],
            "history_sids": [S1],
        }
        score = score_chain(completion, sample)
        spans = correct_match_spans(score, sample, "chain")
        self.assertEqual({item["kind"] for item in spans}, {"correct_event", "correct_sid"})
        sid_span = next(item for item in spans if item["kind"] == "correct_sid")
        self.assertEqual(completion[sid_span["start"]:sid_span["end"]], S1)

    def test_probe_event_and_step_aggregation(self):
        action_sample = {"sample_id": "a", "route": "action", "bucket": 1}
        chain_sample = {"sample_id": "c", "route": "chain", "bucket": 2}
        action_candidates = [
            {"reward": value, "f1": value, "precision": value, "recall": value, "exact_match": False}
            for value in (0.25, 0.5, 0.75, 1.0)
        ]
        chain_candidates = [
            {"reward": 0.5, "total_reward": 0.5, "action_alignment": 0.75, "logic_alignment": 0.25}
            for _ in range(4)
        ]
        events = [
            build_probe_event(action_sample, action_candidates, step=10, reason="periodic", seed=1, probe_wall_sec=2),
            build_probe_event(chain_sample, chain_candidates, step=10, reason="periodic", seed=1, probe_wall_sec=2),
        ]
        row = aggregate_probe_steps(events)[0]
        self.assertEqual(row["step"], 10)
        self.assertEqual(row["action_f1"], 0.625)
        self.assertEqual(row["action_exact"], 0.0)
        self.assertEqual(row["chain_action_alignment"], 0.75)
        self.assertEqual(row["chain_logic_alignment"], 0.25)


if __name__ == "__main__":
    unittest.main()
