import unittest

from run_user_pilot150 import (
    ACTION_BUCKETS,
    CHAIN_BUCKETS,
    build_training_plan,
    direction_decision,
    distribute_step_rows,
    segment_summaries,
    select_pilot_rows,
)


def action_row(bucket, index, length):
    ranges = {
        "1-5": 3,
        "6-10": 8,
        "11-20": 15,
        "21-30": 25,
        "31-40": 35,
        "41+": 45,
    }
    return {
        "sample_id": f"a-{bucket}-{index}",
        "route": "action",
        "gold_sid_count": ranges[bucket],
        "gold_event_count": 0,
        "prompt_token_count": length,
    }


def chain_row(events, index, length):
    return {
        "sample_id": f"c-{events}-{index}",
        "route": "chain",
        "gold_sid_count": 0,
        "gold_event_count": events,
        "prompt_token_count": length,
    }


class Pilot150ContractTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for label, _low, _high, _count in ACTION_BUCKETS:
            rows.extend(action_row(label, index, 100 + index * 17) for index in range(24))
        for label, events, _count in CHAIN_BUCKETS:
            rows.extend(chain_row(events, index, 200 + index * 19) for index in range(28))
        return rows

    def test_selection_is_deterministic_stratified_and_unique(self):
        first, audit = select_pilot_rows(self.fixture(), 20260820)
        second, _ = select_pilot_rows(list(reversed(self.fixture())), 20260820)
        self.assertEqual(
            {route: [row["sample_id"] for row in first[route]] for route in ("action", "chain")},
            {route: [row["sample_id"] for row in second[route]] for route in ("action", "chain")},
        )
        self.assertEqual(75, len(first["action"]))
        self.assertEqual(75, len(first["chain"]))
        self.assertEqual(150, len({row["sample_id"] for route in first.values() for row in route}))
        self.assertEqual(75, sum(audit["bucket_counts"]["action"].values()))
        self.assertEqual(75, sum(audit["bucket_counts"]["chain"].values()))

    def test_plan_is_route_homogeneous_and_stops_at_150(self):
        selected, _ = select_pilot_rows(self.fixture(), 20260820)
        plan = build_training_plan(selected)
        self.assertEqual(20, len(plan))
        self.assertEqual(150, sum(len(item["rows"]) for item in plan))
        self.assertEqual(["action", "chain"] * 10, [item["route"] for item in plan])
        self.assertTrue(all({row["route"] for row in item["rows"]} == {item["route"]} for item in plan))

    def test_tail_ddp_weighting_is_exact_and_has_one_dummy(self):
        rows = [{"sample_id": str(index)} for index in range(3)]
        assignments = [distribute_step_rows(rows, rank) for rank in range(4)]
        self.assertEqual([1, 1, 1, 0], [len(real) for real, _dummy, _scale in assignments])
        self.assertEqual([False, False, False, True], [dummy for _real, dummy, _scale in assignments])
        self.assertAlmostEqual(4.0, sum(scale for _real, _dummy, scale in assignments))

    def test_segment_boundaries_are_19_37_19(self):
        records = []
        for route in ("action", "chain"):
            for index in range(75):
                record = {"route": route, "route_index": index}
                for field in ("f1", "precision", "recall", "wrong_selection", "hallucination", "duplicate"):
                    record[field] = float(index)
                for field in ("reward", "action_alignment", "logic_alignment", "date_mismatch", "action_mismatch"):
                    record[field] = float(index)
                records.append(record)
        summary = segment_summaries(records)
        self.assertEqual([19, 37, 19], [summary["action"][name]["prompt_count"] for name in ("early", "middle", "late")])

    def test_direction_gate_fails_closed(self):
        segment = {
            "action": {
                "early": {"f1": 0.7, "precision": 0.7, "wrong_selection": 0.6},
                "late": {"f1": 0.5, "precision": 0.5, "wrong_selection": 0.8},
            },
            "chain": {
                "early": {"reward": 0.5, "logic_alignment": 0.3, "date_mismatch": 0.2},
                "late": {"reward": 0.3, "logic_alignment": 0.1, "date_mismatch": 0.4},
            },
        }
        global_metrics = {
            "nan_or_inf": False,
            "base_delta": 0.0,
            "lora_delta_l2": 0.1,
            "safety_stop_triggered": False,
        }
        config = {
            "direction_gate": {
                "action_f1_late_minus_early_min": -0.03,
                "action_precision_late_minus_early_min": -0.03,
                "action_wrong_selection_late_minus_early_max": 0.05,
                "chain_reward_late_minus_early_min": -0.03,
                "chain_logic_late_minus_early_min": -0.03,
                "chain_date_mismatch_late_minus_early_max": 0.05,
            }
        }
        self.assertEqual("STOP_AND_ANALYZE", direction_decision(segment, global_metrics, config)["decision"])


if __name__ == "__main__":
    unittest.main()
