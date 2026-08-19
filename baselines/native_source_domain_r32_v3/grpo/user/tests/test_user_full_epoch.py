import json
import random
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_user_probe_light import select_light_probe  # noqa: E402
from preflight_user_full_epoch import build_formal_plan, distribute_formal_step_rows, select_remaining  # noqa: E402


def row(sample_id, route, count, tokens):
    return {
        "sample_id": sample_id,
        "route": route,
        "gold_sid_count": count if route == "action" else 0,
        "gold_event_count": count if route == "chain" else 0,
        "prompt_token_count": tokens,
    }


class FullEpochContractTests(unittest.TestCase):
    def test_remaining_selection_is_natural_deterministic_shuffle(self):
        rows = [row(f"a-{i}", "action", 1, i) for i in range(5)] + [row(f"c-{i}", "chain", 2, i) for i in range(5)]
        selected_a = select_remaining(rows, ["a-0", "c-0"], 20260820)
        selected_b = select_remaining(rows, ["a-0", "c-0"], 20260820)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual({item["sample_id"] for item in selected_a["action"]}, {f"a-{i}" for i in range(1, 5)})
        self.assertEqual({item["sample_id"] for item in selected_a["chain"]}, {f"c-{i}" for i in range(1, 5)})

    def test_formal_plan_is_338_alternating_steps(self):
        selected = {
            "action": [row(f"a-{i}", "action", 1, i) for i in range(1350)],
            "chain": [row(f"c-{i}", "chain", 3, i) for i in range(1350)],
        }
        plan = build_formal_plan(selected, 40)
        self.assertEqual((plan[0]["step"], plan[-1]["step"]), (41, 378))
        self.assertEqual([item["route"] for item in plan[:4]], ["action", "chain", "action", "chain"])
        self.assertEqual([len(item["rows"]) for item in plan[-2:]], [6, 6])
        self.assertEqual(len({row["sample_id"] for item in plan for row in item["rows"]}), 2700)

    def test_six_prompt_tail_scaling_matches_global_mean(self):
        rows = [row(str(i), "action", 1, i) for i in range(6)]
        partitions = [distribute_formal_step_rows(rows, rank) for rank in range(4)]
        self.assertEqual([len(item[0]) for item in partitions], [2, 2, 1, 1])
        self.assertEqual([item[2] for item in partitions], [4 / 3, 4 / 3, 2 / 3, 2 / 3])
        for local, _dummy, scale in partitions:
            self.assertAlmostEqual(scale / (4 * len(local)), 1 / 6)

    def test_light_probe_uses_metadata_and_required_chain_events(self):
        rows = [row(f"a-{i}", "action", i + 1, 100 + i * 100) for i in range(12)]
        rows += [row(f"c-{event}-{i}", "chain", event, 1000 + event * 100 + i) for event in (2, 3, 4, 5) for i in range(2)]
        selected, audit = select_light_probe(rows)
        self.assertEqual(sum(item["route"] == "action" for item in selected), 3)
        self.assertEqual(sum(item["route"] == "chain" for item in selected), 3)
        self.assertEqual({item["gold_event_count"] for item in selected if item["route"] == "chain"}, {2, 3, 4})
        self.assertFalse(audit["selection_uses_rewards"])
        self.assertNotIn("reward", " ".join(audit["selection_fields"]))


if __name__ == "__main__":
    unittest.main()
