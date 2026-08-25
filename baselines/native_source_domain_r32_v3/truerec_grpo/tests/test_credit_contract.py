from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "credit"))
sys.path.insert(0, str(ROOT / "analysis"))
from frontier_credit_v1 import FORMAT_INVALID_TOTAL, GATED, NEGATIVE, POSITIVE, plan_frontier_credit
from hpr_plan_v1 import plan_hpr, validate_plan

SPEC = importlib.util.spec_from_file_location("replay_credit_contract", ROOT / "analysis/replay_credit_contract.py")
replay = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(replay)


GOLD = ["<s_a_1><s_b_2><s_c_3>", "<s_a_1><s_b_4><s_c_5>", "<s_a_6><s_b_7><s_c_8>"]


def candidate(frontier, abc="<s_a_9><s_b_9><s_c_9>", sample=0, wrong=False):
    values = {
        "INVALID_FORMAT": (False, False, False, False), "A_FAIL": (True, False, False, False),
        "B_FAIL": (True, True, False, False), "C_FAIL": (True, True, True, False), "EXACT": (True, True, True, True),
    }[frontier]
    return {"format_valid": values[0], "A_hit": values[1], "AB_hit": values[2], "exact": values[3], "parsed_abc": abc if values[0] else None, "sample_index": sample, "frontier": frontier, "wrong_history_copy": wrong}


def g8(item): return [{**item, "sample_index": index} for index in range(8)]


class CreditContractTest(unittest.TestCase):
    def test_01_a_fail_only_a_negative(self):
        row = plan_frontier_credit(g8(candidate("A_FAIL"))).candidates[0]; self.assertEqual(row.kinds, (NEGATIVE, GATED, GATED))
    def test_02_b_fail_a_positive_b_negative(self):
        row = plan_frontier_credit(g8(candidate("B_FAIL", "<s_a_1><s_b_9><s_c_9>"))).candidates[0]; self.assertEqual(row.kinds, (POSITIVE, NEGATIVE, GATED))
    def test_03_c_fail_a_b_positive_c_negative(self):
        row = plan_frontier_credit(g8(candidate("C_FAIL", "<s_a_1><s_b_2><s_c_9>"))).candidates[0]; self.assertEqual(row.kinds, (POSITIVE, POSITIVE, NEGATIVE))
    def test_04_exact_all_positive(self):
        row = plan_frontier_credit(g8(candidate("EXACT", GOLD[0]))).candidates[0]; self.assertEqual(row.kinds, (POSITIVE, POSITIVE, POSITIVE))
    def test_05_prefix_gating(self):
        self.assertEqual(plan_frontier_credit(g8(candidate("A_FAIL"))).candidates[0].credits[1:], (0.0, 0.0))
    def test_06_g8_milestone_mean(self):
        rows = g8(candidate("A_FAIL")); rows[-1] = candidate("EXACT", GOLD[0], 7)
        self.assertEqual(plan_frontier_credit(rows).milestone_means, (0.125, 0.125, 0.125))
    def test_07_fixed_scale(self):
        row = plan_frontier_credit(g8(candidate("A_FAIL"))).candidates[0]; self.assertEqual(row.credits[0], -0.5 / 8)
    def test_08_domain_has_no_credit(self): self.assertFalse(plan_frontier_credit(g8(candidate("A_FAIL"))).domain_credit)
    def test_09_invalid_separate(self):
        row = plan_frontier_credit(g8(candidate("INVALID_FORMAT"))).candidates[0]; self.assertEqual(row.credits, (0.0, 0.0, 0.0)); self.assertEqual(row.format_credit_total, FORMAT_INVALID_TOTAL)
    def test_10_hpr_a_multi_positive(self):
        plan = plan_hpr(g8(candidate("A_FAIL")), GOLD); self.assertEqual(plan.trigger, "HPR_A"); self.assertEqual(plan.sites[0].target_tokens, ("<s_a_1>", "<s_a_6>"))
    def test_11_hpr_b_conditioned_targets(self):
        plan = plan_hpr(g8(candidate("B_FAIL", "<s_a_1><s_b_9><s_c_9>")), GOLD); self.assertEqual(plan.sites[0].target_tokens, ("<s_b_2>", "<s_b_4>"))
    def test_12_hpr_c_conditioned_targets(self):
        plan = plan_hpr(g8(candidate("C_FAIL", "<s_a_1><s_b_2><s_c_9>")), GOLD); self.assertEqual(plan.sites[0].target_tokens, ("<s_c_3>",))
    def test_13_sites_deduplicate_prefix(self): self.assertEqual(len(plan_hpr(g8(candidate("B_FAIL", "<s_a_1><s_b_9><s_c_9>")), GOLD).sites), 1)
    def test_14_group_average_site_contract(self):
        rows = g8(candidate("B_FAIL", "<s_a_1><s_b_9><s_c_9>"))
        rows[-1] = candidate("B_FAIL", "<s_a_6><s_b_9><s_c_9>", 7)
        plan = plan_hpr(rows, GOLD)
        self.assertEqual(len(plan.sites), 2)
        self.assertEqual(plan.site_reduction, "mean_within_group")
    def test_15_trigger_partition(self):
        cases = (("A_FAIL", "HPR_A"), ("B_FAIL", "HPR_B"), ("C_FAIL", "HPR_C"), ("EXACT", "HPR_NONE"))
        for frontier, trigger in cases: self.assertEqual(plan_hpr(g8(candidate(frontier, GOLD[0])), GOLD).trigger, trigger)
    def test_16_existing_policy_position_mapping(self):
        for frontier in ("A_FAIL", "B_FAIL", "C_FAIL"):
            plan = plan_hpr(g8(candidate(frontier, GOLD[0])), GOLD); validate_plan(plan); self.assertTrue(all(site.onpolicy_positions for site in plan.sites))
    def test_17_no_extra_forward(self): self.assertFalse(plan_hpr(g8(candidate("A_FAIL")), GOLD).extra_model_forward)
    def test_18_zero_std_hpr_independent(self): self.assertEqual(replay.set_intersection_counts({"a", "b"}, {"b", "c"}), {"ZERO_STD_AND_HPR_A": 1, "ZERO_STD_NOT_HPR_A": 1, "HPR_A_NOT_ZERO_STD": 1})
    def test_19_wrong_history_breakdown(self):
        rows = [candidate(name, wrong=True) for name in ("A_FAIL", "B_FAIL", "C_FAIL", "EXACT")]; self.assertEqual(replay.wrong_history_frontier(rows), {"A_FAIL": 1, "B_FAIL": 1, "C_FAIL": 1, "EXACT": 1})
    def test_20_raw_sha_count_group_gate(self):
        rows = []
        for index in range(8): rows.append({"group_id": "g", "sample_index": index})
        raw = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"; path.write_bytes(raw)
            self.assertEqual(len(replay.load_raw_gate(path, hashlib.sha256(raw).hexdigest(), 8, 1)), 8)
            with self.assertRaises(replay.ReplayError): replay.load_raw_gate(path, "0" * 64, 8, 1)
    def test_21_sample_index_exact(self):
        rows = [{"group_id": "g", "sample_index": index} for index in range(7)] + [{"group_id": "g", "sample_index": 6}]
        with self.assertRaises(replay.ReplayError): replay.validate_raw_records(rows, 1)


if __name__ == "__main__": unittest.main()
