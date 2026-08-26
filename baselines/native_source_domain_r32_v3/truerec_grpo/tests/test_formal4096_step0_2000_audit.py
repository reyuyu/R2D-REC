from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "analysis" / "audit_formal4096_step0_2000.py"
SPEC = importlib.util.spec_from_file_location("audit_formal4096_step0_2000", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def candidate(index: int, a=False, ab=False, exact=False, credits=(1.0, 0.0, -1.0)):
    return {
        "candidate_index": index, "A_hit": a, "AB_hit": ab, "exact": exact,
        "action_tokens": [
            {"position_name": position, "effective_signed_credit": credit}
            for position, credit in zip(module.POSITIONS, credits)
        ],
    }


def joined_row(index: int, trigger="HPR_A", hits=(False, False, False)):
    a, ab, exact = hits
    return {
        "group": {
            "group_index": index, "recommendation_group_id": f"g{index}",
            "format_valid_rate": 1.0, "wrong_history_copy_rate": 0.125,
        },
        "explain": {
            "group_index": index, "recommendation_group_id": f"g{index}",
            "hpr_trigger": trigger,
            "candidates": [candidate(i, a and i == 0, ab and i == 0, exact and i == 0) for i in range(8)],
        },
    }


class FormalStep2000AuditTest(unittest.TestCase):
    def test_bucket_boundaries_cover_exactly_2000_groups(self):
        covered = [index for start, end in module.BUCKETS for index in range(start, end + 1)]
        self.assertEqual(covered, list(range(2000)))

    def test_training_summary_uses_group_any_and_signed_credit(self):
        rows = [joined_row(0, "HPR_A"), joined_row(1, "HPR_B", (True, True, False))]
        result = module.summarize_training(rows)
        self.assertEqual(result["HPR_A_rate"], 0.5)
        self.assertEqual(result["HPR_B_rate"], 0.5)
        self.assertEqual(result["GROUP_ANY_A_rate"], 0.5)
        self.assertEqual(result["GROUP_ANY_AB_rate"], 0.5)
        self.assertEqual(result["A_positive_tokens_per_group"], 8.0)
        self.assertEqual(result["B_nonzero_credit_token_rate"], 0.0)
        self.assertEqual(result["C_negative_tokens_per_group"], 8.0)

    def test_prefix_reader_rejects_missing_or_duplicate_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('\n'.join(json.dumps({"group_index": i}) for i in range(1999)) + '\n', encoding="utf-8")
            with self.assertRaisesRegex(module.AuditError, "GROUP_RANGE_INCOMPLETE"):
                module.read_jsonl_prefix(path)
            rows = [{"group_index": i} for i in range(2000)] + [{"group_index": 100}]
            path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding="utf-8")
            with self.assertRaisesRegex(module.AuditError, "DUPLICATE_GROUP_INDEX"):
                module.read_jsonl_prefix(path)

    def test_probe_summary_and_hpr_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "probe" / "0"
            root.mkdir(parents=True)
            groups = []
            explains = []
            for i in range(20):
                groups.append({
                    "recommendation_group_id": f"g{i}", "A_hit_rate": 0.125 if i == 0 else 0.0,
                    "AB_hit_rate": 0.125 if i == 0 else 0.0, "exact_rate": 0.0,
                    "hpr_trigger": "HPR_B" if i == 0 else "HPR_A",
                })
                explains.append({"recommendation_group_id": f"g{i}"})
            summary = {"probe_step": 0, "A_hit_rate": 0.00625, "AB_hit_rate": 0.00625, "exact_rate": 0.0}
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / "groups.jsonl").write_text('\n'.join(map(json.dumps, groups)) + '\n', encoding="utf-8")
            (root / "explain.jsonl").write_text('\n'.join(map(json.dumps, explains)) + '\n', encoding="utf-8")
            result = module.summarize_probe_step(root)
            self.assertEqual(result["GROUP_ANY_A"], 1)
            self.assertEqual(result["HPR_A_count"], 19)
            self.assertEqual(result["HPR_B_count"], 1)

    def test_trend_gates_are_mechanical(self):
        buckets = {}
        for start, end in module.BUCKETS:
            name = f"{start}-{end}"
            late = start >= 1536
            buckets[name] = {
                "groups": end - start + 1,
                "HPR_A_rate": 0.60 if late else 0.70,
                "B_nonzero_credit_token_rate": 0.09 if late else 0.08,
                "C_nonzero_credit_token_rate": 0.02 if late else 0.02,
            }
        probes = {"step0_vs_latest": {
            key: {"delta": 0.0} for key in (
                "A_hit_rate", "AB_hit_rate", "exact_rate",
                "GROUP_ANY_A", "GROUP_ANY_AB", "GROUP_ANY_EXACT",
            )
        }}
        result = module.classify_trends(buckets, probes)
        self.assertEqual(result["HPR_A_TREND"], "DECREASING")
        self.assertEqual(result["FINE_GRAINED_SIGNAL_TREND"], "FLAT")
        self.assertEqual(result["PROBE_PROGRESS"], "FLAT")
        self.assertEqual(result["B_C_NONZERO_TRAINING_CREDIT_CLEARLY_INCREASED"], "NO")


if __name__ == "__main__":
    unittest.main()
