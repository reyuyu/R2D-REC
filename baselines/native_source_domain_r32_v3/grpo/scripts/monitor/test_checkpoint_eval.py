"""CPU-only tests for Recommendation checkpoint history-copy diagnostics."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from checkpoint_eval import (
    _aggregate,
    _route_outcome,
    build_cohort,
    extract_sids,
    load_validation_pool,
)


VIDEO_HISTORY = ("video", 1, 2, 3)
VIDEO_NOVEL = ("video", 4, 5, 6)
PROD_SAME_COMPONENTS = ("prod", 1, 2, 3)


def old_metrics(predicted, gold):
    valid = [sid for sid in predicted if sid is not None]
    return {
        "hit": any(sid in gold for sid in valid),
        "ab_hit": any(any(sid[:3] == target[:3] for target in gold) for sid in valid),
        "a_hit": any(any(sid[:2] == target[:2] for target in gold) for sid in valid),
        "invalid_count": len(predicted) - len(valid),
    }


class RouteOutcomeTests(unittest.TestCase):
    def test_all_novel(self):
        result = _route_outcome([VIDEO_NOVEL] * 4, set(), {VIDEO_HISTORY})
        self.assertEqual(result["history_copy_prediction_count"], 0)
        self.assertEqual(result["novel_prediction_count"], 4)
        self.assertEqual(result["history_copy_rate"], 0.0)
        self.assertEqual(result["novel_prediction_rate"], 1.0)

    def test_all_history_and_duplicate_unique_counts(self):
        result = _route_outcome([VIDEO_HISTORY] * 32, set(), {VIDEO_HISTORY})
        self.assertEqual(result["valid_prediction_count"], 32)
        self.assertEqual(result["history_copy_prediction_count"], 32)
        self.assertEqual(result["history_copy_rate"], 1.0)
        self.assertEqual(result["unique_valid_sid_count"], 1)
        self.assertEqual(result["unique_history_copy_sid_count"], 1)
        self.assertEqual(result["unique_history_copy_rate"], 1.0)

    def test_mixed_history_and_novel(self):
        result = _route_outcome(
            [VIDEO_HISTORY, VIDEO_HISTORY, VIDEO_NOVEL, VIDEO_NOVEL],
            set(),
            {VIDEO_HISTORY},
        )
        self.assertEqual(result["history_copy_rate"], 0.5)
        self.assertEqual(result["novel_prediction_rate"], 0.5)
        self.assertEqual(result["unique_history_copy_rate"], 0.5)
        self.assertEqual(result["unique_novel_rate"], 0.5)

    def test_invalid_predictions_are_excluded_from_denominator(self):
        result = _route_outcome(
            [VIDEO_HISTORY, VIDEO_NOVEL, None, None], set(), {VIDEO_HISTORY}
        )
        self.assertEqual(result["invalid_count"], 2)
        self.assertEqual(result["valid_prediction_count"], 2)
        self.assertEqual(result["history_copy_rate"], 0.5)
        invalid_only = _route_outcome([None, None], set(), {VIDEO_HISTORY})
        self.assertIsNone(invalid_only["history_copy_rate"])
        self.assertIsNone(invalid_only["unique_history_copy_rate"])

    def test_history_membership_requires_exact_domain_and_components(self):
        result = _route_outcome(
            [PROD_SAME_COMPONENTS], set(), {VIDEO_HISTORY}
        )
        self.assertEqual(result["history_copy_prediction_count"], 0)
        self.assertEqual(result["novel_prediction_count"], 1)

    def test_existing_metrics_are_unchanged(self):
        predicted = [VIDEO_HISTORY, VIDEO_NOVEL, None]
        gold = {VIDEO_NOVEL}
        result = _route_outcome(predicted, gold, {VIDEO_HISTORY})
        self.assertEqual(
            {key: result[key] for key in old_metrics(predicted, gold)},
            old_metrics(predicted, gold),
        )


class AggregateTests(unittest.TestCase):
    def test_aggregate_uses_counts_not_mean_group_rates(self):
        first = _route_outcome(
            [VIDEO_HISTORY, VIDEO_HISTORY, VIDEO_HISTORY, None],
            {VIDEO_HISTORY},
            {VIDEO_HISTORY},
        )
        second = _route_outcome([VIDEO_NOVEL], {VIDEO_HISTORY}, {VIDEO_HISTORY})
        rows = [
            {"think": {**first, "closed": True}},
            {"think": {**second, "closed": False}},
        ]
        result = _aggregate(rows, "think")
        self.assertEqual(result["valid_prediction_count"], 4)
        self.assertEqual(result["history_copy_prediction_count"], 3)
        self.assertEqual(result["history_copy_rate"], 0.75)
        self.assertEqual(result["unique_valid_sid_count"], 2)
        self.assertEqual(result["unique_history_copy_sid_count"], 1)
        self.assertEqual(result["unique_history_copy_rate"], 0.5)
        self.assertEqual(result["closure_rate"], 0.5)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["hit_rate"], 0.5)
        self.assertEqual(result["ab_hit_rate"], 0.5)
        self.assertEqual(result["a_hit_rate"], 0.5)
        self.assertEqual(result["invalid_rate"], 1 / 64)


class ValidationPoolTests(unittest.TestCase):
    def test_history_extraction_and_validation_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "dev.jsonl"
            leakage = root / "leakage.json"
            train = root / "train.jsonl"
            group_id = hashlib.sha256(b"validation").hexdigest()
            train_id = hashlib.sha256(b"train").hexdigest()
            base_prompt = (
                "旧提示\n历史 "
                "<|video_begin|><s_a_1><s_b_2><s_c_3> 和 "
                "<|prod_begin|><s_a_1><s_b_2><s_c_3>/think"
            )
            row = {
                "instruction": base_prompt,
                "input": "",
                "source_segment": "recommendation_cot",
                "aux_metadata_json": json.dumps(
                    {
                        "recommendation_group_id": group_id,
                        "recommendation_all_gold_sids": [
                            "<|video_begin|><s_a_4><s_b_5><s_c_6>"
                        ],
                    }
                ),
            }
            validation.write_text(json.dumps(row) + "\n", encoding="utf-8")
            train.write_text(
                json.dumps({"recommendation_group_id": train_id}) + "\n",
                encoding="utf-8",
            )
            leakage.write_text(
                json.dumps(
                    {
                        "validation_safe": True,
                        "dev": {
                            "recommendation_group_id_overlap": 0,
                            "recommendation_history_domain_overlap": 0,
                            "exact_full_row_overlap": 0,
                            "canonical_prompt_overlap": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            pool = load_validation_pool(validation, leakage, train)
            self.assertEqual(len(pool), 1)
            self.assertEqual(
                set(pool[0]["history_sids"]),
                {VIDEO_HISTORY, PROD_SAME_COMPONENTS},
            )
            self.assertEqual(extract_sids(pool[0]["base_prompt"]), set(pool[0]["history_sids"]))
            self.assertEqual(build_cohort(pool, 1, 20260822)[0]["group_id"], group_id)

    def test_dirty_leakage_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "dev.jsonl"
            leakage = root / "leakage.json"
            train = root / "train.jsonl"
            validation.write_text("", encoding="utf-8")
            train.write_text("", encoding="utf-8")
            leakage.write_text(
                json.dumps({"validation_safe": False, "dev": {}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "leakage"):
                load_validation_pool(validation, leakage, train)


if __name__ == "__main__":
    unittest.main()
