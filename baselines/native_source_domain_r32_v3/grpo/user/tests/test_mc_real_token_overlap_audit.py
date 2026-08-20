import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_mc_real_token_overlap import (  # noqa: E402
    classify_overlap_pair,
    final_conclusion,
    summarize_candidate_overlaps,
)


def _unit(delta, char_span, token_span=(0, 1), generated=(0,)):
    return {
        "delta": delta,
        "char_start": char_span[0],
        "char_end": char_span[1],
        "token_start": token_span[0],
        "token_end": token_span[1],
        "generated_token_indices": list(generated),
    }


class RealTokenOverlapAuditTests(unittest.TestCase):
    def test_overlap_statistics_nonzero_and_mixed_sign(self):
        units = [
            _unit(0.5, (0, 4), generated=(7,)),
            _unit(-0.2, (5, 9), generated=(7,)),
            _unit(0.0, (10, 12), generated=(7,)),
        ]
        summary = summarize_candidate_overlaps(units, generated=True)
        self.assertEqual(summary["overlap_token_count"], 1)
        self.assertEqual(summary["overlap_unit_pair_count"], 1)
        self.assertEqual(summary["max_units_per_token"], 2)
        self.assertEqual(summary["same_sign_overlap_token_count"], 0)
        self.assertEqual(summary["mixed_sign_overlap_token_count"], 1)
        self.assertEqual(summary["overlaps"], {7: [0, 1]})

    def test_tokenizer_boundary_overlap_classification(self):
        first = _unit(0.5, (10, 14))
        second = _unit(0.4, (15, 19))
        actual = classify_overlap_pair(
            first,
            second,
            token_offset=(13, 16),
            canonical_overlap=True,
            generated_overlap=True,
            canonical_equals_generated=True,
        )
        self.assertEqual(actual, "TOKENIZER_BOUNDARY_OVERLAP")

    def test_char_span_overlap_has_priority(self):
        first = _unit(0.5, (10, 16))
        second = _unit(-0.4, (15, 19))
        actual = classify_overlap_pair(
            first,
            second,
            token_offset=(15, 16),
            canonical_overlap=True,
            generated_overlap=True,
            canonical_equals_generated=True,
        )
        self.assertEqual(actual, "CHAR_SPAN_OVERLAP")

    def test_projection_only_overlap_classification(self):
        first = _unit(0.5, (10, 14))
        second = _unit(-0.4, (15, 19))
        actual = classify_overlap_pair(
            first,
            second,
            token_offset=None,
            canonical_overlap=False,
            generated_overlap=True,
            canonical_equals_generated=False,
        )
        self.assertEqual(actual, "PROJECTION_BUG")

    def test_conclusion_mapping(self):
        route = {
            "candidates": [
                {
                    "overlap_records": [
                        {
                            "pair_classifications": [
                                {"classification": "TOKENIZER_BOUNDARY_OVERLAP"}
                            ]
                        }
                    ]
                }
            ]
        }
        self.assertEqual(
            final_conclusion([route]), "TOKENIZER_BOUNDARY_OVERLAP_CONFIRMED"
        )


if __name__ == "__main__":
    unittest.main()
