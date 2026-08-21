"""CPU-only pure helper tests for the Smoke24 forensic summary."""

import unittest

from summarize_frontier_smoke import correlation, numeric


class FrontierSmokeSummaryTests(unittest.TestCase):
    def test_numeric_summary(self):
        self.assertEqual(numeric([1, 2, 3])["median"], 2)
        self.assertEqual(numeric([1, 2, 3])["mean"], 2)
        self.assertEqual(numeric([])["n"], 0)

    def test_correlation(self):
        self.assertAlmostEqual(correlation([0, 1, 2], [1, 2, 3]), 1.0)
        self.assertIsNone(correlation([0, 0], [1, 2]))


if __name__ == "__main__":
    unittest.main()
