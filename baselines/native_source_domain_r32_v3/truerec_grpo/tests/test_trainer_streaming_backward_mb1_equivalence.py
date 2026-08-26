"""CPU gates for the long-context streaming microbatch=1 production path."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "diagnostics", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from trainer_streaming_backward_equivalence import run_audit  # noqa: E402


class StreamingBackwardMB1EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = run_audit(streaming_microbatch_size=1)

    def test_all_required_cases_are_value_and_gradient_equivalent(self):
        self.assertEqual(set(self.audit["cases"]), {"HPR_A", "HPR_B", "HPR_C", "HPR_NONE", "FORMAT_INVALID"})
        self.assertTrue(self.audit["frontier_value_equivalent"])
        self.assertTrue(self.audit["hpr_value_equivalent"])
        self.assertTrue(self.audit["total_value_equivalent"])
        self.assertTrue(self.audit["gradient_equivalent"])
        self.assertTrue(all(case["status"] == "PASS" for case in self.audit["cases"].values()))

    def test_mb1_call_and_graph_lifetime_contract(self):
        self.assertEqual(self.audit["trainer_microbatch_size"], 1)
        self.assertEqual(self.audit["physical_forward_calls_per_group"], 8)
        self.assertEqual(self.audit["physical_backward_calls_per_group"], 8)
        self.assertFalse(self.audit["graph_bearing_state_retained_across_chunks"])


if __name__ == "__main__":
    unittest.main()
