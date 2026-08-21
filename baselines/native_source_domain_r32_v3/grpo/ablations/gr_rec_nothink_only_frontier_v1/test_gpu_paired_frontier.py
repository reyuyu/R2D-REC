"""CPU-only static and pure-contract tests for the Frontier paired audit."""

import ast
import inspect
import unittest

import audit_gpu_paired_frontier as audit


class PairedAuditContractTests(unittest.TestCase):
    def test_fixed_contract(self):
        self.assertEqual(audit.M_NO, 8)
        self.assertEqual(audit.EPSILON, 0.2)
        self.assertEqual(audit.REPRESENTATIVE_LABELS["c"], "c_frontier_heavy")
        self.assertIn("checkpoint-1250", audit.HIER_CHECKPOINT)

    def test_harness_has_no_optimizer_or_scheduler_step(self):
        tree = ast.parse(inspect.getsource(audit))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "step":
                    forbidden.append(ast.unparse(node.func))
        self.assertEqual(forbidden, [])

    def test_zero_step_flag_is_required(self):
        with self.assertRaises(SystemExit):
            audit.parse_args(["--output", "/tmp/result.json"])


if __name__ == "__main__":
    unittest.main()
