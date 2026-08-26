from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "trainer"))

from audit_rollout_eos_pad_contract import run


class RolloutEosPadContractTest(unittest.TestCase):
    def test_cpu_contract_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = run(Path(directory))
        self.assertEqual(audit["formal_generate_eos_ids"], [151645, 151643])
        self.assertEqual(audit["formal_generate_pad_id"], 151643)
        self.assertTrue(audit["generation_contract_match_phase07"])
        self.assertEqual(audit["ppo_old_logp_source"], "FULL_FORWARD_RESCORE")
        self.assertFalse(audit["generation_scores_used_for_ppo"])


if __name__ == "__main__":
    unittest.main()
