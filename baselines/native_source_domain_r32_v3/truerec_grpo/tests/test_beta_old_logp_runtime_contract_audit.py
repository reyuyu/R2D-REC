from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
sys.path.insert(0, str(ROOT / "trainer"))

from beta_old_logp_rescore_validation import TRAINER_SHA256
from policy_scoring_v1 import POLICY_SCORING_MODE, SCORING_MICROBATCH_SIZE
from rollout_runtime_v1 import GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, OLD_LOGPS_FROM_FULL_FORWARD_RESCORE, PPO_OLD_LOGP_SOURCE


class OldLogpRuntimeContractAuditTest(unittest.TestCase):
    def test_01_promoted_contract_constants(self):
        self.assertEqual(PPO_OLD_LOGP_SOURCE, "FULL_FORWARD_RESCORE")
        self.assertTrue(OLD_LOGPS_FROM_FULL_FORWARD_RESCORE)
        self.assertFalse(GENERATION_SCORES_USED_FOR_PPO)
        self.assertEqual(GENERATION_SCORE_LOGPS_ROLE, "DIAGNOSTIC_ONLY")

    def test_02_scoring_contract(self):
        self.assertEqual((POLICY_SCORING_MODE, SCORING_MICROBATCH_SIZE), ("eval", 2))

    def test_03_trainer_core_sha_unchanged(self):
        blob = subprocess.check_output(["git", "-C", str(ROOT.parents[2]), "show", "HEAD:baselines/native_source_domain_r32_v3/truerec_grpo/trainer/truerec_grpo_trainer_v1.py"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), TRAINER_SHA256)

    def test_04_audit_has_no_forbidden_calls(self):
        source = (ROOT / "diagnostics" / "beta_old_logp_runtime_contract_audit.py").read_text(encoding="utf-8")
        self.assertNotIn(".generate(", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)


if __name__ == "__main__":
    unittest.main()
