from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "trainer"))

from training_driver_v1 import (  # noqa: E402
    DEFAULT_GROUPS_PER_OPTIMIZER_STEP,
    GRPO_INIT_FAMILY,
    KL_BETA,
    LEARNING_RATE,
    OPTIMIZER_FAMILY,
    POLICY_MODE,
    TRAIN_ACTION,
    TRAIN_BRIDGE,
    TRAIN_FIXED_DOMAIN_IN_CONTEXT,
    WEIGHT_DECAY,
    DriverContractError,
    PolicyMutationError,
    TrueRecTrainingDriverV1,
    audit_pilot4096_admission,
    frozen_contract,
)


class FakeOptimizer:
    def __init__(self, events): self.events = events; self.steps = 0; self.zeroes = 0
    def zero_grad(self, *, set_to_none):
        self.assert_true(set_to_none); self.zeroes += 1; self.events.append("zero_grad")
    def step(self): self.steps += 1; self.events.append("optimizer_step")
    @staticmethod
    def assert_true(value):
        if value is not True: raise AssertionError("set_to_none must be true")


class FakeTrainer:
    def __init__(self, events, fail=False): self.events = events; self.fail = fail; self.calls = 0
    def backward_group_streaming(self, group):
        self.events.append("streaming_backward"); self.calls += 1
        if self.fail: raise RuntimeError("backward failure")
        return {"detached": True, "group": group}


def make_driver(*, rollout_fail=False, rescore_fail=False, backward_fail=False, finite=True, mutate_rollout=False, mutate_rescore=False):
    external = []
    version = {"value": 0}
    optimizer = FakeOptimizer(external)
    trainer = FakeTrainer(external, backward_fail)
    def rollout(record):
        external.append("rollout")
        if rollout_fail: raise RuntimeError("rollout failure")
        if mutate_rollout: version["value"] += 1
        return {"ids": tuple(range(8)), "record": record}
    def rescore(record, rollout_value):
        external.append("old_rescore")
        if rescore_fail: raise RuntimeError("rescore failure")
        if mutate_rescore: version["value"] += 1
        return {"recommendation_group_id": record["recommendation_group_id"], "rollout": rollout_value}
    def gradient_gate(): external.append("gradient_gate"); return finite
    driver = TrueRecTrainingDriverV1(
        rollout_fn=rollout, old_rescore_fn=rescore, trainer=trainer,
        optimizer=optimizer, gradient_finite_fn=gradient_gate,
        policy_fingerprint_fn=lambda: version["value"], groups_per_optimizer_step=1,
    )
    return driver, optimizer, trainer, external


class TrainingDriverLifecycleTest(unittest.TestCase):
    def test_exact_lifecycle_order_and_one_step(self):
        driver, optimizer, _, external = make_driver()
        driver.run_group({"recommendation_group_id": "g0"})
        expected = ["zero_grad", "rollout", "old_rescore", "streaming_backward", "gradient_gate", "optimizer_step"]
        self.assertEqual(external, expected)
        self.assertEqual(driver.lifecycle_events, expected + ["global_step_increment"])
        self.assertEqual((optimizer.zeroes, optimizer.steps, driver.state.global_step), (1, 1, 1))

    def test_three_group_counters(self):
        driver, optimizer, trainer, _ = make_driver()
        driver.run({"recommendation_group_id": f"g{i}"} for i in range(3))
        state = driver.state
        self.assertEqual((state.business_groups_seen, state.rollouts_completed), (3, 3))
        self.assertEqual((state.old_rescores_completed, state.streaming_backwards_completed), (3, 3))
        self.assertEqual((state.optimizer_steps, state.global_step), (3, 3))
        self.assertEqual((optimizer.steps, trainer.calls), (3, 3))

    def _assert_failure_blocks_step(self, **kwargs):
        driver, optimizer, _, _ = make_driver(**kwargs)
        with self.assertRaises((RuntimeError, FloatingPointError, PolicyMutationError)):
            driver.run_group({"recommendation_group_id": "failure"})
        self.assertEqual((optimizer.steps, driver.state.optimizer_steps, driver.state.global_step), (0, 0, 0))
        self.assertTrue(driver.state.failed)
        with self.assertRaises(DriverContractError):
            driver.run_group({"recommendation_group_id": "cannot-continue"})

    def test_rollout_failure_blocks_step(self): self._assert_failure_blocks_step(rollout_fail=True)
    def test_rescore_failure_blocks_step(self): self._assert_failure_blocks_step(rescore_fail=True)
    def test_backward_failure_blocks_step(self): self._assert_failure_blocks_step(backward_fail=True)
    def test_nonfinite_gradient_blocks_step(self): self._assert_failure_blocks_step(finite=False)
    def test_policy_mutation_before_rescore_blocks_step(self): self._assert_failure_blocks_step(mutate_rollout=True)
    def test_policy_mutation_during_rescore_blocks_step(self): self._assert_failure_blocks_step(mutate_rescore=True)

    def test_structure_allows_accumulation_without_hardcoding_one(self):
        external = []
        optimizer = FakeOptimizer(external)
        trainer = FakeTrainer(external)
        driver = TrueRecTrainingDriverV1(
            rollout_fn=lambda record: record, old_rescore_fn=lambda record, rollout: rollout,
            trainer=trainer, optimizer=optimizer, gradient_finite_fn=lambda: True,
            policy_fingerprint_fn=lambda: 0, groups_per_optimizer_step=2,
        )
        driver.run_group({"recommendation_group_id": "g0"})
        self.assertEqual(optimizer.steps, 0)
        driver.run_group({"recommendation_group_id": "g1"})
        self.assertEqual(optimizer.steps, 1)

    def test_frozen_contracts(self):
        contract = frozen_contract()
        self.assertEqual((GRPO_INIT_FAMILY, contract["grpo_init_family"]), ("BETA_BASELINE", "BETA_BASELINE"))
        self.assertEqual((contract["G"], contract["trainer_microbatch_size"]), (8, 2))
        self.assertEqual(contract["ppo_old_logp_source"], "FULL_FORWARD_RESCORE")
        self.assertEqual((POLICY_MODE, contract["policy_mode"]), ("EVAL", "EVAL"))
        self.assertEqual((TRAIN_BRIDGE, TRAIN_FIXED_DOMAIN_IN_CONTEXT, TRAIN_ACTION), (False, True, ("A", "B", "C")))
        self.assertEqual((contract["hpr_lambda"], KL_BETA), (0.02, 0.0))
        self.assertEqual((OPTIMIZER_FAMILY, LEARNING_RATE, WEIGHT_DECAY), ("AdamW", 1e-6, 0.0))
        self.assertEqual(DEFAULT_GROUPS_PER_OPTIMIZER_STEP, 1)

    def test_pilot_admission_unique_business_groups_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.jsonl"
            digest = hashlib.sha256()
            with records.open("wb") as handle:
                for index in range(4096):
                    row = {
                        "recommendation_group_id": f"g{index:04d}", "target_domain": "video",
                        "fixed_domain_token": "<|video_begin|>",
                        "all_gold_abc": ["<s_a_1><s_b_2><s_c_3>"],
                        "system": "system", "user_content_nothink": "prompt /no_think",
                    }
                    raw = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                    handle.write(raw); digest.update(raw)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"groups": 4096, "pilot_records_sha256": digest.hexdigest()}))
            audit = audit_pilot4096_admission(records, manifest)
            self.assertEqual((audit["record_count"], audit["unique_recommendation_group_id"]), (4096, 4096))
            self.assertEqual(audit["pilot_admission"], "PASS")


if __name__ == "__main__":
    unittest.main()
