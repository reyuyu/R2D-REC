from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from run_truerec_pilot_ddp_v1 import (  # noqa: E402
    apply_optimizer_gate_outcome, build_pre_optimizer_gate_diagnostic,
)
from v2_step2586_gate_debug import (  # noqa: E402
    CHECKPOINT_CURSOR, EPOCH2_LOCAL_INDEX, FAILED_GLOBAL_STEP, FAILED_GROUP_INDEX,
    EXPECTED_FAILED_GROUP_ID, trajectory_payload, trajectory_structure, zero_gradient_assessment,
)


def rank_state(rank: int, *, nonzero: int = 2, norm: float = 1.0, plan: str = "same"):
    return {
        "rank": rank,
        "calls": {"forward_calls": 1, "backward_calls": 1, "hpr_extra_forward_calls": 0},
        "gradient": {
            "lora_params_with_grad": 4,
            "lora_params_with_nonzero_grad": nonzero,
            "lora_grad_norm": norm,
            "base_params_with_grad": 0,
            "nan_grad_count": 0,
            "inf_grad_count": 0,
        },
        "runtime_plan_hash": plan,
    }


def diagnostic(states):
    return build_pre_optimizer_gate_diagnostic(
        ratio={"abs_mean": 0.0, "abs_max": 0.0, "ratio_min": 1.0, "ratio_max": 1.0},
        selected_mb=2, expected_calls=1,
        losses={"frontier": 0.0, "hpr_raw": 0.0, "hpr_weighted": 0.0, "total": 0.0},
        rank_states=states,
    )


def solved_noop_diagnostic(states, **signal_overrides):
    signal = {
        "frontier_credits_all_zero": True,
        "format_credits_all_zero": True,
        "hpr_trigger": "HPR_NONE",
        "hpr_site_count": 0,
        **signal_overrides,
    }
    return build_pre_optimizer_gate_diagnostic(
        ratio={"abs_mean": 0.0, "abs_max": 0.0, "ratio_min": 1.0, "ratio_max": 1.0},
        selected_mb=2, expected_calls=1,
        losses={"frontier": 0.0, "hpr_raw": 0.0, "hpr_weighted": 0.0, "total": 0.0},
        rank_states=states, training_signal=signal,
    )


class V2Step2586GateDebugTests(unittest.TestCase):
    def test_all_gate_components_pass(self):
        value = diagnostic([rank_state(rank) for rank in range(4)])
        self.assertTrue(value["pass"])
        self.assertEqual(value["EXACT_FAILED_SUBGATES"], [])
        self.assertTrue(all(value[name]["pass"] for name in (
            "ratio_gate", "call_gate", "loss_gate", "gradient_gate", "runtime_plan_gate",
        )))

    def test_zero_gradient_is_named_exactly(self):
        value = diagnostic([rank_state(rank, nonzero=0, norm=0.0) for rank in range(4)])
        self.assertEqual(value["EXACT_FAILED_SUBGATES"], ["ZERO_GRADIENT"])
        self.assertFalse(value["gradient_gate"]["pass"])

    def test_expected_solved_zero_gradient_passes_without_update(self):
        value = solved_noop_diagnostic([rank_state(rank, nonzero=0, norm=0.0) for rank in range(4)])
        self.assertTrue(value["pass"])
        self.assertTrue(value["pass_with_no_update"])
        self.assertEqual(value["outcome"], "PASS_WITH_NO_UPDATE")
        self.assertEqual(value["RAW_FAILED_SUBGATES"], ["ZERO_GRADIENT"])
        self.assertEqual(value["EXACT_FAILED_SUBGATES"], [])
        self.assertEqual(value["zero_gradient_classification"], "EXPECTED_SOLVED_NOOP")

    def test_expected_solved_noop_never_calls_optimizer_step(self):
        class Optimizer:
            steps = 0
            zeroes = 0
            def step(self): self.steps += 1
            def zero_grad(self, *, set_to_none):
                self.assert_set_to_none = set_to_none
                self.zeroes += 1

        optimizer = Optimizer()
        performed = apply_optimizer_gate_outcome(optimizer, {"pass_with_no_update": True})
        self.assertFalse(performed)
        self.assertEqual((optimizer.steps, optimizer.zeroes), (0, 1))
        self.assertTrue(optimizer.assert_set_to_none)

    def test_zero_gradient_with_any_training_signal_still_fails_fast(self):
        states = [rank_state(rank, nonzero=0, norm=0.0) for rank in range(4)]
        for override in (
            {"format_credits_all_zero": False},
            {"frontier_credits_all_zero": False},
            {"hpr_trigger": "HPR_C"},
            {"hpr_site_count": 1},
        ):
            with self.subTest(override=override):
                value = solved_noop_diagnostic(states, **override)
                self.assertFalse(value["pass"])
                self.assertEqual(value["outcome"], "FAIL_FAST")

    def test_mixed_zero_and_nonzero_rank_gradients_fail_fast(self):
        states = [rank_state(rank, nonzero=0, norm=0.0) for rank in range(4)]
        states[3] = rank_state(3, nonzero=1, norm=0.5)
        value = solved_noop_diagnostic(states)
        self.assertFalse(value["pass"])
        self.assertFalse(value["training_signal_gate"]["gradients_all_zero"])

    def test_runtime_hashes_are_recorded_per_rank(self):
        states = [rank_state(rank, plan="different" if rank == 3 else "same") for rank in range(4)]
        value = diagnostic(states)
        self.assertEqual(value["EXACT_FAILED_SUBGATES"], ["RUNTIME_PLAN_HASH_MISMATCH"])
        self.assertEqual(len(value["runtime_plan_gate"]["each_rank_runtime_plan_hash"]), 4)

    def test_expected_zero_gradient_requires_no_training_signal(self):
        value = diagnostic([rank_state(rank, nonzero=0, norm=0.0) for rank in range(4)])
        explain = {
            "hpr_trigger": "HPR_NONE", "hpr_sites": [],
            "candidates": [{"action_tokens": [{
                "frontier_token_credit": 0.0, "format_token_credit": 0.0,
            }]}],
        }
        self.assertEqual(zero_gradient_assessment(value, explain)["classification"], "ZERO_GRADIENT_EXPECTED")
        explain["candidates"][0]["action_tokens"][0]["frontier_token_credit"] = 1.0
        self.assertEqual(zero_gradient_assessment(value, explain)["classification"], "ZERO_GRADIENT_BUG")

    def test_trajectory_payload_ignores_only_writer_metadata(self):
        value = {"global_step": 1, "group_index": 0, "mode": "TRAIN", "optimizer_update": True, "x": [1]}
        self.assertEqual(trajectory_payload(value), {"x": [1]})

    def test_trajectory_structure_ignores_float_drift_but_not_sample_ids(self):
        value = {
            "recommendation_group_id": "g", "hpr_trigger": "HPR_NONE", "hpr_sites": [],
            "candidates": [{
                "candidate_index": 0, "source_rank": 0, "completion_ids": [1, 2, 3],
                "parsed_abc": [1, 2, 3], "format_valid": True, "A_hit": False,
                "AB_hit": False, "exact": False, "wrong_history_copy": False,
                "action_tokens": [{"old_logp": -1.0}],
            }],
        }
        drifted = {**value, "candidates": [{**value["candidates"][0], "action_tokens": [{"old_logp": -2.0}]}]}
        self.assertEqual(trajectory_structure(value), trajectory_structure(drifted))
        drifted["candidates"][0]["completion_ids"] = [9, 2, 3]
        self.assertNotEqual(trajectory_structure(value), trajectory_structure(drifted))

    def test_frozen_failure_coordinates(self):
        self.assertEqual((CHECKPOINT_CURSOR, FAILED_GROUP_INDEX, FAILED_GLOBAL_STEP), (2560, 2585, 2586))
        self.assertEqual(EPOCH2_LOCAL_INDEX, 537)
        self.assertEqual(EXPECTED_FAILED_GROUP_ID, "6005f443f291be018ce1ec7dbf3a1e512e21b477a7ff9acc10803752e2c4edcd")


if __name__ == "__main__":
    unittest.main()
