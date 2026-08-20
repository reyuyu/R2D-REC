import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_real_smoke import (  # noqa: E402
    SMOKE_CONFIG,
    MCRealSmokeError,
    fixed_completion_texts,
    gpu_preflight,
    run_cli,
    select_fixed_rows,
    validate_fixed_rollout,
)
from user_marginal_credit import action_marginal_credit, chain_marginal_credit  # noqa: E402


A = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
B = "<|prod_begin|><s_a_4><s_b_5><s_c_6>"
X = "<|ad_begin|><s_a_7><s_b_8><s_c_9>"
E1 = {
    "date": "2026-01-01",
    "action": "watch alpha",
    "logic": "likes alpha after a clear observed interaction",
}
E2 = {
    "date": "2026-01-02",
    "action": "buy beta",
    "logic": "needs beta after comparing relevant alternatives",
}


def chain_json(events):
    return json.dumps(
        {"logic_chain": {"name": "test", "events": events}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def rows():
    action = {
        "sample_id": "b-action",
        "route": "action",
        "gold_sids": [A, B],
        "history_sids": [A, B],
        "raw_gold_output": json.dumps([A, B], separators=(",", ":")),
    }
    earlier_action = {
        **action,
        "sample_id": "a-action",
    }
    chain = {
        "sample_id": "a-chain",
        "route": "chain",
        "gold_sids": [],
        "history_sids": [],
        "gold_events": [E1, E2],
        "history_events": [
            {"date": E1["date"], "raw": E1["action"]},
            {"date": E2["date"], "raw": E2["action"]},
        ],
        "raw_gold_output": chain_json([E1, E2]),
    }
    vocabulary = {
        "sample_id": "z-vocabulary",
        "route": "action",
        "gold_sids": [X],
        "history_sids": [X],
        "raw_gold_output": json.dumps([X]),
    }
    return [action, chain, vocabulary, earlier_action]


def fake_rollout(route, units):
    return {
        "route": route,
        "K": 2,
        "credit_units_per_candidate": units,
    }


class FixedCompletionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_uses_real_non_gold_sid(self):
        first = select_fixed_rows(rows())
        second = select_fixed_rows(list(reversed(rows())))
        self.assertEqual(first["action"]["sample_id"], "a-action")
        self.assertEqual(first["chain"]["sample_id"], "a-chain")
        self.assertEqual(first["action_false_positive"], X)
        self.assertEqual(
            {key: value["sample_id"] if isinstance(value, dict) else value for key, value in first.items()},
            {key: value["sample_id"] if isinstance(value, dict) else value for key, value in second.items()},
        )

    def test_action_fixed_k2_has_required_positive_and_negative_credit(self):
        selection = select_fixed_rows(rows())
        completions = fixed_completion_texts(selection)["action"]
        results = [action_marginal_credit(text, selection["action"]) for text in completions]
        units = [result["credits"] for result in results]
        summary = validate_fixed_rollout("action", fake_rollout("action", units))
        self.assertGreater(summary["positive_unit_count"], 0)
        self.assertGreater(summary["negative_unit_count"], 0)
        self.assertTrue(any(unit["delta"] > 0 for unit in units[1]))
        self.assertTrue(any(unit["delta"] < 0 for unit in units[1]))

    def test_chain_fixed_k2_has_required_positive_and_negative_credit(self):
        selection = select_fixed_rows(rows())
        completions = fixed_completion_texts(selection)["chain"]
        results = [chain_marginal_credit(text, selection["chain"]) for text in completions]
        units = [result["credits"] for result in results]
        summary = validate_fixed_rollout("chain", fake_rollout("chain", units))
        self.assertGreater(summary["positive_unit_count"], 0)
        self.assertGreater(summary["negative_unit_count"], 0)
        self.assertTrue(any(unit["delta"] < 0 for unit in units[1]))

    def test_route_rollouts_must_be_homogeneous(self):
        with self.assertRaisesRegex(MCRealSmokeError, "route"):
            validate_fixed_rollout("action", fake_rollout("chain", [[], []]))


class GateAndGPUPreflightTests(unittest.TestCase):
    def test_without_execute_never_calls_model_execution(self):
        preflight = Mock(
            return_value={
                "status": "READY_TO_EXECUTE",
                "gpu": {"index": 0},
                "selected_sample_ids": {"action": "a", "chain": "c"},
                "fixed_k2": {},
                "paths": {"train_sha256": "sha"},
            }
        )
        execute = Mock()
        with patch("run_mc_user_real_smoke.load_beta_model") as model_loader:
            output = run_cli(
                ["--gpu-id", "0"], preflight_fn=preflight, execute_fn=execute
            )
        self.assertEqual(output["status"], "READY_TO_EXECUTE")
        execute.assert_not_called()
        model_loader.assert_not_called()

    def test_execute_gate_calls_execution_only_when_explicit(self):
        preflight_result = {"status": "READY_TO_EXECUTE"}
        preflight = Mock(return_value=preflight_result)
        execute = Mock(return_value={"status": "PASS"})
        output = run_cli(
            ["--gpu-id", "0", "--execute"],
            preflight_fn=preflight,
            execute_fn=execute,
        )
        self.assertEqual(output["status"], "PASS")
        execute.assert_called_once()

    def test_busy_gpu_is_refused(self):
        responses = [
            SimpleNamespace(stdout="0, GPU-0, 100, 80000, 0\n"),
            SimpleNamespace(stdout="GPU-0, 123, python, 100\n"),
        ]
        with self.assertRaisesRegex(MCRealSmokeError, "busy"):
            gpu_preflight(0, run_command=Mock(side_effect=responses))

    def test_idle_gpu_passes(self):
        responses = [
            SimpleNamespace(stdout="0, GPU-0, 100, 80000, 0\n"),
            SimpleNamespace(stdout=""),
        ]
        state = gpu_preflight(0, run_command=Mock(side_effect=responses))
        self.assertEqual(state["index"], 0)
        self.assertEqual(state["compute_processes"], [])

    def test_memory_threshold_is_strict(self):
        responses = [
            SimpleNamespace(stdout="0, GPU-0, 1024, 80000, 0\n"),
            SimpleNamespace(stdout=""),
        ]
        with self.assertRaisesRegex(MCRealSmokeError, "busy"):
            gpu_preflight(0, run_command=Mock(side_effect=responses))

    def test_config_contains_no_grpo_or_ppo_fields(self):
        forbidden = {
            "old_per_token_logps",
            "epsilon",
            "kl",
            "sequence_advantage",
            "token_advantage",
            "penalty_mask",
            "group_mean",
            "group_std",
            "ppo",
            "grpo",
        }
        self.assertTrue(set(SMOKE_CONFIG).isdisjoint(forbidden))
        source = (SCRIPTS_DIR / "run_mc_user_real_smoke.py").read_text(encoding="utf-8")
        self.assertNotIn(".generate(", source)


if __name__ == "__main__":
    unittest.main()
