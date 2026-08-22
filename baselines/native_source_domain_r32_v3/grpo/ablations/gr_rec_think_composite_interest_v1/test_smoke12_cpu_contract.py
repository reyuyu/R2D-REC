from __future__ import annotations

import json
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from .preflight_contract import evaluate_preflight_conditions, sid_runtime_observation
from .run_gr_rec_think_composite_interest_v1 import (
    CHECKPOINT_STEPS,
    PROBE_IDS,
    PROBE_STEPS,
    launch_contract,
    parser as formal_parser,
    prepare_plan,
    launch_training,
)
from . import gpu_preflight
from .run_gr_rec_think_composite_interest_v1_smoke12 import future_launch_command
from .single_node_nccl import configure_single_node_nccl
from .smoke12_contract import (
    evaluate_smoke_conditions,
    smoke12_plan,
    summarize_composite_smoke,
    synthetic_smoke12_fixture,
)


def passing_preflight(**overrides):
    observations = {
        "raw_decode_runtime": True,
        "online_reward_parity": True,
        "online_advantage_parity": True,
        "ddp_g4_alignment": True,
        "gold_leakage": False,
        "loss_finite": True,
        "grad_finite": True,
        "lora_has_gradient": True,
        "base_has_gradient": False,
        "checksum_unchanged": True,
        "base_changed": False,
        "nccl_error": False,
        "oom": False,
        "nan": False,
        "inf": False,
    }
    observations.update(overrides)
    return evaluate_preflight_conditions(**observations)


class PreflightContractTests(unittest.TestCase):
    def test_sid_status_is_observed_and_nonblocking(self):
        self.assertEqual(sid_runtime_observation(0), {
            "generated_sid_candidate_count": 0,
            "sid_evidence_runtime_visible": False,
            "sid_evidence_runtime_status": "NO_SID_GENERATED_IN_THIS_BATCH",
        })
        self.assertEqual(sid_runtime_observation(3)["sid_evidence_runtime_status"], "VISIBLE")
        self.assertTrue(passing_preflight()["preflight_pass"])

    def test_hard_gate_failures_are_reported(self):
        cases = {
            "online_reward_parity": {"online_reward_parity": False},
            "ddp_g4_alignment": {"ddp_g4_alignment": False},
            "nccl_error_absent": {"nccl_error": True},
            "base_unchanged": {"base_changed": True},
        }
        for failure, override in cases.items():
            with self.subTest(failure=failure):
                result = passing_preflight(**override)
                self.assertFalse(result["preflight_pass"])
                self.assertIn(failure, result["failure_reasons"])


class SmokeContractTests(unittest.TestCase):
    def test_smoke_topology_and_launch_switches(self):
        plan = smoke12_plan()
        self.assertEqual(plan["optimizer_steps"], 12)
        self.assertEqual(plan["num_iterations"], 2)
        self.assertEqual(plan["fresh_rollouts"], 6)
        self.assertEqual(plan["groups_per_fresh_rollout"], 4)
        self.assertEqual(plan["g4_count"], 24)
        self.assertEqual(plan["candidate_count"], 96)
        smoke = launch_contract(enable_probes=False, enable_checkpoints=False, smoke_mode=True)
        formal = launch_contract()
        self.assertFalse(smoke["enable_probes"])
        self.assertFalse(smoke["enable_checkpoints"])
        self.assertEqual(smoke["save_config"], {"save_strategy": "no"})
        self.assertTrue(formal["enable_probes"])
        self.assertTrue(formal["enable_checkpoints"])
        self.assertEqual(formal["save_config"]["save_strategy"], "steps")

    def test_callbacks_are_guarded_without_mutating_global_schedules(self):
        source = inspect.getsource(launch_training)
        self.assertIn('if contract["enable_checkpoints"]:', source)
        self.assertIn('if contract["enable_probes"]:', source)
        self.assertIn("trainer.add_callback(MilestoneSaveCallback())", source)
        self.assertIn("trainer.add_callback(CompositeProbeCallback(probe_evaluator))", source)

    def test_preflight_source_does_not_hardcode_sid_visibility(self):
        source = inspect.getsource(gpu_preflight)
        self.assertNotIn('"sid_evidence_runtime_visible": True', source)
        self.assertIn("sid_runtime_observation(generated_sid_count)", source)

    def test_formal_contract_and_shared_cohort_are_unchanged(self):
        args = formal_parser().parse_args(["--dry-run"])
        plan = prepare_plan(args)
        self.assertEqual(args.max_steps, 716)
        self.assertEqual(plan["topology"]["training_groups"], 1432)
        self.assertEqual(plan["topology"]["fresh_g4_rollouts"], 358)
        self.assertEqual(len(PROBE_IDS), 12)
        self.assertEqual(PROBE_STEPS, (0, 200, 400, 600, 716))
        self.assertEqual(CHECKPOINT_STEPS, (200, 400, 600, 716))
        self.assertEqual(plan["train_probe_overlap"], 0)

    def test_future_launch_command_is_only_a_template(self):
        command = future_launch_command(29991)
        self.assertEqual(command[:3], ["torchrun", "--nproc_per_node=4", "--master_addr=127.0.0.1"])
        self.assertIn("--master_port=29991", command)


class SmokeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = synthetic_smoke12_fixture()
        self.summary = summarize_composite_smoke(
            self.fixture["manifest"], self.fixture["metrics"],
            self.fixture["rollouts"], self.fixture["composite_events"],
        )

    def test_synthetic_fixture_and_summary(self):
        self.assertTrue(self.fixture["manifest"]["synthetic"])
        self.assertEqual(self.summary["optimizer_steps"], 12)
        self.assertEqual(self.summary["fresh_rollouts"], 6)
        self.assertEqual(self.summary["g4_count"], 24)
        self.assertEqual(self.summary["candidate_count"], 96)
        self.assertEqual(set(self.summary["k_distribution"]), {"0", "1", "2", "3", "4+"})
        self.assertGreater(self.summary["rescued_zero_count"], 0)
        self.assertGreater(self.summary["composite_zero_std_count"], 0)
        self.assertGreater(self.summary["tie_break_count"], 0)
        self.assertGreater(self.summary["strict_reversal_count"], 0)
        self.assertGreater(self.summary["parser_failure_rate"], 0)
        self.assertTrue(evaluate_smoke_conditions(self.summary)["smoke_pass"])

    def test_smoke_pass_evaluator_detects_failure(self):
        broken = dict(self.summary, optimizer_steps=11, nccl_error=True)
        result = evaluate_smoke_conditions(broken)
        self.assertFalse(result["smoke_pass"])
        self.assertIn("optimizer_steps_12", result["failure_reasons"])
        self.assertIn("nccl_error_absent", result["failure_reasons"])

    def test_monitor_apis_consume_future_smoke_schema(self):
        monitor_dir = Path(__file__).parents[2] / "scripts" / "monitor"
        sys.path.insert(0, str(monitor_dir))
        try:
            from server import create_app
            with tempfile.TemporaryDirectory() as directory:
                run = Path(directory) / "smoke"
                run.mkdir()
                (run / "manifest.json").write_text(
                    json.dumps(self.fixture["manifest"]), encoding="utf-8"
                )
                (run / "composite_interest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in self.fixture["composite_events"]),
                    encoding="utf-8",
                )
                client = TestClient(create_app(run_dir=run))
                captured = client.get("/api/composite-interest").json()
                summary = client.get("/api/composite-interest/summary").json()
                advantages = client.get("/api/advantages").json()
                self.assertTrue(captured["supported"])
                self.assertEqual(len(captured["groups"]), 24)
                self.assertTrue(summary["supported"])
                self.assertEqual(sum(row["group_count"] for row in summary["rows"]), 24)
                self.assertEqual(advantages["provenance"]["mode"], "captured")
        finally:
            sys.path.remove(str(monitor_dir))


class NcclMockTests(unittest.TestCase):
    def test_initialize_false_sets_loopback_without_process_group(self):
        calls = []
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda rank: calls.append(("set_device", rank))),
            device=lambda value: value,
        )
        fake_dist = SimpleNamespace(
            is_initialized=lambda: False,
            init_process_group=lambda **kwargs: calls.append(("init_process_group", kwargs)),
        )
        with patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "4"}, clear=False):
            result = configure_single_node_nccl(
                initialize=False, torch_module=fake_torch, dist_module=fake_dist
            )
        self.assertEqual(result["nccl_socket_ifname"], "lo")
        self.assertEqual(calls, [("set_device", 0)])
        self.assertFalse(result["initialized_here"])


if __name__ == "__main__":
    unittest.main()
