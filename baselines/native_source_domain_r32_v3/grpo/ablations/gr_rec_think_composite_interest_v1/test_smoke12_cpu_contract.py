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
import torch

from .preflight_contract import (
    evaluate_preflight_conditions,
    float_vectors_close,
    masked_token_tuples,
    post_shuffle_association_parity,
    prepare_preflight_loss_context,
    raw_decode_diagnostics,
    sid_runtime_observation,
)
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
from . import run_gr_rec_think_composite_interest_v1_smoke12 as smoke_runner
from .single_node_nccl import configure_single_node_nccl
from .smoke12_parameter_audit import Smoke12ParameterAuditCallback
from .runtime_import_provenance import (
    EXPECTED_SCRIPTS_DIR,
    assert_runtime_import_provenance,
    evaluate_runtime_import_provenance,
)
from .summarize_composite_smoke12 import summarize_run, write_summary
from .smoke12_contract import (
    evaluate_smoke_conditions,
    smoke12_plan,
    summarize_composite_smoke,
    synthetic_smoke12_fixture,
)


def passing_preflight(**overrides):
    observations = {
        "raw_decode_runtime": True,
        "beam_fixed_domain_prefix": True,
        "beam_abc3": True,
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
    class Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return ":".join(str(value) for value in token_ids)

    def test_masked_raw_decode_ignores_right_padding(self):
        result = raw_decode_diagnostics(
            self.Tokenizer(),
            torch.tensor([[1, 2, 3, 0, 0]]),
            torch.tensor([[1, 1, 1, 0, 0]]),
            [{"rank": 0, "local_index": 0, "completion": "1:2:3"}],
            rank=0,
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["mismatch_count"], 0)
        self.assertEqual(masked_token_tuples(
            [[1, 2, 3, 0, 0]], [[1, 1, 1, 0, 0]],
        ), [(1, 2, 3)])

    def test_unmasked_decode_reproduces_old_padding_mismatch(self):
        decoded = self.Tokenizer.decode([1, 2, 3, 0, 0], skip_special_tokens=False)
        self.assertNotEqual(decoded, "1:2:3")

    @staticmethod
    def parity_batch(order, advantages):
        prompts = [[10 + value, 0] for value in order]
        completions = [[20 + value, 0] for value in order]
        masks = [[1, 0] for _ in order]
        return {
            "prompt_ids": torch.tensor(prompts),
            "prompt_mask": torch.tensor(masks),
            "completion_ids": torch.tensor(completions),
            "completion_mask": torch.tensor(masks),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
        }

    def test_pre_shuffle_advantage_vector_parity(self):
        self.assertTrue(float_vectors_close([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]))

    def test_synchronized_post_shuffle_preserves_associations(self):
        pre = self.parity_batch([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        post = self.parity_batch([2, 0, 3, 1], [0.3, 0.1, 0.4, 0.2])
        self.assertTrue(post_shuffle_association_parity(pre, post))
        self.assertFalse(float_vectors_close(
            pre["advantages"].tolist(), post["advantages"].tolist(),
        ))

    def test_advantage_only_shuffle_breaks_associations(self):
        pre = self.parity_batch([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
        broken = self.parity_batch([0, 1, 2, 3], [0.3, 0.1, 0.4, 0.2])
        self.assertFalse(post_shuffle_association_parity(pre, broken))

    def test_preflight_loss_context_uses_frozen_accumulation(self):
        trainer = SimpleNamespace(args=SimpleNamespace(gradient_accumulation_steps=1))
        self.assertFalse(hasattr(trainer, "current_gradient_accumulation_steps"))
        value = prepare_preflight_loss_context(trainer)
        self.assertEqual(value, 1)
        self.assertEqual(trainer.current_gradient_accumulation_steps, 1)

    def test_shared_grpo_config_freezes_gradient_accumulation_to_one(self):
        from scripts import run_grpo_trl_smoke

        source = inspect.getsource(run_grpo_trl_smoke.make_grpo_config)
        self.assertIn("gradient_accumulation_steps=1", source)

    def test_preflight_uses_compute_loss_without_training_step(self):
        source = inspect.getsource(gpu_preflight.main)
        context_index = source.index("prepare_preflight_loss_context(trainer)")
        loss_index = source.index("trainer.compute_loss(model, prepared)")
        self.assertLess(context_index, loss_index)
        self.assertIn("with trainer.compute_loss_context_manager():", source)
        self.assertNotIn("trainer._compute_loss(", source)
        self.assertNotIn("trainer.training_step(", source)

    def test_preflight_captures_snapshot_without_production_trainer_change(self):
        source = inspect.getsource(gpu_preflight.PreflightTrainer)
        self.assertIn("self.preflight_pre_shuffle", source)
        production_source = inspect.getsource(
            gpu_preflight.ThinkCompositeInterestRecGRPOTrainer
        )
        self.assertNotIn("preflight_pre_shuffle", production_source)

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
            "beam_fixed_domain_prefix": {"beam_fixed_domain_prefix": False},
            "beam_abc3": {"beam_abc3": False},
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
    def test_current_worktree_runtime_import_provenance(self):
        result = assert_runtime_import_provenance()
        self.assertEqual(result["runtime_import_provenance"], "PASS")
        self.assertTrue(result["monitor_write_composite_available"])
        self.assertEqual(
            Path(result["monitor_writer_module_path"]),
            EXPECTED_SCRIPTS_DIR / "monitor" / "writer.py",
        )

    def test_stale_monitor_provenance_fails_closed(self):
        expected_trainer = EXPECTED_SCRIPTS_DIR / "grpo_trl_trainer.py"
        stale = SimpleNamespace(
            __file__="/stale/scripts/monitor/writer.py",
            MonitorWriter=type("MonitorWriter", (), {}),
        )
        trainer = SimpleNamespace(__file__=str(expected_trainer))
        result = evaluate_runtime_import_provenance(trainer, stale)
        self.assertEqual(result["runtime_import_provenance"], "FAIL")
        self.assertFalse(result["monitor_write_composite_available"])

    def test_runtime_guard_precedes_nccl_and_model_load(self):
        source = inspect.getsource(launch_training)
        self.assertLess(
            source.index("assert_runtime_import_provenance()"),
            source.index("configure_single_node_nccl(initialize=True)"),
        )
        preflight_source = inspect.getsource(gpu_preflight.main)
        self.assertLess(
            preflight_source.index("assert_runtime_import_provenance()"),
            preflight_source.index("configure_single_node_nccl(initialize=True)"),
        )

    def test_training_chain_has_no_stale_absolute_scripts_priority(self):
        scripts_dir = Path(__file__).parents[2] / "scripts"
        for filename in ("grpo_trl_trainer.py", "run_grpo_trl_smoke.py"):
            source = (scripts_dir / filename).read_text(encoding="utf-8")
            self.assertNotIn('sys.path.insert(0, "/data/GRPO/scripts")', source)
            self.assertIn("Path(__file__).resolve().parent", source)
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
        self.assertIn('if contract["smoke_mode"]:', source)
        self.assertIn("trainer.add_callback(parameter_audit)", source)
        self.assertIn("smoke_mode=True", inspect.getsource(smoke_runner.main))
        self.assertNotIn("Smoke12ParameterAuditCallback", inspect.getsource(smoke_runner.main))

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
            self.fixture["parameter_audit"],
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
        broken = dict(self.summary, metrics_optimizer_steps=11, nccl_error=True)
        result = evaluate_smoke_conditions(broken)
        self.assertFalse(result["smoke_pass"])
        self.assertIn("metrics_optimizer_steps_12", result["failure_reasons"])
        self.assertIn("nccl_error_absent", result["failure_reasons"])

    def test_parameter_and_runtime_step_failures_are_hard(self):
        cases = {
            "lora_changed": {"LORA_CHANGED": False},
            "base_unchanged": {"BASE_CHANGED": True, "BASE_DELTA": 1},
            "runtime_optimizer_steps_12": {"runtime_optimizer_steps": 11},
        }
        for failure, override in cases.items():
            with self.subTest(failure=failure):
                result = evaluate_smoke_conditions(dict(self.summary, **override))
                self.assertFalse(result["smoke_pass"])
                self.assertIn(failure, result["failure_reasons"])

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
                (run / "metrics.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in self.fixture["metrics"]),
                    encoding="utf-8",
                )
                (run / "rollouts.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in self.fixture["rollouts"]),
                    encoding="utf-8",
                )
                (run / "smoke_parameter_audit.json").write_text(
                    json.dumps(self.fixture["parameter_audit"]), encoding="utf-8"
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
                smoke_summary = summarize_run(run)
                self.assertTrue(smoke_summary["smoke_pass"])
                self.assertEqual(smoke_summary["runtime_optimizer_steps"], 12)
                self.assertTrue(smoke_summary["LORA_CHANGED"])
                written, output = write_summary(run, run / "smoke12_summary.json")
                self.assertTrue(written["smoke_pass"])
                self.assertTrue(output.is_file())
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


class TinyParameterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_weight = torch.nn.Parameter(torch.zeros(2, 2))


class ParameterAuditTests(unittest.TestCase):
    def test_lora_delta_and_unchanged_base(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyParameterModel()
            callback = Smoke12ParameterAuditCallback(model, rank=0, run_dir=directory)
            callback.on_train_begin(None, None, None)
            with torch.no_grad():
                model.lora_weight.add_(0.5)
            payload = callback.audit(12)
            self.assertTrue(payload["LORA_CHANGED"])
            self.assertGreater(payload["lora_total_l2_delta"], 0)
            self.assertGreater(payload["lora_max_abs_delta"], 0)
            self.assertFalse(payload["BASE_CHANGED"])
            self.assertEqual(payload["BASE_DELTA"], 0)
            self.assertEqual(payload["base_version_changed_count"], 0)
            self.assertTrue(payload["finite"])

    def test_unchanged_lora_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = Smoke12ParameterAuditCallback(
                TinyParameterModel(), rank=0, run_dir=directory
            )
            callback.on_train_begin(None, None, None)
            payload = callback.audit(12)
            self.assertFalse(payload["LORA_CHANGED"])
            self.assertEqual(payload["lora_total_l2_delta"], 0)
            self.assertEqual(payload["lora_max_abs_delta"], 0)

    def test_base_in_place_mutation_trips_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyParameterModel()
            callback = Smoke12ParameterAuditCallback(model, rank=0, run_dir=directory)
            callback.on_train_begin(None, None, None)
            with torch.no_grad():
                model.base.weight.add_(1)
            payload = callback.audit(12)
            self.assertTrue(payload["BASE_CHANGED"])
            self.assertNotEqual(payload["BASE_DELTA"], 0)
            self.assertEqual(payload["base_version_changed_count"], 1)

    def test_runtime_error_artifact_never_looks_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = Smoke12ParameterAuditCallback(
                TinyParameterModel(), rank=0, run_dir=directory
            )
            payload = callback.write_runtime_error(4)
            self.assertTrue(payload["runtime_error"])
            self.assertFalse(payload["finite"])
            self.assertIsNone(payload["LORA_CHANGED"])
            self.assertEqual(payload["runtime_optimizer_steps"], 4)


if __name__ == "__main__":
    unittest.main()
