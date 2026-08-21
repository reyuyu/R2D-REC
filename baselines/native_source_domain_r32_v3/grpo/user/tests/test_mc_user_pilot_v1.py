import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_pilot_v1 import (  # noqa: E402
    ACTION_COUNT,
    CHAIN_COUNT,
    CHECKPOINT_STEPS,
    PILOT_PROMPT_COUNT,
    advance_step_counts,
    assert_gpu_process_owned,
    build_manifest,
    candidate_record,
    claim_gpu_process_ownership,
    display_rollout_records,
    run_cli,
    run_prompt_loop,
    run_preflight,
    save_adapter_checkpoint,
    select_pilot_rows,
    summarize_metrics,
    validate_prompt_metric,
)


def rows():
    result = []
    for route in ("action", "chain"):
        for index in range(24):
            result.append({"sample_id": f"{route}-{index:02d}", "route": route})
    return result


def candidate(route, *, negative=False):
    value = {
        "candidate_index": 0,
        "generated_token_count": 10,
        "format_valid": True,
        "reward": 0.5,
        "projection_required": False,
        "positive_unit_count": 1,
        "negative_unit_count": int(negative),
        "zero_unit_count": 0,
        "positive_credit_mass": 0.2,
        "negative_credit_mass": 0.1 if negative else 0.0,
        "overlap_token_count": 0,
        "mixed_sign_overlap_token_count": 0,
    }
    if route == "action":
        value["predicted_sid_unit_count"] = 2
    else:
        value.update(
            predicted_event_count=3,
            full_action_alignment=0.6,
            full_logic_alignment=0.4,
        )
    return value


class DisplayRolloutContractTests(unittest.TestCase):
    def test_action_and_chain_copy_existing_credit_without_recomputation(self):
        common_candidate = {
            "candidate_index": 0,
            "completion": "completion",
            "generated_token_count": 4,
            "format_valid": True,
            "full_reward": 0.5,
            "projection_required": False,
            "overlap_token_count": 1,
            "same_sign_overlap_token_count": 0,
            "mixed_sign_overlap_token_count": 1,
            "max_active_units_per_token": 2,
        }
        action_unit = {
            "sid": "sid-A",
            "occurrence": 2,
            "delta": -0.25,
            "credit_type": "negative",
            "generated_token_indices": [1, 2],
            "char_start": 3,
            "char_end": 8,
        }
        action = display_rollout_records(
            prompt_step=4, optimizer_step=3, route="action", sample_id="sample",
            candidates=[common_candidate], units_per_candidate=[[action_unit]],
        )[0]
        self.assertEqual(action["credit_units"][0]["delta"], -0.25)
        self.assertEqual(action["credit_units"][0]["occurrence_index"], 2)
        self.assertEqual(action["credit_units"][0]["generated_token_indices"], [1, 2])

        chain_unit = {
            "event_index": 1,
            "delta": 0.4,
            "delta_action_alignment": 0.6,
            "delta_logic_alignment": 0.2,
            "credit_type": "positive",
            "generated_token_indices": [0, 1, 2],
        }
        chain = display_rollout_records(
            prompt_step=5, optimizer_step=4, route="chain", sample_id="sample",
            candidates=[common_candidate], units_per_candidate=[[chain_unit]],
        )[0]
        self.assertEqual(chain["credit_units"][0]["event_index"], 1)
        self.assertEqual(chain["credit_units"][0]["delta_action"], 0.6)
        self.assertEqual(chain["credit_units"][0]["delta_logic"], 0.2)
        self.assertEqual(chain["full_reward"], 0.5)


def prompt_result(route, *, active=True, negative=False):
    candidates = [candidate(route, negative=negative), candidate(route)]
    return {
        "route": route,
        "sample_id": f"{route}-sample",
        "candidates": candidates,
        "active_unit_count": 1 if active else 0,
        "active_token_count": 4 if active else 0,
        "active_token_assignment_count": 4 if active else 0,
        "loss": 0.2 if active else 0.0,
        "grad_norm": 0.3 if active else 0.0,
        "skipped_update": not active,
        "optimizer_step_performed": active,
        "generation_wall_seconds": 1.0,
        "training_wall_seconds": 2.0,
        "generation_peak_vram_mib": 20.0,
        "training_peak_vram_mib": 22.0,
    }


class SelectionTests(unittest.TestCase):
    def test_selects_32_unique_strictly_alternating_and_excludes_smoke(self):
        excluded = {"action-00", "chain-00"}
        selected = select_pilot_rows(rows(), excluded_sample_ids=excluded)
        self.assertEqual(len(selected), PILOT_PROMPT_COUNT)
        self.assertEqual(len({row["sample_id"] for row in selected}), 32)
        self.assertEqual(sum(row["route"] == "action" for row in selected), ACTION_COUNT)
        self.assertEqual(sum(row["route"] == "chain" for row in selected), CHAIN_COUNT)
        self.assertEqual(
            [row["route"] for row in selected],
            [route for _ in range(16) for route in ("action", "chain")],
        )
        self.assertFalse({row["sample_id"] for row in selected} & excluded)
        self.assertEqual(
            [row["sample_id"] for row in selected],
            [row["sample_id"] for row in select_pilot_rows(rows(), excluded_sample_ids=excluded)],
        )

    def test_manifest_records_selection_and_frozen_config(self):
        selected = select_pilot_rows(rows())
        args = SimpleNamespace(base_model="base", adapter="adapter", train_data="train")
        manifest = build_manifest("run", selected, set(), args)
        self.assertEqual(len(manifest["prompts"]), 32)
        self.assertEqual(manifest["config"]["K"], 2)
        self.assertEqual(manifest["config"]["checkpoint_prompt_steps"], [8, 16, 32])
        self.assertNotIn("prompt_count_argument", manifest["config"])


class StepAndCheckpointTests(unittest.TestCase):
    def test_prompt_and_optimizer_steps_diverge_for_no_credit(self):
        self.assertEqual(
            advance_step_counts(3, 2, {"skipped_update": True, "optimizer_step_performed": False}),
            (4, 2),
        )
        self.assertEqual(
            advance_step_counts(4, 2, {"skipped_update": False, "optimizer_step_performed": True}),
            (5, 3),
        )

    def test_loop_uses_one_persistent_optimizer_and_checkpoints_8_16_32(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        seen_optimizer_ids = []
        checkpoints = []
        state = {"value": 0}

        def process(row, actual_optimizer, step):
            seen_optimizer_ids.append(id(actual_optimizer))
            self.assertEqual(state["value"], step - 1)
            state["value"] += 1
            return prompt_result(row["route"], active=step % 3 != 0)

        selected = select_pilot_rows(rows())
        records, optimizer_steps = run_prompt_loop(
            selected,
            optimizer,
            process,
            lambda prompt_step, *_args: checkpoints.append(prompt_step),
            lambda _record: None,
        )
        self.assertEqual(set(seen_optimizer_ids), {id(optimizer)})
        self.assertEqual(checkpoints, list(CHECKPOINT_STEPS))
        self.assertEqual(len(records), 32)
        self.assertEqual(optimizer_steps, 22)
        self.assertEqual(state["value"], 32)

    def test_checkpoint_is_adapter_only_and_has_pilot_state(self):
        class AdapterModel:
            def save_pretrained(self, path, safe_serialization):
                self.safe_serialization = safe_serialization
                (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")
                (Path(path) / "adapter_model.safetensors").write_bytes(b"adapter")

        model = AdapterModel()
        selected = select_pilot_rows(rows())
        records = [
            {"prompt_step": 1, "optimizer_step": 1, **prompt_result("action")}
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            save_adapter_checkpoint(model, run_dir, 8, 7, selected[:8], records, 3.0)
            checkpoint = run_dir / "checkpoints" / "prompt-step-0008"
            state = __import__("json").loads(
                (checkpoint / "pilot_state.json").read_text(encoding="utf-8")
            )
            self.assertTrue(model.safe_serialization)
            self.assertTrue((checkpoint / "adapter_model.safetensors").is_file())
            self.assertFalse((checkpoint / "model.safetensors").exists())
            self.assertEqual(state["prompt_step"], 8)
            self.assertEqual(state["optimizer_step"], 7)
            self.assertEqual(len(state["sample_ids_processed"]), 8)


class MetricsTests(unittest.TestCase):
    def test_candidate_contract_action_and_chain(self):
        raw = {
            "candidate_index": 0,
            "generated_token_count": 8,
            "format_valid": True,
            "full_reward": 0.7,
            "projection_required": False,
            "overlap_token_count": 1,
            "mixed_sign_overlap_token_count": 1,
        }
        units = [{"delta": 0.2}, {"delta": -0.1}, {"delta": 0.0}]
        action = candidate_record(raw, units, {"valid": True, "credits": [1, 2, 3]}, "action")
        chain = candidate_record(
            raw,
            units,
            {
                "valid": True,
                "credits": [1, 2],
                "full_action_alignment": 0.8,
                "full_logic_alignment": 0.6,
            },
            "chain",
        )
        self.assertEqual(action["predicted_sid_unit_count"], 3)
        self.assertEqual(chain["predicted_event_count"], 2)
        self.assertEqual(chain["negative_unit_count"], 1)
        self.assertAlmostEqual(chain["negative_credit_mass"], 0.1)

    def test_metrics_contract_and_chain_negative_monitoring(self):
        records = []
        for step, route in enumerate(("action", "chain"), 1):
            record = {
                "prompt_step": step,
                "optimizer_step": step,
                **prompt_result(route, negative=route == "chain"),
            }
            validate_prompt_metric(record)
            records.append(record)
        summary = summarize_metrics(records, 9.0)
        self.assertEqual(summary["chain"]["candidate_count"], 2)
        self.assertEqual(summary["chain"]["candidate_with_negative_unit_count"], 1)
        self.assertEqual(summary["chain"]["negative_candidate_rate"], 0.5)
        self.assertEqual(summary["chain"]["mean_predicted_event_count"], 3.0)
        self.assertEqual(summary["optimizer_update_count"], 2)


class SafetyGateTests(unittest.TestCase):
    def test_without_execute_never_calls_model_path(self):
        manifest = {
            "run_id": "run",
            "train_sha256": "sha",
            "prompts": [
                {"route": route, "sample_id": str(index)}
                for index, route in enumerate([r for _ in range(16) for r in ("action", "chain")])
            ],
        }
        preflight = Mock(
            return_value={
                "status": "READY_TO_EXECUTE",
                "manifest": manifest,
                "gpu": {"index": 0},
                "run_dir": Path("run"),
            }
        )
        execute = Mock()
        output = run_cli(["--gpu-id", "0"], preflight_fn=preflight, execute_fn=execute)
        self.assertEqual(output["status"], "READY_TO_EXECUTE")
        execute.assert_not_called()

    def test_execute_gate_calls_once(self):
        preflight = Mock(return_value={"status": "READY_TO_EXECUTE"})
        execute = Mock(return_value={"status": "PASS"})
        output = run_cli(
            ["--gpu-id", "0", "--execute"], preflight_fn=preflight, execute_fn=execute
        )
        self.assertEqual(output["status"], "PASS")
        execute.assert_called_once()

    @staticmethod
    def gpu_responses(selected_processes=(), other_processes=()):
        process_lines = [
            f"GPU-0, {pid}, {name}, {used}"
            for pid, name, used in selected_processes
        ] + [
            f"GPU-1, {pid}, {name}, {used}"
            for pid, name, used in other_processes
        ]
        return [
            SimpleNamespace(stdout="0, GPU-0\n1, GPU-1\n"),
            SimpleNamespace(stdout="\n".join(process_lines) + ("\n" if process_lines else "")),
        ]

    def test_namespace_mismatch_claim_and_assert_pass(self):
        with patch("run_mc_user_pilot_v1.os.getpid", return_value=123):
            claim = claim_gpu_process_ownership(
                0,
                run_command=Mock(
                    side_effect=self.gpu_responses([(45678, "python", 16000)])
                ),
            )
        self.assertEqual(claim["nvidia_host_pid"], 45678)
        assert_gpu_process_owned(
            0,
            claimed_gpu_uuid=claim["gpu_uuid"],
            claimed_host_pid=claim["nvidia_host_pid"],
            run_command=Mock(
                side_effect=self.gpu_responses([(45678, "python", 16000)])
            ),
        )

    def test_claim_requires_one_visible_process(self):
        with self.assertRaisesRegex(RuntimeError, "GPU_OWNER_NOT_VISIBLE"):
            claim_gpu_process_ownership(
                0, run_command=Mock(side_effect=self.gpu_responses())
            )
        with self.assertRaisesRegex(RuntimeError, "GPU_FOREIGN_PROCESS_AFTER_LOAD"):
            claim_gpu_process_ownership(
                0,
                run_command=Mock(
                    side_effect=self.gpu_responses(
                        [(45678, "python", 16000), (99999, "python", 100)]
                    )
                ),
            )

    def test_assert_rejects_second_process(self):
        with self.assertRaisesRegex(RuntimeError, "GPU_FOREIGN_PROCESS_AFTER_LOAD"):
            assert_gpu_process_owned(
                0,
                claimed_gpu_uuid="GPU-0",
                claimed_host_pid=45678,
                run_command=Mock(
                    side_effect=self.gpu_responses(
                        [(45678, "python", 16000), (99999, "python", 100)]
                    )
                ),
            )

    def test_assert_rejects_owner_disappearance(self):
        with self.assertRaisesRegex(RuntimeError, "GPU_OWNER_DISAPPEARED"):
            assert_gpu_process_owned(
                0,
                claimed_gpu_uuid="GPU-0",
                claimed_host_pid=45678,
                run_command=Mock(side_effect=self.gpu_responses()),
            )

    def test_assert_rejects_pid_replacement(self):
        with self.assertRaisesRegex(RuntimeError, "GPU_OWNERSHIP_CHANGED"):
            assert_gpu_process_owned(
                0,
                claimed_gpu_uuid="GPU-0",
                claimed_host_pid=45678,
                run_command=Mock(
                    side_effect=self.gpu_responses([(88888, "python", 16000)])
                ),
            )

    def test_other_gpu_process_is_ignored(self):
        assert_gpu_process_owned(
            0,
            claimed_gpu_uuid="GPU-0",
            claimed_host_pid=45678,
            run_command=Mock(
                side_effect=self.gpu_responses(
                    [(45678, "python", 16000)], [(99999, "python", 50000)]
                )
            ),
        )

    def test_initial_busy_gpu_blocks_preflight(self):
        args = SimpleNamespace(
            base_model=Path("base"),
            adapter=Path("adapter"),
            train_data=Path("train"),
            gpu_id=0,
            memory_threshold_mib=1024,
            run_id="run",
            output_root=Path("output"),
        )
        with unittest.mock.patch(
            "run_mc_user_pilot_v1.validate_paths", return_value={"train_sha256": "sha"}
        ), unittest.mock.patch(
            "run_mc_user_pilot_v1.read_jsonl", return_value=rows()
        ), unittest.mock.patch(
            "run_mc_user_pilot_v1.smoke_sample_ids", return_value=set()
        ), self.assertRaisesRegex(RuntimeError, "busy"):
            run_preflight(args, gpu_checker=Mock(side_effect=RuntimeError("GPU busy")))

    def test_source_has_no_forbidden_training_family_features(self):
        source = (SCRIPTS_DIR / "run_mc_user_pilot_v1.py").read_text(encoding="utf-8")
        forbidden = (
            "old_per_token_logps",
            "group_std",
            "group_mean",
            "penalty_mask",
            "DistributedDataParallel",
            "torchrun",
            "restore_lora(",
        )
        for term in forbidden:
            self.assertNotIn(term, source)

    def test_execute_claims_after_lora_validation_before_adamw(self):
        source = (SCRIPTS_DIR / "run_mc_user_pilot_v1.py").read_text(encoding="utf-8")
        execute = source.split("def execute_pilot(", 1)[1]
        positions = [
            execute.index("model = model_loader(args)"),
            execute.index("assert_only_lora_trainable(model)"),
            execute.index("gpu_owner_claim = claim_gpu_process_ownership"),
            execute.index("optimizer = torch.optim.AdamW"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('"gpu_owner_claim": gpu_owner_claim', execute)
        self.assertIn('"diagnostic_only": True', execute)
        self.assertNotIn("current_pid", source)


if __name__ == "__main__":
    unittest.main()
