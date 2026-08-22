import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


USER_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = USER_DIR / "scripts"
MONITOR_DIR = USER_DIR.parent / "scripts" / "monitor"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(MONITOR_DIR))

from run_mc_user_formal_v1 import (  # noqa: E402
    FORMAL_RESUME,
    FROZEN_CONFIG,
    MCFormalError,
    assert_run_target_writable,
    build_manifest,
    git_reproducibility_state,
    load_formal_config,
    public_preflight,
    run_cli,
    run_formal_prompt_loop,
    run_preflight,
    save_formal_checkpoint,
    select_formal_rows,
)
from server import is_mc_user_manifest, normalized_run_kind  # noqa: E402


CONFIG_PATH = USER_DIR / "configs" / "mc_user_formal_stage1_512.json"


def dataset_rows(count=300):
    return [
        {"sample_id": f"{route}-{index:04d}", "route": route}
        for route in ("action", "chain")
        for index in range(count)
    ]


def candidate(route):
    value = {
        "candidate_index": 0,
        "generated_token_count": 10,
        "format_valid": True,
        "reward": 0.5,
        "projection_required": False,
        "positive_unit_count": 1,
        "negative_unit_count": 1,
        "zero_unit_count": 0,
        "positive_credit_mass": 0.2,
        "negative_credit_mass": 0.1,
        "overlap_token_count": 0,
        "same_sign_overlap_token_count": 0,
        "mixed_sign_overlap_token_count": 0,
        "max_active_units_per_token": 1,
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


def prompt_result(route, sample_id, active=True):
    return {
        "route": route,
        "sample_id": sample_id,
        "candidates": [candidate(route), candidate(route)],
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


class FrozenConfigTests(unittest.TestCase):
    def test_stage1_config_is_exactly_frozen(self):
        config = load_formal_config(CONFIG_PATH)
        self.assertEqual(config, FROZEN_CONFIG)
        self.assertEqual(
            (config["prompt_count"], config["action_count"], config["chain_count"]),
            (512, 256, 256),
        )
        self.assertEqual(
            (
                config["K"],
                config["temperature"],
                config["top_p"],
                config["max_new_tokens"],
            ),
            (2, 0.9, 0.95, 512),
        )
        self.assertEqual(config["learning_rate"], 1e-6)
        self.assertEqual(config["weight_decay"], 0.0)
        self.assertEqual(config["forward_batch_size"], 1)
        self.assertEqual(config["checkpoint_steps"], [128, 256, 384, 512])
        self.assertFalse(config["resume_supported"])
        self.assertEqual(config["resume_policy"], "continuous_run_only")

    def test_config_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            changed = dict(FROZEN_CONFIG, K=4)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(MCFormalError, "frozen Stage1 config mismatch"):
                load_formal_config(path)


class SelectionTests(unittest.TestCase):
    def test_selection_is_unique_deterministic_and_strictly_alternating(self):
        rows = dataset_rows()
        first = select_formal_rows(rows, 20260823)
        second = select_formal_rows(list(reversed(rows)), 20260823)
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertEqual(len(first), 512)
        self.assertEqual(len({row["sample_id"] for row in first}), 512)
        self.assertEqual(sum(row["route"] == "action" for row in first), 256)
        self.assertEqual(sum(row["route"] == "chain" for row in first), 256)
        self.assertEqual(
            [row["route"] for row in first],
            [route for _ in range(256) for route in ("action", "chain")],
        )

    def test_different_seed_changes_order(self):
        rows = dataset_rows()
        first = select_formal_rows(rows, 20260823)
        second = select_formal_rows(rows, 20260824)
        self.assertNotEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )


class ManifestAndMonitorTests(unittest.TestCase):
    def test_manifest_is_complete_and_monitor_recognizes_mc_formal(self):
        rows = select_formal_rows(dataset_rows(), 20260823)
        manifest = build_manifest(
            run_id="formal-test",
            config_path=CONFIG_PATH,
            config_sha256="config-sha",
            config=FROZEN_CONFIG,
            rows=rows,
            git_commit="abc123",
        )
        self.assertEqual(manifest["run_kind"], "user_grpo")
        self.assertEqual(manifest["algorithm"], "mc_user_v1")
        self.assertEqual(manifest["experiment_type"], "formal")
        self.assertEqual(manifest["stage"], "stage1_512")
        self.assertEqual(len(manifest["prompts"]), 512)
        self.assertFalse(manifest["resume_supported"])
        self.assertEqual(manifest["FORMAL_RESUME"], "LIMITATION")
        self.assertTrue(is_mc_user_manifest(manifest))
        self.assertEqual(normalized_run_kind(manifest), "user_grpo")


class LoopAndCheckpointTests(unittest.TestCase):
    def test_loop_uses_one_persistent_optimizer_and_no_credit_accounting(self):
        rows = [
            {"route": route, "sample_id": f"sample-{index}"}
            for index, route in enumerate(("action", "chain", "action", "chain"), 1)
        ]
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer_ids = []
        checkpoints = []

        def process(row, actual_optimizer, step):
            optimizer_ids.append(id(actual_optimizer))
            return prompt_result(
                row["route"], row["sample_id"], active=step not in (2, 4)
            )

        records, optimizer_step = run_formal_prompt_loop(
            rows,
            optimizer,
            process,
            lambda prompt_step, *_args: checkpoints.append(prompt_step),
            lambda _record: None,
            (2, 4),
        )
        self.assertEqual(set(optimizer_ids), {id(optimizer)})
        self.assertEqual([record["optimizer_step"] for record in records], [1, 1, 2, 2])
        self.assertEqual(optimizer_step, 2)
        self.assertEqual(checkpoints, [2, 4])

    def test_checkpoint_is_adapter_only_and_records_formal_state(self):
        class AdapterModel:
            def save_pretrained(self, path, safe_serialization):
                self.safe_serialization = safe_serialization
                (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")
                (Path(path) / "adapter_model.safetensors").write_bytes(b"adapter")

        model = AdapterModel()
        rows = select_formal_rows(dataset_rows(), 20260823)
        records = [
            {"prompt_step": 1, "optimizer_step": 1, **prompt_result("action", "a")}
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            save_formal_checkpoint(
                model,
                run_dir,
                128,
                120,
                rows[:128],
                records,
                5.0,
                selection_seed=20260823,
                config_sha256="config-sha",
                train_sha256="train-sha",
                git_commit="commit",
            )
            checkpoint = run_dir / "checkpoints" / "prompt-step-0128"
            state = json.loads(
                (checkpoint / "formal_state.json").read_text(encoding="utf-8")
            )
            self.assertTrue(model.safe_serialization)
            self.assertTrue((checkpoint / "adapter_model.safetensors").is_file())
            self.assertFalse((checkpoint / "model.safetensors").exists())
            self.assertFalse((checkpoint / "optimizer.pt").exists())
            self.assertEqual(state["prompt_step"], 128)
            self.assertEqual(state["optimizer_step"], 120)
            self.assertEqual(len(state["processed_sample_ids"]), 128)
            self.assertFalse(state["resume_supported"])
            self.assertEqual(state["FORMAL_RESUME"], FORMAL_RESUME)


class SafetyGateTests(unittest.TestCase):
    def test_git_clean_and_dirty_contract(self):
        clean_runner = Mock(
            side_effect=[
                SimpleNamespace(stdout="abc123\n"),
                SimpleNamespace(stdout=""),
            ]
        )
        self.assertTrue(
            git_reproducibility_state(Path("repo"), run_command=clean_runner)[
                "working_tree_clean"
            ]
        )
        dirty_runner = Mock(
            side_effect=[
                SimpleNamespace(stdout="abc123\n"),
                SimpleNamespace(stdout=" M file.py\n"),
            ]
        )
        with self.assertRaisesRegex(MCFormalError, "BLOCKED_DIRTY_WORKTREE"):
            git_reproducibility_state(Path("repo"), run_command=dirty_runner)

    def test_preflight_does_not_load_model_and_writes_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                config=CONFIG_PATH,
                gpu_id=0,
                run_id="formal-test",
                memory_threshold_mib=1024,
                output_root=Path(directory),
            )
            model_loader = Mock()
            with patch(
                "run_mc_user_formal_v1.validate_paths",
                return_value={"train_sha256": FROZEN_CONFIG["train_sha256"]},
            ), patch(
                "run_mc_user_formal_v1.read_jsonl", return_value=dataset_rows()
            ):
                result = run_preflight(
                    args,
                    gpu_checker=Mock(return_value={"index": 0}),
                    git_checker=Mock(
                        return_value={
                            "git_commit": "abc123",
                            "working_tree_clean": True,
                        }
                    ),
                    model_loader=model_loader,
                )
            model_loader.assert_not_called()
            self.assertEqual(result["status"], "READY_TO_EXECUTE")
            self.assertIsInstance(result["run_dir"], str)
            self.assertEqual(Path(result["run_dir"]), Path(directory) / "formal-test")
            self.assertEqual(len(result["manifest"]["prompts"]), 512)
            self.assertTrue((Path(directory) / "formal-test" / "manifest.json").is_file())
            preflight_path = Path(directory) / "formal-test" / "preflight.json"
            self.assertTrue(preflight_path.is_file())
            stored = json.loads(preflight_path.read_text(encoding="utf-8"))
            self.assertIsInstance(stored["run_dir"], str)
            self.assertEqual(stored["run_dir"], result["run_dir"])

            public = public_preflight(result)
            self.assertIsInstance(public["run_dir"], str)
            json.dumps(public)
            for value in public.values():
                json.dumps(value)

    def test_real_preflight_contract_prints_json_and_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)

            def real_preflight(args):
                args.output_root = output_root
                with patch(
                    "run_mc_user_formal_v1.validate_paths",
                    return_value={"train_sha256": FROZEN_CONFIG["train_sha256"]},
                ), patch(
                    "run_mc_user_formal_v1.read_jsonl", return_value=dataset_rows()
                ):
                    return run_preflight(
                        args,
                        gpu_checker=Mock(return_value={"index": 0}),
                        git_checker=Mock(
                            return_value={
                                "git_commit": "abc123",
                                "working_tree_clean": True,
                            }
                        ),
                    )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                output = run_cli(
                    [
                        "--config",
                        str(CONFIG_PATH),
                        "--gpu-id",
                        "0",
                        "--run-id",
                        "formal-cli-test",
                    ],
                    preflight_fn=real_preflight,
                    execute_fn=Mock(),
                )

            rendered = stdout.getvalue()
            self.assertIn('"status": "READY_TO_EXECUTE"', rendered)
            self.assertTrue(rendered.rstrip().endswith("READY_TO_EXECUTE"))
            self.assertIsInstance(output["run_dir"], str)
            json.dumps(output)

    def test_without_execute_never_calls_execute_path(self):
        preflight = Mock(
            return_value={
                "status": "READY_TO_EXECUTE",
                "run_id": "run",
                "run_dir": "run",
                "gpu": {"index": 0},
                "git_commit": "commit",
                "working_tree_clean": True,
                "config_sha256": "config",
                "train_sha256": "train",
                "prompt_count": 512,
                "action_count": 256,
                "chain_count": 256,
                "resume_supported": False,
                "resume_policy": "continuous_run_only",
                "FORMAL_RESUME": "LIMITATION",
                "execute_required": True,
            }
        )
        execute = Mock()
        output = run_cli(
            [
                "--config",
                str(CONFIG_PATH),
                "--gpu-id",
                "0",
                "--run-id",
                "run",
            ],
            preflight_fn=preflight,
            execute_fn=execute,
        )
        self.assertEqual(output["status"], "READY_TO_EXECUTE")
        execute.assert_not_called()

    def test_execute_gate_calls_once(self):
        preflight = Mock(return_value={"status": "READY_TO_EXECUTE"})
        execute = Mock(return_value={"status": "PASS"})
        output = run_cli(
            [
                "--config",
                str(CONFIG_PATH),
                "--gpu-id",
                "0",
                "--run-id",
                "run",
                "--execute",
            ],
            preflight_fn=preflight,
            execute_fn=execute,
        )
        self.assertEqual(output["status"], "PASS")
        execute.assert_called_once()

    def test_dirty_preflight_blocks_execute(self):
        preflight = Mock(side_effect=MCFormalError("BLOCKED_DIRTY_WORKTREE"))
        execute = Mock()
        with self.assertRaisesRegex(MCFormalError, "BLOCKED_DIRTY_WORKTREE"):
            run_cli(
                [
                    "--config",
                    str(CONFIG_PATH),
                    "--gpu-id",
                    "0",
                    "--run-id",
                    "run",
                    "--execute",
                ],
                preflight_fn=preflight,
                execute_fn=execute,
            )
        execute.assert_not_called()

    def test_existing_artifacts_block_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(MCFormalError, "BLOCKED_EXISTING_FORMAL_RUN"):
                assert_run_target_writable(run_dir)

    def test_formal_source_reuses_guards_and_has_one_adamw(self):
        source = (SCRIPTS_DIR / "run_mc_user_formal_v1.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("torch.optim.AdamW("), 1)
        self.assertIn("claim_gpu_process_ownership", source)
        self.assertIn("assert_gpu_process_owned", source)
        self.assertIn("assert_only_lora_trainable", source)
        self.assertIn("validate_exact_policy_ids", source)
        self.assertIn('run_dir = Path(preflight["run_dir"])', source)
        self.assertNotIn("random.shuffle", source)
        for forbidden in (
            "old_per_token_logps",
            "group_mean",
            "group_std",
            "penalty_mask",
            "clip_fraction",
            "scheduler",
            "warmup",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
