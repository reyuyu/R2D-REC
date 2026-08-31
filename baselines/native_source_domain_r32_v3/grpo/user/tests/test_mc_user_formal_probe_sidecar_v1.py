import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


USER_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = USER_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_mc_user_formal_probe_sidecar_v1 import (  # noqa: E402
    CHECKPOINT_STEPS,
    PROBE_SEED,
    PROBE_SHA256,
    FormalProbeError,
    checkpoint_complete,
    checkpoint_spec,
    incremental_result,
    load_formal_manifest,
    output_paths,
    run_preflight,
    sample_seed_map,
    validate_probe_disjoint,
    watch_schedule,
)


def probe_rows():
    rows = []
    for index in range(6):
        route = "action" if index < 3 else "chain"
        rows.append(
            {
                "sample_id": f"probe-{index}",
                "route": route,
                "prompt": f"prompt {index}",
                "prompt_token_count": index + 1,
            }
        )
    return rows


def formal_prompts():
    return [
        {
            "prompt_step": index + 1,
            "route": "action" if index % 2 == 0 else "chain",
            "sample_id": f"formal-{index}",
        }
        for index in range(512)
    ]


def candidate(route, reward):
    if route == "action":
        return {
            "completion": "[]",
            "reward": reward,
            "f1": reward,
            "precision": reward,
            "recall": reward,
            "exact_match": False,
        }
    return {
        "completion": "{}",
        "reward": reward,
        "total_reward": reward,
        "action_alignment": reward,
        "logic_alignment": reward,
    }


def checkpoint_result(step, reward):
    samples = []
    for row in probe_rows():
        samples.append(
            {
                "sample_id": row["sample_id"],
                "route": row["route"],
                "seed": PROBE_SEED + len(samples),
                "prompt": row["prompt"],
                "candidates": [candidate(row["route"], reward) for _ in range(4)],
            }
        )
    return {
        "step": step,
        "name": "BETA" if step == 0 else f"prompt-step-{step:04d}",
        "summary": {
            "action": {"f1": reward, "precision": reward, "recall": reward},
            "chain": {
                "total_reward": reward,
                "action_alignment": reward,
                "logic_alignment": reward,
            },
            "overall_user_proxy": reward * 2,
        },
        "samples": samples,
    }


class SelectionAndPreflightTests(unittest.TestCase):
    def test_seed_map_is_deterministic_and_paired(self):
        first = sample_seed_map(probe_rows())
        second = sample_seed_map(probe_rows())
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()), [PROBE_SEED + index for index in range(6)])
        self.assertEqual(PROBE_SHA256, "dc86fc30776b39a715292d9f185fe1c38a56de718ae8ba865f166e50ee6f2d61")

    def test_formal_manifest_and_zero_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(
                json.dumps({"prompts": formal_prompts()}), encoding="utf-8"
            )
            manifest = load_formal_manifest(path)
            validate_probe_disjoint(probe_rows(), manifest)
            changed = list(probe_rows())
            changed[0] = {**changed[0], "sample_id": "formal-0"}
            with self.assertRaisesRegex(FormalProbeError, "overlaps"):
                validate_probe_disjoint(changed, manifest)

    def test_real_preflight_contract_is_gpu1_and_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"prompts": formal_prompts()}), encoding="utf-8"
            )
            base = root / "base"
            beta = root / "beta"
            probe = root / "probe.jsonl"
            base.mkdir()
            beta.mkdir()
            probe.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                base_model=base,
                beta_adapter=beta,
                probe=probe,
                formal_run_dir=run_dir,
                gpu_id=1,
                memory_threshold_mib=1024,
            )
            with patch(
                "run_mc_user_formal_probe_sidecar_v1.validate_adapter_only"
            ), patch(
                "run_mc_user_formal_probe_sidecar_v1.load_probe",
                return_value=(probe_rows(), {"sha256": PROBE_SHA256, "selection_audit": {}}),
            ):
                result = run_preflight(
                    args, gpu_checker=Mock(return_value={"index": 1})
                )
            self.assertEqual(result["status"], "READY_TO_EXECUTE")
            self.assertEqual(result["gpu"]["index"], 1)
            self.assertEqual([item["step"] for item in result["checkpoints"]], list(CHECKPOINT_STEPS))

    def test_strong_parent_200_prompt_schedule_comes_from_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            prompts = formal_prompts()[:200]
            parent = root / "parent"
            base = root / "base"
            probe = root / "probe.jsonl"
            parent.mkdir()
            base.mkdir()
            probe.write_text("{}\n", encoding="utf-8")
            (run_dir / "manifest.json").write_text(json.dumps({
                "prompt_count": 200,
                "action_count": 100,
                "chain_count": 100,
                "checkpoint_steps": [25, 50, 75, 100, 150, 200],
                "probe_parent_label": "Parent",
                "probe_parent_adapter": str(parent),
                "prompts": prompts,
            }), encoding="utf-8")
            args = SimpleNamespace(
                base_model=base,
                parent_adapter=None,
                probe=probe,
                formal_run_dir=run_dir,
                gpu_id=1,
                memory_threshold_mib=1024,
            )
            with patch("run_mc_user_formal_probe_sidecar_v1.validate_adapter_only"), patch(
                "run_mc_user_formal_probe_sidecar_v1.load_probe",
                return_value=(probe_rows(), {"sha256": PROBE_SHA256, "selection_audit": {}}),
            ):
                result = run_preflight(args, gpu_checker=Mock(return_value={"index": 1}))
            self.assertEqual(result["checkpoint_steps"], [0, 25, 50, 75, 100, 150, 200])
            self.assertEqual(result["parent_label"], "Parent")
            self.assertEqual(result["checkpoints"][0]["name"], "Parent")
            self.assertEqual(Path(result["checkpoints"][0]["path"]), parent)


class WatcherTests(unittest.TestCase):
    def test_incomplete_checkpoint_waits_then_evaluates_once(self):
        specs = [{"step": 0}, {"step": 128}]
        polls = {0: 0, 128: 0}
        evaluated = []
        published = []

        def ready(spec):
            step = spec["step"]
            polls[step] += 1
            return step == 0 or polls[step] >= 3

        outputs = watch_schedule(
            specs,
            is_ready=ready,
            evaluate=lambda spec: evaluated.append(spec["step"]) or {"step": spec["step"]},
            publish=lambda rows: published.append([row["step"] for row in rows]),
            sleep=lambda _seconds: None,
            poll_seconds=0,
        )
        self.assertEqual(evaluated, [0, 128])
        self.assertEqual(published, [[0], [0, 128]])
        self.assertEqual([row["step"] for row in outputs], [0, 128])

    def test_checkpoint_complete_requires_formal_state_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = checkpoint_spec(root, root / "beta", 128)
            path = Path(spec["path"])
            path.mkdir(parents=True)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
            self.assertFalse(checkpoint_complete(spec))
            (path / "formal_state.json").write_text(
                json.dumps({"prompt_step": 128}), encoding="utf-8"
            )
            self.assertTrue(checkpoint_complete(spec))

    def test_incremental_results_are_beta_first_paired_and_keep_completions(self):
        beta = checkpoint_result(0, 0.25)
        step128 = checkpoint_result(128, 0.5)
        result = incremental_result(
            [step128, beta],
            probe_sha256=PROBE_SHA256,
            sample_ids=[row["sample_id"] for row in probe_rows()],
            sample_seeds=sample_seed_map(probe_rows()),
        )
        self.assertEqual(result["evaluated_steps"], [0, 128])
        self.assertEqual(result["waiting_steps"], [256, 384, 512])
        self.assertEqual(result["status"], "WAITING_FOR_CHECKPOINTS")
        self.assertAlmostEqual(result["paired_deltas"][0]["delta_action_f1"], 0.25)
        sample = result["checkpoints"][1]["samples"][0]
        self.assertEqual(len(sample["candidates"]), 4)
        self.assertEqual(sample["candidates"][0]["completion"], "[]")
        self.assertAlmostEqual(sample["relative_to_beta_delta"], 0.25)


class IsolationAndRegressionTests(unittest.TestCase):
    def test_sidecar_has_no_training_import_or_primitive(self):
        path = SCRIPTS / "run_mc_user_formal_probe_sidecar_v1.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any("trainer" in name or "objective" in name for name in imports))
        for forbidden in ("optimizer.step", ".backward(", "mc_optimizer_step"):
            self.assertNotIn(forbidden, source)
        self.assertIn("@torch.inference_mode()", source)
        self.assertIn('CUDA_VISIBLE_DEVICES") != "1"', source)

    def test_sidecar_output_is_evaluations_only(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            results, status = output_paths(run_dir)
            expected = run_dir / "evaluations" / "user_light_probe"
            self.assertEqual(results.parent, expected)
            self.assertEqual(status.parent, expected)

    def test_formal_rollout_and_training_math_source_are_unchanged(self):
        source = (SCRIPTS / "run_mc_user_formal_v1.py").read_text(encoding="utf-8")
        self.assertIn('rollouts_path = run_dir / "rollouts.jsonl"', source)
        self.assertIn("display_rollout_records(", source)
        self.assertIn("_append_jsonl(rollouts_path, display_record)", source)
        self.assertEqual(source.count("torch.optim.AdamW("), 1)
        self.assertNotIn("formal_probe_sidecar", source)


if __name__ == "__main__":
    unittest.main()
