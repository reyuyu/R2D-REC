import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_mc_user_light_probe import (  # noqa: E402
    CHECKPOINT_STEPS,
    EVAL_CANDIDATE_COUNT,
    PROBE_SEED,
    checkpoint_specs,
    evaluate_checkpoint_sequence,
    generation_config,
    load_light_probe,
    paired_deltas,
    run_cli,
    run_preflight,
    sample_seed_map,
    summarize_checkpoint,
    validate_adapter_only,
    validate_probe_disjoint,
)


def probe_source_rows():
    action = [
        {
            "sample_id": f"action-{index:02d}",
            "route": "action",
            "gold_sid_count": index + 1,
            "gold_event_count": 0,
            "prompt_token_count": 100 + index * 10,
        }
        for index in range(12)
    ]
    chain = [
        {
            "sample_id": f"chain-{event_count}-{index}",
            "route": "chain",
            "gold_sid_count": 0,
            "gold_event_count": event_count,
            "prompt_token_count": 500 + event_count * 100 + index,
        }
        for event_count in (2, 3, 4, 5)
        for index in range(2)
    ]
    return action + chain


def candidate(route, value):
    if route == "action":
        return {
            "reward": value,
            "f1": value,
            "precision": value / 2,
            "recall": value / 4,
            "exact_match": value == 1.0,
        }
    return {
        "reward": value,
        "total_reward": value,
        "action_alignment": value / 2,
        "logic_alignment": value / 4,
    }


def checkpoint_result(step, action_value, chain_value):
    samples = []
    for route, value in (("action", action_value), ("chain", chain_value)):
        for index in range(3):
            samples.append({
                "sample_id": f"{route}-{index}",
                "route": route,
                "candidates": [candidate(route, value) for _ in range(4)],
            })
    summary = summarize_checkpoint(step, samples)
    return {"step": step, "summary": summary, "samples": samples}


def make_adapter(path):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


class ProbeSelectionTests(unittest.TestCase):
    def test_sha_selection_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in probe_source_rows()),
                encoding="utf-8",
            )
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            first, _ = load_light_probe(path, sha)
            second, _ = load_light_probe(path, sha)
            self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
            self.assertEqual(sum(row["route"] == "action" for row in first), 3)
            self.assertEqual(sum(row["route"] == "chain" for row in first), 3)
            with self.assertRaisesRegex(RuntimeError, "SHA"):
                load_light_probe(path, "0" * 64)

    def test_probe_and_pilot_must_be_disjoint(self):
        rows = [{"sample_id": f"probe-{index}"} for index in range(6)]
        manifest = {"prompts": [{"sample_id": f"pilot-{index}"} for index in range(32)]}
        validate_probe_disjoint(rows, manifest)
        manifest["prompts"][0]["sample_id"] = "probe-0"
        with self.assertRaisesRegex(RuntimeError, "overlaps"):
            validate_probe_disjoint(rows, manifest)


class CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_order_and_adapter_only_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            beta = root / "beta"
            make_adapter(beta)
            specs = checkpoint_specs(root / "pilot", beta)
            self.assertEqual([item["step"] for item in specs], [0, 8, 16, 32])
            validate_adapter_only(beta)
            (beta / "model.safetensors").write_bytes(b"base")
            with self.assertRaisesRegex(RuntimeError, "base-model"):
                validate_adapter_only(beta)

    def test_adapter_unload_reload_sequence(self):
        base = object()
        unload_bases = []
        loaded_paths = []

        class Adapter:
            def __init__(self, parent):
                self.parent = parent

            def unload(self):
                unload_bases.append(self.parent)
                return self.parent

        specs = [{"step": step, "path": Path(str(step))} for step in CHECKPOINT_STEPS]
        final_base, outputs = evaluate_checkpoint_sequence(
            base,
            specs,
            lambda current, path: loaded_paths.append(path) or Adapter(current),
            lambda _model, spec: {"step": spec["step"]},
            Mock(),
        )
        self.assertIs(final_base, base)
        self.assertEqual([str(path) for path in loaded_paths], ["0", "8", "16", "32"])
        self.assertEqual(unload_bases, [base] * 4)
        self.assertEqual([item["step"] for item in outputs], list(CHECKPOINT_STEPS))


class SeedAndAggregationTests(unittest.TestCase):
    def test_same_sample_seeds_and_four_eval_candidates(self):
        rows = [{"sample_id": f"sample-{index}"} for index in range(6)]
        first = sample_seed_map(rows)
        second = sample_seed_map(rows)
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()), [PROBE_SEED + index for index in range(6)])
        self.assertEqual(generation_config()["candidate_count"], 4)
        self.assertEqual(EVAL_CANDIDATE_COUNT, 4)

    def test_action_chain_and_proxy_aggregation(self):
        result = checkpoint_result(0, 0.8, 0.6)["summary"]
        self.assertAlmostEqual(result["action"]["f1"], 0.8)
        self.assertAlmostEqual(result["action"]["precision"], 0.4)
        self.assertAlmostEqual(result["action"]["recall"], 0.2)
        self.assertEqual(result["action"]["exact_match_rate"], 0.0)
        self.assertAlmostEqual(result["chain"]["total_reward"], 0.6)
        self.assertAlmostEqual(result["chain"]["action_alignment"], 0.3)
        self.assertAlmostEqual(result["chain"]["logic_alignment"], 0.15)
        self.assertAlmostEqual(result["overall_user_proxy"], 1.4)

    def test_paired_delta_and_sample_level_results(self):
        checkpoints = [
            checkpoint_result(0, 0.5, 0.4),
            checkpoint_result(8, 0.6, 0.45),
            checkpoint_result(16, 0.7, 0.5),
            checkpoint_result(32, 0.8, 0.55),
        ]
        deltas = paired_deltas(checkpoints)
        self.assertEqual([item["step"] for item in deltas], [8, 16, 32])
        self.assertAlmostEqual(deltas[0]["delta_action_f1"], 0.1)
        self.assertAlmostEqual(deltas[0]["delta_chain_total"], 0.05)
        self.assertAlmostEqual(deltas[0]["delta_overall_user_proxy"], 0.15)
        self.assertEqual(len(deltas[0]["samples"]), 6)
        self.assertAlmostEqual(deltas[0]["samples"][0]["paired_delta"], 0.1)


class GateTests(unittest.TestCase):
    def test_execute_gate_and_waiting_status(self):
        ready = {
            "status": "READY_TO_EXECUTE",
            "gpu": {"index": 0},
            "probe_sha256": "sha",
            "sample_ids": [str(index) for index in range(6)],
            "checkpoints": [{"step": step} for step in CHECKPOINT_STEPS],
            "missing_pilot_checkpoint_steps": [],
            "generation_config": generation_config(),
        }
        execute = Mock(return_value={"status": "PASS"})
        output = run_cli(
            ["--pilot-run-dir", "run", "--gpu-id", "0"],
            preflight_fn=Mock(return_value=ready),
            execute_fn=execute,
        )
        self.assertEqual(output["status"], "READY_TO_EXECUTE")
        execute.assert_not_called()
        waiting = dict(ready, status="WAITING_FOR_PILOT_CHECKPOINTS", missing_pilot_checkpoint_steps=[8])
        output = run_cli(
            ["--pilot-run-dir", "run", "--gpu-id", "0", "--execute"],
            preflight_fn=Mock(return_value=waiting),
            execute_fn=execute,
        )
        self.assertEqual(output["status"], "WAITING_FOR_PILOT_CHECKPOINTS")
        execute.assert_not_called()

    def test_busy_gpu_blocks_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            beta = root / "beta"
            pilot = root / "pilot"
            probe = root / "probe.jsonl"
            base.mkdir()
            make_adapter(beta)
            pilot.mkdir()
            (pilot / "manifest.json").write_text(
                json.dumps({"prompts": [{"sample_id": f"pilot-{index}"} for index in range(32)]}),
                encoding="utf-8",
            )
            probe.write_text("", encoding="utf-8")
            args = SimpleNamespace(
                base_model=base,
                beta_adapter=beta,
                probe=probe,
                pilot_run_dir=pilot,
                gpu_id=0,
                memory_threshold_mib=1024,
            )
            with patch(
                "eval_mc_user_light_probe.load_light_probe",
                return_value=([{"sample_id": f"probe-{index}", "route": "action" if index < 3 else "chain"} for index in range(6)], {"sha256": "sha", "selection_audit": {}}),
            ), self.assertRaisesRegex(RuntimeError, "busy"):
                run_preflight(args, gpu_checker=Mock(side_effect=RuntimeError("GPU busy")))

    def test_source_is_independent_of_local_constraint_compilation(self):
        source = (SCRIPTS / "eval_mc_user_light_probe.py").read_text(encoding="utf-8")
        forbidden = (
            "compile_penalty_mask",
            "user_penalty_mask",
            "penalty_mask",
            "old_per_token_logps",
            ".backward(",
            "torch.optim",
        )
        for term in forbidden:
            self.assertNotIn(term, source)


if __name__ == "__main__":
    unittest.main()
