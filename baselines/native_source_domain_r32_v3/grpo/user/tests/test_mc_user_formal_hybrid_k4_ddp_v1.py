import json
import sys
import tempfile
import unittest
from pathlib import Path


USER_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = USER_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_formal_hybrid_k4_ddp_v1 import (  # noqa: E402
    ALGORITHM,
    FROZEN_CONFIG,
    GRPO3_DETERMINISM_CONFIG,
    STRONG_PARENT_CONFIG,
    append_rank0_prompt_artifacts,
    build_manifest,
    load_k4_config,
    probe_queue_value,
    validate_parent_adapter_contract,
    validate_checkpoint_root,
)
from compare_grpo3_user_determinism import compare  # noqa: E402
from run_mc_user_formal_probe_sidecar_v1 import checkpoint_spec  # noqa: E402


CONFIG = USER_DIR / "configs" / "mc_user_formal_stage1_512_hybrid_k4.json"
STRONG_CONFIG = USER_DIR / "configs" / "mc_user_hybrid_strongparent_lr3e7_200.json"
GRPO3_CONFIG = USER_DIR / "configs" / "grpo3_user_from_grpo2_step300_determinism_v1.json"


class HybridFormalRunnerTests(unittest.TestCase):
    def test_frozen_config_and_parent(self):
        self.assertEqual(load_k4_config(CONFIG), FROZEN_CONFIG)
        self.assertEqual(FROZEN_CONFIG["K"], 4)
        self.assertEqual(FROZEN_CONFIG["world_size"], 4)
        self.assertEqual(FROZEN_CONFIG["sequence_weight"], 1.0)
        self.assertEqual(FROZEN_CONFIG["local_weight"], 0.3)
        self.assertEqual(FROZEN_CONFIG["parent_checkpoint_step"], 1500)
        self.assertTrue(FROZEN_CONFIG["adapter"].endswith("/checkpoint-1500"))

    def test_manifest_algorithm_and_candidate_parallel_contract(self):
        rows = [
            {"sample_id": f"sample-{index}", "route": "action" if index % 2 else "chain"}
            for index in range(1, 513)
        ]
        manifest = build_manifest("run", CONFIG, "a" * 64, FROZEN_CONFIG, rows, "b" * 40)
        self.assertEqual(manifest["algorithm"], ALGORITHM)
        self.assertEqual((manifest["K"], manifest["world_size"]), (4, 4))
        self.assertEqual(manifest["parallelism"], "candidate_parallel")
        self.assertEqual(manifest["prompt_count"], 512)

    def test_strong_parent_frozen_config_and_first200_contract(self):
        config = load_k4_config(STRONG_CONFIG)
        self.assertEqual(config, STRONG_PARENT_CONFIG)
        self.assertEqual(config["learning_rate"], 3e-7)
        self.assertEqual(config["checkpoint_steps"], [25, 50, 75, 100, 150, 200])
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertTrue(config["adapter"].endswith("checkpoint-250"))
        rows = [
            {"sample_id": f"sample-{index}", "route": "action" if index % 2 else "chain"}
            for index in range(1, 513)
        ]
        original = build_manifest("old", CONFIG, "a" * 64, FROZEN_CONFIG, rows, "b" * 40)
        strong = build_manifest("new", STRONG_CONFIG, "c" * 64, config, rows, "d" * 40)
        self.assertEqual(strong["prompts"], original["prompts"][:200])
        self.assertEqual((strong["prompt_count"], strong["action_count"], strong["chain_count"]), (200, 100, 100))
        self.assertEqual([item["step"] for item in probe_queue_value(config)["items"]], [0, 25, 50, 75, 100, 150, 200])
        self.assertEqual(probe_queue_value(config)["items"][0]["label"], "Parent")

    def test_grpo3_config_and_five_prompt_smoke_contract(self):
        config = load_k4_config(GRPO3_CONFIG)
        self.assertEqual(config, GRPO3_DETERMINISM_CONFIG)
        self.assertEqual(config["learning_rate"], 3e-7)
        self.assertEqual(config["registered_dataset_name"], "user_grpo")
        rows = [
            {"sample_id": f"sample-{index}", "route": "action" if index % 2 else "chain"}
            for index in range(1, 101)
        ]
        manifest = build_manifest(
            "grpo3-smoke", GRPO3_CONFIG, "a" * 64, config, rows, "b" * 40,
            smoke_prompts=5,
        )
        self.assertEqual(manifest["prompt_count"], 5)
        self.assertEqual((manifest["action_count"], manifest["chain_count"]), (3, 2))
        self.assertEqual(manifest["checkpoint_steps"], [])

    def test_grpo3_comparator_requires_exact_step_evidence_and_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for name in ("a", "b"):
                run = root / name
                (run / "final_adapter").mkdir(parents=True)
                (run / "summary.json").write_text(json.dumps({
                    "status": "PASS",
                    "optimizer_step": 5,
                    "registered_dataset_used_by_trainer": True,
                    "training_semantics_changed": False,
                }), encoding="utf-8")
                rows = [{"prompt_step": step, "fingerprint": f"step-{step}"} for step in range(1, 6)]
                (run / "determinism_evidence.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                (run / "final_adapter" / "adapter_model.safetensors").write_bytes(b"same")
                runs.append(run)
            result = compare(*runs)
            self.assertEqual(result["DETERMINISM_LEVEL"], "BYTE_EXACT")
            self.assertEqual(result["FIRST_DIVERGENCE"], "NONE")
            self.assertEqual(result["READY_FOR_GRPO3_FORMAL_TRAINING"], "YES")
            rows = [json.loads(line) for line in (runs[1] / "determinism_evidence.jsonl").read_text().splitlines()]
            rows[2]["fingerprint"] = "different"
            (runs[1] / "determinism_evidence.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = compare(*runs)
            self.assertEqual(result["FIRST_DIVERGENCE"]["step"], 3)
            self.assertEqual(result["READY_FOR_GRPO3_FORMAL_TRAINING"], "NO")

    def test_parent_adapter_contract_requires_exact_504_lora_tensors(self):
        from safetensors.torch import save_file
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensors = {f"layer.{index}.lora_A.weight": torch.zeros(1) for index in range(504)}
            save_file(tensors, root / "adapter_model.safetensors")
            result = validate_parent_adapter_contract(root)
            self.assertEqual((result["trainable_lora_tensor_count"], result["lora_tensor_count"]), (504, 504))
            tensors.pop("layer.503.lora_A.weight")
            save_file(tensors, root / "adapter_model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "BLOCKED_PARENT_CONTRACT"):
                validate_parent_adapter_contract(root)

    def test_checkpoint_root_rejects_data_and_accepts_external(self):
        with self.assertRaisesRegex(RuntimeError, "must not be under /data"):
            validate_checkpoint_root(Path("/data/checkpoints"))
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(validate_checkpoint_root(Path(temporary), minimum_free_bytes=0), Path(temporary).resolve())

    def test_rank0_only_artifact_logging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, rollouts = root / "metrics.jsonl", root / "rollouts.jsonl"
            append_rank0_prompt_artifacts(1, metrics, rollouts, {"step": 1}, [{"candidate": 0}])
            self.assertFalse(metrics.exists())
            append_rank0_prompt_artifacts(0, metrics, rollouts, {"step": 1}, [{"candidate": index} for index in range(4)])
            self.assertEqual(len(metrics.read_text().splitlines()), 1)
            self.assertEqual(len(rollouts.read_text().splitlines()), 4)

    def test_probe_resolves_manifest_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            external = root / "external"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({"checkpoint_root": str(external)}), encoding="utf-8")
            spec = checkpoint_spec(run, root / "beta", 128)
            self.assertEqual(Path(spec["path"]), external / "run" / "checkpoints" / "prompt-step-0128")


if __name__ == "__main__":
    unittest.main()
