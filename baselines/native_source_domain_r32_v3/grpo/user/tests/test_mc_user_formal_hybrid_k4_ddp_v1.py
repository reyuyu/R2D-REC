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
    append_rank0_prompt_artifacts,
    build_manifest,
    load_k4_config,
    validate_checkpoint_root,
)
from run_mc_user_formal_probe_sidecar_v1 import checkpoint_spec  # noqa: E402


CONFIG = USER_DIR / "configs" / "mc_user_formal_stage1_512_hybrid_k4.json"


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
