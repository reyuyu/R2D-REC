import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PIPELINE = Path(__file__).resolve().parents[1]
RUN_SH = PIPELINE / "run.sh"
CONTROL = PIPELINE / "scripts" / "pipeline_control.py"


class PipelineTests(unittest.TestCase):
    def make_registry(self, root: Path) -> None:
        datasets = {}
        for key, rows in (
            ("recommendation_grpo_bilateral", 2),
            ("recommendation_grpo_think_only", 1),
            ("user_grpo", 2),
        ):
            path = root / f"{key}.jsonl"
            path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(rows)))
            import hashlib

            datasets[key] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": rows,
                "split": "train",
            }
        (root / "registry.json").write_text(
            json.dumps({"schema_version": 1, "datasets": datasets}), encoding="utf-8"
        )

    def dry_run(self, *, grpo2: bool) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_registry(root)
            command = ["bash", str(RUN_SH)]
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "DRY_RUN": "1",
                "REPRO_DATA_ROOT": str(root),
                "RUN_GRPO2": "1" if grpo2 else "0",
            }
            return subprocess.run(command, env=env, check=True, text=True, capture_output=True).stdout

    def test_default_dry_run_skips_grpo2_and_uses_grpo1_parent(self):
        output = self.dry_run(grpo2=False)
        self.assertIn("PIPELINE=SFT -> GRPO1 -> GRPO3", output)
        self.assertIn("GRPO1_STEPS=300", output)
        self.assertIn("GRPO2_STATUS=SKIPPED", output)
        self.assertIn("GRPO2_DATASET_RESOLUTION=SKIPPED", output)
        self.assertIn("GRPO3_PARENT=GRPO1_FINAL_ADAPTER", output)

    def test_optional_grpo2_dry_run_switches_grpo3_parent(self):
        output = self.dry_run(grpo2=True)
        self.assertIn("PIPELINE=SFT -> GRPO1 -> GRPO2 -> GRPO3", output)
        self.assertIn("GRPO2_STATUS=ENABLED", output)
        self.assertIn("GRPO2_STEPS=250", output)
        self.assertIn("GRPO3_PARENT=GRPO2_FINAL_ADAPTER", output)


if __name__ == "__main__":
    unittest.main()
