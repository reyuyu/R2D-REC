import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path


PIPELINE = Path(__file__).resolve().parents[1]
REPOSITORY = PIPELINE.parent
RUN_SH = PIPELINE / "run.sh"
CONTROL = PIPELINE / "scripts" / "pipeline_control.py"
RESOLVER = PIPELINE / "scripts" / "resolve_repro_dataset.py"


class PipelineTests(unittest.TestCase):
    def make_registry(self, root: Path) -> Path:
        datasets = {}
        for key, rows in (
            ("recommendation_grpo_bilateral", 2),
            ("recommendation_grpo_think_only", 1),
            ("user_grpo", 2),
        ):
            path = root / f"{key}.jsonl"
            path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(rows)))
            datasets[key] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": rows,
                "split": "train",
            }
        raw = root / "onereason_final_chain_20260901" / "00官方数据集"
        (raw / "group_a/0.0.0").mkdir(parents=True)
        raw_file = raw / "group_a/0.0.0/rank0-0.parquet"
        raw_file.write_bytes(b"test raw parquet contract")
        raw_manifest = root / "RAW_800K_MANIFEST.json"
        raw_manifest.write_text(json.dumps({
            "schema_version": 1,
            "files": {
                "group_a/0.0.0/rank0-0.parquet": {
                    "rows": 7,
                    "bytes": raw_file.stat().st_size,
                    "sha256": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
                }
            },
            "file_count": 1,
            "total_rows": 7,
        }), encoding="utf-8")
        datasets["rec_fdr_v43_full_sft_raw_800k"] = {
            "kind": "raw_parquet_directory",
            "path": raw.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
            "files": 1,
            "rows": 7,
            "split": "raw",
        }
        (root / "registry.json").write_text(
            json.dumps({"schema_version": 1, "datasets": datasets}), encoding="utf-8"
        )
        return raw_manifest

    def dry_run(self, *, grpo2: bool) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_manifest = self.make_registry(root)
            command = ["bash", str(RUN_SH)]
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "DRY_RUN": "1",
                "REPRO_DATA_ROOT": str(root),
                "RUN_GRPO2": "1" if grpo2 else "0",
                "REPRO_TEST_SFT_MANIFEST": str(raw_manifest),
            }
            return subprocess.run(command, env=env, check=True, text=True, capture_output=True).stdout

    def test_default_dry_run_skips_grpo2_and_uses_grpo1_parent(self):
        output = self.dry_run(grpo2=False)
        self.assertIn("PIPELINE=SFT -> GRPO1 -> GRPO3", output)
        self.assertIn("GRPO1_STEPS=300", output)
        self.assertIn("SFT_REGISTERED_DATASET=rec_fdr_v43_full_sft_raw_800k", output)
        self.assertIn("SFT_RAW_TO_TRAINING_DATA=AUTOMATIC_TMPFS", output)
        self.assertIn("GRPO2_STATUS=SKIPPED", output)
        self.assertIn("GRPO2_DATASET_RESOLUTION=SKIPPED", output)
        self.assertIn("GRPO3_PARENT=GRPO1_FINAL_ADAPTER", output)

    def test_optional_grpo2_dry_run_switches_grpo3_parent(self):
        output = self.dry_run(grpo2=True)
        self.assertIn("PIPELINE=SFT -> GRPO1 -> GRPO2 -> GRPO3", output)
        self.assertIn("GRPO2_STATUS=ENABLED", output)
        self.assertIn("GRPO2_STEPS=250", output)
        self.assertIn("GRPO3_PARENT=GRPO2_FINAL_ADAPTER", output)

    def test_raw_registry_rejects_any_non_raw_extra_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_registry(root)
            (root / "onereason_final_chain_20260901/00官方数据集/README.md").write_text(
                "not raw"
            )
            result = subprocess.run(
                [
                    "python3", str(RESOLVER), "--root", str(root),
                    "--key", "rec_fdr_v43_full_sft_raw_800k",
                    "--manifest", str(manifest),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("file set mismatch", result.stderr)

    def test_sft_pipeline_has_no_finished_dataset_path_dependency(self):
        run_source = RUN_SH.read_text(encoding="utf-8")
        rebuild_source = (
            REPOSITORY
            / "reproduction/rec_fdr_v43_strictdet/scripts/rebuild_rec_fdr_v43_from_raw_800k.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("SFT_DATA_ROOT", run_source)
        self.assertNotIn('package_root / "data', rebuild_source)
        self.assertIn("launch_rec_fdr_v43_from_registered_raw.sh", run_source)

    def test_sft_launcher_runs_unittest_by_importable_module_name(self):
        launcher = (
            REPOSITORY
            / "reproduction/rec_fdr_v43_strictdet/scripts/launch_rec_fdr_v43_reproduction.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m unittest -v test_rec_fdr_v43_hcr", launcher)
        self.assertNotIn('unittest -v "$package_root/training/', launcher)


if __name__ == "__main__":
    unittest.main()
