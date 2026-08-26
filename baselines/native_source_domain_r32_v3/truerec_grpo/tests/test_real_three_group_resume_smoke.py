from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
sys.path.insert(0, str(ROOT / "diagnostics"))

from checkpoint_v1 import CHECKPOINT_SCHEMA_VERSION, load_checkpoint, order_metadata, save_checkpoint_atomic  # noqa: E402
from checkpoint_resume_determinism_audit import DATASET_IDENTITY, build_components  # noqa: E402
from real_three_group_resume_smoke import ORDER_SHA256, load_pilot_order  # noqa: E402


GROUP_IDS = [f"pilot-{index:04d}" for index in range(4096)]


class RealThreeGroupResumeSmokeContractTest(unittest.TestCase):
    def test_cuda_rng_roundtrip_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / "checkpoint"
            model, optimizer, driver = build_components([])
            _, order = order_metadata(GROUP_IDS)
            saved_cuda_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
            with patch("checkpoint_v1.torch.cuda.is_available", return_value=True), patch(
                "checkpoint_v1.torch.cuda.get_rng_state", return_value=saved_cuda_state
            ):
                save_checkpoint_atomic(
                    checkpoint, model=model, optimizer=optimizer, driver=driver,
                    dataset_identity=DATASET_IDENTITY, epoch=0, next_group_index=0,
                    order=order, cuda_device="cuda:0",
                )
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            self.assertTrue(metadata["cuda_rng_tested"])
            self.assertEqual(metadata["cuda_device_count_saved"], 1)
            fresh_model, fresh_optimizer, fresh_driver = build_components([])
            with patch("checkpoint_v1.torch.cuda.is_available", return_value=True), patch(
                "checkpoint_v1.torch.cuda.set_rng_state"
            ) as restore:
                load_checkpoint(
                    checkpoint, model=fresh_model, optimizer=fresh_optimizer, driver=fresh_driver,
                    current_dataset_identity=DATASET_IDENTITY, current_group_ids=GROUP_IDS,
                    cuda_device="cuda:0",
                )
            self.assertTrue(torch.equal(restore.call_args.args[0], saved_cuda_state))

    def test_cpu_checkpoint_path_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            model, optimizer, driver = build_components([]); _, order = order_metadata(GROUP_IDS)
            save_checkpoint_atomic(
                checkpoint, model=model, optimizer=optimizer, driver=driver,
                dataset_identity=DATASET_IDENTITY, epoch=0, next_group_index=0, order=order,
            )
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            self.assertFalse(metadata["cuda_rng_tested"])
            self.assertEqual(metadata["cuda_device_count_saved"], 0)

    def test_schema_bumped_for_cuda_contract(self):
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 2)

    def test_runner_has_exact_process_slices_and_no_fourth_group(self):
        source = (ROOT / "diagnostics" / "real_three_group_resume_smoke.py").read_text()
        self.assertIn("order, 0, 2, device", source)
        self.assertIn("order, 2, 3, device", source)
        self.assertNotIn("order, 3, 4, device", source)
        self.assertNotIn("torch.distributed", source)
        self.assertNotIn("scheduler.step", source.lower())

    def test_frozen_order_sha_constant(self):
        self.assertEqual(ORDER_SHA256, "b4bf02f9b0591e1e684fa1e42e4528682848c615994e926aa33236d78337b4f6")


if __name__ == "__main__":
    unittest.main()
