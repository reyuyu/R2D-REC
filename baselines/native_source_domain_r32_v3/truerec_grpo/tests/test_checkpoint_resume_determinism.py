from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
sys.path.insert(0, str(ROOT / "diagnostics"))

from checkpoint_v1 import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_ORDER_SEED,
    CheckpointContractError,
    order_metadata,
    save_checkpoint_atomic,
)
from checkpoint_resume_determinism_audit import (  # noqa: E402
    DATASET_IDENTITY,
    build_components,
    run_equivalence,
    run_rejection_audit,
)


GROUP_IDS = [f"pilot-{index:04d}" for index in range(4096)]


class CheckpointResumeDeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.audit = run_equivalence(GROUP_IDS, cls.root)
        cls.rejections = run_rejection_audit(GROUP_IDS, cls.root / "checkpoint-step-3", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_uninterrupted_resume_exact_equivalence(self):
        for field in (
            "group_id_sequence_equal", "rollout_random_output_sequence_equal",
            "final_model_state_equal", "final_optimizer_state_equal",
            "final_driver_state_equal", "final_global_step_equal",
            "final_next_group_index_equal",
        ):
            self.assertEqual(self.audit[field], "PASS")

    def test_all_cpu_rng_restored(self):
        self.assertEqual(self.audit["python_rng_restored"], "PASS")
        self.assertEqual(self.audit["numpy_rng_restored"], "PASS")
        self.assertEqual(self.audit["torch_cpu_rng_restored"], "PASS")

    def test_adamw_internal_state_exact(self):
        self.assertEqual(self.audit["final_optimizer_state_equal"], "PASS")
        self.assertEqual(self.audit["optimizer_state_fields"], ["exp_avg", "exp_avg_sq", "step"])

    def test_cursor_no_duplicate_no_skip(self):
        self.assertEqual(self.audit["cursor_after_load"], {"epoch": 0, "next_group_index": 3})
        self.assertEqual(self.audit["no_duplicate_group_after_resume"], "PASS")
        self.assertEqual(self.audit["no_skipped_group_after_resume"], "PASS")

    def test_order_is_deterministic_and_complete(self):
        first, first_meta = order_metadata(GROUP_IDS, DEFAULT_ORDER_SEED)
        second, second_meta = order_metadata(list(reversed(GROUP_IDS)), DEFAULT_ORDER_SEED)
        self.assertEqual(first, second)
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(first_meta["count"], 4096)
        self.assertEqual(self.audit["order"], first_meta)

    def test_identity_and_corruption_rejections(self):
        for field in (
            "dataset_mismatch_rejected", "order_mismatch_rejected", "contract_mismatch_rejected",
            "missing_checkpoint_rejected", "corrupt_checkpoint_rejected",
            "partial_window_checkpoint_rejected", "failed_state_checkpoint_rejected",
        ):
            self.assertEqual(self.rejections[field], "PASS")

    def test_save_rejects_partial_accumulation_boundary(self):
        model, optimizer, driver = build_components([])
        driver.state.groups_in_accumulation_window = 1
        _, order = order_metadata(GROUP_IDS)
        with self.assertRaises(CheckpointContractError):
            save_checkpoint_atomic(
                self.root / "partial-save", model=model, optimizer=optimizer, driver=driver,
                dataset_identity=DATASET_IDENTITY, epoch=0, next_group_index=0, order=order,
            )

    def test_save_rejects_failed_driver(self):
        model, optimizer, driver = build_components([])
        driver.state.failed = True
        _, order = order_metadata(GROUP_IDS)
        with self.assertRaises(CheckpointContractError):
            save_checkpoint_atomic(
                self.root / "failed-save", model=model, optimizer=optimizer, driver=driver,
                dataset_identity=DATASET_IDENTITY, epoch=0, next_group_index=0, order=order,
            )

    def test_atomic_layout_and_schema(self):
        checkpoint = self.root / "checkpoint-step-3"
        self.assertEqual({path.name for path in checkpoint.iterdir()}, {"metadata.json", "state.pt"})
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        self.assertEqual(metadata["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertFalse(any(path.name.startswith(".checkpoint-step-3.tmp-") for path in self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
