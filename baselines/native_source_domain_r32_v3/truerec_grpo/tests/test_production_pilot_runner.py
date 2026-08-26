from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "trainer"))

from run_truerec_pilot_v1 import (  # noqa: E402
    ORDER_SHA256, PILOT_RECORDS_SHA256, ProductionTrainingError,
    append_jsonl, run_index_loop, validate_existing_prefix, write_frozen_manifest,
)


class ProductionPilotRunnerTest(unittest.TestCase):
    def test_exact_max_groups_boundary_and_no_group20(self):
        seen = []
        run_index_loop(
            [f"g{i}" for i in range(4096)], start_index=0, stop_exclusive=20,
            execute_group=lambda index, group_id: seen.append((index, group_id)) or {"group_index": index},
            checkpoint_every=10, save_checkpoint=lambda step: None, append_record=lambda row: None,
        )
        self.assertEqual([index for index, _ in seen], list(range(20)))
        self.assertNotIn(20, [index for index, _ in seen])

    def test_resume_cursor_starts_exactly_at_cursor(self):
        seen = []
        run_index_loop(
            [f"g{i}" for i in range(30)], start_index=10, stop_exclusive=20,
            execute_group=lambda index, group_id: seen.append(index) or index,
            checkpoint_every=10, save_checkpoint=lambda step: None, append_record=lambda row: None,
        )
        self.assertEqual(seen, list(range(10, 20)))

    def test_checkpoint_cadence(self):
        checkpoints = []
        run_index_loop(
            [f"g{i}" for i in range(30)], start_index=0, stop_exclusive=20,
            execute_group=lambda index, group_id: index, checkpoint_every=10,
            save_checkpoint=checkpoints.append, append_record=lambda row: None,
        )
        self.assertEqual(checkpoints, [10, 20])

    def test_jsonl_append_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            for index in range(3): append_jsonl(path, {"group_index": index})
            self.assertEqual([json.loads(line)["group_index"] for line in path.read_text().splitlines()], [0, 1, 2])

    def test_manifest_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"; value = {"frozen": 1}
            write_frozen_manifest(path, value, resume=False)
            write_frozen_manifest(path, value, resume=True)
            with self.assertRaises(ProductionTrainingError): write_frozen_manifest(path, {"frozen": 2}, resume=True)

    def test_existing_prefix_identity(self):
        order = ["a", "b", "c"]
        validate_existing_prefix([
            {"group_index": 0, "recommendation_group_id": "a"},
            {"group_index": 1, "recommendation_group_id": "b"},
        ], order, 2)
        with self.assertRaises(ProductionTrainingError):
            validate_existing_prefix([{"group_index": 0, "recommendation_group_id": "b"}], order, 1)

    def test_dataset_order_constants(self):
        self.assertEqual(PILOT_RECORDS_SHA256, "ed144262df3c852ba7fd63eba61c6cf03eb4dbed23be7d3d6c79b36a1a2ec879")
        self.assertEqual(ORDER_SHA256, "b4bf02f9b0591e1e684fa1e42e4528682848c615994e926aa33236d78337b4f6")

    def test_failure_stops_loop(self):
        seen = []
        def execute(index, group_id):
            seen.append(index)
            if index == 3: raise RuntimeError("stop")
            return index
        with self.assertRaises(RuntimeError):
            run_index_loop(
                [f"g{i}" for i in range(10)], start_index=0, stop_exclusive=10,
                execute_group=execute, checkpoint_every=5,
                save_checkpoint=lambda step: None, append_record=lambda row: None,
            )
        self.assertEqual(seen, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
