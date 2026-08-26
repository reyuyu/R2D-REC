"""CPU-only orchestration contracts for the formal four-rank Pilot4096 runner."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "analysis", ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from monitoring_v1 import MonitoringWriterV1  # noqa: E402
from run_truerec_formal4096_ddp_v1 import (  # noqa: E402
    CHECKPOINT_EVERY, PROBE_EVERY, RUN_ID, TOTAL_STEPS, Formal4096Error,
    checkpoint_steps, driver_state_for_cursor, execute_with_rng_restored,
    formal_group_indices, probe_steps, run, run_probe20, validate_fresh_launch,
    validate_monitoring_prefix,
)


class Formal4096DDPRunnerTests(unittest.TestCase):
    def test_exact_frozen_interval(self):
        indices = formal_group_indices(0)
        self.assertEqual((indices.start, indices.stop, len(indices)), (0, 4096, 4096))
        self.assertEqual((indices[0], indices[-1]), (0, 4095))

    def test_resume_cursor_has_no_duplicate_or_skip(self):
        indices = formal_group_indices(768)
        self.assertEqual((indices[0], indices[-1], len(indices)), (768, 4095, 4096 - 768))
        state = driver_state_for_cursor(768)
        self.assertEqual((state["global_step"], state["optimizer_steps"], state["groups_in_accumulation_window"]), (768, 768, 0))

    def test_checkpoint_schedule_is_every_256_through_4096(self):
        expected = tuple(range(256, 4097, 256))
        self.assertEqual(CHECKPOINT_EVERY, 256)
        self.assertEqual(checkpoint_steps(), expected)

    def test_probe_schedule_includes_zero_and_every_checkpoint(self):
        self.assertEqual(PROBE_EVERY, 256)
        self.assertEqual(probe_steps(), (0, *range(256, 4097, 256)))

    def test_probe_rng_restores_on_success_and_failure(self):
        state = {"value": 7}
        capture = lambda: dict(state)
        restore = lambda saved: state.update(saved) or {"cpu": True}
        fingerprint = lambda value: value["value"]
        result, exact = execute_with_rng_restored(
            capture_fn=capture, restore_fn=restore,
            seed_fn=lambda: state.update(value=99), fingerprint_fn=fingerprint,
            action=lambda: state.update(value=123) or "done",
        )
        self.assertEqual((result, exact, state["value"]), ("done", True, 7))
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            execute_with_rng_restored(
                capture_fn=capture, restore_fn=restore,
                seed_fn=lambda: state.update(value=99), fingerprint_fn=fingerprint,
                action=lambda: (_ for _ in ()).throw(RuntimeError("probe failure")),
            )
        self.assertEqual(state["value"], 7)

    def test_probe_has_no_optimizer_or_backward_entrypoint(self):
        signature = inspect.signature(run_probe20)
        source = inspect.getsource(run_probe20)
        self.assertNotIn("optimizer", signature.parameters)
        self.assertNotIn(".step(", source)
        self.assertNotIn(".backward(", source)
        self.assertIn("optimizer_steps_before", source)
        self.assertIn("optimizer_steps_after", source)

    def test_monitoring_prefix_detects_duplicate_skip_and_wrong_group(self):
        order = ["g0", "g1", "g2"]
        valid = [
            {"group_index": 0, "global_step": 1, "recommendation_group_id": "g0"},
            {"group_index": 1, "global_step": 2, "recommendation_group_id": "g1"},
        ]
        validate_monitoring_prefix(valid, order, 2)
        for broken in (
            [valid[0], {**valid[1], "group_index": 0}],
            [valid[0], {**valid[1], "global_step": 3}],
            [valid[0], {**valid[1], "recommendation_group_id": "g2"}],
        ):
            with self.assertRaises(Formal4096Error):
                validate_monitoring_prefix(broken, order, 2)

    def test_fresh_beta_requires_new_run_directory_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            fresh = parent / RUN_ID
            validate_fresh_launch(fresh, None)
            fresh.mkdir()
            with self.assertRaises(Formal4096Error):
                validate_fresh_launch(fresh, None)
        source = inspect.getsource(run)
        self.assertIn("load_runtime(device)", source)
        self.assertIn("if resume_checkpoint is not None", source)
        self.assertNotIn("phase1_3c", source.lower())

    def test_runner_uses_monitoring_writer_in_group_order(self):
        source = inspect.getsource(run)
        self.assertIn("for group_index in formal_group_indices(cursor)", source)
        self.assertIn("writer.record_train_group", source)
        self.assertIn("save_distributed_checkpoint", source)
        with tempfile.TemporaryDirectory() as raw:
            writer = MonitoringWriterV1(Path(raw), total_steps=TOTAL_STEPS, rank=0)
            base = {
                "recommendation_group_id": "g0", "target_domain": "video",
                "context_token_count": 10, "selected_microbatch_size": 2,
                "format_valid_rate": 1.0, "A_hit_rate": 0.0, "AB_hit_rate": 0.0,
                "exact_rate": 0.0, "wrong_history_copy_rate": 0.0,
                "frontier_value": 0.1, "hpr_value_raw": 1.0,
                "hpr_value_weighted": 0.02, "total_value": 0.12,
                "gradient_norm": 1.0, "wall_time_seconds": 2.0,
                "rank_memory": [
                    {"rank": rank, "allocated_gb": 1.0, "reserved_gb": 2.0,
                     "peak_allocated_gb": 3.0, "peak_reserved_gb": 4.0}
                    for rank in range(4)
                ],
            }
            explain = {"global_candidate_indices": list(range(8))}
            writer.record_train_group({**base, "global_step": 1, "group_index": 0}, explain, last_checkpoint_step=None)
            writer.record_train_group({**base, "recommendation_group_id": "g1", "global_step": 2, "group_index": 1}, explain, last_checkpoint_step=None)
            rows = [json.loads(line) for line in (Path(raw) / "train_groups.jsonl").read_text().splitlines()]
            self.assertEqual([row["group_index"] for row in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
