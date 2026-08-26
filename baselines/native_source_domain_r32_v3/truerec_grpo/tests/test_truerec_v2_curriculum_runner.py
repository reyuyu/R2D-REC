"""CPU orchestration contracts for TrueRec V2 Curriculum2048 x 2."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "analysis", ROOT / "data", ROOT / "diagnostics", ROOT / "initialization", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from run_truerec_v2_curriculum_ddp_v1 import (  # noqa: E402
    DOMAIN_ORDER, EPOCH1_ORDER_SHA256, EPOCH_STEPS, HPR_PRIORITY,
    PARENT_CHECKPOINT, PARENT_V1_STEP, RECORDS_SHA256, RUN_ID, TOTAL_STEPS,
    build_epoch2_order, checkpoint_steps, driver_state_for_cursor,
    driver_state_for_progress, formal_group_indices, load_curriculum,
    load_parent_weights_only, probe_steps, validate_every64,
    validate_restored_driver_state,
)


class TrueRecV2CurriculumRunnerTests(unittest.TestCase):
    def test_frozen_data_identity_and_epoch1_balance(self):
        records, order, metadata = load_curriculum()
        self.assertEqual((len(records), len(order), len(set(order))), (2048, 2048, 2048))
        self.assertEqual(metadata["sha256"], EPOCH1_ORDER_SHA256)
        self.assertEqual(len(RECORDS_SHA256), 64)
        validate_every64(order, records)

    def test_epoch2_is_exact_balanced_priority_replay(self):
        records = {}
        rows = []
        triggers = tuple(HPR_PRIORITY)
        index = 0
        for domain in DOMAIN_ORDER:
            for local in range(512):
                group_id = f"{domain}-{local:03d}"
                records[group_id] = {"target_domain": domain}
                rows.append({
                    "global_step": index + 1,
                    "recommendation_group_id": group_id,
                    "hpr_trigger": triggers[local % len(triggers)],
                })
                index += 1
        order = build_epoch2_order(rows, records)
        self.assertEqual((len(order), len(set(order)), set(order)), (2048, 2048, set(records)))
        validate_every64(order, records)
        for domain in DOMAIN_ORDER:
            priorities = [HPR_PRIORITY[rows_by_id(rows)[gid]] for gid in order if records[gid]["target_domain"] == domain]
            self.assertEqual(priorities, sorted(priorities))

    def test_epoch2_rejects_missing_or_duplicate_inventory(self):
        with self.assertRaisesRegex(Exception, "exactly 2048"):
            build_epoch2_order([], {})

    def test_exact_4096_schedule_and_fresh_driver(self):
        self.assertEqual((TOTAL_STEPS, EPOCH_STEPS), (4096, 2048))
        self.assertEqual(formal_group_indices(0), range(4096))
        self.assertEqual(checkpoint_steps(), tuple(range(256, 4097, 256)))
        self.assertEqual(probe_steps(), (0, *range(256, 4097, 256)))
        state = driver_state_for_cursor(0)
        self.assertEqual((state["global_step"], state["optimizer_steps"]), (0, 0))

    def test_parent_loader_is_weight_only(self):
        source = inspect.getsource(load_parent_weights_only)
        self.assertEqual(PARENT_V1_STEP, 4096)
        self.assertIn("checkpoint-step-4096", str(PARENT_CHECKPOINT))
        self.assertIn('payload["model_state_dict"]', source)
        self.assertNotIn("optimizer_state_dict", source)
        self.assertNotIn("rank_rng_states", source)
        self.assertIn("if optimizer.state", source)

    def test_solved_noop_advances_cursor_without_optimizer_step(self):
        state = driver_state_for_progress(2586, optimizer_steps=2585, solved_noop_groups=1)
        self.assertEqual(state["global_step"], 2586)
        self.assertEqual(state["optimizer_steps"], 2585)
        self.assertEqual(state["solved_noop_groups"], 1)
        self.assertEqual(validate_restored_driver_state(state, 2586), (2585, 1))

    def test_legacy_checkpoint_driver_state_remains_loadable(self):
        legacy = driver_state_for_cursor(2560)
        legacy.pop("solved_noop_groups")
        self.assertEqual(validate_restored_driver_state(legacy, 2560), (2560, 0))

    def test_optimizer_and_noop_counts_must_partition_cursor(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            driver_state_for_progress(10, optimizer_steps=8, solved_noop_groups=1)

    def test_run_identity(self):
        self.assertEqual(RUN_ID, "TRUEREC-V2-V1STEP4096-CURRICULUM2048-2E-4GPU")


def rows_by_id(rows):
    return {str(row["recommendation_group_id"]): str(row["hpr_trigger"]) for row in rows}


if __name__ == "__main__":
    unittest.main()
