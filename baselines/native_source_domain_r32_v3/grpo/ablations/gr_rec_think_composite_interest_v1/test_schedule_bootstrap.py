from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from .composite_probe import CompositeProbeCallback, CompositeThinkProbeEvaluator
from .gpu_preflight import main as preflight_main
from .run_gr_rec_think_composite_interest_v1 import (
    CHECKPOINT_STEPS,
    PROBE_DOMAIN_ORDER,
    PROBE_IDS,
    PROBE_ROUNDS,
    PROBE_STEPS,
    launch_training,
    prepare_plan,
    should_run_probe,
    should_save_checkpoint,
)
from .single_node_nccl import configure_single_node_nccl


class RunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = prepare_plan(SimpleNamespace(
            run_id="CPU-CONTRACT",
            max_steps=716,
            output_dir="/tmp/cpu-contract",
            grpo_data=Path("/data/GRPO/data/rec_mp_grpo_v2/train.jsonl"),
            gold_data=Path(
                "/data/lf_data_versions/alltrain/bata_baseline_v1/"
                "onereason_bata_baseline.jsonl"
            ),
        ))

    def test_twelve_unique_eligible_probes_and_domain_balance(self):
        self.assertEqual(len(PROBE_IDS), 12)
        self.assertEqual(len(set(PROBE_IDS)), 12)
        counts = {domain: 0 for domain in PROBE_DOMAIN_ORDER}
        for domain in self.plan["probe_domains"].values():
            counts[domain] += 1
        self.assertEqual(counts, {"video": 3, "prod": 3, "ad": 3, "living": 3})
        self.assertEqual(set(self.plan["probe_records"]), set(PROBE_IDS))

    def test_three_round_rank_mapping_covers_each_domain_once(self):
        self.assertEqual(len(PROBE_ROUNDS), 3)
        for round_ids in PROBE_ROUNDS:
            self.assertEqual(len(round_ids), 4)
            self.assertEqual(
                tuple(self.plan["probe_domains"][group_id] for group_id in round_ids),
                PROBE_DOMAIN_ORDER,
            )

    def test_new_topology_and_zero_overlap(self):
        topology = self.plan["topology"]
        self.assertEqual(topology["parser_valid_gold_groups"], 1446)
        self.assertEqual(topology["post_probe_groups"], 1434)
        self.assertEqual(topology["sampler_drop_groups"], 2)
        self.assertEqual(topology["training_groups"], 1432)
        self.assertEqual(topology["fresh_g4_rollouts"], 358)
        self.assertEqual(topology["optimizer_steps_num_iterations_2"], 716)
        self.assertEqual(self.plan["train_probe_overlap"], 0)

    def test_explicit_checkpoint_and_probe_schedules(self):
        self.assertEqual(CHECKPOINT_STEPS, (200, 400, 600, 716))
        self.assertEqual(PROBE_STEPS, (0, 200, 400, 600, 716))
        self.assertEqual(
            [step for step in range(717) if should_run_probe(step)],
            list(PROBE_STEPS),
        )
        self.assertFalse(should_save_checkpoint(700))
        self.assertFalse(should_save_checkpoint(720))


class ProbeStateTests(unittest.TestCase):
    def test_partial_step_is_not_complete_but_all_twelve_is(self):
        evaluator = CompositeThinkProbeEvaluator.__new__(CompositeThinkProbeEvaluator)
        evaluator.group_ids = list(PROBE_IDS)
        evaluator.trainer = SimpleNamespace(
            accelerator=SimpleNamespace(is_main_process=True)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.jsonl"
            evaluator.monitor = SimpleNamespace(run_dir=Path(directory))
            path.write_text("\n".join(
                json.dumps({"step": 200, "group_id": group_id})
                for group_id in PROBE_IDS[:-1]
            ) + "\n", encoding="utf-8")
            with mock.patch("grpo_probe.dist.is_initialized", return_value=False):
                self.assertFalse(evaluator._already_complete(200))
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": 200, "group_id": PROBE_IDS[-1]}) + "\n")
                self.assertTrue(evaluator._already_complete(200))

    def test_callback_ignores_non_milestones(self):
        calls = []
        evaluator = SimpleNamespace(
            probe_steps=PROBE_STEPS,
            evaluate=lambda step, reason: calls.append((step, reason)),
        )
        callback = CompositeProbeCallback(evaluator)
        callback.on_train_begin(None, SimpleNamespace(global_step=0), None)
        callback.on_step_end(None, SimpleNamespace(global_step=100), None)
        callback.on_step_end(None, SimpleNamespace(global_step=200), None)
        self.assertEqual(calls, [(0, "baseline"), (200, "milestone")])


class NcclBootstrapTests(unittest.TestCase):
    def test_loopback_and_device_are_set_before_dist_init(self):
        events = []

        class FakeCuda:
            @staticmethod
            def set_device(rank):
                events.append(("set_device", rank, os.environ.get("NCCL_SOCKET_IFNAME")))

        fake_torch = SimpleNamespace(cuda=FakeCuda(), device=lambda value: value)

        class FakeDist:
            @staticmethod
            def is_initialized():
                return False

            @staticmethod
            def init_process_group(**kwargs):
                events.append(("init", kwargs, os.environ.get("NCCL_SOCKET_IFNAME")))

        with mock.patch.dict(os.environ, {
            "LOCAL_RANK": "2", "WORLD_SIZE": "4", "NCCL_SOCKET_IFNAME": "bad0",
        }):
            result = configure_single_node_nccl(
                torch_module=fake_torch, dist_module=FakeDist,
            )
        self.assertEqual(events[0], ("set_device", 2, "lo"))
        self.assertEqual(events[1][0], "init")
        self.assertEqual(events[1][1], {"backend": "nccl", "device_id": "cuda:2"})
        self.assertEqual(events[1][2], "lo")
        self.assertTrue(result["initialized_here"])

    def test_preflight_and_formal_use_the_shared_helper(self):
        for function in (preflight_main, launch_training):
            source = inspect.getsource(function)
            self.assertIn("configure_single_node_nccl", source)
            self.assertIn("initialize=True", source)


if __name__ == "__main__":
    unittest.main()
