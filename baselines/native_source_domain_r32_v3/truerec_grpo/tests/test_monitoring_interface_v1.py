"""CPU contracts for detached TrueRec training and probe monitoring."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "analysis", ROOT / "credit", ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from monitoring_v1 import (  # noqa: E402
    MonitoringContractError, MonitoringWriterV1, TOKEN_FIELDS,
    build_train_explain, build_train_group_record, capture_frontier_chunk,
    detached_jsonable, local_payload_from_backward, runtime_monitoring_snapshot,
)
from rollout_metrics import assess_candidate  # noqa: E402
from rollout_runtime_v1 import BusinessGroupRollout, RolloutCandidate  # noqa: E402
from truerec_grpo_trainer_v1 import format_credit_tensors  # noqa: E402
from truerec_loss_v1 import frontier_ppo_loss  # noqa: E402
from truerec_runtime_v1 import build_group_runtime_plan  # noqa: E402


TOKENS = {
    1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>", 4: "<s_a_4>",
    5: "<s_b_5>", 6: "<s_c_6>", 7: "<s_c_7>", 20: "bad", 21: "also_bad",
}
TOKEN_IDS = {value: key for key, value in TOKENS.items()}
GOLD = ("<s_a_1><s_b_2><s_c_3>", "<s_a_1><s_b_2><s_c_7>")


def id_to_token(value):
    return TOKENS.get(int(value), f"token_{value}")


CASE_COMPLETIONS = {
    "HPR_A": [[4, 5, 6]] * 8,
    "HPR_B": [[1, 5, 6]] * 8,
    "HPR_C": [[1, 2, 6]] * 8,
    "HPR_NONE": [[1, 2, 3]] * 8,
    "FORMAT_INVALID": [[20, 21]] * 8,
}


def make_group(case_name: str) -> BusinessGroupRollout:
    candidates = []
    for index, ids in enumerate(CASE_COMPLETIONS[case_name]):
        metrics = assess_candidate(ids, id_to_token, GOLD, "<|video_begin|>", ())
        candidates.append(RolloutCandidate(index, tuple(ids), tuple(-1.0 for _ in ids), metrics))
    return BusinessGroupRollout("group-0", (10, 11, 12), "<|video_begin|>", GOLD, tuple(candidates))


def runtime_for(group):
    return build_group_runtime_plan(
        [candidate.metrics for candidate in group.candidates], group.all_gold_abc, TOKEN_IDS.__getitem__,
    )


def token_row(candidate, position):
    token_id = candidate.completion_ids[position]
    return {
        "action_position": position, "position_name": "ABC"[position], "token_id": token_id,
        "old_logp": -1.0, "current_logp_pre_step": -1.0, "ppo_ratio": 1.0,
        "prefix_gate_active": True, "frontier_token_credit": 0.0,
        "format_token_credit": 0.0, "effective_signed_credit": 0.0,
        "ppo_unclipped_surrogate": 0.0, "ppo_clipped_surrogate": 0.0,
        "ppo_selected_surrogate": 0.0, "frontier_loss_contribution": 0.0,
    }


def rank_payloads(group, runtime):
    payloads = [{"candidates": [], "hpr_positions": []} for _ in range(4)]
    for candidate in group.candidates:
        payloads[candidate.sample_index // 2]["candidates"].append({
            "candidate_index": candidate.sample_index,
            "tokens": [token_row(candidate, position) for position in range(len(candidate.completion_ids))],
        })
    for site_index, site in enumerate(runtime.hpr.sites):
        weight = 1.0 / (len(runtime.hpr.sites) * len(site.onpolicy_positions))
        for candidate_index, action_position in site.onpolicy_positions:
            payloads[candidate_index // 2]["hpr_positions"].append({
                "site_index": site_index, "target_position": site.level,
                "candidate_index": candidate_index, "action_position": action_position,
                "target_token_ids": list(site.target_token_ids),
                "multi_positive_log_mass": -2.0, "hpr_position_loss": 2.0,
                "position_weight": weight, "raw_hpr_contribution": 2.0 * weight,
                "weighted_hpr_contribution": 0.02 * 2.0 * weight,
            })
    return payloads


def memory_rows():
    return [
        {"rank": rank, "healthy": True, "allocated_gb": 10.0 + rank,
         "reserved_gb": 11.0 + rank, "peak_allocated_gb": 12.0 + rank,
         "peak_reserved_gb": 13.0 + rank}
        for rank in range(4)
    ]


class MonitoringInterfaceV1Tests(unittest.TestCase):
    def test_monitoring_tensor_must_be_detached(self):
        with self.assertRaises(MonitoringContractError):
            detached_jsonable(torch.tensor(1.0, requires_grad=True))
        self.assertEqual(detached_jsonable(torch.tensor([1.0, 2.0]).detach()), [1.0, 2.0])

    def test_backward_payload_helper_rejects_graph_state(self):
        safe = SimpleNamespace(local_candidate_monitoring=({"value": 1.0},), local_hpr_monitoring=())
        self.assertEqual(local_payload_from_backward(safe), {"candidates": [{"value": 1.0}], "hpr_positions": []})
        unsafe = SimpleNamespace(
            local_candidate_monitoring=({"value": torch.tensor(1.0, requires_grad=True)},),
            local_hpr_monitoring=(),
        )
        with self.assertRaises(MonitoringContractError):
            local_payload_from_backward(unsafe)

    def test_capture_preserves_real_frontier_loss_and_gradient(self):
        group = make_group("HPR_A")
        runtime = runtime_for(group)
        format_credits, format_mask = format_credit_tensors(group)
        old = torch.full((1, 2, 3), -1.0)
        current_reference = torch.tensor([[[-0.9, -1.1, -0.8], [-1.2, -0.7, -1.3]]], requires_grad=True)
        current_monitored = current_reference.detach().clone().requires_grad_(True)

        reference = frontier_ppo_loss(
            current_reference, old, runtime.token_credits[:2].unsqueeze(0),
            runtime.token_credit_mask[:2].unsqueeze(0), 0.2,
        )
        reference_format = frontier_ppo_loss(
            current_reference, old, format_credits[:2].unsqueeze(0), format_mask[:2].unsqueeze(0), 0.2,
        )
        reference_total = reference.loss + reference_format.loss
        reference_total.backward()

        monitored = frontier_ppo_loss(
            current_monitored, old, runtime.token_credits[:2].unsqueeze(0),
            runtime.token_credit_mask[:2].unsqueeze(0), 0.2,
        )
        monitored_format = frontier_ppo_loss(
            current_monitored, old, format_credits[:2].unsqueeze(0), format_mask[:2].unsqueeze(0), 0.2,
        )
        captured = capture_frontier_chunk(
            group=group, start=0, stop=2, current=current_monitored[0], runtime=runtime,
            format_credits=format_credits, hierarchy=monitored, format_loss=monitored_format,
        )
        monitored_total = monitored.loss + monitored_format.loss
        monitored_total.backward()
        torch.testing.assert_close(monitored_total, reference_total)
        torch.testing.assert_close(current_monitored.grad, current_reference.grad)
        self.assertIsInstance(json.dumps(captured), str)

    def test_frontier_and_format_credit_are_separate(self):
        group = make_group("FORMAT_INVALID")
        runtime = runtime_for(group)
        format_credits, format_mask = format_credit_tensors(group)
        current = torch.full((1, 2, 3), -1.0, requires_grad=True)
        old = current.detach().clone()
        hierarchy = frontier_ppo_loss(current, old, runtime.token_credits[:2].unsqueeze(0), runtime.token_credit_mask[:2].unsqueeze(0), 0.2)
        formatting = frontier_ppo_loss(current, old, format_credits[:2].unsqueeze(0), format_mask[:2].unsqueeze(0), 0.2)
        rows = capture_frontier_chunk(group=group, start=0, stop=2, current=current[0], runtime=runtime, format_credits=format_credits, hierarchy=hierarchy, format_loss=formatting)
        token = rows[0]["tokens"][0]
        self.assertEqual(token["frontier_token_credit"], 0.0)
        self.assertNotEqual(token["format_token_credit"], 0.0)
        self.assertNotIn("normalized_advantage", token)

    def test_all_hpr_triggers_serialize_with_complete_candidate_schema(self):
        expected = {"HPR_A": "HPR_A", "HPR_B": "HPR_B", "HPR_C": "HPR_C", "HPR_NONE": "HPR_NONE", "FORMAT_INVALID": "HPR_A"}
        for case_name, trigger in expected.items():
            with self.subTest(case=case_name):
                group = make_group(case_name)
                runtime = runtime_for(group)
                explain = build_train_explain(group, runtime_monitoring_snapshot(runtime), rank_payloads(group, runtime), id_to_token)
                self.assertEqual(explain["hpr_trigger"], trigger)
                self.assertEqual(explain["global_candidate_indices"], list(range(8)))
                self.assertEqual([row["source_rank"] for row in explain["candidates"]], [0, 0, 1, 1, 2, 2, 3, 3])
                for candidate in explain["candidates"]:
                    for token in candidate["action_tokens"]:
                        self.assertEqual(set(token), set(TOKEN_FIELDS))
                self.assertIsInstance(json.dumps(explain), str)

    def test_missing_or_reordered_global_candidate_is_rejected(self):
        group = make_group("HPR_A")
        runtime = runtime_for(group)
        payloads = rank_payloads(group, runtime)
        payloads[0]["candidates"].pop()
        with self.assertRaises(MonitoringContractError):
            build_train_explain(group, runtime_monitoring_snapshot(runtime), payloads, id_to_token)

    def _group_record(self, index=0):
        group = make_group("HPR_A")
        runtime = runtime_for(group)
        backward = SimpleNamespace(
            global_frontier_value=0.1, global_hpr_value_raw=2.0,
            global_hpr_value_weighted=0.04, global_total_value=0.14,
        )
        record = build_train_group_record(
            record={"target_domain": "video"}, group=group,
            backward=backward, global_step=index + 1, group_index=index,
            selected_microbatch_size=2, gradient_norm=1.5,
            wall_time_seconds=4.0, rank_memory=memory_rows(),
        )
        explain = build_train_explain(group, runtime_monitoring_snapshot(runtime), rank_payloads(group, runtime), id_to_token)
        return record, explain

    def test_rank_zero_jsonl_order_and_atomic_live_replace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            writer = MonitoringWriterV1(root, total_steps=4096, rank=0)
            writer.record_train_group(*self._group_record(0), last_checkpoint_step=None)
            writer.record_train_group(*self._group_record(1), last_checkpoint_step=1)
            groups = [json.loads(line) for line in (root / "train_groups.jsonl").read_text().splitlines()]
            explains = [json.loads(line) for line in (root / "train_explain.jsonl").read_text().splitlines()]
            self.assertEqual([row["group_index"] for row in groups], [0, 1])
            self.assertEqual([row["group_index"] for row in explains], [0, 1])
            live = json.loads((root / "live_state.json").read_text())
            self.assertEqual((live["current_step"], live["last_checkpoint_step"]), (2, 1))
            self.assertEqual(len(live["rank_health"]), 4)
            self.assertFalse((root / "live_state.json.tmp").exists())
            with self.assertRaises(MonitoringContractError):
                writer.record_train_group(*self._group_record(3), last_checkpoint_step=1)

    def test_nonzero_rank_never_writes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            writer = MonitoringWriterV1(root, total_steps=4096, rank=2)
            self.assertFalse(writer.record_train_group(*self._group_record(), last_checkpoint_step=None))
            self.assertFalse(writer.write_probe(50, summary={}, groups=[], explains=[]))
            self.assertEqual(list(root.iterdir()), [])

    def test_probe_schema_writer_does_not_run_optimizer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            writer = MonitoringWriterV1(root, total_steps=4096, rank=0)
            group, explain = self._group_record()
            writer.write_probe(50, summary={"groups": 20}, groups=[group], explains=[explain])
            directory = root / "probe" / "50"
            summary = json.loads((directory / "summary.json").read_text())
            probe_group = json.loads((directory / "groups.jsonl").read_text())
            probe_explain = json.loads((directory / "explain.jsonl").read_text())
            for value in (summary, probe_group, probe_explain):
                self.assertEqual(value["mode"], "PROBE")
                self.assertIs(value["optimizer_update"], False)
                self.assertEqual(value["probe_step"], 50)


if __name__ == "__main__":
    unittest.main()
