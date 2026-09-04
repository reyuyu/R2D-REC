#!/usr/bin/env python3
"""CPU-only contracts for V4.3-HCR. This file does not launch training."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_rec_fdr_v43_hcr_dataset import (  # noqa: E402
    behavior_occurrences,
    exact_gold_in_cot,
    fit_behavior_reliability,
)
from hcr_v43 import (  # noqa: E402
    GroupMetadata,
    HCRMetadata,
    _candidate_evidence,
    compute_hcr_loss,
    history_fdr_loss,
    novelty_type,
    positive_indices,
    safe_negative_indices,
    topk_boundary_loss,
    validate_hcr_config,
    validate_hcr_stage_contract,
)
from rec_fdr_v43_hcr_fullft_sft import load_v43_config  # noqa: E402
from validate_rec_fdr_v43_hcr_smoke import validate_smoke  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HCRMathTest(unittest.TestCase):
    def test_novelty_partition(self) -> None:
        history = [(1, 2, 3), (1, 4, 5)]
        self.assertEqual(novelty_type((9, 8, 7), history), "T0")
        self.assertEqual(novelty_type((1, 8, 7), history), "T1")
        self.assertEqual(novelty_type((1, 2, 7), history), "T2")
        self.assertEqual(novelty_type((1, 2, 3), history), "T3")

    def test_safe_negatives_exclude_all_group_positives(self) -> None:
        anchor = (1, 2, 3)
        positives = [(1, 2, 3), (4, 5, 6), (1, 7, 8)]
        inventory = positives + [(9, 9, 9), (1, 6, 6), (1, 2, 5)]
        self.assertEqual(safe_negative_indices("a", anchor, positives, inventory), [9])
        self.assertEqual(safe_negative_indices("b", anchor, positives, inventory), [6])
        self.assertEqual(safe_negative_indices("c", anchor, positives, inventory), [5])
        self.assertEqual(positive_indices("a", anchor, positives), [1, 4])
        self.assertEqual(positive_indices("b", anchor, positives), [2, 7])

    def test_topk_boundary_monotonicity_and_skip(self) -> None:
        logits = torch.zeros(64)
        negatives = list(range(32))
        positive = [40]
        base, active = topk_boundary_loss(logits, positive, negatives, 32, 0.1, 1.0)
        self.assertTrue(active)
        raised = logits.clone()
        raised[40] = 2.0
        improved, _ = topk_boundary_loss(raised, positive, negatives, 32, 0.1, 1.0)
        self.assertLess(float(improved), float(base))
        lowered = logits.clone()
        lowered[negatives] = -2.0
        easier, _ = topk_boundary_loss(lowered, positive, negatives, 32, 0.1, 1.0)
        self.assertLess(float(easier), float(base))
        skipped, active = topk_boundary_loss(logits, positive, negatives[:31], 32, 0.1, 1.0)
        self.assertFalse(active)
        self.assertEqual(float(skipped), 0.0)

    def test_disabled_hcr_is_exact_zero_addition(self) -> None:
        logits = {level: torch.randn(8192) for level in ("a", "b", "c")}
        metadata = GroupMetadata(
            group_index=1,
            target_domain="video",
            history=((1, 2, 3),),
            target_history_n_sa=1,
            level_counts={"a": {"1": 1}, "b": {"1,2": 1}, "c": {"1,2,3": 1}},
            behavior_level_counts={"a": {}, "b": {}, "c": {}},
        )
        auxiliary, metrics = compute_hcr_loss(
            logits, (1, 2, 3), [(1, 2, 3)], [(1, 2, 3), (4, 5, 6)],
            metadata, {}, {"enabled": False}, 7,
        )
        base = torch.tensor(1.234567)
        self.assertEqual(float(auxiliary), 0.0)
        self.assertTrue(torch.equal(base + auxiliary, base))
        self.assertTrue(all(float(value) == 0.0 for value in metrics.values()))

    def test_compact_sidecar_reuses_one_history_record(self) -> None:
        payload = {
            "schema_version": 1,
            "groups": {"1": "prompt|video", "2": "prompt|video"},
            "history_records": {
                "prompt|video": {
                    "target_domain": "video",
                    "target_history_n_sa": 1,
                    "target_history_level_counts": {
                        "a": {"1": 2}, "b": {"1,2": 2}, "c": {"1,2,3": 2}
                    },
                    "behavior_level_counts": {"a": {}, "b": {}, "c": {}},
                }
            },
            "behavior_reliability": {"video": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            metadata = HCRMetadata.load(path)
        self.assertEqual(set(metadata.groups), {1, 2})
        self.assertEqual(metadata.groups[1].history, ((1, 2, 3),))

    def test_s3_s1_plus_video_multi_a_uses_existing_logits_and_backpropagates(self) -> None:
        logits = {level: torch.zeros(8192, requires_grad=True) for level in ("a", "b", "c")}
        anchor = (1, 2, 3)
        positives = [anchor, (4, 5, 6)]
        inventory = [
            *[(100 + i, 7, 8) for i in range(40)],
            *[(1, 100 + i, 8) for i in range(40)],
            *[(1, 2, 100 + i) for i in range(40)],
        ]
        history = tuple(inventory[:8] + inventory[40:48] + inventory[80:88])
        level_counts = {
            "a": {str(item[0]): 1 for item in history},
            "b": {f"{item[0]},{item[1]}": 1 for item in history},
            "c": {f"{item[0]},{item[1]},{item[2]}": 1 for item in history},
        }
        metadata = GroupMetadata(
            group_index=1,
            target_domain="video",
            history=history,
            target_history_n_sa=len({item[0] for item in history}),
            level_counts=level_counts,
            behavior_level_counts={level: {"深度观看": counts} for level, counts in level_counts.items()},
        )
        config = load_v43_config(
            HERE / "configs/rec_fdr_v43_s3_multi_a_from0_fullft_32k_4gpu.yaml"
        )["hcr"]
        auxiliary, metrics = compute_hcr_loss(
            logits, anchor, positives, inventory, metadata,
            {"video": {"深度观看": {"a": 0.5, "b": 0.7, "c": 1.0}}}, config, 7,
        )
        self.assertTrue(torch.isfinite(auxiliary))
        self.assertGreater(float(auxiliary.detach()), 0.0)
        self.assertEqual(float(metrics["video_multi_a_active"]), 1.0)
        self.assertEqual(float(metrics["history_fdr_total"]), 0.0)
        self.assertEqual(float(metrics["history_pair_count_a"]), 0.0)
        self.assertGreater(float(metrics["weighted_multi_a"]), 0.0)
        self.assertEqual(float(metrics["video_multi_a_eligible"]), 1.0)
        auxiliary.backward()
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all() for value in logits.values()))

    def test_s3_multi_a_is_video_only(self) -> None:
        logits = {level: torch.zeros(256) for level in ("a", "b", "c")}
        positives = [(1, 2, 3), (4, 5, 6)]
        inventory = [*positives, *[(20 + index, 7, 8) for index in range(40)]]
        metadata = GroupMetadata(
            group_index=1,
            target_domain="prod",
            history=(),
            target_history_n_sa=0,
            level_counts={"a": {}, "b": {}, "c": {}},
            behavior_level_counts={"a": {}, "b": {}, "c": {}},
        )
        config = load_v43_config(
            HERE / "configs/rec_fdr_v43_s3_multi_a_from0_fullft_32k_4gpu.yaml"
        )["hcr"]
        _, metrics = compute_hcr_loss(
            logits, positives[0], positives, inventory, metadata, {}, config, 7
        )
        self.assertEqual(float(metrics["multi_a_loss"]), 0.0)
        self.assertEqual(float(metrics["video_multi_a_active"]), 0.0)

    def test_s1_pushes_anchor_only_and_protects_other_group_positive(self) -> None:
        logits = {level: torch.zeros(8192, requires_grad=True) for level in ("a", "b", "c")}
        anchor = (1, 2, 3)
        other_positive = (4, 5, 6)
        inventory = [anchor, other_positive, *[(100 + i, 200 + i, 300 + i) for i in range(40)]]
        metadata = GroupMetadata(
            group_index=1,
            target_domain="video",
            history=(),
            target_history_n_sa=0,
            level_counts={"a": {}, "b": {}, "c": {}},
            behavior_level_counts={"a": {}, "b": {}, "c": {}},
        )
        config = load_v43_config(
            HERE / "configs/rec_fdr_v43_s1_topk_from0_fullft_32k_4gpu.yaml"
        )["hcr"]
        auxiliary, _ = compute_hcr_loss(
            logits, anchor, [anchor, other_positive], inventory, metadata, {}, config, 7,
        )
        auxiliary.backward()
        self.assertNotEqual(float(logits["a"].grad[anchor[0]]), 0.0)
        self.assertEqual(float(logits["a"].grad[other_positive[0]]), 0.0)

    def test_behavior_evidence_is_nonnegative_monotonic_and_unknown_zero(self) -> None:
        def metadata(count: int) -> GroupMetadata:
            return GroupMetadata(
                group_index=count,
                target_domain="video",
                history=(),
                target_history_n_sa=0,
                level_counts={"a": {}, "b": {}, "c": {}},
                behavior_level_counts={
                    "a": {"deep": {"1": count}, "unknown": {"1": 100}},
                    "b": {},
                    "c": {},
                },
            )

        reliability = {"video": {"deep": {"a": 0.8}, "unknown": {"a": 0.0}}}
        one = _candidate_evidence(metadata(1), reliability, "a", "1")
        many = _candidate_evidence(metadata(10), reliability, "a", "1")
        self.assertGreaterEqual(one, 0.0)
        self.assertGreater(many, one)

    def test_history_fdr_reports_selected_competitor_support_and_frequency(self) -> None:
        metadata = GroupMetadata(
            group_index=1,
            target_domain="video",
            history=((4, 8, 9), (4, 8, 9), (4, 8, 9), (5, 8, 9)),
            target_history_n_sa=2,
            level_counts={
                "a": {"4": 3, "5": 1},
                "b": {"4,8": 3, "5,8": 1},
                "c": {"4,8,9": 3, "5,8,9": 1},
            },
            behavior_level_counts={
                "a": {"deep": {"4": 3, "5": 1}},
                "b": {"deep": {"4,8": 3, "5,8": 1}},
                "c": {"deep": {"4,8,9": 3, "5,8,9": 1}},
            },
        )
        loss, metrics = history_fdr_loss(
            {level: torch.zeros(16) for level in ("a", "b", "c")},
            (1, 2, 3),
            [(1, 2, 3)],
            metadata,
            {"video": {"deep": {"a": 0.5, "b": 0.5, "c": 0.5}}},
            {"max_history_hard_pairs_per_level": 2, "history_hard_fraction": 1.0},
            7,
        )
        expected_support = (0.5 * math.log1p(3) + 0.5 * math.log1p(1)) / 2
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(metrics["history_pair_count_a"]), 2.0)
        self.assertAlmostEqual(
            float(metrics["selected_history_competitor_behavior_support"]),
            expected_support,
            places=6,
        )
        self.assertEqual(float(metrics["selected_history_competitor_raw_frequency"]), 2.0)


class DatasetContractTest(unittest.TestCase):
    def test_exact_gold_leak_checks_think_only(self) -> None:
        sid = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
        row = {"response": f"<think>{sid}</think>bridge {sid}", "final_target_sid": sid}
        self.assertTrue(exact_gold_in_cot(row))
        row["response"] = f"<think>clean</think>bridge {sid}"
        self.assertFalse(exact_gold_in_cot(row))

    def test_behavior_parser_separates_hierarchy_evidence(self) -> None:
        deep = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
        normal = "<|video_begin|><s_a_4><s_b_5><s_c_6>"
        parsed = behavior_occurrences(f"用户视频行为: 深度观看了 {deep}，看过 {normal}。", "video")
        self.assertIn((1, 2, 3), parsed["深度观看"])
        self.assertIn((4, 5, 6), parsed["普通观看"])
        self.assertNotIn((1, 2, 3), parsed["普通观看"])

    def test_reliability_fit_contract_is_explicit(self) -> None:
        source = inspect.getsource(fit_behavior_reliability)
        self.assertIn('behavior == "unknown"', source)
        self.assertIn("effective_rates[behavior][level] / maximum", source)


class ConfigurationContractTest(unittest.TestCase):
    def test_stage_configs_resolve_expected_flags(self) -> None:
        stages = {
            "s1": (False, False),
            "s2": (True, False),
            "s3": (False, True),
        }
        for stage, (history_enabled, multi_enabled) in stages.items():
            path = next((HERE / "configs").glob(f"rec_fdr_v43_{stage}_*_from0_fullft_32k_4gpu.yaml"))
            config = load_v43_config(path)
            self.assertTrue(config["hcr"]["novelty_topk"]["enabled"])
            self.assertEqual(config["hcr"]["history_fdr"]["enabled"], history_enabled)
            self.assertEqual(config["hcr"]["multi_a"]["enabled"], multi_enabled)
            self.assertEqual(config["v43_experiment"]["initialization"], "from_pretrain")
            self.assertEqual(config["v43_experiment"]["gold_sid_cot_leak_policy"], "preserve_audit_only")
            self.assertEqual(config["model_name_or_path"], "/data/LLm-8B/code/OneReason-8B-pretrain-competition")
            validate_hcr_config(config["hcr"])
            validate_hcr_stage_contract(stage, config["hcr"])

    def test_s3_stage_contract_rejects_history_fdr(self) -> None:
        config = load_v43_config(
            HERE / "configs/rec_fdr_v43_s3_multi_a_from0_fullft_32k_4gpu.yaml"
        )["hcr"]
        config["history_fdr"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "s3 HCR modules"):
            validate_hcr_stage_contract("s3", config)

    def test_hcr_off_and_continuation_pilot_are_separate_controls(self) -> None:
        disabled = load_v43_config(
            HERE / "configs/rec_fdr_v43_hcr_off_regression_from0_fullft_32k_4gpu.yaml"
        )
        self.assertFalse(disabled["hcr"]["enabled"])
        pilot = load_v43_config(
            HERE / "configs/rec_fdr_v43_s1_topk_from_v42a_continuation_pilot.yaml"
        )
        self.assertEqual(pilot["v43_experiment"]["role"], "continuation_pilot")
        self.assertEqual(pilot["learning_rate"], 2.0e-6)
        self.assertEqual(pilot["curriculum_max_steps_override"], 50)

    def test_v42_core_helpers_are_source_identical(self) -> None:
        v42 = load_module("v42_contract_source", HERE / "rec_fdr_v42_fullft_sft.py")
        v43 = load_module("v43_contract_source", HERE / "rec_fdr_v43_hcr_fullft_sft.py")
        for name in (
            "_content_token_fields",
            "multi_positive_set_loss",
            "first_divergence_pairs",
            "first_divergence_rank_loss",
            "make_token_plan",
        ):
            self.assertEqual(inspect.getsource(getattr(v42, name)), inspect.getsource(getattr(v43, name)))

    def test_hcr_metrics_do_not_create_undeclared_mode_reduction_keys(self) -> None:
        source = inspect.getsource(
            load_module("v43_mode_metric_contract", HERE / "rec_fdr_v43_hcr_fullft_sft.py")
            ._patch_collator_trainer_and_model
        )
        self.assertIn("if mode_key in MODE_METRICS", source)


class SmokeValidatorTest(unittest.TestCase):
    @staticmethod
    def _write_s2_smoke(directory: Path, overrides: dict[str, float] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "smoke_world_size.txt").write_text("4\n", encoding="utf-8")
        metrics = {
            "train/history_fdr_total": 0.25,
            "train/history_pair_count_a": 2.0,
            "train/history_pair_count_b": 0.0,
            "train/history_pair_count_c": 0.0,
            "train/behavior_evidence_pos_A": 0.0,
            "train/behavior_evidence_pos_B": 0.0,
            "train/behavior_evidence_pos_C": 0.0,
            "train/behavior_evidence_neg_A": 0.4,
            "train/behavior_evidence_neg_B": 0.0,
            "train/behavior_evidence_neg_C": 0.0,
            "train/selected_history_competitor_behavior_support": 0.4,
            "train/selected_history_competitor_raw_frequency": 2.0,
        }
        metrics.update(overrides or {})
        rows = [{"step": step, "loss": 1.0, **metrics} for step in (1, 2)]
        (directory / "trainer_log.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_s2_smoke_requires_active_history_fdr_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_s2_smoke(run_dir)
            summary = validate_smoke(run_dir, "s2")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["history_fdr"]["pair_count_sum"], 4.0)

    def test_s2_smoke_rejects_inactive_history_fdr_signals(self) -> None:
        inactive_cases = {
            "loss": {"train/history_fdr_total": 0.0},
            "pairs": {"train/history_pair_count_a": 0.0},
            "behavior": {
                "train/behavior_evidence_neg_A": 0.0,
                "train/selected_history_competitor_behavior_support": 0.0,
            },
            "raw_frequency": {"train/selected_history_competitor_raw_frequency": 0.0},
        }
        for label, overrides in inactive_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                self._write_s2_smoke(run_dir, overrides)
                with self.assertRaises(AssertionError):
                    validate_smoke(run_dir, "s2")

    @staticmethod
    def _write_s3_smoke(directory: Path, overrides: dict[str, float] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "smoke_world_size.txt").write_text("4\n", encoding="utf-8")
        metrics = {
            "train/hcr_topk_total": 0.3,
            "train/multi_a_loss": 0.2,
            "train/video_multi_a_active": 1.0,
            "train/video_positive_a_count": 2.0,
            "train/hcr_aux_total": 0.15,
            "train/rec_base_v42": 1.0,
            "train/weighted_multi_a": 0.05,
            "train/video_multi_a_active_instances_per_pack": 2.0,
            "train/video_multi_a_unique_eligible_groups_per_pack": 1.0,
            "train/video_multi_a_repeat_factor": 2.0,
            "train/history_fdr_total": 0.0,
            "train/history_pair_count_a": 0.0,
            "train/history_pair_count_b": 0.0,
            "train/history_pair_count_c": 0.0,
            "train/selected_history_competitor_behavior_support": 0.0,
            "train/selected_history_competitor_raw_frequency": 0.0,
        }
        metrics.update(overrides or {})
        rows = [{"step": step, "loss": 1.0, **metrics} for step in (1, 2)]
        (directory / "trainer_log.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_s3_smoke_requires_s1_plus_video_multi_a_and_zero_history_fdr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_s3_smoke(run_dir)
            summary = validate_smoke(run_dir, "s3")
        self.assertTrue(summary["s1_video_multi_a"]["history_fdr_exact_zero"])
        self.assertEqual(
            summary["s1_video_multi_a"]["auxiliary_gate"], "automatic_pass"
        )
        self.assertAlmostEqual(
            summary["s1_video_multi_a"]["max_hcr_aux_over_base_rec"], 0.15
        )

    def test_s3_smoke_auxiliary_scale_requires_review_and_has_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_s3_smoke(run_dir, {"train/hcr_aux_total": 0.25})
            with self.assertRaisesRegex(AssertionError, "manual-review band"):
                validate_smoke(run_dir, "s3")
            summary = validate_smoke(run_dir, "s3", allow_aux_review=True)
            self.assertEqual(
                summary["s1_video_multi_a"]["auxiliary_gate"],
                "manual_review_approved",
            )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_s3_smoke(run_dir, {"train/hcr_aux_total": 0.31})
            with self.assertRaisesRegex(AssertionError, "30% hard limit"):
                validate_smoke(run_dir, "s3", allow_aux_review=True)

    def test_s3_smoke_accepts_clean_stop_record_without_joint_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_s3_smoke(run_dir)
            rows = [
                json.loads(line)
                for line in (run_dir / "trainer_log.jsonl").read_text().splitlines()
            ]
            rows[1] = {"step": 2, "loss": 0.9}
            (run_dir / "trainer_log.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = validate_smoke(run_dir, "s3")
        self.assertEqual(summary["s1_video_multi_a"]["auxiliary_gate"], "automatic_pass")
        self.assertEqual(summary["s1_video_multi_a"]["max_effective_repeat_factor"], 2.0)

    def test_s3_smoke_rejects_wrong_module_activity(self) -> None:
        invalid_cases = {
            "topk": {"train/hcr_topk_total": 0.0},
            "multi_loss": {"train/multi_a_loss": 0.0},
            "multi_active": {"train/video_multi_a_active": 0.0},
            "positive_modes": {"train/video_positive_a_count": 1.0},
            "history_loss": {"train/history_fdr_total": 0.1},
            "history_pairs": {"train/history_pair_count_a": 1.0},
            "history_behavior": {
                "train/selected_history_competitor_behavior_support": 0.1
            },
        }
        for label, overrides in invalid_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                self._write_s3_smoke(run_dir, overrides)
                with self.assertRaises(AssertionError):
                    validate_smoke(run_dir, "s3")


if __name__ == "__main__":
    unittest.main()
