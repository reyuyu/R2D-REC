import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

from .run_think_suffix_sid_train import (
    PARENT_ADAPTER,
    PARENT_ADAPTER_SHA256,
    RUN_ID_PREFIX,
    build_think_only_dataset,
    suffix_config_kwargs,
)
from .think_suffix_sid_trainer import (
    ThinkG8SingleGroupSampler,
    ThinkSuffixSIDTrainer,
    audit_think_g8_sampler,
)
from .runtime_import_provenance import (
    EXPECTED_SCRIPTS_DIR,
    assert_runtime_import_provenance,
    evaluate_runtime_import_provenance,
)


class FakeDataset(list):
    def select(self, indices):
        return FakeDataset([self[index] for index in indices])


class TrainerContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = FakeDataset([
            {"route": "think", "recommendation_group_id": "g1"},
            {"route": "no_think", "recommendation_group_id": "g1"},
            {"route": "think", "recommendation_group_id": "g2"},
            {"route": "no_think", "recommendation_group_id": "g2"},
        ])

    def test_dataset_and_sampler_are_think_only_g8_single_group(self):
        dataset = build_think_only_dataset(self.rows)
        sampler = ThinkG8SingleGroupSampler(dataset, repeat_count=2, shuffle=False)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(list(sampler), [0] * 16 + [1] * 16)
        audit = audit_think_g8_sampler(dataset, sampler)
        self.assertEqual(audit["unique_groups_per_global_rollout"], 1)
        self.assertEqual(audit["candidates_per_round"], 8)
        self.assertEqual(audit["max_candidates_per_group"], 32)
        self.assertEqual(audit["optimizer_steps"], 4)

    def test_config_preserves_sampling_and_uses_g8(self):
        cfg = suffix_config_kwargs()
        self.assertEqual(cfg["per_device_train_batch_size"], 2)
        self.assertEqual(cfg["generation_batch_size"], 8)
        self.assertEqual(cfg["num_generations"], 8)
        self.assertEqual(cfg["num_iterations"], 2)
        self.assertEqual(cfg["temperature"], 0.9)
        self.assertEqual(cfg["top_p"], 0.95)
        self.assertEqual(cfg["beta"], 0.0)
        self.assertEqual(cfg["epsilon"], 0.2)

    def test_loss_uses_full_attention_and_separate_suffix_mask(self):
        source = inspect.getsource(ThinkSuffixSIDTrainer._compute_loss)
        self.assertIn('attention_mask = torch.cat([prompt_mask, completion_mask]', source)
        self.assertIn('per_token_loss * loss_mask', source)
        self.assertNotIn('attention_mask = torch.cat([prompt_mask, loss_mask]', source)

    def test_monitor_requests_all_eight_global_candidates(self):
        self.assertTrue(ThinkSuffixSIDTrainer._capture_global_completion_ids(None))

    def test_generation_contract_allows_think_g8_without_changing_sampling(self):
        self.assertEqual(
            ThinkSuffixSIDTrainer._route_generation_contract(None, "think"),
            {"group_size": 8, "temperature": 0.9, "top_p": 0.95},
        )

    def test_generation_explicitly_continues_past_think_close(self):
        source = inspect.getsource(ThinkSuffixSIDTrainer.__init__)
        self.assertIn("self._stop_think_at_closure = False", source)

    def test_runtime_import_provenance_passes_current_worktree(self):
        result = assert_runtime_import_provenance()
        self.assertEqual(result["runtime_import_provenance"], "PASS")
        self.assertTrue(result["baseline_trainer_identity"])

    def test_runtime_import_provenance_rejects_stale_scripts(self):
        import grpo_model
        import grpo_trl_trainer
        import monitor.writer
        import run_grpo_trl_train

        stale = SimpleNamespace(
            __file__="/data/GRPO/scripts/grpo_trl_trainer.py",
            RecGRPOTrainer=grpo_trl_trainer.RecGRPOTrainer,
        )
        result = evaluate_runtime_import_provenance(
            run_grpo_trl_train,
            grpo_model,
            stale,
            monitor.writer,
            expected_scripts_dir=EXPECTED_SCRIPTS_DIR,
        )
        self.assertEqual(result["runtime_import_provenance"], "FAIL")

    def test_shared_formal_runner_has_no_stale_scripts_precedence(self):
        import run_grpo_trl_train

        source = Path(run_grpo_trl_train.__file__).read_text(encoding="utf-8")
        self.assertNotIn('sys.path.insert(0, "/data/GRPO/scripts")', source)

    def test_parent_is_immutable_step1500(self):
        self.assertTrue(str(PARENT_ADAPTER).endswith("checkpoint-1500"))
        self.assertEqual(len(PARENT_ADAPTER_SHA256), 64)
        self.assertTrue(RUN_ID_PREFIX.startswith("GR-REC-THINK-SUFFIX-SID"))


if __name__ == "__main__":
    unittest.main()
