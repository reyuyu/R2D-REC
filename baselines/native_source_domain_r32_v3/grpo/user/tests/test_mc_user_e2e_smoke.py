import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_e2e_smoke import (  # noqa: E402
    FORWARD_BATCH_SIZE,
    K,
    LEARNING_RATE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    cleanup_generation_cache,
    restore_identical_route_start,
    run_cli,
    set_generation_mode,
    set_training_mode,
    validate_update_contract,
)
from run_mc_user_generation_smoke import (  # noqa: E402
    gpu_preflight,
    validate_exact_policy_ids,
)
from user_mc_train_step import mc_optimizer_step  # noqa: E402


class ModeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.config = SimpleNamespace(use_cache=False)
        self.gc_enable_kwargs = None
        self.gc_disable_calls = 0

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gc_enable_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.gc_disable_calls += 1


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.output = torch.nn.Linear(8, 32, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=None,
    ):
        logits = self.output(self.embedding(input_ids))
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def tiny_k2_batch():
    prompt_ids = torch.tensor([[1, 2], [1, 2]])
    completion_ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": torch.ones_like(completion_ids, dtype=torch.float32),
    }


def active_update():
    return {
        "active_unit_count": 2,
        "finite": True,
        "loss": 0.5,
        "grad_norm": 1.2,
        "skipped_update": False,
        "optimizer_step_performed": True,
    }


def skipped_update():
    return {
        "active_unit_count": 0,
        "finite": True,
        "loss": 0.0,
        "grad_norm": 0.0,
        "skipped_update": True,
        "optimizer_step_performed": False,
    }


class ModeAndCacheTests(unittest.TestCase):
    def test_generation_to_training_mode_switch(self):
        model = ModeModel()
        model.train()
        set_generation_mode(model)
        self.assertFalse(model.training)
        self.assertTrue(model.config.use_cache)
        self.assertEqual(model.gc_disable_calls, 1)
        set_training_mode(model)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(model.gc_enable_kwargs, {"use_reentrant": False})

    def test_cache_cleanup_calls_collect_and_empty_then_records_memory(self):
        calls = []
        result = cleanup_generation_cache(
            torch.device("cpu"),
            collect_fn=lambda: calls.append("collect"),
            empty_cache_fn=lambda: calls.append("empty"),
            allocated_fn=lambda _device: 3 * 1024 * 1024,
            reserved_fn=lambda _device: 5 * 1024 * 1024,
        )
        self.assertEqual(calls, ["collect", "empty"])
        self.assertEqual(result["allocated_mib_after_cleanup"], 3.0)
        self.assertEqual(result["reserved_mib_after_cleanup"], 5.0)


class UpdateContractTests(unittest.TestCase):
    def test_real_tiny_active_credit_executes_one_adamw_step(self):
        model = TinyCausalLM()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.0
        )
        result = mc_optimizer_step(
            model,
            optimizer,
            tiny_k2_batch(),
            [[{"delta": 0.5, "generated_token_indices": [1]}], []],
            forward_batch_size=1,
        )
        self.assertTrue(result["optimizer_step_performed"])
        self.assertFalse(result["skipped_update"])
        self.assertGreater(result["parameter_delta_l2"], 0.0)
        self.assertTrue(optimizer.state)

    def test_real_tiny_no_credit_skips_without_optimizer_state(self):
        model = TinyCausalLM()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.0
        )
        result = mc_optimizer_step(
            model,
            optimizer,
            tiny_k2_batch(),
            [[], []],
            forward_batch_size=1,
        )
        self.assertTrue(result["skipped_update"])
        self.assertFalse(result["optimizer_step_performed"])
        self.assertEqual(result["parameter_delta_l2"], 0.0)
        self.assertFalse(optimizer.state)

    def test_active_credit_requires_real_lora_only_step_and_frozen_base(self):
        result = validate_update_contract(
            active_update(),
            base_hash_before="base",
            base_hash_after="base",
            lora_hash_before="lora0",
            lora_hash_after="lora1",
            lora_delta_l2=0.2,
            lora_delta_max_abs=0.01,
            optimizer_state_lora_only=True,
            optimizer_state_empty=False,
        )
        self.assertTrue(result["active_credit"])
        self.assertTrue(result["base_hash_unchanged"])

    def test_no_credit_skips_step_and_preserves_optimizer_and_parameters(self):
        result = validate_update_contract(
            skipped_update(),
            base_hash_before="base",
            base_hash_after="base",
            lora_hash_before="lora0",
            lora_hash_after="lora0",
            lora_delta_l2=0.0,
            lora_delta_max_abs=0.0,
            optimizer_state_lora_only=False,
            optimizer_state_empty=True,
        )
        self.assertFalse(result["active_credit"])

    def test_base_change_fails_for_active_and_no_credit(self):
        for update in (active_update(), skipped_update()):
            with self.subTest(active=update["active_unit_count"]), self.assertRaisesRegex(
                RuntimeError, "base"
            ):
                validate_update_contract(
                    update,
                    base_hash_before="base0",
                    base_hash_after="base1",
                    lora_hash_before="lora0",
                    lora_hash_after="lora1",
                    lora_delta_l2=0.2,
                    lora_delta_max_abs=0.01,
                    optimizer_state_lora_only=True,
                    optimizer_state_empty=False,
                )


class RouteParityAndIDTests(unittest.TestCase):
    def test_action_and_chain_restore_same_lora_snapshot(self):
        model = ModeModel()
        initial = {"weight": torch.tensor([2.0])}
        seen = []

        def restore(target, snapshot):
            target.weight.data.copy_(snapshot["weight"])
            seen.append(float(target.weight))

        def hash_fn(target, lora):
            return ("initial" if float(target.weight) == 2.0 else "drift", 1)

        for _route in ("action", "chain"):
            model.weight.data.fill_(9.0)
            restore_identical_route_start(
                model,
                initial,
                "initial",
                restore_fn=restore,
                hash_fn=hash_fn,
            )
        self.assertEqual(seen, [2.0, 2.0])

    def test_exact_generated_ids_survive_policy_bridge(self):
        generated = [[11, 12], [21, 22, 23]]
        completion_ids = torch.tensor([[11, 12, 0], [21, 22, 23]])
        completion_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
        batch = {
            "candidate_count": K,
            "completion_lengths": [2, 3],
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "credit_units_per_candidate": [
                [{"generated_token_indices": [0, 1]}],
                [{"generated_token_indices": [2]}],
            ],
        }
        validate_exact_policy_ids(batch, generated)

    def test_frozen_generation_and_optimizer_contract_constants(self):
        self.assertEqual(K, 2)
        self.assertEqual(TEMPERATURE, 0.9)
        self.assertEqual(TOP_P, 0.95)
        self.assertEqual(MAX_NEW_TOKENS, 512)
        self.assertEqual(LEARNING_RATE, 1e-6)
        self.assertEqual(FORWARD_BATCH_SIZE, 1)


class SafetyGateTests(unittest.TestCase):
    def test_without_execute_never_calls_e2e_execution(self):
        preflight = Mock(
            return_value={
                "status": "READY_TO_EXECUTE",
                "gpu": {"index": 0},
                "selected_sample_ids": {"action": "a", "chain": "c"},
                "prompt_token_counts": {"action": 1, "chain": 1},
                "paths": {"train_sha256": "sha"},
            }
        )
        execute = Mock()
        output = run_cli(
            ["--gpu-id", "0"], preflight_fn=preflight, execute_fn=execute
        )
        self.assertEqual(output["status"], "READY_TO_EXECUTE")
        execute.assert_not_called()

    def test_execute_gate_calls_once_only_when_explicit(self):
        preflight = Mock(return_value={"status": "READY_TO_EXECUTE"})
        execute = Mock(return_value={"status": "PASS"})
        output = run_cli(
            ["--gpu-id", "0", "--execute"],
            preflight_fn=preflight,
            execute_fn=execute,
        )
        self.assertEqual(output["status"], "PASS")
        execute.assert_called_once()

    def test_busy_gpu_is_refused(self):
        responses = [
            SimpleNamespace(stdout="0, GPU-0, 100, 80000, 0\n"),
            SimpleNamespace(stdout="GPU-0, 123, python, 100\n"),
        ]
        with self.assertRaisesRegex(RuntimeError, "busy"):
            gpu_preflight(0, run_command=Mock(side_effect=responses))

    def test_runner_contains_no_loop_checkpoint_ddp_or_clipping(self):
        source = (SCRIPTS_DIR / "run_mc_user_e2e_smoke.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("clip_grad", source)
        self.assertNotIn("DistributedDataParallel", source)
        self.assertNotIn("save_pretrained", source)
        self.assertNotIn("for step in", source)


if __name__ == "__main__":
    unittest.main()
