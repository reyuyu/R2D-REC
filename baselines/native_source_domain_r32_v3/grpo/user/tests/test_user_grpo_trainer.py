import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from user_generated_token_projection import (
    GeneratedTokenProjectionError,
    project_compiled_mask_to_generated,
)
from user_grpo_trainer import UserGRPOTrainer, prepare_scored_rollout
from user_training_objective import standard_sequence_grpo_loss_from_logps


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=19, hidden_size=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None, **kwargs):
        logits = self.head(self.embed(input_ids))
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def trainer_inputs():
    prompt_ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    prompt_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    completion_ids = torch.tensor([[6, 7, 8, 9], [10, 11, 12, 0]])
    completion_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)
    advantages = torch.tensor([0.7, -0.4])
    return prompt_ids, prompt_mask, completion_ids, completion_mask, advantages


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


class FakeScore:
    def __init__(self, reward, route):
        self.reward = reward
        self.violations = []
        if route == "action":
            self.f1 = reward
            self.precision = reward
            self.recall = reward
        else:
            self.action_f1 = reward
            self.logic_f1 = reward


def fake_compiler(completion, violations, tokenizer, route):
    count = len(completion)
    return {
        "input_ids": list(range(count)),
        "token_count": count,
        "records": [],
        "per_kind_masks": {},
        "penalty_mask": [False] * count,
        "masked_token_count": 0,
        "masked_token_fraction": 0.0,
    }


class UserGRPOTrainerTests(unittest.TestCase):
    def test_real_trainer_no_mask_model_gradient_parity(self):
        torch.manual_seed(3)
        user_model = TinyCausalLM()
        standard_model = TinyCausalLM()
        standard_model.load_state_dict(user_model.state_dict())
        user_trainer = UserGRPOTrainer.for_correctness_smoke(user_model)
        standard_trainer = UserGRPOTrainer.for_correctness_smoke(standard_model)
        prompt_ids, prompt_mask, completion_ids, completion_mask, advantages = trainer_inputs()
        with torch.no_grad():
            combined = torch.cat([prompt_ids, completion_ids], dim=1)
            attention = torch.cat([prompt_mask, completion_mask], dim=1)
            old = user_trainer._get_user_per_token_logps(
                user_model, combined, attention, completion_ids.size(1)
            )
        token_advantages = advantages[:, None].expand_as(completion_mask).clone()
        inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "token_advantages": token_advantages,
            "old_per_token_logps": old,
        }
        user_loss = user_trainer._compute_loss(user_model, inputs)
        standard_logps = standard_trainer._get_user_per_token_logps(
            standard_model, combined, attention, completion_ids.size(1)
        )
        standard_loss, _, _ = standard_sequence_grpo_loss_from_logps(
            standard_logps, old, advantages, completion_mask
        )
        user_loss.backward()
        standard_loss.backward()
        self.assertLessEqual(abs(float(user_loss - standard_loss)), 1e-7)
        for user_parameter, standard_parameter in zip(
            user_model.parameters(), standard_model.parameters()
        ):
            self.assertLessEqual(
                float((user_parameter.grad - standard_parameter.grad).abs().max()), 1e-7
            )

    def test_projection_shape_and_partial_replacement_fail_closed(self):
        base = {
            "input_ids": [1, 20, 3],
            "token_count": 3,
            "records": [
                {
                    "included": True,
                    "masked_token_indices": [1],
                    "token_spans": [{"token_start": 1, "token_end": 2, "token_ids": [20]}],
                }
            ],
            "per_kind_masks": {"hallucinated_sid": [False, True, False]},
            "penalty_mask": [False, True, False],
            "masked_token_count": 1,
            "masked_token_fraction": 1 / 3,
        }
        projected = project_compiled_mask_to_generated(base, [1, 21, 22, 3])
        self.assertEqual(projected["token_count"], 4)
        self.assertEqual(len(projected["penalty_mask"]), 4)
        unsafe = {**base, "input_ids": [1, 20, 30, 3]}
        unsafe["records"] = [
            {
                "included": True,
                "masked_token_indices": [1],
                "token_spans": [{"token_start": 1, "token_end": 2, "token_ids": [20]}],
            }
        ]
        unsafe["per_kind_masks"] = {"hallucinated_sid": [False, True, False, False]}
        with self.assertRaises(GeneratedTokenProjectionError):
            project_compiled_mask_to_generated(unsafe, [1, 99, 3])

    def test_action_and_chain_route_to_existing_scorers(self):
        tokenizer = FakeTokenizer()
        identity_projection = lambda compiled, ids, **kwargs: {
            **compiled,
            "input_ids": list(ids),
            "token_count": len(ids),
        }
        for route, scorer_name in (("action", "score_action"), ("chain", "score_chain")):
            rows = [{"sample_id": f"{route}-sample", "route": route}]
            completions = [[1, 2], [3, 4], [5, 6], [7, 8]]
            scores = [FakeScore(value, route) for value in (0.0, 0.2, 0.6, 1.0)]
            with (
                patch("user_grpo_trainer.compile_penalty_mask", side_effect=fake_compiler),
                patch("user_grpo_trainer.project_compiled_mask_to_generated", side_effect=identity_projection),
                patch(f"user_grpo_trainer.{scorer_name}", side_effect=scores) as scorer,
            ):
                output = prepare_scored_rollout(rows, completions, tokenizer)
            self.assertEqual(scorer.call_count, 4)
            self.assertEqual(output["route"], route)
            self.assertEqual(output["token_advantages"].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
