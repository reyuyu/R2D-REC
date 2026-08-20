import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_objective import mc_unit_credit_loss  # noqa: E402
from user_mc_policy import (  # noqa: E402
    MC_TEMPERATURE,
    compute_mc_model_loss,
    get_mc_completion_logps,
)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=17, hidden_size=8, max_positions=32):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = torch.nn.Embedding(max_positions, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.last_logits = None

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        logits = self.output(hidden)
        if logits.requires_grad:
            logits.retain_grad()
        self.last_logits = logits
        return SimpleNamespace(logits=logits)


class KeepingTinyCausalLM(TinyCausalLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keep_calls = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        self.keep_calls.append(logits_to_keep)
        if logits_to_keep is None:
            return outputs
        return SimpleNamespace(logits=outputs.logits[:, -logits_to_keep:, :])


class SilentlyIgnoringTinyCausalLM(TinyCausalLM):
    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        return super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=kwargs.get("use_cache", False),
        )


def batch(prompt_ids=None, prompt_mask=None, completion_ids=None, completion_mask=None):
    prompt_ids = prompt_ids if prompt_ids is not None else torch.tensor([[1, 2]])
    prompt_mask = prompt_mask if prompt_mask is not None else torch.ones_like(prompt_ids)
    completion_ids = completion_ids if completion_ids is not None else torch.tensor([[3, 4, 5]])
    completion_mask = (
        completion_mask if completion_mask is not None else torch.ones_like(completion_ids)
    )
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
    }


def credit(delta, indices):
    return {"delta": delta, "generated_token_indices": list(indices)}


class MCPolicyCorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260821)

    def test_causal_shift_matches_manual_full_logits_gather_exactly(self):
        model = TinyCausalLM()
        inputs = batch(
            prompt_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            prompt_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            completion_ids=torch.tensor([[6, 7], [8, 9]]),
            completion_mask=torch.ones((2, 2), dtype=torch.long),
        )
        actual = get_mc_completion_logps(model, **inputs)

        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        positions = attention.cumsum(-1) - 1
        positions.masked_fill_(attention == 0, 0)
        full_logits = model(input_ids, attention_mask=attention, position_ids=positions).logits
        shifted = full_logits[:, :-1, :][:, -inputs["completion_ids"].size(1) :, :]
        expected = torch.gather(
            torch.log_softmax(shifted / MC_TEMPERATURE, dim=-1),
            2,
            inputs["completion_ids"].unsqueeze(-1),
        ).squeeze(-1)
        self.assertTrue(torch.equal(actual, expected))

    def test_left_padding_preserves_valid_completion_logps_exactly(self):
        model = KeepingTinyCausalLM()
        completion_ids = torch.tensor([[5, 6, 7]])
        completion_mask = torch.ones_like(completion_ids)
        unpadded = get_mc_completion_logps(
            model,
            torch.tensor([[3, 4]]),
            torch.tensor([[1, 1]]),
            completion_ids,
            completion_mask,
        )
        padded = get_mc_completion_logps(
            model,
            torch.tensor([[0, 0, 3, 4]]),
            torch.tensor([[0, 0, 1, 1]]),
            completion_ids,
            completion_mask,
        )
        self.assertTrue(torch.equal(unpadded, padded))

    def test_supported_model_receives_completion_length_plus_one(self):
        model = KeepingTinyCausalLM()
        inputs = batch()
        get_mc_completion_logps(model, **inputs)
        self.assertEqual(model.keep_calls, [inputs["completion_ids"].size(1) + 1])

    def test_kwargs_model_cannot_silently_ignore_logits_to_keep(self):
        with self.assertRaisesRegex(ValueError, "kept"):
            get_mc_completion_logps(SilentlyIgnoringTinyCausalLM(), **batch())

    def test_memory_safe_and_full_logits_logps_match_exactly(self):
        full_model = TinyCausalLM()
        kept_model = KeepingTinyCausalLM()
        kept_model.load_state_dict(full_model.state_dict())
        inputs = batch(
            prompt_ids=torch.tensor([[1, 2], [3, 4]]),
            completion_ids=torch.tensor([[5, 6, 7], [8, 9, 10]]),
        )
        full = get_mc_completion_logps(
            full_model, **inputs, forward_batch_size=inputs["prompt_ids"].size(0)
        )
        kept = get_mc_completion_logps(kept_model, **inputs, forward_batch_size=1)
        self.assertTrue(torch.equal(full, kept))

    def test_forward_batch_size_one_matches_full_batch_exactly(self):
        model = KeepingTinyCausalLM()
        inputs = batch(
            prompt_ids=torch.tensor([[1, 2], [3, 4], [5, 6]]),
            completion_ids=torch.tensor([[7, 8], [9, 10], [11, 12]]),
        )
        chunked = get_mc_completion_logps(model, **inputs, forward_batch_size=1)
        full_batch = get_mc_completion_logps(model, **inputs, forward_batch_size=3)
        self.assertTrue(torch.equal(chunked, full_batch))
        self.assertEqual(model.keep_calls, [3, 3, 3, 3])

    def test_memory_safe_and_full_loss_and_parameter_gradients_match_exactly(self):
        full_model = TinyCausalLM()
        kept_model = KeepingTinyCausalLM()
        kept_model.load_state_dict(full_model.state_dict())
        inputs = batch(
            prompt_ids=torch.tensor([[1, 2], [3, 4]]),
            completion_ids=torch.tensor([[5, 6, 7], [8, 9, 10]]),
        )
        units = [[credit(0.3, [0, 1])], [credit(-0.2, [2])]]
        full_loss, _, _ = compute_mc_model_loss(
            full_model, inputs, units, forward_batch_size=2
        )
        kept_loss, _, _ = compute_mc_model_loss(
            kept_model, inputs, units, forward_batch_size=1
        )
        self.assertTrue(torch.equal(full_loss, kept_loss))
        full_loss.backward()
        kept_loss.backward()
        gradient_max_error = 0.0
        for full_parameter, kept_parameter in zip(
            full_model.parameters(), kept_model.parameters()
        ):
            gradient_max_error = max(
                gradient_max_error,
                float((full_parameter.grad - kept_parameter.grad).abs().max()),
            )
        self.assertLessEqual(gradient_max_error, torch.finfo(torch.float32).eps)

    def test_positive_credit_direct_signal_raises_target_probability(self):
        model = TinyCausalLM()
        inputs = batch()
        loss, _, logps = compute_mc_model_loss(model, inputs, [[credit(0.5, [1])]])
        logps.retain_grad()
        loss.backward()
        self.assertLess(float(logps.grad[0, 1]), 0.0)
        predictor_position = inputs["prompt_ids"].size(1)
        target_id = int(inputs["completion_ids"][0, 1])
        self.assertLess(float(model.last_logits.grad[0, predictor_position, target_id]), 0.0)

    def test_negative_credit_direct_signal_lowers_target_probability(self):
        model = TinyCausalLM()
        inputs = batch()
        loss, _, logps = compute_mc_model_loss(model, inputs, [[credit(-0.5, [1])]])
        logps.retain_grad()
        loss.backward()
        self.assertGreater(float(logps.grad[0, 1]), 0.0)
        predictor_position = inputs["prompt_ids"].size(1)
        target_id = int(inputs["completion_ids"][0, 1])
        self.assertGreater(float(model.last_logits.grad[0, predictor_position, target_id]), 0.0)

    def test_zero_credit_has_zero_direct_position_signal(self):
        model = TinyCausalLM()
        inputs = batch()
        loss, _, logps = compute_mc_model_loss(model, inputs, [[credit(0.0, [1])]])
        logps.retain_grad()
        loss.backward()
        self.assertTrue(torch.equal(logps.grad, torch.zeros_like(logps)))

    def test_uncredited_positions_have_zero_direct_signal(self):
        model = TinyCausalLM()
        inputs = batch()
        loss, _, logps = compute_mc_model_loss(model, inputs, [[credit(0.5, [1])]])
        logps.retain_grad()
        loss.backward()
        self.assertEqual(float(logps.grad[0, 0]), 0.0)
        self.assertEqual(float(logps.grad[0, 2]), 0.0)
        self.assertNotEqual(float(logps.grad[0, 1]), 0.0)

    def test_bridge_loss_matches_direct_objective_exactly(self):
        model = TinyCausalLM()
        inputs = batch(
            prompt_ids=torch.tensor([[1, 2], [3, 4]]),
            completion_ids=torch.tensor([[5, 6, 7], [8, 9, 10]]),
        )
        units = [[credit(0.3, [0, 1])], [credit(-0.2, [2])]]
        bridge_loss, bridge_metadata, bridge_logps = compute_mc_model_loss(
            model, inputs, units
        )
        direct_logps = get_mc_completion_logps(model, **inputs)
        direct_loss, direct_metadata = mc_unit_credit_loss(
            direct_logps, units, inputs["completion_mask"]
        )
        self.assertTrue(torch.equal(bridge_logps, direct_logps))
        self.assertTrue(torch.equal(bridge_loss, direct_loss))
        self.assertTrue(
            torch.equal(
                bridge_metadata["candidate_losses"],
                direct_metadata["candidate_losses"],
            )
        )

    def test_temperature_is_frozen_at_point_nine_in_bridge(self):
        model = TinyCausalLM()
        inputs = batch()
        _, _, bridge_logps = compute_mc_model_loss(model, inputs, [[credit(0.5, [0])]])
        direct_logps = get_mc_completion_logps(model, **inputs, temperature=0.9)
        self.assertEqual(MC_TEMPERATURE, 0.9)
        self.assertTrue(torch.equal(bridge_logps, direct_logps))


if __name__ == "__main__":
    unittest.main()
