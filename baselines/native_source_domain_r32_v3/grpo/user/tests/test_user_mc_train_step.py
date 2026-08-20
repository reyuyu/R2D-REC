import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_policy import compute_mc_model_loss, get_mc_completion_logps  # noqa: E402
from user_mc_train_step import MCTrainStepError, mc_optimizer_step  # noqa: E402


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=17, hidden_size=8, max_positions=32):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = torch.nn.Embedding(max_positions, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=None,
    ):
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        logits = self.output(hidden)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class FiniteForwardNanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class NanGradientTinyCausalLM(TinyCausalLM):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return SimpleNamespace(logits=FiniteForwardNanBackward.apply(outputs.logits))


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def batch(prompt_ids=None, completion_ids=None):
    prompt_ids = prompt_ids if prompt_ids is not None else torch.tensor([[1, 2]])
    completion_ids = (
        completion_ids if completion_ids is not None else torch.tensor([[3, 4, 5]])
    )
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": torch.ones_like(completion_ids),
    }


def unit(delta, indices=(1,)):
    return {"delta": delta, "generated_token_indices": list(indices)}


def target_probability(model, inputs, token_index=1):
    with torch.no_grad():
        logps = get_mc_completion_logps(model, **inputs)
    return float(logps[0, token_index].exp())


class MCOptimizerStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260821)

    def test_positive_credit_increases_target_probability(self):
        model = TinyCausalLM()
        inputs = batch()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        before = target_probability(model, inputs)
        result = mc_optimizer_step(model, optimizer, inputs, [[unit(0.5)]])
        after = target_probability(model, inputs)
        self.assertGreater(after, before)
        self.assertGreater(result["grad_norm"], 0.0)
        self.assertTrue(result["finite"])

    def test_negative_credit_decreases_target_probability(self):
        model = TinyCausalLM()
        inputs = batch()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        before = target_probability(model, inputs)
        result = mc_optimizer_step(model, optimizer, inputs, [[unit(-0.5)]])
        after = target_probability(model, inputs)
        self.assertLess(after, before)
        self.assertGreater(result["grad_norm"], 0.0)

    def test_zero_credit_changes_no_parameters(self):
        model = TinyCausalLM()
        result = mc_optimizer_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.5),
            batch(),
            [[unit(0.0)]],
        )
        self.assertEqual(result["loss"], 0.0)
        self.assertEqual(result["grad_norm"], 0.0)
        self.assertEqual(result["parameter_delta_l2"], 0.0)
        self.assertEqual(result["parameter_delta_max_abs"], 0.0)

    def test_empty_candidate_changes_no_parameters(self):
        model = TinyCausalLM()
        result = mc_optimizer_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.5),
            batch(),
            [[]],
        )
        self.assertEqual(result["loss"], 0.0)
        self.assertEqual(result["grad_norm"], 0.0)
        self.assertEqual(result["parameter_delta_l2"], 0.0)
        self.assertEqual(result["parameter_delta_max_abs"], 0.0)

    def test_k2_candidate_losses_are_meaned_without_normalization(self):
        model = TinyCausalLM()
        inputs = batch(
            prompt_ids=torch.tensor([[1, 2], [1, 2]]),
            completion_ids=torch.tensor([[3, 4, 5], [6, 7, 8]]),
        )
        units = [[unit(0.5)], [unit(-0.5)]]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        _, probe_metadata, probe_logps = compute_mc_model_loss(model, inputs, units)
        positive_direct = torch.autograd.grad(
            probe_metadata["candidate_losses"][0], probe_logps, retain_graph=True
        )[0]
        negative_direct = torch.autograd.grad(
            probe_metadata["candidate_losses"][1], probe_logps
        )[0]
        self.assertGreater(float(positive_direct[0].abs().sum()), 0.0)
        self.assertEqual(float(positive_direct[1].abs().sum()), 0.0)
        self.assertEqual(float(negative_direct[0].abs().sum()), 0.0)
        self.assertGreater(float(negative_direct[1].abs().sum()), 0.0)

        result = mc_optimizer_step(model, optimizer, inputs, units)
        candidate_losses = result["objective_metadata"]["candidate_losses"]
        self.assertEqual(candidate_losses.shape, (2,))
        self.assertAlmostEqual(result["loss"], float(candidate_losses.mean()), places=7)
        self.assertEqual(result["objective_metadata"]["positive_unit_count"], 1)
        self.assertEqual(result["objective_metadata"]["negative_unit_count"], 1)
        self.assertGreater(result["grad_norm"], 0.0)

    def test_manual_optimizer_step_matches_bridge(self):
        manual_model = TinyCausalLM()
        bridge_model = TinyCausalLM()
        bridge_model.load_state_dict(manual_model.state_dict())
        inputs = batch(
            prompt_ids=torch.tensor([[1, 2], [3, 4]]),
            completion_ids=torch.tensor([[5, 6, 7], [8, 9, 10]]),
        )
        units = [[unit(0.3, (0, 1))], [unit(-0.2, (2,))]]
        manual_optimizer = torch.optim.SGD(manual_model.parameters(), lr=0.1)
        bridge_optimizer = torch.optim.SGD(bridge_model.parameters(), lr=0.1)

        manual_optimizer.zero_grad(set_to_none=True)
        manual_loss, _, _ = compute_mc_model_loss(
            manual_model, inputs, units, forward_batch_size=1
        )
        manual_loss.backward()
        manual_optimizer.step()
        mc_optimizer_step(
            bridge_model,
            bridge_optimizer,
            inputs,
            units,
            forward_batch_size=1,
        )
        max_error = max(
            float((manual - bridge).abs().max())
            for manual, bridge in zip(
                manual_model.parameters(), bridge_model.parameters()
            )
        )
        self.assertLessEqual(max_error, 1e-7)

    def test_nonfinite_loss_does_not_step(self):
        model = TinyCausalLM()
        with torch.no_grad():
            model.output.weight.fill_(float("nan"))
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        with self.assertRaisesRegex(MCTrainStepError, "loss"):
            mc_optimizer_step(model, optimizer, batch(), [[unit(0.5)]])
        self.assertEqual(optimizer.step_calls, 0)
        for initial, parameter in zip(before, model.parameters()):
            unchanged = (initial == parameter) | (
                torch.isnan(initial) & torch.isnan(parameter)
            )
            self.assertTrue(bool(unchanged.all()))

    def test_nonfinite_gradient_does_not_step(self):
        model = NanGradientTinyCausalLM()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        with self.assertRaisesRegex(MCTrainStepError, "gradient"):
            mc_optimizer_step(model, optimizer, batch(), [[unit(0.5)]])
        self.assertEqual(optimizer.step_calls, 0)
        for initial, parameter in zip(before, model.parameters()):
            self.assertTrue(torch.equal(initial, parameter))


if __name__ == "__main__":
    unittest.main()
