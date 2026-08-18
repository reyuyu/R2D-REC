"""CPU gradient, reward parity, and baseline loss parity tests."""
import math
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from grpo_sid import q_reward
from grpo_trl_trainer import RecGRPOTrainer, group_advantages_population, make_nothink_reward_func
from gr_rec_dsr_v1.dsr_objectives import build_nothink_rescue_plan, nothink_unlikelihood_loss
from gr_rec_dsr_v1.dsr_runtime import (
    get_capture,
    make_dsr_nothink_reward_func,
    make_dsr_think_reward_func,
    reset_global_capture,
)
from gr_rec_dsr_v1.dsr_trainer import DsrGRPOTrainer


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(31, 8)
        self.head = torch.nn.Linear(8, 31, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


class FakeSemanticTokenizer:
    def __init__(self):
        tokens = ["<|prod_begin|>", "<s_a_1>", "<s_a_8>", "<s_b_2>", "<s_b_9>", "<s_c_3>", "<s_c_9>"]
        self.vocab = {token: index + 1 for index, token in enumerate(tokens)}
        self.inverse = {value: key for key, value in self.vocab.items()}

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [1000 + ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inverse[value] if value in self.inverse else chr(value - 1000) for value in ids)


def make_loss_trainer(cls):
    trainer = cls.__new__(cls)
    trainer.temperature = 0.9
    trainer.model_kwarg_keys = set()
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.args = SimpleNamespace(delta=None, report_to=[])
    trainer.loss_type = "grpo"
    trainer.current_gradient_accumulation_steps = 2
    trainer._smoke_policy_epoch = {}
    trainer._smoke_rollout_id = 1
    trainer._smoke_log = [{"route": "think"}]
    trainer._detailed_monitor = False
    trainer.state = SimpleNamespace(global_step=0)
    if cls is DsrGRPOTrainer:
        trainer.dsr_think_lambda = 0.0
        trainer.dsr_nothink_scale = 0.0
    upstream = RecGRPOTrainer._get_per_token_logps_and_entropies

    def get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
        return upstream(self, model, input_ids, attention_mask, logits_to_keep, compute_entropy=False)

    trainer._get_per_token_logps_and_entropies = MethodType(get_logps, trainer)
    return trainer


def parity_inputs():
    return {
        "prompt_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "prompt_mask": torch.ones(2, 3, dtype=torch.long),
        "completion_ids": torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14]]),
        "completion_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "advantages": torch.tensor([0.75, -0.5]),
        "old_per_token_logps": torch.tensor([[-2.0, -2.1, -2.2, -2.3], [-2.4, -2.5, -2.6, -2.7]]),
        "route_id": torch.zeros(2, dtype=torch.long),
        "dsr_aux_advantages": torch.zeros(2),
        "dsr_sa_positions": torch.full((2,), -1, dtype=torch.long),
        "dsr_frequency_weights": torch.zeros(2),
        "dsr_rescue_coefficients": torch.zeros(2),
    }


def test_baseline_loss_and_gradient_exact_parity_when_disabled():
    torch.manual_seed(20260818)
    baseline_model = TinyCausalLM()
    dsr_model = TinyCausalLM()
    dsr_model.load_state_dict(baseline_model.state_dict())
    baseline = make_loss_trainer(RecGRPOTrainer)
    dsr = make_loss_trainer(DsrGRPOTrainer)
    inputs = parity_inputs()
    baseline_loss = baseline._compute_loss(baseline_model, inputs)
    dsr_loss = dsr._compute_loss(dsr_model, inputs)
    baseline_loss.backward()
    dsr_loss.backward()
    assert torch.equal(baseline_loss, dsr_loss)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(baseline_model.named_parameters(), dsr_model.named_parameters()):
        assert name_a == name_b and torch.equal(parameter_a.grad, parameter_b.grad)


def test_primary_reward_and_advantage_parity():
    gold = {("prod", 1, 2, 3)}
    predictions = [("prod", 1, 2, 3), ("prod", 1, 2, 4), ("prod", 1, 9, 9), ("prod", 8, 9, 9)]
    rewards = [q_reward(value, gold) for value in predictions]
    assert rewards == [8.0, 2.0, 0.5, 0.0]
    expected = group_advantages_population(rewards, 4)
    observed = group_advantages_population(list(rewards), 4)
    assert torch.equal(expected, observed)


def test_reward_wrappers_return_primary_exactly():
    tokenizer = FakeSemanticTokenizer()
    completion = [tokenizer.vocab[token] for token in ("<|prod_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>")]
    kwargs = {
        "all_gold_sids": [["<|prod_begin|><s_a_1><s_b_2><s_c_3>"]],
        "route": ["no_think"],
        "recommendation_group_id": ["g"],
        "target_domain": ["prod"],
    }
    reset_global_capture()
    baseline = make_nothink_reward_func(tokenizer)(["p"], ["c"], [completion], **kwargs)
    dsr = make_dsr_nothink_reward_func(tokenizer)(["p"], ["c"], [completion], **kwargs)
    assert dsr == baseline == [8.0]

    reset_global_capture()
    capture = get_capture(tokenizer)

    def fake_beam(prompts, completions, completion_ids, gold_sets):
        capture.pending_beam_results = [{
            "beam_sids": [("prod", 1, 2, 3)] * 32,
            "exact": 1,
            "ab": 0,
            "a": 0,
            "invalid": 0,
        }] * len(prompts)
        return [8.0] * len(prompts)

    think_kwargs = {
        "all_gold_sids": [["<|prod_begin|><s_a_1><s_b_2><s_c_3>"]],
        "route": ["think"],
        "recommendation_group_id": ["g"],
        "target_domain": ["prod"],
    }
    assert make_dsr_think_reward_func(fake_beam)(["prompt"], ["cot"], [[]], **think_kwargs) == [8.0]


def test_nothink_activation_rules_and_coefficients():
    same = build_nothink_rescue_plan([0] * 8, [17] * 8, [1] * 8, {1, 2})
    split = build_nothink_rescue_plan([0] * 8, [17] * 4 + [28] * 4, [1] * 8, {1, 2, 3})
    diverse = build_nothink_rescue_plan([0] * 8, list(range(8)), [1] * 8, {1, 2})
    nonzero = build_nothink_rescue_plan([0] * 7 + [0.5], [17] * 8, [1] * 8, {1, 2})
    assert same.active and same.concentration == 1.0 and same.lambda_a == 0.10
    assert split.active and split.concentration == 0.5 and split.lambda_a == 0.20
    assert diverse.active and diverse.concentration == 0.125
    assert not nonzero.active and nonzero.coefficient == 0.0


def test_sa_only_gradient_and_primary_zero_rescue_gradient():
    logps = torch.full((8, 5), -2.0, requires_grad=True)
    positions = torch.tensor([2] * 8)
    weights = torch.ones(8)
    coefficients = torch.full((8,), 0.10)
    loss = nothink_unlikelihood_loss(logps, positions, weights, coefficients)
    assert float(loss) > 0.0
    loss.backward()
    assert torch.count_nonzero(logps.grad[:, 2]) == 8
    assert torch.count_nonzero(logps.grad[:, [0, 1, 3, 4]]) == 0


def test_dsr_step_log_uses_existing_python_scalars():
    class Monitor:
        def __init__(self):
            self.rows = []

        def _append(self, filename, row):
            self.rows.append((filename, row))

    trainer = DsrGRPOTrainer.__new__(DsrGRPOTrainer)
    trainer._monitor = Monitor()
    trainer.accelerator = SimpleNamespace(process_index=0)
    trainer.state = SimpleNamespace(global_step=9)
    trainer._smoke_rollout_id = 4
    trainer._smoke_policy_epoch = {4: 2}
    trainer._smoke_log = [{
        "rollout_id": 4,
        "route": "think",
        "primary_loss_ep2": 0.4,
        "think_aux_loss_ep2": 0.3,
        "nothink_rescue_loss_ep2": 0.0,
        "dsr_total_loss_ep2": 0.43,
        "ratio_mean_ep2": 1.01,
        "clip_fraction_ep2": 0.02,
        "approx_kl_ep2": 0.003,
    }]
    trainer.dsr_think_lambda = 0.10
    trainer._monitor_enabled = MethodType(lambda self: True, trainer)
    with patch.object(RecGRPOTrainer, "log", return_value="baseline"):
        assert trainer.log({"loss": 0.43, "grad_norm": 0.7}) == "baseline"
    filename, row = trainer._monitor.rows[0]
    assert filename == "dsr_steps.jsonl"
    assert row["think_aux_loss_raw"] == 0.3
    assert math.isclose(row["think_aux_contribution"], 0.03)
    assert row["dsr_total_loss"] == 0.43
    assert row["grad_norm"] == 0.7


if __name__ == "__main__":
    from gr_rec_dsr_v1.run_cpu_tests import main
    raise SystemExit(main([__name__]))
