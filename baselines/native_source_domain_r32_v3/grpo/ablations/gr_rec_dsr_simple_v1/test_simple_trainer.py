"""Loss, gradient, primary-reward, and NoThink reuse tests."""
import torch

from grpo_trl_trainer import RecGRPOTrainer, make_nothink_reward_func
from gr_rec_dsr_v1.dsr_objectives import (
    build_nothink_rescue_plan,
    nothink_unlikelihood_loss,
)
from gr_rec_dsr_v1.dsr_runtime import make_dsr_nothink_reward_func
from gr_rec_dsr_v1.dsr_trainer import DsrGRPOTrainer
from gr_rec_dsr_v1.test_dsr_trainer import (
    FakeSemanticTokenizer,
    TinyCausalLM,
    make_loss_trainer,
    parity_inputs,
)

from .simple_trainer import SimpleDsrGRPOTrainer


def _simple_loss_trainer(think_lambda=0.0, nothink_scale=0.0):
    trainer = make_loss_trainer(SimpleDsrGRPOTrainer)
    trainer.dsr_think_lambda = think_lambda
    trainer.dsr_nothink_scale = nothink_scale
    return trainer


def test_baseline_loss_and_gradient_exact_parity_when_disabled():
    torch.manual_seed(20260818)
    baseline_model = TinyCausalLM()
    simple_model = TinyCausalLM()
    simple_model.load_state_dict(baseline_model.state_dict())
    baseline = make_loss_trainer(RecGRPOTrainer)
    simple = _simple_loss_trainer(0.0, 0.0)
    inputs = parity_inputs()
    baseline_loss = baseline._compute_loss(baseline_model, inputs)
    simple_loss = simple._compute_loss(simple_model, inputs)
    baseline_loss.backward()
    simple_loss.backward()
    assert torch.equal(baseline_loss, simple_loss)
    for left, right in zip(baseline_model.parameters(), simple_model.parameters()):
        assert torch.equal(left.grad, right.grad)


def test_primary_signal_aux_gradient_is_exact_zero_with_lambda_enabled():
    torch.manual_seed(9)
    primary_model = TinyCausalLM()
    simple_model = TinyCausalLM()
    simple_model.load_state_dict(primary_model.state_dict())
    primary = _simple_loss_trainer(0.0, 0.0)
    simple = _simple_loss_trainer(0.10, 0.0)
    inputs = parity_inputs()
    inputs["dsr_aux_advantages"] = torch.zeros(2)
    primary_loss = primary._compute_loss(primary_model, inputs)
    simple_loss = simple._compute_loss(simple_model, inputs)
    primary_loss.backward()
    simple_loss.backward()
    assert torch.equal(primary_loss, simple_loss)
    for left, right in zip(primary_model.parameters(), simple_model.parameters()):
        assert torch.equal(left.grad, right.grad)


def test_zero_std_auxiliary_has_nonzero_gradient():
    torch.manual_seed(10)
    model = TinyCausalLM()
    trainer = _simple_loss_trainer(0.10, 0.0)
    inputs = parity_inputs()
    inputs["advantages"] = torch.zeros(2)
    inputs["dsr_aux_advantages"] = torch.tensor([1.0, -1.0])
    loss = trainer._compute_loss(model, inputs)
    loss.backward()
    assert any(parameter.grad is not None and bool((parameter.grad != 0).any()) for parameter in model.parameters())


def test_nothink_is_the_exact_old_implementation():
    assert make_dsr_nothink_reward_func is __import__(
        "gr_rec_dsr_v1.dsr_runtime", fromlist=["make_dsr_nothink_reward_func"]
    ).make_dsr_nothink_reward_func
    assert nothink_unlikelihood_loss is __import__(
        "gr_rec_dsr_v1.dsr_objectives", fromlist=["nothink_unlikelihood_loss"]
    ).nothink_unlikelihood_loss
    plan = build_nothink_rescue_plan([0] * 8, [3] * 8, [1] * 8, {1, 2})
    assert plan.active and plan.lambda_a == 0.10 and plan.concentration == 1.0


def test_nothink_primary_reward_wrapper_matches_baseline():
    tokenizer = FakeSemanticTokenizer()
    completion = [tokenizer.vocab[token] for token in (
        "<|prod_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>",
    )]
    kwargs = {
        "all_gold_sids": [["<|prod_begin|><s_a_1><s_b_2><s_c_3>"]],
        "route": ["no_think"],
        "recommendation_group_id": ["g"],
        "target_domain": ["prod"],
    }
    baseline = make_nothink_reward_func(tokenizer)(["p"], ["c"], [completion], **kwargs)
    simple = make_dsr_nothink_reward_func(tokenizer)(["p"], ["c"], [completion], **kwargs)
    assert simple == baseline == [8.0]
