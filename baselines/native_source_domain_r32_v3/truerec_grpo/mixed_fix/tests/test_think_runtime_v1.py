from dataclasses import replace
import inspect
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).parents[1]
TRUEREC_ROOT = ROOT.parent
for dependency in (ROOT, TRUEREC_ROOT / "credit", TRUEREC_ROOT / "trainer"):
    sys.path.insert(0, str(dependency))

import think_credit_v1  # noqa: E402
import think_runtime_v1 as runtime  # noqa: E402
from batch_collator_v1 import collate_business_group  # noqa: E402
from frontier_credit_v1 import FORMAT_INVALID_TOTAL  # noqa: E402
from phase3a_runtime_audit import (  # noqa: E402
    PAD_ID, TOKEN_TO_ID, abc, deterministic_groups, fresh_policy, make_group, parity_case,
)
from think_trainer_v1 import ThinkGRPOTrainerV1, format_credit_tensors  # noqa: E402
from truerec_loss_v1 import frontier_ppo_loss  # noqa: E402


def plan_for(name):
    group = deterministic_groups()[name]
    return runtime.build_think_runtime_plan(
        [candidate.metrics for candidate in group.candidates], group.all_gold_abc,
        TOKEN_TO_ID.__getitem__, dtype=torch.float64,
    )


def test_runtime_consumes_phase1_plan(monkeypatch):
    called = {"count": 0}
    original = think_credit_v1.plan_think_credit

    def wrapped(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(think_credit_v1, "plan_think_credit", wrapped)
    plan_for("NO_GOLD_A")
    assert called["count"] == 1


def test_runtime_does_not_reference_truerec_frontier_planner():
    source = inspect.getsource(runtime)
    assert "plan_frontier_credit" not in source and "build_group_runtime_plan" not in source


def test_no_semantic_negative_survives():
    plan = plan_for("A_MODE_COLLAPSE_HPR_B")
    assert bool((plan.token_credits >= 0).all())


def test_a_target_token_id_mapping_is_exact_and_deduplicated():
    assert plan_for("NO_GOLD_A").a_rescue.target_token_ids == (1, 2, 3)


@pytest.mark.parametrize(
    "case,trigger,level,targets",
    [
        ("A_MODE_COLLAPSE_HPR_B", "HPR_B", "B", (11,)),
        ("A_MODE_COLLAPSE_HPR_C", "HPR_C", "C", (21,)),
        ("A_MODE_COLLAPSE_HPR_NONE", "HPR_NONE", None, ()),
        ("NO_GOLD_A", "NONE", None, ()),
    ],
)
def test_bc_hpr_trigger_and_direct_target_mapping(case, trigger, level, targets):
    plan = plan_for(case)
    assert plan.bc_hpr.trigger == trigger
    if level is None:
        assert plan.bc_hpr.sites == ()
    else:
        assert plan.bc_hpr.sites[0].level == level
        assert plan.bc_hpr.sites[0].target_token_ids == targets


def test_hpr_a_fails_closed(monkeypatch):
    group = deterministic_groups()["A_MODE_COLLAPSE_HPR_B"]
    source = think_credit_v1.plan_think_credit(
        [candidate.metrics for candidate in group.candidates], group.all_gold_abc,
    )
    monkeypatch.setattr(
        think_credit_v1, "plan_think_credit",
        lambda *args, **kwargs: replace(source, bc_hpr=replace(source.bc_hpr, trigger="HPR_A")),
    )
    with pytest.raises(ValueError, match="HPR_A"):
        runtime.build_think_runtime_plan(
            [candidate.metrics for candidate in group.candidates], group.all_gold_abc,
            TOKEN_TO_ID.__getitem__,
        )


def test_non_bijective_target_mapping_fails_closed():
    group = deterministic_groups()["NO_GOLD_A"]
    with pytest.raises(ValueError, match="one-to-one"):
        runtime.build_think_runtime_plan(
            [candidate.metrics for candidate in group.candidates], group.all_gold_abc,
            lambda _: 1,
        )


def test_shared_context_prefix_identity():
    batch = collate_business_group(deterministic_groups()["NO_GOLD_A"], PAD_ID)
    assert runtime.assert_shared_a_prefix(batch) == (60, 61, 62, 63)


def test_changed_context_prefix_fails_closed():
    batch = collate_business_group(deterministic_groups()["NO_GOLD_A"], PAD_ID)
    changed = batch.input_ids.clone()
    changed[7, 1] = 55
    with pytest.raises(ValueError, match="identical"):
        runtime.assert_shared_a_prefix(replace(batch, input_ids=changed))


def test_causal_mock_a_logits_equal_across_g8_and_bc_can_differ():
    group = make_group("causal", [abc(1, 1, 1), abc(2, 2, 2)] * 4)
    batch = collate_business_group(group, PAD_ID)
    logits = fresh_policy()(batch.input_ids, batch.attention_mask).logits
    a_logits = torch.stack([logits[row, batch.causal_logit_indices[row, 0]] for row in range(8)])
    b_logits = torch.stack([logits[row, batch.causal_logit_indices[row, 1]] for row in range(8)])
    c_logits = torch.stack([logits[row, batch.causal_logit_indices[row, 2]] for row in range(8)])
    torch.testing.assert_close(a_logits, a_logits[:1].expand_as(a_logits), rtol=0, atol=0)
    assert not torch.equal(b_logits[0], b_logits[1])
    assert not torch.equal(c_logits[0], c_logits[1])


@pytest.mark.parametrize("case", list(deterministic_groups()))
def test_full_vs_streaming_mb1_and_mb2_loss_gradient_parity(case):
    result = parity_case(case, deterministic_groups()[case])
    assert result["full_vs_mb1_loss_max_abs_diff"] <= 1e-10
    assert result["full_vs_mb2_loss_max_abs_diff"] <= 1e-10
    assert result["full_vs_mb1_grad_max_abs_diff"] <= 1e-10
    assert result["full_vs_mb2_grad_max_abs_diff"] <= 1e-10


@pytest.mark.parametrize("microbatch,forward_calls", [(1, 8), (2, 4)])
def test_no_extra_rescue_or_hpr_forward(microbatch, forward_calls):
    policy = fresh_policy()
    trainer = ThinkGRPOTrainerV1(
        policy, TOKEN_TO_ID.__getitem__, PAD_ID, streaming_microbatch_size=microbatch,
    )
    output = trainer.backward_group_streaming(deterministic_groups()["A_MODE_COLLAPSE_HPR_C"])
    assert policy.forward_calls == forward_calls
    assert output.monitoring["A_RESCUE_EXTRA_FORWARD_CALLS"] == 0
    assert output.monitoring["BC_HPR_EXTRA_FORWARD_CALLS"] == 0


def test_a_rescue_once_group_and_candidate0_invalid_still_rescues():
    group = deterministic_groups()["CANDIDATE0_FORMAT_INVALID_RESCUE"]
    policy = fresh_policy()
    trainer = ThinkGRPOTrainerV1(policy, TOKEN_TO_ID.__getitem__, PAD_ID, streaming_microbatch_size=2)
    output = trainer.compute_group(group)
    assert output.a_rescue_loss_raw.item() > 0
    assert output.monitoring["A_RESCUE_TRIGGERED"] == 1
    batch = collate_business_group(group, PAD_ID)
    logits = policy(batch.input_ids[:2], batch.attention_mask[:2]).logits
    from think_rescue_v1 import uniform_positive_soft_ce
    expected = uniform_positive_soft_ce(
        logits[0, batch.causal_logit_indices[0, 0]], (1, 2, 3),
    )
    torch.testing.assert_close(output.a_rescue_loss_raw, expected)


def test_no_rescue_returns_graph_connected_zero():
    policy = fresh_policy()
    trainer = ThinkGRPOTrainerV1(policy, TOKEN_TO_ID.__getitem__, PAD_ID)
    output = trainer.compute_group(deterministic_groups()["NO_RESCUE_EXACT"])
    assert output.a_rescue_loss_raw.requires_grad and output.a_rescue_loss_raw.item() == 0.0


def test_format_penalty_reuses_frozen_distribution_once():
    group = deterministic_groups()["CANDIDATE0_FORMAT_INVALID_RESCUE"]
    credits, mask = format_credit_tensors(group)
    assert credits[0, :2].tolist() == pytest.approx([FORMAT_INVALID_TOTAL / 2] * 2)
    assert mask[0].tolist() == [True, True, False]
    assert int(mask.sum()) == 2
    assert plan_for("CANDIDATE0_FORMAT_INVALID_RESCUE").token_credit_mask[0].tolist() == [False] * 3


def test_zero_semantic_active_credit_has_exact_zero_gradient():
    current = torch.tensor([[[-2.0, -3.0, -4.0]]], dtype=torch.float64, requires_grad=True)
    old = current.detach().clone()
    credit = torch.zeros_like(current)
    mask = torch.tensor([[[True, False, False]]])
    loss = frontier_ppo_loss(current, old, credit, mask, 0.2).loss
    loss.backward()
    assert torch.equal(current.grad, torch.zeros_like(current))


def test_gated_semantic_mask_is_false():
    assert plan_for("CANDIDATE0_FORMAT_INVALID_RESCUE").token_credit_mask[0].tolist() == [False] * 3


def test_ppo_ratio_one_and_controlled_positive_delta():
    old = torch.zeros((1, 1, 3), dtype=torch.float64)
    current = old.clone()
    credit = torch.ones_like(old)
    mask = torch.ones_like(old, dtype=torch.bool)
    equal = frontier_ppo_loss(current, old, credit, mask, 0.2)
    assert torch.equal(equal.ratios, torch.ones_like(equal.ratios))
    increased = frontier_ppo_loss(current + 0.1, old, credit, mask, 0.2)
    assert bool((increased.ratios > 1).all())


def test_ppo_clipping_epsilon_point_two():
    old = torch.zeros((1, 1, 3), dtype=torch.float64)
    current = torch.full_like(old, 1.0)
    result = frontier_ppo_loss(current, old, torch.ones_like(old), torch.ones_like(old, dtype=torch.bool), 0.2)
    torch.testing.assert_close(result.clipped_ratios, torch.full_like(old, 1.2))


@pytest.mark.parametrize(
    "case,rescue,trigger",
    [
        ("NO_GOLD_A", "NO_GOLD_A", "NONE"),
        ("A_MODE_COLLAPSE_HPR_B", "A_MODE_COLLAPSE", "HPR_B"),
        ("A_MODE_COLLAPSE_HPR_C", "A_MODE_COLLAPSE", "HPR_C"),
        ("A_MODE_COLLAPSE_HPR_NONE", "A_MODE_COLLAPSE", "HPR_NONE"),
        ("NO_RESCUE_EXACT", "NONE", "HPR_NONE"),
    ],
)
def test_deterministic_rescue_and_hpr_cases(case, rescue, trigger):
    plan = plan_for(case)
    assert plan.a_rescue.reason == rescue
    assert plan.bc_hpr.trigger == trigger


def test_k_a_one_high_frequency_has_no_collapse_rescue():
    group = make_group("ka1", [abc(1, 1, 1)] * 8, (abc(1, 1, 1),))
    plan = runtime.build_think_runtime_plan(
        [candidate.metrics for candidate in group.candidates], group.all_gold_abc,
        TOKEN_TO_ID.__getitem__,
    )
    assert not plan.a_rescue.triggered
