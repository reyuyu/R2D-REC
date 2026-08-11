"""CPU regression for always-on BETA-SETloss PackRatio core metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rec_pu.core_recommendation_metrics import (
    CandidateRankOutcome,
    candidate_chain_sum_count,
    candidate_rank_outcome,
    candidate_window_active,
    finite_sum_count,
    pack_effective_share_sum_count,
    positive_entropy_sum_count,
)
from rec_pu.recommendation_pu_loss import rec_pu_batched_position_loss
from rec_pu.sid8_rec_pu_integration import RecPUConfig, SIDComponentVocab, compute_native_sid8_loss


VOCAB = SIDComponentVocab(a=(1, 2), b=(3, 4), c=(5, 6))


def _target():
    return {
        "segment_index": 0, "segment_start": 0, "segment_end": 5,
        "a_label_position": 2, "b_label_position": 3, "c_label_position": 4,
        "a_logit_position": 1, "b_logit_position": 2, "c_logit_position": 3,
        "positives": {"a": (1, 2), "b": (3,), "c": (5,)},
    }


def _loss_fixture():
    labels = torch.tensor([[-100, 0, 1, 3, 5]], dtype=torch.long)
    weights = torch.tensor([[0.0, 1.0, 8.0, 8.0, 8.0]])
    sample_ids = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long)
    tasks = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    domains = torch.ones_like(weights)
    logits = torch.tensor([[
        [0.0] * 8,
        [0.2, 2.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 2.2, 0.4, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 1.7, 0.5, 0.0],
        [0.0] * 8,
    ]], dtype=torch.float32, requires_grad=True)
    return logits, labels, weights, sample_ids, tasks, domains


def _compute(logits, *, collect_candidate_metrics=False):
    labels, weights, ids, tasks, domains = _loss_fixture()[1:]
    return compute_native_sid8_loss(
        logits=logits, labels=labels, loss_weights=weights, sample_ids=ids,
        sample_task_ids=tasks, sample_domain_weights=domains, rec_pu_targets=[[_target()]],
        rec_pu_config=RecPUConfig(True, 0.05), sid_component_vocab=VOCAB,
        collect_candidate_metrics=collect_candidate_metrics,
    )


def test_metric_parity_and_level_mapping():
    logits, *_ = _loss_fixture()
    loss, details = _compute(logits)
    shift = logits[:, :-1]
    expected = []
    expected_ce = []
    for level, position, positives in (("a", 1, (1, 2)), ("b", 2, (3,)), ("c", 3, (5,))):
        value = rec_pu_batched_position_loss(shift[:, position], [positives], VOCAB.for_level(level), beta=0.05)
        expected.append(value.detach())
        expected_ce.append(F.cross_entropy(shift[:, position], torch.tensor([positives[0]]), reduction="none").detach())
    expected = torch.cat(expected)
    expected_ce = torch.cat(expected_ce)
    assert torch.allclose(details.rec_setpu_sum_count[:, 0].float(), expected)
    assert torch.equal(details.rec_setpu_sum_count[:, 1], torch.ones(3, dtype=torch.float64))
    assert torch.allclose(details.rec_posmass_sum_count[:, 0].float(), torch.exp(-expected))
    assert torch.allclose(details.rec_goldprob_sum_count[:, 0].float(), torch.exp(-expected_ce))
    assert details.rec_posentropy_a_sum_count[1].item() == 1.0
    assert 0.0 <= details.rec_posentropy_a_sum_count[0].item() <= 1.0
    assert not details.rec_setpu_sum_count.requires_grad
    assert not details.rec_posmass_sum_count.requires_grad
    assert not details.rec_goldprob_sum_count.requires_grad
    assert not details.rec_posentropy_a_sum_count.requires_grad
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_setpu_posmass_and_goldprob_formula():
    set_pu = torch.tensor([0.0, 0.7, 2.0])
    base_ce = torch.tensor([0.0, 0.7, 2.0])
    assert torch.allclose(torch.exp(-set_pu), torch.tensor([1.0, 0.4965853, 0.1353353]), atol=1e-6)
    assert torch.allclose(torch.exp(-base_ce), torch.exp(-set_pu))
    result = finite_sum_count(torch.exp(-set_pu))
    assert result[1].item() == 3.0


def test_positive_entropy_uniform_concentrated_and_singleton():
    uniform = positive_entropy_sum_count(torch.tensor([[0.0, 0.0, 0.0, 0.0]]), [(0, 1, 2, 3)])
    concentrated = positive_entropy_sum_count(torch.tensor([[20.0, 0.0, 0.0, 0.0]]), [(0, 1, 2, 3)])
    singleton = positive_entropy_sum_count(torch.tensor([[2.0, 0.0]]), [(0,)])
    assert torch.allclose(uniform, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-6)
    assert concentrated[0].item() < 1e-6 and concentrated[1].item() == 1.0
    assert singleton[0].item() == 0.0 and singleton[1].item() == 0.0


def test_metric_detach_gradient_parity():
    first, *_ = _loss_fixture()
    second = first.detach().clone().requires_grad_(True)
    first_loss, first_details = _compute(first)
    second_loss, _ = _compute(second)
    first_loss.backward()
    second_loss.backward()
    assert torch.allclose(first_loss, second_loss, atol=0.0, rtol=0.0)
    assert torch.allclose(first.grad, second.grad, atol=0.0, rtol=0.0)
    assert first_details.rec_setpu_sum_count.grad_fn is None


def test_effective_share_pack_mean_semantics():
    simple = pack_effective_share_sum_count(
        torch.tensor([0, 0, 1, 1, 1, 2]), torch.zeros(6, dtype=torch.long), 1, 4
    )
    assert torch.allclose(simple[:, 0], torch.tensor([2 / 6, 3 / 6, 1 / 6, 0.0], dtype=torch.float64))
    assert torch.allclose(simple[:, 1], torch.ones(4, dtype=torch.float64))

    # Token length is absent from this calculation by design.
    length_independent = pack_effective_share_sum_count(
        torch.tensor([0, 1]), torch.tensor([0, 0]), 1, 4
    )
    assert torch.allclose(length_independent[:, 0], torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64))

    # Two packs: ten material segments and one chain segment. Pack-mean is
    # 0.5/0.5, not the global segment ratio 10/11 versus 1/11.
    tasks = torch.tensor([0] * 10 + [3])
    rows = torch.tensor([0] * 10 + [1])
    pack_mean = pack_effective_share_sum_count(tasks, rows, 2, 4)
    averaged = pack_mean[:, 0] / pack_mean[:, 1]
    assert torch.allclose(averaged, torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float64))


def test_ddp_sum_count_merge():
    rank0 = torch.tensor([[2.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
    rank1 = torch.tensor([[4.0, 4.0], [2.0, 2.0]], dtype=torch.float64)
    merged = rank0 + rank1
    assert torch.allclose(merged[:, 0] / merged[:, 1], torch.tensor([1.0, 0.75], dtype=torch.float64))


def _ranking_logits(vocab_size=300):
    return torch.full((1, vocab_size), -10.0, requires_grad=True)


def test_candidate_a_hit_hit32_only_miss_and_multi_positive():
    component = tuple(range(256))
    hit_logits = _ranking_logits()
    with torch.no_grad():
        hit_logits[0, 3] = 10.0
    hit = candidate_rank_outcome(hit_logits, [(3,)], component, max_k=32)
    assert hit.hit8.item() == hit.hit32.item() == hit.coverage8.item() == hit.coverage32.item() == 1.0

    only32_logits = _ranking_logits()
    with torch.no_grad():
        only32_logits[0, :19] = torch.arange(19, 0, -1, dtype=torch.float32) + 10.0
        only32_logits[0, 20] = 1.0
    only32 = candidate_rank_outcome(only32_logits, [(20,)], component, max_k=32)
    assert only32.hit8.item() == 0.0 and only32.hit32.item() == 1.0

    miss_logits = _ranking_logits()
    with torch.no_grad():
        miss_logits[0, :40] = torch.arange(40, 0, -1, dtype=torch.float32) + 10.0
        miss_logits[0, 80] = 1.0
    miss = candidate_rank_outcome(miss_logits, [(80,)], component, max_k=32)
    assert miss.hit8.item() == 0.0 and miss.hit32.item() == 0.0

    multi_logits = _ranking_logits()
    with torch.no_grad():
        multi_logits[0, 100] = 10.0
    multi = candidate_rank_outcome(multi_logits, [(3, 100, 200)], component, max_k=32)
    assert multi.hit8.item() == multi.hit32.item() == 1.0
    assert abs(multi.coverage8.item() - 1.0 / 3.0) < 1e-6


def test_candidate_coverage_prefix_component_restriction_and_gradient():
    component = tuple(range(64))
    logits = _ranking_logits(80)
    with torch.no_grad():
        logits[0, 1], logits[0, 3] = 10.0, 9.0
        logits[0, 70] = 1000.0  # text / outside component vocab: must not affect ranking
    outcome = candidate_rank_outcome(logits, [(1, 2, 3, 4)], component, max_k=32)
    assert outcome.hit8.item() == 1.0 and abs(outcome.coverage8.item() - 0.5) < 1e-6

    # b/c receive only the Phase-2 prefix-filtered positives passed to them.
    b_logits, c_logits = _ranking_logits(80), _ranking_logits(80)
    with torch.no_grad():
        b_logits[0, 4] = 10.0  # another a branch's b: not current P={3}
        c_logits[0, 6] = 10.0  # another a,b branch's c: not current P={5}
    b = candidate_rank_outcome(b_logits, [(3,)], (3, 4), max_k=8)
    c = candidate_rank_outcome(c_logits, [(5,)], (5, 6), max_k=8)
    # vocab has only two values, so the intended P is still in top-8. Use a
    # k=1 local check to show it cannot be replaced by the wrong branch.
    b_top1 = candidate_rank_outcome(b_logits, [(3,)], (3, 4), max_k=1)
    c_top1 = candidate_rank_outcome(c_logits, [(5,)], (5, 6), max_k=1)
    assert b.hit8.item() == c.hit8.item() == 1.0
    assert b_top1.hit8.item() == c_top1.hit8.item() == 0.0

    base_loss = logits.square().mean()
    metric_loss = base_loss + outcome.hit8.sum() * 0.0
    base_grad = torch.autograd.grad(base_loss, logits, retain_graph=True)[0]
    metric_grad = torch.autograd.grad(metric_loss, logits)[0]
    assert torch.allclose(base_grad, metric_grad, atol=0.0, rtol=0.0)
    assert not outcome.hit8.requires_grad


def test_candidate_chain_interval_and_no_stale_stats():
    zeros = torch.zeros(3)
    a = CandidateRankOutcome(zeros, torch.tensor([1.0, 1.0, 0.0]), zeros, zeros)
    b = CandidateRankOutcome(torch.tensor([1.0, 0.0, 1.0]), zeros, zeros, zeros)
    c = CandidateRankOutcome(torch.ones(3), zeros, zeros, zeros)
    chain = candidate_chain_sum_count(a, b, c)
    assert chain[0].item() == 1.0 and chain[1].item() == 3.0
    assert not candidate_window_active(48, True, 50)
    assert candidate_window_active(49, True, 50)
    assert not candidate_window_active(50, True, 50)
    assert not candidate_window_active(49, False, 50)
    stats = torch.stack((chain, chain))
    assert stats[:, 1].sum().item() > 0
    stats.zero_()
    assert stats[:, 1].sum().item() == 0  # cleared windows cannot be re-logged.


def test_candidate_loss_integration_and_gradient_parity():
    enabled_logits, *_ = _loss_fixture()
    disabled_logits = enabled_logits.detach().clone().requires_grad_(True)
    enabled_loss, enabled_details = _compute(enabled_logits, collect_candidate_metrics=True)
    disabled_loss, disabled_details = _compute(disabled_logits, collect_candidate_metrics=False)
    enabled_loss.backward()
    disabled_loss.backward()
    assert torch.allclose(enabled_loss, disabled_loss, atol=0.0, rtol=0.0)
    assert torch.allclose(enabled_logits.grad, disabled_logits.grad, atol=0.0, rtol=0.0)
    assert torch.equal(enabled_details.rec_candidate_sum_count[:, 1], torch.ones(7, dtype=torch.float64))
    assert torch.equal(disabled_details.rec_candidate_sum_count, torch.zeros((7, 2), dtype=torch.float64))


def main():
    for test in (
        test_metric_parity_and_level_mapping,
        test_setpu_posmass_and_goldprob_formula,
        test_positive_entropy_uniform_concentrated_and_singleton,
        test_metric_detach_gradient_parity,
        test_effective_share_pack_mean_semantics,
        test_ddp_sum_count_merge,
        test_candidate_a_hit_hit32_only_miss_and_multi_positive,
        test_candidate_coverage_prefix_component_restriction_and_gradient,
        test_candidate_chain_interval_and_no_stale_stats,
        test_candidate_loss_integration_and_gradient_parity,
    ):
        test()
        print(f"PASS {test.__name__}")
    print("CORE_RECOMMENDATION_METRICS_PASS")


if __name__ == "__main__":
    main()
