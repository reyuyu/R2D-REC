"""Regression proving monitor-only metrics cannot alter baseline CE updates."""

from __future__ import annotations

import torch

from rec_pu.sid8_rec_pu_integration import RecPUConfig, SIDComponentVocab, compute_native_sid8_loss


VOCAB = SIDComponentVocab(a=(1, 2), b=(3, 4), c=(5, 6))


def _target():
    return {
        "segment_index": 0, "segment_start": 0, "segment_end": 5,
        "a_label_position": 2, "b_label_position": 3, "c_label_position": 4,
        "a_logit_position": 1, "b_logit_position": 2, "c_logit_position": 3,
        "positives": {"a": (1, 2), "b": (3,), "c": (5,)},
    }


def _fixture():
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


def test_monitor_only_loss_and_gradient_bitwise_parity():
    monitored_logits, labels, weights, ids, tasks, domains = _fixture()
    plain_logits = monitored_logits.detach().clone().requires_grad_(True)
    common = dict(
        labels=labels, loss_weights=weights, sample_ids=ids,
        sample_task_ids=tasks, sample_domain_weights=domains, rec_pu_targets=[[_target()]],
        rec_pu_config=RecPUConfig(False, 0.05),
    )
    monitored_loss, monitored = compute_native_sid8_loss(
        logits=monitored_logits, sid_component_vocab=VOCAB,
        collect_candidate_metrics=True, collect_rec_metrics=True, **common,
    )
    plain_loss, plain = compute_native_sid8_loss(logits=plain_logits, **common)
    monitored_loss.backward()
    plain_loss.backward()

    assert torch.equal(monitored_loss, plain_loss)
    assert torch.equal(monitored_logits.grad, plain_logits.grad)
    assert torch.equal(monitored.contributions, plain.contributions)
    assert monitored.changed_positions == ()
    assert torch.equal(monitored.rec_setpu_sum_count[:, 1], torch.ones(3, dtype=torch.float64))
    assert torch.equal(monitored.rec_candidate_sum_count[:, 1], torch.ones(7, dtype=torch.float64))
    assert torch.equal(plain.rec_setpu_sum_count, torch.zeros((3, 2), dtype=torch.float64))
