"""CPU regression tests for REC-PU Phase 3 replacement in actual NSD SID8 math."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rec_pu.recommendation_pu_phase2 import RecPUPackedTarget, TokenPrefixPositiveSets
from rec_pu.recommendation_pu_loss import rec_pu_position_loss
from rec_pu.sid8_rec_pu_integration import (
    RecPUConfig,
    SIDComponentVocab,
    compute_native_sid8_loss,
    pop_rec_pu_metadata,
)


VOCAB = 12
COMPONENTS = SIDComponentVocab(a=(2, 3, 4, 5), b=(6, 7), c=(8, 9))


def target(start: int, positives: TokenPrefixPositiveSets | None = None) -> RecPUPackedTarget:
    positives = positives or TokenPrefixPositiveSets(a=(2,), b=(6,), c=(8,))
    return RecPUPackedTarget(
        segment_index=0, segment_start=start - 1, segment_end=start + 5,
        a_label_position=start, b_label_position=start + 1, c_label_position=start + 2,
        a_logit_position=start - 1, b_logit_position=start, c_logit_position=start + 1,
        positives=positives,
    )


def packed_inputs():
    # There are 10 shifted positions.  Positions 2/3/4 are an intentionally
    # visible non-target think SID; positions 6/7/8 are the final target SID.
    labels = torch.tensor([[-100, 1, 2, 6, 8, 1, 2, 6, 8, 1, -100]], dtype=torch.long)
    weights = torch.where(labels == -100, torch.zeros_like(labels, dtype=torch.float32), torch.ones_like(labels, dtype=torch.float32))
    for position in (2, 3, 4, 6, 7, 8):
        weights[0, position] = 8.0
    ids = torch.zeros_like(labels)
    tasks = torch.zeros_like(labels)
    domains = torch.ones_like(labels, dtype=torch.float32)
    torch.manual_seed(19)
    logits = torch.randn((1, labels.size(1), VOCAB), dtype=torch.float64, requires_grad=True)
    return logits, labels, weights, ids, tasks, domains


def loss_and_grad(logits, labels, weights, ids, tasks, domains, *, targets=None, enabled=False, beta=.05):
    loss, detail = compute_native_sid8_loss(
        logits=logits, labels=labels, loss_weights=weights, sample_ids=ids,
        sample_task_ids=tasks, sample_domain_weights=domains, rec_pu_targets=targets,
        rec_pu_config=RecPUConfig(rec_pu_enabled=enabled, rec_pu_unlabeled_sid_grad_scale=beta),
        sid_component_vocab=COMPONENTS if enabled else None,
    )
    grad = torch.autograd.grad(loss, logits)[0]
    return loss, detail, grad


def legacy_nsd_sid8_loss(logits, labels, weights, ids, tasks, domains):
    """Independent copy of NSD-R32-V3's pre-REC-PU loss arithmetic."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = weights[..., 1:].float().contiguous()
    shift_ids = ids[..., 1:].long().contiguous()
    shift_domains = domains[..., 1:].float().contiguous()
    valid = (shift_labels != -100) & (shift_ids >= 0)
    ce = F.cross_entropy(shift_logits.reshape(-1, VOCAB), shift_labels.reshape(-1), ignore_index=-100, reduction="none").view_as(shift_labels)
    stride = shift_ids.max().clamp_min(0) + 1
    offsets = torch.arange(shift_ids.size(0), device=shift_ids.device).unsqueeze(1) * stride
    global_ids = torch.where(valid, shift_ids + offsets, torch.full_like(shift_ids, -1))
    flat_valid = valid.reshape(-1)
    flat_ids = global_ids.reshape(-1)[flat_valid]
    unique, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
    flat_loss = (ce.float() * shift_weights).reshape(-1)[flat_valid]
    flat_domains = shift_domains.reshape(-1)[flat_valid]
    numerator = torch.zeros(unique.numel(), device=logits.device, dtype=flat_loss.dtype)
    denominator = torch.zeros_like(numerator)
    domain_sum = torch.zeros_like(numerator)
    numerator.scatter_add_(0, inverse, flat_loss)
    denominator.scatter_add_(0, inverse, torch.ones_like(flat_loss))
    domain_sum.scatter_add_(0, inverse, flat_domains)
    return ((numerator / denominator.clamp_min(1.0)) * (domain_sum / denominator.clamp_min(1.0))).mean()


def test_disabled_equivalence():
    values = packed_inputs()
    reference_loss = legacy_nsd_sid8_loss(*values)
    reference_grad = torch.autograd.grad(reference_loss, values[0])[0]
    disabled = loss_and_grad(*values, targets=[[target(6)]], enabled=False)
    assert torch.equal(reference_loss, disabled[0])
    assert torch.equal(reference_grad, disabled[2])


def test_singleton_beta_one_full_loss_and_gradient_equivalence():
    values = packed_inputs()
    base = loss_and_grad(*values, enabled=False)
    enabled = loss_and_grad(*values, targets=[[target(6)]], enabled=True, beta=1.0)
    assert torch.allclose(base[0], enabled[0], rtol=0, atol=1e-12)
    assert torch.allclose(base[2], enabled[2], rtol=0, atol=1e-12)


def test_denominator_and_no_double_counting():
    values = packed_inputs()
    base = loss_and_grad(*values, enabled=False)
    changed = loss_and_grad(*values, targets=[[target(6)]], enabled=True, beta=.05)
    assert torch.equal(base[1].denominator, changed[1].denominator)
    assert torch.equal(base[1].sample_weight_mass, changed[1].sample_weight_mass)
    selected = [("a", 5, (2,)), ("b", 6, (6,)), ("c", 7, (8,))]
    replacement_sum = base[1].contributions.new_zeros(())
    baseline_sum = base[1].contributions.new_zeros(())
    for level, position, positives in selected:  # shifted positions for labels 6/7/8
        pu, _ = rec_pu_position_loss(values[0][0, position], positives, COMPONENTS.for_level(level), beta=.05)
        expected = (8.0 * pu).to(dtype=changed[1].contributions.dtype)
        assert torch.allclose(changed[1].contributions[0, position], expected, rtol=0, atol=1e-6)
        replacement_sum = replacement_sum + changed[1].contributions[0, position]
        baseline_sum = baseline_sum + base[1].contributions[0, position]
    expected_numerator = base[1].base_sample_numerators[0] - baseline_sum + replacement_sum
    assert torch.allclose(changed[1].sample_numerators[0], expected_numerator, rtol=0, atol=1e-12)
    return {
        "baseline_numerator": base[1].base_sample_numerators[0].item(),
        "baseline_denominator": base[1].denominator[0].item(),
        "baseline_loss": base[0].item(),
        "rec_pu_numerator": changed[1].sample_numerators[0].item(),
        "rec_pu_denominator": changed[1].denominator[0].item(),
        "rec_pu_loss": changed[0].item(),
        "sid_weight_mass": base[1].sample_weight_mass[0].item(),
    }


def test_candidate_count_weight_invariant():
    values = packed_inputs()
    one = loss_and_grad(*values, targets=[[target(6)]], enabled=True)
    multi_pos = TokenPrefixPositiveSets(a=(2, 3, 4, 5), b=(6, 7), c=(8, 9))
    multi = loss_and_grad(*values, targets=[[target(6, multi_pos)]], enabled=True)
    assert torch.equal(one[1].denominator, multi[1].denominator)
    assert torch.equal(one[1].sample_weight_mass, multi[1].sample_weight_mass)


def test_only_final_and_think_sid_untouched():
    labels = torch.tensor([[-100, 1, 1, 2, 6, 8, 2, 6, 8, 1, 1, 2, 6, 8]], dtype=torch.long)
    weights = torch.where(labels == -100, torch.zeros_like(labels, dtype=torch.float32), torch.ones_like(labels, dtype=torch.float32))
    for pos in (3, 4, 5, 6, 7, 8, 11, 12, 13):
        weights[0, pos] = 8.0
    ids = torch.tensor([[-1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3]], dtype=torch.long)
    tasks = torch.tensor([[-1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1]], dtype=torch.long)
    domains = torch.where(ids >= 0, torch.ones_like(weights), torch.zeros_like(weights))
    torch.manual_seed(71)
    logits = torch.randn((1, labels.size(1), VOCAB), dtype=torch.float64, requires_grad=True)
    values = (logits, labels, weights, ids, tasks, domains)
    base = loss_and_grad(*values, enabled=False)
    changed = loss_and_grad(*values, targets=[[target(3), target(11)]], enabled=True)
    changed_indices = set(changed[1].changed_positions)
    assert changed_indices == {(0, 2), (0, 3), (0, 4), (0, 10), (0, 11), (0, 12)}
    for index in range(base[1].contributions.size(1)):
        if (0, index) not in changed_indices:
            assert torch.equal(base[1].contributions[0, index], changed[1].contributions[0, index])


def test_integrated_set_pu_gradient_matches_scalar_reference():
    # The first selected position predicts a. Later b/c components ensure this
    # is a complete Phase-2 target while the replacement gradient is inspected.
    labels = torch.tensor([[-100, 2, 6, 8]], dtype=torch.long)
    weights = torch.tensor([[0.0, 8.0, 8.0, 8.0]])
    ids = tasks = torch.zeros_like(labels)
    domains = torch.ones_like(weights)
    logits0 = torch.tensor([[
        [0.2, -0.1, 1.1, .4, -.3, .5, -.2, .7, .1, -.4, .3, .9],
        [0.0] * VOCAB,
        [0.0] * VOCAB,
        [0.0] * VOCAB,
    ]], dtype=torch.float64, requires_grad=True)
    pu_target = target(1, TokenPrefixPositiveSets(a=(2,), b=(6,), c=(8,)))
    changed = loss_and_grad(logits0, labels, weights, ids, tasks, domains, targets=[[pu_target]], enabled=True, beta=.05)
    ref_logits = logits0.detach().clone().requires_grad_(True)
    reference, _ = rec_pu_position_loss(ref_logits[0, 0], (2,), COMPONENTS.a, alpha=.05)
    # All three final components have equal weights and one valid segment; the
    # a-position receives exactly SID8 / valid-token-count of this scalar.
    reference_grad = torch.autograd.grad(reference * (8.0 / 3.0), ref_logits)[0]
    torch.testing.assert_close(changed[2][0, 0], reference_grad[0, 0], rtol=0, atol=1e-6)
    return {"u_gradient": changed[2][0, 0, 3].item(), "o_gradient": changed[2][0, 0, 6].item()}


def test_metadata_stripped_before_forward():
    inputs = {"input_ids": torch.tensor([[1]]), "rec_pu_targets": [[target(1)]], "rec_pu_config": "opaque"}
    targets, config = pop_rec_pu_metadata(inputs)
    assert targets and config == "opaque"
    assert set(inputs) == {"input_ids"}
    class StrictDummyModel:
        def __call__(self, *, input_ids):
            assert torch.equal(input_ids, torch.tensor([[1]]))
            return "one-forward-only"
    assert StrictDummyModel()(**inputs) == "one-forward-only"


def main():
    tests = [
        test_disabled_equivalence,
        test_singleton_beta_one_full_loss_and_gradient_equivalence,
        test_denominator_and_no_double_counting,
        test_candidate_count_weight_invariant,
        test_only_final_and_think_sid_untouched,
        test_integrated_set_pu_gradient_matches_scalar_reference,
        test_metadata_stripped_before_forward,
    ]
    for fn in tests:
        result = fn()
        print(f"PASS {fn.__name__}" + (f" {result}" if result is not None else ""))


if __name__ == "__main__":
    main()
