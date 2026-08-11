"""CPU math regressions for the direct-autograd Set-PU scalar objective."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1] / "rec_pu"
sys.path.insert(0, str(ROOT))
from recommendation_pu_loss import build_rec_pu_masks, rec_pu_batched_position_loss, rec_pu_position_loss  # noqa: E402


DTYPE = torch.float64
VOCAB = 23
LEVEL = (2, 5, 8, 11, 14, 17)
POSITIVES = ((2,), (5, 8), (11, 14, 17), (2, 17))


def test_alpha_one_singleton_is_onehot_ce() -> None:
    logits = torch.tensor([[.2, -.1, .8, .4, -.7]], dtype=DTYPE, requires_grad=True)
    loss, _ = rec_pu_position_loss(logits[0], (2,), (1, 2), alpha=1.0)
    grad = torch.autograd.grad(loss, logits)[0]
    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(ref_logits, torch.tensor([2]))
    ref_grad = torch.autograd.grad(ref_loss, ref_logits)[0]
    torch.testing.assert_close(loss, ref_loss, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(grad, ref_grad, rtol=0.0, atol=1e-12)
    print("PASS alpha=1 singleton CE equivalence")


def test_alpha_one_multi_is_standard_set_nll() -> None:
    logits = torch.tensor([.2, -.1, .8, .4, -.7], dtype=DTYPE, requires_grad=True)
    loss, _ = rec_pu_position_loss(logits, (1, 3), (1, 2, 3), alpha=1.0)
    grad = torch.autograd.grad(loss, logits)[0]
    ref_logits = logits.detach().clone().requires_grad_(True)
    ref = torch.logsumexp(ref_logits, 0) - torch.logsumexp(ref_logits[torch.tensor([1, 3])], 0)
    ref_grad = torch.autograd.grad(ref, ref_logits)[0]
    torch.testing.assert_close(loss, ref, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(grad, ref_grad, rtol=0.0, atol=1e-12)
    print("PASS alpha=1 multi-positive set-NLL equivalence")


def test_alpha_weighted_u_matches_manual_scalar() -> None:
    logits = torch.tensor([.6, -.4, .2, -.1, .3], dtype=DTYPE, requires_grad=True)
    # P=0, U=1, O=2/3/4.  The manual denominator is explicit.
    loss, _ = rec_pu_position_loss(logits, (0,), (0, 1), alpha=.05)
    manual = torch.logsumexp(torch.stack((logits[0], logits[1] + torch.log(torch.tensor(.05, dtype=DTYPE)), logits[2], logits[3], logits[4])), 0) - logits[0]
    torch.testing.assert_close(loss, manual, rtol=0.0, atol=1e-12)
    grad = torch.autograd.grad(loss, logits)[0]
    manual_logits = logits.detach().clone().requires_grad_(True)
    manual_ref = torch.logsumexp(torch.stack((manual_logits[0], manual_logits[1] + torch.log(torch.tensor(.05, dtype=DTYPE)), manual_logits[2], manual_logits[3], manual_logits[4])), 0) - manual_logits[0]
    manual_grad = torch.autograd.grad(manual_ref, manual_logits)[0]
    torch.testing.assert_close(grad, manual_grad, rtol=0.0, atol=1e-12)
    print(f"PASS alpha=.05 U scalar contribution: grad_U={grad[1].item():.12f}")


def test_masks_are_partition() -> None:
    masks = build_rec_pu_masks(10, (2, 5), (2, 3, 5, 7))
    assert not torch.any(masks.positive & masks.unlabeled)
    assert not torch.any(masks.positive & masks.other)
    assert not torch.any(masks.unlabeled & masks.other)
    assert torch.all(masks.positive | masks.unlabeled | masks.other)
    assert masks.unlabeled.nonzero().flatten().tolist() == [3, 7]
    print("PASS P/U/O disjoint exhaustive partition")


def test_batched_matches_direct_positions() -> None:
    torch.manual_seed(20260811)
    logits = torch.randn((len(POSITIVES), VOCAB), dtype=DTYPE, requires_grad=True)
    batched = rec_pu_batched_position_loss(logits, POSITIVES, LEVEL, alpha=.05)
    direct = torch.stack([
        rec_pu_position_loss(logits[row], positives, LEVEL, alpha=.05, reduction="none")[0]
        for row, positives in enumerate(POSITIVES)
    ])
    torch.testing.assert_close(batched, direct, rtol=0.0, atol=1e-12)
    print("PASS singleton/multi-positive direct-batched equivalence")


if __name__ == "__main__":
    test_alpha_one_singleton_is_onehot_ce()
    test_alpha_one_multi_is_standard_set_nll()
    test_alpha_weighted_u_matches_manual_scalar()
    test_masks_are_partition()
    test_batched_matches_direct_positions()
