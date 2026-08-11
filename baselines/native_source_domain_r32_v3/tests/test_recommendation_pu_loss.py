"""CPU checks for Set-PU's direct-autograd P/U/O scalar objective."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rec_pu"))
from recommendation_pu_loss import build_rec_pu_masks, rec_pu_position_loss  # noqa: E402


DTYPE = torch.float64


def test_alpha_one_singleton_is_ce() -> None:
    logits = torch.tensor([.6, -.4, .2, -.1], dtype=DTYPE, requires_grad=True)
    value, _ = rec_pu_position_loss(logits, (0,), (0, 1), alpha=1.0)
    grad = torch.autograd.grad(value, logits)[0]
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference = F.cross_entropy(reference_logits.unsqueeze(0), torch.tensor([0]))
    ref_grad = torch.autograd.grad(reference, reference_logits)[0]
    torch.testing.assert_close(value, reference, rtol=0, atol=1e-12)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=1e-12)
    print("PASS alpha=1 singleton CE")


def test_multi_positive_is_set_objective() -> None:
    logits = torch.tensor([.2, -.7, .4, .1], dtype=DTYPE, requires_grad=True)
    value, _ = rec_pu_position_loss(logits, (0, 2), (0, 1, 2), alpha=1.0)
    reference = torch.logsumexp(logits, 0) - torch.logsumexp(logits[torch.tensor([0, 2])], 0)
    torch.testing.assert_close(value, reference, rtol=0, atol=1e-12)
    print("PASS alpha=1 multi-positive Set-NLL")


def test_alpha_changes_only_u_denominator_weight() -> None:
    logits = torch.tensor([.6, -.4, .2, -.1], dtype=DTYPE, requires_grad=True)
    value, _ = rec_pu_position_loss(logits, (0,), (0, 1), alpha=.05)
    manual = torch.logsumexp(torch.stack((logits[0], logits[1] + torch.log(torch.tensor(.05, dtype=DTYPE)), logits[2], logits[3])), 0) - logits[0]
    torch.testing.assert_close(value, manual, rtol=0, atol=1e-12)
    print("PASS alpha=.05 weighted U denominator")


def test_same_level_masks_are_exact() -> None:
    masks = build_rec_pu_masks(12, (4,), (1, 4, 7))
    assert masks.positive[4] and masks.unlabeled[1] and masks.unlabeled[7]
    for token in (2, 5, 9, 11):
        assert masks.other[token]
    assert torch.all(masks.positive | masks.unlabeled | masks.other)
    assert not torch.any(masks.positive & masks.unlabeled)
    assert not torch.any(masks.positive & masks.other)
    assert not torch.any(masks.unlabeled & masks.other)
    print("PASS same-level P/U/O partition")


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
