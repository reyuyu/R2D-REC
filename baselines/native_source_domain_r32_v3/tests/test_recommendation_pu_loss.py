"""CPU-only mathematical tests for REC-PU Phase 1."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rec_pu"))
from recommendation_pu_loss import (  # noqa: E402
    attenuate_unlabeled_logits,
    build_rec_pu_masks,
    rec_pu_position_loss,
)


DTYPE = torch.float64


def close(actual: torch.Tensor | float, expected: torch.Tensor | float, *, atol: float = 1e-10) -> None:
    actual_tensor = torch.as_tensor(actual)
    expected_tensor = torch.as_tensor(expected, dtype=actual_tensor.dtype, device=actual_tensor.device)
    torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=atol)


def ce_grad(logits: torch.Tensor, positive_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = logits.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(value.unsqueeze(0), torch.tensor([positive_id]))
    loss.backward()
    return loss.detach(), value.grad.detach()


def pu_grad(logits: torch.Tensor, positives: list[int], level: list[int], beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    value = logits.detach().clone().requires_grad_(True)
    loss, _ = rec_pu_position_loss(value, positives, level, beta=beta)
    loss.backward()
    return loss.detach(), value.grad.detach()


def test_forward_invariant() -> None:
    logits = torch.randn(3, 11, dtype=DTYPE)
    mask = build_rec_pu_masks(11, [3], [1, 3, 7]).unlabeled
    transformed = attenuate_unlabeled_logits(logits, mask, beta=0.05)
    assert torch.equal(transformed.detach(), logits.detach())
    print("PASS forward invariant: transformed logits are bit-identical")


def test_beta_one_ce_equivalence() -> None:
    logits = torch.tensor([0.6, -0.4, 0.2, -0.1], dtype=DTYPE)
    baseline_loss, baseline_grad = ce_grad(logits, 0)
    pu_loss, pu_gradient = pu_grad(logits, [0], [0, 1], beta=1.0)
    close(pu_loss, baseline_loss)
    close(pu_gradient, baseline_grad)
    print(f"PASS beta=1 CE equivalence: loss={pu_loss.item():.12f}")


def test_unlabeled_gradient_ratio() -> None:
    # vocab: positive a0, unlabeled a1, wrong-level b0, ordinary token.
    logits = torch.tensor([0.6, -0.4, 0.2, -0.1], dtype=DTYPE)
    _, baseline = ce_grad(logits, 0)
    _, pu = pu_grad(logits, [0], [0, 1], beta=0.05)
    unlabeled_ratio = pu[1] / baseline[1]
    close(unlabeled_ratio, 0.05)
    print(
        "PASS unlabeled gradient ratio: "
        f"baseline={baseline[1].item():.12f} REC-PU={pu[1].item():.12f} ratio={unlabeled_ratio.item():.12f}"
    )


def test_wrong_token_gradient_ratio() -> None:
    logits = torch.tensor([0.6, -0.4, 0.2, -0.1], dtype=DTYPE)
    _, baseline = ce_grad(logits, 0)
    _, pu = pu_grad(logits, [0], [0, 1], beta=0.05)
    wrong_level_ratio = pu[2] / baseline[2]
    normal_ratio = pu[3] / baseline[3]
    close(wrong_level_ratio, 1.0)
    close(normal_ratio, 1.0)
    print(
        "PASS wrong-token gradient ratio: "
        f"wrong-level baseline={baseline[2].item():.12f} REC-PU={pu[2].item():.12f} ratio={wrong_level_ratio.item():.12f}; "
        f"normal ratio={normal_ratio.item():.12f}"
    )


def test_positive_gradient() -> None:
    logits = torch.tensor([-1.0, 0.7, 0.2, -0.3], dtype=DTYPE)
    _, gradient = pu_grad(logits, [0], [0, 1], beta=0.05)
    assert gradient[0].item() < 0.0
    print(f"PASS positive gradient: grad_positive={gradient[0].item():.12f} (< 0, gradient descent raises it)")


def test_multi_positive() -> None:
    logits = torch.tensor([0.0, -1.0, -1.5, 0.0], dtype=DTYPE)
    _, gradient = pu_grad(logits, [0, 3], [0, 3], beta=0.05)
    assert gradient[0].item() < 0.0
    assert gradient[3].item() < 0.0
    print(f"PASS multi-positive: grad_a1={gradient[0].item():.12f}, grad_a4={gradient[3].item():.12f}")


def test_same_level_mask() -> None:
    # Predicting s_a: other a ids are U; b/c and ordinary ids must remain O.
    masks = build_rec_pu_masks(12, [4], [1, 4, 7])
    assert masks.positive[4] and not masks.unlabeled[4]
    assert masks.unlabeled[1] and masks.unlabeled[7]
    for token_id in (2, 5, 9, 11):  # representative b/c/ordinary vocabulary ids
        assert masks.other[token_id] and not masks.unlabeled[token_id]
    assert torch.equal(masks.positive | masks.unlabeled | masks.other, torch.ones(12, dtype=torch.bool))
    assert not torch.any(masks.positive & masks.unlabeled)
    assert not torch.any(masks.positive & masks.other)
    assert not torch.any(masks.unlabeled & masks.other)
    print("PASS same-level mask: only other s_a ids are U; b/c/ordinary ids are O")


def test_beta_zero() -> None:
    logits = torch.tensor([0.6, -0.4, 0.2, -0.1], dtype=DTYPE)
    _, gradient = pu_grad(logits, [0], [0, 1], beta=0.0)
    close(gradient[1], 0.0)
    assert gradient[2].item() > 0.0 and gradient[3].item() > 0.0
    print(f"PASS beta=0 invariant: grad_unlabeled={gradient[1].item():.12f}")


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
