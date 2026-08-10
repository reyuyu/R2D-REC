"""Reference-vs-vectorized autograd equivalence for REC-PU Phase 4.5."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent / "rec_pu_phase3"
if not ROOT.is_dir():
    ROOT = Path(__file__).resolve().parents[1] / "rec_pu"
sys.path.insert(0, str(ROOT))
from recommendation_pu_loss import rec_pu_batched_position_loss, rec_pu_position_loss  # noqa: E402


DTYPE = torch.float64
VOCAB = 23
LEVEL = (2, 5, 8, 11, 14, 17)
POSITIVES = ((2,), (5, 8), (11, 14, 17), (2, 17))


def reference(logits: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack(
        [rec_pu_position_loss(logits[row], positive, LEVEL, beta=beta, reduction="none")[0] for row, positive in enumerate(POSITIVES)]
    )
    gradient = torch.autograd.grad(values.sum(), logits)[0]
    return values.detach(), gradient.detach()


def optimized(logits: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = rec_pu_batched_position_loss(logits, POSITIVES, LEVEL, beta=beta)
    gradient = torch.autograd.grad(values.sum(), logits)[0]
    return values.detach(), gradient.detach()


def test_reference_equivalence() -> None:
    torch.manual_seed(20260810)
    seed = torch.randn((len(POSITIVES), VOCAB), dtype=DTYPE)
    max_loss_error = 0.0
    max_grad_error = 0.0
    for beta in (0.0, 0.05, 0.5, 1.0):
        ref_loss, ref_gradient = reference(seed.detach().clone().requires_grad_(True), beta)
        opt_loss, opt_gradient = optimized(seed.detach().clone().requires_grad_(True), beta)
        loss_error = float((ref_loss - opt_loss).abs().max())
        grad_error = float((ref_gradient - opt_gradient).abs().max())
        max_loss_error = max(max_loss_error, loss_error)
        max_grad_error = max(max_grad_error, grad_error)
        torch.testing.assert_close(ref_loss, opt_loss, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(ref_gradient, opt_gradient, rtol=0.0, atol=1e-12)
    print(f"PASS reference equivalence: max_loss_error={max_loss_error:.3e} max_grad_error={max_grad_error:.3e}")


def test_batched_gradient_ratios() -> None:
    logits = torch.tensor([[0.6, -0.4, 0.2, -0.1, 0.3]], dtype=DTYPE, requires_grad=True)
    # a0=0 is P, a1=1 is U, b0=2 and ordinary=3 are O.
    pu = rec_pu_batched_position_loss(logits, ((0,),), (0, 1), beta=0.05).sum()
    pu_gradient = torch.autograd.grad(pu, logits)[0][0]
    baseline_logits = logits.detach().clone().requires_grad_(True)
    baseline = torch.nn.functional.cross_entropy(baseline_logits, torch.tensor([0]))
    baseline_gradient = torch.autograd.grad(baseline, baseline_logits)[0][0]
    u_ratio = pu_gradient[1] / baseline_gradient[1]
    wrong_ratio = pu_gradient[2] / baseline_gradient[2]
    normal_ratio = pu_gradient[3] / baseline_gradient[3]
    torch.testing.assert_close(u_ratio, torch.tensor(0.05, dtype=DTYPE), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(wrong_ratio, torch.tensor(1.0, dtype=DTYPE), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(normal_ratio, torch.tensor(1.0, dtype=DTYPE), rtol=0.0, atol=1e-12)
    assert pu_gradient[0] < 0
    print(f"PASS batched ratios: U={u_ratio:.12f} wrong={wrong_ratio:.12f} normal={normal_ratio:.12f}")


if __name__ == "__main__":
    test_reference_equivalence()
    test_batched_gradient_ratios()
