"""CPU integration regression: native SID8 replacement / disabled parity."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rec_pu.mini_topk_loss import MiniTopKConfig
from rec_pu.recommendation_pu_phase2 import RecPUPackedTarget, TokenPrefixPositiveSets
from rec_pu.sid8_rec_pu_integration import RecPUConfig, SIDComponentVocab, compute_native_sid8_loss


def inputs(logits):
    labels = torch.tensor([[-100, 1, 5, 8, 11, 2]])
    weights = torch.tensor([[0., 1., 8., 8., 8., 1.]])
    ids = torch.zeros_like(labels)
    task = torch.ones_like(labels)
    domain = torch.ones_like(weights)
    target = RecPUPackedTarget(0, 0, 6, 2, 3, 4, 1, 2, 3, TokenPrefixPositiveSets((5,6), (8,), (11,)), "recommendation_cot")
    return dict(logits=logits, labels=labels, loss_weights=weights, sample_ids=ids,
                sample_task_ids=task, sample_domain_weights=domain, rec_pu_targets=[[target]],
                sid_component_vocab=SIDComponentVocab((5,6,7), (8,9,10), (11,12,13)))


def run():
    torch.manual_seed(3)
    z0 = torch.randn(1, 6, 16, requires_grad=True)
    plain, _ = compute_native_sid8_loss(**inputs(z0), rec_pu_config=RecPUConfig(False), mini_topk_config=MiniTopKConfig(False))
    plain_g = torch.autograd.grad(plain, z0)[0]
    z1 = z0.detach().clone().requires_grad_(True)
    off, _ = compute_native_sid8_loss(**inputs(z1), rec_pu_config=RecPUConfig(False), mini_topk_config=MiniTopKConfig(False))
    off_g = torch.autograd.grad(off, z1)[0]
    assert torch.equal(plain, off) and torch.equal(plain_g, off_g)

    z2 = z0.detach().clone().requires_grad_(True)
    enabled, details = compute_native_sid8_loss(**inputs(z2), rec_pu_config=RecPUConfig(False), mini_topk_config=MiniTopKConfig(True))
    grad = torch.autograd.grad(enabled, z2)[0]
    assert torch.isfinite(enabled) and torch.isfinite(grad).all()
    assert tuple(sorted(details.changed_positions)) == ((0,1),(0,2),(0,3))
    assert torch.equal(details.denominator, torch.tensor([5.]))
    # One A multi-positive term differs; singleton B/C retain native CE exactly.
    assert torch.equal(details.contributions[0,2], details.base_contributions[0,2])
    assert torch.equal(details.contributions[0,3], details.base_contributions[0,3])
    assert details.mini_topk_sum_count[16,1].item() == 3
    print("MINI_TOPK_CPU_INTEGRATION = PASS")

if __name__ == "__main__": run()
