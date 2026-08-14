"""CPU regression suite for Alpha-Mini Multi-Positive Set-NLL + TopK."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rec_pu.mini_topk_loss import MiniTopKConfig, mini_topk_position_loss, mini_topk_batched_position_loss
from rec_pu.recommendation_pu_phase2 import RecommendationMetadata, SemanticSID, build_prefix_positive_sets


def close(a, b, tol=1e-6):
    assert torch.allclose(a, b, atol=tol, rtol=tol), (a, b)


def run():
    cfg = MiniTopKConfig(enabled=True, slack_a=1, slack_b=2, slack_c=3)
    vocab = tuple(range(10))
    # disabled caller parity is exercised in integration; singleton is exact CE.
    z = torch.tensor([.2, -.1, .5, .3, .0, .1, -.2, .8, -.4, .6], requires_grad=True)
    r = mini_topk_position_loss(z, gold_id=3, positive_ids=(3,), same_level_ids=vocab, level="a", config=cfg)
    ce = F.cross_entropy(z.float().unsqueeze(0), torch.tensor([3]))
    close(r.loss, ce); close(r.rank_loss, torch.zeros_like(ce)); close(torch.autograd.grad(r.loss, z, retain_graph=True)[0], torch.autograd.grad(ce, z)[0])

    # Multi-positive Set-NLL and every positive gets gradient.
    z2 = torch.tensor([-.2, .1, .4, -.3, .8, .0, .2, -.1], requires_grad=True)
    r2 = mini_topk_position_loss(z2, gold_id=1, positive_ids=(1, 4), same_level_ids=tuple(range(8)), level="a", config=cfg)
    expected = torch.logsumexp(z2.float(), 0) - torch.logsumexp(z2[[1, 4]].float(), 0)
    close(r2.set_nll, expected)
    g = torch.autograd.grad(r2.set_nll, z2, retain_graph=True)[0]
    assert g[1] < 0 and g[4] < 0
    assert mini_topk_position_loss(z2.detach().clone().index_add(0, torch.tensor([1]), torch.tensor([1.])), gold_id=1, positive_ids=(1,4), same_level_ids=tuple(range(8)), level="a", config=cfg).set_nll < r2.set_nll

    # Prefix trie: B and C must follow the teacher path, never Cartesian-expand.
    sid = lambda a,b,c: SemanticSID("<|video_begin|>", f"<s_a_{a}>", f"<s_b_{b}>", f"<s_c_{c}>")
    meta = RecommendationMetadata("g", 3, (sid(1,1,1), sid(1,2,2), sid(2,9,9)), sid(1,1,1))
    p = build_prefix_positive_sets(meta)
    assert p.a == ("<s_a_1>", "<s_a_2>") and p.b == ("<s_b_1>", "<s_b_2>") and p.c == ("<s_c_1>",)

    # r+1 boundary / dynamic K / all-in behavior; wrong normal vocab cannot enter boundary.
    z3 = torch.tensor([0., 4., 3., 2., 1., 10., 9.], requires_grad=True)
    rr = mini_topk_position_loss(z3, gold_id=1, positive_ids=(1,2), same_level_ids=(0,1,2,3,4), level="a", config=cfg)
    assert rr.k == 3 and float(rr.all_in) == 1.0  # positives plus exactly one slack slot
    # With r=1, negative boundary is second-highest among ids {0,3,4}: id4=1.
    weakest = -(torch.logsumexp(-z3[[1,2]], 0) - math.log(2.0))
    close(rr.rank_loss, F.softplus(z3[4] - weakest))
    cfg_large = MiniTopKConfig(enabled=True, slack_a=3, slack_b=8, slack_c=12)
    zero_rank = mini_topk_position_loss(z3, gold_id=1, positive_ids=(1,2), same_level_ids=(0,1,2,3,4), level="a", config=cfg_large)
    close(zero_rank.rank_loss, torch.zeros_like(zero_rank.rank_loss))

    # weakest positive receives stronger ranking pressure than already-strong positive.
    z4 = torch.tensor([0., -3., 3., 2., 1.], requires_grad=True)
    q = mini_topk_position_loss(z4, gold_id=1, positive_ids=(1,2), same_level_ids=(0,1,2,3,4), level="a", config=cfg)
    qg = torch.autograd.grad(q.rank_loss, z4)[0]
    assert abs(qg[1]) > abs(qg[2])
    # BF16/FP32 finite.
    bf = mini_topk_position_loss(z4.detach().bfloat16(), gold_id=1, positive_ids=(1,2), same_level_ids=(0,1,2,3,4), level="a", config=cfg)
    assert torch.isfinite(bf.loss)
    # Batched A/B/C route is numerically/gradient equivalent to individual math.
    zb = torch.stack((z2.detach(), z2.detach() + 0.2)).requires_grad_(True)
    positives = ((1,4), (1,2)); gold = torch.tensor((1,1))
    batched = mini_topk_batched_position_loss(zb, gold_ids=gold, positive_sets=positives, same_level_ids=tuple(range(8)), level="a", config=cfg)
    individual = torch.stack((
        mini_topk_position_loss(zb[0], gold_id=1, positive_ids=positives[0], same_level_ids=tuple(range(8)), level="a", config=cfg).loss,
        mini_topk_position_loss(zb[1], gold_id=1, positive_ids=positives[1], same_level_ids=tuple(range(8)), level="a", config=cfg).loss,
    ))
    close(batched.loss, individual)
    close(torch.autograd.grad(batched.loss.sum(), zb, retain_graph=True)[0], torch.autograd.grad(individual.sum(), zb)[0])
    print("MINI_TOPK_CPU_CORE = PASS")


if __name__ == "__main__": run()
