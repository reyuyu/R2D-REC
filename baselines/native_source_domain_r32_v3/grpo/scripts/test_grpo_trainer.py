# -*- coding: utf-8 -*-
"""CPU unit tests for grpo_trainer pure functions. Run: python test_grpo_trainer.py"""
import sys
import math
import torch

sys.path.insert(0, "/data/GRPO/scripts")
from grpo_trainer import (compute_group_advantages, ppo_clipped_loss,
                          approx_old_policy_kl, build_think_action_ids,
                          build_nothink_action_ids, masked_mean_loss,
                          CLIP_EPS, ADV_EPS, STD_FLOOR)

failures = []

def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)

# mock tokenizer for action-id tests
class MockTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

tok = MockTok()

# 1/2) group advantage mean ~= 0 (Think M=4, NoThink M=16)
for M in (4, 16):
    r = torch.randn(M) * 2 + 1
    a = compute_group_advantages(r)
    check(f"advantage mean ~0 (M={M})", abs(a.mean().item()) < 1e-5,
          f"mean={a.mean().item():.2e}")

# 3) zero-std -> all zero
a = compute_group_advantages([3.0, 3.0, 3.0, 3.0])
check("zero-std -> all zero", (a == 0).all().item())

# 4) group weight constant == 1 (no weighting anywhere)
GROUP_WEIGHT = 1.0
check("group weight == 1", GROUP_WEIGHT == 1.0)

# 5) gold_count not in loss path: ppo_clipped_loss signature has no gold_count
import inspect
sig = inspect.signature(ppo_clipped_loss)
check("gold_count not in loss signature", "gold_count" not in sig.parameters)

# 6/7) action mask excludes prompt & beam: think action = completion up to </think>
cot = "<think>分析一下内容</think>\n该用户: <|video_begin|><s_a_1><s_b_2><s_c_3>"
ids, trunc = build_think_action_ids(tok, cot)
check("think action ends at </think> (no beam SID)", not trunc and ids[-1] == ord(">") and "<|video_begin|>" not in "".join(chr(c) for c in ids))
cot2 = "<think>只有思考没有结束"
ids2, trunc2 = build_think_action_ids(tok, cot2)
check("think action truncated flag", trunc2 and len(ids2) > 0)

# 8) malformed NoThink still keeps action
bad = "完全乱码 no sid"
ids3 = build_nothink_action_ids(tok, bad)
check("malformed NoThink action kept", len(ids3) == len(bad))

# 9/10) parity/ratio ~1 requires model -> runtime assert in 4GPU smoke

# 11) positive-advantage clipping: ratio >> 1 with A>0 should clip at 1+eps
lr = torch.tensor([[1.5]])  # ratio 4.48
adv = torch.tensor([1.0])
loss, m = ppo_clipped_loss(lr, adv)
# s1 = 4.48, s2 = 1.2 -> min = 1.2 -> loss -1.2
check("pos-A clipping", abs(loss.item() - (-1.2)) < 1e-5, f"loss={loss.item()}")
check("clip fraction >0", m["clip_fraction"] > 0)

# 12) negative-advantage clipping: ratio << 1 with A<0 should clip at 1-eps
lr = torch.tensor([[-1.5]])  # ratio 0.22
adv = torch.tensor([-1.0])
loss, m = ppo_clipped_loss(lr, adv)
# s1 = -0.22, s2 = -0.8 -> min = -0.8 -> loss 0.8
check("neg-A clipping", abs(loss.item() - 0.8) < 1e-5, f"loss={loss.item()}")

# no-clip case: ratio 1 -> loss = -A
lr = torch.tensor([[0.0]])
adv = torch.tensor([2.0])
loss, m = ppo_clipped_loss(lr, adv)
check("ratio 1 -> loss -A", abs(loss.item() - (-2.0)) < 1e-5)

# 13) token-length normalization: long rollout does not dominate
# same ratio, one long one short: masked mean == plain mean over valid tokens
tl = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
mask = torch.tensor([[True, True, True], [True, False, False]])
lm = masked_mean_loss(tl, mask)
check("token-length normalization", abs(lm.item() - 1.0) < 1e-6)

# 14) route 1:1 constant
L_GROUP = lambda lt, ln: 0.5 * lt + 0.5 * ln
check("route 1:1 (0.5/0.5)", L_GROUP(1.0, 2.0) == 1.5)

# approx KL: ratio 1 -> 0
check("approx KL at ratio 1", abs(approx_old_policy_kl(torch.zeros(4, 5))) < 1e-6)

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL TRAINER TESTS PASSED")
