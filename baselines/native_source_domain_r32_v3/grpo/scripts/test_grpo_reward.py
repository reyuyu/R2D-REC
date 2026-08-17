# -*- coding: utf-8 -*-
"""CPU unit tests for grpo_sid reward logic. Run: python test_grpo_reward.py"""
import sys
sys.path.insert(0, __file__ and "/data/GRPO/scripts")
from grpo_sid import (parse_sid, final_sid, all_sids, q_reward,
                      think_credits, think_reward, raw_w)

V = ("video", 111, 222, 333)
G = [("video", 111, 222, 333), ("video", 444, 555, 666), ("prod", 777, 888, 999)]
GS = set(G)

failures = []

def check(name, got, expect):
    ok = got == expect
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} expect={expect}")
    if not ok:
        failures.append(name)

# 1) NoThink reward six levels
check("q exact", q_reward(("video", 111, 222, 333), GS), 8.0)
check("q AB-only", q_reward(("video", 111, 222, 999), GS), 2.0)
check("q A-only", q_reward(("video", 111, 999, 999), GS), 0.5)
check("q domain-only", q_reward(("video", 999, 999, 999), GS), 0.0)
check("q wrong-domain", q_reward(("ad", 111, 222, 333), GS), -0.25)
check("q unparsable", q_reward(None, GS), -1.0)

# 2) Exact > AB > A (no double counting)
check("exact not plus AB", q_reward(("video", 111, 222, 333), GS), 8.0)

# 3) Think 1/2/3 exact geometric reward
r1, ec1, _, _ = think_reward([("video", 111, 222, 333)], GS)
check("think 1 exact", (r1, ec1), (8.0, 1))
r2, ec2, _, _ = think_reward([("video", 111, 222, 333), ("video", 444, 555, 666)], GS)
check("think 2 exact", (r2, ec2), (12.0, 2))
r3, ec3, _, _ = think_reward(
    [("video", 111, 222, 333), ("video", 444, 555, 666), ("prod", 777, 888, 999)], GS)
check("think 3 exact", (r3, ec3), (14.0, 3))

# 4) exact + AB (AB from a DIFFERENT gold, not covered by exact)
r, ec, ac, _ = think_reward(
    [("video", 111, 222, 333), ("video", 444, 555, 999)], GS)  # 2nd matches Y AB
check("1 exact + 1 AB", (r, ec, ac), (9.0, 1, 1))
r, ec, ac, _ = think_reward(
    [("video", 111, 222, 333), ("video", 444, 555, 666), ("prod", 777, 888, 123)], GS)
check("2 exact + 1 AB", (r, ec, ac), (12.5, 2, 1))

# 5) AB + A (A from a different gold than AB)
r, ec, ac, a2 = think_reward(
    [("video", 111, 222, 999), ("video", 444, 999, 999)], GS)  # AB of X, A of Y
check("1 AB + 1 A", (r, ec, ac, a2), (2.25, 0, 1, 1))
r, _, _, _ = think_reward([("video", 111, 222, 999)], GS)
check("1 AB", r, 2.0)
r, _, _, _ = think_reward(
    [("video", 111, 222, 999), ("video", 111, 222, 888)], GS)
check("2 AB (same AB prefix dedup)", r, 2.0)
r, _, _, _ = think_reward([("video", 111, 999, 999)], GS)
check("1 A", r, 0.5)
r, _, _, _ = think_reward(
    [("video", 111, 999, 999), ("video", 111, 888, 888)], GS)
check("2 A (same A prefix dedup)", r, 0.5)

# 6) Beam duplicate does not double credit
r, ec, _, _ = think_reward(
    [("video", 111, 222, 333), ("video", 111, 222, 333), ("video", 111, 222, 333)], GS)
check("duplicate exact dedup", (r, ec), (8.0, 1))

# 7) Exact covers same-gold AB/A (no lower-level credit for covered gold)
r, ec, ac, a2 = think_reward(
    [("video", 111, 222, 333), ("video", 111, 222, 999), ("video", 111, 999, 999)], GS)
check("exact covers same-gold AB/A", (r, ec, ac, a2), (8.0, 1, 0, 0))

# 8) malformed SID
check("malformed -> unparsable", q_reward(None, GS), -1.0)
check("malformed final_sid", final_sid("garbage no sid here"), None)

# 9) wrong-domain SID
check("wrong domain q", q_reward(("ad", 111, 222, 333), GS), -0.25)

# 10) raw_w sanity
check("raw_w(3)", raw_w(3), 1.0)
check("raw_w(1) capped", raw_w(1), 1.40)
check("raw_w(20) floored", raw_w(20), 0.65)

# parser sanity
check("parse sid", parse_sid("<|video_begin|><s_a_1><s_b_2><s_c_3>"), ("video", 1, 2, 3))
check("final after think",
      final_sid("x<think>a</think>\n该用户: <|video_begin|><s_a_9><s_b_8><s_c_7>"),
      ("video", 9, 8, 7))

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL TESTS PASSED")
