# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1 smoke: SID parsing + hierarchical reward (CPU-only, testable)."""
import re

SID_RE = re.compile(r"<\|(?P<domain>ad|video|prod|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
THINK_CLOSE = "</think>"


def parse_sid(text: str):
    """Return (domain, a, b, c) tuple or None if no complete SID found."""
    if not text:
        return None
    m = SID_RE.search(text)
    if not m:
        return None
    return (m.group("domain"), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def final_sid(text: str):
    """Extract the FINAL SID (after </think> if present, else last complete SID).
    Returns (domain, a, b, c) or None."""
    if not text:
        return None
    tail = text
    idx = text.rfind(THINK_CLOSE)
    if idx >= 0:
        tail = text[idx + len(THINK_CLOSE):]
    matches = list(SID_RE.finditer(tail))
    if not matches:
        # fallback: last complete SID anywhere in text
        matches = list(SID_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return (m.group("domain"), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def all_sids(text: str):
    """All complete SIDs in text (deduped, order-preserved)."""
    out = []
    seen = set()
    for m in SID_RE.finditer(text):
        t = (m.group("domain"), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------- NoThink reward q(y, G) ----------------

def q_reward(sid, gold_set):
    """Hierarchical per-SID reward against a gold set of complete SIDs.
    gold_set: iterable of (domain, a, b, c) tuples.
    Returns float in {-1, -0.25, 0, 0.5, 2, 8}. Mutually exclusive levels."""
    if sid is None:
        return -1.0
    dom, a, b, c = sid
    exact = False
    ab = False
    a_ok = False
    dom_ok = False
    for gd, ga, gb, gc in gold_set:
        if (dom, a, b, c) == (gd, ga, gb, gc):
            exact = True
            break
        if dom == gd and a == ga and b == gb:
            ab = True
        if dom == gd and a == ga:
            a_ok = True
        if dom == gd:
            dom_ok = True
    if exact:
        return 8.0
    if ab:
        return 2.0
    if a_ok:
        return 0.5
    if dom_ok:
        return 0.0
    return -0.25


# ---------------- Think hierarchical credits ----------------

def think_credits(beam_sids, gold_set):
    """Hierarchical credits for a Beam32 output (per-gold coverage).
    Returns (credits_sorted_desc, exact_count, ab_count, a_count)."""
    golds = list(gold_set)
    exact_golds = set()
    ab_golds = set()
    a_golds = set()
    for g in golds:
        gd, ga, gb, gc = g
        exact = any(s == g for s in beam_sids)
        ab = any(s is not None and s[0] == gd and s[1] == ga and s[2] == gb for s in beam_sids)
        a_ok = any(s is not None and s[0] == gd and s[1] == ga for s in beam_sids)
        if exact:
            exact_golds.add(g)
        elif ab:
            ab_golds.add(g)
        elif a_ok:
            a_golds.add(g)
    # count by distinct prefixes (dedupe)
    ab_prefixes = {(g[0], g[1], g[2]) for g in ab_golds}
    a_prefixes = {(g[0], g[1]) for g in a_golds}
    n_exact = len(exact_golds)
    n_ab = len(ab_prefixes)
    n_a = len(a_prefixes)
    credits = [8.0] * n_exact + [2.0] * n_ab + [0.5] * n_a
    credits.sort(reverse=True)
    return credits, n_exact, n_ab, n_a


def think_reward(beam_sids, gold_set):
    """R = sum_j c_j * 0.5^(j-1) with credits sorted desc. 0 if no hits."""
    credits, ec, ac, acnt = think_credits(beam_sids, gold_set)
    if not credits:
        return 0.0, ec, ac, acnt
    r = sum(c * (0.5 ** j) for j, c in enumerate(credits))
    return r, ec, ac, acnt


# ---------------- difficulty weight (record only) ----------------

def raw_w(K: int) -> float:
    """clip((3/K)^0.35, 0.65, 1.40). Only recorded this round."""
    v = (3.0 / K) ** 0.35 if K > 0 else 1.0
    return max(0.65, min(1.40, v))
