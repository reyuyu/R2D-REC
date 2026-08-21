"""Deterministic diagnostic-only parser for interest counts and grounding."""

from __future__ import annotations

import re


SID_RE = re.compile(
    r"<\|(?P<domain>ad|video|prod|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
INTEREST_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*[ \t]*)?"
    r"【[ \t]*兴趣归纳[ \t]*】(?:[ \t]*\*\*)?[ \t]*$"
)
NEXT_MAJOR_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:#{1,6})[ \t]+(?:\*\*)?[^\n]+|"
    r"(?:\*\*)?【(?![ \t]*兴趣归纳[ \t]*】)[^\n】]+】(?:\*\*)?)[ \t]*$"
)
BULLET_RE = re.compile(
    r"(?ms)^[ \t]*(?P<number>\d{1,2})[.、][ \t]*(?P<body>.*?)"
    r"(?=^[ \t]*\d{1,2}[.、][ \t]+|\Z)"
)


def _extract_sids(text: str) -> set[tuple[str, int, int, int]]:
    return {
        (match.group("domain"), int(match.group("a")), int(match.group("b")), int(match.group("c")))
        for match in SID_RE.finditer(text or "")
    }


def interest_diagnostics(completion: str, prompt: str) -> dict:
    """Return Raw N, Grounded N and Coverage; never used by reward or loss."""
    cot = (completion or "").replace("\r\n", "\n").replace("\r", "\n")
    heading = INTEREST_HEADING_RE.search(cot)
    if heading is None:
        return {"raw_n": 0, "grounded_n": 0, "coverage": None, "parser_success": False}
    start = heading.end()
    ends = []
    think_close = cot.find("</think>", start)
    if think_close >= 0:
        ends.append(think_close)
    next_heading = NEXT_MAJOR_HEADING_RE.search(cot, start)
    if next_heading is not None:
        ends.append(next_heading.start())
    section = cot[start:min(ends) if ends else len(cot)].strip()
    prompt_sids = _extract_sids(prompt)
    bullets = [match.group("body") for match in BULLET_RE.finditer(section)]
    grounded = sum(bool(_extract_sids(body) & prompt_sids) for body in bullets)
    raw = len(bullets)
    return {
        "raw_n": raw,
        "grounded_n": grounded,
        "coverage": grounded / raw if raw else None,
        "parser_success": bool(raw),
    }
