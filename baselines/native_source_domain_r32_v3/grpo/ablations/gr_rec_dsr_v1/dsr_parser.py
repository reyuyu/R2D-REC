# -*- coding: utf-8 -*-
"""Deterministic CPU parser for the DSR grounded-interest objective."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


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


Sid = tuple[str, int, int, int]


def sid_from_match(match: re.Match[str]) -> Sid:
    return (
        match.group("domain"),
        int(match.group("a")),
        int(match.group("b")),
        int(match.group("c")),
    )


def canonical_sid(sid: Sid) -> str:
    domain, a, b, c = sid
    return f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def extract_sids(text: str) -> tuple[Sid, ...]:
    """Return complete SIDs in first-seen order, without duplicates."""
    seen: set[Sid] = set()
    values: list[Sid] = []
    for match in SID_RE.finditer(text or ""):
        sid = sid_from_match(match)
        if sid not in seen:
            seen.add(sid)
            values.append(sid)
    return tuple(values)


def _interest_title(body: str) -> str:
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    first_line = re.sub(r"^[\-*#\s]+", "", first_line)
    first_line = first_line.replace("**", "").strip()
    title = re.split(r"[：:]", first_line, maxsplit=1)[0].strip()
    return title


@dataclass(frozen=True)
class InterestBullet:
    number: int
    title: str
    text: str
    evidence: tuple[Sid, ...]
    grounded_evidence: tuple[Sid, ...]

    @property
    def grounded(self) -> bool:
        return bool(self.grounded_evidence)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "text": self.text,
            "evidence": [canonical_sid(value) for value in self.evidence],
            "grounded_evidence": [canonical_sid(value) for value in self.grounded_evidence],
            "grounded": self.grounded,
        }


@dataclass(frozen=True)
class InterestParseResult:
    section_found: bool
    parser_success: bool
    section_text: str
    bullets: tuple[InterestBullet, ...]
    numbering: tuple[int, ...]
    numbering_sequential: bool

    @property
    def bullet_count(self) -> int:
        return len(self.bullets)

    @property
    def grounded_bullets(self) -> tuple[InterestBullet, ...]:
        return tuple(item for item in self.bullets if item.grounded)

    @property
    def grounded_count(self) -> int:
        return len(self.grounded_bullets)

    def to_dict(self) -> dict:
        return {
            "section_found": self.section_found,
            "parser_success": self.parser_success,
            "section_text": self.section_text,
            "bullet_count": self.bullet_count,
            "grounded_count": self.grounded_count,
            "numbering": list(self.numbering),
            "numbering_sequential": self.numbering_sequential,
            "bullets": [item.to_dict() for item in self.bullets],
        }


def _section_bounds(cot: str) -> tuple[int, int] | None:
    heading = INTEREST_HEADING_RE.search(cot)
    if heading is None:
        return None
    start = heading.end()
    candidates = []
    think_close = cot.find("</think>", start)
    if think_close >= 0:
        candidates.append(think_close)
    next_heading = NEXT_MAJOR_HEADING_RE.search(cot, start)
    if next_heading is not None:
        candidates.append(next_heading.start())
    end = min(candidates) if candidates else len(cot)
    return start, end


def parse_interest_section(cot: str, prompt: str) -> InterestParseResult:
    """Parse numbered interests and retain only prompt-grounded SID evidence.

    Hallucinated complete SIDs remain visible in ``evidence`` for auditing, but
    never contribute to grounding or evidence diversity.
    """
    cot = (cot or "").replace("\r\n", "\n").replace("\r", "\n")
    prompt_sids = set(extract_sids(prompt or ""))
    bounds = _section_bounds(cot)
    if bounds is None:
        return InterestParseResult(False, False, "", (), (), False)
    start, end = bounds
    section = cot[start:end].strip()
    bullets: list[InterestBullet] = []
    numbers: list[int] = []
    for match in BULLET_RE.finditer(section):
        number = int(match.group("number"))
        body = match.group("body").strip()
        evidence = extract_sids(body)
        grounded = tuple(value for value in evidence if value in prompt_sids)
        bullets.append(InterestBullet(
            number=number,
            title=_interest_title(body),
            text=body,
            evidence=evidence,
            grounded_evidence=grounded,
        ))
        numbers.append(number)
    sequential = bool(numbers) and numbers == list(range(numbers[0], numbers[0] + len(numbers)))
    return InterestParseResult(
        section_found=True,
        parser_success=bool(bullets),
        section_text=section,
        bullets=tuple(bullets),
        numbering=tuple(numbers),
        numbering_sequential=sequential,
    )
