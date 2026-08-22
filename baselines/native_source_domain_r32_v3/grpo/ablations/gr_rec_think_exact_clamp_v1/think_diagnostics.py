"""Deterministic shared parser for Think interest units and grounding."""

from __future__ import annotations

from dataclasses import dataclass
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

@dataclass(frozen=True)
class InterestUnit:
    index: int
    raw_text: str
    normalized_text: str
    evidence_sids: tuple[str, ...]
    grounded_evidence_sids: tuple[str, ...]


@dataclass(frozen=True)
class InterestParse:
    parser_success: bool
    units: tuple[InterestUnit, ...]
    failure_reason: str | None = None



def extract_sids(text: str) -> set[str]:
    return {match.group(0) for match in SID_RE.finditer(text or "")}

def normalize_interest_text(text: str) -> str:
    value = SID_RE.sub(" ", text or "")
    value = re.sub(r"(?m)^[ \t]*\d{1,2}[.、][ \t]*", "", value)
    value = re.sub(r"(?:\*\*|__|\x60|#{1,6})", "", value)
    value = re.sub(r"</?think>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n:：;；,，。")
    return value


def extract_interest_units(cot: str, prompt: str | None = None) -> InterestParse:
    """Parse numbered interest units from the canonical interest section."""
    text = (cot or "").replace("\r\n", "\n").replace("\r", "\n")
    heading = INTEREST_HEADING_RE.search(text)
    if heading is None:
        return InterestParse(False, (), "missing_interest_heading")
    start = heading.end()
    ends: list[int] = []
    think_close = text.find("</think>", start)
    if think_close >= 0:
        ends.append(think_close)
    next_heading = NEXT_MAJOR_HEADING_RE.search(text, start)
    if next_heading is not None:
        ends.append(next_heading.start())
    section = text[start:min(ends) if ends else len(text)].strip()
    prompt_sids = extract_sids(prompt or "")
    units = []
    for match in BULLET_RE.finditer(section):
        item = match.group("body").strip()
        evidence = tuple(sorted(extract_sids(item)))
        grounded = tuple(sid for sid in evidence if sid in prompt_sids)
        units.append(InterestUnit(
            index=int(match.group("number")),
            raw_text=item,
            normalized_text=normalize_interest_text(item),
            evidence_sids=evidence,
            grounded_evidence_sids=grounded,
        ))
    if not units:
        return InterestParse(False, (), "empty_interest_section")
    return InterestParse(True, tuple(units))



def interest_diagnostics(completion: str, prompt: str) -> dict:
    """Return Raw N, Grounded N and Coverage; never used by reward or loss."""
    parsed = extract_interest_units(completion, prompt)
    raw = len(parsed.units)
    grounded = sum(bool(unit.grounded_evidence_sids) for unit in parsed.units)
    return {
        "raw_n": raw,
        "grounded_n": grounded,
        "coverage": grounded / raw if raw else None,
        "parser_success": parsed.parser_success,
    }
