"""Strict, model-free validation for NoThink completion text."""

from __future__ import annotations

from dataclasses import dataclass
import re

from grpo_sid import SID_RE


DOMAIN_DECLARATIONS = {
    "video": "该用户最近喜欢的视频有:",
    "prod": "该用户最近点击了商品:",
    "ad": "该用户最近感兴趣的广告有:",
    "living": "该用户最近首次打赏了主播:",
}
TRAILING_SPECIAL_RE = re.compile(r"(?:\s*<\|(?:im_end|endoftext)\|>\s*)+$")


@dataclass(frozen=True)
class FormatValidation:
    valid: bool
    reason: str | None
    mode: str | None
    parsed_sid: tuple[str, int, int, int] | None


def _sid_tuple(match: re.Match[str]) -> tuple[str, int, int, int]:
    return (
        match.group("domain"),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def validate_nothink_completion(text: str) -> FormatValidation:
    """Accept only empty-think standard output or a strict direct-SID fallback."""
    if not isinstance(text, str):
        return FormatValidation(False, "other", None, None)
    body = TRAILING_SPECIAL_RE.sub("", text).strip()
    had_empty_think = False
    if body.startswith("<think>"):
        close = body.find("</think>", len("<think>"))
        if close < 0:
            return FormatValidation(False, "malformed_template", None, None)
        think_body = body[len("<think>"):close]
        if think_body.strip():
            match = list(SID_RE.finditer(body))
            sid = _sid_tuple(match[-1]) if match else None
            return FormatValidation(False, "nonempty_think", None, sid)
        body = body[close + len("</think>"):].strip()
        had_empty_think = True
        if "<think>" in body or "</think>" in body:
            return FormatValidation(False, "malformed_template", None, None)
    elif "<think>" in body or "</think>" in body:
        return FormatValidation(False, "malformed_template", None, None)

    match = SID_RE.fullmatch(body)
    if match:
        return FormatValidation(True, None, "direct_sid_fallback", _sid_tuple(match))

    sid_matches = list(SID_RE.finditer(body))
    if not sid_matches:
        return FormatValidation(False, "invalid_sid", None, None)
    final_match = sid_matches[-1]
    parsed_sid = _sid_tuple(final_match)
    suffix = body[final_match.end():]
    prefix = body[:final_match.start()]
    if suffix.strip():
        return FormatValidation(False, "malformed_template", None, parsed_sid)
    expected = DOMAIN_DECLARATIONS[parsed_sid[0]]
    if had_empty_think and prefix.rstrip() == expected:
        return FormatValidation(True, None, "branch", parsed_sid)
    return FormatValidation(False, "unexpected_prose_before_sid", None, parsed_sid)


def effective_scalar_reward(base_reward: float, validation: FormatValidation) -> float:
    """Format violation is an invalid completion regardless of its parsed tail SID."""
    return float(base_reward) if validation.valid else -1.0
