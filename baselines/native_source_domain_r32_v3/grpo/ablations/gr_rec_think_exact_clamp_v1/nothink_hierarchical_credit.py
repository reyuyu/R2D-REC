"""Conditional hierarchy-state credit and NoThink decision-token placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


HIERARCHY_SCALE = 8.0
STAGE_INCREMENTS = (0.25, 0.5, 1.5, 6.0)
TEXT_DOMAIN_SPANS = {
    "video": "视频",
    "prod": "商品",
    "ad": "广告",
    "living": "主播",
}
DOMAIN_DECLARATIONS = {
    "video": "该用户最近喜欢的视频有: ",
    "prod": "该用户最近点击了商品: ",
    "ad": "该用户最近感兴趣的广告有: ",
    "living": "该用户最近首次打赏了主播: ",
}


@dataclass(frozen=True)
class HierarchyState:
    valid: bool
    domain_correct: bool
    a_correct: bool
    ab_correct: bool
    exact: bool


@dataclass(frozen=True)
class DomainTextAlignment:
    """Token-level alignment between the semantic domain decision and final SID."""

    text_domain: str | None
    text_domain_token_position: int | None
    sid_domain: str
    sid_domain_token_position: int
    sid_token_positions: tuple[int, int, int, int]
    valid: bool
    failure: str | None

    @property
    def hierarchy_token_positions(self) -> tuple[int | None, int, int, int]:
        return (
            self.text_domain_token_position,
            self.sid_token_positions[1],
            self.sid_token_positions[2],
            self.sid_token_positions[3],
        )


@dataclass(frozen=True)
class DomainCommitmentAlignment:
    """Candidate-level earliest Domain commitment with direct-SID fallback."""

    commitment_domain: str
    commitment_token_position: int
    sid_domain_token_position: int
    sid_token_positions: tuple[int, int, int, int]
    mode: str
    eligible: bool = True
    failure: str | None = None

    @property
    def hierarchy_token_positions(self) -> tuple[int, int, int, int]:
        return (
            self.commitment_token_position,
            self.sid_token_positions[1],
            self.sid_token_positions[2],
            self.sid_token_positions[3],
        )


def hierarchy_state(final_sid, gold_sids: Iterable[Sequence], target_domain: str) -> HierarchyState:
    """Derive monotonic final-SID correctness flags without using scalar reward."""
    gold = {tuple(value) for value in gold_sids if value is not None and len(value) == 4}
    valid = final_sid is not None and len(final_sid) == 4
    if not valid:
        return HierarchyState(False, False, False, False, False)
    sid = tuple(final_sid)
    domain_correct = sid[0] == target_domain
    a_correct = domain_correct and any(sid[:2] == item[:2] for item in gold)
    ab_correct = a_correct and any(sid[:3] == item[:3] for item in gold)
    exact = ab_correct and sid in gold
    return HierarchyState(valid, domain_correct, a_correct, ab_correct, exact)


def conditional_hierarchical_credits(
    states: Sequence[HierarchyState],
    domain_eligible: Sequence[bool] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Return per-candidate (Domain, A, B, C) credit for one NoThink G8."""
    if len(states) != 8:
        raise ValueError("conditional hierarchical credit requires exactly one G8")
    credits = [[0.0, 0.0, 0.0, 0.0] for _ in states]
    if domain_eligible is None:
        domain_eligible = [state.valid for state in states]
    if len(domain_eligible) != len(states):
        raise ValueError("Domain eligibility must align one-to-one with G8 states")
    domain_indices = [
        index for index, (state, eligible) in enumerate(zip(states, domain_eligible))
        if state.valid and eligible
    ]
    domain_indicators = [float(states[index].domain_correct) for index in domain_indices]
    if domain_indicators and len(set(domain_indicators)) > 1:
        mean = sum(domain_indicators) / len(domain_indicators)
        for index, indicator in zip(domain_indices, domain_indicators):
            credits[index][0] = STAGE_INCREMENTS[0] * (indicator - mean) / HIERARCHY_SCALE

    stages = (
        (lambda state: state.domain_correct, lambda state: state.a_correct),
        (lambda state: state.a_correct, lambda state: state.ab_correct),
        (lambda state: state.ab_correct, lambda state: state.exact),
    )
    for column, ((eligible, correct), increment) in enumerate(
        zip(stages, STAGE_INCREMENTS[1:]), 1
    ):
        indices = [index for index, state in enumerate(states) if eligible(state)]
        indicators = [float(correct(states[index])) for index in indices]
        if not indicators or len(set(indicators)) == 1:
            continue
        mean = sum(indicators) / len(indicators)
        for index, indicator in zip(indices, indicators):
            credits[index][column] = increment * (indicator - mean) / HIERARCHY_SCALE
    return [tuple(row) for row in credits]


def _subsequence_starts(
    values: Sequence[int], pattern: Sequence[int], start: int, stop: int,
) -> list[int]:
    return [
        index
        for index in range(start, stop - len(pattern) + 1)
        if list(values[index:index + len(pattern)]) == list(pattern)
    ]


def _declaration_spec(tokenizer) -> tuple[dict[str, list[int]], int]:
    declarations = {
        domain: [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        for domain, text in DOMAIN_DECLARATIONS.items()
    }
    sequences = list(declarations.values())
    if any(not sequence for sequence in sequences):
        raise RuntimeError("Domain declaration must encode to at least one token")
    branch_index = 0
    for column in zip(*sequences):
        if len(set(column)) != 1:
            break
        branch_index += 1
    if any(branch_index >= len(sequence) for sequence in sequences):
        raise RuntimeError("one Domain declaration is a token-prefix of another")
    return declarations, branch_index


def locate_domain_commitment_token(
    completion_ids: Sequence[int], final_sid, tokenizer,
) -> DomainCommitmentAlignment:
    """Prefer the standard declaration branch; otherwise use explicit SID Domain."""
    sid_positions = find_final_sid_token_positions(completion_ids, final_sid, tokenizer)
    sid_domain = str(final_sid[0])
    ids = [int(value) for value in completion_ids]
    close_ids = [int(value) for value in tokenizer.encode("</think>", add_special_tokens=False)]
    close_starts = _subsequence_starts(ids, close_ids, 0, sid_positions[0]) if close_ids else []
    if close_starts:
        search_start = close_starts[-1] + len(close_ids)
        declarations, branch_index = _declaration_spec(tokenizer)
        matching_starts = _subsequence_starts(
            ids, declarations[sid_domain], search_start, sid_positions[0]
        )
        if len(matching_starts) == 1:
            return DomainCommitmentAlignment(
                sid_domain,
                matching_starts[0] + branch_index,
                sid_positions[0],
                sid_positions,
                "branch",
            )
    return DomainCommitmentAlignment(
        sid_domain,
        sid_positions[0],
        sid_positions[0],
        sid_positions,
        "direct_sid_fallback",
        failure="standard_declaration_not_uniquely_matched",
    )


def apply_domain_text_alignment_gate(
    credits: Sequence[tuple[float, float, float, float]], alignment_valid: bool,
) -> list[tuple[float, float, float, float]]:
    """Disable only the Domain stage when a G8 text/SID alignment is unsafe."""
    if alignment_valid:
        return [tuple(row) for row in credits]
    return [(0.0, row[1], row[2], row[3]) for row in credits]


def find_final_sid_token_positions(
    completion_ids: Sequence[int], final_sid, tokenizer,
) -> tuple[int, int, int, int]:
    """Find Domain/A/B/C in the last exact contiguous final-SID token block."""
    if final_sid is None or len(final_sid) != 4:
        raise ValueError("a valid final SID is required")
    domain, a, b, c = final_sid
    token_texts = (
        f"<|{domain}_begin|>",
        f"<s_a_{int(a)}>",
        f"<s_b_{int(b)}>",
        f"<s_c_{int(c)}>",
    )
    block = []
    for text in token_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"final SID token must encode singly: {text!r} -> {encoded}")
        block.append(int(encoded[0]))
    ids = [int(value) for value in completion_ids]
    for start in range(len(ids) - 4, -1, -1):
        if ids[start:start + 4] == block:
            return start, start + 1, start + 2, start + 3
    raise RuntimeError("parsed final SID has no matching contiguous token block")


def locate_text_domain_token(
    completion_ids: Sequence[int], final_sid, tokenizer,
) -> DomainTextAlignment:
    """Locate the last natural-language Domain span before the final SID block.

    Search is token-native and restricted to the region after the last
    ``</think>`` and before the final contiguous Domain/A/B/C SID block.
    Missing or mismatched text is returned as an invalid alignment; callers
    must not fall back to the SID Domain token.
    """
    sid_positions = find_final_sid_token_positions(completion_ids, final_sid, tokenizer)
    sid_domain = str(final_sid[0])
    ids = [int(value) for value in completion_ids]
    close_ids = [int(value) for value in tokenizer.encode("</think>", add_special_tokens=False)]
    if not close_ids:
        raise RuntimeError("</think> must encode to at least one token")
    close_starts = [
        start
        for start in range(0, sid_positions[0] - len(close_ids) + 1)
        if ids[start:start + len(close_ids)] == close_ids
    ]
    if not close_starts:
        return DomainTextAlignment(
            None, None, sid_domain, sid_positions[0], sid_positions,
            False, "missing_think_close_before_final_sid",
        )
    search_start = close_starts[-1] + len(close_ids)

    token_to_domain = {}
    for domain, span in TEXT_DOMAIN_SPANS.items():
        encoded = tokenizer.encode(span, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"text Domain span must encode singly: {span!r} -> {encoded}")
        token_id = int(encoded[0])
        if token_id in token_to_domain:
            raise RuntimeError("text Domain spans must have distinct token ids")
        token_to_domain[token_id] = domain

    occurrences = [
        (position, token_to_domain[token_id])
        for position, token_id in enumerate(ids[search_start:sid_positions[0]], search_start)
        if token_id in token_to_domain
    ]
    if not occurrences:
        return DomainTextAlignment(
            None, None, sid_domain, sid_positions[0], sid_positions,
            False, "missing_text_domain_before_final_sid",
        )
    position, text_domain = occurrences[-1]
    valid = text_domain == sid_domain
    return DomainTextAlignment(
        text_domain, position, sid_domain, sid_positions[0], sid_positions,
        valid, None if valid else "text_sid_domain_mismatch",
    )
