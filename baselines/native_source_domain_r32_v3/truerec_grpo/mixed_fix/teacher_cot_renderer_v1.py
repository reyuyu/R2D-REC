"""Tokenizer-only Teacher-CoT context serialization contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol


NOTHINK_MARKER = "/no_think"
THINK_MARKER = "/think"


class Renderer(Protocol):
    def prompt_ids(self, system: str, user_content: str) -> list[int]: ...
    def encode(self, text: str) -> list[int]: ...
    def full_sft_ids(self, system: str, user_content: str, response: str) -> list[int]: ...


@dataclass(frozen=True)
class TokenParity:
    passed: bool
    candidate_length: int
    reference_prefix_length: int
    first_mismatch_index: int | None


def transform_terminal_route(user_content_nothink: str) -> str:
    if not user_content_nothink.endswith(NOTHINK_MARKER):
        raise ValueError("user content must end with terminal /no_think")
    prefix = user_content_nothink[: -len(NOTHINK_MARKER)]
    result = prefix + THINK_MARKER
    if result[: -len(THINK_MARKER)] != prefix:
        raise AssertionError("non-route prompt bytes changed")
    return result


def validate_teacher_cot(teacher_cot: str) -> None:
    if not teacher_cot or not teacher_cot.startswith("<think>"):
        raise ValueError("Teacher COT must start with <think>")
    first = teacher_cot.find("</think>")
    if first < 0:
        raise ValueError("Teacher COT lacks </think>")
    if first + len("</think>") != len(teacher_cot):
        raise ValueError("Teacher COT contains suffix after first </think>")


def discover_canonical_separator(
    rows: Iterable[tuple[str, str, str]],
) -> str:
    """Inspect (response, teacher_cot, domain_token) triples."""
    variants = set()
    count = 0
    for response, teacher_cot, domain_token in rows:
        count += 1
        validate_teacher_cot(teacher_cot)
        if not response.startswith(teacher_cot):
            raise ValueError("reference response does not start with Teacher COT")
        domain_start = response.find(domain_token, len(teacher_cot))
        if domain_start < 0:
            raise ValueError("reference response lacks fixed domain token")
        variants.add(response[len(teacher_cot) : domain_start])
    if count == 0:
        raise ValueError("separator audit requires reference rows")
    if len(variants) != 1:
        raise ValueError(f"separator variants are not canonical: {sorted(map(repr, variants))}")
    return next(iter(variants))


def think_rl_context_ids(
    renderer: Renderer,
    system: str,
    user_content_think: str,
    teacher_cot: str,
    fixed_domain_token: str,
    canonical_separator: str,
) -> list[int]:
    if not user_content_think.endswith(THINK_MARKER):
        raise ValueError("Think user content must end with /think")
    validate_teacher_cot(teacher_cot)
    domain_ids = renderer.encode(fixed_domain_token)
    if len(domain_ids) != 1:
        raise ValueError("fixed domain token must encode to exactly one token")
    context = renderer.prompt_ids(system, user_content_think) + renderer.encode(
        teacher_cot + canonical_separator + fixed_domain_token
    )
    if not context or context[-1] != domain_ids[0]:
        raise ValueError("Think context must end at fixed domain token")
    return context


def audit_sft_prefix_parity(
    renderer: Renderer,
    system: str,
    user_content_think: str,
    reference_response: str,
    context_ids: list[int],
) -> TokenParity:
    reference = renderer.full_sft_ids(system, user_content_think, reference_response)
    prefix = reference[: len(context_ids)]
    mismatch = next(
        (index for index, (left, right) in enumerate(zip(context_ids, prefix)) if left != right),
        None,
    )
    passed = len(prefix) == len(context_ids) and mismatch is None
    if not passed and mismatch is None:
        mismatch = min(len(prefix), len(context_ids))
    return TokenParity(passed, len(context_ids), len(prefix), mismatch)


def assert_action_boundary(
    renderer: Renderer, context_ids: list[int], fixed_domain_token: str, abc_text: str
) -> dict[str, Any]:
    domain_ids = renderer.encode(fixed_domain_token)
    action_ids = renderer.encode(abc_text)
    if len(domain_ids) != 1 or context_ids[-1:] != domain_ids:
        raise ValueError("fixed domain boundary contract failed")
    if len(action_ids) != 3:
        raise ValueError("ABC action must encode to exactly three tokens")
    return {
        "domain_token_id": domain_ids[0],
        "action_token_ids": action_ids,
        "A_causal_logit_index": len(context_ids) - 1,
    }
