"""Fixed target-domain contract for Think CoT -> Beam32 continuation."""
from __future__ import annotations

import re
from collections.abc import Iterable

DOMAIN_PREFIX = {
    "video": "<|video_begin|>",
    "prod": "<|prod_begin|>",
    "ad": "<|ad_begin|>",
    "living": "<|living_begin|>",
}
_ABC_TOKEN_PATTERNS = (
    re.compile(r"<s_a_(\d+)>").fullmatch,
    re.compile(r"<s_b_(\d+)>").fullmatch,
    re.compile(r"<s_c_(\d+)>").fullmatch,
)


def domain_prefix(target_domain: str) -> str:
    """Return the sole fixed prefix for dataset.target_domain; fail closed."""
    try:
        return DOMAIN_PREFIX[target_domain]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"UNKNOWN_TARGET_DOMAIN: {target_domain!r}") from exc


def validate_gold_domains(gold_set: Iterable[tuple], target_domain: str) -> None:
    """Check consistency without using Gold to select the Beam domain."""
    domain_prefix(target_domain)
    mismatches = sorted({
        item[0] for item in gold_set if item and item[0] != target_domain
    })
    if mismatches:
        raise ValueError(
            f"GOLD_DOMAIN_MISMATCH: target_domain={target_domain!r}, "
            f"gold_domains={mismatches!r}"
        )


def build_fixed_domain_beam_input(
    tokenizer, prompt_ids, cot_ids, target_domain: str,
):
    """Append exactly the encoded fixed-domain prefix to prompt + closed CoT."""
    prefix_text = domain_prefix(target_domain)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if not prefix_ids:
        raise ValueError(f"EMPTY_DOMAIN_PREFIX_TOKENIZATION: {target_domain!r}")
    return (
        list(prompt_ids) + list(cot_ids) + list(prefix_ids),
        prefix_text,
        list(prefix_ids),
    )


def parse_strict_abc3_ids(tokenizer, generated_ids, target_domain: str):
    """Parse exactly three raw A/B/C token IDs under the fixed target domain."""
    domain_prefix(target_domain)
    ids = list(generated_ids or [])
    if len(ids) != 3:
        return None
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    if isinstance(tokens, str):
        tokens = [tokens]
    if len(tokens) != 3:
        return None
    matches = [pattern(token) for pattern, token in zip(_ABC_TOKEN_PATTERNS, tokens)]
    if any(match is None for match in matches):
        return None
    return (target_domain, *(int(match.group(1)) for match in matches))


def parse_fixed_domain_beam_sid(tokenizer, generated_ids, target_domain: str):
    """Compatibility name for the production strict raw-ID ABC3 parser."""
    return parse_strict_abc3_ids(tokenizer, generated_ids, target_domain)
