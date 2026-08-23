"""Fixed target-domain contract for Think CoT -> Beam32 continuation."""
from __future__ import annotations

import re
from collections.abc import Iterable

from grpo_sid import final_sid


DOMAIN_PREFIX = {
    "video": "<|video_begin|>",
    "prod": "<|prod_begin|>",
    "ad": "<|ad_begin|>",
    "living": "<|living_begin|>",
}
_LEADING_DOMAIN_PREFIX_RE = re.compile(
    r"^\s*<\|(?:video|prod|ad|living)_begin\|>"
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


def parse_fixed_domain_beam_sid(generated_text: str, target_domain: str):
    """Parse A/B/C continuation while dataset.target_domain stays authoritative."""
    prefix_text = domain_prefix(target_domain)
    continuation = _LEADING_DOMAIN_PREFIX_RE.sub(
        "", generated_text or "", count=1,
    )
    parsed = final_sid(prefix_text + continuation)
    if parsed is None or parsed[0] != target_domain:
        return None
    return parsed
