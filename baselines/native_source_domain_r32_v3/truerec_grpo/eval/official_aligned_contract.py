"""Shared no-bridge, fixed-domain ABC3 contract for TrueRec train and eval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONTRACT_ID = "official_aligned_fixed_domain_abc3_v1"
CONTRACT_SOURCE = "USER_PROVIDED_OFFICIAL_PROTOCOL"
OFFICIAL_EVAL_SOURCE_CODE_AVAILABLE = False
OFFICIAL_COMPOSITE_FORMULA_AVAILABLE = False
ACTION_LEVELS = ("A", "B", "C")
ACTION_TOKENS = 3
TRAIN_ROLLOUT_G = 8
EVAL_BEAM_SIZE = 32
DOMAIN_TOKENS = {
    "video": "<|video_begin|>",
    "prod": "<|prod_begin|>",
    "ad": "<|ad_begin|>",
    "living": "<|living_begin|>",
}
BRIDGE_FIELD_FRAGMENTS = (
    "legacy_bridge", "bridge_prefix", "answer_bridge", "beta_bridge",
    "natural_language_bridge",
)
UNKNOWN_OFFICIAL_DECODING_DETAILS = (
    "abc_vocabulary_constraint",
    "length_penalty",
    "early_stopping",
    "eos_semantics",
    "num_return_sequences",
    "beam_score_normalization",
    "position_constrained_decoding",
    "special_logits_processors",
)


class ContractError(ValueError):
    pass


def _active(value: Any) -> bool:
    if value is None or value is False or value == "":
        return False
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return False
    return True


def validate_no_bridge_contract(config: Mapping[str, Any]) -> None:
    for key, value in config.items():
        normalized = str(key).lower()
        if (normalized == "bridge" or any(fragment in normalized for fragment in BRIDGE_FIELD_FRAGMENTS)) and _active(value):
            raise ContractError(f"active bridge field is forbidden: {key}")


@dataclass(frozen=True)
class PrefixContract:
    mode: str
    bridge: bool = False
    fixed_domain_in_context: bool = True
    domain_generated: bool = False
    action_levels: tuple[str, ...] = ACTION_LEVELS
    action_tokens: int = ACTION_TOKENS

    def __post_init__(self) -> None:
        validate_no_bridge_contract(vars(self))
        if not self.fixed_domain_in_context or self.domain_generated:
            raise ContractError("target domain must be fixed in context and never generated")
        if self.action_levels != ACTION_LEVELS or self.action_tokens != ACTION_TOKENS:
            raise ContractError("action must be exactly A/B/C, three tokens")
        if self.mode not in {"train", "eval"}:
            raise ContractError("mode must be train or eval")


def train_prefix_contract() -> PrefixContract:
    return PrefixContract("train")


def eval_prefix_contract() -> PrefixContract:
    return PrefixContract("eval")


def shared_prefix_contract_pass() -> bool:
    train, evaluation = train_prefix_contract(), eval_prefix_contract()
    return (
        train.bridge == evaluation.bridge
        and train.fixed_domain_in_context == evaluation.fixed_domain_in_context
        and train.domain_generated == evaluation.domain_generated
        and train.action_levels == evaluation.action_levels
        and train.action_tokens == evaluation.action_tokens
    )
