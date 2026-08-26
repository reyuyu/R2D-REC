"""Thin adapter over the frozen BetaGammaRenderer fixed-domain context method."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from official_aligned_contract import CONTRACT_ID, ContractError, DOMAIN_TOKENS


RENDERER_CONTRACT_ID = "beta_gamma_renderer_qwen3_nothink_fixed_domain_v1"


@dataclass(frozen=True)
class RenderedOfficialPrefix:
    recommendation_group_id: str
    target_domain: str
    fixed_domain_token: str
    context_ids: tuple[int, ...]
    action_start_position: int
    renderer_contract_id: str = RENDERER_CONTRACT_ID
    eval_contract_id: str = CONTRACT_ID


def render_official_aligned_prefix(record: dict, renderer) -> RenderedOfficialPrefix:
    domain = str(record["target_domain"])
    expected = DOMAIN_TOKENS.get(domain)
    if expected is None or record["fixed_domain_token"] != expected:
        raise ContractError("fixed domain token does not match the record target domain")
    context_ids = tuple(renderer.rl_context_ids(record["system"], record["user_content_nothink"], expected))
    domain_ids: Sequence[int] = renderer.encode(expected)
    if len(domain_ids) != 1 or not context_ids or context_ids[-1] != int(domain_ids[0]):
        raise ContractError("context must end immediately after the one-token fixed domain")
    return RenderedOfficialPrefix(
        str(record["recommendation_group_id"]), domain, expected, context_ids, len(context_ids)
    )
