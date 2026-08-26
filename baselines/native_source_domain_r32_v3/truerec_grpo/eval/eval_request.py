"""Validated request for an Official-Aligned Beam32 evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from model_spec import ModelSpec
from official_aligned_contract import ACTION_TOKENS, CONTRACT_ID, EVAL_BEAM_SIZE, ContractError, validate_no_bridge_contract
from official_aligned_renderer import RENDERER_CONTRACT_ID


DECODER_CONTRACT_ID = "beam32_abc3_frontend_v1"
SPLITS = ("probe20", "dev512", "final2048")


@dataclass(frozen=True)
class OfficialAlignedEvalRequest:
    model_spec: ModelSpec
    split: str
    beam_size: int = EVAL_BEAM_SIZE
    fixed_domain: bool = True
    domain_generated: bool = False
    action_tokens: int = ACTION_TOKENS
    bridge: bool = False
    renderer_contract_id: str = RENDERER_CONTRACT_ID
    decoder_contract_id: str = DECODER_CONTRACT_ID
    eval_contract_id: str = CONTRACT_ID

    def __post_init__(self) -> None:
        validate_no_bridge_contract(vars(self))
        if self.split not in SPLITS:
            raise ContractError(f"unsupported frozen split: {self.split}")
        if self.beam_size != 32 or not self.fixed_domain or self.domain_generated or self.action_tokens != 3:
            raise ContractError("eval requires fixed-domain Beam32 ABC3")
        if self.eval_contract_id != CONTRACT_ID or self.model_spec.eval_contract_id != CONTRACT_ID:
            raise ContractError("model and request must share the official-aligned contract")


@dataclass(frozen=True)
class FrozenSplitInterface:
    name: str
    group_ids: tuple[str, ...]
    checkpoint_selection_allowed: bool
    optimizer_allowed: bool = False
    backward_allowed: bool = False


def frozen_split_interface(name: str, group_ids: Sequence[str], dev_group_ids: Sequence[str] = ()) -> FrozenSplitInterface:
    values = tuple(group_ids)
    expected = {"probe20": 20, "dev512": 512, "final2048": 2048}
    if name not in expected or len(values) != expected[name] or len(set(values)) != len(values):
        raise ContractError("frozen split name/count/uniqueness mismatch")
    if name == "probe20" and not set(values) <= set(dev_group_ids):
        raise ContractError("Probe20 must be a Dev512 subset")
    return FrozenSplitInterface(name, values, checkpoint_selection_allowed=(name == "dev512"))
