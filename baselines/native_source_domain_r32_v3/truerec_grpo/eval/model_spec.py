"""Checkpoint provenance only; model family cannot alter prompt serialization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from official_aligned_contract import CONTRACT_ID, ContractError


MODEL_FAMILIES = ("beta", "beta_gamma", "truerec")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    checkpoint_path: str
    base_model_path: str
    adapter_path: str | None
    tokenizer_path: str
    eval_contract_id: str = CONTRACT_ID

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILIES:
            raise ContractError(f"unsupported model family: {self.family}")
        if not all((self.name, self.checkpoint_path, self.base_model_path, self.tokenizer_path)):
            raise ContractError("ModelSpec provenance paths cannot be empty")
        if self.eval_contract_id != CONTRACT_ID:
            raise ContractError("model family cannot select a different eval prefix contract")

    def provenance(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "family": self.family,
            "checkpoint_path": str(Path(self.checkpoint_path)),
            "base_model_path": str(Path(self.base_model_path)),
            "adapter_path": str(Path(self.adapter_path)) if self.adapter_path else None,
            "tokenizer_path": str(Path(self.tokenizer_path)),
            "eval_contract_id": self.eval_contract_id,
        }
