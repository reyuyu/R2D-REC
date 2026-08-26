"""Business-group evaluation records with complete multi-positive Gold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from beam32_contract import BeamCandidate
from official_aligned_contract import ContractError


@dataclass(frozen=True)
class EvaluationBusinessSample:
    recommendation_group_id: str
    target_domain: str
    all_gold_sids: tuple[str, ...]
    all_gold_abc: tuple[str, ...]
    candidates: tuple[BeamCandidate, ...]

    def __post_init__(self) -> None:
        if not self.recommendation_group_id or not self.all_gold_sids or not self.all_gold_abc:
            raise ContractError("evaluation unit needs group id and complete multi-positive Gold")
        if len(set(self.all_gold_sids)) != len(self.all_gold_sids) or len(set(self.all_gold_abc)) != len(self.all_gold_abc):
            raise ContractError("Gold values must be unique")
        if len(self.candidates) != 32 or any(item.recommendation_group_id != self.recommendation_group_id for item in self.candidates):
            raise ContractError("one evaluation business sample must contain its own Beam32")


def deduplicate_source_rows(rows: Iterable[dict]) -> tuple[dict, ...]:
    """Collapse Think/NoThink source rows to one group without discarding Gold."""
    groups: dict[str, dict] = {}
    for row in rows:
        group_id = str(row["recommendation_group_id"])
        normalized = {
            "recommendation_group_id": group_id,
            "target_domain": row["target_domain"],
            "all_gold_sids": tuple(dict.fromkeys(row["all_gold_sids"])),
            "all_gold_abc": tuple(dict.fromkeys(row["all_gold_abc"])),
        }
        previous = groups.get(group_id)
        if previous is not None and previous != normalized:
            raise ContractError(f"conflicting duplicated route rows: {group_id}")
        groups[group_id] = normalized
    return tuple(groups[key] for key in sorted(groups))
