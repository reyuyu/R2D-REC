"""Beam32 frontend/result contract; contains no real model.generate implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Sequence

from official_aligned_contract import ACTION_TOKENS, EVAL_BEAM_SIZE, UNKNOWN_OFFICIAL_DECODING_DETAILS, ContractError

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from rollout_metrics import parse_raw_abc  # noqa: E402


@dataclass(frozen=True)
class BeamDecoderContract:
    beam_size: int = EVAL_BEAM_SIZE
    action_tokens: int = ACTION_TOKENS
    real_gpu_decoder_implemented: bool = False
    unknown_official_fields: tuple[str, ...] = UNKNOWN_OFFICIAL_DECODING_DETAILS

    def __post_init__(self) -> None:
        if self.beam_size != 32 or self.action_tokens != 3:
            raise ContractError("Official-Aligned frontend requires Beam32 ABC3")
        if not self.unknown_official_fields:
            raise ContractError("unknown official decoding details must remain explicit")


@dataclass(frozen=True)
class BeamCandidate:
    recommendation_group_id: str
    target_domain: str
    beam_rank: int
    raw_token_ids: tuple[int, ...]
    raw_text_with_special_tokens: str
    parsed_abc: str | None
    parsed_sid: str | None
    format_valid: bool
    beam_score: float | None


def make_beam_candidate(
    recommendation_group_id: str,
    target_domain: str,
    fixed_domain_token: str,
    beam_rank: int,
    raw_token_ids: Sequence[int],
    id_to_token: Callable[[int], str],
    beam_score: float | None = None,
) -> BeamCandidate:
    if not 0 <= beam_rank < EVAL_BEAM_SIZE:
        raise ContractError("beam rank must be in [0,31]")
    ids = tuple(int(value) for value in raw_token_ids)
    parsed = parse_raw_abc(list(ids), id_to_token)
    abc = "".join(parsed) if parsed else None
    raw_text = "".join(str(id_to_token(value)) for value in ids)
    return BeamCandidate(
        str(recommendation_group_id), str(target_domain), beam_rank, ids, raw_text,
        abc, fixed_domain_token + abc if abc else None, parsed is not None,
        float(beam_score) if beam_score is not None else None,
    )


def synthetic_beam_fixture(rows: Sequence[tuple[Sequence[int], float | None]], **kwargs) -> tuple[BeamCandidate, ...]:
    """CPU schema/ranking fixture only; it is not an official decoder."""
    if len(rows) != EVAL_BEAM_SIZE:
        raise ContractError("synthetic Beam32 fixture requires exactly 32 candidates")
    ordered = sorted(enumerate(rows), key=lambda item: (float("inf") if item[1][1] is None else -item[1][1], item[0]))
    return tuple(make_beam_candidate(beam_rank=rank, raw_token_ids=row[0], beam_score=row[1], **kwargs) for rank, (_, row) in enumerate(ordered))
