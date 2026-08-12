"""REC-PU Phase 2: strict BETA metadata, prefix positives, and packed locating.

No Trainer, model forward, loss composition, or dataset mutation lives here.
The helpers receive already-tokenized labels and explicit segment boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

import torch


IGNORE_INDEX = -100
SID_RE = re.compile(
    r"^(?P<domain><\|[a-z]+_begin\|>)"
    r"(?P<a><s_a_\d+>)"
    r"(?P<b><s_b_\d+>)"
    r"(?P<c><s_c_\d+>)$"
)
METADATA_FIELDS = (
    "recommendation_group_id",
    "recommendation_group_size",
    "recommendation_all_gold_sids",
    "recommendation_current_gold_sid",
)


class RecPUMetadataError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def __reduce__(self):
        return (type(self), (self.code, str(self)))


class RecPULocationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def __reduce__(self):
        return (type(self), (self.code, str(self)))


@dataclass(frozen=True, order=True)
class SemanticSID:
    domain: str
    a: str
    b: str
    c: str

    @classmethod
    def parse(cls, value: Any) -> "SemanticSID":
        if not isinstance(value, str):
            raise RecPUMetadataError("sid_not_string", "SID must be a string.")
        match = SID_RE.fullmatch(value)
        if match is None:
            raise RecPUMetadataError("sid_malformed", f"Malformed complete SID: {value!r}")
        return cls(**match.groupdict())

    def render(self) -> str:
        return self.domain + self.a + self.b + self.c


@dataclass(frozen=True)
class RecommendationMetadata:
    group_id: str
    group_size: int
    all_gold_sids: tuple[SemanticSID, ...]
    current_gold_sid: SemanticSID


@dataclass(frozen=True)
class PrefixPositiveSets:
    a: tuple[str, ...]
    b: tuple[str, ...]
    c: tuple[str, ...]


@dataclass(frozen=True)
class SIDTokenIds:
    domain: int
    a: int
    b: int
    c: int


@dataclass(frozen=True)
class TokenPrefixPositiveSets:
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: tuple[int, ...]


@dataclass(frozen=True)
class PackedSegment:
    start: int
    end: int  # exclusive
    task_name: str
    metadata: RecommendationMetadata | Mapping[str, Any] | None = None
    source_segment: str | None = None


@dataclass(frozen=True)
class RecPUPackedTarget:
    segment_index: int
    segment_start: int
    segment_end: int
    a_label_position: int
    b_label_position: int
    c_label_position: int
    a_logit_position: int
    b_logit_position: int
    c_logit_position: int
    positives: TokenPrefixPositiveSets
    source_segment: str | None = None


class TokenizerLike(Protocol):
    unk_token_id: int | None

    def convert_tokens_to_ids(self, token: str) -> int: ...

    def convert_ids_to_tokens(self, token_id: int) -> str: ...

    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]: ...


def parse_recommendation_metadata(raw: Mapping[str, Any]) -> RecommendationMetadata:
    """Strictly parse actual BETA recommendation metadata without fallback."""

    missing = [field for field in METADATA_FIELDS if field not in raw]
    if missing:
        raise RecPUMetadataError("metadata_missing", f"Missing recommendation metadata fields: {missing}")
    group_id = raw["recommendation_group_id"]
    group_size = raw["recommendation_group_size"]
    raw_gold = raw["recommendation_all_gold_sids"]
    raw_current = raw["recommendation_current_gold_sid"]
    if not isinstance(group_id, str) or not group_id:
        raise RecPUMetadataError("group_id_invalid", "recommendation_group_id must be a non-empty string.")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise RecPUMetadataError("group_size_invalid", "recommendation_group_size must be a positive integer.")
    if not isinstance(raw_gold, list) or not raw_gold:
        raise RecPUMetadataError("all_gold_invalid", "recommendation_all_gold_sids must be a non-empty list.")

    gold = tuple(SemanticSID.parse(value) for value in raw_gold)
    if len(set(gold)) != len(gold):
        raise RecPUMetadataError("gold_duplicate", "recommendation_all_gold_sids contains duplicate complete SIDs.")
    if group_size != len(gold):
        raise RecPUMetadataError(
            "group_size_mismatch",
            f"group_size={group_size} does not equal unique all_gold_sids={len(gold)}.",
        )
    current = SemanticSID.parse(raw_current)
    if current not in set(gold):
        raise RecPUMetadataError("current_not_in_gold", "current_gold_sid is not contained in all_gold_sids.")
    return RecommendationMetadata(group_id=group_id, group_size=group_size, all_gold_sids=gold, current_gold_sid=current)


def build_prefix_positive_sets(metadata: RecommendationMetadata) -> PrefixPositiveSets:
    """Construct P_a/P_b/P_c using the teacher-forced current SID prefix."""

    current = metadata.current_gold_sid
    positives_a = tuple(sorted({sid.a for sid in metadata.all_gold_sids}))
    positives_b = tuple(
        sorted(
            {
                sid.b
                for sid in metadata.all_gold_sids
                if sid.domain == current.domain and sid.a == current.a
            }
        )
    )
    positives_c = tuple(
        sorted(
            {
                sid.c
                for sid in metadata.all_gold_sids
                if sid.domain == current.domain and sid.a == current.a and sid.b == current.b
            }
        )
    )
    if not positives_a or not positives_b or not positives_c:
        raise RecPUMetadataError("prefix_candidates_empty", "A prefix-conditioned positive set is empty.")
    return PrefixPositiveSets(a=positives_a, b=positives_b, c=positives_c)


def _single_token_id(tokenizer: TokenizerLike, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0:
        raise RecPUMetadataError("token_id_invalid", f"Tokenizer returned invalid id for {token!r}.")
    if tokenizer.convert_ids_to_tokens(token_id) != token:
        raise RecPUMetadataError("token_not_registered", f"{token!r} is not a registered exact tokenizer token.")
    encoded = list(tokenizer.encode(token, add_special_tokens=False))
    if encoded != [token_id]:
        raise RecPUMetadataError("token_not_single", f"{token!r} does not encode as exactly one token id.")
    return token_id


def sid_to_token_ids(sid: SemanticSID, tokenizer: TokenizerLike) -> SIDTokenIds:
    """Map the validated domain/a/b/c SID structure to exact single token IDs."""

    return SIDTokenIds(
        domain=_single_token_id(tokenizer, sid.domain),
        a=_single_token_id(tokenizer, sid.a),
        b=_single_token_id(tokenizer, sid.b),
        c=_single_token_id(tokenizer, sid.c),
    )


def prefix_positive_token_ids(positives: PrefixPositiveSets, tokenizer: TokenizerLike) -> TokenPrefixPositiveSets:
    return TokenPrefixPositiveSets(
        a=tuple(_single_token_id(tokenizer, token) for token in positives.a),
        b=tuple(_single_token_id(tokenizer, token) for token in positives.b),
        c=tuple(_single_token_id(tokenizer, token) for token in positives.c),
    )


def _label_values(labels: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(labels, torch.Tensor):
        if labels.ndim != 1:
            raise RecPULocationError("labels_not_1d", "Packed labels must be one-dimensional.")
        return [int(value) for value in labels.detach().cpu().tolist()]
    return [int(value) for value in labels]


def _validate_segments(segments: Sequence[PackedSegment], labels_length: int) -> None:
    previous_end = 0
    for index, segment in enumerate(segments):
        if segment.start < 0 or segment.end <= segment.start or segment.end > labels_length:
            raise RecPULocationError("segment_boundary_invalid", f"Invalid bounds for segment {index}: {segment.start}:{segment.end}")
        if index and segment.start < previous_end:
            raise RecPULocationError("segment_overlap", f"Segment {index} overlaps its predecessor.")
        previous_end = segment.end


def _find_unique_current_sid(
    labels: list[int], segment: PackedSegment, token_ids: SIDTokenIds, *, final_occurrence: bool = False
) -> tuple[int, int, int]:
    sequence = (token_ids.domain, token_ids.a, token_ids.b, token_ids.c)
    matches = []
    for start in range(segment.start, segment.end - len(sequence) + 1):
        if tuple(labels[start : start + len(sequence)]) == sequence:
            matches.append(start)
    if not matches:
        raise RecPULocationError("current_sid_not_found", "Current full SID is absent from supervised labels in its segment.")
    if len(matches) != 1 and not final_occurrence:
        raise RecPULocationError("current_sid_not_unique", "Current full SID occurs multiple times in supervised labels.")
    # CoT may cite a gold SID in <think>; the final answer is its last response occurrence.
    domain_position = matches[-1]
    a_position, b_position, c_position = domain_position + 1, domain_position + 2, domain_position + 3
    for position in (a_position, b_position, c_position):
        if labels[position] == IGNORE_INDEX:
            raise RecPULocationError("sid_label_ignored", "Final SID component cannot be IGNORE_INDEX.")
        if position - 1 < segment.start:
            raise RecPULocationError("causal_crosses_boundary", "Causal logit position crosses a segment boundary.")
    return a_position, b_position, c_position


def locate_packed_rec_pu_targets(
    labels: Sequence[int] | torch.Tensor,
    segments: Sequence[PackedSegment],
    tokenizer: TokenizerLike,
    *,
    final_occurrence: bool = False,
) -> list[RecPUPackedTarget]:
    """Locate independently valid REC-PU targets for recommendation segments.

    For every returned component, caller must use ``logits[label_position - 1]``
    to predict ``labels[label_position]``. No state is retained between segments.
    """

    values = _label_values(labels)
    _validate_segments(segments, len(values))
    targets: list[RecPUPackedTarget] = []
    for index, segment in enumerate(segments):
        if segment.task_name != "recommendation":
            continue
        if segment.metadata is None:
            raise RecPUMetadataError("metadata_missing", f"Recommendation segment {index} has no metadata.")
        metadata = segment.metadata if isinstance(segment.metadata, RecommendationMetadata) else parse_recommendation_metadata(segment.metadata)
        positives = prefix_positive_token_ids(build_prefix_positive_sets(metadata), tokenizer)
        current_ids = sid_to_token_ids(metadata.current_gold_sid, tokenizer)
        a_position, b_position, c_position = _find_unique_current_sid(
            values, segment, current_ids, final_occurrence=final_occurrence
        )
        targets.append(
            RecPUPackedTarget(
                segment_index=index,
                segment_start=segment.start,
                segment_end=segment.end,
                a_label_position=a_position,
                b_label_position=b_position,
                c_label_position=c_position,
                a_logit_position=a_position - 1,
                b_logit_position=b_position - 1,
                c_logit_position=c_position - 1,
                positives=positives,
                source_segment=segment.source_segment,
            )
        )
    return targets
