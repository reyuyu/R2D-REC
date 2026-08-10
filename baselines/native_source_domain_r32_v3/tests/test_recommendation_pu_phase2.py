"""CPU-only unit tests for REC-PU Phase 2 helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rec_pu"))
from recommendation_pu_phase2 import (  # noqa: E402
    IGNORE_INDEX,
    PackedSegment,
    RecPUMetadataError,
    RecPULocationError,
    build_prefix_positive_sets,
    locate_packed_rec_pu_targets,
    parse_recommendation_metadata,
    prefix_positive_token_ids,
    sid_to_token_ids,
)


DOMAIN = "<|prod_begin|>"
A1, A4 = "<s_a_1>", "<s_a_4>"
B1, B3, B7 = "<s_b_1>", "<s_b_3>", "<s_b_7>"
C1, C2, C8 = "<s_c_1>", "<s_c_2>", "<s_c_8>"
SID_111 = DOMAIN + A1 + B1 + C1
SID_112 = DOMAIN + A1 + B1 + C2
SID_138 = DOMAIN + A1 + B3 + C8
SID_472 = DOMAIN + A4 + B7 + C2


class FakeTokenizer:
    def __init__(self) -> None:
        tokens = [DOMAIN, A1, A4, B1, B3, B7, C1, C2, C8]
        self.vocab = {token: index + 10 for index, token in enumerate(tokens)}
        self.inverse = {token_id: token for token, token_id in self.vocab.items()}
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.inverse.get(token_id, "<unk>")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.convert_tokens_to_ids(text)] if text in self.vocab else [self.unk_token_id]


TOKENIZER = FakeTokenizer()


def metadata(current: str, all_gold: list[str]) -> dict:
    return {
        "recommendation_group_id": "group-test",
        "recommendation_group_size": len(all_gold),
        "recommendation_all_gold_sids": all_gold,
        "recommendation_current_gold_sid": current,
    }


def labels_for(sid: str) -> list[int]:
    ids = sid_to_token_ids(parse_recommendation_metadata(metadata(sid, [sid])).current_gold_sid, TOKENIZER)
    return [ids.domain, ids.a, ids.b, ids.c]


def expect_metadata_error(code: str, raw: dict) -> None:
    try:
        parse_recommendation_metadata(raw)
    except RecPUMetadataError as error:
        assert error.code == code
    else:
        raise AssertionError(f"Expected metadata error {code}")


def test_1_singleton_metadata() -> None:
    parsed = parse_recommendation_metadata(metadata(SID_111, [SID_111]))
    positives = build_prefix_positive_sets(parsed)
    assert positives.a == (A1,) and positives.b == (B1,) and positives.c == (C1,)
    print("PASS 1 singleton: P_a/P_b/P_c each retain one positive")


def test_2_multi_positive_prefix() -> None:
    parsed = parse_recommendation_metadata(metadata(SID_112, [SID_111, SID_112, SID_138, SID_472]))
    positives = build_prefix_positive_sets(parsed)
    assert positives.a == (A1, A4)
    assert positives.b == (B1, B3)
    assert positives.c == (C1, C2)
    print("PASS 2 multi-positive prefix: P_a={a1,a4}, P_b={b1,b3}, P_c={c1,c2}")


def test_3_wrong_branch_excluded() -> None:
    parsed = parse_recommendation_metadata(metadata(SID_112, [SID_111, SID_112, SID_138, SID_472]))
    positives = build_prefix_positive_sets(parsed)
    assert B7 not in positives.b and C8 not in positives.c
    print("PASS 3 wrong branch: a4/b7/c2 never enters current a1/b1 P_b/P_c")


def test_4_current_in_all_gold_invariant() -> None:
    expect_metadata_error("current_not_in_gold", metadata(SID_112, [SID_111]))
    print("PASS 4 current_gold in all_gold invariant")


def test_5_duplicate_and_malformed_metadata() -> None:
    expect_metadata_error("gold_duplicate", metadata(SID_111, [SID_111, SID_111]))
    malformed = metadata(SID_111, [SID_111])
    malformed["recommendation_all_gold_sids"] = ["<|prod_begin|><s_a_1><s_b_1>"]
    expect_metadata_error("sid_malformed", malformed)
    print("PASS 5 duplicate and malformed metadata detection")


def test_6_single_token_sid_mapping() -> None:
    parsed = parse_recommendation_metadata(metadata(SID_112, [SID_111, SID_112, SID_138, SID_472]))
    current = sid_to_token_ids(parsed.current_gold_sid, TOKENIZER)
    positives = prefix_positive_token_ids(build_prefix_positive_sets(parsed), TOKENIZER)
    assert (current.domain, current.a, current.b, current.c) == (10, 11, 13, 17)
    assert positives.a == (11, 12) and positives.b == (13, 14) and positives.c == (16, 17)
    print("PASS 6 SID single-token mapping")


def test_7_final_component_label_positions() -> None:
    raw = metadata(SID_112, [SID_111, SID_112, SID_138, SID_472])
    labels = torch.tensor(labels_for(SID_112), dtype=torch.long)
    target = locate_packed_rec_pu_targets(labels, [PackedSegment(0, 4, "recommendation", raw)], TOKENIZER)[0]
    assert (target.a_label_position, target.b_label_position, target.c_label_position) == (1, 2, 3)
    print("PASS 7 final a/b/c label positions")


def test_8_causal_shift_positions() -> None:
    raw = metadata(SID_112, [SID_111, SID_112, SID_138, SID_472])
    target = locate_packed_rec_pu_targets(labels_for(SID_112), [PackedSegment(0, 4, "recommendation", raw)], TOKENIZER)[0]
    assert (target.a_logit_position, target.b_logit_position, target.c_logit_position) == (0, 1, 2)
    print("PASS 8 causal shift: logits[position - 1] predicts each final component")


def test_9_mixed_pack_no_metadata_leakage() -> None:
    first = metadata(SID_112, [SID_111, SID_112, SID_138, SID_472])
    second = metadata(SID_472, [SID_472])
    labels = labels_for(SID_111) + labels_for(SID_112) + labels_for(SID_111) + labels_for(SID_472)
    segments = [
        PackedSegment(0, 4, "material"),
        PackedSegment(4, 8, "recommendation", first),
        PackedSegment(8, 12, "user"),
        PackedSegment(12, 16, "recommendation", second),
    ]
    targets = locate_packed_rec_pu_targets(labels, segments, TOKENIZER)
    assert [target.segment_index for target in targets] == [1, 3]
    assert targets[0].positives.a == (11, 12)
    assert targets[1].positives.a == (12,)
    assert targets[1].positives.b == (15,) and targets[1].positives.c == (17,)
    print("PASS 9 packed mixed segments: recommendation metadata remains segment-local")


def test_10_non_recommendation_segments_inactive() -> None:
    labels = labels_for(SID_111) + labels_for(SID_112)
    segments = [PackedSegment(0, 4, "material"), PackedSegment(4, 8, "user")]
    assert locate_packed_rec_pu_targets(labels, segments, TOKENIZER) == []
    print("PASS 10 material/user segments do not activate REC-PU")


def test_11_two_recommendation_segments_independent() -> None:
    first = metadata(SID_111, [SID_111, SID_112])
    second = metadata(SID_472, [SID_472])
    labels = labels_for(SID_111) + labels_for(SID_472)
    targets = locate_packed_rec_pu_targets(
        labels,
        [PackedSegment(0, 4, "recommendation", first), PackedSegment(4, 8, "recommendation", second)],
        TOKENIZER,
    )
    assert targets[0].positives.c == (16, 17)
    assert targets[1].positives.c == (17,)
    print("PASS 11 two recommendation segments work independently")


def test_12_ignore_and_boundary_fail_closed() -> None:
    raw = metadata(SID_111, [SID_111])
    ids = labels_for(SID_111)
    try:
        locate_packed_rec_pu_targets([IGNORE_INDEX, *ids[1:]], [PackedSegment(0, 4, "recommendation", raw)], TOKENIZER)
    except RecPULocationError as error:
        assert error.code == "current_sid_not_found"
    else:
        raise AssertionError("IGNORE_INDEX must not yield a REC-PU target")
    try:
        locate_packed_rec_pu_targets(ids, [PackedSegment(1, 4, "recommendation", raw)], TOKENIZER)
    except RecPULocationError as error:
        assert error.code == "current_sid_not_found"
    else:
        raise AssertionError("Cross-boundary SID must not yield a REC-PU target")
    print("PASS 12 IGNORE_INDEX and boundary cases fail closed")


def test_13_final_occurrence_skips_think_citation() -> None:
    raw = metadata(SID_111, [SID_111])
    labels = labels_for(SID_111) + [99] + labels_for(SID_111)
    target = locate_packed_rec_pu_targets(
        labels,
        [PackedSegment(0, len(labels), "recommendation", raw)],
        TOKENIZER,
        final_occurrence=True,
    )[0]
    assert (target.a_label_position, target.b_label_position, target.c_label_position) == (6, 7, 8)
    try:
        locate_packed_rec_pu_targets(labels, [PackedSegment(0, len(labels), "recommendation", raw)], TOKENIZER)
    except RecPULocationError as error:
        assert error.code == "current_sid_not_unique"
    else:
        raise AssertionError("Strict Phase-2 locator must continue rejecting duplicate occurrences")
    print("PASS 13 final occurrence selects answer SID, not a think citation")


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
