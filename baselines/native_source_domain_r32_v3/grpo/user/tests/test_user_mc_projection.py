import json
import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_projection import (  # noqa: E402
    MCCreditProjectionError,
    project_mc_credit_units_to_generated,
)
from user_mc_rollout import prepare_mc_scored_rollout  # noqa: E402


A = "<|video_begin|><s_a_101><s_b_201><s_c_301>"


class FixedTokenizer:
    def __init__(self, completion, canonical_ids, offsets, decoded_pieces, pad_token_id=0):
        self.completion = completion
        self.canonical_ids = list(canonical_ids)
        self.offsets = list(offsets)
        self.decoded_pieces = dict(decoded_pieces)
        self.pad_token_id = pad_token_id

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if text != self.completion:
            raise AssertionError(f"unexpected text: {text!r}")
        result = {"input_ids": list(self.canonical_ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = list(self.offsets)
        return result

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.decoded_pieces[int(token_id)] for token_id in ids)


def unit(start, end, token_ids, unit_type="sid"):
    value = {
        "unit_type": unit_type,
        "delta": 0.5,
        "credit_type": "positive",
        "char_start": 0,
        "char_end": 1,
        "token_start": start,
        "token_end": end,
        "token_ids": list(token_ids),
    }
    if unit_type == "sid":
        value.update(sid=A, occurrence=1)
    else:
        value.update(event_index=0, delta_action_alignment=0.5, delta_logic_alignment=0.5)
    return value


class MCCreditProjectionTests(unittest.TestCase):
    def test_identity_action_sid_four_tokens(self):
        tokenizer = FixedTokenizer(
            A,
            [1, 2, 3, 4],
            [(0, 15), (15, 24), (24, 33), (33, len(A))],
            {1: A[:15], 2: A[15:24], 3: A[24:33], 4: A[33:]},
        )
        result = project_mc_credit_units_to_generated(
            A, [unit(0, 4, [1, 2, 3, 4])], [1, 2, 3, 4], tokenizer
        )
        projected = result["projected_units"][0]
        self.assertFalse(result["projection_required"])
        self.assertFalse(projected["projection_required"])
        self.assertEqual(projected["generated_token_indices"], [0, 1, 2, 3])
        self.assertEqual(projected["generated_token_ids"], [1, 2, 3, 4])

    def test_identity_chain_event_span(self):
        tokenizer = FixedTokenizer(
            "EVENT",
            [1, 2, 3],
            [(0, 1), (1, 4), (4, 5)],
            {1: "E", 2: "VEN", 3: "T"},
        )
        result = project_mc_credit_units_to_generated(
            "EVENT", [unit(0, 3, [1, 2, 3], "event")], [1, 2, 3], tokenizer
        )
        projected = result["projected_units"][0]
        self.assertEqual(projected["canonical_token_ids"], [1, 2, 3])
        self.assertEqual(projected["generated_token_indices"], [0, 1, 2])

    def test_full_unit_safe_replace_maps_all_generated_indices(self):
        tokenizer = FixedTokenizer(
            "ABCD",
            [1, 2, 3],
            [(0, 1), (1, 3), (3, 4)],
            {1: "A", 2: "BC", 3: "D", 4: "B", 5: "C"},
        )
        result = project_mc_credit_units_to_generated(
            "ABCD", [unit(1, 2, [2])], [1, 4, 5, 3], tokenizer
        )
        projected = result["projected_units"][0]
        self.assertTrue(result["projection_required"])
        self.assertEqual(projected["canonical_token_ids"], [2])
        self.assertEqual(projected["generated_token_indices"], [1, 2])
        self.assertEqual(projected["generated_token_ids"], [4, 5])
        self.assertEqual(projected["token_start"], 1)
        self.assertEqual(projected["token_end"], 3)

    def test_partial_replace_fails_closed(self):
        tokenizer = FixedTokenizer(
            "ABCD",
            [1, 2, 3, 4],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            {1: "A", 2: "B", 3: "C", 4: "D", 5: "B", 6: "C"},
        )
        with self.assertRaisesRegex(MCCreditProjectionError, "partially intersects"):
            project_mc_credit_units_to_generated(
                "ABCD", [unit(1, 2, [2])], [1, 5, 6, 4], tokenizer
            )

    def test_insert_opcode_fails_closed(self):
        tokenizer = FixedTokenizer(
            "AB",
            [1, 2],
            [(0, 1), (1, 2)],
            {1: "A", 2: "B", 9: ""},
        )
        with self.assertRaisesRegex(MCCreditProjectionError, "insert"):
            project_mc_credit_units_to_generated("AB", [unit(0, 1, [1])], [1, 9, 2], tokenizer)

    def test_delete_opcode_fails_closed(self):
        tokenizer = FixedTokenizer(
            "AB",
            [1, 9, 2],
            [(0, 1), (1, 1), (1, 2)],
            {1: "A", 2: "B", 9: ""},
        )
        with self.assertRaisesRegex(MCCreditProjectionError, "delete"):
            project_mc_credit_units_to_generated("AB", [unit(0, 1, [1])], [1, 2], tokenizer)

    def test_generated_decode_mismatch_fails(self):
        tokenizer = FixedTokenizer(
            "AB", [1, 2], [(0, 1), (1, 2)], {1: "A", 2: "B", 3: "X"}
        )
        with self.assertRaisesRegex(MCCreditProjectionError, "generated token IDs"):
            project_mc_credit_units_to_generated("AB", [unit(0, 1, [1])], [1, 3], tokenizer)

    def test_canonical_decode_mismatch_fails(self):
        tokenizer = FixedTokenizer(
            "AB", [1, 3], [(0, 1), (1, 2)], {1: "A", 2: "B", 3: "X"}
        )
        with self.assertRaisesRegex(MCCreditProjectionError, "canonical token IDs"):
            project_mc_credit_units_to_generated("AB", [unit(0, 1, [1])], [1, 2], tokenizer)

    def test_empty_projected_unit_fails(self):
        tokenizer = FixedTokenizer("A", [1], [(0, 1)], {1: "A"})
        with self.assertRaisesRegex(MCCreditProjectionError, "empty or invalid"):
            project_mc_credit_units_to_generated("A", [unit(0, 0, [])], [1], tokenizer)

    def test_padding_projection_fails(self):
        tokenizer = FixedTokenizer("A", [0], [(0, 1)], {0: "A"})
        with self.assertRaisesRegex(MCCreditProjectionError, "padding"):
            project_mc_credit_units_to_generated("A", [unit(0, 1, [0])], [0], tokenizer)

    def test_rollout_outputs_generated_space_and_keeps_canonical_units(self):
        completion = json.dumps([A], separators=(",", ":"))
        sid_start = completion.index(A)
        boundaries = [
            sid_start,
            sid_start + A.index("><s_a_") + 1,
            sid_start + A.index("><s_b_") + 1,
            sid_start + A.index("><s_c_") + 1,
            sid_start + len(A),
        ]
        canonical_ids = [10, 11, 12, 13, 14, 15]
        offsets = [(0, sid_start)] + [
            (boundaries[index], boundaries[index + 1]) for index in range(4)
        ] + [(boundaries[-1], len(completion))]
        pieces = {
            10: completion[:sid_start],
            11: completion[boundaries[0] : boundaries[1]],
            12: completion[boundaries[1] : boundaries[2]],
            13: completion[boundaries[2] : boundaries[3]],
            14: completion[boundaries[3] : boundaries[4]],
            15: completion[boundaries[4] :],
        }
        split_piece = pieces[12]
        split_at = len(split_piece) // 2
        pieces[21] = split_piece[:split_at]
        pieces[22] = split_piece[split_at:]
        tokenizer = FixedTokenizer(completion, canonical_ids, offsets, pieces)
        generated_ids = [10, 11, 21, 22, 13, 14, 15]
        row = {
            "sample_id": "action-projection",
            "route": "action",
            "gold_sids": [A],
            "history_sids": [A],
        }
        result = prepare_mc_scored_rollout(
            [row], [generated_ids, generated_ids], tokenizer
        )
        canonical = result["canonical_credit_units_per_candidate"][0][0]
        projected = result["credit_units_per_candidate"][0][0]
        self.assertEqual(result["token_span_space"], "generated_completion_ids")
        self.assertEqual(canonical["token_end"] - canonical["token_start"], 4)
        self.assertEqual(projected["canonical_token_end"] - projected["canonical_token_start"], 4)
        self.assertEqual(projected["generated_token_indices"], [1, 2, 3, 4, 5])
        self.assertEqual(projected["token_ids"], [11, 21, 22, 13, 14])
        self.assertTrue(result["projection_per_candidate"][0]["projection_required"])


if __name__ == "__main__":
    unittest.main()
