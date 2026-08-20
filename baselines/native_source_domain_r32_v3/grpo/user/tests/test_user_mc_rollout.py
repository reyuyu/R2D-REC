import json
import re
import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_rollout import prepare_mc_scored_rollout  # noqa: E402


SID_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
A = "<|video_begin|><s_a_101><s_b_201><s_c_301>"
B = "<|prod_begin|><s_a_102><s_b_202><s_c_302>"
X = "<|ad_begin|><s_a_103><s_b_203><s_c_303>"


class CanonicalTokenizer:
    def __init__(self):
        self.piece_to_id = {}
        self.id_to_piece = {}
        self.decode_skip_special_tokens = []

    def _id(self, piece):
        if piece not in self.piece_to_id:
            token_id = len(self.piece_to_id) + 1
            self.piece_to_id[piece] = token_id
            self.id_to_piece[token_id] = piece
        return self.piece_to_id[piece]

    @staticmethod
    def _sid_pieces(sid, base):
        boundaries = [0, sid.index("><s_a_") + 1, sid.index("><s_b_") + 1, sid.index("><s_c_") + 1, len(sid)]
        return [
            (sid[boundaries[index] : boundaries[index + 1]], base + boundaries[index], base + boundaries[index + 1])
            for index in range(4)
        ]

    def _pieces(self, text):
        pieces = []
        cursor = 0
        for match in SID_RE.finditer(text):
            pieces.extend((text[index], index, index + 1) for index in range(cursor, match.start()))
            pieces.extend(self._sid_pieces(match.group(), match.start()))
            cursor = match.end()
        pieces.extend((text[index], index, index + 1) for index in range(cursor, len(text)))
        return pieces

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        pieces = self._pieces(text)
        result = {"input_ids": [self._id(piece) for piece, _, _ in pieces]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(start, end) for _, start, end in pieces]
        return result

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids, skip_special_tokens=False):
        self.decode_skip_special_tokens.append(skip_special_tokens)
        return "".join(self.id_to_piece[int(token_id)] for token_id in ids)


def action_row(sample_id="action-1", gold=(A, B), history=(A, B, X)):
    return {
        "sample_id": sample_id,
        "route": "action",
        "gold_sids": list(gold),
        "history_sids": list(history),
    }


def event(date, action, logic):
    return {"date": date, "action": action, "logic": logic}


E1 = event("2026-01-01", "watch alpha", "likes alpha")
E2 = event("2026-01-02", "buy beta", "needs beta")
EX = event("2026-01-03", "mc_unrelated_action", "mc_unrelated_logic")


def chain_row(sample_id="chain-1"):
    return {
        "sample_id": sample_id,
        "route": "chain",
        "gold_events": [E1, E2],
        "history_events": [],
        "history_sids": [],
    }


def chain_json(events):
    return json.dumps(
        {"logic_chain": {"name": "test", "events": events}},
        separators=(",", ":"),
    )


def encoded(tokenizer, *completions):
    return [tokenizer.encode(completion, add_special_tokens=False) for completion in completions]


class MCRolloutContractTests(unittest.TestCase):
    def test_one_prompt_two_completions(self):
        tokenizer = CanonicalTokenizer()
        completions = [json.dumps([A, B]), json.dumps([A, X])]
        result = prepare_mc_scored_rollout(
            [action_row()], encoded(tokenizer, *completions), tokenizer
        )
        self.assertEqual(result["K"], 2)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["completions"], completions)
        self.assertEqual([row["sample_id"] for row in result["expanded_rows"]], ["action-1", "action-1"])
        self.assertEqual(tokenizer.decode_skip_special_tokens, [False, False])
        self.assertEqual(result["valid_candidate_rate"], 1.0)
        self.assertEqual(result["positive_unit_count"], 3)
        self.assertEqual(result["negative_unit_count"], 1)
        self.assertEqual(result["zero_unit_count"], 0)

    def test_two_prompts_expand_candidate_major_within_each_row(self):
        tokenizer = CanonicalTokenizer()
        rows = [action_row("row-1"), action_row("row-2", gold=(B,), history=(A, B))]
        completions = [json.dumps([A]), json.dumps([A, X]), json.dumps([B]), json.dumps([B, A])]
        result = prepare_mc_scored_rollout(rows, encoded(tokenizer, *completions), tokenizer)
        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(
            [row["sample_id"] for row in result["expanded_rows"]],
            ["row-1", "row-1", "row-2", "row-2"],
        )
        self.assertEqual(result["completions"], completions)

    def test_non_k2_counts_raise(self):
        tokenizer = CanonicalTokenizer()
        ids = tokenizer.encode(json.dumps([A]))
        with self.assertRaisesRegex(ValueError, "K=2"):
            prepare_mc_scored_rollout([action_row()], [ids], tokenizer)
        with self.assertRaisesRegex(ValueError, "K=2"):
            prepare_mc_scored_rollout([action_row()], [ids, ids, ids], tokenizer)

    def test_mixed_routes_raise(self):
        tokenizer = CanonicalTokenizer()
        ids = tokenizer.encode(json.dumps([A]))
        with self.assertRaisesRegex(ValueError, "route-homogeneous"):
            prepare_mc_scored_rollout([action_row(), chain_row()], [ids] * 4, tokenizer)

    def test_action_tp_fp_units_have_exact_canonical_spans(self):
        tokenizer = CanonicalTokenizer()
        completion = json.dumps([A, X])
        result = prepare_mc_scored_rollout(
            [action_row()], encoded(tokenizer, completion, completion), tokenizer
        )
        units = result["credit_units_per_candidate"][0]
        by_sid = {unit["sid"]: unit for unit in units}
        self.assertGreater(by_sid[A]["delta"], 0.0)
        self.assertLess(by_sid[X]["delta"], 0.0)
        for sid in (A, X):
            unit = by_sid[sid]
            self.assertEqual(completion[unit["char_start"] : unit["char_end"]], sid)
            self.assertEqual(unit["token_end"] - unit["token_start"], 4)
            self.assertEqual(len(unit["token_ids"]), 4)

    def test_duplicate_zero_unit_is_retained(self):
        tokenizer = CanonicalTokenizer()
        completion = json.dumps([A, A, B])
        result = prepare_mc_scored_rollout(
            [action_row()], encoded(tokenizer, completion, completion), tokenizer
        )
        duplicate = next(
            unit
            for unit in result["credit_units_per_candidate"][0]
            if unit["sid"] == A and unit["occurrence"] == 2
        )
        self.assertEqual(duplicate["delta"], 0.0)
        self.assertEqual(duplicate["credit_type"], "zero")
        self.assertGreaterEqual(result["zero_unit_count"], 2)

    def test_chain_positive_and_negative_event_units_have_exact_spans(self):
        tokenizer = CanonicalTokenizer()
        clean = chain_json([E1, E2])
        with_extra = chain_json([E1, E2, EX])
        result = prepare_mc_scored_rollout(
            [chain_row()], encoded(tokenizer, clean, with_extra), tokenizer
        )
        clean_units = result["credit_units_per_candidate"][0]
        extra_units = result["credit_units_per_candidate"][1]
        self.assertTrue(all(unit["delta"] > 0.0 for unit in clean_units))
        self.assertLess(extra_units[-1]["delta"], 0.0)
        self.assertIn("delta_action_alignment", clean_units[0])
        self.assertIn("delta_logic_alignment", clean_units[0])
        for completion, units, events in (
            (clean, clean_units, [E1, E2]),
            (with_extra, extra_units, [E1, E2, EX]),
        ):
            for unit, expected_event in zip(units, events):
                fragment = completion[unit["char_start"] : unit["char_end"]]
                self.assertEqual(json.loads(fragment), expected_event)
                self.assertEqual(unit["token_end"] - unit["token_start"], len(unit["token_ids"]))

    def test_malformed_candidate_has_no_units_and_keeps_diagnostics(self):
        tokenizer = CanonicalTokenizer()
        valid = json.dumps([A])
        malformed = "not-json"
        result = prepare_mc_scored_rollout(
            [action_row()], encoded(tokenizer, valid, malformed), tokenizer
        )
        self.assertEqual(result["valid_candidate_rate"], 0.5)
        self.assertEqual(result["credit_units_per_candidate"][1], [])
        self.assertFalse(result["marginal_results"][1]["valid"])
        self.assertIn("parser_errors", result["marginal_results"][1]["diagnostics"])

    def test_contract_contains_no_advantage_group_stats_or_penalty_fields(self):
        tokenizer = CanonicalTokenizer()
        completion = json.dumps([A])
        result = prepare_mc_scored_rollout(
            [action_row()], encoded(tokenizer, completion, completion), tokenizer
        )

        keys = set()

        def collect(value):
            if isinstance(value, dict):
                keys.update(value)
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)

        collect(result)
        forbidden = {
            "group_mean",
            "group_std",
            "group_reward_means",
            "group_reward_stds",
            "sequence_advantage",
            "sequence_advantages",
            "token_advantage",
            "token_advantages",
            "ppo_ratio",
            "penalty_mask",
        }
        self.assertTrue(keys.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
